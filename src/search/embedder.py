import numpy as np
import onnxruntime as ort
from openinference.semconv.trace import EmbeddingAttributes, SpanAttributes
from tokenizers import Tokenizer
from pathlib import Path

from src.monitoring import get_tracer

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "Xenova" / "all-MiniLM-L6-v2"
MAX_TOKENS = 256
MODEL_NAME = "Xenova/all-MiniLM-L6-v2"

tracer = get_tracer(__name__)


class Embedder:
    def __init__(self, path=DEFAULT_MODEL_PATH):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def encode(self, text, normalize=True):
        # Only the single-text path is traced. encode_batch() is also the
        # ingest path, where one span per 32-document batch would bury the
        # query traces under thousands of spans.
        with tracer.start_as_current_span(
            "embed_query", openinference_span_kind="embedding"
        ) as span:
            span.set_input(text)
            span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, MODEL_NAME)
            span.set_attribute(
                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0."
                f"{EmbeddingAttributes.EMBEDDING_TEXT}",
                text,
            )
            return self.encode_batch([text], normalize=normalize)[0]

    def encode_batch(self, texts, normalize=True):
        self.tokenizer.enable_padding()
        encoded = self.tokenizer.encode_batch(texts)
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = np.array([e.ids for e in encoded], dtype=np.int64)
        if "attention_mask" in self.input_names:
            feed["attention_mask"] = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )
        if "token_type_ids" in self.input_names:
            feed["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64
            )
        hidden = self.session.run(None, feed)[0]
        mask = feed["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
        if normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled