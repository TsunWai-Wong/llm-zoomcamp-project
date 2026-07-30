from tqdm import tqdm

import duckdb
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.monitoring import get_tracer, set_documents


tracer = get_tracer(__name__)


class TextSearch:
    es_client: Elasticsearch
    index_name: str

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "lyrics",
    ) -> None:
        self.es_client = Elasticsearch(es_url)
        self.index_name = index_name

    def build_index(self, data: duckdb.DuckDBPyRelation) -> None:
        """Create the index if it does not exist and bulk-index all rows.

        Idempotent: documents are indexed with their id as the ES _id, so
        re-running overwrites existing documents instead of duplicating them.
        """
        index_settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
        index_mappings = {
            "properties": {
                "title": {"type": "text"},
                "tag": {"type": "keyword"},
                "artist": {"type": "keyword"},
                "year": {"type": "integer"},
                "views": {"type": "long"},
                "lyrics": {"type": "text"},
                "id": {"type": "keyword"},
            }
        }

        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(
                index=self.index_name,
                settings=index_settings,
                mappings=index_mappings,
            )

        columns = data.columns
        rows = data.fetchall()
        actions = (
            {"_index": self.index_name, "_id": doc["id"], **doc}
            for doc in (dict(zip(columns, row)) for row in rows)
        )
        bulk(self.es_client, tqdm(actions, total=len(rows)))

    def text_search(self, query: str, num_results: int = 5) -> list[dict]:
        """Full-text search over titles and lyrics, best matches first."""
        with tracer.start_as_current_span(
            "text_search", openinference_span_kind="retriever"
        ) as span:
            span.set_input(query)
            span.set_attribute("retriever.num_results", num_results)
            hits = self._text_search(query, num_results)
            set_documents(
                span,
                [hit["_source"] for hit in hits],
                scores=[hit["_score"] for hit in hits],
            )
            return [hit["_source"] for hit in hits]

    def _text_search(self, query: str, num_results: int) -> list[dict]:
        search_query = {
            "size": num_results,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["lyrics^4", "title"],
                            "type": "best_fields"
                        }
                    },
                }
            }
        }
        response = self.es_client.search(
            index=self.index_name,
            body=search_query,
        )
        return response["hits"]["hits"]
    

