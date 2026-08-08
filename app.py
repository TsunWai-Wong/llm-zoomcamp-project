import datetime
import textwrap

import duckdb
import streamlit as st

from src.agent.conversation import Conversation
from src.agent.llm_service import LLMService
from src.agent.prompts import Prompt
from src.agent.rag_agent import RAGAgent
from src.agent.tool_registry import ToolRegistry
from src.etl.data_loader import DataLoader
from src.monitoring.tracing import setup_tracing
from src.search.embedder import Embedder
from src.search.hybrid_search import HybridSearch
from src.search.text_search import TextSearch
from src.search.vector_search import VectorSearch


st.set_page_config(page_title="Songs Finder", page_icon="🎧")

# -----------------------------------------------------------------------------
# Set things up.

FEEDBACK_PARQUET = "data/curated/feedback.parquet"
PHOENIX_URL = "http://localhost:6006"

SUGGESTIONS = {
    ":blue[:material/ac_unit:] Something for a winter funeral": (
        "Find me some songs which can be played at a funeral in the winter. "
        "It should be a sad song."
    ),
    ":green[:material/celebration:] Songs for a summer party": (
        "What are some upbeat songs I could play at a summer party outdoors?"
    ),
    ":orange[:material/directions_car:] Long night drive alone": (
        "I want songs for driving alone at night on the highway. "
        "Something moody but not depressing."
    ),
    ":violet[:material/favorite:] Getting over a breakup": (
        "Songs about realising a relationship is over and deciding to move on."
    ),
}

DISCLAIMER = textwrap.dedent("""
    This assistant answers using a fixed corpus of song lyrics and an LLM.
    It can only recommend songs that are in that corpus, and it can still be
    wrong about what a song is about. Lyrics are shown as retrieved and are
    the property of their respective rights holders. Questions you type are
    sent to the model provider, so do not enter private or sensitive
    information.
""")


@st.cache_resource
def get_agent_parts():
    """Build the retrieval stack once and share it across sessions.

    Everything here is stateless and expensive: the ONNX embedder, the vector
    index built from the parquet, and the Elasticsearch client. Conversation
    history is deliberately NOT cached — that lives in session state, one per
    browser tab.
    """
    setup_tracing()

    loader = DataLoader()
    data = loader.load_data()

    text_search = TextSearch()
    vector_search = VectorSearch(Embedder())
    vector_search.build_index(data)
    hybrid_search = HybridSearch(text_search, vector_search)

    tools = ToolRegistry()
    tools.register("search", hybrid_search.hybrid_search)

    return RAGAgent(tools, LLMService(), Prompt.get_agent_instruction())


def get_conversation():
    """One Conversation per browser session, so histories never mix."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation(get_agent_parts())
    return st.session_state.conversation


def record_feedback(question, answer, rating, details, include_history):
    """Append one feedback row to the parquet the monitoring dashboard reads."""
    history = ""
    if include_history:
        history = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in st.session_state.messages
        )

    con = duckdb.connect()
    con.execute(
        "CREATE TABLE feedback ("
        "ts TIMESTAMP, question VARCHAR, answer VARCHAR, "
        "rating BIGINT, details VARCHAR, history VARCHAR)"
    )
    con.execute(
        "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
        [datetime.datetime.now(), question, answer, rating, details, history],
    )

    # Read-append-write rather than an in-place append: parquet files are
    # immutable, and at feedback volumes this stays cheap.
    try:
        con.execute(
            f"INSERT INTO feedback SELECT * FROM read_parquet('{FEEDBACK_PARQUET}')"
        )
    except duckdb.IOException:
        pass  # First piece of feedback, nothing to merge with yet.

    con.execute(f"COPY feedback TO '{FEEDBACK_PARQUET}' (FORMAT PARQUET)")
    con.close()


def show_feedback_controls(message_index):
    """Shows the "How did I do?" control under an assistant message."""
    st.write("")

    with st.popover("How did I do?"):
        with st.form(key=f"feedback-{message_index}", border=False):
            st.markdown(":small[Rating]")
            rating = st.feedback(options="stars")

            details = st.text_area("What was wrong or right? (optional)")
            include_history = st.checkbox("Include chat history", True)

            ""  # Add some space

            if st.form_submit_button("Send feedback"):
                answer = st.session_state.messages[message_index]["content"]
                question = ""
                if message_index > 0:
                    question = st.session_state.messages[message_index - 1]["content"]

                record_feedback(question, answer, rating, details, include_history)
                st.success("Thanks — recorded.")


@st.dialog("About this assistant")
def show_disclaimer_dialog():
    st.caption(DISCLAIMER)


# -----------------------------------------------------------------------------
# Draw the UI.

st.html("<div style='font-size: 5rem; line-height: 1'>🎧</div>")

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("Songs Finder", anchor=False, width="stretch")

user_just_asked_initial_question = bool(st.session_state.get("initial_question"))
user_just_clicked_suggestion = bool(st.session_state.get("selected_suggestion"))
user_first_interaction = (
    user_just_asked_initial_question or user_just_clicked_suggestion
)
has_message_history = bool(st.session_state.get("messages"))

# Show a different UI when the user hasn't asked a question yet.
if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Describe the songs you're looking for...", key="initial_question")

        st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: About this assistant]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    st.stop()

# Show chat input at the bottom once a question has been asked.
user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

with title_row:

    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None
        # The agent keeps its own copy of the history, so clearing only the
        # display list would leave the model still seeing the old turns.
        st.session_state.pop("conversation", None)

    st.button("Restart", icon=":material/refresh:", on_click=clear_conversation)

# Display chat messages from history as speech bubbles.
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()  # Fix ghost message bug.

        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_feedback_controls(i)

if user_message:
    # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
    # display math). The line below fixes it.
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        # The agentic loop searches, reads results, and searches again before
        # answering, so this is a single blocking call rather than a stream.
        with st.spinner("Searching the lyrics corpus..."):
            try:
                response = get_conversation().ask(user_message)
            except Exception as error:
                response = (
                    f"Something went wrong while answering: "
                    f"`{type(error).__name__}: {error}`"
                )

        # Put everything after the spinner in a container to fix the
        # ghost message bug.
        with st.container():
            st.markdown(response)

            st.session_state.messages.append(
                {"role": "user", "content": user_message}
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

            show_feedback_controls(len(st.session_state.messages) - 1)
