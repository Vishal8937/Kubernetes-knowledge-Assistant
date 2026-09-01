import os
import streamlit as st
import requests
import time
import uuid
import logfire
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

env_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".env")
)

load_dotenv(dotenv_path=env_path)


# ============================================================
# LOGFIRE
# ============================================================

try:
    token = os.getenv("LOGFIRE_TOKEN")

    if not token:
        print("WARNING: LOGFIRE_TOKEN is empty or None!")

    logfire.configure(token=token)

    LOGFIRE_STATUS = "Connected & Tracing"

except Exception as e:
    print(f"Logfire Init Error in UI: {e}")
    LOGFIRE_STATUS = "Standby"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Kubernetes Knowledge Assistant",
    page_icon="☸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# KUBERNETES LOGO
# ============================================================

KUBERNETES_LOGO = (
    "https://raw.githubusercontent.com/kubernetes/kubernetes/"
    "master/logo/logo.png"
)


# ============================================================
# DARK THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    html {
        background: #000000 !important;
        color-scheme: dark;
    }

    body {
        background: #000000 !important;
        color: #ffffff !important;
    }

    .stApp {
        background: #000000 !important;
        color: #ffffff !important;
    }

    [data-testid="stAppViewContainer"] {
        background: #000000 !important;
    }

    [data-testid="stAppViewBlockContainer"] {
        background: #000000 !important;
    }

    .main {
        background: #000000 !important;
    }

    .main .block-container {
        max-width: 1200px !important;
        padding-top: 1rem !important;
        padding-bottom: 7rem !important;
    }


    /* ========================================================
       STREAMLIT TOP BAR
       ======================================================== */

    [data-testid="stHeader"] {
        background: #000000 !important;
    }

    header {
        background: #000000 !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #050505 !important;
        border-right: 1px solid #1b1b1b !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #050505 !important;
    }

    section[data-testid="stSidebar"] .block-container {
        background: #050505 !important;
        padding-top: 1.2rem !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }


    /* ========================================================
       SIDEBAR BUTTONS
       ======================================================== */

    section[data-testid="stSidebar"] button {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #292929 !important;
        border-radius: 9px !important;
    }

    section[data-testid="stSidebar"] button:hover {
        background: #181818 !important;
        border-color: #444444 !important;
    }


    /* ========================================================
       ALL MARKDOWN TEXT
       ======================================================== */

    [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        line-height: 1.7;
    }

    [data-testid="stMarkdownContainer"] li {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] b {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #ffffff !important;
    }


    /* ========================================================
       TITLE
       ======================================================== */

    [data-testid="stHeading"] {
        color: #ffffff !important;
    }

    [data-testid="stHeading"] h1,
    [data-testid="stHeading"] h2,
    [data-testid="stHeading"] h3 {
        color: #ffffff !important;
    }


    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    [data-testid="stChatMessage"] {
        background: #080808 !important;
        border: 1px solid #202020 !important;
        border-radius: 14px !important;
        padding: 14px 17px !important;
        margin-bottom: 12px !important;
    }

    [data-testid="stChatMessage"] p {
        color: #ffffff !important;
        font-size: 15px !important;
        line-height: 1.75 !important;
    }

    [data-testid="stChatMessage"] li {
        color: #ffffff !important;
    }

    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] b {
        color: #ffffff !important;
    }


    /* ========================================================
       CODE
       ======================================================== */

    [data-testid="stMarkdownContainer"] code {
        background: #151515 !important;
        color: #ffffff !important;
        border: 1px solid #292929 !important;
        border-radius: 5px !important;
        padding: 2px 6px !important;
    }

    [data-testid="stMarkdownContainer"] pre {
        background: #050505 !important;
        border: 1px solid #252525 !important;
        border-radius: 9px !important;
        padding: 15px !important;
    }

    [data-testid="stMarkdownContainer"] pre code {
        background: transparent !important;
        color: #ffffff !important;
        border: none !important;
    }


    /* ========================================================
       TABLES
       ======================================================== */

    [data-testid="stMarkdownContainer"] table {
        background: #080808 !important;
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] th {
        background: #151515 !important;
        color: #ffffff !important;
        border: 1px solid #292929 !important;
        padding: 10px !important;
    }

    [data-testid="stMarkdownContainer"] td {
        background: #080808 !important;
        color: #ffffff !important;
        border: 1px solid #292929 !important;
        padding: 10px !important;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        background: #090909 !important;
        border: 1px solid #202020 !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] summary {
        background: #090909 !important;
        color: #ffffff !important;
    }

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        color: #ffffff !important;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    [data-testid="stStatusWidget"] {
        background: #090909 !important;
        border: 1px solid #202020 !important;
        border-radius: 10px !important;
    }

    [data-testid="stStatusWidget"] * {
        color: #ffffff !important;
    }


    /* ========================================================
       INPUT AREA
       ======================================================== */

    [data-testid="stBottomBlockContainer"] {
        background: #000000 !important;
        border-top: 1px solid #171717 !important;
        box-shadow: none !important;
    }

    [data-testid="stBottomBlockContainer"] > div {
        background: #000000 !important;
    }

    [data-testid="stChatInput"] {
        background: #0a0a0a !important;
        border: 1px solid #292929 !important;
        border-radius: 14px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] > div {
        background: #0a0a0a !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #0a0a0a !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777777 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background: #161616 !important;
        color: #ffffff !important;
        border: none !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #1b1b1b !important;
    }


    /* ========================================================
       CAPTION
       ======================================================== */

    [data-testid="stCaptionContainer"] p {
        color: #888888 !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        background: #0b0b0b !important;
        color: #ffffff !important;
    }

    [data-testid="stAlert"] p {
        color: #ffffff !important;
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }

    ::-webkit-scrollbar-track {
        background: #000000;
    }

    ::-webkit-scrollbar-thumb {
        background: #292929;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #3a3a3a;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# RETRIEVAL + RERANKING DEBUG VIEW
# ============================================================

def _score(value):
    """Safely format a retrieval/reranking score."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def render_retrieval_debug(retrieval_debug):
    """
    Render the retrieval pipeline in the UI.

    Expected backend response:

    retrieval_debug = {
        "query": "...",
        "retrieval": {
            "method": "Qdrant vector similarity search",
            "candidate_count": 15,
            "candidates": [
                {
                    "content": "...",
                    "source": "...",
                    "score": 0.82,
                    "retrieved_rank": 1
                }
            ]
        },
        "reranking": {
            "method": "FlashRank Cross-Encoder",
            "model": "ms-marco-MiniLM-L-6-v2",
            "input_count": 15,
            "output_count": 5,
            "results": [
                {
                    "content": "...",
                    "source": "...",
                    "vector_score": 0.82,
                    "rerank_score": 0.91,
                    "original_rank": 3,
                    "rerank_rank": 1
                }
            ]
        }
    }
    """

    if not retrieval_debug:
        return

    retrieval = retrieval_debug.get("retrieval", {})
    reranking = retrieval_debug.get("reranking", {})

    candidates = retrieval.get("candidates", [])
    reranked = reranking.get("results", [])

    with st.expander("🔬 Retrieval & Reranking", expanded=False):

        # --------------------------------------------------------
        # QUERY
        # --------------------------------------------------------

        query = retrieval_debug.get("query", "")

        if query:
            st.markdown("### 🔎 Query")
            st.code(query, language=None)

        # --------------------------------------------------------
        # STAGE 1 — QDRANT RETRIEVAL
        # --------------------------------------------------------

        st.markdown("### 1️⃣ Vector Retrieval")

        candidate_count = retrieval.get(
            "candidate_count",
            len(candidates),
        )

        retrieval_method = retrieval.get(
            "method",
            "Qdrant vector similarity search",
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Candidates Retrieved", candidate_count)

        with col2:
            st.caption("Retrieval Method")
            st.write(retrieval_method)

        if candidates:

            st.caption(
                "Chunks returned by Qdrant before semantic reranking."
            )

            for index, chunk in enumerate(candidates, start=1):

                rank = chunk.get("retrieved_rank", index)
                score = chunk.get("score")
                source = chunk.get("source", "Unknown")
                content = chunk.get("content", "")

                preview = (
                    content[:100]
                    .replace("\n", " ")
                    .strip()
                )

                if len(content) > 100:
                    preview += "..."

                with st.expander(
                    f"#{rank}  ·  Vector Score: {_score(score)}  ·  {preview}"
                ):

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.caption("Original Rank")
                        st.write(f"#{rank}")

                    with c2:
                        st.caption("Qdrant Score")
                        st.write(_score(score))

                    with c3:
                        st.caption("Source")
                        st.write(source)

                    st.markdown("**Chunk Content**")
                    st.info(content)

        else:
            st.info(
                "No raw retrieval candidates were returned. "
                "The backend must return `retrieval_debug.retrieval.candidates` "
                "to display them."
            )

        # --------------------------------------------------------
        # STAGE 2 — FLASHRANK
        # --------------------------------------------------------

        st.divider()

        st.markdown("### 2️⃣ Semantic Reranking")

        rerank_method = reranking.get(
            "method",
            "FlashRank",
        )

        rerank_model = reranking.get(
            "model",
            "ms-marco-MiniLM-L-6-v2",
        )

        input_count = reranking.get(
            "input_count",
            len(candidates),
        )

        output_count = reranking.get(
            "output_count",
            len(reranked),
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Reranker Input", input_count)

        with c2:
            st.metric("Final Chunks", output_count)

        with c3:
            st.metric(
                "Dropped",
                max(input_count - output_count, 0),
            )

        st.caption(f"Method: {rerank_method}")
        st.caption(f"Model: {rerank_model}")

        st.caption(
            "FlashRank reorders the retrieved candidates according "
            "to query-document semantic relevance."
        )

        if reranked:

            st.markdown("#### 🏆 Final Ranking")

            for item in reranked:

                original_rank = item.get("original_rank", "?")
                final_rank = item.get("rerank_rank", "?")

                vector_score = item.get("vector_score")
                rerank_score = item.get("rerank_score")

                source = item.get(
                    "source",
                    "Unknown",
                )

                content = item.get(
                    "content",
                    "",
                )

                preview = (
                    content[:100]
                    .replace("\n", " ")
                    .strip()
                )

                if len(content) > 100:
                    preview += "..."

                with st.expander(
                    f"🏆 Final #{final_rank}  ·  "
                    f"Original #{original_rank}  ·  "
                    f"Rerank Score: {_score(rerank_score)}  ·  "
                    f"{preview}"
                ):

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.caption("Final Rank")
                        st.write(f"#{final_rank}")

                    with c2:
                        st.caption("Original Rank")
                        st.write(f"#{original_rank}")

                    with c3:
                        st.caption("Qdrant Score")
                        st.write(_score(vector_score))

                    with c4:
                        st.caption("FlashRank Score")
                        st.write(_score(rerank_score))

                    st.caption(f"Source: {source}")

                    st.markdown("**Chunk Content**")
                    st.info(content)

                    try:
                        movement = int(original_rank) - int(final_rank)

                        if movement > 0:
                            st.success(
                                f"⬆️ Promoted by {movement} position(s)"
                            )
                        elif movement < 0:
                            st.warning(
                                f"⬇️ Moved down by {abs(movement)} position(s)"
                            )
                        else:
                            st.caption(
                                "↔️ Ranking position unchanged"
                            )
                    except (TypeError, ValueError):
                        pass

        else:
            st.info(
                "No reranking metadata was returned. "
                "The backend must return `retrieval_debug.reranking.results` "
                "to display the reranking process."
            )

        # --------------------------------------------------------
        # DROPPED CHUNKS
        # --------------------------------------------------------

        if candidates and reranked:

            selected_ranks = {
                item.get("original_rank")
                for item in reranked
            }

            dropped = []

            for index, chunk in enumerate(candidates, start=1):

                rank = chunk.get(
                    "retrieved_rank",
                    index,
                )

                if rank not in selected_ranks:
                    dropped.append(chunk)

            if dropped:

                st.divider()

                st.markdown("### 🗑️ Removed After Reranking")

                st.caption(
                    f"{len(dropped)} retrieved chunk(s) were not "
                    "passed to the final answering context."
                )

                for index, chunk in enumerate(dropped, start=1):

                    rank = chunk.get(
                        "retrieved_rank",
                        index,
                    )

                    score = chunk.get("score")
                    source = chunk.get("source", "Unknown")
                    content = chunk.get("content", "")

                    preview = (
                        content[:100]
                        .replace("\n", " ")
                        .strip()
                    )

                    if len(content) > 100:
                        preview += "..."

                    with st.expander(
                        f"❌ Original #{rank}  ·  "
                        f"Vector Score: {_score(score)}  ·  "
                        f"{preview}"
                    ):

                        st.caption(f"Source: {source}")
                        st.markdown("**Chunk Content**")
                        st.info(content)

        st.divider()

        st.caption(
            "Note: Qdrant similarity scores and FlashRank scores "
            "are different scoring systems. FlashRank scores are "
            "not percentages."
        )



# ============================================================
# SESSION MANAGEMENT
# ============================================================

if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())

    logfire.info(
        f"New Kubernetes Assistant session created: "
        f"{st.session_state.session_id}"
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # LOGO
    # --------------------------------------------------------

    st.image(
        KUBERNETES_LOGO,
        width=65,
    )

    st.markdown(
        "### Kubernetes Knowledge Assistant"
    )

    st.caption(
        "Intelligent Kubernetes Assistant"
    )

    st.divider()


    # --------------------------------------------------------
    # NEW CONVERSATION
    # --------------------------------------------------------

    if st.button(
        "＋  New Conversation",
        width="stretch",
    ):

        logfire.info(
            f"New conversation requested for session "
            f"{st.session_state.session_id}"
        )

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()


    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.markdown("### System Status")

    with st.container(border=True):

        st.caption("OBSERVABILITY")

        st.write(
            f"🟢 {LOGFIRE_STATUS}"
        )

    with st.container(border=True):

        st.caption("SESSION")

        st.write(
            f"{st.session_state.session_id[:8]}..."
        )


    # --------------------------------------------------------
    # KNOWLEDGE AREAS
    # --------------------------------------------------------

    st.markdown("### Knowledge Areas")

    topics = [
        "Pods & Deployments",
        "Services & Networking",
        "Ingress",
        "ConfigMaps & Secrets",
        "RBAC & Security",
        "Storage",
        "Troubleshooting",
    ]

    for topic in topics:

        st.markdown(
            f"◈ {topic}"
        )


    st.divider()


    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    if st.button(
        "🗑 Clear Conversation",
        width="stretch",
    ):

        logfire.warn(
            f"Conversation cleared for session "
            f"{st.session_state.session_id}"
        )

        st.session_state.messages = []

        st.rerun()


# ============================================================
# TOP WEBSITE HEADER
# ============================================================

logo_col, title_col = st.columns(
    [0.08, 0.92],
    vertical_alignment="center",
)

with logo_col:

    st.image(
        KUBERNETES_LOGO,
        width=52,
    )


with title_col:

    st.markdown(
        "# Kubernetes Knowledge Assistant"
    )

    st.caption(
        "Kubernetes documentation & troubleshooting assistant"
    )

st.divider()


# ============================================================
# HOME / HERO
# ============================================================

if not st.session_state.messages:

    st.markdown(
        "<br>",
        unsafe_allow_html=False,
    )

    st.image(
        KUBERNETES_LOGO,
        width=75,
    )

    st.markdown(
        "# Kubernetes Knowledge Assistant"
    )

    st.markdown(
        """
        Ask questions about Kubernetes concepts,
        configurations, troubleshooting and best practices.

        Get grounded answers from your Kubernetes knowledge base.
        """
    )

    st.caption(
        "RAG-powered  •  Guardrails protected  •  Production-ready"
    )

    st.markdown("### Try asking")

    col1, col2 = st.columns(2)


    with col1:

        with st.container(border=True):

            st.markdown(
                "🚨 **Troubleshoot CrashLoopBackOff**"
            )

            st.caption(
                "Understand common causes and how to diagnose "
                "failing Kubernetes containers."
            )


        with st.container(border=True):

            st.markdown(
                "🌐 **Explain Kubernetes Networking**"
            )

            st.caption(
                "Learn how Pods, Services, Ingress and DNS "
                "work together."
            )


    with col2:

        with st.container(border=True):

            st.markdown(
                "📦 **Deployment vs StatefulSet**"
            )

            st.caption(
                "Understand when to use Deployments, "
                "StatefulSets and other workload controllers."
            )


        with st.container(border=True):

            st.markdown(
                "🔐 **Explain Kubernetes RBAC**"
            )

            st.caption(
                "Learn how Roles, ClusterRoles and "
                "ServiceAccounts control access."
            )


else:

    st.markdown(
        "### Conversation"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "assistant":

        with st.chat_message(
            "assistant",
            avatar="☸️",
        ):

            st.markdown(
                message["content"]
            )

    else:

        with st.chat_message(
            "user",
            avatar="👤",
        ):

            st.markdown(
                message["content"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

if prompt := st.chat_input(
    "Ask a Kubernetes question..."
):

    # --------------------------------------------------------
    # LOG INTERACTION
    # --------------------------------------------------------

    with logfire.span(
        "Kubernetes Assistant Chat",
        user_query=prompt,
        session_id=st.session_state.session_id,
    ):

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message(
            "user",
            avatar="👤",
        ):

            st.markdown(prompt)


        # ----------------------------------------------------
        # ASSISTANT RESPONSE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="☸️",
        ):

            with st.status(
                "🔎 Searching Kubernetes knowledge base...",
                expanded=True,
            ) as status:

                try:

                    # ----------------------------------------
                    # BACKEND REQUEST
                    # ----------------------------------------

                    with logfire.span(
                        "Calling Kubernetes RAG Backend"
                    ):

                        base_url = os.getenv(
                            "BACKEND_URL",
                            "http://localhost:8000",
                        )

                        url = f"{base_url}/query"

                        payload = {
                            "q": prompt,
                            "thread_id":
                                st.session_state.session_id,
                        }

                        response = requests.post(
                            url,
                            json=payload,
                            timeout=60,
                        )

                        response.raise_for_status()

                        data = response.json()


                    # ----------------------------------------
                    # THOUGHT PROCESS
                    # ----------------------------------------

                    steps = data.get(
                        "thought_process",
                        [],
                    )

                    if steps:

                        st.markdown(
                            "**Retrieval Pipeline**"
                        )

                        for step in steps:

                            st.write(
                                f"⚙️ {step}"
                            )


                    # ----------------------------------------
                    # STATUS
                    # ----------------------------------------

                    status.update(
                        label=(
                            "✅ Knowledge retrieved & "
                            "answer synthesized"
                        ),
                        state="complete",
                        expanded=False,
                    )


                    # ----------------------------------------
                    # RETRIEVAL + RERANKING DEBUG
                    # ----------------------------------------

                    retrieval_debug = data.get(
                        "retrieval_debug",
                        {},
                    )

                    if retrieval_debug:
                        render_retrieval_debug(
                            retrieval_debug
                        )


                    # ----------------------------------------
                    # SOURCES
                    # ----------------------------------------

                    sources = data.get(
                        "sources",
                        [],
                    )

                    if sources:

                        with st.expander(
                            f"📚 Retrieved Sources · "
                            f"{len(sources)} chunks"
                        ):

                            for i, source in enumerate(
                                sources
                            ):

                                st.markdown(
                                    f"**SOURCE {i + 1}**"
                                )

                                preview = (
                                    source[:120]
                                    .replace("\n", " ")
                                    .strip()
                                )

                                st.caption(
                                    preview + "..."
                                )

                                with st.expander(
                                    "View retrieved context"
                                ):

                                    st.markdown(
                                        source
                                    )

                                if i < len(sources) - 1:

                                    st.divider()


                # --------------------------------------------
                # CONNECTION ERROR
                # --------------------------------------------

                except requests.exceptions.ConnectionError:

                    logfire.error(
                        "Kubernetes RAG backend connection failed."
                    )

                    status.update(
                        label="❌ Backend unavailable",
                        state="error",
                    )

                    st.error(
                        "Unable to connect to the Kubernetes "
                        "RAG backend. Make sure FastAPI is running."
                    )

                    st.stop()


                # --------------------------------------------
                # TIMEOUT
                # --------------------------------------------

                except requests.exceptions.Timeout:

                    logfire.error(
                        "Kubernetes RAG backend request timed out."
                    )

                    status.update(
                        label="❌ Request timed out",
                        state="error",
                    )

                    st.error(
                        "The backend took too long to respond. "
                        "Please try again."
                    )

                    st.stop()


                # --------------------------------------------
                # GENERAL ERROR
                # --------------------------------------------

                except Exception as e:

                    logfire.error(
                        f"UI-Backend connection failed: {e}"
                    )

                    status.update(
                        label="❌ Request failed",
                        state="error",
                    )

                    st.error(
                        f"Request failed: {e}"
                    )

                    st.stop()


            # =================================================
            # ANSWER
            # =================================================

            full_answer = data.get(
                "answer",
                "No response received.",
            )

            st.caption("✦ ANSWER")


            # -------------------------------------------------
            # TYPING EFFECT
            # -------------------------------------------------

            answer_placeholder = st.empty()

            curr_text = ""

            for char in full_answer:

                curr_text += char

                answer_placeholder.markdown(
                    curr_text + "▌"
                )

                time.sleep(0.003)


            # -------------------------------------------------
            # FINAL ANSWER
            # -------------------------------------------------

            answer_placeholder.markdown(
                full_answer
            )


            # -------------------------------------------------
            # SAVE
            # -------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_answer,
                }
            )

            logfire.info(
                "Kubernetes Assistant chat cycle completed."
            )


# ============================================================
# FOOTER
# ============================================================

if st.session_state.messages:

    st.caption(
        "Kubernetes Knowledge Assistant · "
        "Answers are generated from the connected knowledge base"
    )