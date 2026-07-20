"""
Groq + LangGraph "Mother & Child" chatbot -- Streamlit UI.

Run with:
    pip install -r requirements.txt
    streamlit run mother_child_chatbot_streamlit.py

--------------------------------------------------------------------------
CLOUD DB SETUP (Neon Postgres)
--------------------------------------------------------------------------
Create a file named `.env` next to this script with a single line:

    DATABASE_URL=postgresql://<user>:<password>@<your-neon-host>/<dbname>?sslmode=require

(Copy this straight from the Neon dashboard -> Connection Details ->
"Connection string". Make sure `sslmode=require` is present -- Neon
requires TLS.)

--------------------------------------------------------------------------
Design
--------------------------------------------------------------------------
- Every user (child) is their own thread_id -> fully isolated conversation.
- Sign up with a brand-new username, or log in with an existing one.
  Usernames are enforced unique.
- LIVE state (what's sent to the LLM) stays small and bounded:
    * only the last KEEP_RECENT messages are kept verbatim in the graph state
    * everything older is folded into a running `summary` (also persisted)
- NOTHING IS EVER DELETED FROM MEMORY. Before old messages are dropped from
  live state, they are written verbatim to an `archive` table keyed by
  username, searchable via Postgres full-text search.
- Because everything is persisted in Neon and looked up by username
  (thread_id), reopening the app (even from a different machine) and
  logging back in with the same username picks the conversation up
  exactly where it left off.
- All Postgres access goes through a pooled, health-checked connection
  pool (psycopg_pool.ConnectionPool). This fixes the "connection is
  closed" error that shows up after deployment: Neon suspends idle
  connections, and a single long-lived cached connection (the old
  `st.cache_resource` pattern) goes stale silently. The pool validates
  connections before handing them out and reconnects automatically, and
  db_cursor() retries once if a query still races a drop.

The Groq API key is still entered in the sidebar (never hard-coded / never
read from an env var), so anyone can bring their own key.
"""

import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Annotated, TypedDict, Optional

import streamlit as st
from dotenv import load_dotenv
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import tuple_row

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    RemoveMessage,
    trim_messages,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

load_dotenv()  # reads .env in the current working directory

DATABASE_URL = os.getenv("DATABASE_URL")

DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"

MAX_CONTEXT_TOKENS = 4000
SUMMARIZE_THRESHOLD = 30
KEEP_RECENT = 12

RECALL_MAX_RESULTS = 3
RECALL_MIN_WORD_LEN = 4
RECALL_STOPWORDS = {
    "what", "when", "where", "were", "there", "your", "about", "have",
    "with", "this", "that", "from", "they", "them", "then", "does",
    "remember", "recall", "tell", "told", "said",
}

MOTHER_SYSTEM_PROMPT = (
    "You are speaking as a warm, patient, loving mother talking with your child, {name}. "
    "You remember everything {name} has ever told you and bring it up naturally when it "
    "fits, the way a real mother would. You are encouraging, gentle when correcting, and "
    "genuinely curious about {name}'s day, feelings, and interests. Keep your tone caring "
    "and age-appropriate, never clinical or robotic."
)


def _require_database_url():
    if not DATABASE_URL:
        st.error(
            "DATABASE_URL is not set. Create a `.env` file next to this script with:\n\n"
            "DATABASE_URL=postgresql://user:password@your-neon-host/dbname?sslmode=require"
        )
        st.stop()


# --------------------------------------------------------------------------
# 0. Pooled DB connection (replaces the old single cached psycopg.connect)
# --------------------------------------------------------------------------

@st.cache_resource
def get_db_pool() -> ConnectionPool:
    """A process-wide pool, cached once per app process.

    Unlike a single cached connection, the pool validates connections
    before handing them out (check=ConnectionPool.check_connection) and
    transparently opens replacements for any Neon closed server-side
    (e.g. after the endpoint auto-suspends from being idle). This is what
    prevents 'the connection is closed' errors at runtime after
    deployment.
    """
    _require_database_url()
    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=5,
        kwargs={
            "autocommit": True,
            "row_factory": tuple_row,
            "prepare_threshold": 0,  # avoid stale prepared statements after reconnects
        },
        check=ConnectionPool.check_connection,
        reconnect_timeout=30,
    )
    pool.wait(timeout=30)
    return pool


def _is_connection_error(exc: BaseException) -> bool:
    return isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError))


@contextmanager
def db_cursor():
    """Yields a live pooled connection, retrying once if it turns out to
    have been dropped between the pool's health check and our query
    (e.g. Neon suspends the endpoint mid-request)."""
    pool = get_db_pool()
    try:
        with pool.connection() as conn:
            yield conn
        return
    except Exception as e:
        if not _is_connection_error(e):
            raise
    # One retry, giving the pool a moment to replace the bad connection.
    time.sleep(0.5)
    with pool.connection() as conn:
        yield conn


# --------------------------------------------------------------------------
# 1. Users table (login lookup, enforces unique usernames)
# --------------------------------------------------------------------------

def init_users_table():
    with db_cursor() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            created_at TEXT
        )""")


def username_exists(username: str) -> bool:
    with db_cursor() as conn:
        return conn.execute(
            "SELECT 1 FROM users WHERE username=%s", (username,)
        ).fetchone() is not None


def create_user(username: str):
    with db_cursor() as conn:
        conn.execute(
            "INSERT INTO users (username, created_at) VALUES (%s,%s)",
            (username, datetime.now(timezone.utc).isoformat()),
        )


# --------------------------------------------------------------------------
# 2. Archive table (permanent, unbounded raw transcript)
# --------------------------------------------------------------------------

def init_archive_table():
    with db_cursor() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS archive (
            id SERIAL PRIMARY KEY,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            archived_at TEXT NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_user ON archive(username)")
        # Full-text search replacement for SQLite's FTS5 virtual table:
        # a GIN index over to_tsvector(content) lets to_tsquery searches
        # (used in search_archive below) run fast without a shadow table.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_archive_fts "
            "ON archive USING GIN (to_tsvector('english', content))"
        )


def archive_messages(username: str, messages: list[BaseMessage]):
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (username, m.type, m.content, now)
        for m in messages
        if getattr(m, "content", None)
    ]
    if not rows:
        return
    with db_cursor() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO archive (username, role, content, archived_at) VALUES (%s,%s,%s,%s)",
                rows,
            )


def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return list(dict.fromkeys(
        w for w in words
        if len(w) >= RECALL_MIN_WORD_LEN and w not in RECALL_STOPWORDS
    ))


def search_archive(username: str, query_text: str, limit: int = RECALL_MAX_RESULTS) -> list[str]:
    keywords = _extract_keywords(query_text)
    if not keywords:
        return []
    # OR the keywords together, same intent as the old FTS5 MATCH expression.
    tsquery_expr = " | ".join(keywords)
    try:
        with db_cursor() as conn:
            cursor = conn.execute(
                """SELECT content FROM archive
                   WHERE username = %s
                     AND to_tsvector('english', content) @@ to_tsquery('english', %s)
                   ORDER BY id DESC LIMIT %s""",
                (username, tsquery_expr, limit),
            )
            return [row[0] for row in cursor.fetchall()]
    except psycopg.errors.SyntaxError:
        return []


def full_transcript(username: str) -> list[tuple[str, str, str]]:
    """Every archived message for this user, oldest first: (role, content, archived_at)."""
    with db_cursor() as conn:
        return conn.execute(
            "SELECT role, content, archived_at FROM archive WHERE username=%s ORDER BY id ASC",
            (username,),
        ).fetchall()


def archive_count(username: str) -> int:
    with db_cursor() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM archive WHERE username=%s", (username,)
        ).fetchone()[0]


# --------------------------------------------------------------------------
# 3. Graph state
# --------------------------------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: Optional[str]
    recall: Optional[str]


def approx_token_counter(messages: list[BaseMessage]) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token) used to keep
    trim_messages() from needing a real tokenizer (avoids the optional
    `transformers` dependency that some chat models fall back to).
    """
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        total += max(1, len(content) // 4)
    return total


# --------------------------------------------------------------------------
# 4. Build the graph
# --------------------------------------------------------------------------

def build_graph(checkpointer: PostgresSaver, username: str, api_key: str, model_name: str):
    llm = ChatGroq(model=model_name, temperature=0.7, api_key=api_key)
    summarizer = ChatGroq(model=model_name, temperature=0, api_key=api_key)

    def maybe_summarize(state: ChatState) -> dict:
        messages = state["messages"]
        if len(messages) <= SUMMARIZE_THRESHOLD:
            return {}

        to_archive = messages[:-KEEP_RECENT]
        archive_messages(username, to_archive)

        existing_summary = state.get("summary") or ""
        archive_text = "\n".join(
            f"{m.type}: {m.content}" for m in to_archive if getattr(m, "content", None)
        )
        summary_prompt = (
            "You maintain a running summary of an ongoing relationship between a "
            "mother and her child, for the mother's use. Update the summary below "
            "to incorporate the new messages. Keep the child's name, personality, "
            "preferences, important events, feelings, and unresolved things the "
            "child cares about. Be concise (under 200 words), written as notes a "
            "caring parent would keep.\n\n"
            f"EXISTING SUMMARY:\n{existing_summary or '(none yet)'}\n\n"
            f"NEW MESSAGES TO FOLD IN:\n{archive_text}\n\n"
            "UPDATED SUMMARY:"
        )
        new_summary = summarizer.invoke([HumanMessage(content=summary_prompt)]).content

        removals = [RemoveMessage(id=m.id) for m in to_archive if m.id is not None]
        return {"summary": new_summary, "messages": removals}

    def recall_memory(state: ChatState) -> dict:
        messages = state["messages"]
        latest_human = next((m for m in reversed(messages) if m.type == "human"), None)
        if latest_human is None:
            return {"recall": None}

        hits = search_archive(username, latest_human.content)
        if not hits:
            return {"recall": None}

        recall_text = "\n".join(f"- {h}" for h in hits)
        return {"recall": recall_text}

    def call_model(state: ChatState) -> dict:
        trimmed = trim_messages(
            state["messages"],
            max_tokens=MAX_CONTEXT_TOKENS,
            token_counter=approx_token_counter,
            strategy="last",
            start_on="human",
        )

        payload = [SystemMessage(content=MOTHER_SYSTEM_PROMPT.format(name=username))]
        summary = state.get("summary")
        if summary:
            payload.append(
                SystemMessage(content=f"What you remember about {username} so far:\n{summary}")
            )
        recall = state.get("recall")
        if recall:
            payload.append(
                SystemMessage(
                    content=(
                        f"Something {username} mentioned a while back that seems "
                        f"relevant right now (only bring it up if it naturally fits):\n{recall}"
                    )
                )
            )
        payload += trimmed

        response = llm.invoke(payload)
        return {"messages": [response]}

    def archive_new_turn(state: ChatState) -> dict:
        archive_messages(username, state["messages"][-2:])
        return {}

    builder = StateGraph(ChatState)
    builder.add_node("maybe_summarize", maybe_summarize)
    builder.add_node("recall_memory", recall_memory)
    builder.add_node("call_model", call_model)
    builder.add_node("archive_new_turn", archive_new_turn)
    builder.add_edge(START, "maybe_summarize")
    builder.add_edge("maybe_summarize", "recall_memory")
    builder.add_edge("recall_memory", "call_model")
    builder.add_edge("call_model", "archive_new_turn")
    builder.add_edge("archive_new_turn", END)

    return builder.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------
# 5. Checkpointer (Postgres-backed, shares the same pool as everything else)
# --------------------------------------------------------------------------

@st.cache_resource
def get_checkpointer() -> PostgresSaver:
    pool = get_db_pool()
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()  # idempotent -- creates checkpoint tables if missing
    return checkpointer


# --------------------------------------------------------------------------
# 6. Streamlit app
# --------------------------------------------------------------------------

st.set_page_config(page_title="Mother & Child Chat", page_icon="💛", layout="centered")

for key, default in {
    "logged_in": False,
    "username": None,
    "graph": None,
    "config": None,
    "display_messages": [],
    "api_key": "",
    "model_name": DEFAULT_MODEL_NAME,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

init_users_table()
init_archive_table()


def bmessage_to_display(m: BaseMessage) -> Optional[dict]:
    if m.type == "human":
        return {"role": "user", "content": m.content}
    if m.type == "ai":
        return {"role": "assistant", "content": m.content}
    return None


def do_logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.graph = None
    st.session_state.config = None
    st.session_state.display_messages = []


# ---------- Sidebar: API key + auth ----------

with st.sidebar:
    st.header("Settings")
    # IMPORTANT: use key= and let Streamlit own the session_state value.
    # Manually doing st.session_state.api_key = st.text_input(..., value=st.session_state.api_key)
    # with no key= causes the widget's identity to shift across reruns
    # (Streamlit derives identity partly from `value` for keyless widgets),
    # which is why the app kept "forgetting" the key you just typed.
    st.text_input(
        "Groq API key",
        type="password",
        key="api_key",
        help="Your key is only kept in this browser session, never stored on disk.",
    )
    st.text_input(
        "Model",
        key="model_name",
    )

    st.divider()

    if not st.session_state.logged_in:
        st.header("Sign in")
        mode = st.radio("I am...", ["Returning (log in)", "New (sign up)"], label_visibility="collapsed")
        username_input = st.text_input("Your name")
        submit = st.button("Continue", use_container_width=True)

        if submit:
            username_input = username_input.strip()
            api_key = st.session_state.api_key.strip()
            if not api_key:
                st.error("Please enter your Groq API key first.")
            elif not username_input:
                st.error("Please enter a name.")
            elif mode.startswith("New") and username_exists(username_input):
                st.error(f"'{username_input}' is already taken. Please choose another name, or log in instead.")
            elif mode.startswith("Returning") and not username_exists(username_input):
                st.error(f"No account found for '{username_input}'. Please sign up first.")
            else:
                is_returning = username_exists(username_input)
                if not is_returning:
                    create_user(username_input)

                checkpointer = get_checkpointer()
                graph = build_graph(
                    checkpointer, username_input,
                    api_key, st.session_state.model_name.strip(),
                )
                config = {"configurable": {"thread_id": username_input}}

                existing_state = graph.get_state(config)
                live_messages = existing_state.values.get("messages", []) if existing_state.values else []

                st.session_state.logged_in = True
                st.session_state.username = username_input
                st.session_state.graph = graph
                st.session_state.config = config
                st.session_state.display_messages = [
                    d for d in (bmessage_to_display(m) for m in live_messages) if d
                ]
                st.rerun()
    else:
        st.success(f"Signed in as **{st.session_state.username}**")
        total_archived = archive_count(st.session_state.username)
        st.caption(f"{total_archived} messages remembered in total")

        with st.expander("View full remembered history"):
            rows = full_transcript(st.session_state.username)
            if not rows:
                st.write("Nothing archived yet.")
            else:
                for role, content, archived_at in rows:
                    speaker = "You" if role == "human" else "Mom" if role == "ai" else role
                    st.markdown(f"**{speaker}:** {content}")

        if st.button("Log out", use_container_width=True):
            do_logout()
            st.rerun()


# ---------- Main: chat ----------

st.title("💛 Talk with Mom")

if not st.session_state.logged_in:
    st.info("Enter your Groq API key and sign in from the sidebar to start chatting.")
else:
    if not st.session_state.display_messages:
        st.markdown(f"*Hi {st.session_state.username}, it's so good to meet you! Let's talk.*")

    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"], avatar="🧒" if msg["role"] == "user" else "💛"):
            st.markdown(msg["content"])

    user_input = st.chat_input("Message Mom...")
    if user_input:
        st.session_state.display_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧒"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="💛"):
            with st.spinner("Mom is thinking..."):
                try:
                    result = st.session_state.graph.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=st.session_state.config,
                    )
                    reply = result["messages"][-1].content
                except Exception as e:
                    reply = f"(Something went wrong talking to the model: {e})"
            st.markdown(reply)

        st.session_state.display_messages.append({"role": "assistant", "content": reply})