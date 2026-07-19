# """
# Groq + LangGraph "Mother & Child" chatbot -- Streamlit UI.

# Run with:
#     pip install streamlit langgraph langgraph-checkpoint-sqlite langchain-groq python-dotenv
#     streamlit run mother_child_chatbot_streamlit.py

# Design (unchanged from the original CLI version):
# - Every user (child) is their own thread_id -> fully isolated conversation.
# - Sign up with a brand-new username, or log in with an existing one.
#   Usernames are enforced unique -- you cannot sign up with a name that's
#   already taken, and you cannot "log in" to a name that doesn't exist yet.
# - LIVE state (what's sent to the LLM) stays small and bounded:
#     * only the last KEEP_RECENT messages are kept verbatim in the graph state
#     * everything older is folded into a running `summary` (also persisted)
# - NOTHING IS EVER DELETED FROM MEMORY. Before old messages are dropped from
#   live state, they are written verbatim to an `archive` SQLite table keyed
#   by username. So "mother" has:
#     1) an always-available structured summary of the whole relationship
#     2) the last N turns verbatim, for natural short-term context
#     3) a full raw transcript in `archive.db`, queryable any time
# - Because everything is persisted (users.db / checkpoints.db / archive.db)
#   and looked up by username (thread_id), reopening the app and logging back
#   in with the same username picks the conversation up exactly where it left
#   off -- nothing from earlier sessions is lost.

# The Groq API key is entered in the sidebar (never hard-coded / never read
# from an env var), so anyone can bring their own key.
# """

# import os
# import re
# import sqlite3
# from datetime import datetime, timezone
# from typing import Annotated, TypedDict, Optional

# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_core.messages import (
#     BaseMessage,
#     SystemMessage,
#     HumanMessage,
#     RemoveMessage,
#     trim_messages,
# )
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver

# # --------------------------------------------------------------------------
# # Constants
# # --------------------------------------------------------------------------

# USERS_DB_PATH = "users.db"
# CHECKPOINT_DB_PATH = "checkpoints.db"
# ARCHIVE_DB_PATH = "archive.db"
# DEFAULT_MODEL_NAME = "llama-3.3-70b-versatile"

# MAX_CONTEXT_TOKENS = 4000
# SUMMARIZE_THRESHOLD = 30
# KEEP_RECENT = 12

# RECALL_MAX_RESULTS = 3
# RECALL_MIN_WORD_LEN = 4
# RECALL_STOPWORDS = {
#     "what", "when", "where", "were", "there", "your", "about", "have",
#     "with", "this", "that", "from", "they", "them", "then", "does",
#     "remember", "recall", "tell", "told", "said",
# }

# MOTHER_SYSTEM_PROMPT = (
#     "You are speaking as a warm, patient, loving mother talking with your child, {name}. "
#     "You remember everything {name} has ever told you and bring it up naturally when it "
#     "fits, the way a real mother would. You are encouraging, gentle when correcting, and "
#     "genuinely curious about {name}'s day, feelings, and interests. Keep your tone caring "
#     "and age-appropriate, never clinical or robotic."
# )


# # --------------------------------------------------------------------------
# # 1. Users table (login lookup, enforces unique usernames)
# # --------------------------------------------------------------------------

# @st.cache_resource
# def get_users_conn():
#     conn = sqlite3.connect(USERS_DB_PATH, check_same_thread=False)
#     conn.execute("""CREATE TABLE IF NOT EXISTS users (
#         username TEXT PRIMARY KEY,
#         created_at TEXT
#     )""")
#     conn.commit()
#     return conn


# def username_exists(conn: sqlite3.Connection, username: str) -> bool:
#     return conn.execute(
#         "SELECT 1 FROM users WHERE username=?", (username,)
#     ).fetchone() is not None


# def create_user(conn: sqlite3.Connection, username: str):
#     conn.execute(
#         "INSERT INTO users (username, created_at) VALUES (?,?)",
#         (username, datetime.now(timezone.utc).isoformat()),
#     )
#     conn.commit()


# # --------------------------------------------------------------------------
# # 2. Archive table (permanent, unbounded raw transcript)
# # --------------------------------------------------------------------------

# @st.cache_resource
# def get_archive_conn():
#     conn = sqlite3.connect(ARCHIVE_DB_PATH, check_same_thread=False)
#     conn.execute("""CREATE TABLE IF NOT EXISTS archive (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         username TEXT NOT NULL,
#         role TEXT NOT NULL,
#         content TEXT NOT NULL,
#         archived_at TEXT NOT NULL
#     )""")
#     conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_user ON archive(username)")
#     conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS archive_fts USING fts5(
#         content,
#         username UNINDEXED,
#         archive_id UNINDEXED,
#         tokenize='porter'
#     )""")
#     conn.commit()
#     return conn


# def archive_messages(conn: sqlite3.Connection, username: str, messages: list[BaseMessage]):
#     now = datetime.now(timezone.utc).isoformat()
#     rows = [
#         (username, m.type, m.content, now)
#         for m in messages
#         if getattr(m, "content", None)
#     ]
#     if not rows:
#         return
#     cursor = conn.cursor()
#     fts_rows = []
#     for row in rows:
#         cursor.execute(
#             "INSERT INTO archive (username, role, content, archived_at) VALUES (?,?,?,?)",
#             row,
#         )
#         fts_rows.append((row[2], row[0], cursor.lastrowid))
#     cursor.executemany(
#         "INSERT INTO archive_fts (content, username, archive_id) VALUES (?,?,?)",
#         fts_rows,
#     )
#     conn.commit()


# def _extract_keywords(text: str) -> list[str]:
#     words = re.findall(r"[a-zA-Z']+", text.lower())
#     return list(dict.fromkeys(
#         w for w in words
#         if len(w) >= RECALL_MIN_WORD_LEN and w not in RECALL_STOPWORDS
#     ))


# def search_archive(conn: sqlite3.Connection, username: str, query_text: str,
#                     limit: int = RECALL_MAX_RESULTS) -> list[str]:
#     keywords = _extract_keywords(query_text)
#     if not keywords:
#         return []
#     match_expr = " OR ".join(f'"{kw}"' for kw in keywords)
#     try:
#         cursor = conn.execute(
#             """SELECT content FROM archive_fts
#                WHERE archive_fts MATCH ? AND username = ?
#                ORDER BY rank LIMIT ?""",
#             (match_expr, username, limit),
#         )
#         return [row[0] for row in cursor.fetchall()]
#     except sqlite3.OperationalError:
#         return []


# def full_transcript(conn: sqlite3.Connection, username: str) -> list[tuple[str, str, str]]:
#     """Every archived message for this user, oldest first: (role, content, archived_at)."""
#     return conn.execute(
#         "SELECT role, content, archived_at FROM archive WHERE username=? ORDER BY id ASC",
#         (username,),
#     ).fetchall()


# # --------------------------------------------------------------------------
# # 3. Graph state
# # --------------------------------------------------------------------------

# class ChatState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]
#     summary: Optional[str]
#     recall: Optional[str]


# def approx_token_counter(messages: list[BaseMessage]) -> int:
#     """Cheap, dependency-free token estimate (~4 chars/token) used to keep
#     trim_messages() from needing a real tokenizer (avoids the optional
#     `transformers` dependency that some chat models fall back to).
#     """
#     total = 0
#     for m in messages:
#         content = getattr(m, "content", "") or ""
#         total += max(1, len(content) // 4)
#     return total


# # --------------------------------------------------------------------------
# # 4. Build the graph
# # --------------------------------------------------------------------------

# def build_graph(checkpointer: SqliteSaver, archive_conn: sqlite3.Connection,
#                  username: str, api_key: str, model_name: str):
#     llm = ChatGroq(model=model_name, temperature=0.7, api_key=api_key)
#     summarizer = ChatGroq(model=model_name, temperature=0, api_key=api_key)

#     def maybe_summarize(state: ChatState) -> dict:
#         messages = state["messages"]
#         if len(messages) <= SUMMARIZE_THRESHOLD:
#             return {}

#         to_archive = messages[:-KEEP_RECENT]
#         archive_messages(archive_conn, username, to_archive)

#         existing_summary = state.get("summary") or ""
#         archive_text = "\n".join(
#             f"{m.type}: {m.content}" for m in to_archive if getattr(m, "content", None)
#         )
#         summary_prompt = (
#             "You maintain a running summary of an ongoing relationship between a "
#             "mother and her child, for the mother's use. Update the summary below "
#             "to incorporate the new messages. Keep the child's name, personality, "
#             "preferences, important events, feelings, and unresolved things the "
#             "child cares about. Be concise (under 200 words), written as notes a "
#             "caring parent would keep.\n\n"
#             f"EXISTING SUMMARY:\n{existing_summary or '(none yet)'}\n\n"
#             f"NEW MESSAGES TO FOLD IN:\n{archive_text}\n\n"
#             "UPDATED SUMMARY:"
#         )
#         new_summary = summarizer.invoke([HumanMessage(content=summary_prompt)]).content

#         removals = [RemoveMessage(id=m.id) for m in to_archive if m.id is not None]
#         return {"summary": new_summary, "messages": removals}

#     def recall_memory(state: ChatState) -> dict:
#         messages = state["messages"]
#         latest_human = next((m for m in reversed(messages) if m.type == "human"), None)
#         if latest_human is None:
#             return {"recall": None}

#         hits = search_archive(archive_conn, username, latest_human.content)
#         if not hits:
#             return {"recall": None}

#         recall_text = "\n".join(f"- {h}" for h in hits)
#         return {"recall": recall_text}

#     def call_model(state: ChatState) -> dict:
#         trimmed = trim_messages(
#             state["messages"],
#             max_tokens=MAX_CONTEXT_TOKENS,
#             token_counter=approx_token_counter,
#             strategy="last",
#             start_on="human",
#         )

#         payload = [SystemMessage(content=MOTHER_SYSTEM_PROMPT.format(name=username))]
#         summary = state.get("summary")
#         if summary:
#             payload.append(
#                 SystemMessage(content=f"What you remember about {username} so far:\n{summary}")
#             )
#         recall = state.get("recall")
#         if recall:
#             payload.append(
#                 SystemMessage(
#                     content=(
#                         f"Something {username} mentioned a while back that seems "
#                         f"relevant right now (only bring it up if it naturally fits):\n{recall}"
#                     )
#                 )
#             )
#         payload += trimmed

#         response = llm.invoke(payload)
#         return {"messages": [response]}

#     def archive_new_turn(state: ChatState) -> dict:
#         archive_messages(archive_conn, username, state["messages"][-2:])
#         return {}

#     builder = StateGraph(ChatState)
#     builder.add_node("maybe_summarize", maybe_summarize)
#     builder.add_node("recall_memory", recall_memory)
#     builder.add_node("call_model", call_model)
#     builder.add_node("archive_new_turn", archive_new_turn)
#     builder.add_edge(START, "maybe_summarize")
#     builder.add_edge("maybe_summarize", "recall_memory")
#     builder.add_edge("recall_memory", "call_model")
#     builder.add_edge("call_model", "archive_new_turn")
#     builder.add_edge("archive_new_turn", END)

#     return builder.compile(checkpointer=checkpointer)


# # --------------------------------------------------------------------------
# # 5. Streamlit app
# # --------------------------------------------------------------------------

# st.set_page_config(page_title="Mother & Child Chat", page_icon="💛", layout="centered")

# for key, default in {
#     "logged_in": False,
#     "username": None,
#     "graph": None,
#     "config": None,
#     "display_messages": [],
#     "api_key": "",
#     "model_name": DEFAULT_MODEL_NAME,
# }.items():
#     if key not in st.session_state:
#         st.session_state[key] = default

# users_conn = get_users_conn()
# archive_conn = get_archive_conn()


# def bmessage_to_display(m: BaseMessage) -> Optional[dict]:
#     if m.type == "human":
#         return {"role": "user", "content": m.content}
#     if m.type == "ai":
#         return {"role": "assistant", "content": m.content}
#     return None


# def do_logout():
#     st.session_state.logged_in = False
#     st.session_state.username = None
#     st.session_state.graph = None
#     st.session_state.config = None
#     st.session_state.display_messages = []


# # ---------- Sidebar: API key + auth ----------

# with st.sidebar:
#     st.header("Settings")
#     st.session_state.api_key = st.text_input(
#         "Groq API key", value=st.session_state.api_key, type="password",
#         help="Your key is only kept in this browser session, never stored on disk.",
#     )
#     st.session_state.model_name = st.text_input(
#         "Model", value=st.session_state.model_name,
#     )

#     st.divider()

#     if not st.session_state.logged_in:
#         st.header("Sign in")
#         mode = st.radio("I am...", ["Returning (log in)", "New (sign up)"], label_visibility="collapsed")
#         username_input = st.text_input("Your name")
#         submit = st.button("Continue", use_container_width=True)

#         if submit:
#             username_input = username_input.strip()
#             if not st.session_state.api_key:
#                 st.error("Please enter your Groq API key first.")
#             elif not username_input:
#                 st.error("Please enter a name.")
#             elif mode.startswith("New") and username_exists(users_conn, username_input):
#                 st.error(f"'{username_input}' is already taken. Please choose another name, or log in instead.")
#             elif mode.startswith("Returning") and not username_exists(users_conn, username_input):
#                 st.error(f"No account found for '{username_input}'. Please sign up first.")
#             else:
#                 is_returning = username_exists(users_conn, username_input)
#                 if not is_returning:
#                     create_user(users_conn, username_input)

#                 checkpoint_conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
#                 checkpointer = SqliteSaver(checkpoint_conn)
#                 graph = build_graph(
#                     checkpointer, archive_conn, username_input,
#                     st.session_state.api_key, st.session_state.model_name,
#                 )
#                 config = {"configurable": {"thread_id": username_input}}

#                 existing_state = graph.get_state(config)
#                 live_messages = existing_state.values.get("messages", []) if existing_state.values else []

#                 st.session_state.logged_in = True
#                 st.session_state.username = username_input
#                 st.session_state.graph = graph
#                 st.session_state.config = config
#                 st.session_state.display_messages = [
#                     d for d in (bmessage_to_display(m) for m in live_messages) if d
#                 ]
#                 st.rerun()
#     else:
#         st.success(f"Signed in as **{st.session_state.username}**")
#         total_archived = archive_conn.execute(
#             "SELECT COUNT(*) FROM archive WHERE username=?", (st.session_state.username,)
#         ).fetchone()[0]
#         st.caption(f"{total_archived} messages remembered in total")

#         with st.expander("View full remembered history"):
#             rows = full_transcript(archive_conn, st.session_state.username)
#             if not rows:
#                 st.write("Nothing archived yet.")
#             else:
#                 for role, content, archived_at in rows:
#                     speaker = "You" if role == "human" else "Mom" if role == "ai" else role
#                     st.markdown(f"**{speaker}:** {content}")

#         if st.button("Log out", use_container_width=True):
#             do_logout()
#             st.rerun()


# # ---------- Main: chat ----------

# st.title("💛 Talk with Mom")

# if not st.session_state.logged_in:
#     st.info("Enter your Groq API key and sign in from the sidebar to start chatting.")
# else:
#     if not st.session_state.display_messages:
#         st.markdown(f"*Hi {st.session_state.username}, it's so good to meet you! Let's talk.*")

#     for msg in st.session_state.display_messages:
#         with st.chat_message(msg["role"], avatar="🧒" if msg["role"] == "user" else "💛"):
#             st.markdown(msg["content"])

#     user_input = st.chat_input("Message Mom...")
#     if user_input:
#         st.session_state.display_messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user", avatar="🧒"):
#             st.markdown(user_input)

#         with st.chat_message("assistant", avatar="💛"):
#             with st.spinner("Mom is thinking..."):
#                 try:
#                     result = st.session_state.graph.invoke(
#                         {"messages": [{"role": "user", "content": user_input}]},
#                         config=st.session_state.config,
#                     )
#                     reply = result["messages"][-1].content
#                 except Exception as e:
#                     reply = f"(Something went wrong talking to the model: {e})"
#             st.markdown(reply)

#         st.session_state.display_messages.append({"role": "assistant", "content": reply})
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

Everything that used to live in three local SQLite files (users.db,
checkpoints.db, archive.db) now lives in three sets of tables inside the
SAME Neon Postgres database pointed to by DATABASE_URL. Nothing else about
the design changed:

- Every user (child) is their own thread_id -> fully isolated conversation.
- Sign up with a brand-new username, or log in with an existing one.
  Usernames are enforced unique.
- LIVE state (what's sent to the LLM) stays small and bounded:
    * only the last KEEP_RECENT messages are kept verbatim in the graph state
    * everything older is folded into a running `summary` (also persisted)
- NOTHING IS EVER DELETED FROM MEMORY. Before old messages are dropped from
  live state, they are written verbatim to an `archive` table keyed by
  username, searchable via Postgres full-text search (replaces the old
  SQLite FTS5 virtual table -- FTS5 is SQLite-only and has no Postgres
  equivalent, so `to_tsvector` / `to_tsquery` do the same job here).
- Because everything is persisted in Neon and looked up by username
  (thread_id), reopening the app (even from a different machine) and
  logging back in with the same username picks the conversation up
  exactly where it left off.

The Groq API key is still entered in the sidebar (never hard-coded / never
read from an env var), so anyone can bring their own key.
"""

import os
import re
from datetime import datetime, timezone
from typing import Annotated, TypedDict, Optional

import streamlit as st
from dotenv import load_dotenv
import psycopg
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
# 1. Users table (login lookup, enforces unique usernames)
# --------------------------------------------------------------------------

@st.cache_resource
def get_users_conn():
    _require_database_url()
    conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=tuple_row)
    conn.execute("""CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        created_at TEXT
    )""")
    return conn


def username_exists(conn: psycopg.Connection, username: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM users WHERE username=%s", (username,)
    ).fetchone() is not None


def create_user(conn: psycopg.Connection, username: str):
    conn.execute(
        "INSERT INTO users (username, created_at) VALUES (%s,%s)",
        (username, datetime.now(timezone.utc).isoformat()),
    )


# --------------------------------------------------------------------------
# 2. Archive table (permanent, unbounded raw transcript)
# --------------------------------------------------------------------------

@st.cache_resource
def get_archive_conn():
    _require_database_url()
    conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=tuple_row)
    conn.execute("""CREATE TABLE IF NOT EXISTS archive (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        archived_at TEXT NOT NULL
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_user ON archive(username)")
    # Full-text search replacement for SQLite's FTS5 virtual table:
    # a GIN index over to_tsvector(content) lets `to_tsquery` searches
    # (used in search_archive below) run fast without a separate shadow table.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_archive_fts "
        "ON archive USING GIN (to_tsvector('english', content))"
    )
    return conn


def archive_messages(conn: psycopg.Connection, username: str, messages: list[BaseMessage]):
    now = datetime.now(timezone.utc).isoformat()
    rows = [
        (username, m.type, m.content, now)
        for m in messages
        if getattr(m, "content", None)
    ]
    if not rows:
        return
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


def search_archive(conn: psycopg.Connection, username: str, query_text: str,
                    limit: int = RECALL_MAX_RESULTS) -> list[str]:
    keywords = _extract_keywords(query_text)
    if not keywords:
        return []
    # OR the keywords together, same intent as the old FTS5 MATCH expression.
    tsquery_expr = " | ".join(keywords)
    try:
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


def full_transcript(conn: psycopg.Connection, username: str) -> list[tuple[str, str, str]]:
    """Every archived message for this user, oldest first: (role, content, archived_at)."""
    return conn.execute(
        "SELECT role, content, archived_at FROM archive WHERE username=%s ORDER BY id ASC",
        (username,),
    ).fetchall()


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

def build_graph(checkpointer: PostgresSaver, archive_conn: psycopg.Connection,
                 username: str, api_key: str, model_name: str):
    llm = ChatGroq(model=model_name, temperature=0.7, api_key=api_key)
    summarizer = ChatGroq(model=model_name, temperature=0, api_key=api_key)

    def maybe_summarize(state: ChatState) -> dict:
        messages = state["messages"]
        if len(messages) <= SUMMARIZE_THRESHOLD:
            return {}

        to_archive = messages[:-KEEP_RECENT]
        archive_messages(archive_conn, username, to_archive)

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

        hits = search_archive(archive_conn, username, latest_human.content)
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
        archive_messages(archive_conn, username, state["messages"][-2:])
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
# 5. Checkpointer (Postgres-backed, replaces SqliteSaver)
# --------------------------------------------------------------------------

@st.cache_resource
def get_checkpointer() -> PostgresSaver:
    _require_database_url()
    # autocommit + no statement caching, matching langgraph's documented
    # Postgres setup; the connection is kept open for the app's lifetime
    # via st.cache_resource, same pattern as the old SQLite connection.
    conn = psycopg.connect(DATABASE_URL, autocommit=True, prepare_threshold=0)
    checkpointer = PostgresSaver(conn)
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

users_conn = get_users_conn()
archive_conn = get_archive_conn()


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
    st.session_state.api_key = st.text_input(
        "Groq API key", value=st.session_state.api_key, type="password",
        help="Your key is only kept in this browser session, never stored on disk.",
    )
    st.session_state.model_name = st.text_input(
        "Model", value=st.session_state.model_name,
    )

    st.divider()

    if not st.session_state.logged_in:
        st.header("Sign in")
        mode = st.radio("I am...", ["Returning (log in)", "New (sign up)"], label_visibility="collapsed")
        username_input = st.text_input("Your name")
        submit = st.button("Continue", use_container_width=True)

        if submit:
            username_input = username_input.strip()
            if not st.session_state.api_key:
                st.error("Please enter your Groq API key first.")
            elif not username_input:
                st.error("Please enter a name.")
            elif mode.startswith("New") and username_exists(users_conn, username_input):
                st.error(f"'{username_input}' is already taken. Please choose another name, or log in instead.")
            elif mode.startswith("Returning") and not username_exists(users_conn, username_input):
                st.error(f"No account found for '{username_input}'. Please sign up first.")
            else:
                is_returning = username_exists(users_conn, username_input)
                if not is_returning:
                    create_user(users_conn, username_input)

                checkpointer = get_checkpointer()
                graph = build_graph(
                    checkpointer, archive_conn, username_input,
                    st.session_state.api_key, st.session_state.model_name,
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
        total_archived = archive_conn.execute(
            "SELECT COUNT(*) FROM archive WHERE username=%s", (st.session_state.username,)
        ).fetchone()[0]
        st.caption(f"{total_archived} messages remembered in total")

        with st.expander("View full remembered history"):
            rows = full_transcript(archive_conn, st.session_state.username)
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