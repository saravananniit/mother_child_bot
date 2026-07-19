
# 💛 Mother & Child Chatbot

A memory-rich AI chatbot built with **LangGraph**, **Groq**, **PostgreSQL (Neon)**, and **Streamlit**.

The chatbot acts as a loving and caring mother who remembers conversations over time while maintaining efficient context management.

---

## ✨ Features

### Persistent Long-Term Memory

The application stores:

- User accounts
- Conversation checkpoints
- Complete conversation archives

All data is stored in **Neon PostgreSQL**, allowing conversations to continue seamlessly across sessions and devices.

---

### Smart Memory Architecture

Instead of sending the entire conversation to the LLM every time:

- Recent messages remain in active context
- Older messages are automatically summarized
- Full transcripts are permanently archived
- Relevant memories can be recalled using semantic search

This keeps token usage low while preserving complete conversation history.

---

### User Authentication

Users can:

- Sign up with a unique username
- Log in with an existing username
- Maintain isolated conversation threads

Each username maps to its own LangGraph thread.

---

### Context Compression

The system automatically:

1. Keeps the most recent messages in memory
2. Summarizes older conversations
3. Stores raw transcripts permanently
4. Recalls relevant memories when needed

This enables effectively unlimited memory without growing prompt sizes indefinitely.

---

## 🏗️ Tech Stack

- Streamlit
- LangGraph
- LangChain
- Groq LLMs
- Neon PostgreSQL
- Psycopg
- Python 3.11+

---

## 📂 Project Structure

```text
.
├── mother_child_chatbot_streamlit.py
├── requirements.txt
├── .env
└── README.md
```

---

## 📦 Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / Mac

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔧 Environment Variables

Create a `.env` file in the project root.

```env
DATABASE_URL=postgresql://username:password@your-neon-host/database?sslmode=require
```

### Important

- Use the connection string from your Neon dashboard.
- Ensure `sslmode=require` is included.
- The Groq API key is **NOT stored in the database**.
- Users provide their own Groq API key through the Streamlit sidebar.

---

## 🗄️ Neon Database Setup

### Step 1: Create a Neon Account

Visit:

```text
https://neon.tech
```

Create a PostgreSQL project.

---

### Step 2: Copy Connection String

Navigate to:

```text
Dashboard → Project → Connection Details
```

Copy:

```text
postgresql://username:password@host/database?sslmode=require
```

Add it to your `.env` file.

---

### Step 3: Database Initialization

No manual SQL setup is required.

The application automatically creates:

#### users

Stores registered usernames.

#### archive

Stores complete conversation history.

#### LangGraph Checkpoint Tables

Created automatically by:

```python
checkpointer.setup()
```

---

## 🚀 Run Locally

Start the Streamlit application:

```bash
streamlit run mother_child_chatbot_streamlit.py
```

Open:

```text
http://localhost:8501
```

---

# ☁️ Deploying to Streamlit Community Cloud

## Step 1: Push Code to GitHub

Create a GitHub repository and push:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <repo-url>
git push -u origin main
```

---

## Step 2: Sign in to Streamlit Cloud

Open:

```text
https://share.streamlit.io
```

Login using GitHub.

---

## Step 3: Create a New App

Click:

```text
New App
```

Select:

- Repository
- Branch
- Main file path

Example:

```text
mother_child_chatbot_streamlit.py
```

---

## Step 4: Configure Secrets

In Streamlit Cloud:

```text
App Settings → Secrets
```

Add:

```toml
DATABASE_URL = "postgresql://username:password@host/database?sslmode=require"
```

---

## Step 5: Use Streamlit Secrets (Recommended)

Replace:

```python
DATABASE_URL = os.getenv("DATABASE_URL")
```

with:

```python
DATABASE_URL = st.secrets.get(
    "DATABASE_URL",
    os.getenv("DATABASE_URL")
)
```

This allows both:

- Local `.env`
- Streamlit Cloud Secrets

to work seamlessly.

---

## ✅ requirements.txt

Example:

```text
streamlit
langgraph
langgraph-checkpoint-postgres
langchain
langchain-groq
python-dotenv
psycopg[binary]
```

Recommended pinned versions:

```text
streamlit>=1.45.0
langgraph>=0.6.0
langgraph-checkpoint-postgres>=2.0.0
langchain>=0.3.0
langchain-groq>=0.3.0
python-dotenv>=1.0.0
psycopg[binary]>=3.2.0
```

---

## 🔐 Security Notes

- Groq API keys are never stored in PostgreSQL.
- Groq API keys are never committed to GitHub.
- Neon credentials should be stored in:
  - `.env` locally
  - Streamlit Secrets in production
- Each user conversation is isolated using a unique LangGraph `thread_id`.

---

## 🧠 Memory Flow

```text
User Message
      │
      ▼
Recent Context
      │
      ▼
LangGraph State
      │
      ├── Summarization
      │
      ├── Recall Search
      │
      └── Checkpoint Storage
      ▼
Neon PostgreSQL
      │
      ├── users
      ├── archive
      └── checkpoint tables
```

---

## 💡 Model Support

The application supports any Groq chat model.

Examples:

```text
llama-3.3-70b-versatile
llama-3.1-8b-instant
deepseek-r1-distill-llama-70b
gemma2-9b-it
```

Users can change the model directly from the sidebar.

---

## 📜 License

MIT License

Feel free to modify, extend, and use this project for learning, experimentation, and production deployments.

---

## ❤️ Acknowledgements

Built using:

- Streamlit
- LangGraph
- LangChain
- Groq
- Neon PostgreSQL

Special thanks to the open-source communities that make modern AI applications possible.


One important change before deploying to Streamlit Cloud:

Update your code from:

DATABASE_URL = os.getenv("DATABASE_URL")


to:

DATABASE_URL = st.secrets.get(
    "DATABASE_URL",
    os.getenv("DATABASE_URL")
)


This ensures the app works both locally (.env) and on Streamlit Cloud (Secrets).