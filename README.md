# JimmyCore

**A RAG analyst over Malaysian government open data.**

Search for a topic in plain English, pick a dataset, and have a conversation
about it. JimmyCore grounds every answer in the actual rows — computed on the
fly, not recalled from the model's memory — and visualizes what it finds.

**[Try it live →](https://jimmycore.streamlit.app)** · [API](https://jimmycore-api.onrender.com)

> Screenshot: 
<img width="1919" height="993" alt="image" src="https://github.com/user-attachments/assets/f7e33907-7c11-4449-828e-211b028d4359" />

<img width="1919" height="915" alt="image" src="https://github.com/user-attachments/assets/29ab6ff0-ef79-4f65-928e-11aee6006ac9" />

<img width="1919" height="991" alt="image" src="https://github.com/user-attachments/assets/03983b60-34f2-46e1-9c3c-755be4eb74e5" />

<img width="1919" height="909" alt="image" src="https://github.com/user-attachments/assets/78a55115-7bef-4408-88de-90e3b3c57619" />

---

## What it does

JimmyCore is a search-and-chat layer on top of Malaysia's official open-data
catalog ([data.gov.my](https://data.gov.my)). Underneath, it's a small RAG
system: semantic search over a local index, plus a tool-calling agent that
queries the underlying dataset.

- **Search** — free-text queries are embedded and matched against a local
  copy of the government's dataset catalog. No more clicking through
  navigation trees to find the right CSV.
- **Understand** — every dataset opens with a plain-English overview, a
  chart of its most informative dimension, and suggested starter questions.
- **Ask** — chat about the data. Jimmy has six tools (`filter_rows`,
  `aggregate`, `value_counts`, `describe_column`, `get_sample_rows`,
  `get_schema`) that it calls against the real DataFrame, so answers come from
  computed results rather than the model's memory.
- **Visualize** — when a question is best answered with a breakdown or a
  trend, the chat renders a chart below the response.

## How it works

Three services:

1. A **catalog sync** job pulls the official dataset list
   ([data.gov.my's parquet dump](https://storage.data.gov.my/metrics/dataset_list.parquet))
   into a Postgres table and generates a vector embedding for each row.
2. When you search, your query is embedded and ranked by cosine similarity
   against every catalog row. At this catalog's scale (hundreds of datasets,
   not millions), a brute-force pass is instant — no pgvector required.
3. When you pick a dataset, JimmyCore fetches the real rows from
   `api.data.gov.my` (respecting their 4-requests/minute rate limit, with a
   60-minute cache), profiles the shape, and asks an LLM to write an overview
   plus suggested questions plus a chart hint.
4. When you chat, the model runs a tool-calling loop against the actual
   DataFrame. Every tool result is capped, repeated calls are caught, and
   the loop has a wrap-up fallback if the model gets stuck.

**Stack:** FastAPI · Streamlit · PostgreSQL (via Neon) · OpenRouter ·
Alembic · pandas · SQLAlchemy.

## Running it locally

Prerequisites: Python 3.12, a Postgres database (Neon's free tier works),
an [OpenRouter API key](https://openrouter.ai).

```bash
git clone https://github.com/thaqifbrutus/JimmyCore
cd JimmyCore
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in DATABASE_URL and OPENROUTER_API_KEY

alembic stamp 0001_baseline      # marks the baseline as already-applied
alembic upgrade head             # runs the real migrations

# Backend
uvicorn app.main:app --reload

# Frontend (in a second terminal)
streamlit run frontend.py
