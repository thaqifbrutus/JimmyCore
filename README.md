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
```

The first search will return zero results — the catalog table starts empty.
Seed it once:

```bash
python scripts/sync_catalog_job.py
```

That same script also runs daily via `.github/workflows/sync-catalog.yml`.

### Deployment notes

Backend runs on Render's free tier, database on Neon's free tier.

Neon suspends compute after a few minutes of idle time and closes its
connections; Render kills idle TCP connections too. The SQLAlchemy engine in
`db/database.py` is configured with `pool_pre_ping`, `pool_recycle`, and TCP
keepalives to survive both — without those, a stale pooled connection produces
`SSL connection has been closed unexpectedly` on the next request. Neon's
pooled connection string (`-pooler` in the hostname) also requires stripping
`channel_binding=require`, since PgBouncer in transaction mode doesn't support
it.

An [UptimeRobot](https://uptimerobot.com) ping keeps the Render instance from
spinning down after its 15-minute idle window.

## Testing

```bash
pytest -q
```

`110 passed, 5 skipped` as of the last commit.

The five skipped tests in `tests/test_catalog_analyze.py` are integration
tests for the `/catalog/{id}/analyze` endpoint. They require a Postgres
instance and a shared `client` fixture that hasn't been built yet; they're
kept on disk as a specification of the endpoint's expected behaviour. The
rest of the suite runs against in-memory SQLite and mocked network calls.

## Limitations

- **Tabular only.** JimmyCore handles CSV and JSON-array data. It doesn't
  ingest PDFs, Word documents, or web pages yet — that's the next major
  feature.
- **5000-row fetch cap.** The government API has no pagination. When a dataset
  exceeds the cap, JimmyCore tells the model its view may be partial, and the
  model tells you.
- **Free-tier AI quota.** The default OpenRouter model chain uses free
  models. If the daily budget runs out, JimmyCore surfaces a clear
  "quota exhausted, back after the daily reset" message rather than a raw
  error.
- **No chat persistence.** Conversations live in Streamlit session state.
  Reload the tab, lose the thread.
- **Cold-start latency.** The backend runs on Render's free tier, which
  spins down after 15 minutes idle. The UptimeRobot ping usually keeps it
  warm, but the very first request after a long quiet period may take ~20s.
- **National-aggregate rows.** Some government datasets mix a "Malaysia"
  total row with per-state rows. The overview chart currently treats them the
  same, so a national total can dominate the visualization.

## Roadmap

- **Document ingestion** — the next major feature. PDFs and text from
  ministry sites, chunked and embedded alongside the tabular catalog.
- **Chat persistence** — a `chat_messages` table so sessions survive reloads.
- **Agentic multi-dataset queries** — "compare drug arrests to road
  accidents."
- **Better handling of national-aggregate rows** — filter them out of charts
  by default, or split into national/state views.

## License

MIT — see [LICENSE](LICENSE).
