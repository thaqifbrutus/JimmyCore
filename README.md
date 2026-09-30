# JimmyCore

**A RAG analyst over Malaysian government open data.**

Search for a topic in plain English, pick a dataset, and have a conversation
about it. JimmyCore grounds every answer in the actual rows — computed on the
fly, not recalled from the model's memory — and visualizes what it finds.

**[Try it live →](https://jimmycore.streamlit.app)** · [API](https://jimmycore-api.onrender.com)

> Screenshot: the chat view showing "What the data *can* tell you" / "What the
> data *cannot* tell you" — insert the Kelantan opiate cases screenshot here.

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
  trend, the chat renders a chart below the response. No extra clicks.
- **Stick to the data** — when the answer isn't in the dataset, Jimmy says so,
  cites the source agency, and stops. See the sample exchange below.

## A sample exchange

> **You:** Why did Kelantan suddenly have a massive drop in opiate cases from
> 2020 to 2023?
>
> **Jimmy:** The data shows a gradual decline from 2020–2022, then a slight
> uptick in 2023 — not a sudden drop. Here are the exact figures:
>
> | Year | Kelantan Opiate Cases |
> |------|----------------------|
> | 2019 | 158 |
> | 2020 | 157 |
> | 2021 | 56 |
> | 2022 | 37 |
> | 2023 | 77 |
>
> …
>
> ### What the data *cannot* tell you
>
> The dataset contains only counts by state, year, and drug type. It has no
> fields for:
>
> - Enforcement operations, policy changes, or treatment capacity
> - Demographics, socioeconomic factors, or reporting practices
> - Data collection methodology changes
>
> Any causal explanation (policy, enforcement, reporting changes, pandemic
> effects, etc.) would require external information not present in this
> dataset.
>
> *Source: AADK, KDN (Official Government Dataset, Public Safety → Drug
> Addiction, 2015–2023).*

The refusal to invent a cause is the part I care most about. A general-purpose
LLM will happily produce a plausible-sounding explanation. JimmyCore points
you at the source when a real decision is on the line.

> Screenshot: the chat view showing the tool-call trail button and the
> per-year bar chart — insert the Johor/Kedah/Kelantan table screenshot here.

## How it works

Three services, one flow.

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