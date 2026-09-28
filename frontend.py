import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"
# API_BASE = "https://jimmycore-production.up.railway.app"

st.set_page_config(
    page_title="JimmyCore",
    page_icon="🧠",
    layout="wide"
)

st.title("JimmyCore")
st.caption("Search official government datasets, or upload your own CSV — ask questions either way")
st.divider()


# ── API helpers — upload flow ──────────────────────────────────────────────

def upload_file(file):
    response = requests.post(
        f"{API_BASE}/upload",
        files={"file": (file.name, file.getvalue(), "text/csv")}
    )
    return response.json() if response.status_code == 200 else None


def trigger_profile(dataset_id):
    response = requests.post(f"{API_BASE}/reports/datasets/{dataset_id}/profile")
    return response.json() if response.status_code == 200 else None


# ── API helpers — government catalog search flow ───────────────────────────

def search_catalog(query, top_k=5):
    """
    Returns (results, error_message) — one is always None. Distinguishing
    a genuine search failure from "search worked, zero matches" so the UI
    can show the right message for each rather than treating both as
    silent nothing.
    """
    try:
        response = requests.get(
            f"{API_BASE}/catalog/search",
            params={"q": query, "top_k": top_k},
        )
    except requests.exceptions.RequestException as e:
        return None, f"Could not reach the API: {e}"

    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith("application/json") else response.text
        return None, detail

    return response.json().get("results", []), None


def analyze_catalog_dataset(catalog_dataset_id, force_refresh=False):
    response = requests.post(
        f"{API_BASE}/catalog/{catalog_dataset_id}/analyze",
        params={"force_refresh": force_refresh},
    )
    if response.status_code == 200:
        return response.json(), None

    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return None, detail


def ask_question(report_id, question, history):
    response = requests.post(
        f"{API_BASE}/reports/{report_id}/ask",
        json={"question": question, "conversation_history": history}
    )
    return response.json() if response.status_code == 200 else None


# ── Result dict helpers ────────────────────────────────────────────────────
# AI functions return {"status": "ok"|"failed", "reason": ..., "content": ...}.
# extract_ai_content pulls .content out safely and surfaces failures.

_QUOTA_EXHAUSTED_MESSAGE = (
    "AI quota exhausted for today. Jimmy will be back after the free-tier "
    "daily reset (midnight UTC)."
)


def extract_ai_content(result_dict, field_name="content"):
    """
    Pulls .content out of a structured AI result dict.
    Returns (content, error_message) — one is always None.

    Special-cases the daily-quota error so the user sees an explanatory
    message rather than the raw OpenRouter 429 payload.
    """
    if result_dict is None:
        return None, "No response received from the API."
    if isinstance(result_dict, str):
        # Old-shape response from an endpoint not yet updated — pass through.
        return result_dict, None
    if result_dict.get("error_type") == "daily_quota_exhausted":
        return None, _QUOTA_EXHAUSTED_MESSAGE
    status = result_dict.get("status")
    if status == "ok":
        return result_dict.get(field_name), None
    elif status == "failed":
        reason = result_dict.get("reason") or "Unknown error."
        return None, reason
    # Unexpected shape — surface raw so nothing is silently swallowed.
    return None, f"Unexpected response shape: {result_dict}"


def render_failed_ai(label: str, reason: str):
    """Consistent UI treatment for a failed AI result."""
    st.warning(
        f"⚠️ **{label} could not be generated.**\n\n"
        f"Reason: {reason}",
        icon=None
    )


def _render_chart(chart_data: list[dict] | None, x_label: str | None = None,
                  y_label: str | None = None):
    """
    Render a small bar chart from a chart_data list of {key, value} dicts.

    x_label / y_label are passed through to st.bar_chart so the axes are
    labeled. Without them, the y-axis renders as bare numbers and the
    x-axis as bare category names — fine if you already know what the
    chart is showing, unreadable otherwise. st.bar_chart is backed by
    Altair/Vega-Lite, which uses these as both the axis titles and the
    tooltip field labels.

    The chart is a presentation bonus — a broken chart must not break the
    page it renders on, so the whole thing is wrapped in try/except and
    silently skipped on any failure. This is the one place in the codebase
    where silent failure is acceptable, and it's on purpose.
    """
    if not chart_data:
        return
    try:
        import pandas as pd
        chart_df = pd.DataFrame(chart_data).set_index(
            list(chart_data[0].keys())[0]
        )
        st.bar_chart(chart_df, x_label=x_label, y_label=y_label)
    except Exception:
        pass  # charts are a bonus; never break the page over one


def _chart_axis_labels(tool_name: str | None, tool_args: dict | None,
                       chart_data: list[dict] | None) -> tuple[str | None, str | None]:
    """
    Derive human-readable x/y axis labels from a chat tool call's name and
    arguments. Used only for chat charts (which come from tool responses
    where we know exactly which tool produced them). The overview chart
    uses the kind/metric fields on the chart dict directly.
    """
    args = tool_args or {}
    if tool_name == "value_counts":
        return args.get("column"), "count"
    if tool_name == "aggregate":
        group = args.get("group_by")
        agg_col = args.get("agg_column")
        agg_func = args.get("agg_func")
        if agg_col and agg_func:
            return group, f"{agg_func} of {agg_col}"
        return group, agg_func or agg_col
    # Unknown tool — pull labels from the data keys as a last resort.
    if chart_data:
        keys = list(chart_data[0].keys())
        if len(keys) >= 2:
            return keys[0], keys[1]
    return None, None


def _render_tool_calls_button(tool_calls: list):
    """
    Renders the tool-call trail behind a single discreet icon button.
    Hidden unless the user explicitly opens it — the answer and its chart
    are the visible output; the tool trail is for verification.

    Uses st.popover (Streamlit >= 1.31). AppTest element accessibility for
    popovers varies across Streamlit versions, so the test that asserts
    this button exists checks the underlying data rather than the widget.
    """
    if not tool_calls:
        return
    with st.popover(f"⚙️ Tool calls ({len(tool_calls)})"):
        for i, call in enumerate(tool_calls, 1):
            if i > 1:
                st.markdown("---")
            st.markdown(f"**{i}. `{call['name']}`**")
            st.code(json.dumps(call.get("arguments", {}), indent=2), language="json")
            st.caption(call.get("result_summary", "—"))


# ── Session state initialisation ───────────────────────────────────────────

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Search government data"
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "report_id" not in st.session_state:
    st.session_state.report_id = None
if "profile_result" not in st.session_state:
    st.session_state.profile_result = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_error" not in st.session_state:
    st.session_state.search_error = None
if "source_label" not in st.session_state:
    st.session_state.source_label = None
if "source_kind" not in st.session_state:
    st.session_state.source_kind = None  # "government" | "upload" | None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "_search_triggered" not in st.session_state:
    st.session_state._search_triggered = False


def _queue_search():
    """Callback that fires when the user presses Enter in the search box."""
    st.session_state._search_triggered = True


def _reset_all():
    for key in [
        "dataset_id", "report_id", "profile_result",
        "search_results", "search_error", "source_label", "source_kind",
    ]:
        st.session_state[key] = None
    st.session_state.suggested_questions = []
    st.session_state.chat_history = []
    st.session_state.messages = []


def _do_ask(question: str):
    """
    Submits one question to /ask, appends user + assistant turns to
    st.session_state.messages, and updates chat_history on success.

    The assistant message carries tool_calls from the response so the
    render loop can show the inline chart and the tool-calls popover.
    """
    st.session_state.messages.append({"role": "user", "content": question})

    response = ask_question(
        st.session_state.report_id, question, st.session_state.chat_history
    )

    answer_raw = response.get("answer") if response else None
    answer_content, answer_error = extract_ai_content(answer_raw)
    tool_log = (response or {}).get("tool_calls_log", []) or []

    if answer_error:
        content = f"⚠️ {answer_error}"
    elif answer_content:
        content = answer_content
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": content})
    else:
        content = "⚠️ No response from the API."

    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "tool_calls": tool_log,
    })


# ── Step 1: choose input method ────────────────────────────────────────────

st.subheader("Step 1 — Find a dataset")

st.session_state.input_mode = st.radio(
    "How do you want to start?",
    ["Search government data", "Upload a CSV"],
    horizontal=True,
    label_visibility="collapsed",
)

if st.session_state.input_mode == "Search government data" and not st.session_state.profile_result:
    st.caption("Search official Malaysian government open data (data.gov.my) by topic")

    search_col, button_col = st.columns([5, 1])
    with search_col:
        query = st.text_input(
            "Search query",
            placeholder="e.g. drunk driving accidents, fuel prices, unemployment rate",
            label_visibility="collapsed",
            on_change=_queue_search,
        )
    with button_col:
        run_search = st.button("🔍 Search", type="primary", use_container_width=True)

    search_was_triggered = st.session_state._search_triggered
    st.session_state._search_triggered = False

    if run_search or (search_was_triggered and query.strip()):
        with st.spinner("Searching official datasets..."):
            results, error = search_catalog(query.strip())
            st.session_state.search_results = results
            st.session_state.search_error = error

    if st.session_state.search_error:
        st.error(f"Search failed: {st.session_state.search_error}")

    elif st.session_state.search_results is not None:
        if len(st.session_state.search_results) == 0:
            st.info("No matching datasets found — try a broader or different search term.")
        else:
            st.markdown(f"**Found {len(st.session_state.search_results)} matching datasets:**")
            for result in st.session_state.search_results:
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(f"**{result['title_en']}**")
                        meta_bits = [
                            b for b in [
                                result.get("category_en"),
                                result.get("subcategory_en"),
                                result.get("source"),
                            ] if b
                        ]
                        if meta_bits:
                            st.caption(" · ".join(meta_bits))
                        if result.get("dataset_begin") and result.get("dataset_end"):
                            st.caption(f"Coverage: {result['dataset_begin']}–{result['dataset_end']}")
                        st.caption(f"Match relevance: {result['score']:.0%}")
                    with c2:
                        if st.button("Analyze", key=f"analyze_{result['id']}", use_container_width=True):
                            with st.spinner(f"Fetching and analyzing \"{result['title_en']}\"..."):
                                analysis, error = analyze_catalog_dataset(result["id"])
                                if analysis:
                                    st.session_state.profile_result = analysis
                                    st.session_state.report_id = analysis["report_id"]
                                    st.session_state.source_label = result["title_en"]
                                    st.session_state.source_kind = "government"
                                    st.rerun()
                                else:
                                    st.error(f"Analysis failed: {error}")

elif st.session_state.input_mode == "Upload a CSV" and not st.session_state.profile_result:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload any CSV file up to 10MB"
    )

    if uploaded_file and not st.session_state.dataset_id:
        with st.spinner("Uploading..."):
            result = upload_file(uploaded_file)
            if result:
                st.session_state.dataset_id = result["dataset_id"]
                st.session_state.source_label = result["original_name"]
                st.session_state.source_kind = "upload"
                st.success(f"✅ Uploaded: **{result['original_name']}**")
            else:
                st.error("Upload failed. Check your API is running.")

    if st.session_state.dataset_id and not st.session_state.profile_result:
        if st.button("🔍 Run AI Analysis", type="primary"):
            with st.spinner("Analyzing dataset..."):
                result = trigger_profile(st.session_state.dataset_id)
                if result:
                    st.session_state.profile_result = result
                    st.session_state.report_id = result["report_id"]
                    st.rerun()
                else:
                    st.error("Profiling failed. Check your API logs.")

st.divider()


# ── Step 2: detail view — overview + chart + suggested questions + chat ────
# Shared by both flows — /reports/datasets/{id}/profile (upload) and
# /catalog/{id}/analyze (search) return the same shape, so everything
# below renders identically regardless of where the data came from.

if st.session_state.profile_result:
    result = st.session_state.profile_result

    back_col, _ = st.columns([1, 5])
    with back_col:
        if st.button("← Back to search", use_container_width=True):
            _reset_all()
            st.rerun()

    title = st.session_state.source_label or "Dataset"
    st.title(title)
    if st.session_state.source_kind == "government":
        st.caption("Official government dataset")
    elif st.session_state.source_kind == "upload":
        st.caption(f"Uploaded file: {st.session_state.source_label}")

    # ── Overview ────────────────────────────────────────────────────────
    st.markdown("### Overview")
    overview_raw = result.get("overview")
    overview_content, overview_error = extract_ai_content(overview_raw)

    if overview_error:
        render_failed_ai("Overview", overview_error)
    elif isinstance(overview_content, dict):
        st.markdown(overview_content.get("overview", ""))

        # Overview chart — sits between the prose and the suggested
        # questions below. Never inside an expander: this is part of the
        # answer, not metadata about it. Axis labels and caption branch
        # on the chart's "kind" — sum-based charts (from the model's
        # primary_metric hint) read "sum of X by Y", count-based charts
        # read "count of X".
        chart = overview_raw.get("chart") if isinstance(overview_raw, dict) else None
        if chart and chart.get("values"):
            chart_kind = chart.get("kind", "count")
            chart_metric = chart.get("metric")
            if chart_kind == "sum" and chart_metric:
                y_label = f"sum of {chart_metric}"
                chart_caption = (
                    f"Top {len(chart['values'])} `{chart['column']}` "
                    f"by `{chart_metric}`"
                )
            else:
                y_label = "count"
                chart_caption = (
                    f"Top {len(chart['values'])} values of `{chart['column']}`"
                )
            _render_chart(
                [{"value": v["value"], "count": v["count"]} for v in chart["values"]],
                x_label=chart.get("column"),
                y_label=y_label,
            )
            st.caption(chart_caption)

        st.session_state.suggested_questions = (
            overview_content.get("suggested_questions", []) or []
        )
    elif overview_content:
        st.markdown(overview_content)

    # ── Suggested questions ─────────────────────────────────────────────
    if st.session_state.suggested_questions:
        st.markdown("### Try asking")
        sq_cols = st.columns(2)
        for i, q in enumerate(st.session_state.suggested_questions):
            with sq_cols[i % 2]:
                if st.button(q, key=f"suggested_{i}", use_container_width=True):
                    with st.spinner("Thinking..."):
                        _do_ask(q)
                    st.rerun()

    # ── Chat ────────────────────────────────────────────────────────────
    st.markdown("### Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # Inline chart from the most recent tool call that has one.
                # Shown directly, not in an expander — the answer and its
                # visualization belong together. Reverse-iterate so the
                # latest chartable call wins when a message has multiple.
                chart_call = None
                for call in reversed(msg.get("tool_calls", []) or []):
                    if call.get("chart_data"):
                        chart_call = call
                        break
                if chart_call:
                    x_label, y_label = _chart_axis_labels(
                        chart_call.get("name"),
                        chart_call.get("arguments"),
                        chart_call.get("chart_data"),
                    )
                    _render_chart(
                        chart_call["chart_data"],
                        x_label=x_label,
                        y_label=y_label,
                    )

                # Demoted tool-call trail — single discreet icon button.
                _render_tool_calls_button(msg.get("tool_calls", []) or [])

    if prompt := st.chat_input("Ask a question about this dataset..."):
        with st.spinner("Thinking..."):
            _do_ask(prompt)
        st.rerun()

else:
    if st.session_state.input_mode == "Search government data":
        st.info("Search for a topic above to find official government datasets.")
    else:
        st.info("Upload a CSV file above to get started.")


# ── Reset button ───────────────────────────────────────────────────────────

if st.session_state.dataset_id or st.session_state.profile_result:
    st.divider()
    if st.button("🔄 Start over"):
        _reset_all()
        st.rerun()