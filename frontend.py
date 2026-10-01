import json
import os

import requests
import streamlit as st

API_BASE = os.getenv("JIMMYCORE_API_URL", "http://localhost:8000")
APP_URL_BASE = os.getenv("JIMMYCORE_APP_URL", "http://localhost:8501")

st.set_page_config(
    page_title="JimmyCore",
    page_icon="🧠",
    layout="wide"
)

st.title("JimmyCore")
st.caption("Search official government datasets, or upload your own CSV — ask questions either way")
st.divider()


# ── API helpers ────────────────────────────────────────────────────────────

def upload_file(file):
    response = requests.post(
        f"{API_BASE}/upload",
        files={"file": (file.name, file.getvalue(), "text/csv")}
    )
    return response.json() if response.status_code == 200 else None


def trigger_profile(dataset_id):
    response = requests.post(f"{API_BASE}/reports/datasets/{dataset_id}/profile")
    return response.json() if response.status_code == 200 else None


def search_catalog(query, top_k=5):
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


def load_report(report_id):
    try:
        response = requests.get(f"{API_BASE}/reports/{report_id}", timeout=30)
    except requests.exceptions.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


# ── Result dict helpers ────────────────────────────────────────────────────

_QUOTA_EXHAUSTED_MESSAGE = (
    "AI quota exhausted for today. Jimmy will be back after the free-tier "
    "daily reset (midnight UTC)."
)


def extract_ai_content(result_dict, field_name="content"):
    if result_dict is None:
        return None, "No response received from the API."
    if isinstance(result_dict, str):
        return result_dict, None
    if result_dict.get("error_type") == "daily_quota_exhausted":
        return None, _QUOTA_EXHAUSTED_MESSAGE
    status = result_dict.get("status")
    if status == "ok":
        return result_dict.get(field_name), None
    elif status == "failed":
        reason = result_dict.get("reason") or "Unknown error."
        return None, reason
    return None, f"Unexpected response shape: {result_dict}"


def render_failed_ai(label: str, reason: str):
    st.warning(
        f"⚠️ **{label} could not be generated.**\n\n"
        f"Reason: {reason}",
        icon=None
    )


def _render_chart(chart_data, x_label=None, y_label=None):
    if not chart_data:
        return
    try:
        import pandas as pd
        chart_df = pd.DataFrame(chart_data).set_index(
            list(chart_data[0].keys())[0]
        )
        st.bar_chart(chart_df, x_label=x_label, y_label=y_label)
    except Exception:
        pass


def _chart_axis_labels(tool_name, tool_args, chart_data):
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
    if chart_data:
        keys = list(chart_data[0].keys())
        if len(keys) >= 2:
            return keys[0], keys[1]
    return None, None


def _render_tool_calls_button(tool_calls):
    if not tool_calls:
        return
    with st.popover(f"⚙️ Tool calls ({len(tool_calls)})"):
        for i, call in enumerate(tool_calls, 1):
            if i > 1:
                st.markdown("---")
            st.markdown(f"**{i}. `{call['name']}`**")
            st.code(json.dumps(call.get("arguments", {}), indent=2), language="json")
            st.caption(call.get("result_summary", "—"))


def _build_markdown_export(profile_result, messages, source_label):
    lines = [f"# {source_label or 'JimmyCore session'}", ""]
    ov = (profile_result or {}).get("overview") or {}
    content = ov.get("content") if isinstance(ov, dict) else None
    if isinstance(content, dict):
        lines.append("## Overview")
        lines.append("")
        lines.append(content.get("overview", ""))
        lines.append("")
    lines.append("## Conversation")
    lines.append("")
    for msg in messages:
        role = "**You**" if msg["role"] == "user" else "**Jimmy**"
        lines.append(f"{role}:")
        lines.append("")
        lines.append(msg.get("content") or "")
        lines.append("")
    return "\n".join(lines)


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
    st.session_state.source_kind = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "_search_triggered" not in st.session_state:
    st.session_state._search_triggered = False
if "_last_stream_metadata" not in st.session_state:
    st.session_state._last_stream_metadata = None


# ── URL-driven session loading ────────────────────────────────────────────

_url_report_id = st.query_params.get("report")
if _url_report_id and st.session_state.profile_result is None:
    with st.spinner("Loading session..."):
        _report = load_report(_url_report_id)

    if _report is not None:
        st.session_state.profile_result = {
            "report_id": _report["id"],
            "dataset_stats": (_report.get("profile_data") or {}).get("overview", {}),
            "columns": (_report.get("profile_data") or {}).get("columns", []),
            "overview": _report.get("ai_summary"),
            "source": None,
        }
        st.session_state.report_id = _report["id"]
        st.session_state.messages = _report.get("chat_messages") or []
        st.session_state.source_label = _report.get("source_title") or "Session"
        st.session_state.source_kind = (
            "government" if _report.get("catalog_dataset_id") else "upload"
        )
        st.rerun()
    else:
        st.warning("Could not load the shared session — it may have been deleted.")


def _queue_search():
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
    st.session_state._last_stream_metadata = None
    try:
        st.query_params.clear()
    except Exception:
        pass


def _stream_answer(question: str, report_id: str):
    """
    Generator consumed by st.write_stream. Yields text chunks. Stores the
    final metadata event (done or error) on session state for _do_ask to
    pick up after the generator is exhausted.
    """
    st.session_state._last_stream_metadata = None
    try:
        response = requests.post(
            f"{API_BASE}/reports/{report_id}/ask/stream",
            json={"question": question},
            stream=True,
            timeout=180,
        )
    except requests.exceptions.RequestException as e:
        yield f"⚠️ Could not reach the API: {e}"
        return

    if response.status_code != 200:
        text_preview = (response.text or "")[:200]
        yield f"⚠️ API error {response.status_code}: {text_preview}"
        return

    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        event_type = event.get("type")
        if event_type == "token":
            yield event.get("content", "")
        elif event_type == "done":
            st.session_state._last_stream_metadata = event
        elif event_type == "error":
            st.session_state._last_stream_metadata = event
            yield f"\n\n⚠️ {event.get('message', 'Unknown error')}"


def _do_ask(question: str):
    """
    Streams the answer inline, then stores the turn on session state. The
    caller must follow with st.rerun() so the message list replays from
    session_state and the streamed content is rendered from there — this
    avoids double-rendering the same message.
    """
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        full_text = st.write_stream(_stream_answer(question, st.session_state.report_id))

    metadata = st.session_state.get("_last_stream_metadata") or {}
    if metadata.get("type") == "done":
        # Prefer write_stream's return; fall back to the done event's
        # content field if write_stream returned something falsy.
        full_text = full_text or metadata.get("content", "")
        tool_calls = metadata.get("tool_calls_log", []) or []
        error_type = None
    elif metadata.get("type") == "error":
        tool_calls = []
        error_type = metadata.get("error_type")
    else:
        tool_calls = []
        error_type = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text or "",
        "tool_calls": tool_calls,
        "error_type": error_type,
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
                                    # Sync the URL bar so a plain F5 (without
                                    # the user having to copy the share link)
                                    # reloads the same session.
                                    try:
                                        st.query_params["report"] = analysis["report_id"]
                                    except Exception:
                                        pass
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
                    try:
                        st.query_params["report"] = result["report_id"]
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.error("Profiling failed. Check your API logs.")

st.divider()


# ── Step 2: detail view ────────────────────────────────────────────────────

if st.session_state.profile_result:
    result = st.session_state.profile_result

    share_col, back_col, _spacer = st.columns([3, 1, 4])
    with share_col:
        st.code(
            f"{APP_URL_BASE}?report={st.session_state.report_id}",
            language=None,
        )
    with back_col:
        if st.button("← Back to search", use_container_width=True):
            _reset_all()
            st.rerun()

    title = st.session_state.source_label or "Dataset"
    st.title(title)
    if st.session_state.source_kind == "government":
        st.caption("Official government dataset")
    elif st.session_state.source_kind == "upload":
        st.caption("Uploaded CSV file")

    # ── Overview ────────────────────────────────────────────────────────
    st.markdown("### Overview")
    overview_raw = result.get("overview")
    overview_content, overview_error = extract_ai_content(overview_raw)

    if overview_error:
        render_failed_ai("Overview", overview_error)
    elif isinstance(overview_content, dict):
        st.markdown(overview_content.get("overview", ""))

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
                    _do_ask(q)
                    st.rerun()

    # ── Chat ────────────────────────────────────────────────────────────
    st.markdown("### Chat")

    if st.session_state.messages:
        reset_col, download_col, _ = st.columns([2, 2, 4])
        with reset_col:
            if st.button("🗑️ Reset conversation"):
                try:
                    requests.post(
                        f"{API_BASE}/reports/{st.session_state.report_id}/reset",
                        timeout=30,
                    )
                except requests.exceptions.RequestException:
                    pass
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.rerun()
        with download_col:
            md = _build_markdown_export(
                st.session_state.profile_result,
                st.session_state.messages,
                st.session_state.source_label,
            )
            st.download_button(
                "📥 Download as Markdown",
                data=md,
                file_name=f"jimmycore-session-{st.session_state.report_id[:8]}.md",
                mime="text/markdown",
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content") or "")

            if msg["role"] == "assistant":
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

                _render_tool_calls_button(msg.get("tool_calls", []) or [])

    if prompt := st.chat_input("Ask a question about this dataset..."):
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