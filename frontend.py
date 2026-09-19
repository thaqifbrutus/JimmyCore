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
st.caption("Search official government datasets, or upload your own CSV — AI-summarized either way")
st.divider()


# ── API helpers — upload flow (existing) ────────────────────────────────────

def upload_file(file):
    response = requests.post(
        f"{API_BASE}/upload",
        files={"file": (file.name, file.getvalue(), "text/csv")}
    )
    return response.json() if response.status_code == 200 else None


def trigger_profile(dataset_id):
    response = requests.post(f"{API_BASE}/reports/datasets/{dataset_id}/profile")
    return response.json() if response.status_code == 200 else None


# ── API helpers — government catalog search flow (new) ─────────────────────

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


def get_technical_context(report_id):
    response = requests.post(f"{API_BASE}/reports/{report_id}/technical-context")
    return response.json() if response.status_code == 200 else None


def ask_question(report_id, question, history):
    response = requests.post(
        f"{API_BASE}/reports/{report_id}/ask",
        json={"question": question, "conversation_history": history}
    )
    return response.json() if response.status_code == 200 else None


# ── Result dict helpers ────────────────────────────────────────────────────────
# All three AI functions now return {"status": "ok"|"failed", "reason": ...,
# "content": ...} instead of a raw string. These helpers extract content
# safely and surface failures consistently across the UI.

def extract_ai_content(result_dict, field_name="content"):
    """
    Pulls .content out of a structured AI result dict.
    Returns (content, error_message) — one is always None.
    """
    if result_dict is None:
        return None, "No response received from the API."
    if isinstance(result_dict, str):
        # Old-shape response from an endpoint not yet updated — pass through.
        return result_dict, None
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
        f"Reason: {reason}\n\n"
        f"Try clicking Regenerate, or upload a different file.",
        icon=None
    )


# ── Badge / status helpers ─────────────────────────────────────────────────────

def render_severity_badge(severity):
    colours = {
        "critical": "🔴",
        "warning":  "🟡",
        "info":     "🔵"
    }
    return colours.get(severity, "⚪")


def render_overall_status(status):
    mapping = {
        "good":            ("✅", "Good", "success"),
        "good_with_notes": ("✅", "Good with notes", "success"),
        "needs_attention": ("⚠️", "Needs attention", "warning"),
        "critical":        ("🔴", "Critical issues found", "error")
    }
    return mapping.get(status, ("⚪", status, "info"))


# ── Technical brief renderer ───────────────────────────────────────────────────
# technical_brief["content"] is a structured dict (see app/schemas.py),
# not a markdown string — render each section explicitly.

def render_technical_brief(content: dict):
    """
    Renders the structured TechnicalContext dict as readable Streamlit UI.
    Each section maps to its own visual treatment.
    """

    effort = content.get("estimated_effort", {})
    level = effort.get("level", "—")
    justification = effort.get("justification", "—")

    level_colours = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    badge = level_colours.get(level, "⚪")
    st.markdown(f"**Estimated effort:** {badge} {level}")
    st.caption(justification)

    st.markdown("---")

    schema = content.get("suggested_schema", [])
    if schema:
        st.markdown("##### 🗂️ Suggested database schema")
        schema_rows = [
            {
                "Column": row.get("column_name", ""),
                "Detected type": row.get("detected_type", ""),
                "Suggested SQL type": row.get("suggested_sql_type", ""),
                "Notes": row.get("notes", ""),
            }
            for row in schema
        ]
        st.dataframe(schema_rows, use_container_width=True, hide_index=True)

    rules = content.get("validation_rules", [])
    if rules:
        st.markdown("##### ✅ Validation rules")
        rules_rows = [
            {
                "Column": row.get("column_name", ""),
                "Rules": row.get("rules", ""),
            }
            for row in rules
        ]
        st.dataframe(rules_rows, use_container_width=True, hide_index=True)

    steps = content.get("transformation_steps", [])
    if steps:
        st.markdown("##### 🔄 Transformation steps")
        steps_rows = [
            {
                "Column": row.get("column_name", ""),
                "Transformation needed": row.get("transformation", ""),
            }
            for row in steps
        ]
        st.dataframe(steps_rows, use_container_width=True, hide_index=True)

    risks = content.get("risks_and_warnings", [])
    if risks:
        st.markdown("##### ⚠️ Risks and warnings")
        for risk in risks:
            st.markdown(f"- {risk}")


# ── Session state initialisation ───────────────────────────────────────────────

if "input_mode" not in st.session_state:
    st.session_state.input_mode = "Search government data"
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "report_id" not in st.session_state:
    st.session_state.report_id = None
if "profile_result" not in st.session_state:
    st.session_state.profile_result = None
if "tech_brief" not in st.session_state:
    st.session_state.tech_brief = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_error" not in st.session_state:
    st.session_state.search_error = None
if "source_label" not in st.session_state:
    st.session_state.source_label = None  # what Step 2's header calls the source


def _reset_all():
    for key in ["dataset_id", "report_id", "profile_result", "tech_brief",
                "search_results", "search_error", "source_label"]:
        st.session_state[key] = None
    st.session_state.chat_history = []
    st.session_state.messages = []


# ── Step 1: choose input method ─────────────────────────────────────────────

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
        )
    with button_col:
        run_search = st.button("🔍 Search", type="primary", use_container_width=True)

    if run_search and query.strip():
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
                st.success(f"✅ Uploaded: **{result['original_name']}**")
            else:
                st.error("Upload failed. Check your API is running.")

    if st.session_state.dataset_id and not st.session_state.profile_result:
        if st.button("🔍 Run AI Analysis", type="primary"):
            with st.spinner("Profiling dataset and generating AI summary — this may take a moment"):
                result = trigger_profile(st.session_state.dataset_id)
                if result:
                    st.session_state.profile_result = result
                    st.session_state.report_id = result["report_id"]
                    st.session_state.source_label = None  # upload flow shows the filename separately
                    st.rerun()
                else:
                    st.error("Profiling failed. Check your API logs.")

st.divider()


# ── Step 2: Results ────────────────────────────────────────────────────────────
# Shared by both flows — /reports/datasets/{id}/profile (upload) and
# /catalog/{id}/analyze (search) return the same shape, so everything below
# renders identically regardless of where the data came from.

if st.session_state.profile_result:
    result = st.session_state.profile_result
    overview = result["overview"]
    issues = result["issues"]
    status = result["overall_status"]
    emoji, label, alert_type = render_overall_status(status)

    st.subheader("Step 2 — Quality report")
    if st.session_state.source_label:
        st.caption(f"Source: {st.session_state.source_label} (official government dataset)")

    if alert_type == "success":
        st.success(f"{emoji} Overall status: **{label}**")
    elif alert_type == "warning":
        st.warning(f"{emoji} Overall status: **{label}**")
    else:
        st.error(f"{emoji} Overall status: **{label}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{overview['row_count']:,}")
    col2.metric("Columns", overview["column_count"])
    col3.metric("Nulls", f"{overview['null_percentage']}%")
    col4.metric("Duplicates", overview["duplicate_row_count"])

    st.markdown("#### 🤖 AI Summary")
    with st.container(border=True):
        ai_summary_raw = result.get("ai_summary")
        summary_content, summary_error = extract_ai_content(ai_summary_raw)

        if summary_error:
            render_failed_ai("AI Summary", summary_error)
        elif summary_content:
            st.markdown(summary_content)
        else:
            st.info("No AI summary available for this report.")

    if issues:
        st.markdown("#### ⚠️ Issues detected")
        for issue in issues:
            badge = render_severity_badge(issue["severity"])
            with st.expander(f"{badge} {issue['message']}"):
                st.write(f"**Type:** `{issue['type']}`")
                st.write(f"**Affected:** `{issue['affected']}`")
                st.write(f"**Severity:** {issue['severity'].upper()}")
    else:
        st.success("No issues detected. This dataset looks clean!")

    if "profile_data" in result:
        st.markdown("#### 📊 Column breakdown")
        columns = result["profile_data"].get("columns", [])
        for col in columns:
            with st.expander(f"📋 `{col['name']}` — {col['dtype']}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Nulls", f"{col['null_percentage']}%")
                c2.metric("Unique values", col["unique_count"])
                c3.metric("Null count", col["null_count"])

                if col.get("stats"):
                    st.markdown("**Statistics**")
                    stats = col["stats"]
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Min", stats.get("min", "—"))
                    s2.metric("Max", stats.get("max", "—"))
                    s3.metric("Mean", stats.get("mean", "—"))
                    s4.metric("Std dev", stats.get("std_dev", "—"))

                if col.get("top_values"):
                    st.markdown("**Top values**")
                    for tv in col["top_values"]:
                        st.write(f"- `{tv['value']}` — {tv['count']} occurrences")

                if col.get("sampling_note"):
                    st.caption(f"ℹ️ {col['sampling_note']}")

    st.markdown("#### 🛠️ Technical brief for developers")

    if st.session_state.tech_brief is None:
        if st.button("Generate technical brief"):
            with st.spinner("Generating technical brief..."):
                tech_response = get_technical_context(st.session_state.report_id)
                if tech_response:
                    st.session_state.tech_brief = tech_response.get("technical_brief")
                    st.rerun()
                else:
                    st.error("Could not reach the API. Check your API logs.")
    else:
        tech_content, tech_error = extract_ai_content(st.session_state.tech_brief)

        if tech_error:
            render_failed_ai("Technical brief", tech_error)
            if st.button("🔄 Retry technical brief"):
                st.session_state.tech_brief = None
                st.rerun()
        elif tech_content:
            with st.container(border=True):
                render_technical_brief(tech_content)
            if st.button("🔄 Regenerate technical brief"):
                st.session_state.tech_brief = None
                st.rerun()
        else:
            st.info("Technical brief content was empty.")

    st.divider()

    st.subheader("Step 3 — Ask JimmyCore AI")
    st.caption("Ask any question about your dataset in plain English")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your dataset..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(
                    st.session_state.report_id,
                    prompt,
                    st.session_state.chat_history
                )

                answer_raw = response.get("answer") if response else None
                answer_content, answer_error = extract_ai_content(answer_raw)

                if answer_error:
                    render_failed_ai("Answer", answer_error)
                elif answer_content:
                    st.markdown(answer_content)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_content
                    })
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": prompt
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_content
                    })
                else:
                    st.error("Could not get a response. Check your API.")

else:
    if st.session_state.input_mode == "Search government data":
        st.info("Search for a topic above to find official government datasets.")
    else:
        st.info("Upload a CSV file above to get started.")


# ── Reset button ───────────────────────────────────────────────────────────────

if st.session_state.dataset_id or st.session_state.profile_result:
    st.divider()
    if st.button("🔄 Start over"):
        _reset_all()
        st.rerun()
