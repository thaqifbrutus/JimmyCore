import streamlit as st
import requests
import json

# API_BASE = "http://localhost:8000/api"
API_BASE = "jimmycore-production.up.railway.app"

st.set_page_config(
    page_title="JimmyCore",
    page_icon="To_Be_Inserted",
    layout="wide"
)

st.title("JimmyCore")
st.caption("AI-powered data quality platform")
st.divider()

def upload_file(file):
    response = requests.post(
        f"{API_BASE}/upload",
        files={"file": (file.name, file.getvalue(), "text/csv")}
    )
    return response.json() if response.status_code == 200 else None


def trigger_profile(dataset_id):
    response = requests.post(f"{API_BASE}/datasets/{dataset_id}/profile")
    return response.json() if response.status_code == 200 else None


def get_technical_context(report_id):
    response = requests.post(f"{API_BASE}/reports/{report_id}/technical-context")
    return response.json() if response.status_code == 200 else None


def ask_question(report_id, question, history):
    response = requests.post(
        f"{API_BASE}/reports/{report_id}/ask",
        json={"question": question, "conversation_history": history}
    )
    return response.json() if response.status_code == 200 else None


def render_severity_badge(severity):
    colours = {
        "critical": "placeholder",
        "warning": "placeholder",
        "info": "placeholder"
    }
    return mapping.get(status, ("placeholder_white", status, "info"))

# ── Session state initialisation ──────────────────────────────────────────────
# Streamlit re-runs the whole script on every interaction.
# st.session_state persists values across those reruns.
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

if "report_id" not in st.session_state:
    st.session_state.report_id = None

if "profile_result" not in st.session_state:
    st.session_state.profile_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.subheader("Step 1: Upload your dataset")

upload_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload any CSV file up to 10MB"
)

if upload_file and not st.session_state.dataset_id:
    with st.spinner("Uploading..."):
        result = upload_file(upload_file)
        if result:
            st.session_state.dataset_id = result["dataset_id"]
            st.success(f"Uploaded: **{result['original_name']}**")
        else:
            st.error("Upload failed. Check your API is running.")

if st.session_state.dataset_id and not st.session_state.profile_result:
    if st.button("Run AI Analysis", type="primary"):
        with st.spinner("Profiling dataset and generating AI summary...this may take a moment"):
            result = trigger_profile(st.session_state.dataset_id)
            if result:
                st.session_state.profile_result = result
                st.session_state.report_id = result["report_id"]
                st.rerun()
            else:
                st.error("Profiling failed. Check your API logs.")

st.divider()

# ── Step 2: Results ────────────────────────────────────────────────────────────
if st.session_state.profile_result:
    result = st.session_state.profile_result
    overview = result["overview"]
    issues = result["issues"]
    status = result["overall_status"]
    emoji, label, alert_type = render_overall_status(status)

    st.subheader("Step 2 - Quality report")

    # Overall status banner

    if alert_type == "success":
        st.success(f"{emoji} Overall status: **{label}**")
    elif alert_type == "warning":
        st.warning(f"{emoji} Overall status: **{label}**")
    else:
        st.error(f"{emoji} Overall status: **{label}**")

    # Overview metrics
    col1, col2, col3, col4, = st.columns(4)
    col1.metric("Rows", f"{overview['row_count']:,}")
    col2.metric("Columns", overview["column_count"])
    col3.metric("Nulls", f"{overview['null_percentage']}%")
    col4.metric("Duplicates", overview["duplicate_row_count"])

    # AI Summary
    st.markdown("##### AI Summary")
    with st.container(border=True):
        st.markdown(result.get("ai_summary", "No summary available."))

    # Issues breakdown
    if issues:
        st.markdown("#### Column Breakdown")
        columns = result["profile_data"].get("columns", [])
        for col in columns:
            with st.expander(f"'{col['name']}' - {col['dtype']}"):
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

        # Technical Context
        st.markdown("#### Technical brief for developers")
        if st.button("Generate technical context"):
            with st.spinner("Generating technical brief..."):
                tech = get_technical_context(st.session_state.report_id)
                if tech:
                    with st.container(border=True):
                        st.markdown(tech["technical_brief"])

st.divider()

# ── Step 3: Chat ───────────────────────────────────────────────────────────
st.subheader("Step 3 - Ask JimmyCore")
st.caption("Ask any question about your dataset")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your dataset..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_question(
                st.session_state.report_id,
                prompt,
                st.session_state.chat_history
            )
            if response:
                answer = response["answer"]
                st.markdown(answer)

                # Update both display message and API history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": prompt
                })
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                st.error("Could not get a response. Check your API.")
else:
    st.info("Upload a CSV file above to get started.")

# ── Reset button ───────────────────────────────────────────────────────────────
if st.session_state.dataset_id:
    st.divider()
    if st.button("Analyse a new dataset"):
        for key in ["dataset_id", "report_id", "profile_result",
                    "chat_history", "messages"]:
            st.session_state[key] = None if key != "chat_history" \
                and key != "messages" else []
        st.rerun()
             
          


