import os
import re
import uuid

import requests
import streamlit as st


st.set_page_config(page_title="DocQA System", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def init_session_state():
    """Initialize all session state variables if they don't exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "confidence_scores" not in st.session_state:
        st.session_state.confidence_scores = []
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = False


def display_sidebar_metrics():
    """Show total queries and average confidence in the sidebar."""
    with st.sidebar:
        st.header("Analytics")
        st.metric("Total Queries", st.session_state.query_count)
        avg_conf = 0.0
        if st.session_state.confidence_scores:
            avg_conf = sum(st.session_state.confidence_scores) / len(
                st.session_state.confidence_scores
            )
        st.metric("Avg Confidence", f"{avg_conf:.2%}")

        st.header("Question History")
        if st.session_state.history:
            for idx, (question, answer, _, _) in enumerate(
                st.session_state.history[-5:]
            ):
                with st.expander(f"{idx + 1}. {question[:50]}"):
                    st.write(f"**Answer:** {answer[:100]}...")
        else:
            st.info("No questions asked yet.")


def upload_files(files):
    """Upload PDF files to backend and update session state."""
    if not files:
        return False

    url = f"{BACKEND_URL}/upload"
    data = {"session_id": st.session_state.session_id}
    files_data = [
        ("files", (file.name, file.getvalue(), "application/pdf")) for file in files
    ]

    try:
        with st.spinner("Uploading and processing documents..."):
            response = requests.post(url, data=data, files=files_data, timeout=120)
            response.raise_for_status()
            result = response.json()
            st.success(
                f"Upload successful! Processed {result.get('total_chunks', 0)} chunks."
            )
            st.session_state.uploaded = True
            return True
    except requests.exceptions.RequestException as exc:
        st.error(f"Upload failed: {_format_request_error(exc)}")
        return False


def ask_question(question):
    """Send question to backend and return answer data."""
    url = f"{BACKEND_URL}/ask"
    payload = {
        "session_id": st.session_state.session_id,
        "question": question,
    }
    try:
        with st.spinner("Searching documents..."):
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Error getting answer: {_format_request_error(exc)}")
        return None


def _format_request_error(exc):
    """Return a readable API error message when the backend sends JSON details."""
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    try:
        data = response.json()
    except ValueError:
        return str(exc)

    detail = data.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail

    return str(exc)


def update_analytics(score):
    """Update query count and confidence scores."""
    st.session_state.query_count += 1
    st.session_state.confidence_scores.append(score)


def display_answer(data, question):
    """Display the answer, simplified answer, score, and source chunk."""
    answer = data.get("answer", "")
    simplified = data.get("simplified_answer", "")
    score = data.get("score", 0.0)
    source_text = data.get("source_text", "")
    source_idx = data.get("source_chunk_index", -1)

    update_analytics(score)
    st.session_state.history.append((question, answer, simplified, score))

    st.subheader("Answer")
    if "upload documents first" in answer.lower() or "not found" in answer.lower():
        st.warning(answer)
    else:
        st.success(answer)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence Score", f"{score:.2%}")
    with col2:
        st.info(f"Source Chunk Index: {source_idx}")

    with st.expander("Simplified Answer"):
        st.write(simplified)

    if source_text:
        with st.expander("Source Text (with keyword highlighting)"):
            keywords = [word for word in question.split() if len(word) > 4]
            highlighted = source_text
            for keyword in keywords:
                pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
                highlighted = pattern.sub(
                    "<mark style='background-color: #FFFF00'>\\g<0></mark>",
                    highlighted,
                )
            st.markdown(highlighted, unsafe_allow_html=True)
    else:
        st.info("No source text available.")


def main():
    st.title("Document Question Answering System")
    st.markdown(
        "Upload PDF documents, then ask questions. The system will query the configured backend service."
    )

    init_session_state()
    display_sidebar_metrics()

    with st.container():
        st.header("1. Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if st.button("Upload and Process", type="primary"):
            if uploaded_files:
                success = upload_files(uploaded_files)
                if success:
                    st.rerun()
            else:
                st.warning("Please select at least one PDF file.")

    st.header("2. Ask a Question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic of the document?",
    )

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button(
            "Get Answer", type="primary", disabled=not st.session_state.uploaded
        )
    with col2:
        clear_button = st.button("Clear History")

    if clear_button:
        st.session_state.history = []
        st.session_state.query_count = 0
        st.session_state.confidence_scores = []
        st.rerun()

    if ask_button and question.strip():
        answer_data = ask_question(question)
        if answer_data:
            display_answer(answer_data, question)
    elif ask_button and not question.strip():
        st.warning("Please enter a question.")
    elif ask_button and not st.session_state.uploaded:
        st.warning("Please upload documents first.")

    if st.session_state.history:
        with st.expander("Recent Questions & Answers"):
            for idx, (question_text, answer, _, _) in enumerate(
                reversed(st.session_state.history[-5:])
            ):
                st.write(f"**Q{len(st.session_state.history) - idx}:** {question_text}")
                st.write(f"**A:** {answer[:200]}...")
                st.divider()


if __name__ == "__main__":
    main()
