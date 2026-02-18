from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATA_RAW_DIR = Path("data/raw")

TOPICS = [
    "any",
    "python",
    "backend",
    "fastapi",
    "django",
    "devops",
    "linux",
    "databases",
    "algorithms",
    "networking",
    "other",
]


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="RAG Chat Assistant", layout="wide")
st.title("RAG Chat Assistant")
st.caption("Local-first retrieval over your ChatGPT exports with citation-grounded answers")

col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("Ingestion")
    uploaded = st.file_uploader("Upload ChatGPT export", type=["zip", "json", "html", "htm"])

    if uploaded is not None and st.button("Ingest Uploaded Export"):
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = DATA_RAW_DIR / uploaded.name
        file_path.write_bytes(uploaded.getvalue())
        try:
            result = api_post("/admin/ingest", {"input_path": str(file_path)})
            st.success(f"Ingested {result.get('processed_message_count', 0)} messages")
            st.json(result)
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")

    path_hint = st.text_input("Or ingest file already on disk", value="data/raw/sample_export_stub.json")
    if st.button("Ingest From Path"):
        try:
            result = api_post("/admin/ingest", {"input_path": path_hint})
            st.success(f"Ingested {result.get('processed_message_count', 0)} messages")
            st.json(result)
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")

    if st.button("Show Index Stats"):
        try:
            stats = api_get("/admin/stats")
            st.json(stats)
        except Exception as exc:
            st.error(f"Stats failed: {exc}")

with col_b:
    st.subheader("Ask")
    question = st.text_area("Question", height=120, placeholder="Ask about your prior conversations...")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        topic = st.selectbox("Topic", TOPICS, index=0)
    with fcol2:
        top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
    with fcol3:
        mode = st.selectbox("Mode", ["extractive", "llm"], index=0)
    with fcol4:
        chat_ids_raw = st.text_input("Chat IDs (csv)", value="")

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        date_from = st.text_input("Date From (ISO)", value="")
    with dcol2:
        date_to = st.text_input("Date To (ISO)", value="")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            payload = {
                "question": question,
                "top_k": top_k,
                "topic": None if topic == "any" else topic,
                "date_from": date_from or None,
                "date_to": date_to or None,
                "chat_ids": [x.strip() for x in chat_ids_raw.split(",") if x.strip()] or None,
                "mode": mode,
            }
            try:
                result = api_post("/ask", payload)
                st.markdown("### Answer")
                st.write(result["answer"])
                st.caption(
                    f"Confidence: {result['confidence']:.3f} | Latency: {result['latency_ms']:.1f} ms"
                )

                st.markdown("### Citations")
                for idx, citation in enumerate(result.get("citations", []), start=1):
                    with st.expander(f"C{idx} | chat={citation['chat_id']} | score={citation['score']:.3f}"):
                        st.write(citation["snippet"])
                        st.code(
                            f"message_ids={citation['message_ids']}\ncreated_at={citation.get('created_at')}",
                            language="text",
                        )
            except Exception as exc:
                st.error(f"Ask failed: {exc}")
