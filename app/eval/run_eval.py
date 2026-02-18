from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from app.api.routes_chat import get_chat_service
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.eval.metrics import abstain_rate, average_latency_ms, hit_at_k, keyword_hit
from app.rag.chunking import build_chunks, load_chunks_jsonl, write_chunks_jsonl
from app.rag.ingest.export_reader import ingest_export, load_messages_jsonl


def _prepare_index() -> None:
    settings = get_settings()
    service = get_chat_service()

    chunks = load_chunks_jsonl(settings.chunks_jsonl_path)
    if chunks:
        return

    sample_path = settings.raw_data_dir / "sample_export_stub.json"
    if not sample_path.exists():
        return

    ingest_export(
        input_path=sample_path,
        output_messages_path=settings.messages_jsonl_path,
        allowlist_it_only=False,
        exclude_title_keywords=[],
    )
    messages = load_messages_jsonl(settings.messages_jsonl_path)
    chunks = build_chunks(messages, max_tokens=settings.max_chunk_tokens, overlap_messages=settings.overlap_messages)
    write_chunks_jsonl(settings.chunks_jsonl_path, chunks)

    service.store.create_collection(reset=True)
    if chunks:
        vectors = service.embedder.embed_texts([c.text for c in chunks])
        service.store.upsert_chunks(chunks, vectors)


def run_eval(dataset_path: Path, top_k: int) -> None:
    settings = get_settings()
    service = get_chat_service()

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    latencies: list[float] = []
    abstains: list[bool] = []
    hit_scores: list[float] = []

    print("| Question | Hit@5 | Keyword Hit | Latency (ms) | Abstained |")
    print("|---|---:|---:|---:|---:|")

    for row in dataset:
        question = row["question"]
        expected_chat = row.get("expected_chat_id")
        expected_keywords = row.get("expected_keywords", [])

        start = perf_counter()
        contexts = service.retriever.retrieve(question=question, top_k=top_k)
        answer, _ = service.answerer.generate(question, contexts, mode=settings.mode)
        latency_ms = (perf_counter() - start) * 1000

        chat_hit = hit_at_k([ctx.chat_id for ctx in contexts[:top_k]], expected_chat)
        kw_hit = keyword_hit([ctx.text for ctx in contexts[:top_k]], expected_keywords)
        is_abstain = answer.lower().startswith("insufficient context")

        latencies.append(latency_ms)
        abstains.append(is_abstain)
        hit_scores.append(max(chat_hit, kw_hit))

        print(
            f"| {question} | {chat_hit:.0f} | {kw_hit:.0f} | {latency_ms:.1f} | {str(is_abstain)} |"
        )

    hit_at_5 = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0
    avg_latency = average_latency_ms(latencies)
    abstain = abstain_rate(abstains)

    print("\n## Summary")
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| Hit@5 | {hit_at_5:.2f} |")
    print(f"| Avg latency (ms) | {avg_latency:.1f} |")
    print(f"| Abstain rate | {abstain:.2f} |")
    print("| Embedding cost | 0 (local model) |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for the RAG Chat Assistant")
    parser.add_argument("--dataset", default="app/eval/dataset.example.json", help="Path to eval dataset JSON")
    parser.add_argument("--top-k", default=5, type=int, help="Top-k retrieval cutoff")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    _prepare_index()
    run_eval(Path(args.dataset), top_k=args.top_k)


if __name__ == "__main__":
    main()
