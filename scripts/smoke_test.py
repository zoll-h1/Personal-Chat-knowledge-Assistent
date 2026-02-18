from __future__ import annotations

import os
import sys
from pprint import pprint

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def main() -> None:
    print(f"Using API: {API_BASE_URL}")

    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=15)
        health.raise_for_status()
        print("Health:", health.json())

        reset = _post("/admin/collection/reset", {})
        print("Collection reset:", reset)

        ingest = _post("/admin/ingest", {"input_path": "data/raw/sample_export_stub.json"})
        print("Ingest summary:")
        pprint(ingest)

        ask = _post(
            "/ask",
            {
                "question": "How do I run FastAPI with Uvicorn and where are vectors stored?",
                "top_k": 5,
                "mode": "extractive",
            },
        )
        print("\nAnswer:")
        print(ask["answer"])
        print("\nTop citation:")
        if ask.get("citations"):
            pprint(ask["citations"][0])
        else:
            print("No citations returned")

    except Exception as exc:
        print(f"Smoke test failed: {exc}")
        print("Make sure docker compose services are running (qdrant + api).")
        sys.exit(1)


if __name__ == "__main__":
    main()
