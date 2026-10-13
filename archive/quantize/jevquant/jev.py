"""TypeSafe Jev client: POST /v1/systemone with a disk cache and usage tracking.

Auth: if ``TYPESAFE_API_KEY`` is set it is sent as a Bearer token; otherwise the
request goes out unauthenticated, which works where a proxy injects credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = os.environ.get("TYPESAFE_URL", "https://api.typesafe.ai/v1/systemone")
CACHE_DIR = os.environ.get("JEV_CACHE", ".jev_cache")


def ask(state, questions: dict, model: str = "jev-latest", cache: bool = True, retries: int = 4) -> dict:
    """One Jev request.  Returns the API response plus ``latency_s`` and ``cached``."""
    body = {"model": model, "state": state, "questions": questions}
    blob = json.dumps(body, sort_keys=True)
    key = hashlib.sha256(blob.encode()).hexdigest()[:32]
    path = os.path.join(CACHE_DIR, key + ".json")
    if cache and os.path.exists(path):
        out = json.load(open(path))
        out["cached"] = True
        return out
    headers = {"Content-Type": "application/json"}
    if os.environ.get("TYPESAFE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TYPESAFE_API_KEY"]
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = requests.post(URL, data=blob, headers=headers, timeout=60)
            latency = time.perf_counter() - t0
            if r.status_code == 200:
                out = r.json()
                out["latency_s"] = round(latency, 4)
                if cache:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    json.dump(out, open(path, "w"))
                out["cached"] = False
                return out
            if r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"Jev {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("Jev request failed after retries")


def ask_many(jobs: list[tuple], workers: int = 8, **kw) -> list[dict]:
    """Run ``ask(state, questions, model)`` for many jobs in parallel, keeping order."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: ask(*j, **kw), jobs))
