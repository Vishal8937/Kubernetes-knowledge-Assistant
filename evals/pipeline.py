"""
Phase 1 — Live Pipeline.

Calls the running FastAPI /query endpoint for each golden sample.

Captures:
    - actual_response
    - actual_contexts
    - actual_tools_called

IMPORTANT:
    golden_dataset.json is treated as immutable ground truth.

Live results are persisted separately to:
    eval_results.json
"""

import copy
import json
import os
import time

import logfire
import requests


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000/query"

DELAY_BETWEEN_CALLS = 10
REQUEST_TIMEOUT = 120

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "eval_results.json",
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from the /query response to a tool name.

    Planner sets:
        'Intent: Technical' + 'Search Term: ...'
            -> retrieve_documents

        'Intent: Conversational/Memory'
            -> direct_answer

    main.py sets:
        'Intent: Guardrails Fired'
            -> guardrails
    """

    if not thought_process:
        return "unknown"

    joined = " ".join(
        str(item) for item in thought_process
    ).lower()

    if "guardrails fired" in joined:
        return "guardrails"

    if (
        "intent: technical" in joined
        or "search term:" in joined
        or "context retrieved" in joined
    ):
        return "retrieve_documents"

    if (
        "conversational" in joined
        or "memory" in joined
    ):
        return "direct_answer"

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Live Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    golden_dataset: dict,
    progress_callback=None,
) -> dict:
    """
    Enriches a deep copy of the golden dataset with live API results.

    The original golden dataset is NEVER modified.

    Each RAG sample gets:

        actual_response
        actual_contexts
        actual_tools_called

    Returns the enriched dataset.
    """

    # Never mutate the original golden dataset.
    dataset = copy.deepcopy(golden_dataset)

    samples = dataset.get("rag_samples", [])
    n = len(samples)

    if n == 0:
        raise ValueError(
            "No rag_samples found in golden dataset."
        )

    with logfire.span(
        "🚀 Eval Phase 1 — Live Pipeline",
        total_samples=n,
    ):

        for i, sample in enumerate(samples):

            question = sample.get("question", "")

            # ─────────────────────────────────────────────────────────────
            # Progress callback: starting request
            # ─────────────────────────────────────────────────────────────

            if progress_callback:
                progress_callback(
                    i,
                    n,
                    question,
                    "calling",
                )

            # ─────────────────────────────────────────────────────────────
            # Call FastAPI
            # ─────────────────────────────────────────────────────────────

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):

                try:

                    resp = requests.post(
                        API_URL,
                        json={
                            "q": question,
                            "thread_id": f"eval_run_{i}",
                        },
                        timeout=REQUEST_TIMEOUT,
                    )

                    resp.raise_for_status()

                    data = resp.json()

                    # ─────────────────────────────────────────────────────
                    # Extract response
                    # ─────────────────────────────────────────────────────

                    raw_answer = data.get("answer") or ""

                    thought_process = (
                        data.get("thought_process") or []
                    )

                    sources = data.get("sources") or []

                    # ─────────────────────────────────────────────────────
                    # IMPORTANT:
                    #
                    # Store the FULL answer.
                    #
                    # Do NOT truncate here.
                    # RAGAS decides later how much data to send to the judge.
                    # ─────────────────────────────────────────────────────

                    sample["actual_response"] = raw_answer

                    # ─────────────────────────────────────────────────────
                    # Store ALL retrieved contexts.
                    #
                    # Do NOT truncate here either.
                    # metrics.py handles token reduction when calling RAGAS.
                    # ─────────────────────────────────────────────────────

                    sample["actual_contexts"] = sources

                    # ─────────────────────────────────────────────────────
                    # Detect tool
                    # ─────────────────────────────────────────────────────

                    detected_tool = detect_tool(
                        thought_process
                    )

                    sample["actual_tools_called"] = [
                        detected_tool
                    ]

                    # ─────────────────────────────────────────────────────
                    # Logging
                    # ─────────────────────────────────────────────────────

                    logfire.info(
                        "✅ Response captured",
                        sample_id=sample.get("id"),
                        tool=detected_tool,
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                # ─────────────────────────────────────────────────────────
                # Connection error
                # ─────────────────────────────────────────────────────────

                except requests.exceptions.ConnectionError:

                    logfire.error(
                        "❌ Cannot reach FastAPI — "
                        "is the app running on :8000?"
                    )

                    sample["actual_response"] = ""
                    sample["actual_contexts"] = []
                    sample["actual_tools_called"] = [
                        "unknown"
                    ]

                # ─────────────────────────────────────────────────────────
                # HTTP errors / JSON errors / other errors
                # ─────────────────────────────────────────────────────────

                except Exception as e:

                    logfire.error(
                        f"❌ Query failed: {e}"
                    )

                    sample["actual_response"] = ""
                    sample["actual_contexts"] = []
                    sample["actual_tools_called"] = [
                        "unknown"
                    ]

            # ─────────────────────────────────────────────────────────────
            # Progress callback: finished request
            # ─────────────────────────────────────────────────────────────

            if progress_callback:
                progress_callback(
                    i,
                    n,
                    question,
                    "done",
                    sample.get(
                        "actual_response",
                        "",
                    ),
                )

            # ─────────────────────────────────────────────────────────────
            # Rate-limit protection
            # ─────────────────────────────────────────────────────────────

            if i < n - 1:
                time.sleep(
                    DELAY_BETWEEN_CALLS
                )

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Save Results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    dataset: dict,
    path: str = RESULTS_PATH,
) -> None:
    """
    Save live evaluation results.

    IMPORTANT:
        This writes to eval_results.json,
        NOT golden_dataset.json.
    """

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dataset,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Load Existing Results
# ─────────────────────────────────────────────────────────────────────────────

def load_results(
    path: str = RESULTS_PATH,
) -> dict:
    """
    Load previously collected live evaluation results.

    This allows RAGAS evaluation to run without
    calling the FastAPI application again.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Evaluation results not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Load Golden Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_golden_dataset() -> dict:
    """
    Load the immutable ground-truth dataset.

    golden_dataset.json should contain:
        - questions
        - references
        - relevant contexts
        - expected tools
        - guardrail cases

    It should NOT contain live model results.
    """

    golden_path = os.path.join(
        os.path.dirname(__file__),
        "golden_dataset.json",
    )

    if not os.path.exists(golden_path):
        raise FileNotFoundError(
            f"Golden dataset not found: {golden_path}"
        )

    with open(
        golden_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)