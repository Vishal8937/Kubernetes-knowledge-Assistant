"""
Phase 2 — RAGAS + Tool Correctness metrics.

Uses JUDGE_GROQ key so production GROQ_API_KEY is never exhausted
by evaluation runs.

All LLM-based metrics run one sample at a time with cooldowns
between samples/experiments to stay within Groq's 6,000 TPM
on-demand tier.

IMPORTANT:

    eval_results.json stores the FULL response and FULL retrieved
    contexts.

    Context truncation happens ONLY here, immediately before the
    data is sent to RAGAS.

Metrics:
    1. Faithfulness
    2. Answer Relevancy
    3. Context Precision
    4. Context Recall
    5. Answer Correctness
    6. Tool Correctness
"""

import asyncio
import os

import logfire
import pandas as pd
from openai import AsyncOpenAI

from app.config import settings

from ragas import SingleTurnSample
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory

from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────



JUDGE_MODEL = "gpt-4o-mini"

# Cooldown between experiments.
COOLDOWN_STANDARD = 62

# Cooldown between individual samples.
COOLDOWN_MINI = 40

# IMPORTANT:
# RAGAS abatch_score can generate concurrent LLM calls.
# Keep this at 1 for Groq's 6,000 TPM limit.
GENERAL_BATCH_SIZE = 1

# Token protection.
#
# These ONLY affect what is sent to RAGAS.
# The original eval_results.json keeps the complete contexts.
CONTEXT_TRUNCATE = 300
CONTEXT_LIMIT = 2


# ─────────────────────────────────────────────────────────────────────────────
# Build RAGAS Judge
# ─────────────────────────────────────────────────────────────────────────────

def _build_judge():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.openai.com/v1",
    )

    llm = llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=4096,
    )

    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )

    return llm, embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Cooldown
# ─────────────────────────────────────────────────────────────────────────────

async def _cooldown(
    seconds: int,
    label: str,
    status_cb=None,
):

    msg = (
        f"⏳ {seconds}s cooldown after "
        f"{label} (Groq TPM buffer)..."
    )

    if status_cb:
        status_cb(msg)

    # Sleep in 10-second chunks so Streamlit receives updates.
    remaining = seconds

    while remaining > 0:

        sleep_for = min(
            10,
            remaining,
        )

        await asyncio.sleep(
            sleep_for
        )

        remaining -= sleep_for

    if status_cb:
        status_cb(
            "✅ Ready — starting next experiment."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prepare Samples
# ─────────────────────────────────────────────────────────────────────────────

def _prep_samples(
    golden_dataset: dict,
) -> list:
    """
    Prepare live evaluation results for RAGAS.

    Only samples with an actual_response are evaluated.

    IMPORTANT:

        The full response/context data remains in eval_results.json.

        Contexts are truncated only in this function so the RAGAS
        judge request stays below Groq's TPM limit.
    """

    valid = []

    for s in golden_dataset.get(
        "rag_samples",
        [],
    ):

        response = (
            s.get(
                "actual_response",
                "",
            )
            or ""
        ).strip()

        # Skip failed API calls.
        if not response:
            continue

        raw_contexts = (
            s.get("actual_contexts")
            or s.get("relevant_contexts")
            or []
        )

        # Keep only the first CONTEXT_LIMIT contexts
        # and truncate each one for the RAGAS request.
        contexts = [
            str(context)[:CONTEXT_TRUNCATE]
            for context in raw_contexts[
                :CONTEXT_LIMIT
            ]
        ]

        valid.append(
            {
                **s,
                "actual_contexts": contexts,
            }
        )

    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Convert Scores to DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _score_df(
    metric_key: str,
    samples: list,
    scores,
) -> pd.DataFrame:

    rows = []

    for s, result in zip(
        samples,
        scores,
    ):

        rows.append(
            {
                "question": s["question"][:65],
                metric_key: round(
                    float(result.value),
                    3,
                ),
            }
        )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Batched Scoring
# ─────────────────────────────────────────────────────────────────────────────

async def _batched_score(
    metric,
    inputs: list,
    samples: list,
    status_cb=None,
    label: str = "",
) -> list:
    """
    Runs metric.abatch_score() in small batches.

    GENERAL_BATCH_SIZE is intentionally 1 because RAGAS can make
    multiple concurrent LLM calls inside abatch_score().
    """

    all_scores = []

    batches = [
        inputs[
            i : i + GENERAL_BATCH_SIZE
        ]
        for i in range(
            0,
            len(inputs),
            GENERAL_BATCH_SIZE,
        )
    ]

    for b_idx, batch in enumerate(
        batches
    ):

        if b_idx > 0:

            await _cooldown(
                COOLDOWN_MINI,
                f"{label} batch {b_idx}",
                status_cb,
            )

        if status_cb:
            status_cb(
                f"🧪 {label}: "
                f"sample {b_idx + 1}/{len(batches)}..."
            )

        scores = await metric.abatch_score(
            batch
        )

        all_scores.extend(
            scores
        )

    return all_scores


# ─────────────────────────────────────────────────────────────────────────────
# Run All Metrics
# ─────────────────────────────────────────────────────────────────────────────

async def run_all_metrics(
    golden_dataset: dict,
    status_cb=None,
) -> dict:
    """
    Runs all 6 evaluation experiments.

    Returns:

        {
            "faithfulness": DataFrame,
            "answer_relevancy": DataFrame,
            "context_precision": DataFrame,
            "context_recall": DataFrame,
            "answer_correctness": DataFrame,
            "tool_correctness": DataFrame,
        }
    """

    judge_llm, ragas_embeddings = (
        _build_judge()
    )

    samples = _prep_samples(
        golden_dataset
    )

    if not samples:
        raise ValueError(
            "No samples with actual_response found. "
            "Run Phase 1 first."
        )

    results = {}

    with logfire.span(
        "🧪 Eval Phase 2 — All Metrics",
        total_samples=len(samples),
    ):

        # ═════════════════════════════════════════════════════════════════
        # EXP 1 — Faithfulness
        # ═════════════════════════════════════════════════════════════════

        if status_cb:
            status_cb(
                f"🧪 Exp 1/6 — Faithfulness "
                f"({len(samples)} samples)..."
            )

        with logfire.span(
            "🧪 Exp 1 — Faithfulness"
        ):

            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s[
                        "actual_contexts"
                    ],
                }
                for s in samples
            ]

            scores = await _batched_score(
                Faithfulness(
                    llm=judge_llm
                ),
                inputs,
                samples,
                status_cb,
                "Faithfulness",
            )

            df = _score_df(
                "faithfulness",
                samples,
                scores,
            )

            results[
                "faithfulness"
            ] = df

            logfire.info(
                "🧪 Faithfulness done",
                avg=round(
                    df[
                        "faithfulness"
                    ].mean(),
                    3,
                ),
            )

        await _cooldown(
            COOLDOWN_STANDARD,
            "Faithfulness",
            status_cb,
        )

        # ═════════════════════════════════════════════════════════════════
        # EXP 2 — Answer Relevancy
        # ═════════════════════════════════════════════════════════════════

        if status_cb:
            status_cb(
                f"🧪 Exp 2/6 — Answer Relevancy "
                f"({len(samples)} samples)..."
            )

        with logfire.span(
            "🧪 Exp 2 — Answer Relevancy"
        ):

            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                }
                for s in samples
            ]

            scores = await _batched_score(
                AnswerRelevancy(
                    llm=judge_llm,
                    embeddings=ragas_embeddings,
                ),
                inputs,
                samples,
                status_cb,
                "Answer Relevancy",
            )

            df = _score_df(
                "answer_relevancy",
                samples,
                scores,
            )

            results[
                "answer_relevancy"
            ] = df

            logfire.info(
                "🧪 Answer Relevancy done",
                avg=round(
                    df[
                        "answer_relevancy"
                    ].mean(),
                    3,
                ),
            )

        await _cooldown(
            COOLDOWN_STANDARD,
            "Answer Relevancy",
            status_cb,
        )

        # ═════════════════════════════════════════════════════════════════
        # EXP 3 — Context Precision
        # ═════════════════════════════════════════════════════════════════

        if status_cb:
            status_cb(
                f"🧪 Exp 3/6 — Context Precision "
                f"({len(samples)} samples)..."
            )

        with logfire.span(
            "🧪 Exp 3 — Context Precision"
        ):

            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s[
                        "actual_contexts"
                    ],
                }
                for s in samples
            ]

            scores = await _batched_score(
                ContextPrecision(
                    llm=judge_llm
                ),
                inputs,
                samples,
                status_cb,
                "Context Precision",
            )

            df = _score_df(
                "context_precision",
                samples,
                scores,
            )

            results[
                "context_precision"
            ] = df

            logfire.info(
                "🧪 Context Precision done",
                avg=round(
                    df[
                        "context_precision"
                    ].mean(),
                    3,
                ),
            )

        await _cooldown(
            COOLDOWN_STANDARD,
            "Context Precision",
            status_cb,
        )

        # ═════════════════════════════════════════════════════════════════
        # EXP 4 — Context Recall
        # ═════════════════════════════════════════════════════════════════

        if status_cb:
            status_cb(
                f"🧪 Exp 4/6 — Context Recall "
                f"({len(samples)} samples)..."
            )

        with logfire.span(
            "🧪 Exp 4 — Context Recall"
        ):

            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s[
                        "actual_contexts"
                    ],
                }
                for s in samples
            ]

            scores = await _batched_score(
                ContextRecall(
                    llm=judge_llm
                ),
                inputs,
                samples,
                status_cb,
                "Context Recall",
            )

            df = _score_df(
                "context_recall",
                samples,
                scores,
            )

            results[
                "context_recall"
            ] = df

            logfire.info(
                "🧪 Context Recall done",
                avg=round(
                    df[
                        "context_recall"
                    ].mean(),
                    3,
                ),
            )

        await _cooldown(
            COOLDOWN_STANDARD,
            "Context Recall",
            status_cb,
        )

        # ═════════════════════════════════════════════════════════════════
        # EXP 5 — Answer Correctness
        # ═════════════════════════════════════════════════════════════════

        if status_cb:
            status_cb(
                f"🧪 Exp 5/6 — Answer Correctness "
                f"({len(samples)} samples)..."
            )

        with logfire.span(
            "🧪 Exp 5 — Answer Correctness"
        ):

            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]

            all_scores = await _batched_score(
                AnswerCorrectness(
                    llm=judge_llm,
                    embeddings=ragas_embeddings,
                ),
                inputs,
                samples,
                status_cb,
                "Answer Correctness",
            )

            df = _score_df(
                "answer_correctness",
                samples,
                all_scores,
            )

            results[
                "answer_correctness"
            ] = df

            logfire.info(
                "🧪 Answer Correctness done",
                avg=round(
                    df[
                        "answer_correctness"
                    ].mean(),
                    3,
                ),
            )

        await _cooldown(
            COOLDOWN_STANDARD,
            "Answer Correctness",
            status_cb,
        )

        # ═════════════════════════════════════════════════════════════════
        # EXP 6 — Tool Correctness
        #
        # No LLM calls.
        # Uses Jaccard similarity between expected and actual tools.
        # ═════════════════════════════════════════════════════════════════

        if status_cb:
            status_cb(
                "⚡ Exp 6/6 — Tool Correctness "
                "(zero LLM calls)..."
            )

        with logfire.span(
            "🧪 Exp 6 — Tool Correctness"
        ):

            tool_rows = []

            for s in samples:

                called = set(
                    s.get(
                        "actual_tools_called"
                    )
                    or []
                )

                expected = set(
                    s.get(
                        "expected_tools"
                    )
                    or []
                )

                union = len(
                    called | expected
                )

                if union > 0:
                    score = (
                        len(
                            called & expected
                        )
                        / union
                    )
                else:
                    score = 0.0

                tool_rows.append(
                    {
                        "question": s[
                            "question"
                        ][:65],
                        "tool_correctness": round(
                            score,
                            3,
                        ),
                    }
                )

            df = pd.DataFrame(
                tool_rows
            )

            results[
                "tool_correctness"
            ] = df

            logfire.info(
                "🧪 Tool Correctness done",
                avg=round(
                    df[
                        "tool_correctness"
                    ].mean(),
                    3,
                ),
            )

        if status_cb:
            status_cb(
                "✅ All 6 experiments complete!"
            )

    return results