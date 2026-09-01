# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: logfire must be configured before all other imports
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)


from dotenv import load_dotenv

load_dotenv()


import logfire

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="evals",
)


# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import json

import nest_asyncio
import pandas as pd
import streamlit as st

nest_asyncio.apply()


from evals.pipeline import (
    run_pipeline,
    load_golden_dataset,
    save_results,
    load_results,
)

from evals.guardrails_eval import (
    run_guardrails_eval,
    compute_guardrails_metrics,
)

from evals.metrics import (
    run_all_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Enterprise RAG — Eval Suite",
    page_icon="🧪",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SCORE_COLORS = {
    "green": "#d4edda",
    "yellow": "#fff3cd",
    "red": "#f8d7da",
}


def _badge(score: float) -> str:

    if score >= 0.75:
        return "🟢"

    elif score >= 0.5:
        return "🟡"

    return "🔴"


def _grade(score: float) -> str:

    if score >= 0.75:
        return "✅ Good"

    elif score >= 0.5:
        return "⚠️ Fair"

    return "❌ Poor"


def _color_score(val):

    if not isinstance(
        val,
        (int, float),
    ):
        return ""

    if val >= 0.75:
        return (
            f"background-color: "
            f"{SCORE_COLORS['green']}"
        )

    elif val >= 0.5:
        return (
            f"background-color: "
            f"{SCORE_COLORS['yellow']}"
        )

    return (
        f"background-color: "
        f"{SCORE_COLORS['red']}"
    )


def _render_metric_table(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
):

    if df.empty:
        st.warning(
            f"No results available for {title}."
        )
        return

    avg = df[
        metric_col
    ].mean()

    st.markdown(
        f"**{title}** — "
        f"AVG: {_badge(avg)} "
        f"`{avg:.2f}` "
        f"{_grade(avg)}"
    )

    styled = (
        df.style
        .applymap(
            _color_score,
            subset=[metric_col],
        )
        .format(
            {
                metric_col: "{:.3f}"
            }
        )
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
    )


def _run_async(coro):

    loop = asyncio.get_event_loop()

    return loop.run_until_complete(
        coro
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────────────────────

# Golden dataset is always loaded from the immutable source.
if "golden" not in st.session_state:

    st.session_state.golden = (
        load_golden_dataset()
    )


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT:
#
# Try to restore a previously collected eval_results.json.
#
# This means restarting Streamlit does NOT force you to call FastAPI again.
# ─────────────────────────────────────────────────────────────────────────────

if "enriched_dataset" not in st.session_state:

    try:

        st.session_state.enriched_dataset = (
            load_results()
        )

        st.session_state.pipeline_done = True

    except FileNotFoundError:

        st.session_state.enriched_dataset = None
        st.session_state.pipeline_done = False


if "pipeline_done" not in st.session_state:

    st.session_state.pipeline_done = (
        st.session_state.enriched_dataset
        is not None
    )


if "guardrails_results" not in st.session_state:
    st.session_state.guardrails_results = None


if "metric_results" not in st.session_state:
    st.session_state.metric_results = None


if "pipeline_rows" not in st.session_state:
    st.session_state.pipeline_rows = []


golden = st.session_state.golden


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title(
    "🧪 Enterprise RAG — Evaluation Suite"
)

st.caption(
    "Step 1: Review ground truth → "
    "Step 2: Run live pipeline → "
    "Step 3: Score with RAGAS"
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    [
        "📋 Step 1 — Ground Truth",
        "🚀 Step 2 — Live Pipeline",
        "📊 Step 3 — Eval Metrics",
    ]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════

with tab1:

    st.subheader(
        "Ground Truth Dataset"
    )

    st.markdown(
        "These are the **golden Q&A pairs** built "
        "by parsing your real enterprise documents. "
        "Each entry has a question, a reference answer "
        "(ground truth), and the expected tool the RAG "
        "agent should call."
    )

    rag_rows = []

    for s in golden.get(
        "rag_samples",
        [],
    ):

        reference = s.get(
            "reference",
            "",
        )

        expected_tools = s.get(
            "expected_tools",
            [],
        )

        rag_rows.append(
            {
                "ID": s.get(
                    "id",
                    "—",
                ),

                "Domain": s.get(
                    "domain",
                    "",
                ).replace(
                    "_",
                    " ",
                ).title(),

                "Question": s.get(
                    "question",
                    "",
                ),

                "Reference Answer": (
                    reference[:120] + "..."
                    if len(reference) > 120
                    else reference
                ),

                "Expected Tool": (
                    expected_tools[0]
                    if expected_tools
                    else "—"
                ),
            }
        )

    df_golden = pd.DataFrame(
        rag_rows
    )

    st.dataframe(
        df_golden,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"✅ {len(rag_rows)} golden RAG samples "
        f"from 5 enterprise docs"
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # Guardrails
    # ─────────────────────────────────────────────────────────────────────

    st.subheader(
        "Guardrails Test Cases"
    )

    st.markdown(
        "These inputs test whether the safety rails "
        "correctly **block adversarial inputs** "
        "and **let through legitimate questions**."
    )

    g_rows = []

    for g in golden.get(
        "guardrails_samples",
        [],
    ):

        expected_label = (
            "🛡️ Block"
            if g["expected_blocked"]
            else "✅ Pass"
        )

        g_rows.append(
            {
                "ID": g["id"],
                "Input": g["input"],
                "Expected": expected_label,
                "Type": g["type"],
                "Description": g[
                    "description"
                ],
            }
        )

    st.dataframe(
        pd.DataFrame(g_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "6 guardrails test cases: "
        "3 adversarial (should block) + "
        "3 legit (should pass)"
    )

    with st.expander(
        "View raw golden_dataset.json"
    ):

        st.json(
            golden
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════

with tab2:

    st.subheader(
        "Live Pipeline — Collect Real Responses"
    )

    st.markdown(
        "Sends each golden question to your "
        "**running FastAPI app** (`localhost:8000/query`). "
        "Captures the actual response, all retrieved contexts, "
        "and the tool called. "
        "**Full responses and contexts are saved to "
        "`eval_results.json`.** "
        "Token reduction happens only when data is sent to RAGAS."
    )

    st.info(
        "⚠️ Make sure your FastAPI backend is running first: "
        "`uvicorn app.main:app --reload --port 8000`",
        icon="⚠️",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Buttons
    # ─────────────────────────────────────────────────────────────────────

    col_p1, col_p2, col_p3 = st.columns(
        [1, 1, 2]
    )

    run_pipeline_btn = col_p1.button(
        "▶️ Run Live Pipeline",
        type="primary",
        width="stretch",
        disabled=st.session_state.pipeline_done,
    )

    reset_btn = col_p2.button(
        "🔄 Reset & Re-run",
        width="stretch",
        disabled=not st.session_state.pipeline_done,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────

    if reset_btn:

        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = None
        st.session_state.pipeline_rows = []

        st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # Run Live Pipeline
    # ─────────────────────────────────────────────────────────────────────

    if run_pipeline_btn:

        st.session_state.pipeline_rows = []

        progress_bar = st.progress(
            0,
            text="Starting pipeline...",
        )

        live_table_slot = st.empty()
        status_slot = st.empty()

        # ─────────────────────────────────────────────────────────────────
        # Progress callback
        # ─────────────────────────────────────────────────────────────────

        def pipeline_cb(
            i,
            total,
            question,
            stage,
            response="",
        ):

            pct = int(
                (i / total) * 100
            )

            if stage == "calling":

                progress_bar.progress(
                    pct,
                    text=(
                        f"[{i + 1}/{total}] "
                        f"Calling /query: "
                        f"{question[:60]}..."
                    ),
                )

            else:

                short_q = (
                    question[:55] + "..."
                    if len(question) > 55
                    else question
                )

                short_r = (
                    response[:80] + "..."
                    if len(response) > 80
                    else response
                )

                st.session_state.pipeline_rows.append(
                    {
                        "#": i + 1,
                        "Question": short_q,
                        "Live Response (truncated)": (
                            short_r
                            if short_r
                            else "⚠️ No response"
                        ),
                        "Status": (
                            "✅"
                            if short_r
                            else "❌"
                        ),
                    }
                )

                live_table_slot.dataframe(
                    pd.DataFrame(
                        st.session_state.pipeline_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                progress_bar.progress(
                    int(
                        (
                            (i + 1)
                            / total
                        )
                        * 100
                    ),
                    text=(
                        f"[{i + 1}/{total}] "
                        f"✅ Done"
                    ),
                )

        # ─────────────────────────────────────────────────────────────────
        # Execute pipeline
        # ─────────────────────────────────────────────────────────────────

        with logfire.span(
            "🚀 Streamlit — Run Pipeline Button"
        ):

            enriched = run_pipeline(
                golden,
                progress_callback=pipeline_cb,
            )

            # ─────────────────────────────────────────────────────────────
            # CRITICAL FIX:
            #
            # Persist the enriched dataset.
            #
            # This is the part that was missing.
            # ─────────────────────────────────────────────────────────────

            save_results(
                enriched
            )

            st.session_state.enriched_dataset = (
                enriched
            )

        progress_bar.progress(
            100,
            text="✅ All responses collected!",
        )

        status_slot.success(
            f"💾 {len(enriched['rag_samples'])} "
            f"responses saved to "
            f"`eval_results.json`."
        )

        # ─────────────────────────────────────────────────────────────────
        # Guardrails tests
        # ─────────────────────────────────────────────────────────────────

        st.divider()

        st.subheader(
            "Guardrails Tests"
        )

        g_progress = st.progress(
            0,
            text="Running guardrails tests...",
        )

        g_status_slot = st.empty()

        def g_cb(
            i,
            total,
            input_text,
        ):

            g_progress.progress(
                int(
                    (i / total) * 100
                ),
                text=(
                    f"[{i + 1}/{total}] "
                    f"Testing: "
                    f"{input_text[:60]}..."
                ),
            )

        with logfire.span(
            "🛡️ Streamlit — Guardrails Tests"
        ):

            g_results = run_guardrails_eval(
                enriched[
                    "guardrails_samples"
                ],
                progress_callback=g_cb,
            )

            g_metrics = (
                compute_guardrails_metrics(
                    g_results
                )
            )

            st.session_state.guardrails_results = (
                g_results
            )

            st.session_state.pipeline_done = True

        g_progress.progress(
            100,
            text="✅ Guardrails tests complete!",
        )

        g_rows_live = []

        for r in g_results:

            result_label = {
                "TP": "🛡️ Blocked ✅",
                "TN": "✅ Passed ✅",
                "FP": (
                    "🛡️ Blocked ❌ "
                    "(False Positive)"
                ),
                "FN": (
                    "✅ Passed ❌ "
                    "(Missed)"
                ),
            }.get(
                r["result"],
                r["result"],
            )

            g_rows_live.append(
                {
                    "ID": r["id"],
                    "Input": r["input"][:70],
                    "Expected": (
                        "🛡️ Block"
                        if r["expected_blocked"]
                        else "✅ Pass"
                    ),
                    "Actual": (
                        "Blocked"
                        if r["actual_blocked"]
                        else "Passed"
                    ),
                    "Result": result_label,
                }
            )

        st.dataframe(
            pd.DataFrame(
                g_rows_live
            ),
            use_container_width=True,
            hide_index=True,
        )

        mc1, mc2, mc3, mc4 = st.columns(
            4
        )

        mc1.metric(
            "Correct",
            f"{g_metrics['correct']}/{g_metrics['total']}",
        )

        mc2.metric(
            "Precision",
            f"{g_metrics['precision']:.2f}",
        )

        mc3.metric(
            "Recall",
            f"{g_metrics['recall']:.2f}",
        )

        mc4.metric(
            "Accuracy",
            f"{g_metrics['accuracy']:.2f}",
        )

    # ─────────────────────────────────────────────────────────────────────
    # Existing saved results
    # ─────────────────────────────────────────────────────────────────────

    elif st.session_state.pipeline_done:

        st.success(
            "✅ Pipeline results loaded from "
            "`eval_results.json`."
        )

        resp_rows = []

        enriched_dataset = (
            st.session_state.enriched_dataset
        )

        for s in enriched_dataset.get(
            "rag_samples",
            [],
        ):

            actual_response = s.get(
                "actual_response",
                "",
            )

            actual_tools = s.get(
                "actual_tools_called",
                [],
            )

            actual_contexts = s.get(
                "actual_contexts",
                [],
            )

            resp_rows.append(
                {
                    "#": s.get(
                        "id",
                        "—",
                    ),

                    "Domain": s.get(
                        "domain",
                        "",
                    ).replace(
                        "_",
                        " ",
                    ).title(),

                    "Question": s.get(
                        "question",
                        "",
                    )[:60],

                    "Live Response": (
                        actual_response[:100] + "..."
                        if len(actual_response) > 100
                        else actual_response
                    ),

                    "Tool Called": (
                        actual_tools[0]
                        if actual_tools
                        else "—"
                    ),

                    "Contexts Retrieved": len(
                        actual_contexts
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(resp_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ─────────────────────────────────────────────────────────────────
        # Show saved file details
        # ─────────────────────────────────────────────────────────────────

        st.caption(
            "📁 Source: `eval_results.json` "
            "— these results can be reused for RAGAS "
            "without calling FastAPI again."
        )

        with st.expander(
            "View saved evaluation results"
        ):

            st.json(
                enriched_dataset
            )

        # ─────────────────────────────────────────────────────────────────
        # Previous guardrails
        # ─────────────────────────────────────────────────────────────────

        if st.session_state.guardrails_results:

            st.divider()

            st.subheader(
                "Guardrails Results "
                "(from previous run)"
            )

            g_rows_prev = []

            for r in (
                st.session_state.guardrails_results
            ):

                result_label = {
                    "TP": "🛡️ Blocked ✅",
                    "TN": "✅ Passed ✅",
                    "FP": "Blocked ❌ FP",
                    "FN": "Passed ❌ FN",
                }.get(
                    r["result"],
                    r["result"],
                )

                g_rows_prev.append(
                    {
                        "ID": r["id"],
                        "Input": r["input"][:70],
                        "Result": result_label,
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    g_rows_prev
                ),
                use_container_width=True,
                hide_index=True,
            )

            gm = compute_guardrails_metrics(
                st.session_state.guardrails_results
            )

            mc1, mc2, mc3, mc4 = st.columns(
                4
            )

            mc1.metric(
                "Correct",
                f"{gm['correct']}/{gm['total']}",
            )

            mc2.metric(
                "Precision",
                f"{gm['precision']:.2f}",
            )

            mc3.metric(
                "Recall",
                f"{gm['recall']:.2f}",
            )

            mc4.metric(
                "Accuracy",
                f"{gm['accuracy']:.2f}",
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════

with tab3:

    st.subheader(
        "Eval Metrics — RAGAS + Tool Correctness"
    )

    if not st.session_state.pipeline_done:

        st.warning(
            "⚠️ Complete Step 2 "
            "(Live Pipeline) first to collect responses."
        )

    else:

        st.markdown(
            "Runs all **6 metric experiments** on the "
            "stored responses. "
            "LLM-based metrics use `JUDGE_GROQ` key — "
            "samples are scored one at a time with "
            "cooldowns between samples to stay within "
            "Groq's **6,000 TPM** on-demand limit."
        )

        st.info(
            "Token key used: `JUDGE_GROQ` "
            "(separate from production key). "
            "The complete responses and contexts remain "
            "in `eval_results.json`; only the data sent "
            "to RAGAS is truncated.",
            icon="ℹ️",
        )

        run_metrics_btn = st.button(
            "▶️ Run Eval Metrics",
            type="primary",
            disabled=not st.session_state.pipeline_done,
        )

        if run_metrics_btn:

            status_slot = st.empty()

            results_slots = {}

            metric_display_names = {
                "faithfulness":
                    "Exp 1 — Faithfulness",

                "answer_relevancy":
                    "Exp 2 — Answer Relevancy",

                "context_precision":
                    "Exp 3 — Context Precision",

                "context_recall":
                    "Exp 4 — Context Recall",

                "answer_correctness":
                    "Exp 5 — Answer Correctness",

                "tool_correctness":
                    "Exp 6 — Tool Correctness",
            }

            for (
                key,
                title,
            ) in metric_display_names.items():

                results_slots[key] = st.empty()

            def status_cb(msg: str):

                status_slot.info(
                    msg
                )

            with logfire.span(
                "📊 Streamlit — Run Metrics Button"
            ):

                metric_results = _run_async(
                    run_all_metrics(
                        st.session_state.enriched_dataset,
                        status_cb=status_cb,
                    )
                )

                st.session_state.metric_results = (
                    metric_results
                )

            status_slot.success(
                "✅ All 6 experiments complete!"
            )

            for (
                key,
                title,
            ) in metric_display_names.items():

                if key in metric_results:

                    with results_slots[
                        key
                    ].container():

                        _render_metric_table(
                            metric_results[key],
                            key,
                            title,
                        )

        elif st.session_state.metric_results:

            st.success(
                "✅ Metrics already computed. "
                "Showing results below."
            )

            metric_display_names = {
                "faithfulness":
                    "Exp 1 — Faithfulness",

                "answer_relevancy":
                    "Exp 2 — Answer Relevancy",

                "context_precision":
                    "Exp 3 — Context Precision",

                "context_recall":
                    "Exp 4 — Context Recall",

                "answer_correctness":
                    "Exp 5 — Answer Correctness",

                "tool_correctness":
                    "Exp 6 — Tool Correctness",
            }

            for (
                key,
                title,
            ) in metric_display_names.items():

                if key in (
                    st.session_state.metric_results
                ):

                    _render_metric_table(
                        st.session_state.metric_results[key],
                        key,
                        title,
                    )

        # ─────────────────────────────────────────────────────────────────
        # Final Summary
        # ─────────────────────────────────────────────────────────────────

        if st.session_state.metric_results:

            st.divider()

            st.subheader(
                "Final Summary"
            )

            mr = (
                st.session_state.metric_results
            )

            summary = [
                (
                    "Faithfulness",
                    mr.get(
                        "faithfulness",
                        pd.DataFrame(),
                    ).get(
                        "faithfulness",
                        pd.Series(
                            dtype=float
                        ),
                    ).mean(),
                ),

                (
                    "Answer Relevancy",
                    mr.get(
                        "answer_relevancy",
                        pd.DataFrame(),
                    ).get(
                        "answer_relevancy",
                        pd.Series(
                            dtype=float
                        ),
                    ).mean(),
                ),

                (
                    "Context Precision",
                    mr.get(
                        "context_precision",
                        pd.DataFrame(),
                    ).get(
                        "context_precision",
                        pd.Series(
                            dtype=float
                        ),
                    ).mean(),
                ),

                (
                    "Context Recall",
                    mr.get(
                        "context_recall",
                        pd.DataFrame(),
                    ).get(
                        "context_recall",
                        pd.Series(
                            dtype=float
                        ),
                    ).mean(),
                ),

                (
                    "Answer Correctness",
                    mr.get(
                        "answer_correctness",
                        pd.DataFrame(),
                    ).get(
                        "answer_correctness",
                        pd.Series(
                            dtype=float
                        ),
                    ).mean(),
                ),

                (
                    "Tool Correctness",
                    mr.get(
                        "tool_correctness",
                        pd.DataFrame(),
                    ).get(
                        "tool_correctness",
                        pd.Series(
                            dtype=float
                        ),
                    ).mean(),
                ),
            ]

            cols = st.columns(
                len(summary)
            )

            for (
                col,
                (
                    name,
                    score,
                ),
            ) in zip(
                cols,
                summary,
            ):

                if pd.notna(score):

                    col.metric(
                        label=name,
                        value=f"{score:.2f}",
                        delta=_grade(
                            score
                        ),
                    )

            # ─────────────────────────────────────────────────────────────
            # Summary DataFrame
            # ─────────────────────────────────────────────────────────────

            summary_df = pd.DataFrame(
                [
                    {
                        "Metric": name,
                        "Score": (
                            f"{score:.3f}"
                            if pd.notna(score)
                            else "—"
                        ),
                        "Grade": (
                            _grade(score)
                            if pd.notna(score)
                            else "—"
                        ),
                    }
                    for (
                        name,
                        score,
                    ) in summary
                ]
            )

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

            # ─────────────────────────────────────────────────────────────
            # Guardrails Accuracy
            # ─────────────────────────────────────────────────────────────

            if st.session_state.guardrails_results:

                gm = compute_guardrails_metrics(
                    st.session_state.guardrails_results
                )

                st.metric(
                    label="🛡️ Guardrails Accuracy",
                    value=(
                        f"{gm['correct']}/"
                        f"{gm['total']}"
                    ),
                    delta=(
                        f"Precision "
                        f"{gm['precision']:.2f} | "
                        f"Recall "
                        f"{gm['recall']:.2f}"
                    ),
                )