"""
Streamlit Page: LLM Evaluation
Direct Q&A testing without RAG
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import time

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.llm_clients import (
    get_client,
    check_api_keys,
    AVAILABLE_MODELS,
)
from src.evaluation import evaluate_response
from src.hallucination import detect_hallucination

st.set_page_config(
    page_title="LLM Evaluation | Financial LLM Study",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LLM Evaluation (No-RAG)")
st.markdown(
    """
This page tests **LLMs answering financial questions *without* retrieval**.  
We’ll later compare this to your RAG page.

**Goal:** See *what the model actually answered* first, then look at  
accuracy scores and hallucination metrics.
"""
)

# -------------------------------------------------------------------
# API keys + model availability
# -------------------------------------------------------------------
st.sidebar.header("API Configuration")
keys_status = check_api_keys()

for provider, status in keys_status.items():
    if status:
        st.sidebar.success(f"✅ {provider} API Key")
    else:
        st.sidebar.error(f"❌ {provider} API Key Missing")

available_models = []
for model_name, config in AVAILABLE_MODELS.items():
    key_name = config["key_name"]
    provider = key_name.replace("_API_KEY", "").replace("_", " ").title()
    if provider == "Openai":
        provider = "OpenAI"
    if keys_status.get(provider, False):
        available_models.append(model_name)

if not available_models:
    st.error("No API keys configured. Please add keys to `.streamlit/secrets.toml`")
    st.code(
        """# .streamlit/secrets.toml
[api_keys]
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
GOOGLE_API_KEY = "..."
"""
    )
    st.stop()

# -------------------------------------------------------------------
# Dataset presence check
# -------------------------------------------------------------------
if "merged_df" not in st.session_state or st.session_state.merged_df is None:
    st.warning("⚠️ Dataset not loaded. Please go to Data Processing page first.")
    if st.button("Go to Data Processing"):
        st.switch_page("pages/1_Data_Processing.py")
    st.stop()

df = st.session_state.merged_df

# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Evaluation Settings")

selected_models = st.sidebar.multiselect(
    "Select Models to Test",
    options=available_models,
    default=[available_models[0]] if available_models else [],
)

sample_size = st.sidebar.slider(
    "Sample Size (questions)",
    min_value=5,
    max_value=min(100, len(df)),
    value=10,
    step=5,
)

include_context = st.sidebar.checkbox(
    "Include Context (if available)",
    value=True,
    help="Send context along with question – needed for FinQA-style questions.",
)

question_types = st.sidebar.multiselect(
    "Question Types",
    options=df["question_type"].unique().tolist(),
    default=df["question_type"].unique().tolist(),
)

compute_semantic = st.sidebar.checkbox(
    "Compute Semantic Similarity",
    value=False,
    help="Uses sentence-transformers (slower but more accurate).",
)

use_llm_judge = st.sidebar.checkbox(
    "Use LLM-as-Judge",
    value=True,
    help="Judge model scores faithfulness / hallucination (needs Groq API key).",
)

# results storage
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []
if "eval_questions_df" not in st.session_state:
    st.session_state.eval_questions_df = None

# -------------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------------
st.header("Run Evaluation")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"**Questions:** {sample_size}")
    st.markdown(f"**Models:** {len(selected_models)}")
    st.markdown(f"**Total API calls:** {sample_size * len(selected_models)}")

    est_tokens = df["total_input_tokens"].mean() * sample_size
    st.markdown(f"**Est. input tokens:** ~{int(est_tokens):,}")

with col2:
    if st.button(
        "🚀 Start Evaluation",
        type="primary",
        use_container_width=True,
        disabled=not selected_models,
    ):
        # Filter and sample data
        eval_df = (
            df[df["question_type"].isin(question_types)]
            .sample(n=min(sample_size, len(df)), random_state=42)
            .reset_index(drop=True)
        )

        # Save questions used in this run (for display later)
        st.session_state.eval_questions_df = eval_df.copy()

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_steps = len(eval_df) * len(selected_models)
        current_step = 0

        for model_name in selected_models:
            status_text.text(f"Loading {model_name}...")
            try:
                client = get_client(model_name)
            except Exception as e:
                st.error(f"Failed to initialize {model_name}: {e}")
                continue

            for idx, row in eval_df.iterrows():
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                status_text.text(f"[{model_name}] Question {idx + 1}/{len(eval_df)}")

                context = row["context_linearized"] if include_context else None
                response = client.query(row["question"], context)

                eval_result = evaluate_response(
                    response.answer,
                    row["ground_truth_answer"],
                    compute_semantic=compute_semantic,
                )

                hall_result = None
                if include_context and context:
                    hall_result = detect_hallucination(
                        answer=response.answer,
                        context=context,
                        question=row["question"],
                        use_semantic=compute_semantic,
                        use_llm_judge=use_llm_judge,
                    )

                results.append(
                    {
                        "question_id": row["question_id"],
                        "model": model_name,
                        "question": row["question"],
                        "ground_truth": row["ground_truth_answer"],
                        "model_answer": response.answer,
                        "question_type": row["question_type"],
                        "source_dataset": row["source_dataset"],
                        "context_provided": include_context,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "latency_ms": response.latency_ms,
                        "cost_usd": response.cost_usd,
                        "error": response.error,
                        # evaluation metrics
                        "exact_match": eval_result.exact_match,
                        "f1_score": eval_result.f1_score,
                        "numerical_accuracy": eval_result.numerical_accuracy,
                        "rouge_l": eval_result.rouge_l,
                        "bleu_score": eval_result.bleu_score,
                        "semantic_similarity": eval_result.semantic_similarity,
                        # hallucination-related
                        "token_overlap": hall_result.token_overlap
                        if hall_result
                        else None,
                        "semantic_faithfulness": hall_result.semantic_faithfulness
                        if hall_result
                        else None,
                        "llm_judge_score": hall_result.llm_judge_score
                        if hall_result
                        else None,
                        "llm_judge_verdict": hall_result.llm_judge_verdict
                        if hall_result
                        else None,
                        "hallucination_score": hall_result.hallucination_score
                        if hall_result
                        else None,
                        "is_hallucination": hall_result.is_hallucination
                        if hall_result
                        else None,
                    }
                )

                time.sleep(0.1)

        progress_bar.progress(1.0)
        status_text.text("✅ Evaluation complete!")

        st.session_state.eval_results = results
        st.session_state.eval_df = pd.DataFrame(results)

# -------------------------------------------------------------------
# Display results (if any)
# -------------------------------------------------------------------
if "eval_df" in st.session_state and len(st.session_state.eval_df) > 0:
    results_df = st.session_state.eval_df

    # 1) QUESTIONS ONLY – list of what we actually asked
    st.header("🧪 Questions Used in This Evaluation")

    if st.session_state.eval_questions_df is not None:
        qdf = st.session_state.eval_questions_df.copy()
    else:
        qdf = (
            results_df[
                [
                    "question_id",
                    "source_dataset",
                    "question_type",
                    "question",
                    "ground_truth",
                ]
            ]
            .drop_duplicates("question_id")
        )

    display_cols = ["source_dataset", "question_type", "question"]
    if "ground_truth_answer" in qdf.columns:
        display_cols.append("ground_truth_answer")
    elif "ground_truth" in qdf.columns:
        display_cols.append("ground_truth")

    st.dataframe(qdf[display_cols], use_container_width=True)

    # 2) QUESTION + MODEL ANSWER TABLE (front and centre)
    st.header("💬 Model Answers for Each Question")
    st.markdown(
        """
This table shows **exactly what the model said** vs the ground truth.

Use the model filter to see how a specific model behaves.
"""
    )

    filter_model = st.selectbox(
        "Filter by Model",
        options=["All"] + results_df["model"].unique().tolist(),
    )

    display_df = results_df.copy()
    if filter_model != "All":
        display_df = display_df[display_df["model"] == filter_model]

    st.dataframe(
        display_df[
            [
                "model",
                "question_type",
                "question",
                "ground_truth",
                "model_answer",
                "llm_judge_verdict",
                "hallucination_score",
                "is_hallucination",
            ]
        ].head(50),
        use_container_width=True,
        height=400,
    )

    # 3) SINGLE ANSWER INSPECTION
    st.header("🔍 Inspect a Single Answer")
    st.markdown(
        "Pick one row to show all details, including LLM-as-judge scores and token stats."
    )

    inspect_idx = st.selectbox(
        "Select a question to inspect",
        options=range(min(20, len(display_df))),
        format_func=lambda x: f"{display_df.iloc[x]['question'][:60]}...",
    )

    if inspect_idx is not None:
        inspect_row = display_df.iloc[inspect_idx]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Question:**")
            st.info(inspect_row["question"])

            st.markdown("**Ground Truth Answer:**")
            st.success(inspect_row["ground_truth"])

            st.markdown("**Model Answer:**")
            if inspect_row["error"]:
                st.error(f"Error: {inspect_row['error']}")
            else:
                st.warning(inspect_row["model_answer"])

        with col2:
            st.markdown("**Usage / Cost (for completeness):**")
            st.json(
                {
                    "Model": inspect_row["model"],
                    "Input Tokens": inspect_row["input_tokens"],
                    "Output Tokens": inspect_row["output_tokens"],
                    "Cost (USD)": float(inspect_row["cost_usd"]),
                }
            )

            st.markdown("**Accuracy Scores for this answer:**")
            eval_scores = {
                "Exact Match (1 = perfect)": float(inspect_row["exact_match"]),
                "F1 Score (0–1, higher better)": float(inspect_row["f1_score"]),
                "ROUGE-L (text overlap, higher better)": float(
                    inspect_row["rouge_l"]
                ),
                "BLEU (text overlap, higher better)": float(
                    inspect_row["bleu_score"]
                ),
            }
            if inspect_row.get("numerical_accuracy") is not None:
                eval_scores[
                    "Numerical Accuracy (0–1, higher better)"
                ] = float(inspect_row["numerical_accuracy"])
            if inspect_row.get("semantic_similarity") is not None:
                eval_scores[
                    "Semantic Similarity (0–1, higher better)"
                ] = float(inspect_row["semantic_similarity"])
            st.json(eval_scores)

            if inspect_row.get("hallucination_score") is not None:
                st.markdown("**Hallucination Analysis (LLM-as-judge + overlap):**")
                hall_info = {
                    "Token Overlap (higher = closer to context)": inspect_row[
                        "token_overlap"
                    ],
                    "Semantic Faithfulness (higher = better)": inspect_row.get(
                        "semantic_faithfulness"
                    ),
                    "LLM Judge Score (higher = more faithful)": inspect_row.get(
                        "llm_judge_score"
                    ),
                    "Hallucination Score (higher = worse)": inspect_row[
                        "hallucination_score"
                    ],
                    "Is Hallucination?": bool(inspect_row["is_hallucination"]),
                    "LLM Judge Verdict": inspect_row.get("llm_judge_verdict"),
                }
                st.json(hall_info)

    # 4) ACCURACY METRICS BY MODEL
    st.header("📊 Accuracy Metrics by Model")
    st.markdown(
        """
**How to read these:**

- **Exact Match / F1 / ROUGE-L / BLEU / Semantic Sim / Numerical Acc**  
  → **Higher = better** (closer to ground-truth answer).
"""
    )

    eval_summary = (
        results_df.groupby("model")
        .agg(
            {
                "exact_match": "mean",
                "f1_score": "mean",
                "rouge_l": "mean",
                "bleu_score": "mean",
                "numerical_accuracy": lambda x: x.dropna().mean()
                if x.dropna().any()
                else None,
                "semantic_similarity": lambda x: x.dropna().mean()
                if x.dropna().any()
                else None,
            }
        )
        .round(4)
    )

    eval_summary.columns = [
        "Exact Match",
        "F1 Score",
        "ROUGE-L",
        "BLEU",
        "Numerical Acc",
        "Semantic Sim",
    ]
    st.dataframe(eval_summary, use_container_width=True)

    chart_data = (
        results_df.groupby("model")[["f1_score", "rouge_l", "exact_match"]]
        .mean()
        .reset_index()
    )
    chart_melted = chart_data.melt(
        id_vars=["model"], var_name="Metric", value_name="Score"
    )

    fig_metrics = px.bar(
        chart_melted,
        x="model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Accuracy Metrics (higher is better)",
    )
    st.plotly_chart(fig_metrics, use_container_width=True)

    # 5) HALLUCINATION METRICS
    if results_df["hallucination_score"].notna().any():
        st.header("🧠 Hallucination Metrics by Model")
        st.markdown(
            """
**How to read these:**

- **Token Overlap / Semantic Faithfulness / LLM Judge Score** → **Higher is better**  
  (answer stays closer to the given context).
- **Hallucination Score / Hallucination Rate** → **Higher is worse**  
  (model is making stuff up more often).
"""
        )

        hall_summary = (
            results_df.groupby("model")
            .agg(
                {
                    "token_overlap": lambda x: x.dropna().mean()
                    if x.dropna().any()
                    else None,
                    "semantic_faithfulness": lambda x: x.dropna().mean()
                    if x.dropna().any()
                    else None,
                    "llm_judge_score": lambda x: x.dropna().mean()
                    if x.dropna().any()
                    else None,
                    "hallucination_score": lambda x: x.dropna().mean()
                    if x.dropna().any()
                    else None,
                    "is_hallucination": lambda x: x.dropna().sum()
                    / len(x.dropna())
                    if x.dropna().any()
                    else 0.0,
                }
            )
            .round(4)
        )

        hall_summary.columns = [
            "Token Overlap",
            "Semantic Faith.",
            "LLM Judge Score",
            "Halluc. Score",
            "Halluc. Rate",
        ]
        st.dataframe(hall_summary, use_container_width=True)

        hall_rate = (
            results_df.groupby("model")["is_hallucination"].mean().reset_index()
        )
        hall_rate.columns = ["Model", "Hallucination Rate"]

        fig_hall = px.bar(
            hall_rate,
            x="Model",
            y="Hallucination Rate",
            title="Hallucination Rate (lower is better)",
            color="Hallucination Rate",
            color_continuous_scale="RdYlGn_r",
        )
        fig_hall.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_hall, use_container_width=True)

    # 6) EXPORT
    st.header("📥 Export Results")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            "Download Results (CSV)",
            data=csv_data,
            file_name="llm_evaluation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        json_data = results_df.to_json(orient="records", indent=2)
        st.download_button(
            "Download Results (JSON)",
            data=json_data,
            file_name="llm_evaluation_results.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    st.info("👆 Select models and click **Start Evaluation** to begin testing.")

    st.subheader("Sample Questions from Dataset")
    st.dataframe(
        df[
            [
                "source_dataset",
                "question_type",
                "question",
                "ground_truth_answer",
            ]
        ].head(10),
        use_container_width=True,
    )
