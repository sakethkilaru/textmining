"""
Streamlit Page: LLM Evaluation
Direct Q&A testing without RAG

"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.llm_clients import (
    get_client, 
    check_api_keys, 
    AVAILABLE_MODELS,
    LLMResponse
)

st.set_page_config(
    page_title="LLM Evaluation | Financial LLM Study",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLM Evaluation")
st.markdown("Test LLMs on financial Q&A without retrieval augmentation.")

# Check API keys status
st.sidebar.header("API Configuration")
keys_status = check_api_keys()

for provider, status in keys_status.items():
    if status:
        st.sidebar.success(f"✅ {provider} API Key")
    else:
        st.sidebar.error(f"❌ {provider} API Key Missing")

# Filter available models based on API keys
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
    st.code("""# .streamlit/secrets.toml
[api_keys]
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
GOOGLE_API_KEY = "..."
""")
    st.stop()

# Check if dataset is loaded
if "merged_df" not in st.session_state or st.session_state.merged_df is None:
    st.warning("⚠️ Dataset not loaded. Please go to Data Processing page first.")
    if st.button("Go to Data Processing"):
        st.switch_page("pages/1_Data_Processing.py")
    st.stop()

df = st.session_state.merged_df

# Sidebar configuration
st.sidebar.header("Evaluation Settings")

selected_models = st.sidebar.multiselect(
    "Select Models to Test",
    options=available_models,
    default=[available_models[0]] if available_models else []
)

sample_size = st.sidebar.slider(
    "Sample Size (questions)",
    min_value=5, max_value=min(100, len(df)), value=10, step=5
)

include_context = st.sidebar.checkbox(
    "Include Context (if available)",
    value=True,  # Default to True so models get the financial data
    help="Send context along with question - REQUIRED for FinQA questions that need specific data"
)

question_types = st.sidebar.multiselect(
    "Question Types",
    options=df["question_type"].unique().tolist(),
    default=df["question_type"].unique().tolist()
)

# Initialize results storage
if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

# Main evaluation section
st.header("Run Evaluation")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"**Questions:** {sample_size}")
    st.markdown(f"**Models:** {len(selected_models)}")
    st.markdown(f"**Total API calls:** {sample_size * len(selected_models)}")
    
    # Cost estimate
    est_tokens = df["total_input_tokens"].mean() * sample_size
    st.markdown(f"**Est. input tokens:** ~{int(est_tokens):,}")

with col2:
    if st.button("🚀 Start Evaluation", type="primary", use_container_width=True, disabled=not selected_models):
        # Filter and sample data
        eval_df = df[df["question_type"].isin(question_types)].sample(
            n=min(sample_size, len(df)), 
            random_state=42
        ).reset_index(drop=True)
        
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
                
                results.append({
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
                    "error": response.error
                })
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Evaluation complete!")
        
        # Store results
        st.session_state.eval_results = results
        st.session_state.eval_df = pd.DataFrame(results)

# Display results
if "eval_df" in st.session_state and len(st.session_state.eval_df) > 0:
    results_df = st.session_state.eval_df
    
    st.header("📊 Results")
    
    # Summary metrics
    st.subheader("Summary by Model")
    
    summary = results_df.groupby("model").agg({
        "latency_ms": ["mean", "std"],
        "cost_usd": "sum",
        "input_tokens": "sum",
        "output_tokens": "sum",
        "error": lambda x: (x.notna() & (x != "")).sum()
    }).round(2)
    
    summary.columns = ["Avg Latency (ms)", "Std Latency", "Total Cost ($)", 
                       "Input Tokens", "Output Tokens", "Errors"]
    st.dataframe(summary, use_container_width=True)
    
    # Latency comparison
    st.subheader("Latency Comparison")
    fig_latency = px.box(
        results_df, 
        x="model", 
        y="latency_ms",
        color="model",
        title="Response Latency by Model"
    )
    fig_latency.update_layout(showlegend=False)
    st.plotly_chart(fig_latency, use_container_width=True)
    
    # Latency by question type
    st.subheader("Latency by Question Type")
    fig_qtype = px.box(
        results_df,
        x="question_type",
        y="latency_ms",
        color="model",
        title="Response Latency by Question Type and Model"
    )
    st.plotly_chart(fig_qtype, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    
    # Filter options
    filter_model = st.selectbox(
        "Filter by Model",
        options=["All"] + results_df["model"].unique().tolist()
    )
    
    display_df = results_df.copy()
    if filter_model != "All":
        display_df = display_df[display_df["model"] == filter_model]
    
    # Show comparison view
    st.dataframe(
        display_df[[
            "model", "question_type", "question", 
            "ground_truth", "model_answer", "latency_ms", "cost_usd"
        ]].head(50),
        use_container_width=True,
        height=400
    )
    
    # Individual answer inspection
    st.subheader("🔍 Answer Inspection")
    
    inspect_idx = st.selectbox(
        "Select a question to inspect",
        options=range(min(20, len(display_df))),
        format_func=lambda x: f"{display_df.iloc[x]['question'][:60]}..."
    )
    
    if inspect_idx is not None:
        inspect_row = display_df.iloc[inspect_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Question:**")
            st.info(inspect_row["question"])
            
            st.markdown("**Ground Truth:**")
            st.success(inspect_row["ground_truth"])
        
        with col2:
            st.markdown(f"**{inspect_row['model']} Answer:**")
            if inspect_row["error"]:
                st.error(f"Error: {inspect_row['error']}")
            else:
                st.warning(inspect_row["model_answer"])
            
            st.markdown("**Metrics:**")
            st.json({
                "Latency": f"{inspect_row['latency_ms']:.0f} ms",
                "Input Tokens": inspect_row["input_tokens"],
                "Output Tokens": inspect_row["output_tokens"],
                "Cost": f"${inspect_row['cost_usd']:.6f}"
            })
    
    # Export results
    st.header("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            "Download Results (CSV)",
            data=csv_data,
            file_name="llm_evaluation_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        json_data = results_df.to_json(orient="records", indent=2)
        st.download_button(
            "Download Results (JSON)",
            data=json_data,
            file_name="llm_evaluation_results.json",
            mime="application/json",
            use_container_width=True
        )

else:
    st.info("👆 Select models and click 'Start Evaluation' to begin testing.")
    
    # Show sample questions
    st.subheader("Sample Questions from Dataset")
    st.dataframe(
        df[["source_dataset", "question_type", "question", "ground_truth_answer"]].head(10),
        use_container_width=True
    )