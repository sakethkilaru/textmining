"""
Streamlit Page: Probing Tests
Test model robustness with paraphrasing, temporal shifts, counterfactuals, and unanswerable questions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.llm_clients import get_client, check_api_keys, AVAILABLE_MODELS
from src.probing import (
    run_probe,
    run_all_probes,
    aggregate_probe_results,
    ProbeResult,
    generate_paraphrase,
    generate_temporal_shift,
    generate_counterfactual,
)

st.set_page_config(
    page_title="Probing Tests | Financial LLM Study",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Probing Tests")
st.markdown("Test model robustness with adversarial probes.")

# Check API keys
keys_status = check_api_keys()

# Sidebar
st.sidebar.header("Configuration")

# Filter available models
available_models = []
for model_name, config in AVAILABLE_MODELS.items():
    key_name = config["key_name"]
    provider_map = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic", 
        "GOOGLE_API_KEY": "Google",
        "GROQ_API_KEY": "Groq"
    }
    provider = provider_map.get(key_name, "Unknown")
    if keys_status.get(provider, False):
        available_models.append(model_name)

if not available_models:
    st.error("No API keys configured.")
    st.stop()

selected_model = st.sidebar.selectbox(
    "Select Model to Test",
    options=available_models
)

# Check dataset
if "merged_df" not in st.session_state or st.session_state.merged_df is None:
    st.warning("⚠️ Dataset not loaded. Please go to Data Processing page first.")
    if st.button("Go to Data Processing"):
        st.switch_page("pages/1_Data_Processing.py")
    st.stop()

df = st.session_state.merged_df

# Filter to questions with context (needed for meaningful probing)
df_with_context = df[df["context_linearized"].str.len() > 50].copy()

if len(df_with_context) == 0:
    st.error("No questions with context found. Probing tests require context.")
    st.stop()

st.sidebar.markdown(f"**Questions with context:** {len(df_with_context)}")

sample_size = st.sidebar.slider(
    "Number of Questions",
    min_value=1, max_value=min(20, len(df_with_context)), value=5
)

probe_types = st.sidebar.multiselect(
    "Probe Types",
    options=["paraphrase", "temporal", "counterfactual", "unanswerable"],
    default=["paraphrase", "counterfactual"]
)

# Explanation
with st.expander("ℹ️ What are Probing Tests?"):
    st.markdown("""
    **Probing tests evaluate model robustness:**
    
    | Probe | What it Tests | Pass Condition |
    |-------|---------------|----------------|
    | **Paraphrase** | Consistency | Same answer for rephrased question |
    | **Temporal** | Attention | Notices year mismatch in question vs context |
    | **Counterfactual** | Resistance | Rejects false premises |
    | **Unanswerable** | Honesty | Refuses to answer without context |
    """)

# Initialize session state for results
if "probe_results" not in st.session_state:
    st.session_state.probe_results = []

# Main content
st.header("Run Probing Tests")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"**Model:** {selected_model}")
    st.markdown(f"**Questions:** {sample_size}")
    st.markdown(f"**Probes:** {', '.join(probe_types)}")
    st.markdown(f"**Total API calls:** ~{sample_size * (len(probe_types) + 1)}")

with col2:
    if st.button("🚀 Run Probing Tests", type="primary", use_container_width=True):
        # Sample questions
        test_df = df_with_context.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Initialize model client
        try:
            client = get_client(selected_model)
        except Exception as e:
            st.error(f"Failed to initialize {selected_model}: {e}")
            st.stop()
        
        # Query function for probing
        def query_model(question: str, context: str) -> str:
            response = client.query(question, context if context else None)
            return response.answer
        
        all_results = []
        progress = st.progress(0)
        status = st.empty()
        
        for idx, row in test_df.iterrows():
            status.text(f"Testing question {idx + 1}/{len(test_df)}...")
            
            # Get original answer first
            original_answer = query_model(row["question"], row["context_linearized"])
            
            # Run selected probes
            for probe_type in probe_types:
                status.text(f"Q{idx + 1}/{len(test_df)} - {probe_type} probe...")
                
                result = run_probe(
                    probe_type=probe_type,
                    question=row["question"],
                    context=row["context_linearized"],
                    query_func=query_model,
                    original_answer=original_answer
                )
                
                if result:
                    all_results.append({
                        "question_id": row["question_id"],
                        "model": selected_model,
                        "probe_type": result.probe_type,
                        "original_question": result.original_question,
                        "probed_question": result.probed_question,
                        "original_answer": result.original_answer,
                        "probed_answer": result.probed_answer,
                        "passed": result.passed,
                        "reason": result.reason,
                        "similarity_score": result.similarity_score,
                    })
                
                time.sleep(0.1)  # Small delay for rate limits
            
            progress.progress((idx + 1) / len(test_df))
        
        progress.progress(1.0)
        status.text("✅ Probing tests complete!")
        
        st.session_state.probe_results = all_results
        st.session_state.probe_df = pd.DataFrame(all_results)

# Display results
if "probe_df" in st.session_state and len(st.session_state.probe_df) > 0:
    results_df = st.session_state.probe_df
    
    st.header("📊 Results")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    total_probes = len(results_df)
    passed_probes = results_df["passed"].sum()
    pass_rate = passed_probes / total_probes if total_probes > 0 else 0
    
    col1.metric("Total Probes", total_probes)
    col2.metric("Passed", int(passed_probes))
    col3.metric("Pass Rate", f"{pass_rate:.1%}")
    
    # Pass rate by probe type
    st.subheader("Pass Rate by Probe Type")
    
    probe_summary = results_df.groupby("probe_type").agg({
        "passed": ["sum", "count"]
    }).reset_index()
    probe_summary.columns = ["Probe Type", "Passed", "Total"]
    probe_summary["Pass Rate"] = probe_summary["Passed"] / probe_summary["Total"]
    
    fig_probes = px.bar(
        probe_summary,
        x="Probe Type",
        y="Pass Rate",
        color="Pass Rate",
        color_continuous_scale="RdYlGn",
        title="Pass Rate by Probe Type",
        text=probe_summary["Pass Rate"].apply(lambda x: f"{x:.0%}")
    )
    fig_probes.update_traces(textposition="outside")
    fig_probes.update_layout(yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig_probes, use_container_width=True)
    
    # Detailed results table
    st.subheader("Detailed Results")
    
    # Color code pass/fail
    def highlight_pass(val):
        if val == True:
            return "background-color: #90EE90"
        elif val == False:
            return "background-color: #FFB6C1"
        return ""
    
    display_cols = ["probe_type", "passed", "reason", "original_question", "probed_question"]
    st.dataframe(
        results_df[display_cols].style.applymap(
            highlight_pass, subset=["passed"]
        ),
        use_container_width=True,
        height=400
    )
    
    # Individual inspection
    st.subheader("🔍 Probe Inspection")
    
    inspect_idx = st.selectbox(
        "Select a probe to inspect",
        options=range(len(results_df)),
        format_func=lambda x: f"{results_df.iloc[x]['probe_type']} - {results_df.iloc[x]['original_question'][:50]}..."
    )
    
    if inspect_idx is not None:
        row = results_df.iloc[inspect_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Question:**")
            st.info(row["original_question"])
            
            st.markdown("**Original Answer:**")
            st.success(row["original_answer"][:500] if row["original_answer"] else "N/A")
        
        with col2:
            st.markdown(f"**Probed Question ({row['probe_type']}):**")
            st.warning(row["probed_question"])
            
            st.markdown("**Probed Answer:**")
            if row["passed"]:
                st.success(row["probed_answer"][:500] if row["probed_answer"] else "N/A")
            else:
                st.error(row["probed_answer"][:500] if row["probed_answer"] else "N/A")
        
        # Result
        st.markdown("---")
        if row["passed"]:
            st.success(f"✅ **PASSED** - {row['reason']}")
        else:
            st.error(f"❌ **FAILED** - {row['reason']}")
        
        if row.get("similarity_score"):
            st.metric("Similarity Score", f"{row['similarity_score']:.4f}")
    
    # Export
    st.header("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            "Download Results (CSV)",
            data=csv_data,
            file_name=f"probing_results_{selected_model.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        json_data = results_df.to_json(orient="records", indent=2)
        st.download_button(
            "Download Results (JSON)",
            data=json_data,
            file_name=f"probing_results_{selected_model.replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

else:
    st.info("👆 Configure settings and click 'Run Probing Tests' to begin.")
    
    # Preview sample questions
    st.subheader("Sample Questions (with context)")
    st.dataframe(
        df_with_context[["question", "ground_truth_answer", "question_type"]].head(10),
        use_container_width=True
    )