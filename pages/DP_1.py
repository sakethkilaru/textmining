"""
Streamlit Page: Data Processing
Load, preview, and export merged financial QA datasets
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data_utils import (
    merge_datasets, 
    get_dataset_stats, 
    add_token_estimates,
    load_finance_qa,
    load_finqa
)

st.set_page_config(
    page_title="Data Processing | Financial LLM Study",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Processing")
st.markdown("Load and preprocess FinanceQA + FinQA datasets for hallucination testing.")

# Sidebar configuration
st.sidebar.header("Dataset Configuration")

financeqa_size = st.sidebar.slider(
    "FiQA Sample Size",
    min_value=100, max_value=1000, value=500, step=50
)

finqa_size = st.sidebar.slider(
    "FinQA Sample Size", 
    min_value=100, max_value=600, value=400, step=50
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0, max_value=9999, value=42
)

# Session state for caching loaded data
if "merged_df" not in st.session_state:
    st.session_state.merged_df = None

# Load data button
col1, col2 = st.columns([1, 3])
with col1:
    load_button = st.button("🔄 Load Datasets", type="primary", use_container_width=True)

if load_button:
    with st.spinner("Loading datasets from HuggingFace..."):
        try:
            df = merge_datasets(
                financeqa_size=financeqa_size,
                finqa_size=finqa_size,
                seed=random_seed
            )
            df = add_token_estimates(df)
            st.session_state.merged_df = df
            st.success(f"✅ Loaded {len(df)} questions successfully!")
        except Exception as e:
            st.error(f"Error loading datasets: {str(e)}")

# Display data if loaded
if st.session_state.merged_df is not None:
    df = st.session_state.merged_df
    stats = get_dataset_stats(df)
    
    # Statistics overview
    st.header("Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Questions", stats["total_questions"])
    col2.metric("With Context", stats["with_context"])
    col3.metric("Require Calculation", stats["requires_calculation"])
    col4.metric("Avg Question Length", f"{stats['avg_question_length']:.0f} chars")
    
    # Distribution charts
    st.subheader("Data Distribution")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Source distribution
        source_df = pd.DataFrame({
            "Source": list(stats["by_source"].keys()),
            "Count": list(stats["by_source"].values())
        })
        fig_source = px.pie(
            source_df, values="Count", names="Source",
            title="By Source Dataset",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_source, use_container_width=True)
    
    with chart_col2:
        # Question type distribution
        qtype_df = pd.DataFrame({
            "Question Type": list(stats["by_question_type"].keys()),
            "Count": list(stats["by_question_type"].values())
        })
        fig_qtype = px.bar(
            qtype_df, x="Question Type", y="Count",
            title="By Question Type",
            color="Question Type",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_qtype.update_layout(showlegend=False)
        st.plotly_chart(fig_qtype, use_container_width=True)
    
    # Token estimates for cost planning
    st.subheader("Token Estimates (for Cost Planning)")
    
    token_col1, token_col2, token_col3 = st.columns(3)
    
    total_question_tokens = df["question_tokens"].sum()
    total_context_tokens = df["context_tokens"].sum()
    total_input_tokens = df["total_input_tokens"].sum()
    
    token_col1.metric("Total Question Tokens", f"{total_question_tokens:,}")
    token_col2.metric("Total Context Tokens", f"{total_context_tokens:,}")
    token_col3.metric("Total Input Tokens", f"{total_input_tokens:,}")
    
    # Cost estimates
    st.markdown("**Estimated API Costs (per full evaluation run):**")
    cost_data = {
        "Model": ["GPT-3.5-turbo", "GPT-4o", "Claude Sonnet 4.5", "Gemini 1.5 Flash"],
        "Input $/1M": [0.50, 5.00, 3.00, 0.00],
        "Est. Cost": [
            f"${total_input_tokens * 0.50 / 1_000_000:.2f}",
            f"${total_input_tokens * 5.00 / 1_000_000:.2f}",
            f"${total_input_tokens * 3.00 / 1_000_000:.2f}",
            "Free tier"
        ]
    }
    st.table(pd.DataFrame(cost_data))
    
    # Data preview
    st.header("Data Preview")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        source_filter = st.multiselect(
            "Filter by Source",
            options=df["source_dataset"].unique(),
            default=df["source_dataset"].unique()
        )
    with filter_col2:
        type_filter = st.multiselect(
            "Filter by Question Type",
            options=df["question_type"].unique(),
            default=df["question_type"].unique()
        )
    
    filtered_df = df[
        (df["source_dataset"].isin(source_filter)) &
        (df["question_type"].isin(type_filter))
    ]
    
    st.dataframe(
        filtered_df[[
            "question_id", "source_dataset", "question_type",
            "question", "ground_truth_answer", "total_input_tokens"
        ]].head(50),
        use_container_width=True,
        height=400
    )
    
    # Sample question detail view
    st.subheader("Sample Question Detail")
    
    sample_idx = st.selectbox(
        "Select a question to view details",
        options=filtered_df.index[:20],
        format_func=lambda x: f"{filtered_df.loc[x, 'question'][:80]}..."
    )
    
    if sample_idx is not None:
        sample = filtered_df.loc[sample_idx]
        
        detail_col1, detail_col2 = st.columns(2)
        
        with detail_col1:
            st.markdown("**Question:**")
            st.info(sample["question"])
            
            st.markdown("**Ground Truth Answer:**")
            st.success(sample["ground_truth_answer"])
            
            st.markdown("**Metadata:**")
            st.json({
                "ID": sample["question_id"],
                "Source": sample["source_dataset"],
                "Type": sample["question_type"],
                "Requires Calculation": sample["requires_calculation"],
                "Input Tokens (est.)": sample["total_input_tokens"]
            })
        
        with detail_col2:
            st.markdown("**Context (Linearized):**")
            if sample["context_linearized"]:
                st.text_area(
                    "Context",
                    value=sample["context_linearized"],
                    height=300,
                    disabled=True,
                    label_visibility="collapsed"
                )
            else:
                st.warning("No context available for this question (FinanceQA)")
            
            if sample["gold_program"]:
                st.markdown("**Gold Program (calculation steps):**")
                st.code(sample["gold_program"], language="text")
    
    # Export section
    st.header("Export Data")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="financial_qa_merged.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        # JSON export (preserves nested structures)
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name="financial_qa_merged.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Save to session state for other pages
    st.session_state.dataset_ready = True

else:
    st.info("👆 Click 'Load Datasets' to begin processing.")
    
    # Show expected structure
    st.subheader("Expected Data Schema")
    schema_df = pd.DataFrame({
        "Column": [
            "question_id", "source_dataset", "question", "ground_truth_answer",
            "context_text", "context_linearized", "question_type",
            "requires_calculation", "gold_program"
        ],
        "Type": [
            "string", "string", "string", "string",
            "string", "string", "category",
            "boolean", "string (nullable)"
        ],
        "Description": [
            "Unique identifier (MD5 hash)",
            "FinanceQA or FinQA",
            "The financial question",
            "Expected correct answer",
            "Raw text context (FinQA only)",
            "Token-efficient flattened context",
            "factual_lookup / single_calculation / multi_step / comparative",
            "Whether calculation is needed",
            "FinQA calculation program (if available)"
        ]
    })
    st.table(schema_df)