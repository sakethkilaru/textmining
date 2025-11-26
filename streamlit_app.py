"""
Financial LLM Hallucination Study
Main Streamlit Application
"""

import streamlit as st

st.set_page_config(
    page_title="Financial LLM Hallucination Study",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Financial LLM Hallucination Study")

st.markdown("""
**Objective:** Evaluate and compare hallucination rates of Large Language Models 
on financial question-answering tasks.

---

### Research Questions

1. Which LLMs demonstrate the lowest hallucination rates on financial data?
2. How does RAG (Retrieval-Augmented Generation) impact factual accuracy?
3. What types of financial questions are most prone to hallucinations?
4. What is the optimal number of context chunks for financial RAG systems?

---

### Project Workflow

Navigate using the sidebar to access each component:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📊 1. Data Processing
    - Load FinanceQA + FinQA datasets
    - Merge and standardize schema
    - Preview and export data
    - Token estimation for cost planning
    """)
    
with col2:
    st.markdown("""
    #### 🤖 2. LLM Evaluation
    - Test multiple LLMs (GPT, Claude, Gemini, Llama)
    - Direct Q&A without context
    - Hallucination detection metrics
    - Compare model performance
    """)

with col3:
    st.markdown("""
    #### 📚 3. RAG Evaluation
    - Add retrieval-augmented generation
    - Test RAG-3, RAG-5, RAG-10 configs
    - Measure context utilization
    - Compare RAG vs non-RAG
    """)

st.markdown("---")

# Quick status
st.subheader("📈 Current Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    if st.session_state.get("dataset_ready"):
        st.success("✅ Dataset loaded")
        if st.session_state.get("merged_df") is not None:
            st.metric("Questions", len(st.session_state.merged_df))
    else:
        st.warning("⏳ Dataset not loaded")
        
with status_col2:
    if st.session_state.get("llm_results"):
        st.success("✅ LLM evaluation complete")
    else:
        st.info("⏳ LLM evaluation pending")

with status_col3:
    if st.session_state.get("rag_results"):
        st.success("✅ RAG evaluation complete")
    else:
        st.info("⏳ RAG evaluation pending")

st.markdown("---")

# Models overview
st.subheader("🎯 Target Models")

models_data = {
    "Tier": ["Tier 1 (Must Test)", "Tier 1", "Tier 1", "Tier 2", "Tier 2", "Tier 2"],
    "Model": [
        "GPT-3.5-turbo", "Gemini 1.5 Flash", "Llama 3.3 70B",
        "Claude Sonnet 4.5", "GPT-4o", "Mistral Large 2"
    ],
    "Provider": ["OpenAI", "Google", "Meta (via Groq)", "Anthropic", "OpenAI", "Mistral"],
    "Cost": ["$0.50/1M", "Free tier", "Free (Groq)", "$3/1M", "$5/1M", "~$2/1M"],
    "Context": ["16K", "1M", "128K", "200K", "128K", "128K"]
}

st.table(models_data)

st.markdown("---")

st.markdown("""
### 📖 Methodology Summary

**Hallucination Detection (Ensemble Approach):**
1. **RAGAS Framework** - Faithfulness, Answer Relevancy, Context Recall
2. **Token-Level Similarity** - ROUGE-L, BLEU, Token Overlap
3. **LLM-as-Judge** - GPT-4o evaluates factual support

**Probing Types:**
- Standard Q&A across question categories
- Perturbation testing (paraphrasing, temporal shifts)
- Adversarial probing (counterfactuals, unanswerable questions)
- RAG-specific context volume experiments
""")

# Footer
st.markdown("---")
st.caption("Financial LLM Hallucination Study | Syracuse University | 2025")