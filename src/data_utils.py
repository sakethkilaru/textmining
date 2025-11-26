"""
Data utilities for Financial LLM Hallucination Study
Handles loading, processing, and merging FinanceQA and FinQA datasets
"""

import pandas as pd
from datasets import load_dataset
from typing import Optional
import hashlib


# Question type keywords for classification
QUESTION_TYPE_PATTERNS = {
    "factual_lookup": [
        "what was", "what is", "how much", "how many", 
        "what are", "list", "name", "identify"
    ],
    "single_calculation": [
        "calculate", "compute", "what percentage", "growth rate",
        "ratio", "margin", "change from", "difference"
    ],
    "multi_step_reasoning": [
        "compare", "analyze", "explain why", "what factors",
        "step by step", "derive", "based on"
    ],
    "comparative_analysis": [
        "which company", "higher", "lower", "better", "worse",
        "most", "least", "rank", "between"
    ]
}


def classify_question_type(question: str) -> str:
    """Classify question into one of four categories based on keywords."""
    question_lower = question.lower()
    
    # Check patterns in order of specificity
    for qtype, patterns in QUESTION_TYPE_PATTERNS.items():
        if any(pattern in question_lower for pattern in patterns):
            return qtype
    
    return "factual_lookup"  # Default


def generate_question_id(question: str, source: str) -> str:
    """Generate unique ID for each question."""
    hash_input = f"{source}:{question}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def linearize_table(table_data: list) -> str:
    """
    Convert table data to token-efficient linearized text.
    Input: List of lists (rows) where first row is headers
    Output: Flattened string representation
    """
    if not table_data or len(table_data) < 2:
        return ""
    
    headers = table_data[0]
    rows = table_data[1:]
    
    linearized_parts = []
    for row in rows:
        row_parts = []
        for i, cell in enumerate(row):
            if i < len(headers) and cell:
                row_parts.append(f"{headers[i]}: {cell}")
        if row_parts:
            linearized_parts.append(" | ".join(row_parts))
    
    return "\n".join(linearized_parts)


def load_finance_qa(sample_size: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Load FiQA dataset from HuggingFace (open, no auth required).
    FiQA contains financial opinion QA pairs from Reddit/StockTwits.
    Returns standardized DataFrame.
    """
    print(f"Loading FiQA dataset (sample: {sample_size})...")
    
    # Using FiQA - Financial Opinion Mining and Question Answering
    # Alternative: "FinGPT/fingpt-fiqa_qa" or "financial_phrasebank"
    try:
        dataset = load_dataset("FinGPT/fingpt-fiqa_qa", split="train")
    except Exception:
        # Fallback to another open dataset
        print("Trying fallback dataset...")
        dataset = load_dataset("gbharti/finance-alpaca", split="train")
    
    # Sample if dataset is larger than requested
    if len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    
    records = []
    for item in dataset:
        # Handle different column names across datasets
        question = item.get("input", item.get("question", item.get("instruction", "")))
        answer = item.get("output", item.get("answer", item.get("response", "")))
        context = item.get("context", item.get("input_context", ""))
        
        if not question:  # Skip empty questions
            continue
            
        records.append({
            "question_id": generate_question_id(question, "fiqa"),
            "source_dataset": "FiQA",
            "question": question,
            "ground_truth_answer": answer,
            "context_text": context,
            "context_table": None,
            "context_linearized": context,
            "question_type": classify_question_type(question),
            "requires_calculation": False,
            "gold_program": None
        })
    
    # Ensure we have enough samples
    df = pd.DataFrame(records)
    if len(df) < sample_size:
        print(f"Warning: Only {len(df)} samples available")
    
    return df.head(sample_size)


def load_finqa(sample_size: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Load FinQA dataset - trying multiple sources.
    Returns standardized DataFrame with linearized tables.
    """
    print(f"Loading FinQA dataset (sample: {sample_size})...")
    
    dataset = None
    
    # Try different sources in order of preference
    sources = [
        ("dreamerdeo/finqa", "train"),
        ("ChanceFocus/finqa-verified", "train"),
    ]
    
    for source, split in sources:
        try:
            print(f"Trying {source}...")
            dataset = load_dataset(source, split=split, trust_remote_code=True)
            print(f"Successfully loaded from {source}")
            break
        except Exception as e:
            print(f"Failed to load {source}: {e}")
            continue
    
    # If all HuggingFace sources fail, use a synthetic financial QA set
    if dataset is None:
        print("Using fallback: synthetic financial calculation questions")
        return _create_synthetic_finqa(sample_size, seed)
    
    # Sample if needed
    if len(dataset) > sample_size:
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    
    records = []
    for item in dataset:
        # Handle various column naming conventions
        question = item.get("question", item.get("query", ""))
        answer = item.get("answer", item.get("exe_ans", item.get("final_ans", "")))
        
        # Extract context components (different datasets structure this differently)
        pre_text = item.get("pre_text", item.get("pre", []))
        post_text = item.get("post_text", item.get("post", []))
        table = item.get("table", item.get("table_ori", []))
        gold_program = item.get("program", item.get("program_re", ""))
        
        # Handle if pre_text/post_text are strings instead of lists
        if isinstance(pre_text, str):
            pre_text = [pre_text] if pre_text else []
        if isinstance(post_text, str):
            post_text = [post_text] if post_text else []
        
        # Combine text context
        context_text = " ".join(pre_text + post_text)
        
        # Linearize table
        context_linearized = linearize_table(table) if table else ""
        
        # Full linearized context
        full_context = context_text
        if context_linearized:
            full_context += "\n\nTable Data:\n" + context_linearized
        
        # Determine if calculation required
        calc_keywords = ["calculate", "compute", "%", "ratio", "growth", "change"]
        requires_calc = any(kw in question.lower() for kw in calc_keywords) or bool(gold_program)
        
        records.append({
            "question_id": generate_question_id(question, "finqa"),
            "source_dataset": "FinQA",
            "question": question,
            "ground_truth_answer": str(answer),
            "context_text": context_text,
            "context_table": table,
            "context_linearized": full_context,
            "question_type": classify_question_type(question),
            "requires_calculation": requires_calc,
            "gold_program": gold_program if gold_program else None
        })
    
    return pd.DataFrame(records)


def _create_synthetic_finqa(sample_size: int, seed: int) -> pd.DataFrame:
    """
    Create synthetic financial calculation questions as fallback.
    These mimic FinQA style questions with numerical reasoning.
    """
    import random
    random.seed(seed)
    
    templates = [
        {
            "question": "What is the year-over-year revenue growth rate from {year1} to {year2}?",
            "context": "Revenue {year1}: ${rev1} million | Revenue {year2}: ${rev2} million",
            "answer": "{answer}%",
            "calc": lambda r1, r2: round((r2 - r1) / r1 * 100, 2)
        },
        {
            "question": "What was the gross profit margin in {year1}?",
            "context": "Revenue {year1}: ${rev1} million | Cost of goods sold {year1}: ${cogs} million",
            "answer": "{answer}%",
            "calc": lambda rev, cogs: round((rev - cogs) / rev * 100, 2)
        },
        {
            "question": "What is the debt-to-equity ratio for {year1}?",
            "context": "Total debt {year1}: ${debt} million | Total equity {year1}: ${equity} million",
            "answer": "{answer}",
            "calc": lambda d, e: round(d / e, 2)
        },
    ]
    
    records = []
    for i in range(sample_size):
        template = random.choice(templates)
        year1 = random.randint(2020, 2023)
        year2 = year1 + 1
        rev1 = random.randint(100, 5000)
        rev2 = int(rev1 * random.uniform(0.9, 1.3))
        cogs = int(rev1 * random.uniform(0.4, 0.7))
        debt = random.randint(50, 2000)
        equity = random.randint(100, 3000)
        
        if "growth" in template["question"]:
            answer = template["calc"](rev1, rev2)
            context = template["context"].format(year1=year1, year2=year2, rev1=rev1, rev2=rev2)
            question = template["question"].format(year1=year1, year2=year2)
        elif "margin" in template["question"]:
            answer = template["calc"](rev1, cogs)
            context = template["context"].format(year1=year1, rev1=rev1, cogs=cogs)
            question = template["question"].format(year1=year1)
        else:
            answer = template["calc"](debt, equity)
            context = template["context"].format(year1=year1, debt=debt, equity=equity)
            question = template["question"].format(year1=year1)
        
        records.append({
            "question_id": generate_question_id(question + str(i), "finqa_synthetic"),
            "source_dataset": "FinQA_Synthetic",
            "question": question,
            "ground_truth_answer": template["answer"].format(answer=answer),
            "context_text": context,
            "context_table": None,
            "context_linearized": context,
            "question_type": "single_calculation",
            "requires_calculation": True,
            "gold_program": None
        })
    
    return pd.DataFrame(records)


def merge_datasets(
    financeqa_size: int = 500,
    finqa_size: int = 400,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load and merge both datasets into unified format.
    """
    df_financeqa = load_finance_qa(sample_size=financeqa_size, seed=seed)
    df_finqa = load_finqa(sample_size=finqa_size, seed=seed)
    
    # Merge
    df_merged = pd.concat([df_financeqa, df_finqa], ignore_index=True)
    
    # Shuffle merged dataset
    df_merged = df_merged.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"Merged dataset: {len(df_merged)} total questions")
    print(f"  - FinanceQA: {len(df_financeqa)}")
    print(f"  - FinQA: {len(df_finqa)}")
    
    return df_merged


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset."""
    stats = {
        "total_questions": len(df),
        "by_source": df["source_dataset"].value_counts().to_dict(),
        "by_question_type": df["question_type"].value_counts().to_dict(),
        "with_context": len(df[df["context_linearized"].str.len() > 0]),
        "requires_calculation": df["requires_calculation"].sum(),
        "avg_question_length": df["question"].str.len().mean(),
        "avg_answer_length": df["ground_truth_answer"].str.len().mean(),
        "avg_context_length": df["context_linearized"].str.len().mean()
    }
    return stats


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Rough token estimation (OpenAI averages ~4 chars per token)."""
    if not text:
        return 0
    return int(len(text) / chars_per_token)


def add_token_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """Add token count estimates for cost planning."""
    df = df.copy()
    df["question_tokens"] = df["question"].apply(estimate_tokens)
    df["answer_tokens"] = df["ground_truth_answer"].apply(estimate_tokens)
    df["context_tokens"] = df["context_linearized"].apply(estimate_tokens)
    df["total_input_tokens"] = df["question_tokens"] + df["context_tokens"]
    return df