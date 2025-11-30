"""
Probing Tests for Financial LLM Hallucination Study
Tests model robustness: Paraphrasing, Temporal Shift, Counterfactual, Unanswerable
"""

import re
import time
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProbeResult:
    """Result of a single probe test."""
    probe_type: str
    original_question: str
    probed_question: str
    original_answer: str
    probed_answer: str
    passed: bool
    reason: str
    similarity_score: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "probe_type": self.probe_type,
            "original_question": self.original_question,
            "probed_question": self.probed_question,
            "original_answer": self.original_answer,
            "probed_answer": self.probed_answer,
            "passed": self.passed,
            "reason": self.reason,
            "similarity_score": self.similarity_score,
        }


# =============================================================================
# GROQ CLIENT FOR PARAPHRASING
# =============================================================================

_groq_client = None

def get_groq_client():
    """Get Groq client for paraphrasing."""
    global _groq_client
    if _groq_client is None:
        try:
            import streamlit as st
            from groq import Groq
            api_key = st.secrets.get("api_keys", {}).get("GROQ_API_KEY")
            if api_key:
                _groq_client = Groq(api_key=api_key)
        except Exception:
            return None
    return _groq_client


# =============================================================================
# PROBE 1: PARAPHRASING
# =============================================================================

def generate_paraphrase(question: str, max_retries: int = 2) -> Optional[str]:
    """
    Generate a paraphrased version of the question using LLM.
    Meaning should stay the same, only wording changes.
    """
    client = get_groq_client()
    if client is None:
        return None
    
    prompt = f"""Rephrase this question using different words. Keep the exact same meaning.
Only output the rephrased question, nothing else.

Original: {question}

Rephrased:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            paraphrased = response.choices[0].message.content.strip()
            # Clean up any quotes or prefixes
            paraphrased = re.sub(r'^["\'"]|["\'"]$', '', paraphrased)
            paraphrased = re.sub(r'^(Rephrased|Question|Here):\s*', '', paraphrased, flags=re.I)
            return paraphrased
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            continue
    return None


def evaluate_paraphrase(
    original_answer: str, 
    paraphrased_answer: str,
    threshold: float = 0.75
) -> tuple[bool, str, Optional[float]]:
    """
    Check if paraphrased question got similar answer.
    Returns (passed, reason, similarity_score)
    """
    if not original_answer or not paraphrased_answer:
        return False, "Empty answer", 0.0
    
    if original_answer.startswith("[") or paraphrased_answer.startswith("["):
        return False, "Error in response", 0.0
    
    # Try semantic similarity
    try:
        from sentence_transformers import SentenceTransformer
        from numpy import dot
        from numpy.linalg import norm
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([original_answer[:500], paraphrased_answer[:500]])
        similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        similarity = float(max(0, similarity))
        
        if similarity >= threshold:
            return True, f"Consistent (sim={similarity:.2f})", similarity
        else:
            return False, f"Inconsistent (sim={similarity:.2f})", similarity
    except Exception:
        # Fallback to simple token overlap
        orig_tokens = set(original_answer.lower().split())
        para_tokens = set(paraphrased_answer.lower().split())
        if not orig_tokens or not para_tokens:
            return False, "Empty tokens", 0.0
        overlap = len(orig_tokens & para_tokens) / max(len(orig_tokens), len(para_tokens))
        if overlap >= 0.5:
            return True, f"Consistent (overlap={overlap:.2f})", overlap
        return False, f"Inconsistent (overlap={overlap:.2f})", overlap


# =============================================================================
# PROBE 2: TEMPORAL SHIFT
# =============================================================================

def extract_years(text: str) -> list[str]:
    """Extract all 4-digit years from text."""
    return re.findall(r'\b(19|20)\d{2}\b', text)


def generate_temporal_shift(question: str, context: str) -> Optional[tuple[str, str]]:
    """
    Shift year in question to create mismatch with context.
    Returns (shifted_question, expected_behavior) or None if no years found.
    """
    question_years = extract_years(question)
    context_years = extract_years(context)
    
    if not question_years:
        return None
    
    # Find a year to shift to (one not in context)
    original_year = question_years[0]
    
    # Generate alternative years
    try:
        year_int = int(original_year)
        candidates = [str(year_int - 1), str(year_int + 1), str(year_int - 2)]
        
        # Prefer year not in context
        for candidate in candidates:
            if candidate not in context:
                shifted_question = question.replace(original_year, candidate, 1)
                expected = f"Should note {candidate} data not available or refuse"
                return shifted_question, expected
        
        # If all candidates in context, just use previous year
        shifted_question = question.replace(original_year, str(year_int - 1), 1)
        return shifted_question, f"Should note year mismatch"
        
    except ValueError:
        return None


def evaluate_temporal_shift(
    probed_answer: str,
    shifted_year: str
) -> tuple[bool, str]:
    """
    Check if model noticed the temporal mismatch.
    Passes if model mentions the year issue or refuses to answer.
    """
    if not probed_answer:
        return False, "Empty answer"
    
    answer_lower = probed_answer.lower()
    
    # Keywords indicating model noticed the issue
    refusal_keywords = [
        "not available", "no data", "cannot find", "don't have",
        "not provided", "no information", "cannot determine",
        "not in the context", "not mentioned", "outside",
        "doesn't contain", "does not contain", "not included"
    ]
    
    # Check if model mentions the shifted year isn't available
    for keyword in refusal_keywords:
        if keyword in answer_lower:
            return True, "Model noticed missing data"
    
    # Check if model mentions year mismatch
    if shifted_year in probed_answer and ("only" in answer_lower or "but" in answer_lower):
        return True, "Model noted year discrepancy"
    
    # If model gave a confident answer, it likely failed
    if any(word in answer_lower for word in ["is", "was", "were", "the answer"]):
        if "cannot" not in answer_lower and "don't" not in answer_lower:
            return False, "Gave answer without noting year issue"
    
    return False, "Unclear response"


# =============================================================================
# PROBE 3: COUNTERFACTUAL
# =============================================================================

COUNTERFACTUAL_PREFIXES = [
    "Given that the company reported a loss, ",
    "Assuming the revenue decreased by 50%, ",
    "If the company went bankrupt in this period, ",
    "Since the profit margin was negative, ",
    "Considering the company had zero revenue, ",
]


def generate_counterfactual(question: str) -> tuple[str, str]:
    """
    Add a false premise to the question.
    Returns (counterfactual_question, false_premise).
    """
    prefix = random.choice(COUNTERFACTUAL_PREFIXES)
    # Make question start with lowercase if adding prefix
    if question and question[0].isupper():
        question_lower = question[0].lower() + question[1:]
    else:
        question_lower = question
    
    counterfactual = prefix + question_lower
    return counterfactual, prefix.strip().rstrip(',')


def evaluate_counterfactual(
    probed_answer: str,
    false_premise: str,
    context: str
) -> tuple[bool, str]:
    """
    Check if model rejected or corrected the false premise.
    Passes if model doesn't accept the false premise.
    """
    if not probed_answer:
        return False, "Empty answer"
    
    answer_lower = probed_answer.lower()
    
    # Keywords indicating model rejected premise
    rejection_keywords = [
        "however", "but", "actually", "in fact", "incorrect",
        "not accurate", "doesn't match", "contradicts", "contrary",
        "the context shows", "according to the context", "data shows",
        "not true", "false", "wrong", "mistaken", "cannot assume"
    ]
    
    for keyword in rejection_keywords:
        if keyword in answer_lower:
            return True, "Model challenged false premise"
    
    # Check if model just went along with false premise
    premise_keywords = ["loss", "decreased", "bankrupt", "negative", "zero"]
    accepted = any(kw in answer_lower and "not" not in answer_lower[:answer_lower.find(kw)+20] 
                   for kw in premise_keywords if kw in false_premise.lower())
    
    if accepted:
        return False, "Model accepted false premise"
    
    return True, "Model did not accept false premise"


# =============================================================================
# PROBE 4: UNANSWERABLE
# =============================================================================

def generate_unanswerable(question: str) -> tuple[str, str]:
    """
    Create unanswerable scenario by noting context will be empty.
    Returns (question, instruction).
    """
    # We'll pass empty context when querying
    return question, "Context removed - model should refuse to answer"


def evaluate_unanswerable(probed_answer: str) -> tuple[bool, str]:
    """
    Check if model admitted it cannot answer without context.
    """
    if not probed_answer:
        return True, "No answer given"  # This is actually correct behavior
    
    answer_lower = probed_answer.lower()
    
    # Keywords indicating model admitted it can't answer
    refusal_keywords = [
        "cannot determine", "cannot answer", "don't have", "no context",
        "not provided", "no information", "need more", "insufficient",
        "cannot find", "not available", "unable to", "don't know",
        "no data", "not specified", "not mentioned", "would need",
        "cannot be determined", "not enough information"
    ]
    
    for keyword in refusal_keywords:
        if keyword in answer_lower:
            return True, "Model correctly refused"
    
    # If model gave a confident numerical answer, it likely hallucinated
    numbers = re.findall(r'\d+\.?\d*%?', probed_answer)
    if numbers and len(probed_answer) > 20:
        return False, "Model fabricated answer without context"
    
    # Short vague answers might be okay
    if len(probed_answer) < 50:
        return True, "Brief/unclear response"
    
    return False, "Model answered without context"


# =============================================================================
# MAIN PROBING SUITE
# =============================================================================

def run_probe(
    probe_type: str,
    question: str,
    context: str,
    query_func,  # Function to query the model: (question, context) -> answer
    original_answer: Optional[str] = None
) -> Optional[ProbeResult]:
    """
    Run a single probe test.
    
    Args:
        probe_type: One of 'paraphrase', 'temporal', 'counterfactual', 'unanswerable'
        question: Original question
        context: Original context
        query_func: Function to query model, takes (question, context) returns answer
        original_answer: Pre-computed original answer (optional)
    
    Returns:
        ProbeResult or None if probe couldn't be generated
    """
    
    # Get original answer if not provided
    if original_answer is None:
        original_answer = query_func(question, context)
    
    if probe_type == "paraphrase":
        probed_question = generate_paraphrase(question)
        if probed_question is None:
            return None
        probed_answer = query_func(probed_question, context)
        passed, reason, sim = evaluate_paraphrase(original_answer, probed_answer)
        
        return ProbeResult(
            probe_type="paraphrase",
            original_question=question,
            probed_question=probed_question,
            original_answer=original_answer,
            probed_answer=probed_answer,
            passed=passed,
            reason=reason,
            similarity_score=sim
        )
    
    elif probe_type == "temporal":
        result = generate_temporal_shift(question, context)
        if result is None:
            return None
        probed_question, expected = result
        shifted_year = extract_years(probed_question)[0] if extract_years(probed_question) else ""
        probed_answer = query_func(probed_question, context)
        passed, reason = evaluate_temporal_shift(probed_answer, shifted_year)
        
        return ProbeResult(
            probe_type="temporal",
            original_question=question,
            probed_question=probed_question,
            original_answer=original_answer,
            probed_answer=probed_answer,
            passed=passed,
            reason=reason
        )
    
    elif probe_type == "counterfactual":
        probed_question, false_premise = generate_counterfactual(question)
        probed_answer = query_func(probed_question, context)
        passed, reason = evaluate_counterfactual(probed_answer, false_premise, context)
        
        return ProbeResult(
            probe_type="counterfactual",
            original_question=question,
            probed_question=probed_question,
            original_answer=original_answer,
            probed_answer=probed_answer,
            passed=passed,
            reason=reason
        )
    
    elif probe_type == "unanswerable":
        probed_question = question
        probed_answer = query_func(question, "")  # Empty context
        passed, reason = evaluate_unanswerable(probed_answer)
        
        return ProbeResult(
            probe_type="unanswerable",
            original_question=question,
            probed_question=probed_question + " [NO CONTEXT]",
            original_answer=original_answer,
            probed_answer=probed_answer,
            passed=passed,
            reason=reason
        )
    
    return None


def run_all_probes(
    question: str,
    context: str,
    query_func,
    original_answer: Optional[str] = None,
    probe_types: list[str] = None
) -> list[ProbeResult]:
    """
    Run all probe types on a single question.
    
    Returns list of ProbeResults (skips probes that can't be generated).
    """
    if probe_types is None:
        probe_types = ["paraphrase", "temporal", "counterfactual", "unanswerable"]
    
    # Get original answer once
    if original_answer is None:
        original_answer = query_func(question, context)
    
    results = []
    for probe_type in probe_types:
        result = run_probe(probe_type, question, context, query_func, original_answer)
        if result is not None:
            results.append(result)
    
    return results


def aggregate_probe_results(results: list[ProbeResult]) -> dict:
    """Compute aggregate statistics over probe results."""
    if not results:
        return {}
    
    by_type = {}
    for r in results:
        if r.probe_type not in by_type:
            by_type[r.probe_type] = {"passed": 0, "total": 0}
        by_type[r.probe_type]["total"] += 1
        if r.passed:
            by_type[r.probe_type]["passed"] += 1
    
    summary = {
        "total_probes": len(results),
        "total_passed": sum(1 for r in results if r.passed),
        "pass_rate": sum(1 for r in results if r.passed) / len(results),
    }
    
    for probe_type, counts in by_type.items():
        summary[f"{probe_type}_pass_rate"] = counts["passed"] / counts["total"] if counts["total"] > 0 else 0
        summary[f"{probe_type}_total"] = counts["total"]
    
    return summary