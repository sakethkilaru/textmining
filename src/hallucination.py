"""
Hallucination Detection for Financial LLM Study
Three-method ensemble: Token Overlap, Semantic Faithfulness, LLM-as-Judge
"""

import re
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class HallucinationResult:
    """Container for hallucination detection results."""
    token_overlap: float  # 0-1, higher = more overlap with context
    semantic_faithfulness: Optional[float]  # 0-1, higher = more faithful
    llm_judge_score: Optional[float]  # 0-1, higher = more supported
    llm_judge_verdict: Optional[str]  # Supported/Partial/Not Supported/Hallucinated
    hallucination_score: float  # 0-1, higher = more likely hallucination
    is_hallucination: bool  # True if score > threshold
    
    def to_dict(self) -> dict:
        return {
            "token_overlap": self.token_overlap,
            "semantic_faithfulness": self.semantic_faithfulness,
            "llm_judge_score": self.llm_judge_score,
            "llm_judge_verdict": self.llm_judge_verdict,
            "hallucination_score": self.hallucination_score,
            "is_hallucination": self.is_hallucination,
        }


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\.\%\$]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize(text: str) -> set[str]:
    """Tokenize and return unique tokens."""
    return set(normalize_text(text).split())


# =============================================================================
# METHOD 1: TOKEN OVERLAP
# =============================================================================

def compute_token_overlap(answer: str, context: str) -> float:
    """
    Compute what fraction of answer tokens appear in context.
    Higher = more grounded in context.
    """
    if not answer or not context:
        return 0.0
    
    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)
    
    if not answer_tokens:
        return 0.0
    
    # Remove common stopwords that don't indicate grounding
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                 'as', 'into', 'through', 'during', 'before', 'after', 'above',
                 'below', 'between', 'under', 'again', 'further', 'then', 'once',
                 'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                 'neither', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
                 'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
                 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                 'other', 'some', 'such', 'no', 'any', 'this', 'that', 'these',
                 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what',
                 'which', 'who', 'whom', 'if', 'because', 'while', 'although'}
    
    answer_content = answer_tokens - stopwords
    context_content = context_tokens - stopwords
    
    if not answer_content:
        return 1.0  # Only stopwords = likely fine
    
    overlap = answer_content & context_content
    return len(overlap) / len(answer_content)


# =============================================================================
# METHOD 2: SEMANTIC FAITHFULNESS
# =============================================================================

_embedding_model = None

def get_embedding_model():
    """Lazy load embedding model (shared with evaluation.py)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            return None
    return _embedding_model


def compute_semantic_faithfulness(answer: str, context: str) -> Optional[float]:
    """
    Compute semantic similarity between answer and context.
    Higher = answer is semantically grounded in context.
    """
    if not answer or not context:
        return 0.0
    
    model = get_embedding_model()
    if model is None:
        return None
    
    try:
        # Truncate for efficiency
        answer_truncated = answer[:1000]
        context_truncated = context[:3000]
        
        embeddings = model.encode([answer_truncated, context_truncated])
        
        from numpy import dot
        from numpy.linalg import norm
        
        similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        return float(max(0, similarity))
    except Exception:
        return None


# =============================================================================
# METHOD 3: LLM-AS-JUDGE
# =============================================================================

_judge_client = None

def get_judge_client():
    """Get LLM client for judging (uses Groq Llama 3.1 8B - free & fast)."""
    global _judge_client
    if _judge_client is None:
        try:
            import streamlit as st
            from groq import Groq
            api_key = st.secrets.get("api_keys", {}).get("GROQ_API_KEY")
            if api_key:
                _judge_client = Groq(api_key=api_key)
        except Exception:
            return None
    return _judge_client


def compute_llm_judge(
    answer: str, 
    context: str, 
    question: str,
    max_retries: int = 2
) -> tuple[Optional[float], Optional[str]]:
    """
    Use LLM to judge if answer is supported by context.
    Returns (score, verdict) or (None, None) if failed.
    """
    if not answer or not context:
        return None, None
    
    client = get_judge_client()
    if client is None:
        return None, None
    
    # Truncate to avoid token limits
    context_truncated = context[:2000]
    answer_truncated = answer[:500]
    
    prompt = f"""You are evaluating if an AI answer is factually supported by the given context.

Context:
{context_truncated}

Question: {question}

AI Answer: {answer_truncated}

Evaluate the answer and respond with ONLY one of these verdicts:
- SUPPORTED: Answer is fully supported by the context
- PARTIALLY_SUPPORTED: Some claims are supported, others are not
- NOT_SUPPORTED: Answer makes claims not found in context
- HALLUCINATED: Answer contains fabricated information contradicting context

Verdict:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20
            )
            
            verdict_text = response.choices[0].message.content.strip().upper()
            
            # Parse verdict to score
            if "SUPPORTED" in verdict_text and "NOT" not in verdict_text and "PARTIAL" not in verdict_text:
                return 1.0, "Supported"
            elif "PARTIALLY" in verdict_text or "PARTIAL" in verdict_text:
                return 0.5, "Partially Supported"
            elif "NOT_SUPPORTED" in verdict_text or "NOT SUPPORTED" in verdict_text:
                return 0.25, "Not Supported"
            elif "HALLUCINATED" in verdict_text or "HALLUCINATION" in verdict_text:
                return 0.0, "Hallucinated"
            else:
                # Default to partial if unclear
                return 0.5, f"Unclear: {verdict_text[:30]}"
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Brief delay before retry
                continue
            return None, None
    
    return None, None


# =============================================================================
# ENSEMBLE HALLUCINATION DETECTION
# =============================================================================

def detect_hallucination(
    answer: str,
    context: str,
    question: str = "",
    use_semantic: bool = True,
    use_llm_judge: bool = True,
    threshold: float = 0.5
) -> HallucinationResult:
    """
    Detect hallucination using ensemble of methods.
    
    Args:
        answer: Model-generated answer
        context: Source context (required for hallucination detection)
        question: Original question (for LLM judge)
        use_semantic: Whether to use semantic similarity
        use_llm_judge: Whether to use LLM-as-Judge
        threshold: Score above which to flag as hallucination
    
    Returns:
        HallucinationResult with all scores
    """
    # Handle edge cases
    if not context or not context.strip():
        # Can't detect hallucination without context
        return HallucinationResult(
            token_overlap=0.0,
            semantic_faithfulness=None,
            llm_judge_score=None,
            llm_judge_verdict=None,
            hallucination_score=0.0,
            is_hallucination=False
        )
    
    if not answer or answer.startswith("["):
        # Empty/error response = treat as hallucination
        return HallucinationResult(
            token_overlap=0.0,
            semantic_faithfulness=0.0,
            llm_judge_score=0.0,
            llm_judge_verdict="No Response",
            hallucination_score=1.0,
            is_hallucination=True
        )
    
    # Method 1: Token Overlap (always computed)
    token_overlap = compute_token_overlap(answer, context)
    
    # Method 2: Semantic Faithfulness (optional)
    semantic_faithfulness = None
    if use_semantic:
        semantic_faithfulness = compute_semantic_faithfulness(answer, context)
    
    # Method 3: LLM-as-Judge (optional)
    llm_judge_score = None
    llm_judge_verdict = None
    if use_llm_judge:
        llm_judge_score, llm_judge_verdict = compute_llm_judge(answer, context, question)
    
    # Compute ensemble hallucination score
    # Higher score = more likely to be hallucination
    scores = []
    weights = []
    
    # Token overlap: invert (low overlap = high hallucination)
    scores.append(1 - token_overlap)
    weights.append(0.3)
    
    # Semantic faithfulness: invert
    if semantic_faithfulness is not None:
        scores.append(1 - semantic_faithfulness)
        weights.append(0.3)
    
    # LLM judge: invert
    if llm_judge_score is not None:
        scores.append(1 - llm_judge_score)
        weights.append(0.4)
    
    # Normalize weights
    total_weight = sum(weights)
    hallucination_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    
    return HallucinationResult(
        token_overlap=token_overlap,
        semantic_faithfulness=semantic_faithfulness,
        llm_judge_score=llm_judge_score,
        llm_judge_verdict=llm_judge_verdict,
        hallucination_score=hallucination_score,
        is_hallucination=hallucination_score > threshold
    )


def compute_hallucination_rate(results: list[HallucinationResult]) -> dict:
    """Compute aggregate hallucination statistics."""
    if not results:
        return {}
    
    valid_results = [r for r in results if r.token_overlap > 0 or r.semantic_faithfulness is not None]
    
    if not valid_results:
        return {}
    
    hallucination_count = sum(1 for r in valid_results if r.is_hallucination)
    
    return {
        "total_evaluated": len(valid_results),
        "hallucination_count": hallucination_count,
        "hallucination_rate": hallucination_count / len(valid_results),
        "avg_token_overlap": sum(r.token_overlap for r in valid_results) / len(valid_results),
        "avg_hallucination_score": sum(r.hallucination_score for r in valid_results) / len(valid_results),
    }