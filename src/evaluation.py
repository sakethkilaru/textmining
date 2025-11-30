"""
Evaluation Metrics for Financial LLM Hallucination Study
Compares model answers against ground truth
"""

import re
import string
from typing import Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics."""
    exact_match: float
    f1_score: float
    numerical_accuracy: Optional[float]
    rouge_l: float
    bleu_score: float
    semantic_similarity: Optional[float]
    
    def to_dict(self) -> dict:
        return {
            "exact_match": self.exact_match,
            "f1_score": self.f1_score,
            "numerical_accuracy": self.numerical_accuracy,
            "rouge_l": self.rouge_l,
            "bleu_score": self.bleu_score,
            "semantic_similarity": self.semantic_similarity,
        }


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\.\%\$]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenization."""
    return normalize_text(text).split()


# =============================================================================
# EXACT MATCH
# =============================================================================

def exact_match(prediction: str, ground_truth: str) -> float:
    """Binary exact match after normalization."""
    if not prediction or not ground_truth:
        return 0.0
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


# =============================================================================
# F1 SCORE (Token-level)
# =============================================================================

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score."""
    if not prediction or not ground_truth:
        return 0.0
    
    pred_tokens = set(tokenize(prediction))
    truth_tokens = set(tokenize(ground_truth))
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    common = pred_tokens & truth_tokens
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


# =============================================================================
# NUMERICAL ACCURACY
# =============================================================================

def extract_numbers(text: str) -> list[float]:
    """Extract all numerical values from text."""
    if not text:
        return []
    
    # Handle various formats: $1.5M, 1,500,000, 15%, 1.5 billion, etc.
    text = text.lower()
    
    # Replace word multipliers
    text = re.sub(r'(\d+\.?\d*)\s*billion', lambda m: str(float(m.group(1)) * 1e9), text)
    text = re.sub(r'(\d+\.?\d*)\s*million', lambda m: str(float(m.group(1)) * 1e6), text)
    text = re.sub(r'(\d+\.?\d*)\s*thousand', lambda m: str(float(m.group(1)) * 1e3), text)
    text = re.sub(r'(\d+\.?\d*)\s*[mb](?!\w)', lambda m: str(float(m.group(1)) * 1e6), text)
    text = re.sub(r'(\d+\.?\d*)\s*[kb](?!\w)', lambda m: str(float(m.group(1)) * 1e3), text)
    
    # Remove currency symbols and commas
    text = re.sub(r'[\$€£¥,]', '', text)
    
    # Extract numbers (including decimals and percentages)
    pattern = r'-?\d+\.?\d*%?'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            if match.endswith('%'):
                numbers.append(float(match[:-1]))
            else:
                numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers


def numerical_accuracy(prediction: str, ground_truth: str, tolerance: float = 0.01) -> Optional[float]:
    """
    Compare numerical values in prediction vs ground truth.
    Returns accuracy score or None if no numbers found in ground truth.
    """
    truth_numbers = extract_numbers(ground_truth)
    
    if not truth_numbers:
        return None  # No numerical comparison possible
    
    pred_numbers = extract_numbers(prediction)
    
    if not pred_numbers:
        return 0.0  # Ground truth has numbers but prediction doesn't
    
    # Match closest numbers
    matches = 0
    for truth_num in truth_numbers:
        for pred_num in pred_numbers:
            if truth_num == 0:
                if pred_num == 0:
                    matches += 1
                    break
            elif abs(pred_num - truth_num) / abs(truth_num) <= tolerance:
                matches += 1
                break
    
    return matches / len(truth_numbers)


# =============================================================================
# ROUGE-L (Longest Common Subsequence)
# =============================================================================

def lcs_length(s1: list, s2: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence."""
    if not prediction or not ground_truth:
        return 0.0
    
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    lcs = lcs_length(pred_tokens, truth_tokens)
    
    if lcs == 0:
        return 0.0
    
    precision = lcs / len(pred_tokens)
    recall = lcs / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


# =============================================================================
# BLEU SCORE
# =============================================================================

def get_ngrams(tokens: list, n: int) -> dict:
    """Get n-gram counts from tokens."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        ngrams[gram] = ngrams.get(gram, 0) + 1
    return ngrams


def bleu_score(prediction: str, ground_truth: str, max_n: int = 4) -> float:
    """
    Simplified BLEU score (without brevity penalty for single references).
    """
    if not prediction or not ground_truth:
        return 0.0
    
    pred_tokens = tokenize(prediction)
    truth_tokens = tokenize(ground_truth)
    
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
        pred_ngrams = get_ngrams(pred_tokens, n)
        truth_ngrams = get_ngrams(truth_tokens, n)
        
        if not pred_ngrams:
            continue
        
        matches = 0
        for gram, count in pred_ngrams.items():
            matches += min(count, truth_ngrams.get(gram, 0))
        
        precision = matches / sum(pred_ngrams.values())
        precisions.append(precision)
    
    if not precisions or all(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions (with smoothing for zeros)
    import math
    smoothed = [max(p, 1e-10) for p in precisions]
    log_avg = sum(math.log(p) for p in smoothed) / len(smoothed)
    
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(truth_tokens) / max(len(pred_tokens), 1)))
    
    return bp * math.exp(log_avg)


# =============================================================================
# SEMANTIC SIMILARITY (using sentence-transformers)
# =============================================================================

_embedding_model = None

def get_embedding_model():
    """Lazy load embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            return None
    return _embedding_model


def semantic_similarity(prediction: str, ground_truth: str) -> Optional[float]:
    """Compute cosine similarity between embeddings."""
    if not prediction or not ground_truth:
        return 0.0
    
    model = get_embedding_model()
    if model is None:
        return None  # sentence-transformers not installed
    
    try:
        # Truncate very long texts
        pred_truncated = prediction[:2000]
        truth_truncated = ground_truth[:2000]
        
        embeddings = model.encode([pred_truncated, truth_truncated])
        
        # Cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        
        similarity = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        
        return float(max(0, similarity))  # Clamp negative values
    except Exception:
        return None


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_response(
    prediction: str, 
    ground_truth: str,
    compute_semantic: bool = True
) -> EvaluationResult:
    """
    Compute all evaluation metrics for a single response.
    
    Args:
        prediction: Model-generated answer
        ground_truth: Expected correct answer
        compute_semantic: Whether to compute semantic similarity (slower)
    
    Returns:
        EvaluationResult with all metrics
    """
    # Handle empty/error responses
    if not prediction or prediction.startswith("["):
        return EvaluationResult(
            exact_match=0.0,
            f1_score=0.0,
            numerical_accuracy=None,
            rouge_l=0.0,
            bleu_score=0.0,
            semantic_similarity=0.0 if compute_semantic else None
        )
    
    return EvaluationResult(
        exact_match=exact_match(prediction, ground_truth),
        f1_score=f1_score(prediction, ground_truth),
        numerical_accuracy=numerical_accuracy(prediction, ground_truth),
        rouge_l=rouge_l(prediction, ground_truth),
        bleu_score=bleu_score(prediction, ground_truth),
        semantic_similarity=semantic_similarity(prediction, ground_truth) if compute_semantic else None
    )


def evaluate_batch(
    predictions: list[str],
    ground_truths: list[str],
    compute_semantic: bool = True
) -> list[EvaluationResult]:
    """Evaluate a batch of predictions."""
    results = []
    for pred, truth in zip(predictions, ground_truths):
        results.append(evaluate_response(pred, truth, compute_semantic))
    return results


def aggregate_metrics(results: list[EvaluationResult]) -> dict:
    """Compute aggregate statistics over evaluation results."""
    if not results:
        return {}
    
    metrics = {
        "exact_match": [],
        "f1_score": [],
        "numerical_accuracy": [],
        "rouge_l": [],
        "bleu_score": [],
        "semantic_similarity": [],
    }
    
    for r in results:
        metrics["exact_match"].append(r.exact_match)
        metrics["f1_score"].append(r.f1_score)
        if r.numerical_accuracy is not None:
            metrics["numerical_accuracy"].append(r.numerical_accuracy)
        metrics["rouge_l"].append(r.rouge_l)
        metrics["bleu_score"].append(r.bleu_score)
        if r.semantic_similarity is not None:
            metrics["semantic_similarity"].append(r.semantic_similarity)
    
    aggregated = {}
    for name, values in metrics.items():
        if values:
            aggregated[f"{name}_mean"] = sum(values) / len(values)
            aggregated[f"{name}_count"] = len(values)
    
    return aggregated