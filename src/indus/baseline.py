"""Deterministic extractive baselines; no model downloads or third-party imports."""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "was",
        "were",
        "with",
        "by",
        "at",
        "as",
        "from",
        "be",
        "are",
        "this",
        "that",
        "it",
    ]
)


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def sentences(text: str) -> list[str]:
    # Do not split decimal measurements such as 0.45g.
    return [
        part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip()) if part.strip()
    ]


def summarize(text: str, method: str = "frequency", count: int = 2) -> str:
    """Rank sentences by normalized document term frequency, preserving source order."""
    if count <= 0:
        raise ValueError("Sentence count must be positive.")
    if method not in {"lead", "frequency"}:
        raise ValueError("Method must be lead or frequency.")
    parts = sentences(text)
    if method == "lead":
        return " ".join(parts[:count])
    terms = Counter(word for word in words(text) if word not in STOPWORDS)
    scores = [
        sum(terms[word] for word in words(part) if word not in STOPWORDS)
        / math.sqrt(max(len(words(part)), 1))
        for part in parts
    ]
    selected = sorted(sorted(range(len(parts)), key=lambda i: (-scores[i], i))[:count])
    return " ".join(parts[i] for i in selected)


def rouge_l(prediction: str, reference: str) -> float:
    """Token LCS F1, lowercase Unicode words, without stemming; scale 0–1."""
    left, right = words(prediction), words(reference)
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for j, other in enumerate(right, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(left) + len(right))
