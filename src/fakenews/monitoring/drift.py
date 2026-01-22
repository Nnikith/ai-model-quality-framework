from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from collections import Counter
from pathlib import Path
import json

import numpy as np


@dataclass
class DriftResult:
    passed: bool
    warnings: List[str]
    stats: Dict[str, Any]


def _text_lengths(texts: List[str]) -> np.ndarray:
    return np.array([len(t) for t in texts], dtype=float)


def _basic_length_stats(lengths: np.ndarray) -> Dict[str, float]:
    return {
        "count": float(len(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "p50": float(np.percentile(lengths, 50)),
        "p90": float(np.percentile(lengths, 90)),
        "p99": float(np.percentile(lengths, 99)),
    }


def _tokenize_simple(text: str) -> List[str]:
    # simple, deterministic tokenizer for monitoring (not model tokenization)
    return [tok for tok in text.lower().split() if tok.strip()]


def _top_tokens(texts: List[str], top_k: int = 50) -> List[Tuple[str, int]]:
    c = Counter()
    for t in texts:
        c.update(_tokenize_simple(t))
    return c.most_common(top_k)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def detect_data_drift(
    baseline_texts: List[str],
    current_texts: List[str],
    *,
    length_mean_shift_pct_warn: float = 0.20,
    length_p90_shift_pct_warn: float = 0.25,
    top_token_jaccard_warn: float = 0.60,
    top_k_tokens: int = 50,
) -> DriftResult:
    warnings: List[str] = []

    base_len = _text_lengths(baseline_texts)
    cur_len = _text_lengths(current_texts)

    base_stats = _basic_length_stats(base_len)
    cur_stats = _basic_length_stats(cur_len)

    def pct_change(a: float, b: float) -> float:
        # change from a -> b
        if a == 0:
            return float("inf") if b != 0 else 0.0
        return (b - a) / a

    mean_shift = pct_change(base_stats["mean"], cur_stats["mean"])
    p90_shift = pct_change(base_stats["p90"], cur_stats["p90"])

    if abs(mean_shift) >= length_mean_shift_pct_warn:
        warnings.append(
            f"Text length mean shifted by {mean_shift:.2%} (warn >= {length_mean_shift_pct_warn:.0%})"
        )
    if abs(p90_shift) >= length_p90_shift_pct_warn:
        warnings.append(
            f"Text length p90 shifted by {p90_shift:.2%} (warn >= {length_p90_shift_pct_warn:.0%})"
        )

    base_top = _top_tokens(baseline_texts, top_k=top_k_tokens)
    cur_top = _top_tokens(current_texts, top_k=top_k_tokens)

    base_set = set([t for t, _ in base_top])
    cur_set = set([t for t, _ in cur_top])

    token_jaccard = _jaccard(base_set, cur_set)
    if token_jaccard <= top_token_jaccard_warn:
        warnings.append(
            f"Top-{top_k_tokens} token overlap low (jaccard={token_jaccard:.2f}, warn <= {top_token_jaccard_warn:.2f})"
        )

    stats: Dict[str, Any] = {
        "length": {
            "baseline": base_stats,
            "current": cur_stats,
            "mean_shift_pct": mean_shift,
            "p90_shift_pct": p90_shift,
        },
        "tokens": {
            "top_k": top_k_tokens,
            "jaccard_top_tokens": token_jaccard,
            "baseline_top_tokens": base_top,
            "current_top_tokens": cur_top,
        },
    }

    return DriftResult(passed=(len(warnings) == 0), warnings=warnings, stats=stats)


def write_drift_report(path: str | Path, report: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
