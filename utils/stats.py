"""
Statistics helpers (BH-FDR, running enrichment score).
"""

import numpy as np


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values in input order; NaNs preserved."""
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan)
    valid_idx = np.where(~np.isnan(pvals))[0]
    valid_p = pvals[valid_idx]
    n = len(valid_p)
    if n == 0:
        return q
    order = np.argsort(valid_p)
    raw_q = valid_p[order] * n / np.arange(1, n + 1)
    q_monotone = np.clip(np.minimum.accumulate(raw_q[::-1])[::-1], 0, 1)
    q_full = np.empty(n)
    q_full[order] = q_monotone
    q[valid_idx] = q_full
    return q


def running_enrichment_score(ranked_fc, class_mask):
    """GSEA-style running enrichment score; returns (scores, ES, ES index)."""
    N = len(ranked_fc)
    is_in = np.asarray(class_mask, dtype=bool)
    n_in, n_out = is_in.sum(), N - is_in.sum()
    if n_in == 0 or n_out == 0:
        return np.zeros(N), 0.0, 0
    in_weight = np.abs(np.asarray(ranked_fc, dtype=float)) * is_in
    in_norm = in_weight.sum()
    scores = np.zeros(N)
    running = 0.0
    for i in range(N):
        if is_in[i]:
            running += in_weight[i] / in_norm if in_norm > 0 else 0
        else:
            running -= 1.0 / n_out
        scores[i] = running
    es_idx = int(np.argmax(np.abs(scores)))
    return scores, scores[es_idx], es_idx
