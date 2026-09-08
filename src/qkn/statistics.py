"""Paired statistical comparisons across seeds with multiple-comparison correction."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats


def paired_comparison(a: np.ndarray, b: np.ndarray, label: str = "") -> dict:
    """Compare paired samples a (candidate) and b (reference).  Returns t-test, Wilcoxon, Cohen's d_z,
    mean difference and bootstrap 95% CI of the mean difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    out = dict(comparison=label, n=n, mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)) if n > 1 else np.nan)
    if n > 1 and d.std(ddof=1) > 0:
        t, p = stats.ttest_rel(a, b)
        out.update(t=float(t), p_t=float(p), cohen_dz=float(d.mean() / d.std(ddof=1)))
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox")
            out.update(p_wilcoxon=float(w.pvalue))
        except ValueError:
            out.update(p_wilcoxon=np.nan)
    else:
        out.update(t=np.nan, p_t=1.0 if d.std(ddof=1) == 0 else np.nan, cohen_dz=0.0, p_wilcoxon=1.0)
    rng = np.random.default_rng(0)
    boots = [rng.choice(d, n, replace=True).mean() for _ in range(5000)]
    out.update(ci_lo=float(np.percentile(boots, 2.5)), ci_hi=float(np.percentile(boots, 97.5)))
    return out


def holm_correction(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i]) if not np.isnan(p[i]) else np.nan
        running = max(running, val) if not np.isnan(val) else running
        adj[i] = running if not np.isnan(val) else np.nan
    return adj


def compare_table(df: pd.DataFrame, group_cols: list[str], method_col: str, value_col: str,
                  candidate: str, reference: str, seed_col: str = "seed") -> pd.DataFrame:
    """For each group, paired comparison of candidate vs reference across seeds; Holm-adjusted p (t-test)."""
    rows = []
    for key, g in df.groupby(group_cols):
        piv = g.pivot_table(index=seed_col, columns=method_col, values=value_col)
        if candidate in piv and reference in piv:
            pv = piv[[candidate, reference]].dropna()
            r = paired_comparison(pv[candidate].values, pv[reference].values, f"{candidate} - {reference}")
            r.update(dict(zip(group_cols, key if isinstance(key, tuple) else (key,))))
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_t_holm"] = holm_correction(out["p_t"].values)
    return out
