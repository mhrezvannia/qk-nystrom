"""Shared Markdown table formatting for the manuscript and the supplement (presentation only; values are never altered)."""
import numpy as np, pandas as pd

INT_COLS = {"m", "shots", "shots_f", "seeds", "n", "n_train", "n_test", "N", "params", "runs", "S", "layers", "n_params", "rows", "circuits (train)", "circuits (test)"}


def md_table(df, floatfmt=".2f"):
    """Count columns as integers, ∞ for inf, – for missing; numeric columns right-aligned."""
    d = df.copy(); align = []
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            f = (lambda v: f"{int(round(v))}") if c in INT_COLS else (lambda v: f"{v:{floatfmt}}")
            d[c] = d[c].map(lambda v: "–" if pd.isna(v) else ("∞" if np.isinf(v) else f(v))); align.append("right")
        else:
            d[c] = d[c].map(lambda v: "–" if (isinstance(v, float) and np.isnan(v)) else str(v).replace("|", "¦")); align.append("left")
    return d.to_markdown(index=False, disable_numparse=True, colalign=align)
