"""Merge ablation parts into results/ablation_per_seed.csv (dedupe on task x seed x variant; base rows win over alt re-runs)."""
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "results"
order = ["ablation_part0_rerun.csv", "ablation_part1.csv", "ablation_part2.csv", "ablation_alt.csv", "ablation_part0.csv"]
frames = []
for f in order:
    if (R / f).exists():
        d = pd.read_csv(R / f); d["trained"] = d["trained"].astype(str).replace({"True": "1", "False": "0"}); d["_src"] = f; frames.append(d)
df = pd.concat(frames, ignore_index=True)
if "ablation_part0_rerun.csv" in df["_src"].values:  # the re-run supersedes the original seeds-0-4 rows for the same (task, seed, variant)
    df = df[~((df["_src"] == "ablation_part0.csv"))]
df = df.drop_duplicates(["task", "seed", "variant"], keep="first").drop(columns="_src").sort_values(["task", "seed", "variant"])
df.to_csv(R / "ablation_per_seed.csv", index=False)
print(df.shape, df.groupby("task")["seed"].nunique().to_dict(), df["variant"].nunique(), "variants")
