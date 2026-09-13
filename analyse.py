import json, glob
import numpy as np
from scipy import stats
import statsmodels.stats.multicomp as mc

PROJECT = "D:/Documents/dissertation"
RESULTS = f"{PROJECT}/experiment_results"

# load every result into lists per strategy
data = {}
for path in glob.glob(f"{RESULTS}/*.json"):
    r = json.load(open(path))
    data.setdefault(r["mode"], {"full": [], "progress": []})
    data[r["mode"]]["full"].append(r["full"])
    data[r["mode"]]["progress"].append(r["progress"])

order = ["sparse", "progress", "potential", "penalty"]

def analyse(metric):
    print(f"\n{'='*55}\n  METRIC: {metric}\n{'='*55}")

    # means for reference
    for m in order:
        vals = data[m][metric]
        print(f"  {m:>10}: mean {np.mean(vals):5.1f}  (n={len(vals)})")

    # --- ANOVA: is there ANY difference between the four? ---
    groups = [data[m][metric] for m in order]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n  ANOVA:  F = {f_stat:.2f},  p = {p_val:.4f}")
    if p_val < 0.05:
        print("  -> Significant: the strategies do differ (p < 0.05)")
    else:
        print("  -> Not significant: no clear difference detected")

    # --- Tukey HSD: WHICH pairs differ? ---
    values, labels = [], []
    for m in order:
        values += data[m][metric]
        labels += [m] * len(data[m][metric])
    tukey = mc.pairwise_tukeyhsd(np.array(values), np.array(labels), alpha=0.05)
    print("\n  Tukey HSD (which pairs differ):")
    print(tukey)

analyse("full")
analyse("progress")