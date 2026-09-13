import json
import numpy as np
PROJECT = "D:/Documents/dissertation"
results = json.load(open(f"{PROJECT}/difficulty_results.json"))

BANDS = ["easy", "medium", "hard"]
STRATEGIES = ["sparse", "progress", "potential", "penalty"]

# organise: strategy -> band -> list of progress values across seeds
agg = {m: {b: [] for b in BANDS} for m in STRATEGIES}
for r in results.values():
    agg[r["mode"]][r["band"]].append(r["progress"])

print("Average PROGRESS % by strategy and difficulty (mean over seeds):\n")
print(f"{'strategy':>10} | {'easy':>8} | {'medium':>8} | {'hard':>8}")
print("-" * 44)
for m in STRATEGIES:
    row = f"{m:>10} |"
    for b in BANDS:
        vals = agg[m][b]
        row += f" {np.mean(vals):6.1f}  |" if vals else "    -    |"
    print(row)