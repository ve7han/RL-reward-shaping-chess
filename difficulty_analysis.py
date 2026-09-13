import json, glob, os
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from run_experiment import ChessPuzzleEnv, ChessCNN, mask_fn, evaluate_partial, PROJECT

RESULTS = f"{PROJECT}/experiment_results"
BANDS = ["easy", "medium", "hard"]
STRATEGIES = ["sparse", "progress", "potential", "penalty"]
SEEDS = [42, 43, 44, 45, 46]

# load the three difficulty test sets
band_sets = {b: pd.read_csv(f"{PROJECT}/diff_{b}.csv") for b in BANDS}

out_path = f"{PROJECT}/difficulty_results.json"
results = {}
if os.path.exists(out_path):
    results = json.load(open(out_path))          # resume if partly done

for mode in STRATEGIES:
    for seed in SEEDS:
        tag = f"{mode}_seed{seed}"
        model_path = f"{RESULTS}/model_{tag}.zip"
        if not os.path.exists(model_path):
            print(f"  [missing] {tag}, skipping")
            continue
        model = MaskablePPO.load(model_path)
        for b in BANDS:
            key = f"{tag}_{b}"
            if key in results:
                continue                          # already done
            prog, full = evaluate_partial(model, band_sets[b], n=300)
            results[key] = {"mode": mode, "seed": seed, "band": b,
                            "progress": prog, "full": full}
            json.dump(results, open(out_path, "w"))   # save after each
            print(f"  {key}: progress {prog:.1f}%, full {full:.1f}%")

print("\nDifficulty analysis complete.")