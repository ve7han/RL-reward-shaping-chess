import json, glob, numpy as np
PROJECT = "D:/Documents/dissertation"
RESULTS = f"{PROJECT}/experiment_results"

rows = {}
for path in glob.glob(f"{RESULTS}/*.json"):
    r = json.load(open(path))
    rows.setdefault(r["mode"], {"full": [], "progress": []})
    rows[r["mode"]]["full"].append(r["full"])
    rows[r["mode"]]["progress"].append(r["progress"])

print(f"{'strategy':>10} | {'full solve %':>16} | {'avg progress %':>16} | seeds")
print("-" * 62)
for mode in ["sparse", "progress", "potential", "penalty"]:
    if mode in rows:
        f = np.array(rows[mode]["full"]); p = np.array(rows[mode]["progress"])
        print(f"{mode:>10} | {f.mean():6.1f} ± {f.std():4.1f}    | {p.mean():6.1f} ± {p.std():4.1f}    | {len(f)}")