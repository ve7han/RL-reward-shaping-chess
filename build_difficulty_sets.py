import pandas as pd
PROJECT = "D:/Documents/dissertation"

df = pd.read_csv(f"{PROJECT}/lichess_db_puzzle.csv.zst")
df["n_moves"] = df["Moves"].str.split().apply(len)

# multi-step puzzles only, split into three difficulty bands by rating
base = df[df.n_moves >= 4]
bands = {
    "easy":   base[(base.Rating >= 1000) & (base.Rating < 1300)],
    "medium": base[(base.Rating >= 1300) & (base.Rating < 1600)],
    "hard":   base[(base.Rating >= 1600) & (base.Rating < 2000)],
}

for name, pool in bands.items():
    sample = pool.sample(500, random_state=42)
    sample.to_csv(f"{PROJECT}/diff_{name}.csv", index=False)
    print(f"{name}: {len(sample)} puzzles, rating {int(sample.Rating.min())}-{int(sample.Rating.max())}, avg n_moves {sample.n_moves.mean():.1f}")

print("Difficulty sets built.")