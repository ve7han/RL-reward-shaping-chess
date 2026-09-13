import time, pandas as pd, torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from run_experiment import make_env, evaluate_partial, PROJECT

def main():
    train_set = pd.read_csv(f"{PROJECT}/train_set.csv")
    test_set  = pd.read_csv(f"{PROJECT}/test_set.csv")

    env = SubprocVecEnv([make_env(train_set, "progress") for _ in range(8)])
    model = MaskablePPO("MlpPolicy", env, verbose=0, seed=42, device="cuda")  # plain MLP

    chunk = 100000
    for c in range(1, 7):                       # 600k total
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        prog, full = evaluate_partial(model, test_set, n=300)
        print(f"after {c*chunk:>7} steps -> avg progress {prog:.1f}%  |  full solves {full:.1f}%", flush=True)
    env.close()
    print("MLP progress test done.")

if __name__ == "__main__":
    main()