import time, pandas as pd, torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from run_experiment import make_env, PROJECT

def main():
    train_set = pd.read_csv(f"{PROJECT}/train_set.csv")
    env = SubprocVecEnv([make_env(train_set, "progress") for _ in range(8)])
    # plain MLP - no policy_kwargs, no CNN
    model = MaskablePPO("MlpPolicy", env, verbose=0, seed=42, device="cuda")
    t0 = time.time()
    model.learn(total_timesteps=50000)
    dt = time.time() - t0
    env.close()
    print(f"MLP: 50000 steps in {dt:.0f}s -> {50000/dt:.0f} steps/sec")

if __name__ == "__main__":
    main()