import time
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO
import pandas as pd, torch

from run_experiment import ChessPuzzleEnv, ChessCNN, mask_fn, make_env, PROJECT

def time_method(name, vec_env, steps=50000):
    print(f"  running {name} ...", flush=True)          # <-- tells you it started
    pk = dict(features_extractor_class=ChessCNN,
              features_extractor_kwargs=dict(features_dim=256))
    model = MaskablePPO("MlpPolicy", vec_env, policy_kwargs=pk,
                        verbose=0, seed=42, device="cuda")
    t0 = time.time()
    model.learn(total_timesteps=steps)
    dt = time.time() - t0
    vec_env.close()
    print(f"  {name:>14}: {steps} steps in {dt:.0f}s  ->  {steps/dt:.0f} steps/sec\n", flush=True)
    return dt

def main():
    train_set = pd.read_csv(f"{PROJECT}/train_set.csv")
    print("GPU:", torch.cuda.is_available(), "\n")

    for n in [8, 12, 16]:
        env = SubprocVecEnv([make_env(train_set, "progress") for _ in range(n)])
        time_method(f"Subproc x{n}", env)

    for n in [8, 12, 16]:
        env = DummyVecEnv([make_env(train_set, "progress") for _ in range(n)])
        time_method(f"Dummy x{n}", env)

    print("Speed test complete.")

if __name__ == "__main__":
    main()