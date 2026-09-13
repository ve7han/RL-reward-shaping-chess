import time
import pandas as pd, torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import MaskablePPO
from run_experiment import ChessCNN, make_env, PROJECT

def main():
    train_set = pd.read_csv(f"{PROJECT}/train_set.csv")
    pk = dict(features_extractor_class=ChessCNN,
              features_extractor_kwargs=dict(features_dim=256))

    for dev in ["cuda", "cpu"]:
        print(f"running device={dev} ...", flush=True)
        env = SubprocVecEnv([make_env(train_set, "progress") for _ in range(8)])
        model = MaskablePPO("MlpPolicy", env, policy_kwargs=pk,
                            verbose=0, seed=42, device=dev)
        t0 = time.time()
        model.learn(total_timesteps=50000)
        dt = time.time() - t0
        env.close()
        print(f"  device={dev}: 50000 steps in {dt:.0f}s -> {50000/dt:.0f} steps/sec\n", flush=True)

if __name__ == "__main__":
    main()