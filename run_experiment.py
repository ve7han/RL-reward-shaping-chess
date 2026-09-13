import pandas as pd, numpy as np, chess, os, json, glob
import gymnasium as gym
from gymnasium import spaces
import torch, torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

PROJECT = "D:/Documents/dissertation"

# ---------- environment ----------
def move_to_index(move):
    return move.from_square * 64 + move.to_square

def board_potential(board, my_color):
    values = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3,
              chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
    score = 0
    for pt, val in values.items():
        score += val * len(board.pieces(pt, my_color))
        score -= val * len(board.pieces(pt, not my_color))
    if board.is_checkmate(): score += 20
    elif board.is_check():   score += 0.5
    return score

class ChessPuzzleEnv(gym.Env):
    def __init__(self, puzzles, reward_mode="sparse"):
        super().__init__()
        self.puzzles = puzzles.reset_index(drop=True)
        self.reward_mode = reward_mode
        self.observation_space = spaces.Box(low=0, high=1, shape=(13, 8, 8), dtype=np.float32)
        self.action_space = spaces.Discrete(4096)
    def _encode_board(self):
        pts = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        planes = np.zeros((13, 8, 8), dtype=np.float32)
        for i, pt in enumerate(pts):
            for sq in self.board.pieces(pt, chess.WHITE): planes[i, sq//8, sq%8] = 1.0
            for sq in self.board.pieces(pt, chess.BLACK): planes[i+6, sq//8, sq%8] = 1.0
        if self.board.turn == chess.WHITE: planes[12,:,:] = 1.0
        return planes
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.puzzle = self.puzzles.sample(1).iloc[0]
        self.board = chess.Board(self.puzzle["FEN"])
        self.moves = self.puzzle["Moves"].split()
        self.board.push_uci(self.moves[0])
        self.solution = self.moves[1:]
        self.step_idx = 0
        self.my_color = self.board.turn
        self.prev_potential = board_potential(self.board, self.my_color)
        self._legal_cache = list(self.board.legal_moves)   # cache legal moves
        return self._encode_board(), {}

    def action_masks(self):
        mask = np.zeros(4096, dtype=bool)
        # use the cached moves instead of regenerating
        idxs = [mv.from_square * 64 + mv.to_square for mv in self._legal_cache]
        mask[idxs] = True                                  # set all at once, no per-item loop
        return mask

    def step(self, action):
        terminated = False
        reward = 0.0
        # look up the chosen move from the cache, not a fresh generation
        chosen = None
        for mv in self._legal_cache:
            if mv.from_square * 64 + mv.to_square == action:
                chosen = mv
                break
        if chosen is None:
            return self._encode_board(), 0.0, True, False, {}

        expected = chess.Move.from_uci(self.solution[self.step_idx])
        if chosen == expected:
            self.board.push(chosen)
            self.step_idx += 1
            if self.reward_mode == "progress":
                reward = 0.1
            elif self.reward_mode == "potential":
                new_p = board_potential(self.board, self.my_color)
                reward = new_p - self.prev_potential
                self.prev_potential = new_p
            if self.step_idx >= len(self.solution):
                reward = 1.0
                terminated = True
            else:
                self._legal_cache = list(self.board.legal_moves)   # refresh for next step
        else:
            terminated = True
            if self.reward_mode == "penalty":
                reward = -0.1
        return self._encode_board(), reward, terminated, False, {}

class ChessCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        n = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            nf = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.linear = nn.Sequential(nn.Linear(nf, features_dim), nn.ReLU())
    def forward(self, x): return self.linear(self.cnn(x))

def mask_fn(env): return env.action_masks()

def make_env(puzzles, mode):
    def _init():
        return ActionMasker(ChessPuzzleEnv(puzzles, reward_mode=mode), mask_fn)
    return _init

def evaluate_partial(model, puzzles, n=300):
    env = ActionMasker(ChessPuzzleEnv(puzzles, reward_mode="sparse"), mask_fn)
    tot = 0.0; full = 0
    for _ in range(n):
        obs, info = env.reset(); base = env.unwrapped
        sol = len(base.solution); done = False
        while not done:
            m = env.action_masks()
            a, _ = model.predict(obs, action_masks=m, deterministic=True)
            obs, r, term, trunc, info = env.step(int(a)); done = term or trunc
        tot += base.step_idx / sol
        if base.step_idx == sol: full += 1
    return tot/n*100, full/n*100

# ---------- experiment ----------
def main():
    train_set = pd.read_csv(f"{PROJECT}/train_set.csv")
    test_set  = pd.read_csv(f"{PROJECT}/test_set.csv")
    RESULTS = f"{PROJECT}/experiment_results"; os.makedirs(RESULTS, exist_ok=True)

    STRATEGIES = ["sparse", "progress", "potential", "penalty"]
    SEEDS = [42, 43, 44, 45, 46]
    STEPS = 600000
    N_ENVS = 12                       # parallel environments -> uses your 12 cores
    pk = dict(features_extractor_class=ChessCNN,
              features_extractor_kwargs=dict(features_dim=256))

    print("GPU:", torch.cuda.is_available())
    for mode in STRATEGIES:
        print(f"\n===== {mode} =====")
        for seed in SEEDS:
            tag = f"{mode}_seed{seed}"; rp = f"{RESULTS}/{tag}.json"
            if os.path.exists(rp):
                r = json.load(open(rp))
                print(f"  [skip] {tag}: full {r['full']:.1f}%, prog {r['progress']:.1f}%")
                continue
            venv = SubprocVecEnv([make_env(train_set, mode) for _ in range(N_ENVS)])
            model = MaskablePPO("MlpPolicy", venv, policy_kwargs=pk,
                                verbose=0, seed=seed, device="cuda")
            model.learn(total_timesteps=STEPS)
            venv.close()
            prog, full = evaluate_partial(model, test_set, n=300)
            json.dump({"mode":mode,"seed":seed,"progress":prog,"full":full}, open(rp,"w"))
            model.save(f"{RESULTS}/model_{tag}")
            print(f"  [done] {tag}: full {full:.1f}%, prog {prog:.1f}%")
    print("\nAll outstanding runs complete.")

if __name__ == "__main__":       # <-- the Windows multiprocessing guard, essential
    main()