import time, pandas as pd, chess
from run_experiment import ChessPuzzleEnv, PROJECT

train_set = pd.read_csv(f"{PROJECT}/train_set.csv")
env = ChessPuzzleEnv(train_set, reward_mode="progress")

# time each part separately over many calls
N = 20000
env.reset()

t0 = time.time()
for _ in range(N):
    env._encode_board()
print(f"_encode_board: {(time.time()-t0)/N*1e6:.1f} microsec/call")

t0 = time.time()
for _ in range(N):
    env.action_masks()
print(f"action_masks:  {(time.time()-t0)/N*1e6:.1f} microsec/call")

t0 = time.time()
for _ in range(N):
    list(env.board.legal_moves)
print(f"legal_moves:   {(time.time()-t0)/N*1e6:.1f} microsec/call")