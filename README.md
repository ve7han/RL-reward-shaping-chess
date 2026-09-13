# Reward Shaping for Long-Term Strategic Decision Making

**A Comparative Study Using Reinforcement Learning on Chess Puzzles**

MSc Artificial Intelligence Technology dissertation project. This repository contains the code, experiments, and analysis for a study comparing four reward shaping strategies for a reinforcement learning agent trained to solve multi-step chess puzzles.

---

## Overview

Reinforcement learning (RL) agents learn from a reward signal, but on long-horizon tasks a good decision may only pay off many steps later, which makes learning difficult. *Reward shaping* adds intermediate rewards to guide learning, but a poorly designed shaping signal can push an agent toward the wrong behaviour.

This project compares **four reward shaping strategies** under identical conditions, using chess puzzles from the [Lichess database](https://database.lichess.org/) as a controlled long-horizon task. Only the reward changes between conditions, so any difference in performance is due to the reward design alone.

### The four strategies

| Strategy | Description |
|----------|-------------|
| **Sparse** (baseline) | Reward only when the whole puzzle is solved. No shaping. |
| **Progress** | A small reward for each correct move (dense feedback). |
| **Potential** | Reward from a material-based potential function; the only strategy with a theoretical guarantee (Ng, Harada & Russell, 1999). |
| **Penalty** | A small penalty for each incorrect move. |

---

## Key Findings

- **Progress-based shaping performed best**, reaching ~12.6% average solution progress versus the sparse baseline's ~2.6% — roughly five times higher.
- Differences were **statistically significant** (one-way ANOVA, p < 0.001; confirmed with Tukey HSD).
- **Potential-based shaping** helped less than the simpler progress strategy, because its material-based signal is poorly aligned with tactical puzzles that often require sacrifices.
- **Penalty-augmented shaping** was not significantly better than no shaping at all.
- A difficulty analysis showed the strategies **converge as puzzles get harder**: reward shaping helps most when the task is within the agent's reach.

**Main conclusion:** a reward well aligned with the task matters more, in practice, than one that is theoretically sound but poorly matched.

> Note: absolute solve rates are modest by design. This study deliberately excludes search (unlike engines such as Stockfish or AlphaZero) so that the effect of the reward can be isolated. The contribution is the comparison between strategies, not raw puzzle-solving strength.

---

## Method

- **Algorithm:** Proximal Policy Optimisation (PPO) with action masking, via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) and `sb3-contrib` (`MaskablePPO`).
- **Action space:** a fixed 4,096 (64 x 64) representation with action masking, so a given action always refers to the same move. (An earlier design indexing into the legal-move list failed to learn.)
- **Board encoding:** 13 planes of 8x8 (12 piece planes + 1 side-to-move plane).
- **Network:** a small convolutional neural network (CNN).
- **Chess logic:** [python-chess](https://python-chess.readthedocs.io/).
- **Environment interface:** [Gymnasium](https://gymnasium.farama.org/).
- **Evaluation:** 80/20 stratified train/test split (no overlap), five random seeds per strategy, two metrics (full-solve rate and average progress), analysed with ANOVA and Tukey HSD.

---

## Repository Structure

```
.
├── run_experiment.py        # Environment, CNN, reward strategies, and multi-seed training loop
├── experiment.ipynb         # Notebook for data prep, analysis, statistics, and charts
├── test_gpu.py              # Quick check that CUDA / the GPU is available
├── experiment_results/      # Saved results (one JSON per run) and trained models
├── requirements.txt         # Python dependencies
└── README.md
```

*(Adjust the file list above to match your actual repository.)*

---

## Setup and Usage

### Requirements

- Python 3.11
- A CUDA-capable GPU is recommended for training (the code runs on CPU but more slowly)

### Installation

```bash
# clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (optional) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

If you need GPU support, install the CUDA build of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Running the experiment

1. Download the Lichess puzzle database and build the train/test sets (see the notebook).
2. Verify the GPU is available:
   ```bash
   python test_gpu.py
   ```
3. Run the full experiment (four strategies x five seeds). Completed runs are saved and skipped on restart, so training can be paused and resumed:
   ```bash
   python run_experiment.py
   ```
4. Open `experiment.ipynb` to produce the results table, statistical tests, and charts.

---

## Technologies

Python · PyTorch · Stable-Baselines3 · sb3-contrib · Gymnasium · python-chess · pandas · NumPy · SciPy · statsmodels · Matplotlib

---

## Key References

- Ng, A.Y., Harada, D. and Russell, S. (1999) 'Policy invariance under reward transformations', *ICML*.
- Schulman, J. et al. (2017) 'Proximal policy optimization algorithms', *arXiv:1707.06347*.
- Silver, D. et al. (2018) 'A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play', *Science*.
- Henderson, P. et al. (2018) 'Deep reinforcement learning that matters', *AAAI*.
- Knox, W.B. et al. (2023) 'Reward (mis)design for autonomous driving', *Artificial Intelligence*.

---

## Author

**Yasser Baidi**  MSc Artificial Intelligence Technology at Northumbria University London

*This project was submitted as part of an MSc dissertation. The data is from the openly available Lichess puzzle database.*
