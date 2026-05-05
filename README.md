# Soccer-Twos — DRL Project Submission

> **Graders:** Jump straight to [Evaluating the Submitted Agent](#evaluating-the-submitted-agent) to run the agent, or [Repository Structure](#repository-structure) for a map of every file.

This repository contains the full training pipeline for a PPO-based Soccer-Twos agent, developed for the Deep Reinforcement Learning course project. The agent was trained using a multi-phase curriculum learning approach against the CEIA baseline opponent.

**Branch layout**

| Branch | Contents |
|---|---|
| `main` | Original unmodified starter code |
| `sim` / `submission` | All training scripts, trained agent, and results (**this branch**) |

---

## Evaluating the Submitted Agent

### 1. Setup environment

```bash
git clone https://github.com/<your-username>/soccer-twos-starter.git
cd soccer-twos-starter
git checkout submission          # or: git checkout sim

conda create --name soccertwos python=3.8 -y
conda activate soccertwos

pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt
pip install protobuf==3.20.3 pydantic==1.10.13
```

### 2. Watch the trained agent play against the CEIA baseline

```bash
python -m soccer_twos.watch -m1 my_agent -m2 ceia_baseline_agent
```

### 3. Watch the trained agent play against a random opponent

```bash
python -m soccer_twos.watch -m my_agent
```

> **Note:** `my_agent/agent.py` loads a Ray checkpoint. Make sure `CHECKPOINT_PATH` inside `my_agent/agent.py` points to the correct checkpoint on your machine, or set the environment variable:
> ```bash
> export CHECKPOINT_PATH=./ray_results/cpu_curriculum_run1/PPO_Soccer_ae5ed_00000_0_2026-04-21_20-03-24/checkpoint_001000/checkpoint-1000
> ```

---

## Training Pipeline

The agent was trained in three phases. Run each script from the repo root with `conda activate soccertwos`.

### Phase 1 — Curriculum Learning (main training run)

```bash
python train_agent_new.py
```

Trains a single PPO agent through 6 progressively harder stages:

| Stage | Name | Opponent |
|---|---|---|
| 0 | Very Easy Goal | None (ball near goal) |
| 1 | Easy Goal | None |
| 2 | Medium Goal | None |
| 3 | Hard Goal | None |
| 4 | Random Players | Random-action agents |
| 5 | vs Baseline | CEIA baseline (frozen) |

Advances to the next stage when `episode_reward_mean > 0.5`. Saves checkpoints to `ray_results/cpu_curriculum_run1/`.

### Phase 2 — Self-Play Fine-Tuning

```bash
python train_selfplay_finetune.py \
  --checkpoint ray_results/cpu_curriculum_run1/PPO_Soccer_ae5ed_00000_0_2026-04-21_20-03-24/checkpoint_001000/checkpoint-1000
```

Loads the best curriculum checkpoint and fine-tunes using `multiagent_team` self-play. Saves to `ray_results/cpu_selfplay_phase3/`.

### Phase 3 — Final Baseline Training (Ray 2.x)

```bash
python train_agent.py
```

Final training against the CEIA baseline using the Ray 2.x API with a larger network (512×512×256). Saves to `ray_results/cpu_train_run1/`.

### Alternative — Team Shaped Training

```bash
python train_agent_shaped.py
```

Controls both players on team 0 simultaneously with dense reward shaping (ball approach, shoot bonus, blocking, spread). Saves to `ray_results/cpu_team_selfplay_run1/`.

---

## Repository Structure

```
soccer-twos-starter/
│
├── README.md                        ← this file
├── requirements.txt                 ← pip dependencies
├── curriculum.yaml                  ← curriculum stage definitions (ball/player ranges)
│
├── ── Submitted Agent ──────────────────────────────────────────
├── my_agent/
│   ├── __init__.py                  ← exposes MyAgent class
│   └── agent.py                     ← loads Ray checkpoint, implements act()
│
├── ── Training Scripts (our work) ──────────────────────────────
├── train_agent_new.py               ★ PHASE 1: curriculum learning (main run)
├── train_selfplay_finetune.py       ★ PHASE 2: self-play fine-tuning from checkpoint
├── train_agent.py                   ★ PHASE 3: final baseline training (Ray 2.x API)
├── train_agent_shaped.py            ★ PHASE 1 alt: team training + dense reward shaping
├── train_ray_curriculum.py          ← original curriculum prototype (Ray 1.4)
├── train_ray_selfplay.py            ← original self-play prototype (Ray 1.4)
│
├── ── Supporting Code ──────────────────────────────────────────
├── utils.py                         ← RLLibWrapper, create_rllib_env, curriculum samplers
├── baseline_policy.py               ← loads CEIA checkpoint into a callable PyTorch policy
├── cpu_agent.py                     ← lightweight CPU inference agent wrapper
├── patch_checkpoint.py              ← utility to migrate Ray 1.x checkpoints to Ray 2.x
│
├── ── Results & Plots ──────────────────────────────────────────
├── training_curve.pdf               ★ training curve plot (curriculum vs cold-start)
├── training_curve_preview.png       ← PNG preview of the same plot
├── plot_training_curve.py           ← script that generates training_curve.pdf
│
├── ── Training Results ─────────────────────────────────────────
├── ray_results/
│   ├── cpu_curriculum_run1/         ★ Phase 1 results (1000 iters, 10M steps)
│   ├── cpu_selfplay_phase3/         ★ Phase 2 results (self-play fine-tuning)
│   ├── cpu_train_run1/              ★ Phase 3 results (final baseline training)
│   ├── cpu_shaped_curriculum_run1/  ← shaped curriculum experiment
│   ├── cpu_team_selfplay_run1/      ← team self-play experiment
│   ├── PPO_SP/                      ← early self-play prototype
│   └── PPO_SP_vs_Baseline/          ← early self-play vs baseline prototype
│
├── ── Baseline Agent ───────────────────────────────────────────
├── ceia_baseline_agent/
│   ├── agent_ray.py                 ← baseline agent class
│   └── ray_results/PPO_selfplay_twos/
│       └── .../checkpoint_002449/   ← pre-trained baseline weights (Ray 1.x)
│
├── ── Testing ──────────────────────────────────────────────────
├── test_agent.py                    ← runs my_agent for N episodes, prints win rate
├── test_env.py                      ← sanity-checks the environment loads correctly
├── curriculum_test.py               ← tests curriculum stage transitions
│
└── ── Original Starter Examples ────────────────────────────────
    ├── example_player_agent/        ← random player agent example
    ├── example_team_agent/          ← team agent example with model
    ├── example_random_players.py
    ├── example_random_teams.py
    ├── example_ray_ppo_sp_still.py
    ├── example_ray_team_vs_random.py
    ├── example_ray_dqn_sp.py
    ├── example_ray_ma_players.py
    ├── example_ray_ma_players_offline.py
    ├── example_ray_ma_teams.py
    └── example_configuration_channel.py
```

---

## Key Results

| Run | Script | Steps | Final Reward Mean | Notes |
|---|---|---|---|---|
| `cpu_curriculum_run1` | `train_agent_new.py` | 10M | **+1.69** | Main result, curriculum vs CEIA |
| `cpu_selfplay_phase3` | `train_selfplay_finetune.py` | 15M | — | Fine-tuned from curriculum ckpt |
| `cpu_train_run1` | `train_agent.py` | 17.7M | ~−1.95 | Cold-start baseline (comparison) |
| `cpu_team_selfplay_run1` | `train_agent_shaped.py` | 20M | — | Team + reward shaping experiment |

The training curve comparing curriculum vs cold-start is in [`training_curve.pdf`](training_curve.pdf).

---

## Hyperparameters (Final Training Run)

| Parameter | Value |
|---|---|
| Algorithm | PPO (Ray RLLib 1.4) |
| Network | FC 256 → 256, ReLU, shared VF |
| Learning rate | 3 × 10⁻⁴ |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| PPO clip ε | 0.2 |
| Entropy coefficient | 0.01 |
| Train batch size | 10,000 |
| SGD minibatch size | 256 |
| SGD iterations | 8 |
| Workers | 10 × 1 env |
| Total steps | 10,000,000 |
| Curriculum stages | 6 |
| Advance threshold | reward mean > 0.5 |

---

## Environment Setup (Full Reference)

```bash
conda create --name soccertwos python=3.8 -y
conda activate soccertwos
pip install pip==23.3.2 setuptools==65.5.0 wheel==0.38.4
pip cache purge
pip install -r requirements.txt
pip install protobuf==3.20.3 pydantic==1.10.13
```

Original starter code: https://github.com/bryanoliveira/soccer-twos-starter  
Environment spec: https://github.com/bryanoliveira/soccer-twos-env
