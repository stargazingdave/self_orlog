# Orlog Reinforcement Learning

Train RL agents (DDQN, PPO) to play the Orlog dice game using action masking, curriculum learning, and self-play.

---

## 🎮 Live App

[Play Orlog Online](https://orlog-delta.vercel.app/bot)

![Orlog Game Screenshot](./assets/orlog_screenshot.png)

---

## 🧠 Overview

This project includes:

- A full implementation of the **Orlog game engine**
- A custom **DDQN agent**
- A **Maskable PPO agent** (SB3)
- Support for:
  - Action masking (phase-based legality)
  - Curriculum learning (opponent mixes)
  - Self-play with model pools
  - Evaluation suite (win rate, returns, head-to-head)

### Challenges

- Large discrete action space (**138 actions**)
- Stochastic gameplay (dice-based)
- Phase-dependent legal actions
- Sparse terminal rewards

---

## 🏗️ Project Structure

```
root/
├── game/        # core game logic (FSM, transitions)
├── rl/
│   ├── env/     # environment, observation encoding, configs
│   ├── ddqn/    # custom DDQN implementation
│   ├── pg/ppo/  # PPO training (SB3 MaskablePPO)
│   └── eval/    # evaluation scripts
├── outputs/     # models, logs, graphs
```

### `game/`

- Finite State Machine (FSM)
- Pure state transitions:
  - input: state (+ params)
  - output: new state

### `rl/`

- Environment + agents + training + evaluation

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- pip

### Clone

```
git clone https://github.com/stargazingdave/self_orlog.git
cd self_orlog
```

### (Optional) Virtual Environment

#### Windows (PowerShell)

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

---

## 🚀 Training

Run all commands from the project root.

---

### DDQN

#### Tune

```
python -m rl.ddqn.tune.run
```

#### Train

```
python -m rl.ddqn.train.full_5M
```

Outputs:

```
outputs/rl/ddqn/<run_name>/
  best/
  graphs/
  pool/
```

---

### Maskable PPO

#### Tune

```
python -m rl.pg.ppo.tune.lr
python -m rl.pg.ppo.tune.rollout
```

Save best params:

```
outputs/rl/pg/ppo/tune/best_hyperparams.json
```

#### Train

```
python -m rl.pg.ppo.train.full_5M
```

---

## 📊 Evaluation & Results

- Benchmarked against multiple opponent archetypes
- Head-to-head DDQN vs PPO
- Metrics:
  - Win rate
  - Mean return
  - Game length

All outputs stored under:

```
outputs/
```

---

## 🧩 Key Concepts

### Action Masking

- Only valid actions exposed per phase
- Enforced in:
  - DDQN (manual masking)
  - PPO (MaskablePPO)

### Reward Design

- Terminal reward (win/loss)
- Shaping:
  - HP advantage
  - token advantage
- Clipping + truncation penalty

### Training Strategy

- Curriculum learning (progressive opponent difficulty)
- Self-play via model pool
- Periodic evaluation + best model tracking

---

## 📌 Notes

- Models saved per curriculum stage
- Evaluation tracked per opponent
- Supports long training runs + analysis
