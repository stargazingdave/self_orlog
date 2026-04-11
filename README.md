# Orlog RL Project

This repository contains a full implementation of the Orlog game along with Reinforcement Learning (RL) agents trained using both a custom DDQN implementation and PPO (via SB3 MaskablePPO).

---

# Project Setup and Training Guide

## Prerequisites

- Python 3.10+ recommended
- `pip`
- venv (optional, for dependncies isolation)
- Git (optional, for cloning the repo)

## Clone the Repository

```bash
git clone https://github.com/stargazingdave/self_orlog.git
cd self_orlog
```

## Optional: Create a Virtual Environment

If you want to keep dependencies isolated, create and activate a virtual environment.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows (cmd)

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Project Structure

```
root/
  game/
  rl/
  outputs/
```

### `root/game/`

Contains the full game logic.

- Implemented as a **Finite State Machine (FSM)**
- State transitions are defined under:
  ```
  root/game/state_transitions/
  ```
- Each transition:
  - Receives a game state object (+ optional parameters)
  - Returns a **new updated game state**

---

### `root/rl/`

Contains all RL-related code.

#### `root/rl/env/`

- Environment implementation
- Observation encoding/decoding
- Configurations
- Opponent policies
- Helper utilities

#### `root/rl/ddqn/`

- Custom **DDQN implementation**
- Hyperparameter tuning (Optuna)
- Training scripts

#### `root/rl/pg/ppo/`

- PPO training using **SB3 MaskablePPO**
- Manual tuning scripts
- Training scripts

---

### `root/outputs/`

Stores all outputs:

- Saved models
- Training logs
- Evaluation results
- Graphs

---

## Running

All scripts should be executed from the **root directory**.

---

## DDQN

### Tune DDQN

```bash
python -m rl.ddqn.tune.run
```

- Results saved as:
  ```
  root/outputs/rl/ddqn/tune/best_hyperparameters_<timestamp>.json
  ```
- After tuning:
  - Update the training script to use the selected file

---

### Train DDQN

```bash
python -m rl.ddqn.train.full_5M
```

#### Output Structure

```
root/outputs/rl/ddqn/<run_name>/
  best/
  graphs/
  pool/
```

##### `best/`

Per curriculum mix point:

```
best/<curriculum_mix_point_name>/
  model.pt
  model.zip
  eval.txt
```

- Saved whenever a **new best model** is found during evaluation

---

##### `graphs/`

Contains performance plots:

- Aggregated:

  ```
  graphs/<run_name>/mean_return.png
  graphs/<run_name>/winrate.png
  ```

- Per opponent:
  ```
  graphs/conservative/mean_return.png
  graphs/conservative/winrate.png
  ...
  ```

---

##### `pool/`

- Stores models used for **self-play pool during training**

---

## MaskablePPO (SB3)

### Tune MaskablePPO

#### Step 1: Learning Rate

```bash
python -m rl.pg.ppo.tune.lr
```

- Results:
  ```
  root/outputs/rl/pg/ppo/tune/
  ```
- Select the best `learning_rate`

---

#### Step 2: Rollout Size

```bash
python -m rl.pg.ppo.tune.rollout
```

- Select the best `n_steps`

---

#### Save Hyperparameters

Create:

```
root/outputs/rl/pg/tune/best_hyperparams.json
```

Example:

```json
{
  "learning_rate": 1e-4,
  "n_steps": 4096
}
```

---

### Train MaskablePPO

```bash
python -m rl.pg.ppo.train.full_5M
```

- Requires:
  - `best_hyperparams.json` to exist

---

#### Output Structure

Same structure as DDQN, under:

```
root/outputs/rl/pg/ppo/<run_name>/
```

---

## Results

All results are stored under:

```
root/outputs/
```

### DDQN

- Tuning:

  ```
  rl/ddqn/tune/best_hyperparameters_<timestamp>.json
  ```

- Training:
  ```
  rl/ddqn/<run_name>/
  ```

---

### MaskablePPO

- Tuning:
  - Managed manually via output inspection

- Training:
  ```
  rl/pg/ppo/<run_name>/
  ```

---

## Notes

- Training uses **curriculum learning with opponent mixes**
- Evaluation runs periodically and tracks:
  - Mean return
  - Win rate
  - Variance
- Best models are saved **per curriculum stage**
- Self-play is supported via a model pool
