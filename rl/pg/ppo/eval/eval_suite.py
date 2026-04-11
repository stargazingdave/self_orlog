from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from game.types.players import PlayerId
from rl.env.config import OpponentName
from rl.env.env import Opponent
from rl.pg.ppo.utils.env_factory import make_env


def eval_suite(
    model: MaskablePPO,
    base_out: Path,
    max_env_steps: int,
    n_eval_episodes: int = 500,
):
    opponents_list: list[tuple[str, dict[OpponentName, Opponent]]] = [
        ("random", {OpponentName.Random: Opponent(p_mix=1.0, p_action=1.0)}),
        (
            "conservative",
            {OpponentName.Conservative: Opponent(p_mix=1.0, p_action=1.0)},
        ),
        ("semi_aggr", {OpponentName.SemiAggressive: Opponent(p_mix=1.0, p_action=1.0)}),
        (
            "tempo_aggr",
            {OpponentName.TempoAggressive: Opponent(p_mix=1.0, p_action=1.0)},
        ),
        ("aggressive", {OpponentName.Aggressive: Opponent(p_mix=1.0, p_action=1.0)}),
        (
            "pressure",
            {OpponentName.HeuristicPressure: Opponent(p_mix=1.0, p_action=1.0)},
        ),
    ]

    rows = []
    for name, opp in opponents_list:
        env = DummyVecEnv(
            [
                lambda opp=opp: make_env(
                    seed=12345,
                    opponents=opp,
                    terminal_only=True,
                    max_steps=max_env_steps,
                )
            ]
        )
        terminal_infos: list[dict] = []

        def hook(locals_, globals_):
            infos = locals_.get("infos")
            dones = locals_.get("dones")
            if infos is None or dones is None:
                return
            for info, done in zip(infos, dones):
                if done and info is not None:
                    terminal_infos.append(info)

        ep_rewards, _ = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
            callback=hook,
        )

        wins = sum(
            1 for info in terminal_infos if info.get("winner_player_id") == PlayerId.P1
        )
        winrate = wins / float(n_eval_episodes)
        mean_return = float(np.mean(ep_rewards))
        rows.append((name, winrate, mean_return))

    suite_path = base_out / "eval_suite.npz"
    np.savez(
        suite_path,
        names=np.array([r[0] for r in rows]),
        winrate=np.array([r[1] for r in rows], dtype=np.float32),
        mean_return=np.array([r[2] for r in rows], dtype=np.float32),
    )

    txt_path = base_out / "eval_suite.txt"
    lines = ["=== Eval Suite (terminal-only) ===", ""]
    for n, wr, mr in rows:
        lines.append(f"{n:12s}  winrate={wr:.3f}  mean_return={mr:.4f}")
    lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n=== Eval Suite (terminal-only) ===")
    for n, wr, mr in rows:
        print(f"{n:12s}  winrate={wr:.3f}  mean_return={mr:.4f}")
    print(f"[saved] {suite_path}")
    print(f"[saved] {txt_path}\n")
