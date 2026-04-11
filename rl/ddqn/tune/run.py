import time

import optuna
from pathlib import Path

from rl.ddqn.utils.hyperparams import save_best_hyperparameters
from rl.ddqn.tune.objective import objective
from rl.utils.dirs import ensure_dir


def tune_hyperparameters_staged(
    *,
    n_trials_stage1: int = 20,
    save_csv: str = "optuna_results.csv",
):
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=25_000,  # matches report_every in objective
        ),
    )

    study.optimize(
        objective,
        n_trials=n_trials_stage1,
        n_jobs=1,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best eval winrate: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")

    path = Path("outputs") / "rl" / "ddqn" / "tune"
    ensure_dir(path)
    df = study.trials_dataframe()
    df.to_csv(path / save_csv, index=False)
    print(f"Saved: {path / save_csv}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_best_hyperparameters(
        study.best_params, f"best_hyperparameters_{timestamp}.json"
    )

    return study.best_params, study


if __name__ == "__main__":
    best_params, study = tune_hyperparameters_staged()
    print("Tuning complete. Best hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
