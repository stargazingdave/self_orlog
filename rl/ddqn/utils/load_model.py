from __future__ import annotations

import json
from typing import Literal
import io
import os
import zipfile
import json
import torch
from pathlib import Path

from rl.ddqn.config import DDQNConfig
from rl.ddqn.model import DDQN, QNetConfigurable
from rl.ddqn.utils.dirs import build_checkpoint_dir
from rl.env.env import OrlogEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_best_hyperparams() -> dict[str, float | int]:
    path = (
        Path("outputs")
        / "rl"
        / "ddqn"
        / "tune"
        / "best_hyperparameters_20260331_075257.json"
    )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_zip(zip_path: Path | str):
    """
    Expects zip created by your save_checkpoint_zip():
      - model.pt  (torch.save dict with model_state_dict, params, meta)
      - params.json
      - meta.json
    Returns: (ckpt_dict, params_dict, meta_dict)
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        model_bytes = zf.read("model.pt")
        params = json.loads(zf.read("params.json").decode("utf-8"))
        meta = json.loads(zf.read("meta.json").decode("utf-8"))

    ckpt = torch.load(
        io.BytesIO(model_bytes),
        map_location=device,
        weights_only=False,
    )
    return ckpt, params, meta


def load_model(
    run_name: str,
    checkpoint: Literal["best", "last"],
    env: OrlogEnv,
    path: Path | str | None = None,
) -> DDQN:
    if path is None:
        path = build_checkpoint_dir(run_name, checkpoint) / "model.zip"

    ckpt, params, meta = load_checkpoint_zip(path)
    print("[loaded zip]", path)
    print("[loaded meta]", meta)
    print("[loaded params keys]", sorted(list(params.keys())))

    out = DDQN(env, config=DDQNConfig(**params))

    q = QNetConfigurable(
        out.obs_dim, out.n_actions, params["hidden_dim"], params["n_layers"]
    ).to(device)
    q_targ = QNetConfigurable(
        out.obs_dim, out.n_actions, params["hidden_dim"], params["n_layers"]
    ).to(device)

    # Load state from zip
    q.load_state_dict(ckpt["model_state_dict"])
    q_targ.load_state_dict(ckpt["model_state_dict"])
    q_targ.eval()

    out.q = q
    out.q_target = q_targ

    return out


def load_model_from_path(
    path: Path | str,
    env: OrlogEnv,
) -> DDQN:
    return load_model(
        run_name="unused",
        checkpoint="best",
        env=env,
        path=path,
    )
