from __future__ import annotations

import json
import zipfile
import json
import torch
from pathlib import Path

from rl.ddqn.config import DDQNConfig


def save_model_pt(
    path: str | Path, checkpoint: str, q, params: DDQNConfig, meta: dict
) -> str:
    """
    Saves <path>/model.pt
    Returns the full .pt path.
    """
    pt_path = Path(path) / "model.pt"
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"model_state_dict": q.state_dict(), "params": params, "meta": meta},
        pt_path,
    )
    return str(pt_path)


def save_model_zip(
    path: str | Path, pt_path: str, params: DDQNConfig, meta: dict
) -> str:
    """
    Saves <path>/model.zip
    Returns the full .zip path.
    """
    zip_path = Path(path) / "model.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pt_path, arcname="model.pt")
        zf.writestr("params.json", json.dumps(params.__dict__, indent=2))
        zf.writestr("meta.json", json.dumps(meta, indent=2))

    print(f"[zip] {zip_path} saved")
    return str(zip_path)
