import os
from pathlib import Path

import torch


def nerf_synthetic_config(device):
    return {
        "aabb": torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device),
        "near_plane": 0.0,
        "far_plane": 1.0e10,
        "grid_resolution": 128,
        "grid_nlvl": 1,
        "render_step_size": 5e-3,
    }


def get_project_root() -> Path:
    module_dir = Path(os.path.dirname(__file__))
    project_root = module_dir.parent
    return project_root


def get_nerf_synthetic_dataset_dir() -> Path:
    project_root = get_project_root()
    dataset_dir = project_root / "nerf_synthetic"
    return dataset_dir
