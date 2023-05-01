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
