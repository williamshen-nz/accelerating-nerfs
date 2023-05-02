from typing import Optional

import torch

from accelerating_nerfs.render import render_nerf_synthetic
from accelerating_nerfs.utils import NERF_SYNTHETIC_SCENES, set_random_seed

checkpoint_pattern = "../nerf-synthetic-checkpoints/results/{scene}/nerf_20000.pt"
set_random_seed(42)


@torch.no_grad()
def render_nerf_synthetic_all(
    num_downscales: int, max_num_scenes: Optional[int] = None, profile: bool = False
):
    assert num_downscales >= 0
    assert max_num_scenes is None or max_num_scenes > 0

    for scene in NERF_SYNTHETIC_SCENES[:max_num_scenes]:
        checkpoint = checkpoint_pattern.format(scene=scene)
        result_dir = f"results/{scene}"
        render_nerf_synthetic(
            scene=scene,
            checkpoint=checkpoint,
            result_dir=result_dir,
            num_downscales=num_downscales,
            profile=profile,
        )


if __name__ == "__main__":
    # When profiling, set CUDA_LAUNCH_BLOCKING=1
    render_nerf_synthetic_all(num_downscales=2, max_num_scenes=None, profile=False)
