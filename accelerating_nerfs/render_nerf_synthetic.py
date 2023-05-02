import os
from pathlib import Path
from typing import Optional

import torch

from accelerating_nerfs.render import render_nerf_synthetic
from accelerating_nerfs.utils import NERF_SYNTHETIC_SCENES, set_random_seed


def get_checkpoint_pattern() -> str:
    module_dir = Path(os.path.dirname(__file__))
    project_root = module_dir.parent
    checkpoint_dir = project_root / "nerf-synthetic-checkpoints/results"
    checkpoint_pattern = f"{checkpoint_dir}/{{scene}}/nerf_20000.pt"
    return checkpoint_pattern


@torch.no_grad()
def render_nerf_synthetic_all(num_downscales: int, max_num_scenes: Optional[int], profile: bool, random_seed: int = 42):
    assert num_downscales >= 0, "num_downscales must >= 0"
    assert max_num_scenes is None or max_num_scenes > 0, "max_num_scenes must be None or > 0"
    set_random_seed(random_seed)
    checkpoint_pattern = get_checkpoint_pattern()

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
    # When profiling, you should set CUDA_LAUNCH_BLOCKING=1
    render_nerf_synthetic_all(num_downscales=1, max_num_scenes=None, profile=True)
