import os
from typing import Optional

import torch

from accelerating_nerfs.config import get_project_root
from accelerating_nerfs.render import render_nerf_synthetic
from accelerating_nerfs.utils import NERF_SYNTHETIC_SCENES, set_random_seed


def get_checkpoint_pattern() -> str:
    project_root = get_project_root()
    checkpoint_dir = project_root / "nerf-synthetic-checkpoints/results"
    checkpoint_pattern = f"{checkpoint_dir}/{{scene}}/nerf_50000.pt"
    return checkpoint_pattern


def check_cuda_launch_blocking():
    cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", None)
    if cuda_launch_blocking != "1":
        raise RuntimeError("CUDA_LAUNCH_BLOCKING should be set to 1 for accurate profiling results")


@torch.no_grad()
def render_nerf_synthetic_all(
    num_downscales: int, max_num_scenes: Optional[int], profile: bool, quantize: bool, random_seed: int = 42
):
    # Validate arguments
    assert num_downscales >= 0, "num_downscales must >= 0"
    assert max_num_scenes is None or max_num_scenes > 0, "max_num_scenes must be None or > 0"
    if profile:
        check_cuda_launch_blocking()
        print("Profiling enabled and CUDA_LAUNCH_BLOCKING=1, expect a slowdown.")

    set_random_seed(random_seed)
    checkpoint_pattern = get_checkpoint_pattern()
    project_root = get_project_root()

    for scene in NERF_SYNTHETIC_SCENES[:max_num_scenes]:
        checkpoint = checkpoint_pattern.format(scene=scene)
        result_dir = project_root / "results" / scene
        render_nerf_synthetic(
            scene=scene,
            checkpoint=checkpoint,
            result_dir=result_dir,
            num_downscales=num_downscales,
            quantize=quantize,
            profile=profile,
        )


if __name__ == "__main__":
    # When profiling, you need to set CUDA_LAUNCH_BLOCKING=1 to get accurate numbers
    render_nerf_synthetic_all(num_downscales=1, max_num_scenes=None, profile=False, quantize=False)
