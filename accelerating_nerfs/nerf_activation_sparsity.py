"""
Check number of zeros in the activations.
"""
from collections import defaultdict

import torch
from tqdm import tqdm

from accelerating_nerfs.models import VanillaNeRF
from accelerating_nerfs.render import (
    load_checkpoint,
    load_test_dataset,
    near_plane,
    render_step_size,
)
from accelerating_nerfs.utils import render_image_with_occgrid


def register_hooks(model: VanillaNeRF):
    num_zeros = defaultdict(int)
    totals = defaultdict(int)

    def hook(nn_module, _, output_tensor):
        if not isinstance(output_tensor, torch.Tensor):
            return
        num_non_zero = torch.count_nonzero(output_tensor).item()
        total = output_tensor.numel()
        num_zero = total - num_non_zero
        num_zeros[nn_module] += num_zero
        totals[nn_module] += total

    for name, module in model.named_modules():
        module.register_forward_hook(hook)

    return num_zeros, totals


def entrypoint(scene: str = "lego"):
    checkpoint_path = "../nerf-synthetic-checkpoints/results/lego/nerf_20000.pt"
    radiance_field, estimator = load_checkpoint(checkpoint_path)
    num_non_zeros, totals = register_hooks(radiance_field)

    # Load test dataset
    test_dataset = load_test_dataset(scene, num_downscales=2)

    # Render frames
    for idx in tqdm(range(len(test_dataset)), f"Rendering {scene} test images"):
        data = test_dataset[idx]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]

        # Render
        _ = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            # test options
            test_chunk_size=4096,
        )

    print("Done")


if __name__ == "__main__":
    entrypoint()
