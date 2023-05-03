"""
Check number of zeros in the activations.
"""
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from accelerating_nerfs.models import VanillaNeRF
from accelerating_nerfs.render import (
    load_checkpoint,
    load_test_dataset,
    near_plane,
    render_step_size,
)
from accelerating_nerfs.render_nerf_synthetic import get_checkpoint_pattern
from accelerating_nerfs.utils import NERF_SYNTHETIC_SCENES, render_image_with_occgrid


def register_hooks(model: VanillaNeRF):
    num_zeros = defaultdict(int)
    totals = defaultdict(int)
    sparsities = defaultdict(list)

    def hook(nn_module, input_tensor, output_tensor):
        # Only consider linear layers
        if not isinstance(nn_module, torch.nn.Linear):
            return
        # Sometimes the input tensor is a tuple, weird stuff
        if isinstance(input_tensor, tuple):
            assert len(input_tensor) == 1
            input_tensor = input_tensor[0]
        # The batch size is sometimes 0, so skip that
        if input_tensor.numel() == 0:
            return

        num_non_zero = torch.count_nonzero(input_tensor).item()
        total = input_tensor.numel()
        num_zero = total - num_non_zero
        num_zeros[nn_module] += num_zero
        totals[nn_module] += total
        # Within batch sparsity
        sparsities[nn_module].append(num_zero / total)

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook))

    def remove_hooks():
        for h in hooks:
            h.remove()

    return remove_hooks, num_zeros, totals, sparsities


@torch.no_grad()
def get_activation_sparsity():
    """Get activation sparsity over NeRF synthetic datasets."""
    checkpoint_pattern = get_checkpoint_pattern()
    scene_sparsity_results = {}
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("sparsity", exist_ok=True)
    sparsity_path = f"sparsity/{now}_sparsity.json"
    all_layer_sparsities = defaultdict(list)

    for scene in NERF_SYNTHETIC_SCENES:
        print(f"====== Processing {scene} ======")
        # Load model
        checkpoint_path = checkpoint_pattern.format(scene=scene)
        radiance_field, estimator = load_checkpoint(checkpoint_path)
        radiance_field.eval()
        estimator.eval()

        # Register hooks
        remove_hooks, num_zeros, totals, sparsities = register_hooks(radiance_field)

        # Load test dataset, use num_downscales = 2 otherwise run OOM
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

        # Remove hooks
        remove_hooks()

        # Massage results
        sparsity_results = {}
        layer_count = Counter()
        for idx, (layer, num_zero) in enumerate(num_zeros.items()):
            total = totals[layer]
            layer_sparsities = sparsities[layer]  # within layer batch sparsities

            layer_name = str(layer)
            layer_id = f"{layer_name}_{layer_count[layer_name]}"
            sparsity_results[layer_id] = {
                "sparsity": num_zero / total,  # overall sparsity
                "avg_sparsity": np.mean(layer_sparsities),
                "std_sparsity": np.std(layer_sparsities),
                "num_zero": num_zero,
                "total": total,
                "fc_label": f"fc_{idx:02d}",
            }
            layer_count[layer_name] += 1
        scene_sparsity_results[scene] = sparsity_results

        # Write partial results
        with open(sparsity_path, "w") as f:
            json.dump(scene_sparsity_results, f, indent=4)

    # Add overall batch sparsity over all scenes
    all_layer_sparsities = np.array(all_layer_sparsities)
    scene_sparsity_results["overall"] = {
        "avg_sparsity": np.mean(all_layer_sparsities),
        "std_sparsity": np.std(all_layer_sparsities),
    }
    with open(sparsity_path, "w") as f:
        json.dump(scene_sparsity_results, f, indent=4)
    print(f"Done, check results at {sparsity_path}")


if __name__ == "__main__":
    get_activation_sparsity()
