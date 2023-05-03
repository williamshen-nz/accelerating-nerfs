"""
Check volume rendering sparsity by monkey patching.
"""
import json
import os
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from accelerating_nerfs.render import (
    load_checkpoint,
    load_test_dataset,
    near_plane,
    render_step_size,
)
from accelerating_nerfs.render_nerf_synthetic import get_checkpoint_pattern
from accelerating_nerfs.utils import NERF_SYNTHETIC_SCENES, render_image_with_occgrid


def monkey_patch_volrend():
    from nerfacc import accumulate_along_rays, render_weight_from_density

    import accelerating_nerfs.volrend

    results = {
        # Sigma (i.e., density). It should match up roughly with the weights in theory
        "num_zero_sigmas": 0,
        "num_total_sigmas": 0,
        "sigmas_sparsity": [],
        # Weights after computing transmittance stuff
        "num_zero_weights": 0,
        "num_total_weights": 0,
        "weights_sparsity": [],
    }

    def render_weight_from_density_patched(t_starts, t_ends, sigmas, ray_indices, n_rays):
        total = sigmas.numel()
        num_zero = total - torch.count_nonzero(sigmas).item()
        if total > 0:
            results["num_zero_sigmas"] += num_zero
            results["num_total_sigmas"] += total
            results["sigmas_sparsity"].append(num_zero / total)
        return render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=n_rays)

    def accumulate_along_rays_patched(weights, values, ray_indices, n_rays):
        if values is not None and values.ndim == 2 and values.shape[1] == 3:
            # We only record weights for RGB, so we don't double count
            total = weights.numel()
            num_zero = total - torch.count_nonzero(weights).item()
            if total > 0:
                results["num_zero_weights"] += num_zero
                results["num_total_weights"] += total
                results["weights_sparsity"].append(num_zero / total)
        return accumulate_along_rays(weights, values, ray_indices, n_rays)

    # Monkey patch
    accelerating_nerfs.volrend.render_weight_from_density = render_weight_from_density_patched
    accelerating_nerfs.volrend.accumulate_along_rays = accumulate_along_rays_patched
    return results


@torch.no_grad()
def get_volrend_sparsity():
    """Get sparsity of weights when volumetric rendering."""
    checkpoint_pattern = get_checkpoint_pattern()
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("sparsity", exist_ok=True)
    sparsity_path = f"sparsity/{now}_volrend_sparsity.json"
    scene_sparsity_results = {}

    for scene in NERF_SYNTHETIC_SCENES:
        print(f"====== Processing {scene} ======")
        # Load model
        checkpoint_path = checkpoint_pattern.format(scene=scene)
        radiance_field, estimator = load_checkpoint(checkpoint_path)
        radiance_field.eval()
        estimator.eval()

        # Monkey patch
        results = monkey_patch_volrend()

        # Load test dataset, use num_downscales = 1 to match renders
        test_dataset = load_test_dataset(scene, num_downscales=1)

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

        # Massage results
        sparsity_results = {
            # Sigmas
            "num_zero_sigmas": results["num_zero_sigmas"],
            "num_total_sigmas": results["num_total_sigmas"],
            "avg_sigmas_sparsity": np.mean(results["sigmas_sparsity"]),
            "std_sigmas_sparsity": np.std(results["sigmas_sparsity"]),
            # Weights
            "num_zero_weights": results["num_zero_weights"],
            "num_total_weights": results["num_total_weights"],
            "avg_weights_sparsity": np.mean(results["weights_sparsity"]),
            "std_weights_sparsity": np.std(results["weights_sparsity"]),
        }
        scene_sparsity_results[scene] = sparsity_results

        # Write partial results
        with open(sparsity_path, "w") as f:
            json.dump(scene_sparsity_results, f, indent=4)

    print(f"Done, check results at {sparsity_path}")


if __name__ == "__main__":
    get_volrend_sparsity()
