"""
Check number of zeros in the input activations.
"""
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


def register_hooks(model: VanillaNeRF) -> Tuple[Callable, Dict[str, Any]]:
    """Register hooks to collect metrics about the input and output activations."""
    layer_metrics = {
        "layers": [],
        "batch_sizes": defaultdict(list),
        "input": {
            "num_zeros": defaultdict(int),
            "totals": defaultdict(int),
            "sparsities": defaultdict(list),  # within batch sparsity
        },
        "output": {
            "num_zeros": defaultdict(int),
            "totals": defaultdict(int),
            "sparsities": defaultdict(list),  # within batch sparsity
            "activation": {},  # activation function
        },
    }

    def get_output_activation(nn_module):
        """Get output activation function for given module"""
        if nn_module in layer_metrics["output"]["activation"]:
            return layer_metrics["output"]["activation"][nn_module]

        for layers, activation in [
            (model.mlp.base.hidden_layers, model.mlp.base.hidden_activation),
            ([model.mlp.sigma_layer.output_layer], model.mlp.sigma_layer.output_activation),
            ([model.mlp.bottleneck_layer.output_layer], model.mlp.bottleneck_layer.output_activation),
            (model.mlp.rgb_layer.hidden_layers, model.mlp.rgb_layer.hidden_activation),
            ([model.mlp.rgb_layer.output_layer], model.mlp.rgb_layer.output_activation),
        ]:
            if nn_module in layers:
                layer_metrics["output"]["activation"][nn_module] = activation
                return activation
        else:
            raise ValueError(f"Activation function not found for {nn_module}")

    def hook(nn_module, input_tensor, output_tensor):
        """ Count zeros in input and output activations"""
        # Only consider linear layers
        if not isinstance(nn_module, torch.nn.Linear):
            return

        # Sometimes the input and/or output tensor is a tuple, weird stuff
        if isinstance(input_tensor, tuple):
            assert len(input_tensor) == 1
            input_tensor = input_tensor[0]
        if isinstance(output_tensor, tuple):
            assert len(output_tensor) == 1
            output_tensor = output_tensor[0]

        # The batch size is sometimes 0, so skip that
        if input_tensor.numel() == 0:
            assert output_tensor.numel() == 0
            return

        if nn_module not in layer_metrics["layers"]:
            layer_metrics["layers"].append(nn_module)

        # Batch size
        assert input_tensor.shape[0] == output_tensor.shape[0], "batch size mismatch"
        layer_metrics["batch_sizes"][nn_module].append(input_tensor.shape[0])

        # Input activations
        num_non_zero_input = torch.count_nonzero(input_tensor).item()
        total_input = input_tensor.numel()
        num_zero_input = total_input - num_non_zero_input
        layer_metrics["input"]["num_zeros"][nn_module] += num_zero_input
        layer_metrics["input"]["totals"][nn_module] += total_input
        layer_metrics["input"]["sparsities"][nn_module].append(num_zero_input / total_input)

        # Output activations, find the activation function and then apply it
        activation = get_output_activation(nn_module)
        output_activations = activation(output_tensor)
        num_non_zero_output = torch.count_nonzero(output_activations).item()
        total_output = output_activations.numel()
        num_zero_output = total_output - num_non_zero_output
        layer_metrics["output"]["num_zeros"][nn_module] += num_zero_output
        layer_metrics["output"]["totals"][nn_module] += total_output
        layer_metrics["output"]["sparsities"][nn_module].append(num_zero_output / total_output)

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook))

    def remove_hooks():
        for h in hooks:
            h.remove()

    return remove_hooks, layer_metrics


@torch.no_grad()
def get_activation_sparsity():
    """Get input and output activation sparsity over NeRF synthetic datasets."""
    checkpoint_pattern = get_checkpoint_pattern()
    scene_sparsity_results = {}
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("sparsity", exist_ok=True)
    sparsity_path = f"sparsity/{now}_sparsity.json"

    for scene in NERF_SYNTHETIC_SCENES[:2]:
        print(f"====== Processing {scene} ======")
        # Load model
        checkpoint_path = checkpoint_pattern.format(scene=scene)
        radiance_field, estimator = load_checkpoint(checkpoint_path)
        radiance_field.eval()
        estimator.eval()

        # Register hooks
        remove_hooks, layer_metrics = register_hooks(radiance_field)

        # Load test dataset, use num_downscales = 2 otherwise run OOM.
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

        # Add layer metrics to scene results
        scene_sparsity_results[scene] = layer_metrics

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
                "avg_batch_size": np.mean(batch_sizes[layer]),
                "std_batch_size": np.std(batch_sizes[layer]),
                "fc_label": f"fc_{idx:02d}",
            }
            all_layer_num_zeros[layer_id] += num_zero
            all_layer_totals[layer_id] += total
            all_layer_sparsities[layer_id].extend(layer_sparsities)
            all_batch_sizes[layer_id].extend(batch_sizes[layer])
            layer_count[layer_name] += 1
        scene_sparsity_results[scene] = sparsity_results

        # Write partial results
        with open(sparsity_path, "w") as f:
            json.dump(scene_sparsity_results, f, indent=4)

    # Collate overall results
    # overall_results = {
    #     layer_name: {
    #         "sparsity": all_layer_num_zeros[layer_name] / all_layer_totals[layer_name],  # overall sparsity
    #         "avg_sparsity": np.mean(all_layer_sparsities[layer_name]),
    #         "std_sparsity": np.std(all_layer_sparsities[layer_name]),
    #         "num_zero": all_layer_num_zeros[layer_name],
    #         "total": all_layer_totals[layer_name],
    #         "avg_batch_size": np.mean(all_batch_sizes[layer_name]),
    #         "std_batch_size": np.std(all_batch_sizes[layer_name]),
    #         "fc_label": f"fc_{idx:02d}",
    #     }
    #     for idx, layer_name in enumerate(all_layer_num_zeros.keys())
    # }
    # scene_sparsity_results["overall"] = overall_results
    # with open(sparsity_path, "w") as f:
    #     json.dump(scene_sparsity_results, f, indent=4)
    print(f"Done, check results at {sparsity_path}")


if __name__ == "__main__":
    get_activation_sparsity()
