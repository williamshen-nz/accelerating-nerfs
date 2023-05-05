"""
Check number of zeros in the input and output activations.
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


def register_hooks(model: VanillaNeRF, scene: str) -> Tuple[Callable, Dict[str, Any]]:
    """Register hooks to collect metrics about the input and output activations."""
    layer_metrics = {
        "scene": scene,
        "layers": [],
        "fc_labels": [],
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
    layer_count = Counter()
    layer_to_id = {}
    layer_to_activation = {}

    def get_output_activation(nn_module):
        """Get output activation function for given module"""
        if nn_module in layer_to_activation:
            return layer_to_activation[nn_module]

        for layers, activation in [
            (model.mlp.base.hidden_layers, model.mlp.base.hidden_activation),
            ([model.mlp.sigma_layer.output_layer], F.relu),
            ([model.mlp.bottleneck_layer.output_layer], model.mlp.bottleneck_layer.output_activation),
            (model.mlp.rgb_layer.hidden_layers, model.mlp.rgb_layer.hidden_activation),
            ([model.mlp.rgb_layer.output_layer], F.sigmoid),
        ]:
            if nn_module in layers:
                layer_id = layer_to_id[nn_module]
                layer_to_activation[nn_module] = activation
                layer_metrics["output"]["activation"][layer_id] = str(activation)
                return activation
        else:
            raise ValueError(f"Activation function not found for {nn_module}")

    def hook(nn_module, input_tensor, output_tensor):
        """Count zeros in input and output activations"""
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

        if nn_module not in layer_to_id:
            layer_name = str(nn_module)
            layer_id = f"{layer_name}_{layer_count[layer_name]}"
            layer_metrics["layers"].append(layer_id)
            layer_metrics["fc_labels"].append(f"fc_{len(layer_metrics['fc_labels']) + 1}")
            layer_count[layer_name] += 1
            layer_to_id[nn_module] = layer_id
        else:
            layer_id = layer_to_id[nn_module]

        # Batch size
        assert input_tensor.shape[0] == output_tensor.shape[0], "batch size mismatch"
        layer_metrics["batch_sizes"][layer_id].append(input_tensor.shape[0])

        # Input activations
        num_non_zero_input = torch.count_nonzero(input_tensor).item()
        total_input = input_tensor.numel()
        num_zero_input = total_input - num_non_zero_input
        layer_metrics["input"]["num_zeros"][layer_id] += num_zero_input
        layer_metrics["input"]["totals"][layer_id] += total_input
        layer_metrics["input"]["sparsities"][layer_id].append(num_zero_input / total_input)

        # Output activations, find the activation function and then apply it
        activation = get_output_activation(nn_module)
        output_activations = activation(output_tensor)
        num_non_zero_output = torch.count_nonzero(output_activations).item()
        total_output = output_activations.numel()
        num_zero_output = total_output - num_non_zero_output
        layer_metrics["output"]["num_zeros"][layer_id] += num_zero_output
        layer_metrics["output"]["totals"][layer_id] += total_output
        layer_metrics["output"]["sparsities"][layer_id].append(num_zero_output / total_output)

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
    os.makedirs("sparsity", exist_ok=True)

    for scene in NERF_SYNTHETIC_SCENES:
        print(f"====== Processing {scene} ======")
        # Load model
        checkpoint_path = checkpoint_pattern.format(scene=scene)
        radiance_field, estimator = load_checkpoint(checkpoint_path)
        radiance_field.eval()
        estimator.eval()

        # Register hooks
        remove_hooks, layer_metrics = register_hooks(radiance_field, scene)

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

        # Write results for scene
        sparsity_path = f"sparsity/{scene}_sparsity.json"
        with open(sparsity_path, "w") as f:
            json.dump(layer_metrics, f, indent=4)

    print("Done, check results at sparsity/*.json")


if __name__ == "__main__":
    get_activation_sparsity()
