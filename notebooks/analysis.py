""" Common analysis helpers to make the notebooks more readable. """
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import yaml
from notebook_utils import natural_sort


def convert_nerf_to_timeloop(model, batch_size: int, top_dir: str = "workloads", sub_dir: str = "nerf") -> str:
    """
    Convert a NeRF model to Timeloop problems using pytorch2timeloop.

    Parameters
    ----------
    model: VanillaNeRF
        NeRF model to convert.
    batch_size: int
        Batch size to use for the conversion.
    top_dir: str
        Top directory to store the converted Timeloop problems.
    sub_dir: str
        Subdirectory to store the converted Timeloop problems.

    Returns
    -------
    str
        Mapped layer directory
    """
    from accelerating_nerfs.models import VanillaNeRF

    try:
        import pytorch2timeloop
    except ImportError:
        raise ImportError("Could not import pytorch2timeloop. Please make sure you are using the Docker environment.")
    assert isinstance(model, VanillaNeRF), f"model must be a VanillaNeRF, got {type(model)}"

    layer_dir = Path(top_dir) / Path(sub_dir)
    # clear previous conversion results
    if layer_dir.exists():
        shutil.rmtree(layer_dir, ignore_errors=True)

    # convert the model to timeloop files
    pytorch2timeloop.convert_model(
        model=model,
        input_size=(1, 3),
        batch_size=batch_size,
        model_name=sub_dir,
        save_dir=top_dir,
        convert_fc=True,  # must convert FC as NeRF is all FCs
        exception_module_names=[],
    )

    # Check layer directory is not empty
    assert len(os.listdir(layer_dir)) > 0, f"Layer directory {layer_dir} is empty"
    print(f"Converted VanillaNeRF model to Timeloop problems in {layer_dir}")
    return str(layer_dir)


def load_nerf_layer_shapes(layer_dir: str = "workloads/nerf") -> Dict[int, Dict[str, int]]:
    """
    Load layer shape info from the result of the pytorch2timeloop converter.

    Parameters
    ----------
    layer_dir: str
        Path to the directory containing the layer shape info output by pytorch2timeloop.

    Returns
    -------
    Mapping of layer ID to shape dict
    """
    # These are the keys that should be 1, since we're doing a 1x1 convolution in a FC layer
    keys_should_be_1 = ["Hdilation", "Hstride", "P", "Q", "R", "S", "Wdilation", "Wstride"]
    layer_shapes = {}

    for layer_path in natural_sort(os.listdir(layer_dir)):
        layer_path = os.path.join(layer_dir, layer_path)
        layer_id = int(layer_path.split("layer")[1].split(".")[0])

        with open(layer_path, "r") as f:
            layer_config = yaml.safe_load(f)

        instance = layer_config["problem"]["instance"]
        for key in keys_should_be_1:
            assert instance[key] == 1, f"{key} != 1"
            del instance[key]

        assert layer_id not in layer_shapes
        layer_shapes[layer_id] = {"shape": instance}

    assert layer_shapes, "layer_shapes should not be empty"
    return layer_shapes


def load_nerf_sparsities(sparsity_dir: str) -> Dict[str, Dict]:
    """
    Load the NeRF layer sparsity from the results generated by accelerating_nerfs/nerf_activation_sparsity.py.

    Parameters
    ----------
    sparsity_dir: str
        Path to the directory containing the sparsity results.

    Returns
    -------
    Mapping of scene name to sparsity results
    """
    assert os.path.isdir(sparsity_dir), f"Sparsity directory {sparsity_dir} does not exist"

    scene_to_sparsity_results = {}
    for sparsity_results in sorted(os.listdir(sparsity_dir)):
        if not sparsity_results.endswith(".json"):
            continue
        with open(os.path.join(sparsity_dir, sparsity_results), "r") as f:
            sparsity_results_dict = json.load(f)
            scene_to_sparsity_results[sparsity_results_dict["scene"]] = sparsity_results_dict

    print(f"Loaded sparsity results for {scene_to_sparsity_results.keys()}")
    return scene_to_sparsity_results


def compute_layer_sparsities(
    sparsity_results, include_sparsity_list: bool = False
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Compute the average sparsity across all layers across all scenes.

    Parameters
    ----------
    sparsity_results
    include_sparsity_list: bool
        Whether to include the list of sparsities for each layer in the output.

    Returns
    -------
    Mapping of layer ID to average sparsity for input and output activations.
    """
    layer_to_all_sparsities: Dict[int, Dict[str, List[float]]] = defaultdict(dict)

    for scene, results in sparsity_results.items():
        for layer, fc_label in zip(results["layers"], results["fc_labels"]):
            layer_id = int(fc_label.split("_")[1])
            if "input_sparsity" not in layer_to_all_sparsities[layer_id]:
                layer_to_all_sparsities[layer_id]["input_sparsity"] = []
            layer_to_all_sparsities[layer_id]["input_sparsity"].extend(results["input"]["sparsities"][layer])

            if "output_sparsity" not in layer_to_all_sparsities[layer_id]:
                layer_to_all_sparsities[layer_id]["output_sparsity"] = []
            layer_to_all_sparsities[layer_id]["output_sparsity"].extend(results["output"]["sparsities"][layer])

    # Check that all layers have the same number of sparsities
    num_sparsities = [
        (len(sparsities["input_sparsity"]), len(sparsities["output_sparsity"]))
        for sparsities in layer_to_all_sparsities.values()
    ]
    assert all(tup[0] == tup[1] for tup in num_sparsities)

    # Compute mean and std sparsity
    layer_to_sparsity: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for layer_id, sparsities in layer_to_all_sparsities.items():
        layer_to_sparsity[layer_id] = {}
        for key, sparsities in sparsities.items():
            layer_to_sparsity[layer_id][key] = {
                "mean": np.mean(sparsities).item(),
                "std": np.std(sparsities).item(),
                "num": len(sparsities),
            }
            if include_sparsity_list:
                layer_to_sparsity[layer_id][key]["sparsities"] = sparsities
    return layer_to_sparsity


def compute_overall_sparsity(sparsity_results, sparsity_threshold: float = 0.1) -> Dict[str, Dict[str, float]]:
    """
    Compute the overall sparsity across all layers across all scenes.

    Parameters
    ----------
    sparsity_results
    sparsity_threshold: float
        Threshold below which a sparsity is considered 0.

    Returns
    -------
    Mapping of sparsity type to average sparsity.
    """
    layer_to_sparsity = compute_layer_sparsities(sparsity_results, include_sparsity_list=True)
    input_sparsities = []
    output_sparsities = []

    # Delete sparsities below threshold
    for layer_id, sparsities in layer_to_sparsity.items():
        if sparsities["input_sparsity"]["mean"] < sparsity_threshold:
            print(f"Deleting input sparsity for layer {layer_id} with mean {sparsities['input_sparsity']['mean']}")
            del sparsities["input_sparsity"]
        else:
            input_sparsities.extend(sparsities["input_sparsity"]["sparsities"])

        if sparsities["output_sparsity"]["mean"] < sparsity_threshold:
            print(f"Deleting output sparsity for layer {layer_id} with mean {sparsities['output_sparsity']['mean']}")
            del sparsities["output_sparsity"]
        else:
            output_sparsities.extend(sparsities["output_sparsity"]["sparsities"])

    overall_sparsity = {
        "input_sparsity": {
            "mean": np.mean(input_sparsities).item(),
            "std": np.std(input_sparsities).item(),
            "num": len(input_sparsities),
        },
        "output_sparsity": {
            "mean": np.mean(output_sparsities).item(),
            "std": np.std(output_sparsities).item(),
            "num": len(output_sparsities),
        },
    }
    return overall_sparsity


def add_sparsity_to_nerf_layers(
    layer_to_sparsity: Dict[int, Dict[str, Dict[str, float]]],
    layer_dir: str = "workloads/nerf-sparse",
    dry_run: bool = False,
) -> None:
    """
    Add sparsity (it's actually density) to the workload problems for each of the NeRF layers so Timeloop and Accelergy
    can use them. This updates the problem YAML files in-place.

    Parameters
    ----------
    layer_to_sparsity: Dict[int, Dict[str, Dict[str, float]]]
        Mapping of layer ID to input and output sparsity.
    layer_dir: str
        Path to the directory containing the layer shape info output by pytorch2timeloop.
    dry_run: bool
        If True, don't actually write the updated YAML files.

    Returns
    -------
    None
    """
    # Check layer_dir is not empty
    assert os.listdir(layer_dir), f"Layer directory {layer_dir} is empty"
    if dry_run:
        print("=== add_sparsity_to_nerf_layers dry-run: not writing updated YAML files ===")

    for layer_path in natural_sort(os.listdir(layer_dir)):
        layer_path = os.path.join(layer_dir, layer_path)
        layer_id = int(layer_path.split("layer")[1].split(".")[0])

        with open(layer_path, "r") as f:
            layer_config = yaml.safe_load(f)

        # Get density = 1 - sparsity for layer
        input_sparsity = layer_to_sparsity[layer_id]["input_sparsity"]["mean"]
        input_density = 1 - input_sparsity
        output_sparsity = layer_to_sparsity[layer_id]["output_sparsity"]["mean"]
        output_density = 1 - output_sparsity

        # Add densities to problem instance for layer
        layer_config["problem"]["instance"]["densities"] = {
            "Inputs": input_density,
            "Weights": 1.0,  # we showed that weights are fully dense
            "Outputs": output_density,
        }

        # Write updated layer config
        if not dry_run:
            with open(layer_path, "w") as f:
                yaml.dump(layer_config, f)

        print(f"Layer {layer_id} added densities:", layer_config["problem"]["instance"]["densities"])


if __name__ == "__main__":
    print(load_nerf_layer_shapes("workloads/nerf"))
    print(50 * "-")
    s_dict = load_nerf_sparsities("../accelerating_nerfs/sparsity")
    print(s_dict)
    print(50 * "-")
    s_layer = compute_layer_sparsities(s_dict)
    print(s_layer)
    print(50 * "-")
    print(compute_overall_sparsity(s_dict))
    print(50 * "-")
    add_sparsity_to_nerf_layers(s_layer, "workloads/nerf", dry_run=True)
