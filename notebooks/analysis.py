""" Common analysis helpers to make the notebooks more readable. """
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

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


def load_nerf_sparsities(
    sparsity_path: str, sparsity_key: str = "avg_sparsity", sparsity_std_key: str = "std_sparsity"
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Load the NeRF layer sparsity from the results generated by accelerating_nerfs/nerf_activation_sparsity.py.

    Parameters
    ----------
    sparsity_path: str
        Path to the JSON file containing the layer sparsity info output by nerf_activation_sparsity.py.
    sparsity_key: str
        Key that contains the average sparsity.
    sparsity_std_key: str
        Key that contains the standard deviation of the sparsity.

    Returns
    -------
    Mapping of scene name to layer ID to sparsity dict
    """
    assert os.path.exists(sparsity_path), f"Sparsity results {sparsity_path} does not exist"

    with open(sparsity_path, "r") as f:
        sparsity_results_dict = json.load(f)

    sparsity_results = defaultdict(dict)
    for scene, results in sparsity_results_dict.items():
        if scene == "overall":
            continue

        for layer_name, layer_result in results.items():
            # We indexed by 0 in the sparsity results, but timeloop/accelergy use 1-indexing
            _, layer_num = layer_result["fc_label"].split("_")
            layer_num = int(layer_num) + 1

            sparsity_results[scene][layer_num] = {
                "sparsity": layer_result[sparsity_key],
                "sparsity_std": layer_result[sparsity_std_key],
                "layer_name": layer_name,
            }

    print(f"Loaded sparsity results for {sparsity_results.keys()}")
    return sparsity_results


def compute_layer_sparsities(sparsity_results: Dict[str, Dict[int, Dict[str, float]]]) -> Dict[int, float]:
    """
    Compute the average sparsity across all layers across all scenes.

    Parameters
    ----------
    sparsity_results: Dict[str, Dict[str, Dict[str, float]]]
        Mapping of scene name to layer ID to sparsity dict

    Returns
    -------
    Mapping of layer ID to average sparsity
    """
    layer_to_sparsities: Dict[int, List[float]] = defaultdict(list)

    for layer_results in sparsity_results.values():
        for layer_name, layer_result in layer_results.items():
            layer_to_sparsities[layer_name].append(layer_result["sparsity"])

    # Check that all layers have the same number of sparsities
    num_sparsities = [len(sparsities) for sparsities in layer_to_sparsities.values()]
    assert len(set(num_sparsities)) == 1, f"Number of sparsities should be the same across all layers: {num_sparsities}"

    # Compute mean sparsity
    layer_to_avg_sparsity = {
        layer_name: sum(sparsities) / len(sparsities) for layer_name, sparsities in layer_to_sparsities.items()
    }
    return layer_to_avg_sparsity


def add_sparsity_to_nerf_layers(
    layer_to_avg_sparsity: Dict[int, float], layer_dir: str = "workloads/nerf-sparse", dry_run: bool = False
) -> None:
    """
    Add sparsity (it's actually density) to the workload problems for each of the NeRF layers so Timeloop and Accelergy
    can use them. This updates the problem YAML files in-place.

    Parameters
    ----------
    layer_to_avg_sparsity: Dict[str, float]
        Mapping of layer ID to average sparsity.
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
        sparsity = layer_to_avg_sparsity[layer_id]
        density = 1 - sparsity

        # Add densities to problem instance for layer
        layer_config["problem"]["instance"]["densities"] = {
            "Inputs": density,
            "Weights": 1.0,
            # TODO: compute the actual output density in the NeRF activation script
            "Outputs": density,
        }

        # Write updated layer config
        if not dry_run:
            with open(layer_path, "w") as f:
                yaml.dump(layer_config, f)

        print(f"Layer {layer_id} added densities:", layer_config["problem"]["instance"]["densities"])


if __name__ == "__main__":
    print(load_nerf_layer_shapes("workloads/nerf"))
    print(50 * "-")
    s_dict = load_nerf_sparsities("../accelerating_nerfs/sparsity/2023-05-03_00-21-28_input_sparsity.json")
    print(s_dict)
    print(50 * "-")
    s_layer = compute_layer_sparsities(s_dict)
    print(s_layer)
    print(50 * "-")
    add_sparsity_to_nerf_layers(s_layer, "workloads/nerf", dry_run=True)