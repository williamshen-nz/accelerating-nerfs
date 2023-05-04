"""
Modified from Lab 1.
"""

import copy
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

import pytorch2timeloop
import torch
import torch.nn as nn
import yaml
from torchprofile import profile_macs
from torchvision.models.resnet import BasicBlock, Bottleneck
from tqdm import tqdm


def count_activation_size(net, input_size=(1, 3, 224, 224), require_backward=False, activation_bits=32):
    act_byte = activation_bits / 8
    model = copy.deepcopy(net)

    # noinspection PyArgumentList
    def count_convNd(m, x, y):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte // m.groups])  # bytes

    # noinspection PyArgumentList
    def count_linear(m, x, y):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

    # noinspection PyArgumentList
    def count_bn(m, x, _):
        # count activation size required by backward
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    # noinspection PyArgumentList
    def count_relu(m, x, _):
        # count activation size required by backward
        if require_backward:
            m.grad_activations = torch.Tensor([x[0].numel() / 8])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    # noinspection PyArgumentList
    def count_smooth_act(m, x, _):
        # count activation size required by backward
        if require_backward:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
        else:
            m.grad_activations = torch.Tensor([0])
        # temporary memory footprint required by inference
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

    def add_hooks(m_):
        if len(list(m_.children())) > 0:
            return

        m_.register_buffer("grad_activations", torch.zeros(1))
        m_.register_buffer("tmp_activations", torch.zeros(1))

        if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            fn = count_convNd
        elif type(m_) in [nn.Linear]:
            fn = count_linear
        elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]:
            fn = count_bn
        elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
            fn = count_relu
        elif type(m_) in [nn.Sigmoid, nn.Tanh]:
            fn = count_smooth_act
        else:
            fn = None

        if fn is not None:
            _handler = m_.register_forward_hook(fn)

    model.eval()
    model.apply(add_hooks)

    x = torch.zeros(input_size).to(model.parameters().__next__().device)
    with torch.no_grad():
        model(x)

    memory_info_dict = {
        "peak_activation_size": torch.zeros(1),
        "residual_size": torch.zeros(1),
    }

    for m in model.modules():
        if len(list(m.children())) == 0:

            def new_forward(_module):
                def lambda_forward(_x):
                    current_act_size = _module.tmp_activations + memory_info_dict["residual_size"]
                    memory_info_dict["peak_activation_size"] = max(
                        current_act_size, memory_info_dict["peak_activation_size"]
                    )
                    return _module.old_forward(_x)

                return lambda_forward

            m.old_forward = m.forward
            m.forward = new_forward(m)

        if type(m) in [BasicBlock, Bottleneck]:

            def new_forward(_module):
                def lambda_forward(_x):
                    memory_info_dict["residual_size"] = _x.numel() * act_byte
                    result = _module.old_forward(_x)
                    memory_info_dict["residual_size"] = 0
                    return result

                return lambda_forward

            m.old_forward = m.forward
            m.forward = new_forward(m)

    with torch.no_grad():
        model(x)

    return memory_info_dict["peak_activation_size"].item()


def profile_memory_cost(net, input_size=(1, 3, 224, 224), require_backward=False, activation_bits=32, batch_size=1):
    activation_size = count_activation_size(net, input_size, require_backward, activation_bits)
    memory_cost = activation_size * batch_size
    return memory_cost


class Profiler:
    # Metrics per layer to track
    layer_metric_keys: ClassVar[List[str]] = ["energy", "area", "cycle", "gflops", "utilization", "edp"]

    def __init__(
        self,
        sub_dir: str,
        top_dir: str,
        timeloop_dir: str,
        arch_name: str,
        model: torch.nn.Module,
        input_size: Tuple[int, ...],
        batch_size: int,
        convert_fc: bool,
        exception_module_names: Optional[List[str]] = None,
        profiled_lib_dir_pattern: str = "./{arch_name}_profiled_lib.json",
    ):
        self.base_dir = Path(os.getcwd())
        self.sub_dir = sub_dir
        self.top_dir = top_dir
        self.model = model
        self.timeloop_dir = timeloop_dir
        self.arch_name = arch_name
        self.input_size = input_size
        self.batch_size = batch_size
        self.convert_fc = convert_fc
        self.exception_module_names = exception_module_names

        profiled_lib_dir = profiled_lib_dir_pattern.format(arch_name=self.arch_name)
        self.profiled_lib_dir = profiled_lib_dir
        self.profiled_lib = {}
        self.load_profiled_lib()

    def load_profiled_lib(self):
        """Load existing profiled results in-place."""
        if os.path.exists(self.profiled_lib_dir):
            with open(self.profiled_lib_dir, "r") as fid:
                self.profiled_lib = json.load(fid)
            print(f"Loaded profiled lib from {self.profiled_lib_dir}")

    def write_profiled_lib(self):
        """Write profiled results to a json file."""
        with open(self.profiled_lib_dir, "w") as fid:
            json.dump(self.profiled_lib, fid, sort_keys=True, indent=4)
        print(f"Saved profiled lib to {self.profiled_lib_dir}")

    def convert_model(self) -> Path:
        """Convert model to timeloop files and return mapped layer directory."""
        # clear previous conversion results
        layer_dir = self.base_dir / self.top_dir / self.sub_dir
        if layer_dir.exists():
            shutil.rmtree(layer_dir, ignore_errors=True)
        # convert the model to timeloop files
        pytorch2timeloop.convert_model(
            self.model,
            self.input_size,
            self.batch_size,
            self.sub_dir,
            self.top_dir,
            self.convert_fc,
            self.exception_module_names,
        )
        return layer_dir

    def get_timeloop_cmd(self, layer_id: int, layer_info: dict) -> Tuple[str, str]:
        """Get timeloop working directory and command."""
        cwd = f"{self.base_dir / self.timeloop_dir / self.sub_dir / f'layer{layer_id}'}"
        if "M" in layer_info[layer_id]["layer_dict"]["problem"]["instance"]:
            constraint_pth = self.base_dir / self.timeloop_dir / "constraints/*.yaml"
        else:
            # depthwise
            constraint_pth = self.base_dir / self.timeloop_dir / "constraints_dw/*.yaml"

        # Check if sparse_opt is enabled
        sparse_opt_dir = self.base_dir / self.timeloop_dir / "sparse_opt"
        include_sparse_opt = os.path.exists(sparse_opt_dir)
        if include_sparse_opt:
            print(f"Sparse optimization is enabled for {self.arch_name} and layer {layer_id}")

        arch_fname = f"{self.arch_name}.yaml"
        timeloopcmd = (
            f"timeloop-mapper "
            f"{self.base_dir / self.timeloop_dir / 'arch' / arch_fname} "
            f"{self.base_dir / self.timeloop_dir / 'arch/components/*.yaml'} "
            f"{self.base_dir / self.timeloop_dir / 'mapper/mapper.yaml'} "
            f"{constraint_pth} "
            f"{sparse_opt_dir / '*.yaml'} " if include_sparse_opt else " "  # Important: keep the space at the end
            f"{self.base_dir / self.top_dir / self.sub_dir / self.sub_dir}_layer{layer_id}.yaml > /dev/null 2>&1"
        )
        print(timeloopcmd)
        return cwd, timeloopcmd

    def run_timeloop(self, layer_info: dict):
        """Run Timeloop and Accelergy in the required layers."""
        # need to run timeloop on layers that are not already in the profiled_lib
        for layer_id in layer_info.keys():
            os.makedirs(self.base_dir / self.timeloop_dir / self.sub_dir / f"layer{layer_id}", exist_ok=True)
        cmds_list = [
            self.get_timeloop_cmd(layer_id, layer_info)
            for layer_id in layer_info
            if "energy" not in layer_info[layer_id]
        ]
        for cwd, cmd in tqdm(cmds_list, desc="running timeloop to get energy and latency..."):
            os.chdir(cwd)
            os.system(cmd)
        os.chdir(self.base_dir)

    def process_results(self, layer_info: dict):
        """Process the timeloop results and update layer_info in-place."""
        # process the results into layer_info
        for layer_id in layer_info:
            cur_layer_info = layer_info[layer_id]
            if "energy" in cur_layer_info:
                # the layer is in the profiler lib
                continue

            # check if results for the layer exists, skip if not (weird stuff going on)
            stats_fname = (
                self.base_dir / self.timeloop_dir / self.sub_dir / f"layer{layer_id}" / "timeloop-mapper.stats.txt"
            )
            if not stats_fname.exists():
                print(f"CRITICAL WARNING: {stats_fname} does not exist, skipping...")
                continue

            if any(key in cur_layer_info for key in self.layer_metric_keys):
                raise RuntimeError("check with willshen@. layer info already has some metrics")

            # process results
            with open(stats_fname, "r") as fid:
                lines = fid.read().split("\n")[-50:]
                for line in lines:
                    line = line.lower()
                    for key in self.layer_metric_keys:
                        if not line.startswith(key):
                            continue
                        metric = line.split(": ")[1].split(" ")[0]
                        cur_layer_info[key] = eval(metric)

            # check all metrics are there
            for key in self.layer_metric_keys:
                if key not in cur_layer_info:
                    raise RuntimeError(f"missing metric {key} for layer {layer_id}")

    def populate_profiled_lib(self, layer_info: dict):
        """Populate the profiled lib with the layer info."""
        keys_to_include = (
            ["layer_dict"]
            + self.layer_metric_keys
            + ["mapper_timeout", "mapper_algo", "mapper_victory_condition", "mapper_max_permutations"]
        )
        for layer_id in layer_info:
            layer_name = layer_info[layer_id]["name"]
            if layer_name not in self.profiled_lib:
                info = {key: layer_info[layer_id][key] for key in keys_to_include}
                self.profiled_lib[layer_name] = info

    def profile(self) -> Tuple[dict, dict, dict]:
        """Profile the model."""
        # Run the pytorch2timeloop converter
        layer_dir = self.convert_model()

        # check duplicated layer info
        layer_info = {}
        path, dirs, files = next(os.walk(layer_dir))
        file_count = len(files)
        for idx in range(file_count):
            file = layer_dir / f"{self.sub_dir}_layer{idx + 1}.yaml"
            with open(file, "r") as fid:
                layer_dict = yaml.safe_load(fid)
                for layer_id, info in layer_info.items():
                    if info["layer_dict"] == layer_dict:
                        layer_info[layer_id]["num"] += 1
                        break
                else:
                    layer_info[idx + 1] = {"layer_dict": layer_dict, "num": 1, "name": str(file).replace(".yaml", "")}

        # check the mapper info
        with open(self.base_dir / self.timeloop_dir / "mapper/mapper.yaml", "r") as fid:
            mapper_dict = yaml.safe_load(fid)
            for layer_id, info in layer_info.items():
                layer_info[layer_id]["mapper_timeout"] = mapper_dict["mapper"]["timeout"]
                layer_info[layer_id]["mapper_algo"] = mapper_dict["mapper"]["algorithm"]
                layer_info[layer_id]["mapper_victory_condition"] = mapper_dict["mapper"]["victory-condition"]
                layer_info[layer_id]["mapper_max_permutations"] = mapper_dict["mapper"]["max-permutations-per-if-visit"]

        # check whether some layers have been profiled before and exist in the profiled_lib.
        # sometimes the layer_dict are the same but name will be different, in that case make their name the same
        for layer_id, info in layer_info.items():
            for profiled_name, profiled_info in self.profiled_lib.items():
                if (
                    info["layer_dict"] == profiled_info["layer_dict"]
                    and info["mapper_timeout"] == profiled_info["mapper_timeout"]
                    and info["mapper_algo"] == profiled_info["mapper_algo"]
                    and info["mapper_victory_condition"] == profiled_info["mapper_victory_condition"]
                    and info["mapper_max_permutations"] == profiled_info["mapper_max_permutations"]
                ):
                    layer_info[layer_id]["name"] = profiled_name
                    for key in self.layer_metric_keys:
                        layer_info[layer_id][key] = profiled_info[key]

        # Run timeloop and process the results
        self.run_timeloop(layer_info)
        self.process_results(layer_info)

        # Process layer info into profiled_lib and write to file
        self.populate_profiled_lib(layer_info)
        self.write_profiled_lib()

        # Create overall summary and per-layer metrics
        # Since we deduplicated layer info, we need to multiply the num back to get the totals
        total_metrics = defaultdict(float)
        layer_metrics = {}
        for layer_id, info in layer_info.items():
            layer_metrics[layer_id] = {"name": info["name"]}
            for metric in ["area", "energy", "cycle"]:
                layer_total = info[metric] * info["num"]
                total_metrics[f"total_{metric}"] += layer_total
                layer_metrics[layer_id][f"total_{metric}"] = layer_total

            # Add other layer metrics
            layer_metrics[layer_id]["num"] = info["num"]
            for key in self.layer_metric_keys:
                layer_metrics[layer_id][key] = info[key]

        overall = {
            **total_metrics,
            # num_params for the whole model
            "num_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            # MACs for batch size 1
            "macs": profile_macs(self.model, torch.randn([1] + list(self.input_size))),
            # Activation size for batch size 1
            "activation_size": count_activation_size(self.model, [1] + list(self.input_size)),
        }

        return layer_info, overall, layer_metrics


def test():
    import pdb

    import torchvision

    pdb.set_trace()
    profiler = Profiler(
        top_dir="workloads",
        sub_dir="alexnet",
        timeloop_dir="designs/simple_weight_stationary",
        arch_name="simple_weight_stationary",
        model=torchvision.models.alexnet(),
        input_size=(3, 224, 224),
        batch_size=1,
        convert_fc=True,
        exception_module_names=[],
    )
    profiler.profile()


if __name__ == "__main__":
    test()
