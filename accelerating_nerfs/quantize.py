import os
import tempfile

import torch
import torch.quantization
from nerfacc import OccGridEstimator
from torch.quantization import DeQuantStub, QuantStub
from tqdm import tqdm

from accelerating_nerfs.models import VanillaNeRF


def sizeof_fmt(num, suffix="B"):
    """Source: https://stackoverflow.com/a/1094933"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def get_size_of_model(model: torch.nn.Module) -> int:
    """Size of the model in bytes"""
    with tempfile.NamedTemporaryFile() as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name)
    return size


class QuantizedVanillaNeRF(torch.nn.Module):
    def __init__(self, vanilla_nerf: VanillaNeRF):
        super().__init__()
        self.quant_x = QuantStub()
        self.quant_condition = QuantStub()
        self.dequant_rgb = DeQuantStub()
        self.dequant_sigma = DeQuantStub()
        self.vanilla_nerf = vanilla_nerf

    def query_density(self, x):
        # Quantize
        x = self.quant_x(x)
        # Forward pass
        sigma = self.vanilla_nerf.query_density(x)
        # Dequantize
        sigma = self.dequant_sigma(sigma)
        return sigma

    def forward(self, x, condition=None):
        # Quantize
        x = self.quant_x(x)
        if condition is not None:
            condition = self.quant_condition(condition)
        # Forward pass
        rgb_act, sigma_act = self.vanilla_nerf(x, condition)
        # Dequantize
        rgb_act = self.dequant_rgb(rgb_act)
        sigma_act = self.dequant_sigma(sigma_act)
        return rgb_act, sigma_act


def quantize_vanilla_nerf(model: VanillaNeRF, estimator: OccGridEstimator, scene: str) -> QuantizedVanillaNeRF:
    """Quantize a VanillaNeRF model. Don't worry about the estimator."""
    from accelerating_nerfs.render import (
        load_test_dataset,
        near_plane,
        render_step_size,
    )
    from accelerating_nerfs.utils import render_image_with_occgrid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantized_model = QuantizedVanillaNeRF(model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config
    torch.quantization.prepare(quantized_model, inplace=True)

    test_dataset = load_test_dataset(scene, num_downscales=2)

    # Calibrate by rendering the test dataset
    for idx in tqdm(range(len(test_dataset)), f"Running {scene} test images for quantization"):
        data = test_dataset[idx]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]

        # Render
        rgb, acc, depth, _ = render_image_with_occgrid(
            quantized_model,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            # test options
            test_chunk_size=4096,
        )

    quantized_model = quantized_model.cpu()
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    quantized_model = quantized_model.to(device)
    quantized_model.eval()
    return quantized_model
