import os
import tempfile

import torch
import torch.quantization
from torch.quantization import DeQuantStub, QuantStub

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


class QuantizedVanillaNeRF(VanillaNeRF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x, condition=None):
        # Quantize
        x = self.quant(x)
        if condition is not None:
            condition = self.quant(condition)
        # Forward pass
        rgb_act, sigma_act = super().forward(x, condition)
        # Dequantize
        rgb_act = self.dequant(rgb_act)
        sigma_act = self.dequant(sigma_act)
        return rgb_act, sigma_act


def quantize_vanilla_nerf(model: VanillaNeRF) -> QuantizedVanillaNeRF:
    """Quantize a VanillaNeRF model"""
    quantized_model = QuantizedVanillaNeRF()
    quantized_model.load_state_dict(model.state_dict())
    quantized_model.eval()
    return quantized_model
