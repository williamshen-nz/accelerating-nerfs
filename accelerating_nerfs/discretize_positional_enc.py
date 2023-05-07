"""
Reference: https://jvm-gaming.org/t/fast-math-sin-cos-lookup-tables/36660
"""

import math

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAD = torch.pi / 180.0
DEG = 180.0 / torch.pi

SIN_BITS = 8
SIN_MASK = ~(-1 << SIN_BITS)
SIN_COUNT = SIN_MASK + 1
print("SIN COUNT", SIN_COUNT)

rad_full = torch.pi * 2.0
deg_full = 360.0
rad_to_index = SIN_COUNT / rad_full
deg_to_index = SIN_COUNT / deg_full

sin = torch.zeros(SIN_COUNT, device=device)
cos = torch.zeros(SIN_COUNT, device=device)

for i in range(SIN_COUNT):
    sin[i] = math.sin((i + 0.5) / SIN_COUNT * rad_full)
    cos[i] = math.cos((i + 0.5) / SIN_COUNT * rad_full)

# Four cardinal directions (credits: Nate)
for i in range(0, 360, 90):
    sin[int(i * deg_to_index) & SIN_MASK] = math.sin(i * torch.pi / 180.0)
    cos[int(i * deg_to_index) & SIN_MASK] = math.cos(i * torch.pi / 180.0)


def sin_lut(rad):
    return sin[(rad * rad_to_index + 0.5).long() & SIN_MASK]


if __name__ == "__main__":
    # Random angles between -5pi and 5pi
    angles = torch.rand(1000000, device=device) * 10 * torch.pi - 5 * torch.pi

    # Test sin_rad
    discretized = sin_lut(angles)
    continuous = torch.sin(angles)
    error = torch.abs(discretized - continuous)
    print(f"Max error: {error.max():.3e}")
    print(f"Mean error: {error.mean():.3e}")
