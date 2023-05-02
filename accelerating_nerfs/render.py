import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from lpips import LPIPS
from nerfacc import OccGridEstimator
from tqdm import tqdm

from accelerating_nerfs.config import (
    get_nerf_synthetic_dataset_dir,
    nerf_synthetic_config,
)
from accelerating_nerfs.datasets.nerf_synthetic import SubjectLoader
from accelerating_nerfs.models import VanillaNeRF
from accelerating_nerfs.profiler import profiler
from accelerating_nerfs.utils import render_image_with_occgrid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = nerf_synthetic_config(device)
aabb = config["aabb"]

# Scene parameters
near_plane = config["near_plane"]
far_plane = config["far_plane"]

# Model parameters
grid_resolution = config["grid_resolution"]
grid_nlvl = config["grid_nlvl"]

# Render parameters
render_step_size = config["render_step_size"]


@lru_cache(maxsize=1)
def load_lpips() -> LPIPS:
    lpips_net = LPIPS(net="vgg").to(device)
    return lpips_net


def load_checkpoint(model_path: str) -> Tuple[VanillaNeRF, OccGridEstimator]:
    checkpoint = torch.load(model_path)
    # Load NeRF
    radiance_field = VanillaNeRF().to(device)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])

    # Load OccGridEstimator
    estimator = OccGridEstimator(roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl).to(device)
    estimator.load_state_dict(checkpoint["estimator_state_dict"])

    radiance_field.eval()
    estimator.eval()
    return radiance_field, estimator


def load_test_dataset(scene: str, num_downscales: int) -> SubjectLoader:
    test_dataset = SubjectLoader(
        subject_id=scene,
        root_fp=str(get_nerf_synthetic_dataset_dir()),
        split="test",
        num_rays=None,
        device=device,
    )
    test_dataset.downscale(num_downscales=num_downscales)
    return test_dataset


def render_nerf_synthetic(
    scene: str,
    checkpoint: str,
    result_dir: Path,
    num_downscales: int = 0,
    profile: bool = False,
    video_fps: int = 24,
):
    assert num_downscales >= 0
    os.makedirs(result_dir, exist_ok=True)
    profiler.enable(profile)
    profiler.clear()

    # Setup LPIPS
    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    # Load checkpoint
    radiance_field, estimator = load_checkpoint(checkpoint)

    # Load test dataset
    test_dataset = load_test_dataset(scene, num_downscales)
    psnrs, lpips = [], []

    # RGB images
    rgb_dir = os.path.join(result_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    rgbs, rgb_paths = [], []

    # Render frames
    for idx in tqdm(range(len(test_dataset)), f"Rendering {scene} test images"):
        data = test_dataset[idx]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        # Render
        rgb, acc, depth, _ = render_image_with_occgrid(
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
        # TODO: save depths?
        rgb_image = (rgb.cpu().numpy() * 255).astype(np.uint8)
        rgbs.append(rgb_image)

        # Save RGB frame
        rgb_path = os.path.join(rgb_dir, f"rgb_{idx:04d}.png")
        imageio.imwrite(rgb_path, rgb_image)
        rgb_paths.append(rgb_path)

        # Calculate metrics
        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        psnrs.append(psnr.item())
        lpips.append(lpips_fn(rgb, pixels).item())

    print(f"Successfully rendered {len(test_dataset)} images")

    # Stop profiling
    if profile:
        profiler.save(f"{result_dir}/profile.json")

    # Save metrics
    psnr_avg = np.mean(psnrs)
    lpips_avg = np.mean(lpips)
    print(f"PSNR: {psnr_avg:.4f}, LPIPS: {lpips_avg:.4f}")

    metrics = {
        "scene": scene,
        "checkpoint": checkpoint,
        "num_downscales": num_downscales,
        "psnr_avg": psnr_avg,
        "lpips_avg": lpips_avg,
        "psnrs": psnrs,
        "lpips": lpips,
    }
    metrics_path = os.path.join(result_dir, "render_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Create video from RGB images
    video_path = os.path.join(result_dir, "video.mp4")
    imageio.mimwrite(video_path, rgbs, fps=video_fps)
    print(f"Saved results to {result_dir}")
