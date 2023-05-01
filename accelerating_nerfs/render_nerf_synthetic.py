import json
import os
from typing import Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from datasets.nerf_synthetic import SubjectLoader
from lpips import LPIPS
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from nerfacc.estimators.occ_grid import OccGridEstimator
from tqdm import tqdm

from accelerating_nerfs.config import nerf_synthetic_config
from accelerating_nerfs.models import VanillaNeRF
from accelerating_nerfs.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)

device = "cuda:0"
set_random_seed(42)
checkpoint_pattern = "results/{scene}/nerf_20000.pt"

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


def load_test_dataset(scene: str, num_downscales: int) -> SubjectLoader:
    test_dataset = SubjectLoader(
        subject_id=scene,
        root_fp="/home/william/workspace/dl-hardware/nerfacc/nerf_synthetic",
        split="test",
        num_rays=None,
        device=device,
    )
    test_dataset.downscale(num_downscales=num_downscales)
    return test_dataset


def load_checkpoint(model_path: str) -> Tuple[VanillaNeRF, OccGridEstimator]:
    checkpoint = torch.load(model_path)
    # Load NeRF
    radiance_field = VanillaNeRF().to(device)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])

    # Load OccGridEstimator
    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    ).to(device)
    estimator.load_state_dict(checkpoint["estimator_state_dict"])

    radiance_field.eval()
    estimator.eval()
    return radiance_field, estimator


@torch.no_grad()
def render_nerf_synthetic(num_downscales: int, max_num_scenes: Optional[int] = None):
    assert num_downscales >= 0
    assert max_num_scenes is None or max_num_scenes > 0

    # Setup LPIPS
    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

    def render_scene(scene: str):
        print(f"===== Rendering '{scene}' scene =====")
        psnrs = []
        lpips = []

        # Load checkpoint
        model_path = checkpoint_pattern.format(scene=scene)
        radiance_field, estimator = load_checkpoint(model_path)

        # Load test dataset
        test_dataset = load_test_dataset(scene, num_downscales)

        # Create render and image directories
        render_dir = f"renders/{scene}"
        os.makedirs(render_dir, exist_ok=True)
        image_dir = os.path.join(render_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        # Render frames
        image_paths = []
        for idx in tqdm(range(len(test_dataset)), desc="Rendering image"):
            data = test_dataset[idx]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # rendering
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
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())
            lpips.append(lpips_fn(rgb, pixels).item())
            image_path = os.path.join(image_dir, f"rgb_{idx:04d}.png")
            imageio.imwrite(
                image_path,
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            image_paths.append(image_path)
        print(f"Successfully rendered {len(image_paths)} images")

        # Create video
        video_path = os.path.join(render_dir, "video.mp4")
        clip = ImageSequenceClip(image_paths, fps=24)
        clip.write_videofile(video_path, verbose=False)
        print(f"Saved video to {video_path}")

        # Metrics
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
        metrics = {
            "psnr_avg": psnr_avg,
            "lpips_avg": lpips_avg,
            "psnrs": psnrs,
            "lpips": lpips,
        }
        metrics_path = os.path.join(render_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics to {metrics_path}")

    for scene_ in NERF_SYNTHETIC_SCENES[:max_num_scenes]:
        render_scene(scene_)


if __name__ == "__main__":
    render_nerf_synthetic(num_downscales=1, max_num_scenes=None)
