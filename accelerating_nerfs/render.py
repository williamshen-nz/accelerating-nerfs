import argparse
import os

import imageio
import moviepy
import numpy as np
import torch
import torch.nn.functional as F
from datasets.nerf_synthetic import SubjectLoader
from lpips import LPIPS
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.VideoClip import ImageClip, VideoClip
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default="/home/william/workspace/dl-hardware/nerfacc/nerf_synthetic",
    help="the root dir of the dataset",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="results/chair/nerf_20000.pt",
    help="the path of the pretrained model",
)
parser.add_argument(
    "--scene",
    type=str,
    default="chair",
    choices=NERF_SYNTHETIC_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=4096,
)
args = parser.parse_args()

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

# setup the dataset
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
)
test_dataset.downscale(num_downscales=1)

# Load from checkpoint
checkpoint = torch.load(args.model_path)
radiance_field = VanillaNeRF().to(device)
radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)
estimator.load_state_dict(checkpoint["estimator_state_dict"])

radiance_field.eval()
estimator.eval()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

psnrs = []
lpips = []

render_dir = f"renders/{args.scene}"
os.makedirs(render_dir, exist_ok=True)
image_dir = os.path.join(render_dir, "images")
os.makedirs(image_dir, exist_ok=True)
image_paths = []

with torch.no_grad():
    for i in tqdm(range(len(test_dataset)), desc="Rendering image"):
        data = test_dataset[i]
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
            test_chunk_size=args.test_chunk_size,
        )
        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        psnrs.append(psnr.item())
        lpips.append(lpips_fn(rgb, pixels).item())
        image_path = os.path.join(image_dir, f"rgb_{i:04d}.png")
        imageio.imwrite(
            image_path,
            (rgb.cpu().numpy() * 255).astype(np.uint8),
        )
        image_paths.append(image_path)

# concat images into video at 24fps
video_path = os.path.join(render_dir, "video.mp4")
clip = ImageSequenceClip(image_paths, fps=24)
clip.write_videofile(video_path)
print(f"video saved to {video_path}")

# metrics
psnr_avg = sum(psnrs) / len(psnrs)
lpips_avg = sum(lpips) / len(lpips)
print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")

metrics = {
    "psnr_avg": psnr_avg,
    "lpips_avg": lpips_avg,
    "psnrs": psnrs,
    "lpips": lpips,
}
