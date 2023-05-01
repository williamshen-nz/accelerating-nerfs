"""
Train over all nerf synthetic scenes.
"""

import argparse
import os
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader
from lpips import LPIPS
from nerfacc.estimators.occ_grid import OccGridEstimator

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
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=4096,
)
args = parser.parse_args()

# training parameters
max_steps = 20000
eval_every = 2000
init_batch_size = 1024
target_sample_batch_size = 1 << 16
# scene parameters
aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = 5e-3

for scene in NERF_SYNTHETIC_SCENES:
    if scene in {"chair", "drums"}:
        continue

    print(f"Training on {scene}")
    exp_dir = f"results/{scene}"
    os.makedirs(exp_dir, exist_ok=True)

    # setup the dataset
    train_dataset = SubjectLoader(
        subject_id=scene,
        root_fp=args.data_root,
        split=args.train_split,
        num_rays=init_batch_size,
        device=device,
    )
    test_dataset = SubjectLoader(
        subject_id=scene,
        root_fp=args.data_root,
        split="test",
        num_rays=None,
        device=device,
    )

    estimator = OccGridEstimator(
        roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
    ).to(device)

    # setup the radiance field we want to train.
    radiance_field = VanillaNeRF().to(device)
    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            max_steps // 2,
            max_steps * 3 // 4,
            max_steps * 5 // 6,
            max_steps * 9 // 10,
        ],
        gamma=0.33,
    )

    lpips_net = LPIPS(net="vgg").to(device)
    lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
    lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()
    step = 0

    # training
    tic = time.time()
    for step in tqdm.tqdm(range(max_steps + 1)):
        radiance_field.train()
        estimator.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]

        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
        )
        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays * (target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)

        # compute loss
        loss = F.smooth_l1_loss(rgb, pixels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % eval_every == 0:
            elapsed_time = time.time() - tic
            loss = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(loss) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss:.5f} | psnr={psnr:.2f} | "
                f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                f"max_depth={depth.max():.3f} | "
            )

        if step > 0 and (step % eval_every == 0 or step == max_steps):
            model_save_path = os.path.join(exp_dir, f"nerf_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "scene": scene,
                    "radiance_field_state_dict": radiance_field.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "estimator_state_dict": estimator.state_dict(),
                },
                model_save_path,
            )

            # evaluation
            radiance_field.eval()
            estimator.eval()

            psnrs = []
            lpips = []
            with torch.no_grad():
                for i in range(5):
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
                    imageio.imwrite(
                        os.path.join(exp_dir, f"rgb_test_step{step:05d}_{i:03d}.png"),
                        (rgb.cpu().numpy() * 255).astype(np.uint8),
                    )
                    imageio.imwrite(
                        os.path.join(exp_dir, f"rgb_error_{step:05d}_{i:03d}.png"),
                        ((rgb - pixels).norm(dim=-1).cpu().numpy() * 255).astype(
                            np.uint8
                        ),
                    )
            psnr_avg = sum(psnrs) / len(psnrs)
            lpips_avg = sum(lpips) / len(lpips)
            print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
