"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional

from accelerating_nerfs.profiler import profiler
from accelerating_nerfs.volrend import rendering

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
from datasets.utils import Rays, namedtuple_map
from nerfacc.estimators.occ_grid import OccGridEstimator

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]

MIPNERF360_UNBOUNDED_SCENES = [
    "garden",
    "bicycle",
    "bonsai",
    "counter",
    "kitchen",
    "room",
    "stump",
]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # simple quantization
    use_fp16: bool = False,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        with profiler.profile("sigma_fn", len(ray_indices)):
            if use_fp16:  # convert to fp16 if enabled
                t_starts = t_starts.half()
                t_ends = t_ends.half()

            with profiler.profile(f"sigma_fn.ray_to_positions", len(ray_indices)):
                t_origins = rays.origins[ray_indices]
                t_dirs = rays.viewdirs[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            with profiler.profile(f"sigma_fn.radiance_field.query_density", len(positions)):
                sigmas = radiance_field.query_density(positions)
                sigmas = sigmas.squeeze(-1)
        return sigmas

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        with profiler.profile("rgb_sigma_fn", len(ray_indices)):
            if use_fp16:  # convert to fp16 if enabled
                t_starts = t_starts.half()
                t_ends = t_ends.half()

            with profiler.profile("rgb_sigma_fn.ray_to_positions", len(ray_indices)):
                t_origins = chunk_rays.origins[ray_indices]
                t_dirs = chunk_rays.viewdirs[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            with profiler.profile("rgb_sigma_fn.radiance_field", len(positions)):
                rgbs, sigmas = radiance_field(positions, t_dirs)
                sigmas = sigmas.squeeze(-1)
        return rgbs, sigmas

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else test_chunk_size
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            # Need to pass float to estimator as it's custom CUDA code that uses floats
            chunk_rays.origins if not use_fp16 else chunk_rays.origins.float(),
            chunk_rays.viewdirs if not use_fp16 else chunk_rays.viewdirs.float(),
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        with profiler.profile("rendering", len(ray_indices)):
            rgb, opacity, depth, extras = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=render_bkgd,
                use_fp16=use_fp16,
            )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )
