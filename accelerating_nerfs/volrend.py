"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.

Modified from: https://github.com/KAIR-BAIR/nerfacc/blob/master/nerfacc/volrend.py
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from nerfacc import (
    accumulate_along_rays,
    render_weight_from_alpha,
    render_weight_from_density,
)
from torch import Tensor

from accelerating_nerfs.profiler import profiler


def rendering(
    # ray marching results
    t_starts: Tensor,
    t_ends: Tensor,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
    # radiance field
    rgb_sigma_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Render the rays through the radience field defined by `rgb_sigma_fn`."""
    if ray_indices is not None:
        assert (
            t_starts.shape == t_ends.shape == ray_indices.shape
        ), "Since nerfacc 0.5.0, t_starts, t_ends and ray_indices must have the same shape (N,). "

    if rgb_sigma_fn is None:
        raise ValueError("`rgb_sigma_fn` must be specified.")

    # Query sigma/alpha and color with gradients
    rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices)
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(rgbs.shape)
    assert sigmas.shape == t_starts.shape, "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)

    # Rendering: compute weights.
    with profiler.profile("rendering.render_weight_from_density", len(ray_indices)):
        weights, trans, alphas = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
    extras = {
        "weights": weights,
        "alphas": alphas,
        "trans": trans,
        "sigmas": sigmas,
        "rgbs": rgbs,
    }

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    with profiler.profile("rendering.colors.accumulate_along_rays", len(ray_indices)):
        colors = accumulate_along_rays(weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays)

    opacities = accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, extras
