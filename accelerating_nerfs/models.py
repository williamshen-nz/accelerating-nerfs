"""
Modified from https://github.com/KAIR-BAIR/nerfacc/blob/master/examples/radiance_fields/mlp.py

Vanilla NeRF that uses MLP.
"""

import math
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerating_nerfs.discretize_positional_enc import sin_lut
from accelerating_nerfs.profiler import profiler


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: Optional[int] = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init

        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(nn.Linear(in_features, self.net_width, bias=bias_enabled))
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(in_features, self.output_dim, bias=bias_enabled)
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def forward(self, x):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            x = self.hidden_activation(x)
            if (self.skip_layer is not None) and (i % self.skip_layer == 0) and (i > 0):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)

        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None):
        x = self.base(x)
        raw_sigma = self.sigma_layer(x)
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view([num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]).expand(
                    list(x.shape[:-1]) + [n_dim]
                )
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x)
        return raw_rgb, raw_sigma


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self._use_sin_lut = False
        self._sin_fn = torch.sin
        self.register_buffer("scales", torch.tensor([2**i for i in range(min_deg, max_deg)]))

    def enable_sin_lut(self):
        self._use_sin_lut = True
        self._sin_fn = sin_lut
        print("Using sin_lut for positional encoding.")

    @property
    def latent_dim(self) -> int:
        return (int(self.use_identity) + (self.max_deg - self.min_deg) * 2) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = self._sin_fn(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class VanillaNeRF(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
        )

    # def query_opacity(self, x, step_size):
    #     density = self.query_density(x)
    #     # if the density is small enough those two are the same.
    #     # opacity = 1.0 - torch.exp(-density * step_size)
    #     opacity = density * step_size
    #     return opacity

    def query_density(self, x):
        with profiler.profile("nerf.query_density.posi_encoder", x.shape[0]):
            x = self.posi_encoder(x)
        with profiler.profile("nerf.query_density.mlp_query_density", x.shape[0]):
            sigma = self.mlp.query_density(x)
        return F.relu(sigma)

    def forward(self, x, condition=None):
        with profiler.profile("nerf.forward.posi_encoder", x.shape[0]):
            x = self.posi_encoder(x)
        if condition is not None:
            with profiler.profile("nerf.forward.view_encoder", x.shape[0]):
                condition = self.view_encoder(condition)
        with profiler.profile("nerf.forward.mlp_forward", x.shape[0]):
            rgb, sigma = self.mlp(x, condition=condition)
        with profiler.profile("nerf.forward.sigmoid", rgb.shape[0]):
            rgb_act = torch.sigmoid(rgb)
        with profiler.profile("nerf.forward.relu", sigma.shape[0]):
            sigma_act = F.relu(sigma)
        return rgb_act, sigma_act


def patch_forward(model: VanillaNeRF) -> None:
    """Patch the forward of VanillaNeRF to also pass the condition. Use only for debugging purposes."""
    assert isinstance(model, VanillaNeRF)
    model.old_forward = model.forward

    def new_forward(self, x):
        return self.old_forward(x, x)

    model.forward = new_forward.__get__(model)
    warnings.warn(
        "patched forward of VanillaNeRF to also pass the condition. "
        "You should only use this for debugging or with Timeloop and Accelergy"
    )


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nerf = VanillaNeRF().to(device)
    patch_forward(nerf)
    summary(nerf, input_size=(1, 3))
