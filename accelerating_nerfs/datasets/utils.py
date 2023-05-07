"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.

From: https://github.com/KAIR-BAIR/nerfacc
"""

import collections
from typing import NamedTuple

import torch


class Rays(NamedTuple):
    origins: torch.Tensor
    viewdirs: torch.Tensor

    def half(self):
        return Rays(self.origins.half(), self.viewdirs.half())


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))
