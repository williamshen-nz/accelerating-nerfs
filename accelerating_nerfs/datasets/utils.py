"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.

From: https://github.com/KAIR-BAIR/nerfacc
"""

import collections

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))
