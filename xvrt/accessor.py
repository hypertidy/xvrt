"""Optional ``.vrt`` accessor on xarray Datasets.

Enables:

    import xvrt.accessor   # side-effect registers the accessor
    ds.vrt.write_mdim(path, sources=[...], composition="stack", concat_dim="time")

Mirrors the ``rioxarray``/``cf_xarray`` accessor patterns. The Dataset stays
a plain ``xr.Dataset`` everywhere else — this layer just provides a
method-call spelling of :func:`xvrt.write_mdim_vrt`.

Importing this module registers the accessor with xarray's global registry.
The core ``xvrt`` package does **not** import it for you — opt in explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import xarray as xr

from ._writer import write_mdim_vrt, Composition
from ._coords import CoordMode


@xr.register_dataset_accessor("vrt")
class VrtAccessor:
    """``ds.vrt`` — write multidim VRT from an xarray Dataset."""

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def write_mdim(
        self,
        path: str | Path,
        *,
        sources: Sequence[Any],
        composition: Composition,
        concat_dim: str | None = None,
        crs: str | None = None,
        block_size: Any = None,
        coord_mode: CoordMode | Any = "auto",
        inline_threshold: int = 10_000,
        sizes: Sequence[int] | None = None,
        require_crs: bool = False,
        variables: Sequence[str] | None = None,
    ) -> Path:
        """Method-call form of :func:`xvrt.write_mdim_vrt`.

        All keyword args are forwarded unchanged.
        """
        return write_mdim_vrt(
            self._ds,
            sources,
            path,
            composition=composition,
            concat_dim=concat_dim,
            crs=crs,
            block_size=block_size,
            coord_mode=coord_mode,
            inline_threshold=inline_threshold,
            sizes=sizes,
            require_crs=require_crs,
            variables=variables,
        )
