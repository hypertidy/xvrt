"""xvrt — xarray to GDAL multidimensional VRT.

A pure-Python writer. Takes an ``xarray.Dataset`` plus an explicit source
specification and emits a valid GDAL multidim VRT conforming to the
``release/3.12`` ``gdalvrt.xsd``.

Parallel to the R package ``vrtstack``. The two are independent
implementations of the same spec.

Scope and rules: see ``mdim-vrt-writer-plan.md``.
"""
from __future__ import annotations

from ._writer import write_mdim_vrt, encode_datetime_coord
from ._errors import VrtWriterError
from ._stac import from_stac

__all__ = ["write_mdim_vrt", "from_stac", "encode_datetime_coord", "VrtWriterError"]
__version__ = "0.0.1.9000"
