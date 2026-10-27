"""xvrt: write GDAL multidimensional VRT from xarray Datasets and STAC catalogues.

A pure-Python writer. ``write_mdim_vrt`` takes an ``xarray.Dataset`` plus the
source files and emits a GDAL multidim VRT; ``from_stac`` builds one from
STAC catalogue fields alone. Output validates against GDAL's ``gdalvrt.xsd``.

Documentation: https://hypertidy.github.io/xvrt/
"""
from __future__ import annotations

from ._writer import write_mdim_vrt, encode_datetime_coord
from ._errors import VrtWriterError
from ._stac import from_stac

__all__ = ["write_mdim_vrt", "from_stac", "encode_datetime_coord", "VrtWriterError"]
__version__ = "0.1.0"
