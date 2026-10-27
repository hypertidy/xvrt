"""Small on-disk inputs for the documentation pages.

Each function writes files into a fresh temporary directory and returns
their paths, so pages run anywhere with xarray, netCDF4 and GDAL.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

WKT_4326 = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],'
    'AUTHORITY["EPSG","4326"]]'
)

LAT = np.array([-44.5, -43.5, -42.5])
LON = np.array([145.5, 146.5, 147.5, 148.5])


def workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="xvrt-docs-"))


def daily_netcdf(n: int = 4, variables=("sst",), packed: bool = True) -> list[Path]:
    """One file per day, (time, lat, lon), CF time, optionally packed int16."""
    out = workdir()
    rng = np.random.default_rng(1)
    paths = []
    for i in range(n):
        data_vars, encoding = {}, {}
        for v in variables:
            values = 12.0 + rng.normal(0, 1, (1, LAT.size, LON.size))
            data_vars[v] = (("time", "lat", "lon"), values,
                            {"units": "degC", "long_name": f"daily {v}"})
            if packed:
                encoding[v] = {"dtype": "int16", "scale_factor": 0.01,
                               "add_offset": 0.0, "_FillValue": np.int16(-32768)}
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": ("time", [float(i)], {"units": "days since 2026-01-01 12:00:00",
                                              "axis": "T", "standard_name": "time"}),
                "lat": ("lat", LAT, {"units": "degrees_north", "axis": "Y",
                                     "standard_name": "latitude"}),
                "lon": ("lon", LON, {"units": "degrees_east", "axis": "X",
                                     "standard_name": "longitude"}),
            },
            attrs={"title": "documentation fixture"},
        )
        p = out / f"sst_202601{i + 1:02d}.nc"
        ds.to_netcdf(p, encoding=encoding)
        paths.append(p)
    return paths


def slices_2d(n: int = 3) -> list[Path]:
    """2D files without a time dim, e.g. one model level or one date each."""
    out = workdir()
    paths = []
    for i in range(n):
        ds = xr.Dataset(
            {"temp": (("lat", "lon"), np.full((LAT.size, LON.size), 10.0 + i, "f4"))},
            coords={"lat": LAT, "lon": LON},
        )
        p = out / f"temp_level{i}.nc"
        ds.to_netcdf(p)
        paths.append(p)
    return paths


def geotiffs(n: int = 3, name: str = "analysed_sst") -> list[Path]:
    """Packed Int16 GeoTIFFs on EPSG:4326, one per day, like a COG collection."""
    from osgeo import gdal

    out = workdir()
    paths = []
    for i in range(n):
        p = out / f"2026010{i + 1}_{name}.tif"
        d = gdal.GetDriverByName("GTiff").Create(str(p), LON.size, LAT.size, 1, gdal.GDT_Int16)
        d.SetGeoTransform([145.0, 1.0, 0, -42.0, 0, -1.0])
        d.SetProjection(WKT_4326)
        b = d.GetRasterBand(1)
        b.SetNoDataValue(-32768)
        b.SetScale(0.001)
        b.SetOffset(25.0)
        b.SetUnitType("celsius")
        b.WriteArray((np.arange(12).reshape(3, 4) * 50 - 13000 + 100 * i).astype("int16"))
        d = None
        paths.append(p)
    return paths


def show_xml(path, max_lines: int = 60) -> None:
    lines = Path(path).read_text().splitlines()
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"... ({len(lines) - max_lines} more lines)")
