#!/usr/bin/env python3
"""Round-trip validator — run where GDAL 3.12 is available.

Usage
-----
    python scripts/validate_roundtrip.py

Builds synthetic NetCDF sources, writes a VRT via xvrt, then re-opens the
VRT through ``osgeo.gdal``'s mdim API and cross-checks shape / dtype /
coord values / data values against the originating ``xr.Dataset``.

This is the correctness criterion described in the plan. The sandbox where
xvrt was developed doesn't have GDAL, so this script is kept as a
standalone entry point for users / CI to run locally.

Exit codes
----------
0 — all checks passed
1 — mismatch or unexpected error
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

# Adjust if running outside the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xvrt import write_mdim_vrt  # noqa: E402


_WKT = (
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],'
    'AXIS["latitude",north],AXIS["longitude",east],'
    'UNIT["degree",0.0174532925199433]]'
)


def _make_source(tmp: Path, index: int, nt: int, t_start: float) -> tuple[Path, np.ndarray]:
    ny, nx = 4, 6
    y = np.linspace(-10.0, 10.0, ny)
    x = np.linspace(140.0, 150.0, nx)
    t = np.arange(t_start, t_start + nt, dtype=np.float64)
    rng = np.random.default_rng(index)
    vals = rng.standard_normal((nt, ny, nx)).astype(np.float32)
    ds = xr.Dataset(
        {"temp": (("time", "y", "x"), vals, {"units": "degC"})},
        coords={
            "time": ("time", t, {"axis": "T", "standard_name": "time",
                                  "units": "days since 2020-01-01"}),
            "y": ("y", y, {"axis": "Y", "standard_name": "latitude"}),
            "x": ("x", x, {"axis": "X", "standard_name": "longitude"}),
        },
    )
    p = tmp / f"src_{index:02d}.nc"
    ds.to_netcdf(p)
    return p, vals


def main() -> int:
    try:
        from osgeo import gdal
    except ImportError:
        print("ERROR: this script needs osgeo.gdal (GDAL Python bindings).")
        print("Install it in your environment first, e.g. via conda or system package.")
        return 1

    gdal.UseExceptions()
    drv_version = gdal.__version__
    print(f"GDAL {drv_version}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # --- stack case --------------------------------------------------
        n_sources = 4
        path_val_pairs = [_make_source(td, i, nt=1, t_start=float(i)) for i in range(n_sources)]
        paths = [p for p, _ in path_val_pairs]
        expected = np.concatenate([v for _, v in path_val_pairs], axis=0)

        pieces = [xr.open_dataset(p, decode_times=False) for p in paths]
        ds = xr.concat(pieces, dim="time")

        vrt_path = td / "stack.vrt"
        write_mdim_vrt(
            ds, [str(p) for p in paths], vrt_path,
            composition="stack", concat_dim="time", crs=_WKT,
        )
        print(f"wrote {vrt_path}")

        # Re-open through GDAL mdim.
        gds = gdal.OpenEx(str(vrt_path), gdal.OF_MULTIDIM_RASTER)
        if gds is None:
            print("FAIL: GDAL could not open the VRT")
            return 1
        root = gds.GetRootGroup()
        arr = root.OpenMDArrayFromFullname("/temp")
        shape = tuple(arr.GetDimensions()[i].GetSize() for i in range(3))
        expect_shape = (n_sources, 4, 6)
        if shape != expect_shape:
            print(f"FAIL: shape {shape} != {expect_shape}")
            return 1
        got = arr.ReadAsArray()
        if not np.allclose(got, expected, equal_nan=True):
            print("FAIL: values differ between expected and GDAL-read")
            diff = np.abs(got - expected)
            print(f"  max abs diff: {diff.max()}  at {np.unravel_index(diff.argmax(), diff.shape)}")
            return 1
        print(f"OK stack  shape={shape}  dtype={gdal.GetDataTypeName(arr.GetDataType().GetNumericDataType())}")

        # Coord check.
        time_arr = root.OpenMDArrayFromFullname("/time")
        got_time = time_arr.ReadAsArray()
        if not np.allclose(got_time, np.arange(n_sources, dtype=np.float64)):
            print(f"FAIL: time coord {got_time} != expected")
            return 1
        print(f"OK time   {got_time}")

        # --- concat case -------------------------------------------------
        pp = [_make_source(td, 10 + i, nt=3, t_start=float(i * 3)) for i in range(2)]
        cat_paths = [p for p, _ in pp]
        cat_expected = np.concatenate([v for _, v in pp], axis=0)

        cat_ds = xr.concat(
            [xr.open_dataset(p, decode_times=False) for p in cat_paths],
            dim="time",
        )
        cat_vrt = td / "concat.vrt"
        write_mdim_vrt(
            cat_ds,
            [{"path": str(p), "size": 3} for p in cat_paths],
            cat_vrt,
            composition="concat", concat_dim="time", crs=_WKT,
        )
        gds2 = gdal.OpenEx(str(cat_vrt), gdal.OF_MULTIDIM_RASTER)
        arr2 = gds2.GetRootGroup().OpenMDArrayFromFullname("/temp")
        got2 = arr2.ReadAsArray()
        if got2.shape != (6, 4, 6):
            print(f"FAIL: concat shape {got2.shape} != (6,4,6)")
            return 1
        if not np.allclose(got2, cat_expected, equal_nan=True):
            print("FAIL: concat values differ")
            return 1
        print(f"OK concat shape={got2.shape}")

    print("\nAll round-trip checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
