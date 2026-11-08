"""Stack 2D rasters along a new dimension: the vrtstack mode.

The simple case the R package vrtstack covers: a list of GeoTIFF/COG files,
one per position of a new dimension (time, depth, level), becomes one
multidim VRT. No xarray input is needed; a template Dataset is built here
(zero-memory placeholders, as in ``from_stac``) and handed to
:func:`xvrt.write_mdim_vrt`, which emits ``<SourceBand>`` plus the
axis-inserting ``<SourceTranspose>``.

The grid (shape, geotransform, CRS, dtype, nodata) comes from the first
file, read with ``rasterio`` or ``osgeo.gdal`` -- whichever imports -- or is
passed explicitly with ``grid=`` so that nothing is opened at all. All
files are assumed to share the grid, as in vrtstack: identical grid or the
result is wrong, and GDAL's reader is the arbiter.
"""
from __future__ import annotations

import datetime as _dt
import re
from os.path import basename
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import xarray as xr

from ._errors import VrtWriterError


def stack_rasters(
    files: Sequence[str | Path],
    path: str | Path,
    *,
    dim: str = "time",
    values: Sequence[Any] | str | Callable[[str], Any] | None = None,
    format: str | None = None,
    start: float | None = None,
    step: float | None = None,
    units: str | None = None,
    attrs: Mapping[str, Any] | None = None,
    array_name: str | None = None,
    band: int = 1,
    grid: Mapping[str, Any] | None = None,
    crs: str | None = None,
    block_size: Sequence[int] | None = None,
    sort: bool = True,
) -> Path:
    """Write a multidim VRT stacking 2D raster files along a new dimension.

    Parameters
    ----------
    files
        Raster paths or URLs (GeoTIFF/COG and anything else GDAL opens as a
        classic raster). URLs become ``/vsicurl/`` sources.
    path
        Output ``.vrt`` path.
    dim
        Name of the new dimension.
    values
        Dimension values, one of:

        - a sequence of numbers, ``datetime``/``numpy.datetime64`` values,
          or date strings (parsed with ``format=`` when given, else ISO)
        - a regex with one capture group, applied to each basename
          (``r"(\\d{8})"`` pulls ``20230115`` from ``sst_20230115.tif``);
          the captured text is parsed with ``format=`` when given, else as
          a number
        - a callable taking the basename and returning a value
        - ``None`` with ``start=`` and ``step=`` for a regular axis
    format
        ``strptime`` format for parsing date text (e.g. ``"%Y%m%d"``).
        Parsed values become a datetime coordinate, re-encoded to CF time
        by the writer.
    start, step
        Regular-axis alternative to ``values``; files are taken in the
        order given.
    units
        Units for the dimension coordinate (``"cm"``, ``"hPa"``, or a CF
        time unit such as ``"days since 1970-01-01"``, which sets the time
        encoding for a datetime coordinate).
    attrs
        Extra attributes for the dimension coordinate (e.g.
        ``{"axis": "Z", "standard_name": "depth"}``) so the writer can mark
        the ``<Dimension>`` with its type and direction.
    array_name
        Name of the data array. Default: the common prefix of the basenames
        with trailing punctuation stripped, else ``"data"``.
    band
        Band of each file to reference (1-based). Default 1.
    grid
        Grid description, skipping file introspection entirely:
        ``{"shape": (ny, nx), "geotransform": (ulx, xres, xrot, uly, yrot,
        yres), "crs": "<WKT>", "dtype": "float32", "nodata": ...}``.
        ``shape`` and ``geotransform`` are required; the geotransform is
        GDAL order. Without ``grid=``, the first file is read with
        ``rasterio`` or ``osgeo.gdal``.
    crs
        WKT override; else from ``grid["crs"]`` or the inspected file.
    block_size
        Optional ``<BlockSize>`` as (1, blocky, blockx).
    sort
        Sort files by their dimension values (default). With
        ``start=``/``step=`` the order given is kept.

    Returns
    -------
    Path of the written VRT.
    """
    from ._writer import write_mdim_vrt

    files = [str(f) for f in files]
    if not files:
        raise VrtWriterError("stack_rasters: no files")
    n = len(files)

    vals = _resolve_values(files, values, format, start, step, n)
    if sort and values is not None:
        order = np.argsort(vals, kind="stable")
        files = [files[i] for i in order]
        vals = vals[order]

    info = _grid_info(grid, files[0], band)
    ny, nx = info["shape"]
    ulx, xres, xrot, uly, yrot, yres = info["geotransform"]
    if xrot != 0 or yrot != 0:
        raise VrtWriterError(
            f"rotated/sheared geotransform {info['geotransform']} is not "
            "supported"
        )
    x = ulx + xres * (np.arange(nx) + 0.5)
    y = uly + yres * (np.arange(ny) + 0.5)

    name = array_name or _common_prefix([basename(f) for f in files]) or "data"
    dtype = np.dtype(info.get("dtype") or "float32")
    placeholder = np.broadcast_to(np.zeros((), dtype=dtype), (n, ny, nx))
    da = xr.DataArray(placeholder, dims=(dim, "y", "x"), name=name)
    da.encoding["dtype"] = dtype
    if info.get("nodata") is not None:
        da.encoding["_FillValue"] = info["nodata"]

    coord = xr.DataArray(vals, dims=(dim,), name=dim)
    is_time = vals.dtype.kind == "M"
    if units:
        if is_time and "since" in units:
            coord.encoding["units"] = units
        else:
            coord.attrs["units"] = units
    if attrs:
        coord.attrs.update(attrs)

    ds = xr.Dataset(
        {name: da},
        coords={dim: coord, "y": ("y", y), "x": ("x", x)},
    )
    sources = [{"path": f, "band": int(band)} for f in files]
    return write_mdim_vrt(
        ds, sources, path,
        composition="stack", concat_dim=dim,
        crs=crs or info.get("crs"), block_size=block_size,
    )


def open_raster_stack(
    files: Sequence[str | Path],
    path: str | Path | None = None,
    *,
    engine: str = "gdalxarray",
    **kwargs: Any,
) -> xr.Dataset:
    """Stack raster files and open the result as an xarray Dataset.

    Writes the VRT with :func:`stack_rasters` (to ``path``, or next to a
    temporary directory when ``path`` is None) and opens it with the given
    xarray engine. Writing needs nothing beyond this package; reading the
    bytes back is the engine's job, so a GDAL-backed backend such as
    ``gdalxarray`` must be installed.

    All other keyword arguments go to :func:`stack_rasters`.
    """
    if path is None:
        import tempfile
        path = Path(tempfile.mkdtemp(prefix="xvrt-")) / "stack.vrt"
    vrt = stack_rasters(files, path, **kwargs)
    try:
        return xr.open_dataset(vrt, engine=engine)
    except ValueError as exc:
        raise VrtWriterError(
            f"the VRT was written to {vrt} but no reader opened it with "
            f"engine={engine!r}; install that backend (pip install "
            f"{engine}) or open the file with GDAL directly."
        ) from exc


# -- values ---------------------------------------------------------------

def _resolve_values(files, values, format, start, step, n) -> np.ndarray:
    if values is None:
        if start is None or step is None:
            raise VrtWriterError(
                "stack_rasters: pass values= (sequence, regex with one "
                "capture group, or callable), or start= and step= for a "
                "regular axis."
            )
        return np.asarray(start) + np.asarray(step) * np.arange(n, dtype=np.float64)
    if start is not None or step is not None:
        raise VrtWriterError(
            "stack_rasters: values= and start=/step= are mutually exclusive."
        )

    if isinstance(values, str):
        pat = re.compile(values)
        if pat.groups != 1:
            raise VrtWriterError(
                f"stack_rasters: values regex {values!r} needs exactly one "
                "capture group."
            )
        out = []
        for f in files:
            m = pat.search(basename(f))
            if m is None:
                raise VrtWriterError(
                    f"stack_rasters: values regex {values!r} did not match "
                    f"{basename(f)!r}."
                )
            out.append(m.group(1))
        values = out
    elif callable(values):
        values = [values(basename(f)) for f in files]
    else:
        values = list(values)

    if len(values) != n:
        raise VrtWriterError(
            f"stack_rasters: {len(values)} values for {n} files."
        )

    return _coerce_values(values, format)


def _coerce_values(values, format) -> np.ndarray:
    if format is not None:
        parsed = [_dt.datetime.strptime(str(v), format) for v in values]
        return np.array(parsed, dtype="datetime64[ns]")
    first = values[0]
    if isinstance(first, (_dt.date, _dt.datetime, np.datetime64)):
        return np.array(values, dtype="datetime64[ns]")
    if isinstance(first, str):
        try:
            return np.asarray([float(v) for v in values], dtype=np.float64)
        except ValueError:
            try:
                return np.array(values, dtype="datetime64[ns]")
            except ValueError:
                raise VrtWriterError(
                    f"stack_rasters: values are strings that parse as "
                    f"neither numbers nor dates (first: {first!r}); pass "
                    "format= for date text."
                ) from None
    return np.asarray(values, dtype=np.float64)


# -- grid -----------------------------------------------------------------

_GRID_KEYS = ("shape", "geotransform")


def _grid_info(grid, file, band) -> dict[str, Any]:
    if grid is not None:
        missing = [k for k in _GRID_KEYS if k not in grid]
        if missing:
            raise VrtWriterError(
                f"stack_rasters: grid= lacks {missing}; needs "
                "{'shape': (ny, nx), 'geotransform': (ulx, xres, xrot, uly, "
                "yrot, yres)} and optionally 'crs', 'dtype', 'nodata'."
            )
        out = dict(grid)
        out["shape"] = tuple(int(v) for v in out["shape"])
        out["geotransform"] = tuple(float(v) for v in out["geotransform"])
        return out
    return _raster_info(file, band)


def _raster_info(file: str, band: int) -> dict[str, Any]:
    """Grid of one raster via rasterio, else osgeo.gdal, else raise."""
    src = file
    if src.startswith(("http://", "https://")):
        src = "/vsicurl/" + src
    try:
        import rasterio
        with rasterio.open(src) as r:
            t = r.transform  # affine: a, b, c, d, e, f
            return {
                "shape": (r.height, r.width),
                "geotransform": (t.c, t.a, t.b, t.f, t.d, t.e),
                "crs": r.crs.to_wkt() if r.crs else None,
                "dtype": r.dtypes[band - 1],
                "nodata": r.nodatavals[band - 1],
            }
    except ImportError:
        pass
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        ds = gdal.Open(src)
        b = ds.GetRasterBand(band)
        info = {
            "shape": (ds.RasterYSize, ds.RasterXSize),
            "geotransform": tuple(ds.GetGeoTransform()),
            "crs": ds.GetProjectionRef() or None,
            "dtype": gdal.GetDataTypeName(b.DataType).lower()
                     .replace("byte", "uint8"),
            "nodata": b.GetNoDataValue(),
        }
        ds = None
        return info
    except ImportError:
        pass
    raise VrtWriterError(
        "stack_rasters: reading the grid from the first file needs rasterio "
        "or osgeo.gdal; install one, or pass grid={'shape': (ny, nx), "
        "'geotransform': (...), 'crs': ..., 'dtype': ..., 'nodata': ...}."
    )


def _common_prefix(names: Sequence[str]) -> str:
    import os.path
    prefix = os.path.commonprefix(list(names))
    return re.sub(r"[^A-Za-z0-9]+$", "", prefix)
