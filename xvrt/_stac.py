"""Build a multidim VRT straight from STAC items: no file is opened.

The catalogue already says everything the VRT needs: the grid
(``proj:shape``, ``proj:transform``), the CRS (``proj:epsg`` / ``proj:code``
/ ``proj:wkt2``), each item's datetime, and per asset the band's storage
(``raster:bands``: ``data_type``, ``nodata``, ``scale``, ``offset``,
``unit``). From that we build a template :class:`xarray.Dataset` whose data
variables are zero-memory broadcast placeholders of the full shape, plus one
source list per asset, and hand both to :func:`xvrt.write_mdim_vrt`.

This is the catalogue-driven counterpart of the R package vrtstack: "the grid
comes from the catalogue, identical grid or refuse", one array per asset, and
items that lack an asset leave nodata at their position.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from ._errors import VrtWriterError


def from_stac(
    items: str | Path | Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    assets: Sequence[str] | None = None,
    dim: str = "time",
    crs: str | None = None,
    block_size: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
) -> Path:
    """Write a multidim VRT stacking STAC items along ``dim``.

    Parameters
    ----------
    items
        A stac-geoparquet file (path or URL; needs ``pyarrow``), or an
        iterable of STAC item dicts (``pystac.Item.to_dict()`` works). Both
        the flat geoparquet layout and the nested ``"properties"`` layout are
        read.
    path
        Output ``.vrt`` path.
    assets
        Asset keys to write, one array each, in this order. Default: every
        asset with ``raster:bands``, in first-seen order.
    dim
        Name of the stack dimension (one position per distinct datetime).
    crs
        WKT override. Default: from the items' projection fields, turned into
        WKT with ``pyproj`` (or GDAL's ``osr``) when only an EPSG code is
        given, so the axis mapping can be computed.
    block_size
        Optional ``<BlockSize>``; the catalogue does not record it.

    Raises
    ------
    VrtWriterError
        Items on different grids, duplicate datetimes, a band description
        that changes between items, or no usable CRS.
    """
    from ._writer import write_mdim_vrt

    rows = [_flatten(it) for it in _read_items(items)]
    if not rows:
        raise VrtWriterError("no STAC items")

    times = [_item_time(r) for r in rows]
    order = np.argsort(np.array(times, dtype="datetime64[ns]"), kind="stable")
    rows = [rows[i] for i in order]
    times = np.array([times[i] for i in order], dtype="datetime64[ns]")
    if len(np.unique(times)) != len(times):
        dup = times[1:][times[1:] == times[:-1]]
        raise VrtWriterError(
            f"duplicate datetimes in items (first: {dup[0]}); one item per "
            f"position along {dim!r} is required."
        )

    keys = list(assets) if assets is not None else _band_assets(rows)
    if not keys:
        raise VrtWriterError("no assets with raster:bands found in the items")

    shape, transform = _common_grid(rows, keys)
    ny, nx = shape
    a, b, c, d, e, f = transform[:6]
    if b != 0 or d != 0:
        raise VrtWriterError(
            f"rotated/sheared transform {transform[:6]} is not supported"
        )
    x = c + a * (np.arange(nx) + 0.5)
    y = f + e * (np.arange(ny) + 0.5)

    wkt = crs or _common_crs_wkt(rows, keys)

    data_vars = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for k in keys:
        band = _common_band(rows, k)
        dtype = np.dtype(band.get("data_type") or "float32")
        placeholder = np.broadcast_to(np.zeros((), dtype=dtype), (len(rows), ny, nx))
        da = xr.DataArray(placeholder, dims=(dim, "y", "x"), name=k)
        enc: dict[str, Any] = {"dtype": dtype}
        if band.get("nodata") is not None:
            enc["_FillValue"] = band["nodata"]
        if band.get("scale") is not None:
            enc["scale_factor"] = band["scale"]
        if band.get("offset") is not None:
            enc["add_offset"] = band["offset"]
        da.encoding.update(enc)
        if band.get("unit"):
            da.attrs["units"] = band["unit"]
        data_vars[k] = da
        sources[k] = [
            {"path": r["assets"][k]["href"], "band": 1, "index": i}
            for i, r in enumerate(rows)
            if (r.get("assets") or {}).get(k)
        ]

    ds = xr.Dataset(
        data_vars,
        coords={dim: (dim, times), "y": ("y", y), "x": ("x", x)},
    )
    return write_mdim_vrt(
        ds, sources, path,
        composition="stack", concat_dim=dim, crs=wkt, block_size=block_size,
    )


# -- reading -------------------------------------------------------------------

def _read_items(items) -> list[Mapping[str, Any]]:
    if isinstance(items, (str, Path)):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise VrtWriterError(
                "reading stac-geoparquet needs pyarrow (pip install pyarrow)"
            ) from exc
        src = str(items)
        if src.startswith(("http://", "https://")):
            import urllib.request
            import io
            with urllib.request.urlopen(src) as resp:  # noqa: S310
                return pq.read_table(io.BytesIO(resp.read())).to_pylist()
        return pq.read_table(src).to_pylist()
    return [it.to_dict() if hasattr(it, "to_dict") else it for it in items]


def _flatten(item: Mapping[str, Any]) -> dict[str, Any]:
    """Geoparquet rows are flat; STAC JSON nests fields under properties."""
    out = dict(item.get("properties") or {})
    out.update({k: v for k, v in item.items() if k != "properties"})
    return out


def _item_time(r: Mapping[str, Any]) -> np.datetime64:
    t = r.get("datetime") or r.get("start_datetime")
    if t is None:
        raise VrtWriterError(f"item {r.get('id')!r} has no datetime")
    if isinstance(t, _dt.datetime):
        if t.tzinfo is not None:
            t = t.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return np.datetime64(t, "ns")
    s = str(t).replace("Z", "")
    if s.endswith("+00:00"):
        s = s[:-6]
    return np.datetime64(s, "ns")


def _band_assets(rows) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows:
        for k, a in (r.get("assets") or {}).items():
            if a and a.get("raster:bands"):
                seen.setdefault(k, None)
    return list(seen)


def _field(r, asset, name):
    """A projection field from the asset, falling back to the item."""
    a = (r.get("assets") or {}).get(asset) or {}
    v = a.get(name)
    return v if v is not None else r.get(name)


def _common_grid(rows, keys):
    grids = set()
    for r in rows:
        for k in keys:
            if not (r.get("assets") or {}).get(k):
                continue
            shape = _field(r, k, "proj:shape")
            tr = _field(r, k, "proj:transform")
            if shape is None or tr is None:
                raise VrtWriterError(
                    f"item {r.get('id')!r} asset {k!r} lacks proj:shape / "
                    "proj:transform; the VRT cannot be built from the "
                    "catalogue alone."
                )
            grids.add((tuple(int(v) for v in shape),
                       tuple(round(float(v), 12) for v in tr[:6])))
    if len(grids) != 1:
        raise VrtWriterError(
            f"items are on {len(grids)} different grids (identical grid or "
            f"refuse): {sorted(grids)[:3]}"
        )
    shape, tr = grids.pop()
    return shape, tr


_STORAGE = ("data_type", "nodata", "scale", "offset")


def _common_band(rows, k) -> dict[str, Any]:
    """One storage description per asset.

    Storage fields (dtype, nodata, scale, offset) must agree across items:
    they decide how bytes decode. The unit is advisory: items that omit it
    take the unit other items give, and conflicting non-empty units warn and
    use the most common.
    """
    storage = set()
    units: dict[str, int] = {}
    first = None
    for r in rows:
        a = (r.get("assets") or {}).get(k)
        if not a:
            continue
        b = (a.get("raster:bands") or [{}])[0]
        st = tuple(b.get(f) for f in _STORAGE)
        storage.add(st)
        if b.get("unit"):
            units[b["unit"]] = units.get(b["unit"], 0) + 1
        first = first or dict(zip(_STORAGE, st))
    if len(storage) > 1:
        raise VrtWriterError(
            f"asset {k!r}: raster:bands storage ({', '.join(_STORAGE)}) differs "
            f"between items: {sorted(storage, key=str)[:3]}; one array needs one "
            "storage description."
        )
    out = dict(first or {})
    if units:
        unit = max(units, key=units.get)
        if len(units) > 1:
            import warnings
            warnings.warn(
                f"asset {k!r}: items give different units {sorted(units)}; "
                f"using {unit!r}",
                UserWarning, stacklevel=3,
            )
        out["unit"] = unit
    return out


def _common_crs_wkt(rows, keys) -> str:
    found = set()
    for r in rows[:: max(1, len(rows) // 50)] + rows[-1:]:
        for k in keys:
            if not (r.get("assets") or {}).get(k):
                continue
            for name in ("proj:wkt2", "proj:code", "proj:epsg"):
                v = _field(r, k, name)
                if v is not None:
                    found.add((name, str(v)))
                    break
    if not found:
        raise VrtWriterError("items carry no proj:epsg / proj:code / proj:wkt2; pass crs=")
    if len({v for _, v in found}) > 1:
        raise VrtWriterError(f"items declare different CRSs: {sorted(found)}")
    name, v = found.pop()
    if name == "proj:wkt2":
        return v
    code = v if ":" in v else f"EPSG:{v}"
    return _code_to_wkt(code)


def _code_to_wkt(code: str) -> str:
    try:
        import pyproj
        return pyproj.CRS.from_user_input(code).to_wkt()
    except ImportError:
        pass
    try:
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.SetFromUserInput(code)
        return srs.ExportToWkt(["FORMAT=WKT2_2019"])
    except ImportError:
        pass
    raise VrtWriterError(
        f"{code} needs pyproj (or GDAL) to become WKT, which the axis mapping "
        "depends on; install pyproj or pass crs=<WKT>."
    )
