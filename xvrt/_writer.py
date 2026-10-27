"""Main writer.

Public surface: :func:`write_mdim_vrt` and :func:`encode_datetime_coord`.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import xarray as xr

from . import _xml as X
from ._coords import CoordMode, detect_regular, resolve_mode
from ._dtype import numpy_to_vrt_dtype
from ._errors import VrtWriterError
from ._sources import Source, normalise_sources


Composition = Literal["stack", "concat", "mosaic"]


# -- public API -------------------------------------------------------------

def write_mdim_vrt(
    ds: xr.Dataset,
    sources: Sequence[Any],
    path: str | Path,
    *,
    composition: Composition,
    concat_dim: str | None = None,
    crs: str | None = None,
    sizes: Sequence[int] | None = None,
    block_size: Sequence[int] | Mapping[str, Sequence[int]] | None = None,
    coord_mode: CoordMode | Mapping[str, CoordMode] = "auto",
    inline_threshold: int = 10_000,
    require_crs: bool = False,
) -> Path:
    """Emit a GDAL multidim VRT from an xarray Dataset.

    Parameters
    ----------
    ds
        The Dataset whose shape / dims / coords / data-vars describe the
        desired virtual composition.
    sources
        ``list[str | Path | dict]``.
    path
        Output ``.vrt`` path.
    composition
        ``"stack"`` | ``"concat"`` | ``"mosaic"``. **v0 implements stack
        and concat**; mosaic raises ``NotImplementedError``.
    concat_dim
        Required for stack (new dim name) and concat (existing dim name).
    crs
        WKT string. If omitted, we look at ``ds[dv].attrs["crs_wkt"]`` /
        ``ds[dv].rio.crs`` / CF ``grid_mapping``; if still absent the
        writer emits no ``<SRS>`` and warns. Pass ``require_crs=True`` for
        strict R4 behaviour.
    sizes
        Convenience shortcut for concat: ``list[int]`` of per-source sizes
        along ``concat_dim``. If not given, we try (in order):
        per-source ``size`` in dict-form sources, ``ds.chunks[concat_dim]``
        when the number of chunks matches the number of sources, then
        opening each source to read ``.sizes[concat_dim]``.
    block_size
        Sequence matching array dim order, or dict keyed by data-var name.
        ``None`` falls back to ``ds[dv].chunks`` when present.
    coord_mode
        ``"auto"`` (default) | ``"inline"`` | ``"source"`` or dict per coord.
    inline_threshold
        In auto mode, coords with ``size <= threshold`` go inline.
    require_crs
        When True, raise :class:`VrtWriterError` if no CRS can be
        determined. Default False (warn-and-omit); matches the reality of
        ocean NetCDFs and other CRS-less sources.
    """
    if composition not in ("stack", "concat", "mosaic"):
        raise VrtWriterError(
            f"composition: expected 'stack'|'concat'|'mosaic', got {composition!r}"
        )
    if composition == "mosaic":
        raise NotImplementedError(
            "mosaic composition is v1 scope — not implemented yet."
        )
    if composition in ("stack", "concat") and concat_dim is None:
        raise VrtWriterError(
            f"composition={composition!r} requires concat_dim="
        )

    path = Path(path)

    # Normalise sources. For stack, the new dim is concat_dim and each source
    # contributes one element; for concat it exists already on ds.
    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        raise VrtWriterError("ds has no data_vars")
    if len(data_vars) > 1:
        # v0 keeps it simple — one data_var per call. Multi-var would need
        # per-var source dispatch; punt to v0.5.
        raise NotImplementedError(
            f"ds has multiple data_vars ({data_vars}); v0 supports a single "
            "data_var per write call. Split the Dataset with ds[[var]] and "
            "write once per variable."
        )
    (data_var,) = data_vars

    # Early concat_dim validation with a useful hint. Catches case mismatches
    # ("time" vs "Time") before size resolution walks off looking for a
    # nonexistent dim on every source.
    if composition in ("stack", "concat"):
        available = tuple(ds[data_var].dims)
        if concat_dim not in available:
            raise VrtWriterError(
                f"concat_dim={concat_dim!r} is not a dim of "
                f"ds[{data_var!r}]. Available dims: {available}."
            )

    sources_norm = normalise_sources(
        sources,
        default_array=data_var,
        vrt_path=path,
        composition=composition,
    )

    # Resolve per-source sizes for concat (auto-discover before checking).
    if composition == "concat":
        sources_norm = _resolve_concat_sizes(
            ds=ds, data_var=data_var, concat_dim=concat_dim,
            sources=sources_norm, explicit=sizes,
        )

    # R7: array-name disjointness.
    _check_array_name_collisions(ds)

    # R4: CRS — strict or soft.
    crs_wkt = _resolve_crs(ds, data_var, crs, require=require_crs)

    # Composition-specific dim / size arithmetic.
    if composition == "stack":
        _check_stack(ds, data_var, sources_norm, concat_dim)
        sources_norm = _resolve_stack_has_dim(sources_norm, concat_dim)
    elif composition == "concat":
        _check_concat(ds, data_var, sources_norm, concat_dim)

    # Build XML.
    root = _build_vrt(
        ds=ds,
        data_var=data_var,
        sources=sources_norm,
        composition=composition,
        concat_dim=concat_dim,
        crs_wkt=crs_wkt,
        block_size=block_size,
        coord_mode=coord_mode,
        inline_threshold=inline_threshold,
    )

    xml_bytes = X.tostring(root)
    path.write_bytes(xml_bytes)
    return path


def encode_datetime_coord(
    da: xr.DataArray,
    *,
    unit: Literal["seconds", "minutes", "hours", "days"] = "days",
    since: str = "1970-01-01",
) -> xr.DataArray:
    """Convert a datetime64 coord DataArray to numeric + CF ``units`` attr.

    Handy pre-step when you have a ``datetime64[ns]`` time coord and want to
    feed it to the writer (which refuses datetime dtypes per v0 policy).
    """
    if da.dtype.kind != "M":
        raise VrtWriterError(f"encode_datetime_coord: {da.name} is not datetime64")
    origin = np.datetime64(since)
    delta = (da.values - origin)
    div = {"seconds": np.timedelta64(1, "s"),
           "minutes": np.timedelta64(1, "m"),
           "hours":   np.timedelta64(1, "h"),
           "days":    np.timedelta64(1, "D")}[unit]
    numeric = delta / div
    out = xr.DataArray(numeric.astype(np.float64), dims=da.dims, coords=da.coords,
                       attrs={**da.attrs, "units": f"{unit} since {since}"},
                       name=da.name)
    return out


# -- internals --------------------------------------------------------------

def _check_array_name_collisions(ds: xr.Dataset) -> None:
    """R7: the VRT ``<Group>`` has a flat namespace across Dimension and
    Array elements. xarray allows a dim name to coincide with a data-var
    name (e.g. ``xr.DataArray(..., dims=("foo",), name="foo")``); this is
    legal in xarray but emits two children with the same name in the VRT.

    Dim-coords (where ``dim_name == coord_name``) are the normal case and
    are NOT a collision — they're exactly how indexing variables are
    expressed in the XSD. We only flag real duplicates.
    """
    dims = set(ds.dims)
    coords = set(ds.coords)
    data_vars = set(ds.data_vars)

    # Non-dim coord names must not collide with data-var names.
    non_dim_coords = coords - dims
    cc = non_dim_coords & data_vars
    if cc:
        raise VrtWriterError(
            f"array-name collision (R7): {sorted(cc)} appear as both coord "
            "and data_var. VRT requires a flat namespace."
        )
    # Dim names must not collide with data-var names (unless it's a dim-coord).
    dc = (dims & data_vars) - coords
    if dc:
        raise VrtWriterError(
            f"array-name collision (R7): {sorted(dc)} appear as both dim "
            "and data_var without being a dim-coord. VRT requires a flat "
            "namespace."
        )


def _resolve_crs(
    ds: xr.Dataset,
    data_var: str,
    crs_arg: str | None,
    *,
    require: bool = False,
) -> str | None:
    if crs_arg is not None:
        return crs_arg

    # Look on the data_var.
    da = ds[data_var]
    for key in ("crs_wkt", "spatial_ref", "esri_pe_string"):
        v = da.attrs.get(key)
        if isinstance(v, str) and v.strip():
            return v

    # Try rio accessor if available — non-mandatory dep.
    try:
        import rioxarray  # noqa: F401
        maybe = da.rio.crs
        if maybe is not None:
            return maybe.to_wkt()
    except Exception:
        pass

    # Look at any coord that looks like a grid_mapping pointer (CF convention).
    gm = da.attrs.get("grid_mapping")
    if gm and gm in ds.coords:
        return _resolve_crs_coord(ds[gm])
    if gm and gm in ds.variables:
        return _resolve_crs_coord(ds[gm])

    if require:
        raise VrtWriterError(
            "CRS not supplied and not detected on ds. Pass crs= explicitly "
            "(WKT) or attach crs_wkt / grid_mapping per CF conventions. "
            "(require_crs=True mode.)"
        )

    import warnings
    warnings.warn(
        "CRS not supplied and not detected; emitting VRT without <SRS>. "
        "Pass crs=<WKT> to attach one, or require_crs=True to make this an "
        "error.",
        UserWarning,
        stacklevel=3,
    )
    return None


def _resolve_crs_coord(coord: xr.DataArray) -> str | None:
    for key in ("crs_wkt", "spatial_ref"):
        v = coord.attrs.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _check_stack(
    ds: xr.Dataset,
    data_var: str,
    sources: list[Source],
    new_dim: str,
) -> None:
    da = ds[data_var]
    if new_dim not in da.dims:
        raise VrtWriterError(
            f"composition='stack': new dim {new_dim!r} must be present in "
            f"ds[{data_var!r}].dims — it isn't. dims are {da.dims}."
        )
    if new_dim not in ds.coords:
        raise VrtWriterError(
            f"composition='stack': coord for new dim {new_dim!r} is required. "
            "Assign it via ds = ds.assign_coords({...})."
        )
    n_sources = len(sources)
    n_along = ds.sizes[new_dim]
    if n_sources != n_along:
        raise VrtWriterError(
            f"composition='stack': {n_sources} sources provided but "
            f"ds.sizes[{new_dim!r}] = {n_along}. One source per element."
        )


def _resolve_stack_has_dim(sources: list[Source], new_dim: str) -> list[Source]:
    """Record, per stack source, whether it already carries ``new_dim``.

    "Stack" covers two source shapes: 2D slices that lack the stack dim
    (they need a ``[newaxis,...]`` view), and files that already carry it
    with length 1 (one-time-step-per-file NetCDF; adding an axis would make
    the source rank too high). Dict-form sources may say so with
    ``has_dim=True|False``; otherwise the first source is opened (metadata
    only) and its answer applies to all sources without an explicit value.
    """
    from dataclasses import replace

    if all("has_dim" in s.extra for s in sources):
        return sources
    probe = next(s for s in sources if "has_dim" not in s.extra)
    has = _source_has_dim(probe.original or probe.path, probe.array, new_dim)
    return [s if "has_dim" in s.extra
            else replace(s, extra={**s.extra, "has_dim": has})
            for s in sources]


def _source_has_dim(target: str, array: str, dim: str) -> bool:
    """Does ``array`` in ``target`` have dimension ``dim``? Metadata only.

    Unopenable sources (e.g. GeoTIFFs xarray cannot read without rioxarray)
    are treated as lacking the dim, which is right for classic 2D rasters.
    """
    import xarray as _xr

    is_url = target.startswith(("http://", "https://"))
    engines: list[str | None] = ["h5netcdf", None] if is_url else [None, "h5netcdf"]
    name = array.lstrip("/")
    for engine in engines:
        try:
            kw: dict[str, Any] = {"decode_times": False, "decode_cf": False}
            if engine is not None:
                kw["engine"] = engine
            with _xr.open_dataset(target, **kw) as sds:
                if name in sds.variables:
                    return dim in sds[name].dims
                return dim in sds.dims
        except Exception:
            continue
    return False


def _resolve_concat_sizes(
    *,
    ds: xr.Dataset,
    data_var: str,
    concat_dim: str,
    sources: list[Source],
    explicit: Sequence[int] | None,
) -> list[Source]:
    """Fill in ``Source.size`` along the concat dim by any means available.

    Priority:
      1. ``explicit`` — the ``sizes=`` kwarg, one int per source
      2. ``Source.size`` already set (dict-form input)
      3. ``ds[data_var].chunks[concat_axis]`` if #chunks == #sources
         (common case for ``open_mfdataset`` default, one chunk per file)
      4. open each source path and read ``.sizes[concat_dim]``

    Returns a new list of Source records with ``size`` populated.
    """
    n = len(sources)

    # 1. Explicit kwarg wins outright.
    if explicit is not None:
        if len(explicit) != n:
            raise VrtWriterError(
                f"sizes= has length {len(explicit)}, expected {n} "
                "(one per source)."
            )
        return [
            Source(s.path, s.array, s.relative_to_vrt, int(sz), s.extra, s.original)
            for s, sz in zip(sources, explicit)
        ]

    # 2. Already present on every source? Nothing to do.
    if all(s.size is not None for s in sources):
        return list(sources)

    resolved_sizes: list[int | None] = [s.size for s in sources]

    # 3. Try ds chunks along the concat axis.
    da = ds[data_var]
    axis = da.dims.index(concat_dim) if concat_dim in da.dims else None
    if axis is not None:
        chunks = da.chunks  # tuple-of-tuples, or None
        if chunks is not None:
            per_axis = chunks[axis]
            if len(per_axis) == n:
                for i, cs in enumerate(per_axis):
                    if resolved_sizes[i] is None:
                        resolved_sizes[i] = int(cs)

    # 3.5. Uniform guess: if the total divides evenly by n_sources and we
    # still don't know sizes, assume each source contributes total/n along
    # concat_dim. Fires for daily-files-one-timestep-each patterns (OISST
    # etc). Warns — this is a guess, not a fact.
    if any(sz is None for sz in resolved_sizes) and axis is not None:
        total = int(ds.sizes[concat_dim])
        if total % n == 0:
            uniform = total // n
            # Only apply to slots that are still None.
            applied = False
            for i in range(n):
                if resolved_sizes[i] is None:
                    resolved_sizes[i] = uniform
                    applied = True
            if applied:
                import warnings
                warnings.warn(
                    f"assuming {uniform} element(s) per source along "
                    f"{concat_dim!r} ({total} total / {n} sources). This is "
                    "a guess based on even divisibility; pass sizes=... if "
                    "sources contribute unequally.",
                    UserWarning, stacklevel=3,
                )

    # 4. Fallback: open each source without a size, read its concat dim.
    if any(sz is None for sz in resolved_sizes):
        # Lazy import — only pay the cost when we need to.
        import xarray as _xr  # local alias clarifies this is a fallback path
        for i, (src, sz) in enumerate(zip(sources, resolved_sizes)):
            if sz is not None:
                continue
            # Use original user-supplied path for opening, not the
            # possibly-rewritten one in src.path.
            open_target = src.original or src.path
            resolved_sizes[i] = _open_and_read_size(open_target, concat_dim, i)

    return [
        Source(s.path, s.array, s.relative_to_vrt, int(sz), s.extra, s.original)
        for s, sz in zip(sources, resolved_sizes)
    ]


def _open_and_read_size(target: str, concat_dim: str, index: int) -> int:
    """Open a source with whichever engine can read it, return concat size.

    URLs (http/https) need ``h5netcdf`` (fsspec-capable); the default
    ``netCDF4`` engine chokes on them. Local files work with either.
    Tries a small ordered list of engines before giving up.
    """
    import xarray as _xr

    is_url = target.startswith("http://") or target.startswith("https://")
    engines: list[str | None] = ["h5netcdf", None] if is_url else [None, "h5netcdf"]

    errors: list[str] = []
    for engine in engines:
        try:
            open_kw = {"decode_times": False}
            if engine is not None:
                open_kw["engine"] = engine
            with _xr.open_dataset(target, **open_kw) as sds:
                if concat_dim not in sds.sizes:
                    raise VrtWriterError(
                        f"source #{index} ({target}): concat dim "
                        f"{concat_dim!r} not found. Available dims in this "
                        f"source: {tuple(sds.sizes)}. Pass sizes= or "
                        "dict-form sources with size= to override."
                    )
                return int(sds.sizes[concat_dim])
        except VrtWriterError:
            raise
        except Exception as e:  # keep trying other engines
            errors.append(f"{engine or 'default'}: {type(e).__name__}: {e}")

    raise VrtWriterError(
        f"source #{index} ({target}): could not open to read size along "
        f"{concat_dim!r}. Tried engines: " + " | ".join(errors) +
        ". Pass sizes= (list[int]) or dict-form sources with size= to "
        "supply it directly."
    )


def _check_concat(
    ds: xr.Dataset,
    data_var: str,
    sources: list[Source],
    dim: str,
) -> None:
    da = ds[data_var]
    if dim not in da.dims:
        raise VrtWriterError(
            f"composition='concat': concat_dim {dim!r} must be a dim of "
            f"ds[{data_var!r}]. dims are {da.dims}."
        )
    # Each source must declare its size along the concat dim.
    missing = [i for i, s in enumerate(sources) if s.size is None]
    if missing:
        raise VrtWriterError(
            f"composition='concat': sources {missing} missing 'size'. Pass "
            "sources as list[dict] with {'path': ..., 'size': N} for each."
        )
    total = sum(s.size for s in sources)
    n_along = ds.sizes[dim]
    if total != n_along:
        raise VrtWriterError(
            f"composition='concat': source sizes along {dim!r} sum to "
            f"{total}, but ds.sizes[{dim!r}] = {n_along}."
        )
    # R5: monotonicity of concat-dim coord.
    if dim in ds.coords:
        vals = np.asarray(ds[dim].values)
        if vals.dtype.kind in "fiu":
            diffs = np.diff(vals.astype(np.float64))
            if not (np.all(diffs > 0) or np.all(diffs < 0)):
                raise VrtWriterError(
                    f"composition='concat': coord {dim!r} is not strictly "
                    "monotonic across concatenated sources (R5)."
                )


# -- build the element tree -------------------------------------------------

def _build_vrt(
    *,
    ds: xr.Dataset,
    data_var: str,
    sources: list[Source],
    composition: Composition,
    concat_dim: str | None,
    crs_wkt: str | None,
    block_size: Sequence[int] | Mapping[str, Sequence[int]] | None,
    coord_mode: CoordMode | Mapping[str, CoordMode],
    inline_threshold: int,
) -> ET.Element:
    da = ds[data_var]
    dims = list(da.dims)

    root = ET.Element("VRTDataset")
    grp = X.sub(root, "Group", name="/")

    # <Dimension> elements — in the order the data-var dims appear.
    for d in dims:
        size = ds.sizes[d]
        attrs: dict[str, Any] = {"name": d, "size": size}
        if d in ds.coords:
            attrs["indexingVariable"] = d
        dim_type, dim_dir = _axis_role(ds, d)
        if dim_type is not None:
            attrs["type"] = dim_type
        if dim_dir is not None:
            attrs["direction"] = dim_dir
        X.sub(grp, "Dimension", **attrs)

    # Dataset-level <Attribute>s (ds.attrs).
    _emit_attrs(grp, ds.attrs)

    # Coord <Array>s — one per 1D coord that matches a dim.
    for d in dims:
        if d not in ds.coords:
            continue
        _emit_coord_array(
            grp, ds.coords[d], dim=d,
            mode=coord_mode, threshold=inline_threshold, sources=sources,
            composition=composition, concat_dim=concat_dim,
        )

    # Data-var <Array>.
    _emit_data_array(
        grp,
        da=da,
        name=data_var,
        sources=sources,
        composition=composition,
        concat_dim=concat_dim,
        crs_wkt=crs_wkt,
        block_size=_resolve_block_size(da, block_size, data_var),
    )

    return root


def _axis_role(ds: xr.Dataset, dim: str) -> tuple[str | None, str | None]:
    """Return (type, direction) for <Dimension> when detectable from CF attrs.

    Conservative: only emit type/direction when we can justify it. Users can
    override by hand-editing or by attaching CF ``axis``/``standard_name``
    attrs to the coord.
    """
    if dim not in ds.coords:
        return (None, None)
    attrs = ds.coords[dim].attrs
    axis = attrs.get("axis", "").upper()
    stdname = attrs.get("standard_name", "").lower()
    if axis == "T" or "time" in stdname:
        return ("TEMPORAL", None)
    if axis == "Z" or stdname in {"depth", "altitude", "height"}:
        direction = None
        if stdname == "depth":
            direction = "DOWN"
        elif stdname in {"altitude", "height"}:
            direction = "UP"
        return ("VERTICAL", direction)
    if axis == "X" or stdname in {"longitude", "projection_x_coordinate"}:
        return ("HORIZONTAL_X", "EAST")
    if axis == "Y" or stdname in {"latitude", "projection_y_coordinate"}:
        return ("HORIZONTAL_Y", "NORTH")
    return (None, None)


def _emit_attrs(parent: ET.Element, attrs: Mapping[str, Any]) -> None:
    """Emit xarray attrs as <Attribute> children.

    Per the plan's Open Questions: warn and filter on unsupported types
    rather than raise. Here we silently drop attrs whose values aren't
    representable — a future revision should surface a warning list.
    """
    for name, val in attrs.items():
        dt, items = _attr_dtype_and_values(val)
        if dt is None:
            continue  # TODO: accumulate filtered names and emit one warning.
        a = X.sub(parent, "Attribute", name=name)
        X.sub(a, "DataType", text=dt)
        for v in items:
            X.sub(a, "Value", text=_format_attr_value(v))


def _attr_dtype_and_values(val: Any) -> tuple[str | None, list[Any]]:
    if isinstance(val, str):
        return ("String", [val])
    if isinstance(val, (bool, np.bool_)):
        return ("Int32", [int(val)])
    if isinstance(val, (int, np.integer)):
        return ("Int64", [int(val)])
    if isinstance(val, (float, np.floating)):
        return ("Float64", [float(val)])
    if isinstance(val, np.ndarray):
        if val.dtype.kind in "iu":
            return ("Int64", val.tolist())
        if val.dtype.kind == "f":
            return ("Float64", val.tolist())
        if val.dtype.kind in ("U", "S"):
            return ("String", [str(v) for v in val.tolist()])
    if isinstance(val, (list, tuple)) and val:
        # Delegate by element type of first item.
        try:
            arr = np.asarray(val)
            return _attr_dtype_and_values(arr)
        except Exception:
            return (None, [])
    return (None, [])


def _format_attr_value(v: Any) -> str:
    if isinstance(v, float):
        return X.repr_double(v)
    return str(v)


def _reencode_datetime_coord(da: xr.DataArray) -> xr.DataArray:
    """Convert a datetime64 coord back to numeric with a CF ``units`` attr.

    xarray stashes the original CF ``units`` string in ``da.encoding``
    when it decodes on open (e.g. ``"seconds since 1978-01-01"``). We use
    that to round-trip faithfully so reading the VRT reproduces the same
    timestamps. If ``encoding["units"]`` is absent we default to
    ``"days since 1970-01-01"`` — a benign choice since the numeric values
    still represent the same moments in time.
    """
    from xarray.coding.times import encode_cf_datetime

    units = da.encoding.get("units") or da.attrs.get("units")
    if not isinstance(units, str) or "since" not in units:
        units = "days since 1970-01-01"
    calendar = da.encoding.get("calendar") or da.attrs.get("calendar")

    # xarray's own CF encoder: honours a time-of-day in the origin
    # ("seconds since 1981-01-01 12:00:00") and non-standard calendars
    # (cftime objects), both of which a hand-rolled numpy subtraction gets
    # wrong.
    try:
        numeric, _units_out, cal_out = encode_cf_datetime(
            da.values, units=units, calendar=calendar
        )
    except Exception:
        numeric, units, cal_out = encode_cf_datetime(da.values)

    attrs = {k: v for k, v in da.attrs.items() if k != "calendar"}
    attrs["units"] = units
    if calendar or cal_out not in ("standard", "proleptic_gregorian", "gregorian"):
        attrs["calendar"] = calendar or cal_out
    return xr.DataArray(np.asarray(numeric).astype(np.float64), dims=da.dims,
                        attrs=attrs, name=da.name)


def _is_datetime_like(da: xr.DataArray) -> bool:
    """datetime64, or an object array of cftime datetimes."""
    if da.dtype.kind == "M":
        return True
    if da.dtype.kind == "O" and da.size:
        first = np.asarray(da.values).ravel()[0]
        return hasattr(first, "calendar") and hasattr(first, "timetuple")
    return False


def _emit_coord_array(
    parent: ET.Element,
    da: xr.DataArray,
    *,
    dim: str,
    mode: CoordMode | Mapping[str, CoordMode],
    threshold: int,
    sources: list[Source],
    composition: Composition = "concat",
    concat_dim: str | None = None,
) -> None:
    # Auto-encode datetime coords to numeric + CF units. xarray stashes
    # the original "units" string in encoding when it decodes times on open;
    # we round-trip back through it where possible so the VRT reads back to
    # the same timestamps. Fallback origin is the unix epoch.
    if _is_datetime_like(da):
        da = _reencode_datetime_coord(da)

    arr = X.sub(parent, "Array", name=dim)
    vrt_dtype = numpy_to_vrt_dtype(da.dtype)
    X.sub(arr, "DataType", text=vrt_dtype)
    X.sub(arr, "DimensionRef", ref=dim)

    # <Unit> from CF units attr. Emitted before value carrier per XSD.
    units = da.attrs.get("units")
    if isinstance(units, str) and units.strip():
        X.sub(arr, "Unit", text=units)

    # Regular-spacing detection comes first, independent of coord_mode.
    values = np.asarray(da.values)
    reg = detect_regular(values) if values.dtype.kind in "fiu" else None
    if reg is not None:
        start, inc = reg
        X.sub(arr, "RegularlySpacedValues",
              start=X.repr_double(start),
              increment=X.repr_double(inc))
        _emit_attrs(arr, _filter_axis_attrs(da.attrs))
        return

    is_comp_dim = concat_dim is not None and dim == concat_dim
    explicit = mode.get(dim) if isinstance(mode, Mapping) else mode
    if is_comp_dim and explicit in (None, "auto"):
        # The composition dim's values are spread across every source (or,
        # for stack, exist in none of them). Auto mode never points at a
        # single source file for it.
        chosen = "inline"
    else:
        chosen = resolve_mode(dim, values, mode, threshold=threshold)

    if chosen == "source" and is_comp_dim:
        if composition == "stack":
            raise VrtWriterError(
                f"coord_mode 'source' for the stack dim {dim!r}: stacked "
                "sources do not carry this coord. Use 'inline' (or 'auto')."
            )
        # concat: one <Source> per file, slabbed along the coord's only dim.
        # Each file's own coord values are read as stored, so this is only
        # correct when every file encodes the coord with the same units.
        offset = 0
        for src in sources:
            s = X.sub(arr, "Source")
            X.sub(s, "SourceFilename", text=src.path,
                  relativeToVRT=src.relative_to_vrt)
            X.sub(s, "SourceArray", text=_full_array_name(dim))
            X.sub(s, "DestSlab", offset=str(offset))
            offset += int(src.size)  # type: ignore[arg-type]
        _emit_attrs(arr, _filter_axis_attrs(da.attrs))
        return

    if chosen == "inline":
        ivv = X.sub(arr, "InlineValuesWithValueElement")
        if values.dtype.kind == "f":
            for v in values.tolist():
                X.sub(ivv, "Value", text=X.repr_double(v))
        else:
            for v in values.tolist():
                X.sub(ivv, "Value", text=str(v))
    elif chosen == "source":
        # Point at the first source — caller is promising the coord is there
        # and identical across sources (R3 ensures consistency for spatial
        # dims in stack/concat; for the concat dim we inline always).
        src = sources[0]
        s = X.sub(arr, "Source")
        X.sub(s, "SourceFilename", text=src.path,
              relativeToVRT=src.relative_to_vrt)
        X.sub(s, "SourceArray", text=_full_array_name(dim))
    else:
        raise AssertionError(f"unreachable coord mode: {chosen}")

    _emit_attrs(arr, _filter_axis_attrs(da.attrs))


def _filter_axis_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Strip attrs that are already represented as XML attributes / elements.

    ``scale_factor`` / ``add_offset`` are materialised as ``<Scale>`` /
    ``<Offset>``; repeating them as ``<Attribute>`` invites readers that
    apply CF decoding from attributes to unpack twice.
    """
    return {k: v for k, v in attrs.items()
            if k not in {"units", "axis", "standard_name",
                         "_FillValue", "missing_value",
                         "scale_factor", "add_offset"}}


def _storage_packing(da: xr.DataArray) -> dict[str, Any]:
    """On-disk dtype, scale, offset and fill for a data variable.

    The VRT reads the *source* bytes: GDAL's mdim VRT does not apply xarray's
    CF decoding. When xarray has decoded a variable (the default
    ``mask_and_scale=True``), the in-memory dtype is float and the packing
    keys have moved from ``attrs`` into ``encoding``. Declaring the decoded
    dtype with no ``<Scale>`` would hand back raw packed integers as if they
    were physical values, so we recover the storage description from
    ``encoding`` and let GDAL's Scale/Offset/NoData carry the decoding.

    ``attrs`` win over ``encoding`` (``mask_and_scale=False`` keeps them in
    attrs and the dtype is already the storage dtype).
    """
    attrs, enc = da.attrs, da.encoding

    def pick(*keys):
        for k in keys:
            if attrs.get(k) is not None:
                return attrs[k]
        for k in keys:
            if enc.get(k) is not None:
                return enc[k]
        return None

    dtype = da.dtype
    enc_dtype = enc.get("dtype")
    if enc_dtype is not None:
        try:
            ed = np.dtype(enc_dtype)
        except TypeError:
            ed = None
        if ed is not None and ed.kind in "biuf" and da.dtype.kind in "biuf":
            dtype = ed

    return {
        "dtype": dtype,
        "fill": pick("_FillValue", "missing_value"),
        "scale": pick("scale_factor"),
        "offset": pick("add_offset"),
    }


def _full_array_name(name: str) -> str:
    """Return the full-name form GDAL uses inside <SourceArray>: leading slash."""
    return name if name.startswith("/") else f"/{name}"


def _emit_data_array(
    parent: ET.Element,
    *,
    da: xr.DataArray,
    name: str,
    sources: list[Source],
    composition: Composition,
    concat_dim: str | None,
    crs_wkt: str | None,
    block_size: list[int] | None,
) -> None:
    packing = _storage_packing(da)

    arr = X.sub(parent, "Array", name=name)
    X.sub(arr, "DataType", text=numpy_to_vrt_dtype(packing["dtype"]))
    for d in da.dims:
        X.sub(arr, "DimensionRef", ref=d)

    if block_size is not None:
        X.sub(arr, "BlockSize", text=X.comma_list(block_size))

    if crs_wkt:
        # Spatial axis mapping: which array dim carries each SRS axis.
        mapping = _axis_mapping(da, crs_wkt)
        X.sub(arr, "SRS", text=crs_wkt, dataAxisToSRSAxisMapping=mapping)

    # <Unit> / <NoDataValue> / <Offset> / <Scale> from attrs.
    units = da.attrs.get("units")
    if isinstance(units, str) and units.strip():
        X.sub(arr, "Unit", text=units)
    fill = packing["fill"]
    if fill is not None:
        try:
            fv = float(np.asarray(fill).ravel()[0])
            if np.isfinite(fv):
                X.sub(arr, "NoDataValue", text=X.repr_double(fv))
            else:
                # nan/inf — XSD allows the literal "nan"/"NAN".
                X.sub(arr, "NoDataValue", text="nan" if np.isnan(fv) else str(fv))
        except (TypeError, ValueError, IndexError):
            pass  # non-numeric fill — skip rather than emit garbage
    off = packing["offset"]
    if off is not None:
        X.sub(arr, "Offset", text=X.repr_double(float(off)))
    scl = packing["scale"]
    if scl is not None:
        X.sub(arr, "Scale", text=X.repr_double(float(scl)))

    # <Source> composition.
    _emit_sources(
        arr, sources=sources, da=da,
        composition=composition, concat_dim=concat_dim,
    )

    # Remaining user attrs (strip those already materialised).
    _emit_attrs(arr, _filter_axis_attrs(da.attrs))


def _axis_mapping(da: xr.DataArray, crs_wkt: str | None = None) -> str | None:
    """Return the dataAxisToSRSAxisMapping string, or None if unclear.

    GDAL semantics: entry *i* is the 1-based index of the **array dimension**
    that carries SRS axis *i*. Its length is the number of SRS axes, and it
    indexes over all array dims (time and depth included). So a geographic
    CRS with WKT axis order (lat, lon) on an array ``(time, lat, lon)`` gives
    ``"2,3"``; with (lon, lat) / traditional GIS order it gives ``"3,2"``.

    SRS axis order is read from the WKT's ``AXIS[...]`` directions; WKT
    without AXIS defaults to easting-first, as GDAL does for WKT1.
    """
    tags = [_dim_axis_tag(da, d) for d in da.dims]
    if "Y" not in tags or "X" not in tags:
        return None
    order = _srs_axis_order(crs_wkt) if crs_wkt else None
    if order is None:
        order = ["X", "Y"]
    return ",".join(str(tags.index(t) + 1) for t in order)


_Y_NAMES = {"lat", "latitude", "y", "nav_lat", "rlat", "yc"}
_X_NAMES = {"lon", "longitude", "x", "nav_lon", "rlon", "xc"}


def _dim_axis_tag(da: xr.DataArray, d: str) -> str | None:
    coord = da.coords.get(d)
    if coord is not None:
        a = str(coord.attrs.get("axis", "")).upper()
        sn = str(coord.attrs.get("standard_name", "")).lower()
        if a == "Y" or sn in {"latitude", "projection_y_coordinate", "grid_latitude"}:
            return "Y"
        if a == "X" or sn in {"longitude", "projection_x_coordinate", "grid_longitude"}:
            return "X"
        if a in ("T", "Z") or "time" in sn:
            return a or "T"
    low = str(d).lower()
    if low in _Y_NAMES:
        return "Y"
    if low in _X_NAMES:
        return "X"
    return None


def _srs_axis_order(wkt: str) -> list[str] | None:
    """Horizontal axis order of a 2D CRS from its WKT, e.g. ``["Y", "X"]``.

    Takes the last two ``AXIS[...]`` entries: in WKT1 a PROJCS may repeat its
    GEOGCS axes first, and in both WKT1 and WKT2 the CRS's own axes come
    last. Returns None for anything that is not plainly two horizontal axes
    (compound CRS etc.), letting the caller fall back.
    """
    import re as _re

    dirs = _re.findall(r'AXIS\[\s*"[^"]*"\s*,\s*([A-Za-z]+)', wkt)
    if len(dirs) < 2:
        return None
    if "COMPOUNDCRS" in wkt.upper() or "COMPD_CS" in wkt.upper():
        return None
    out = []
    for d in dirs[-2:]:
        d = d.lower()
        if d in ("north", "south"):
            out.append("Y")
        elif d in ("east", "west"):
            out.append("X")
        else:
            return None
    return out if sorted(out) == ["X", "Y"] else None


def _resolve_block_size(
    da: xr.DataArray,
    block_size: Sequence[int] | Mapping[str, Sequence[int]] | None,
    data_var: str,
) -> list[int] | None:
    if block_size is None:
        if da.chunks is None:
            return None
        # ds.chunks is a tuple-of-tuples per dim; use the first chunk of each.
        return [int(cs[0]) for cs in da.chunks]
    if isinstance(block_size, Mapping):
        bs = block_size.get(data_var)
        if bs is None:
            return None
        return list(map(int, bs))
    return list(map(int, block_size))


def _emit_sources(
    arr: ET.Element,
    *,
    sources: list[Source],
    da: xr.DataArray,
    composition: Composition,
    concat_dim: str | None,
) -> None:
    dims = list(da.dims)
    # Which axis participates in the composition.
    comp_axis = dims.index(concat_dim) if concat_dim else 0
    offset_so_far = 0
    for s in sources:
        size_along = 1 if composition == "stack" else int(s.size)  # type: ignore[arg-type]
        src_el = X.sub(arr, "Source")
        X.sub(src_el, "SourceFilename", text=s.path,
              relativeToVRT=s.relative_to_vrt)
        X.sub(src_el, "SourceArray", text=_full_array_name(s.array))
        if composition == "stack" and not s.extra.get("has_dim", False):
            # A stacked source lacks the new dim. GDAL refuses to read a
            # source whose rank differs from the destination ("Inconsistent
            # number of dimensions"), so insert a length-1 axis at the
            # stack position with a view.
            X.sub(src_el, "SourceView",
                  text="[" + ":," * comp_axis + "newaxis,...]")
        # DestSlab always emitted (plan rule — zeros included).
        offsets = [0] * len(dims)
        offsets[comp_axis] = offset_so_far
        X.sub(src_el, "DestSlab", offset=X.comma_list(offsets))
        offset_so_far += size_along
