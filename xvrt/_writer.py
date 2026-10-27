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
    variables: Sequence[str] | None = None,
) -> Path:
    """Emit a GDAL multidim VRT from an xarray Dataset.

    Every data variable becomes an ``<Array>`` in one VRT (OISST-style files
    with sst/anom/err/ice give one VRT with four arrays). Each source file is
    expected to hold every variable; variables that lack the composition dim
    (static fields such as a land mask) are read from the first source.

    Parameters
    ----------
    ds
        The Dataset whose shape / dims / coords / data-vars describe the
        desired virtual composition.
    sources
        ``list[str | Path | dict]``. Dict form: ``{"path": ..., "size": N,
        "array": "/name"}``; with several variables use ``"arrays": {var:
        "/name", ...}`` instead of ``"array"`` for any renames (variables
        not listed default to their own name).
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
    variables
        Subset of ``ds.data_vars`` to write, in order. Default: all data vars
        except CF ``grid_mapping`` containers (their content is carried by
        ``<SRS>``).
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

    data_vars = _select_data_vars(ds, variables)

    # Early concat_dim validation with a useful hint. Catches case mismatches
    # ("time" vs "Time") before size resolution walks off looking for a
    # nonexistent dim on every source.
    comp_vars = [v for v in data_vars if concat_dim in ds[v].dims]
    if not comp_vars:
        available = tuple(dict.fromkeys(d for v in data_vars for d in ds[v].dims))
        raise VrtWriterError(
            f"concat_dim={concat_dim!r} is not a dim of any data var "
            f"({data_vars}). Available dims: {available}."
        )
    lead = comp_vars[0]  # the variable used for size / rank inference

    sources_norm = normalise_sources(
        sources,
        default_array=lead,
        vrt_path=path,
        composition=composition,
        n_vars=len(data_vars),
    )

    # Resolve per-source sizes for concat (auto-discover before checking).
    if composition == "concat":
        sources_norm = _resolve_concat_sizes(
            ds=ds, data_var=lead, concat_dim=concat_dim,
            sources=sources_norm, explicit=sizes,
        )

    # R7: array-name disjointness.
    _check_array_name_collisions(ds)

    # R4: CRS, per variable; strict or soft.
    crs_by_var = _resolve_crs_all(ds, data_vars, crs, require=require_crs)

    # Composition-specific dim / size arithmetic.
    if composition == "stack":
        _check_stack(ds, lead, sources_norm, concat_dim)
        sources_norm = _resolve_stack_has_dim(
            sources_norm, concat_dim, _array_for(sources_norm[0], lead, data_vars)
        )
    elif composition == "concat":
        _check_concat(ds, lead, sources_norm, concat_dim)

    block_sizes = _resolve_block_sizes(ds, data_vars, block_size)

    # Build XML.
    root = _build_vrt(
        ds=ds,
        data_vars=data_vars,
        sources=sources_norm,
        composition=composition,
        concat_dim=concat_dim,
        crs_by_var=crs_by_var,
        block_sizes=block_sizes,
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
    warn: bool = True,
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

    if not warn:
        return None
    import warnings
    warnings.warn(
        "CRS not supplied and not detected; emitting VRT without <SRS>. "
        "Pass crs=<WKT> to attach one, or require_crs=True to make this an "
        "error.",
        UserWarning,
        stacklevel=4,
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


def _resolve_stack_has_dim(
    sources: list[Source], new_dim: str, array: str | None = None
) -> list[Source]:
    """Record, per stack source, whether it already carries ``new_dim``.

    "Stack" covers two source shapes: 2D slices that lack the stack dim
    (they need a ``[newaxis,...]`` view), and files that already carry it
    with length 1 (one-time-step-per-file NetCDF; adding an axis would make
    the source rank too high). Dict-form sources may say so with
    ``has_dim=True|False``; otherwise the first source is opened (metadata
    only) and its answer applies to all sources without an explicit value.
    """
    from dataclasses import replace

    sources = [replace(s, extra={**s.extra, "has_dim": False})
               if s.extra.get("band") is not None and "has_dim" not in s.extra
               else s for s in sources]
    if all("has_dim" in s.extra for s in sources):
        return sources
    probe = next(s for s in sources if "has_dim" not in s.extra)
    has = _source_has_dim(probe.original or probe.path, array or probe.array, new_dim)
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
    data_vars: list[str],
    sources: list[Source],
    composition: Composition,
    concat_dim: str | None,
    crs_by_var: Mapping[str, str | None],
    block_sizes: Mapping[str, list[int] | None],
    coord_mode: CoordMode | Mapping[str, CoordMode],
    inline_threshold: int,
) -> ET.Element:
    # Union of the data vars' dims, in order of first appearance.
    dims = list(dict.fromkeys(d for v in data_vars for d in ds[v].dims))

    root = ET.Element("VRTDataset")
    grp = X.sub(root, "Group", name="/")

    # <Dimension> elements.
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

    # Data-var <Array>s.
    for v in data_vars:
        _emit_data_array(
            grp,
            da=ds[v],
            name=v,
            sources=sources,
            arrays=[_array_for(s, v, data_vars) for s in sources],
            composition=composition,
            concat_dim=concat_dim,
            crs_wkt=crs_by_var.get(v),
            block_size=block_sizes.get(v),
        )

    return root


def _select_data_vars(ds: xr.Dataset, variables: Sequence[str] | None) -> list[str]:
    if variables is not None:
        if isinstance(variables, str):
            variables = [variables]
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            raise VrtWriterError(
                f"variables={missing} not in ds.data_vars ({list(ds.data_vars)})"
            )
        out = list(variables)
    else:
        gm = {ds[v].attrs.get("grid_mapping") for v in ds.data_vars}
        out = [v for v in ds.data_vars if v not in gm]
    if not out:
        raise VrtWriterError("ds has no data_vars")
    return out


def _array_for(src: Source, var: str, data_vars: Sequence[str]) -> str:
    """The <SourceArray> name for ``var`` in ``src``."""
    arrays = src.extra.get("arrays")
    if isinstance(arrays, Mapping) and var in arrays:
        return str(arrays[var])
    if len(data_vars) == 1:
        return src.array
    return var


def _resolve_crs_all(
    ds: xr.Dataset,
    data_vars: Sequence[str],
    crs_arg: str | None,
    *,
    require: bool,
) -> dict[str, str | None]:
    """CRS per variable; ``<SRS>`` only goes on variables with horizontal dims.

    A variable gets an SRS when both an X and a Y dim are detectable on it.
    If no variable has detectable X/Y dims, every variable of rank >= 2 gets
    it (the pre-multi-var behaviour, for unconventionally named grids).
    """
    found = {v: _resolve_crs(ds, v, crs_arg, require=False, warn=False)
             for v in data_vars}
    if not any(found.values()):
        # Re-run on the first var for the original warn/raise behaviour.
        _resolve_crs(ds, data_vars[0], None, require=require, warn=True)
        return {v: None for v in data_vars}
    fallback = next(w for w in found.values() if w)
    spatial = {v for v in data_vars
               if {"X", "Y"} <= {_dim_axis_tag(ds[v], d) for d in ds[v].dims}}
    if not spatial:
        spatial = {v for v in data_vars if ds[v].ndim >= 2}
    return {v: (found[v] or fallback) if v in spatial else None
            for v in data_vars}


def _axis_role(ds: xr.Dataset, dim: str) -> tuple[str | None, str | None]:
    """Return (type, direction) for <Dimension> when detectable from CF attrs.

    Conservative: only emit type/direction when we can justify it. Users can
    override by hand-editing or by attaching CF ``axis``/``standard_name``
    attrs to the coord.
    """
    attrs = ds.coords[dim].attrs if dim in ds.coords else {}
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
    # No CF attrs (e.g. readers that drop them): fall back on common names.
    low = str(dim).lower()
    if low in {"time", "t"}:
        return ("TEMPORAL", None)
    if low in _X_NAMES:
        return ("HORIZONTAL_X", "EAST")
    if low in _Y_NAMES:
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
    arrays: list[str] | None = None,
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
        arrays=arrays or [s.array for s in sources],
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


def _resolve_block_sizes(
    ds: xr.Dataset,
    data_vars: Sequence[str],
    block_size: Sequence[int] | Mapping[str, Sequence[int]] | None,
) -> dict[str, list[int] | None]:
    """Per-variable block sizes. A plain sequence applies to every variable
    of matching rank; a mismatch is an error rather than a silent skip."""
    if block_size is not None and not isinstance(block_size, Mapping):
        bad = [v for v in data_vars if ds[v].ndim != len(block_size)]
        if bad:
            raise VrtWriterError(
                f"block_size={list(block_size)} has {len(block_size)} entries "
                f"but {bad} have a different rank. Pass a dict keyed by "
                "variable name."
            )
    return {v: _resolve_block_size(ds[v], block_size, v) for v in data_vars}


def _resolve_block_size(
    da: xr.DataArray,
    block_size: Sequence[int] | Mapping[str, Sequence[int]] | None,
    data_var: str,
) -> list[int] | None:
    if block_size is None:
        if da.chunks is not None:
            # da.chunks is a tuple-of-tuples per dim; use the first chunk of each.
            return [int(cs[0]) for cs in da.chunks]
        # Backends declare their native chunking in encoding (xarray's
        # "preferred_chunks", or netCDF4's "chunksizes").
        pc = da.encoding.get("preferred_chunks")
        if isinstance(pc, Mapping) and all(d in pc for d in da.dims):
            return [int(pc[d]) for d in da.dims]
        cs = da.encoding.get("chunksizes")
        if cs is not None and len(cs) == da.ndim:
            return [int(c) for c in cs]
        return None
    if isinstance(block_size, Mapping):
        bs = block_size.get(data_var)
        if bs is None:
            return None
        return list(map(int, bs))
    return list(map(int, block_size))


def _is_band_source(src: Source) -> bool:
    return src.extra.get("band") is not None


def _emit_sources(
    arr: ET.Element,
    *,
    sources: list[Source],
    da: xr.DataArray,
    arrays: list[str],
    composition: Composition,
    concat_dim: str | None,
) -> None:
    dims = list(da.dims)
    shape = [int(n) for n in da.shape]

    def slab(el: ET.Element, count: list[int]) -> None:
        # Stating each source's extent lets GDAL open only the sources a
        # read touches; without it every source is opened (for remote COGs,
        # a HEAD plus a header GET each) to learn its shape.
        X.sub(el, "SourceSlab", offset=X.comma_list([0] * len(count)),
              count=X.comma_list(count), step=X.comma_list([1] * len(count)))

    def head(el: ET.Element, s: Source, array: str) -> None:
        X.sub(el, "SourceFilename", text=s.path, relativeToVRT=s.relative_to_vrt)
        if _is_band_source(s):
            X.sub(el, "SourceBand", text=str(int(s.extra["band"])))
        else:
            X.sub(el, "SourceArray", text=_full_array_name(array))

    if concat_dim not in dims:
        # Static variable (no composition dim): the first source supplies it.
        s = sources[0]
        src_el = X.sub(arr, "Source")
        head(src_el, s, arrays[0])
        if dims:
            slab(src_el, shape)
            X.sub(src_el, "DestSlab", offset=X.comma_list([0] * len(dims)))
        return

    # Which axis participates in the composition.
    comp_axis = dims.index(concat_dim)
    offset_so_far = 0
    for s, array in zip(sources, arrays):
        size_along = 1 if composition == "stack" else int(s.size)  # type: ignore[arg-type]
        src_el = X.sub(arr, "Source")
        head(src_el, s, array)
        if _is_band_source(s):
            # A classic raster band is 2D (y, x). SourceTranspose maps the
            # destination dims onto it, -1 marking the new length-1 axis.
            if len(dims) != 3:
                raise VrtWriterError(
                    f"band source {s.path}: a 2D band can only fill a 3D "
                    f"array (stack dim + 2 raster dims); {da.name!r} has "
                    f"dims {tuple(dims)}."
                )
            order, k = [], 0
            for i in range(len(dims)):
                if i == comp_axis:
                    order.append(-1)
                else:
                    order.append(k)
                    k += 1
            X.sub(src_el, "SourceTranspose", text=",".join(map(str, order)))
        elif composition == "stack" and not s.extra.get("has_dim", False):
            # A stacked source lacks the new dim. GDAL refuses to read a
            # source whose rank differs from the destination ("Inconsistent
            # number of dimensions"), so insert a length-1 axis at the
            # stack position with a view.
            X.sub(src_el, "SourceView",
                  text="[" + ":," * comp_axis + "newaxis,...]")
        count = list(shape)
        count[comp_axis] = size_along
        slab(src_el, count)
        # DestSlab always emitted (plan rule, zeros included).
        offsets = [0] * len(dims)
        offsets[comp_axis] = offset_so_far
        X.sub(src_el, "DestSlab", offset=X.comma_list(offsets))
        offset_so_far += size_along
