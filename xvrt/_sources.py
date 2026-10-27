"""Source specification — normalise user input.

Accepts:
  - ``list[str | Path]`` — each source is a path/URL, writer infers the source
    array name per composition mode
  - ``list[dict]`` — each dict carries ``{"path": ..., "array": ..., ...}``

Produces a uniform list of :class:`Source` records for the emitter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ._errors import VrtWriterError


@dataclass(frozen=True)
class Source:
    """One participating source.

    Attributes
    ----------
    path
        The string that goes into ``<SourceFilename>``. May be a local path,
        a URL with ``/vsicurl/`` prefix applied, or a plain ``/vsi*`` VSI path.
    array
        The ``<SourceArray>`` name — the full-name of the array inside the
        source (e.g. ``"/temp"``, or ``"temp"`` for root-group arrays).
    relative_to_vrt
        Emitted as ``relativeToVRT="1"`` when True.
    size
        For ``composition="concat"``: number of elements this source
        contributes along the concat dim. ``None`` for stack (always 1).
    original
        The source string as the user originally passed it, before VSI or
        relative-to-VRT rewriting. Used by size auto-discovery so the
        fallback opener sees a resolvable path.
    """

    path: str
    array: str
    relative_to_vrt: bool = False
    size: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    original: str = ""


_VSI_PREFIXES = ("/vsi",)


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _to_vsi(s: str) -> str:
    """Prefix http(s) URLs with /vsicurl/; pass /vsi* through unchanged."""
    if s.startswith(_VSI_PREFIXES):
        return s
    if _is_url(s):
        return f"/vsicurl/{s}"
    if s.startswith("s3://"):
        return f"/vsis3/{s[5:]}"
    return s


def normalise_sources(
    sources: Sequence[Any],
    *,
    default_array: str,
    vrt_path: Path | None = None,
    composition: str,
    n_vars: int = 1,
) -> list[Source]:
    """Resolve user-supplied sources into :class:`Source` records.

    Parameters
    ----------
    sources
        ``list[str | Path | dict]``.
    default_array
        Array name to use when a dict-form source omits ``"array"``. Typically
        the data-variable name from ``ds``.
    vrt_path
        Path the VRT will be written to. Used to decide whether to mark
        ``relativeToVRT="1"``. If ``None``, always emit absolute paths.
    composition
        ``"stack"`` | ``"concat"`` | ``"mosaic"``. Stack sources must not
        carry a per-source ``size``.
    n_vars
        Number of data variables being written. With more than one, a
        dict-form ``"array"`` is ambiguous and refused; use ``"arrays"``
        (a ``{var: source_array_name}`` mapping) instead.
    """
    if not sources:
        raise VrtWriterError("at least one source is required")

    out: list[Source] = []
    for i, raw in enumerate(sources):
        out.append(_coerce_one(raw, i, default_array, vrt_path, composition, n_vars))
    return out


def _coerce_one(
    raw: Any,
    index: int,
    default_array: str,
    vrt_path: Path | None,
    composition: str,
    n_vars: int = 1,
) -> Source:
    if isinstance(raw, Source):
        return raw

    # R6: no open handles. Check before type dispatch — a Dataset is not a
    # str/Path/dict and would otherwise hit the generic "expected ..." error.
    if hasattr(raw, "variables") and hasattr(raw, "dims"):
        raise VrtWriterError(
            f"source #{index}: open xarray/netCDF4 handles are not accepted. "
            "Pass the path string instead (R6)."
        )

    if isinstance(raw, (str, Path, PurePosixPath)):
        path_str = str(raw)
        arr = default_array
        size = None
        rel = False
        extra: Mapping[str, Any] = {}
    elif isinstance(raw, Mapping):
        if "path" not in raw:
            raise VrtWriterError(
                f"source #{index}: dict form requires a 'path' key"
            )
        path_str = str(raw["path"])
        if n_vars > 1 and "array" in raw:
            raise VrtWriterError(
                f"source #{index}: 'array' names one source array but "
                f"{n_vars} variables are being written. Use "
                "'arrays': {var: '/name', ...} for renames."
            )
        arrays = raw.get("arrays")
        if arrays is not None and not isinstance(arrays, Mapping):
            raise VrtWriterError(
                f"source #{index}: 'arrays' must be a dict {{var: name}}"
            )
        arr = raw.get("array", default_array)
        size = raw.get("size", None)
        rel = bool(raw.get("relative_to_vrt", False))
        extra = {k: v for k, v in raw.items()
                 if k not in {"path", "array", "size", "relative_to_vrt"}}
    else:
        raise VrtWriterError(
            f"source #{index}: expected str, Path, or dict — got {type(raw).__name__}"
        )

    # Classic raster sources (GeoTIFF/COG): <SourceBand> instead of
    # <SourceArray>. Explicit with "band": N in dict form; implied band 1 for
    # .tif/.tiff paths that name no array.
    band = extra.get("band") if isinstance(extra, Mapping) else None
    if band is None and not (isinstance(raw, Mapping) and
                             ("array" in raw or "arrays" in raw)):
        if path_str.lower().split("?")[0].endswith((".tif", ".tiff")):
            band = 1
    if band is not None:
        extra = {**extra, "band": int(band)}
        if composition == "concat":
            raise VrtWriterError(
                f"source #{index}: a raster band source is 2D and has no "
                "concat dim to contribute; use composition='stack'."
            )

    if composition == "stack" and size is not None and size != 1:
        raise VrtWriterError(
            f"source #{index}: stack composition contributes one element per "
            "source; do not pass size=. Use concat if each source has >1 "
            "element along the dim."
        )

    # Apply /vsicurl/ prefixing and relativity.
    path_emitted = _to_vsi(path_str)
    if (vrt_path is not None and not _is_url(path_str)
            and not path_str.startswith(("/vsi", "s3://"))):
        try:
            rel_path = Path(path_str).resolve().relative_to(vrt_path.parent.resolve())
            # Only auto-relativise if the caller didn't explicitly set it.
            if rel is False:
                path_emitted = str(rel_path)
                rel = True
        except (ValueError, OSError):
            pass

    return Source(
        path=path_emitted,
        array=str(arr),
        relative_to_vrt=rel,
        size=size,
        extra=extra,
        original=path_str,
    )
