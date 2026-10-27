"""Coord value carrier dispatch.

Three carriers, per the plan:

- ``<RegularlySpacedValues start= increment=/>`` when uniform spacing is
  detected within a tolerance
- ``<InlineValuesWithValueElement><Value>...</Value>...</InlineValuesWithValueElement>``
  for small non-regular coords
- ``<Source>...</Source>`` pointing at a coord variable in one of the source
  files, for large coords (typical 2D lon/lat grids)

The ``coord_mode`` argument on :func:`xvrt.write_mdim_vrt` selects policy:

- ``"auto"``: regular-spacing detection, else inline below threshold, else
  source-ref
- ``"inline"``: always inline (errors if coord exceeds XML practicality)
- ``"source"``: always source-ref (caller promises the coord is in the source)
- ``dict``: per-coord override, e.g. ``{"time": "inline", "lat": "source"}``
"""
from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

from ._errors import VrtWriterError


CoordMode = Literal["auto", "inline", "source"]

_DEFAULT_INLINE_THRESHOLD = 10_000
_REGULAR_RTOL = 1e-9
_REGULAR_ATOL = 1e-15


def detect_regular(values: np.ndarray) -> tuple[float, float] | None:
    """Return (start, increment) if the values are uniformly spaced, else None.

    Uses :func:`numpy.isclose` with tight tolerances — suitable for real-world
    coord axes (daily time stamps, regular lat/lon grids) but strict enough
    that float drift in concatenated chunks won't be misidentified.

    Empty or single-element arrays are not regular.
    """
    if values.ndim != 1 or values.size < 2:
        return None
    # Float-safe: require all diffs equal to within tolerance.
    diffs = np.diff(values.astype(np.float64))
    if diffs.size == 0:
        return None
    first = diffs[0]
    if not np.all(np.isclose(diffs, first, rtol=_REGULAR_RTOL, atol=_REGULAR_ATOL)):
        return None
    # Reject zero increment — degenerate.
    if first == 0:
        return None
    # Increment from the end points, not the first difference: the first
    # difference of decoded floats carries rounding noise (0.01 comes back as
    # 0.010000000000005116), and the end-point estimate averages it away.
    v = values.astype(np.float64)
    return float(v[0]), float((v[-1] - v[0]) / (v.size - 1))


def resolve_mode(
    coord_name: str,
    values: np.ndarray,
    mode: CoordMode | Mapping[str, CoordMode],
    threshold: int = _DEFAULT_INLINE_THRESHOLD,
) -> CoordMode:
    """Resolve the coord-mode for a specific coord.

    ``mode`` may be a scalar policy or a per-coord dict. ``"auto"`` chooses
    between inline and source based on ``values.size`` and ``threshold``;
    regular-spacing detection is applied *before* this dispatch in
    :func:`xvrt._writer`.
    """
    if isinstance(mode, Mapping):
        chosen = mode.get(coord_name, "auto")
    else:
        chosen = mode

    if chosen not in ("auto", "inline", "source"):
        raise VrtWriterError(
            f"coord_mode for {coord_name!r}: expected 'auto'|'inline'|'source', "
            f"got {chosen!r}"
        )

    if chosen == "auto":
        chosen = "inline" if values.size <= threshold else "source"
    return chosen
