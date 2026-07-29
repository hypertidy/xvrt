"""pytest suite — rules R1..R7 and happy paths.

Runnable with ``pytest tests/test_rules.py`` once the package is installed
(``pip install -e .``) or with ``PYTHONPATH=. pytest``.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xvrt import write_mdim_vrt, VrtWriterError


# --- fixtures --------------------------------------------------------------

WKT = (
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],'
    'AXIS["latitude",north],AXIS["longitude",east],UNIT["degree",0.0174532925199433]]'
)


def _make_nc(tmp_path: Path, index: int, nt: int = 1, nvals: float | None = None) -> Path:
    ny, nx = 4, 6
    y = np.linspace(-10, 10, ny)
    x = np.linspace(140, 150, nx)
    t = np.arange(index, index + nt, dtype=np.float64) if nvals is None else np.full(nt, nvals)
    data = np.broadcast_to(t[:, None, None], (nt, ny, nx)).astype(np.float32).copy()
    ds = xr.Dataset(
        {"temp": (("time", "y", "x"), data, {"units": "degC"})},
        coords={
            "time": ("time", t, {"axis": "T", "standard_name": "time",
                                  "units": "days since 2020-01-01"}),
            "y": ("y", y, {"axis": "Y", "standard_name": "latitude"}),
            "x": ("x", x, {"axis": "X", "standard_name": "longitude"}),
        },
    )
    p = tmp_path / f"src_{index:02d}.nc"
    ds.to_netcdf(p)
    return p


def _stack_ds(tmp_path: Path, n: int = 3):
    paths = [_make_nc(tmp_path, i, nt=1) for i in range(n)]
    pieces = [xr.open_dataset(p, decode_times=False) for p in paths]
    ds = xr.concat(pieces, dim="time")
    return ds, paths


def _concat_ds(tmp_path: Path, per: int = 3, n_sources: int = 2):
    paths = [_make_nc(tmp_path, i * per, nt=per) for i in range(n_sources)]
    pieces = [xr.open_dataset(p, decode_times=False) for p in paths]
    ds = xr.concat(pieces, dim="time")
    return ds, paths


# --- happy paths -----------------------------------------------------------

def test_stack_happy(tmp_path):
    ds, paths = _stack_ds(tmp_path, n=3)
    out = write_mdim_vrt(ds, [str(p) for p in paths], tmp_path / "s.vrt",
                          composition="stack", concat_dim="time", crs=WKT)
    root = ET.parse(out).getroot()
    arr = [a for a in root.find("Group").findall("Array") if a.find("Source") is not None][0]
    srcs = arr.findall("Source")
    assert len(srcs) == 3
    offsets = [s.find("DestSlab").get("offset") for s in srcs]
    assert offsets == ["0,0,0", "1,0,0", "2,0,0"]


def test_block_size_is_comma_separated(tmp_path):
    """<BlockSize> takes comma-separated ints, matching GDAL mdim's
    BLOCKSIZE=bs0,bs1,...,bsN convention and <DestSlab offset>. Space
    separation (the xs:list XSD default) silently misparses."""
    ds, paths = _stack_ds(tmp_path, n=2)
    out = write_mdim_vrt(
        ds, [str(p) for p in paths], tmp_path / "bs.vrt",
        composition="stack", concat_dim="time", crs=WKT,
        block_size=[1, 4, 6],
    )
    root = ET.parse(out).getroot()
    bs = [el.text for el in root.iter("BlockSize")]
    assert bs, "expected at least one <BlockSize>"
    for text in bs:
        assert "," in text and " " not in text, (
            f"<BlockSize> must be comma-separated, got {text!r}"
        )


def test_concat_dim_case_mismatch_clear_error(tmp_path):
    """Case mismatch on concat_dim ('time' vs 'Time') fails up front with
    the available dims listed, not after walking into size resolution."""
    # Build a ds whose dim is "Time" (capital T), mimicking BRAN.
    ny, nx = 4, 6
    nt = 3
    t = np.arange(nt, dtype=np.float64)
    ds = xr.Dataset(
        {"temp": (("Time", "y", "x"), np.zeros((nt, ny, nx), dtype=np.float32))},
        coords={"Time": ("Time", t, {"units": "days since 2020-01-01"}),
                "y": ("y", np.linspace(-10, 10, ny)),
                "x": ("x", np.linspace(140, 150, nx))},
    )
    p = tmp_path / "bran_like.nc"
    ds.to_netcdf(p)

    with pytest.raises(VrtWriterError,
                        match=r"concat_dim='time'.*Available dims.*Time"):
        write_mdim_vrt(ds, [str(p)], tmp_path / "e.vrt",
                        composition="concat", concat_dim="time", crs=WKT)


def test_concat_happy(tmp_path):
    ds, paths = _concat_ds(tmp_path, per=3, n_sources=2)
    out = write_mdim_vrt(
        ds,
        [{"path": str(p), "size": 3} for p in paths],
        tmp_path / "c.vrt",
        composition="concat", concat_dim="time", crs=WKT,
    )
    root = ET.parse(out).getroot()
    arr = [a for a in root.find("Group").findall("Array") if a.find("Source") is not None][0]
    offsets = [s.find("DestSlab").get("offset") for s in arr.findall("Source")]
    assert offsets == ["0,0,0", "3,0,0"]


def test_array_xsd_child_order(tmp_path):
    """Array children come out in XSD-mandated order: DataType first, then
    DimensionRef*, then BlockSize/SRS/Unit/NoDataValue/Offset/Scale, then
    the value carrier and Sources, finally Attribute*."""
    ds, paths = _stack_ds(tmp_path, n=2)
    out = write_mdim_vrt(ds, [str(p) for p in paths], tmp_path / "o.vrt",
                          composition="stack", concat_dim="time", crs=WKT)
    root = ET.parse(out).getroot()
    for arr in root.iter("Array"):
        tags = [c.tag for c in arr]
        assert tags[0] == "DataType", (arr.get("name"), tags)
        # DimensionRef block contiguous and right after DataType
        dref_positions = [i for i, t in enumerate(tags) if t == "DimensionRef"]
        assert dref_positions, tags
        assert dref_positions[0] == 1, (arr.get("name"), tags)
        # Unit — when present — sits before any value-carrier / Source
        if "Unit" in tags and ("Source" in tags or "RegularlySpacedValues" in tags
                                or "InlineValuesWithValueElement" in tags):
            carrier_i = min(tags.index(t) for t in
                            ("Source", "RegularlySpacedValues",
                             "InlineValuesWithValueElement", "InlineValues",
                             "ConstantValue") if t in tags)
            assert tags.index("Unit") < carrier_i, (arr.get("name"), tags)


# --- rules -----------------------------------------------------------------

def test_r4_missing_crs_requires_flag(tmp_path):
    """With require_crs=True the writer raises; default is warn-and-omit."""
    ds, paths = _stack_ds(tmp_path, n=2)
    with pytest.raises(VrtWriterError, match="CRS"):
        write_mdim_vrt(ds, [str(p) for p in paths], tmp_path / "no.vrt",
                        composition="stack", concat_dim="time",
                        require_crs=True)


def test_r4_missing_crs_warns_by_default(tmp_path):
    """Default behaviour: warn and emit VRT without <SRS>."""
    import warnings
    ds, paths = _stack_ds(tmp_path, n=2)
    out = tmp_path / "soft.vrt"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        write_mdim_vrt(ds, [str(p) for p in paths], out,
                        composition="stack", concat_dim="time")
    assert any("CRS" in str(x.message) for x in w), [str(x.message) for x in w]
    root = ET.parse(out).getroot()
    assert root.find(".//SRS") is None, "expected no <SRS> when CRS undetected"


def test_r5_nonmonotonic_concat_raises(tmp_path):
    # Build two sources whose time coords overlap / reverse.
    p0 = _make_nc(tmp_path, 5, nt=2)  # times 5, 6
    p1 = _make_nc(tmp_path, 0, nt=2)  # times 0, 1 — non-monotonic after concat
    ds = xr.concat([xr.open_dataset(p0, decode_times=False),
                    xr.open_dataset(p1, decode_times=False)], dim="time")
    with pytest.raises(VrtWriterError, match="monotonic"):
        write_mdim_vrt(
            ds,
            [{"path": str(p0), "size": 2}, {"path": str(p1), "size": 2}],
            tmp_path / "bad.vrt",
            composition="concat", concat_dim="time", crs=WKT,
        )


def test_r6_open_handle_raises(tmp_path):
    ds, paths = _stack_ds(tmp_path, n=2)
    open_handle = xr.open_dataset(paths[0], decode_times=False)
    with pytest.raises(VrtWriterError, match="R6"):
        write_mdim_vrt(ds, [open_handle, str(paths[1])], tmp_path / "h.vrt",
                        composition="stack", concat_dim="time", crs=WKT)


def test_r7_check_function_catches_coord_datavar_collision():
    """R7 defends against malformed Datasets constructed through low-level
    APIs. xarray's public API already enforces non-collision between dims,
    coords, and data_vars — so this test exercises the check function
    directly with a mock-shaped object, confirming the defence-in-depth."""
    from xvrt._writer import _check_array_name_collisions

    class _FakeDS:
        # Simulate the shape _check_array_name_collisions reads: a Dataset
        # with dim 'foo' AND data_var 'foo' AND no dim-coord. Normal xarray
        # would reject this at construction, but some downstream code paths
        # (e.g. virtual Datasets) could in principle produce it.
        dims = ("foo", "y")
        coords = {"y": None}           # only y is a coord; foo is not
        data_vars = {"foo": None}      # foo is a data_var
        sizes = {"foo": 3, "y": 4}

    with pytest.raises(VrtWriterError, match="R7|collision"):
        _check_array_name_collisions(_FakeDS())  # type: ignore[arg-type]


def test_stack_wrong_source_count_raises(tmp_path):
    ds, paths = _stack_ds(tmp_path, n=3)
    with pytest.raises(VrtWriterError, match="sources provided"):
        write_mdim_vrt(ds, [str(paths[0]), str(paths[1])], tmp_path / "n.vrt",
                        composition="stack", concat_dim="time", crs=WKT)


def test_concat_sizes_kwarg(tmp_path):
    """sizes= kwarg is the one-liner alternative to dict-form sources."""
    ds, paths = _concat_ds(tmp_path, per=3, n_sources=2)
    out = write_mdim_vrt(
        ds, [str(p) for p in paths], tmp_path / "k.vrt",
        composition="concat", concat_dim="time", crs=WKT,
        sizes=[3, 3],
    )
    root = ET.parse(out).getroot()
    arr = [a for a in root.find("Group").findall("Array") if a.find("Source") is not None][0]
    offsets = [s.find("DestSlab").get("offset") for s in arr.findall("Source")]
    assert offsets == ["0,0,0", "3,0,0"]


def test_concat_auto_size_from_opening_sources(tmp_path):
    """Concat without any size hint: writer opens each source to read
    sizes[concat_dim]. Works because the sources are real files on disk."""
    ds, paths = _concat_ds(tmp_path, per=3, n_sources=2)
    out = write_mdim_vrt(
        ds, [str(p) for p in paths], tmp_path / "auto.vrt",
        composition="concat", concat_dim="time", crs=WKT,
        # no sizes=, no dict form — should auto-discover
    )
    root = ET.parse(out).getroot()
    arr = [a for a in root.find("Group").findall("Array") if a.find("Source") is not None][0]
    offsets = [s.find("DestSlab").get("offset") for s in arr.findall("Source")]
    assert offsets == ["0,0,0", "3,0,0"]


def test_concat_unopenable_sources_error_message(tmp_path):
    """When the source is unreachable and no sizes hint exists, the error
    message suggests sizes= / dict-form rather than parroting the IO error.

    Uses total=5 with 2 sources so step 3.5 (uniform guess) cannot apply
    (5 % 2 != 0), forcing us into the file-open fallback."""
    # Build ds with 5 timesteps from 2 real files of sizes 2 and 3.
    p0 = _make_nc(tmp_path, 0, nt=2)
    p1 = _make_nc(tmp_path, 2, nt=3)
    ds_loaded = xr.concat(
        [xr.open_dataset(p0, decode_times=False),
         xr.open_dataset(p1, decode_times=False)], dim="time",
    ).load()
    # Point at nonexistent paths — sizes must come from step 4 (which will fail).
    bad = [str(tmp_path / "nope_0.nc"), str(tmp_path / "nope_1.nc")]
    with pytest.raises(VrtWriterError, match="sizes="):
        write_mdim_vrt(ds_loaded, bad, tmp_path / "u.vrt",
                        composition="concat", concat_dim="time", crs=WKT)


def test_concat_sizes_mismatch_raises(tmp_path):
    ds, paths = _concat_ds(tmp_path, per=3, n_sources=2)
    with pytest.raises(VrtWriterError, match="sum to"):
        write_mdim_vrt(
            ds,
            [{"path": str(paths[0]), "size": 3}, {"path": str(paths[1]), "size": 2}],
            tmp_path / "m.vrt",
            composition="concat", concat_dim="time", crs=WKT,
        )


def test_mosaic_not_implemented(tmp_path):
    ds, paths = _stack_ds(tmp_path, n=2)
    with pytest.raises(NotImplementedError):
        write_mdim_vrt(ds, [str(p) for p in paths], tmp_path / "m.vrt",
                        composition="mosaic", crs=WKT)


def test_datetime_coord_auto_encoded(tmp_path):
    """datetime64 time coord is auto-encoded using xarray's stashed
    encoding['units'] — user does not need decode_times=False."""
    paths = [_make_nc(tmp_path, i, nt=1) for i in range(2)]
    pieces = [xr.open_dataset(p) for p in paths]  # default decodes time
    ds = xr.concat(pieces, dim="time")
    assert ds["time"].dtype.kind == "M"  # precondition: time is datetime64
    out = write_mdim_vrt(ds, [str(p) for p in paths], tmp_path / "dt.vrt",
                          composition="stack", concat_dim="time", crs=WKT)
    root = ET.parse(out).getroot()
    # Time coord Array should have Float64 DataType and a <Unit>.
    time_arr = next(a for a in root.iter("Array") if a.get("name") == "time")
    assert time_arr.find("DataType").text == "Float64"
    unit = time_arr.find("Unit")
    assert unit is not None and "since" in unit.text


# --- accessor --------------------------------------------------------------

def test_accessor(tmp_path):
    import xvrt.accessor  # noqa: F401  — registers the accessor
    ds, paths = _stack_ds(tmp_path, n=2)
    out = ds.vrt.write_mdim(
        tmp_path / "a.vrt",
        sources=[str(p) for p in paths],
        composition="stack", concat_dim="time", crs=WKT,
    )
    assert out.exists()
