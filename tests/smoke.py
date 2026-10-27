"""Smoke tests for v0 writer.

Run with: python -m tests.smoke

These avoid GDAL — they build synthetic NetCDFs, open them with xarray,
write a VRT, and structurally verify the emitted XML (XSD validation, when
the schema is present; structural assertions unconditionally).
"""
from __future__ import annotations

import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import xarray as xr

# Make the in-repo package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xvrt import write_mdim_vrt, encode_datetime_coord, VrtWriterError


def make_source(tmpdir: Path, index: int, t_value: float) -> Path:
    """Write a small NetCDF at a given synthetic time value."""
    ny, nx = 4, 6
    y = np.linspace(-10, 10, ny)
    x = np.linspace(140, 150, nx)
    data = np.full((ny, nx), t_value, dtype=np.float32) + np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) * 0.01
    ds = xr.Dataset(
        {"temp": (("y", "x"), data, {"units": "degC"})},
        coords={
            "y": ("y", y, {"axis": "Y", "standard_name": "latitude", "units": "degrees_north"}),
            "x": ("x", x, {"axis": "X", "standard_name": "longitude", "units": "degrees_east"}),
        },
        attrs={"source_index": index},
    )
    path = tmpdir / f"src_{index:02d}.nc"
    ds.to_netcdf(path)
    return path


def smoke_stack(tmpdir: Path) -> Path:
    """Stack 3 sources along a new 'time' dim; regular spacing; CRS from arg."""
    paths = [make_source(tmpdir, i, float(i)) for i in range(3)]
    # Build the xarray target by opening each source and expanding dims.
    pieces = [xr.open_dataset(p).expand_dims(time=[float(i)]) for i, p in enumerate(paths)]
    ds = xr.concat(pieces, dim="time")
    ds = ds.assign_coords(time=("time", np.array([0.0, 1.0, 2.0]),
                                 {"axis": "T", "standard_name": "time",
                                  "units": "days since 2020-01-01"}))

    vrt_path = tmpdir / "stack.vrt"
    write_mdim_vrt(
        ds, [str(p) for p in paths], vrt_path,
        composition="stack", concat_dim="time",
        crs=_dummy_wkt(),
    )
    return vrt_path


def smoke_concat(tmpdir: Path) -> Path:
    """Concat 2 sources, each contributing 3 elements along 'time'."""
    ny, nx = 4, 6
    y = np.linspace(-10, 10, ny)
    x = np.linspace(140, 150, nx)

    def make(i0: int, n: int) -> Path:
        t = np.arange(i0, i0 + n, dtype=np.float64)
        data = np.broadcast_to(t[:, None, None], (n, ny, nx)).astype(np.float32).copy()
        ds = xr.Dataset(
            {"temp": (("time", "y", "x"), data)},
            coords={
                "time": ("time", t, {"axis": "T", "standard_name": "time",
                                      "units": "days since 2020-01-01"}),
                "y": ("y", y, {"axis": "Y", "standard_name": "latitude"}),
                "x": ("x", x, {"axis": "X", "standard_name": "longitude"}),
            },
        )
        p = tmpdir / f"concat_src_{i0:02d}.nc"
        ds.to_netcdf(p)
        return p

    p0 = make(0, 3)
    p1 = make(3, 3)
    # Open with decode_times=False so the time coord stays numeric. This is
    # the realistic path for users with CF-encoded times who want the
    # writer to emit a numeric time axis + CF <Unit>.
    ds = xr.concat(
        [xr.open_dataset(p0, decode_times=False),
         xr.open_dataset(p1, decode_times=False)],
        dim="time",
    )

    vrt_path = tmpdir / "concat.vrt"
    write_mdim_vrt(
        ds,
        [{"path": str(p0), "size": 3},
         {"path": str(p1), "size": 3}],
        vrt_path,
        composition="concat", concat_dim="time",
        crs=_dummy_wkt(),
    )
    return vrt_path


def _dummy_wkt() -> str:
    # A trivial WGS84 WKT2. Just enough to satisfy R4 in smoke tests.
    return (
        'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
        'ELLIPSOID["WGS 84",6378137,298.257223563]],'
        'CS[ellipsoidal,2],'
        'AXIS["latitude",north],AXIS["longitude",east],'
        'UNIT["degree",0.0174532925199433]]'
    )


def structural_check(vrt_path: Path, *, expected_dims: tuple[str, ...],
                     expected_composition_offsets: list[list[int]]) -> None:
    tree = ET.parse(vrt_path)
    root = tree.getroot()
    assert root.tag == "VRTDataset", root.tag
    groups = list(root.findall("Group"))
    assert len(groups) == 1 and groups[0].get("name") == "/", "expect one root group"
    grp = groups[0]

    # Dimensions first, arrays last (XSD order).
    children_tags = [c.tag for c in grp]
    dim_idx = [i for i, t in enumerate(children_tags) if t == "Dimension"]
    arr_idx = [i for i, t in enumerate(children_tags) if t == "Array"]
    attr_idx = [i for i, t in enumerate(children_tags) if t == "Attribute"]
    assert dim_idx and arr_idx
    assert max(dim_idx) < min(arr_idx), "Dimensions must precede Arrays"
    if attr_idx:
        assert max(dim_idx) < min(attr_idx) < min(arr_idx), \
            "Attributes must be between Dimensions and Arrays"

    # Dim names and order.
    dim_names = tuple(d.get("name") for d in grp.findall("Dimension"))
    assert dim_names == expected_dims, (dim_names, expected_dims)

    # Find the data-var array (the one with Sources).
    data_arrays = [a for a in grp.findall("Array") if a.find("Source") is not None]
    assert len(data_arrays) == 1
    da = data_arrays[0]

    # DataType must be the first child of Array.
    first_child = list(da)[0]
    assert first_child.tag == "DataType", f"Array first child must be DataType, got {first_child.tag}"

    # DestSlab offsets.
    offsets = []
    for s in da.findall("Source"):
        ds_el = s.find("DestSlab")
        assert ds_el is not None, "every <Source> must carry <DestSlab>"
        offsets.append([int(x) for x in ds_el.get("offset").split(",")])
    assert offsets == expected_composition_offsets, (offsets, expected_composition_offsets)

    # Array child order: DataType, then DimensionRef, then BlockSize/SRS/Unit/NoData/Offset/Scale, then Source, then Attribute.
    order_seen = []
    for c in da:
        order_seen.append(c.tag)
    # Every Source must appear after every DimensionRef.
    dref_i = max(i for i, t in enumerate(order_seen) if t == "DimensionRef")
    src_i = min(i for i, t in enumerate(order_seen) if t == "Source")
    assert dref_i < src_i, f"DimensionRef must precede Source: {order_seen}"


def run_all() -> int:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        print("== stack ==")
        p = smoke_stack(td)
        structural_check(
            p,
            expected_dims=("time", "y", "x"),
            expected_composition_offsets=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        )
        print(p.read_text())

        print("\n== concat ==")
        p = smoke_concat(td)
        structural_check(
            p,
            expected_dims=("time", "y", "x"),
            expected_composition_offsets=[[0, 0, 0], [3, 0, 0]],
        )
        print(p.read_text())

        print("\n== negative: missing crs raises R4 (require_crs=True) ==")
        paths = [make_source(td, i, float(i)) for i in range(2)]
        pieces = [xr.open_dataset(pp).expand_dims(time=[float(i)]) for i, pp in enumerate(paths)]
        ds = xr.concat(pieces, dim="time").assign_coords(time=("time", [0.0, 1.0]))
        try:
            write_mdim_vrt(ds, [str(pp) for pp in paths], td / "nocrs.vrt",
                            composition="stack", concat_dim="time",
                            require_crs=True)
        except VrtWriterError as e:
            print("  R4 raised as expected:", e)
        else:
            raise AssertionError("expected VrtWriterError for missing CRS")

    return 0


if __name__ == "__main__":
    sys.exit(run_all())
