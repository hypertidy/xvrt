"""stack_rasters(): the vrtstack mode."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from xvrt import stack_rasters, VrtWriterError

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


GRID = {
    "shape": (4, 6),
    "geotransform": (140.0, 2.0, 0.0, -10.0, 0.0, -2.5),
    "crs": (
        'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
        'ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],'
        'AXIS["latitude",north],AXIS["longitude",east],'
        'UNIT["degree",0.0174532925199433]]'
    ),
    "dtype": "float32",
    "nodata": -9999.0,
}


def _data_array(root, name):
    return next(a for a in root.iter("Array") if a.get("name") == name)


def test_explicit_values_grid_no_io(tmp_path):
    """grid= means nothing is opened: files need not exist."""
    files = [f"soil_{d}cm.tif" for d in (0, 30, 100)]
    out = stack_rasters(
        files, tmp_path / "soil.vrt",
        dim="depth", values=[0, 30, 100], units="cm",
        attrs={"axis": "Z", "standard_name": "depth"},
        grid=GRID, array_name="soil",
    )
    root = ET.parse(out).getroot()
    dims = {d.get("name"): d for d in root.iter("Dimension")}
    assert dims["depth"].get("size") == "3"
    assert dims["depth"].get("type") == "VERTICAL"

    arr = _data_array(root, "soil")
    srcs = arr.findall("Source")
    assert len(srcs) == 3
    assert [s.find("SourceBand").text for s in srcs] == ["1", "1", "1"]
    assert [s.find("DestSlab").get("offset") for s in srcs] == [
        "0,0,0", "1,0,0", "2,0,0"]
    # axis-inserting transpose on every band source
    assert all(s.find("SourceTranspose").text == "-1,0,1" for s in srcs)
    # nodata from the grid dict
    assert arr.find("NoDataValue") is not None

    depth = _data_array(root, "depth")
    assert depth.find("Unit").text == "cm"


def test_values_sorted(tmp_path):
    files = ["b_30.tif", "c_100.tif", "a_0.tif"]
    out = stack_rasters(
        files, tmp_path / "s.vrt",
        dim="depth", values=[30, 100, 0], grid=GRID,
    )
    root = ET.parse(out).getroot()
    arr = next(a for a in root.iter("Array") if a.find("Source") is not None)
    names = [s.find("SourceFilename").text for s in arr.findall("Source")]
    assert [n.split("/")[-1] for n in names] == [
        "a_0.tif", "b_30.tif", "c_100.tif"]


def test_regex_dates_from_filenames(tmp_path):
    files = [f"sst_2023010{i}.tif" for i in (3, 1, 2)]
    out = stack_rasters(
        files, tmp_path / "t.vrt",
        dim="time", values=r"(\d{8})", format="%Y%m%d",
        units="days since 2023-01-01", grid=GRID,
    )
    root = ET.parse(out).getroot()
    t = _data_array(root, "time")
    assert "since" in t.find("Unit").text
    rsv = t.find("RegularlySpacedValues")
    assert rsv is not None
    assert float(rsv.get("start")) == 0.0 and float(rsv.get("increment")) == 1.0
    # sorted by date despite shuffled input
    arr = next(a for a in root.iter("Array") if a.find("Source") is not None)
    names = [s.find("SourceFilename").text.split("/")[-1]
             for s in arr.findall("Source")]
    assert names == [f"sst_2023010{i}.tif" for i in (1, 2, 3)]


def test_start_step_keeps_order(tmp_path):
    files = [f"p_{i}.tif" for i in range(4)]
    out = stack_rasters(
        files, tmp_path / "p.vrt",
        dim="pressure", start=1000, step=-50, units="hPa", grid=GRID,
    )
    root = ET.parse(out).getroot()
    p = _data_array(root, "pressure")
    rsv = p.find("RegularlySpacedValues")
    assert float(rsv.get("start")) == 1000.0
    assert float(rsv.get("increment")) == -50.0


def test_array_name_default_common_prefix(tmp_path):
    files = [f"claytotal_{d}_cm_p.tif" for d in (0, 5, 15)]
    out = stack_rasters(files, tmp_path / "c.vrt",
                        dim="depth", values=[0, 5, 15], grid=GRID)
    root = ET.parse(out).getroot()
    assert any(a.get("name") == "claytotal" for a in root.iter("Array"))


def test_url_sources_get_vsicurl(tmp_path):
    files = [f"https://example.com/sst_{i}.tif" for i in range(2)]
    out = stack_rasters(files, tmp_path / "u.vrt",
                        dim="time", values=[0, 1], grid=GRID)
    root = ET.parse(out).getroot()
    arr = next(a for a in root.iter("Array") if a.find("Source") is not None)
    for s in arr.findall("Source"):
        assert s.find("SourceFilename").text.startswith("/vsicurl/https://")


def test_errors(tmp_path):
    with pytest.raises(VrtWriterError, match="values=|start="):
        stack_rasters(["a.tif"], tmp_path / "e.vrt", grid=GRID)
    with pytest.raises(VrtWriterError, match="mutually exclusive"):
        stack_rasters(["a.tif"], tmp_path / "e.vrt",
                      values=[1], start=0, step=1, grid=GRID)
    with pytest.raises(VrtWriterError, match="one capture group"):
        stack_rasters(["a.tif"], tmp_path / "e.vrt", values=r"\d+", grid=GRID)
    with pytest.raises(VrtWriterError, match="did not match"):
        stack_rasters(["a.tif"], tmp_path / "e.vrt", values=r"(\d{8})", grid=GRID)
    with pytest.raises(VrtWriterError, match="values for"):
        stack_rasters(["a.tif", "b.tif"], tmp_path / "e.vrt",
                      values=[1], grid=GRID)
    with pytest.raises(VrtWriterError, match="rotated"):
        bad = dict(GRID, geotransform=(0.0, 1.0, 0.1, 0.0, 0.0, -1.0))
        stack_rasters(["a.tif"], tmp_path / "e.vrt", values=[1], grid=bad)
    with pytest.raises(VrtWriterError, match="lacks"):
        stack_rasters(["a.tif"], tmp_path / "e.vrt", values=[1],
                      grid={"shape": (2, 2)})


@pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
def test_grid_from_rasterio(tmp_path):
    from rasterio.transform import from_origin
    files = []
    for i in range(3):
        p = tmp_path / f"r_{i}.tif"
        with rasterio.open(
            p, "w", driver="GTiff", height=4, width=6, count=1,
            dtype="float32", crs="EPSG:4326",
            transform=from_origin(140.0, -10.0, 2.0, 2.5), nodata=-1.0,
        ) as dst:
            dst.write(np.full((4, 6), float(i), dtype="float32"), 1)
        files.append(str(p))
    out = stack_rasters(files, tmp_path / "r.vrt",
                        dim="time", values=[0.0, 1.0, 2.0])
    root = ET.parse(out).getroot()
    arr = next(a for a in root.iter("Array") if a.find("Source") is not None)
    assert arr.find("DataType").text == "Float32"
    assert arr.find("SRS") is not None
    assert float(arr.find("NoDataValue").text) == -1.0
    x = _data_array(root, "x")
    rsv = x.find("RegularlySpacedValues")
    assert float(rsv.get("start")) == 141.0  # pixel centre
    assert float(rsv.get("increment")) == 2.0
