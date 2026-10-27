"""Round-trip regression tests through osgeo.gdal's mdim API.

Skipped when GDAL's Python bindings are not importable. Each test writes
small NetCDF sources, opens them with xarray the way a user would, writes a
VRT, reads it back with GDAL and compares against xarray's view.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

gdal = pytest.importorskip("osgeo.gdal")
gdal.UseExceptions()

from xvrt import write_mdim_vrt  # noqa: E402

pytest.importorskip("netCDF4")

# EPSG:4326 as WKT2, authority axis order (lat, lon). Written out literally
# so the tests do not depend on proj.db being findable.
WKT_LATLON = (
    'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'CS[ellipsoidal,2],'
    'AXIS["geodetic latitude (Lat)",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],'
    'AXIS["geodetic longitude (Lon)",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]]]'
)
# WKT1 with no AXIS: GDAL treats it as easting-first.
WKT1_NOAXIS = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
)

LAT = np.array([-1.0, 0.0, 1.0])
LON = np.array([10.0, 11.0, 12.0, 13.0])


def _coords(t, t_units):
    return {
        "time": ("time", np.atleast_1d(np.asarray(t, dtype="f8")),
                 {"units": t_units, "axis": "T", "standard_name": "time"}),
        "lat": ("lat", LAT, {"axis": "Y", "standard_name": "latitude",
                             "units": "degrees_north"}),
        "lon": ("lon", LON, {"axis": "X", "standard_name": "longitude",
                             "units": "degrees_east"}),
    }


def _packed_sources(tmp: Path, times, t_units="days since 1981-01-01 12:00:00"):
    paths = []
    for i, t in enumerate(times):
        raw = (np.arange(12).reshape(1, 3, 4) + 100 * i).astype("int16")
        raw[0, 0, 0] = -999
        ds = xr.Dataset(
            {"sst": (("time", "lat", "lon"), raw,
                     {"units": "degC", "scale_factor": 0.01,
                      "add_offset": 0.0, "_FillValue": np.int16(-999)})},
            coords=_coords(t, t_units),
        )
        p = tmp / f"s{i}.nc"
        ds.to_netcdf(p, encoding={"sst": {"dtype": "int16"}})
        paths.append(p)
    return paths


def _open_vrt(path):
    return gdal.OpenEx(str(path), gdal.OF_MULTIDIM_RASTER).GetRootGroup()


_SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "gdalvrt.xsd"


def _validate_xsd(path):
    etree = pytest.importorskip("lxml.etree")
    schema = etree.XMLSchema(etree.parse(str(_SCHEMA)))
    doc = etree.parse(str(path))
    schema.validate(doc)
    # Known GDAL schema/driver disagreement: the XSD types mdim <BlockSize>
    # as a space-separated integerList, but the VRT driver only parses
    # comma-separated values ("Invalid number of values in BLOCKSIZE"
    # otherwise). We follow the driver.
    errors = [e for e in schema.error_log
              if "'BlockSize'" not in e.message]
    assert not errors, "\n".join(str(e) for e in errors)


def _write(ds, paths, out, **kw):
    kw.setdefault("composition", "concat")
    kw.setdefault("concat_dim", "time")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = write_mdim_vrt(ds, [str(p) for p in paths], out, **kw)
    _validate_xsd(out)
    return out


def test_decoded_packed_data_keeps_storage_dtype_and_scale(tmp_path):
    paths = _packed_sources(tmp_path, [0, 1, 3, 7])
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    assert ds.sst.dtype.kind == "f"  # xarray decoded it

    g = _open_vrt(_write(ds, paths, tmp_path / "o.vrt"))
    a = g.OpenMDArray("sst")
    assert gdal.GetDataTypeName(a.GetDataType().GetNumericDataType()) == "Int16"
    assert a.GetScale() == pytest.approx(0.01)
    assert a.GetNoDataValueAsDouble() == -999

    got = a.GetUnscaled().ReadAsArray()
    want = ds.sst.values
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)])

    text = (tmp_path / "o.vrt").read_text()
    assert 'name="scale_factor"' not in text  # not duplicated as <Attribute>


def test_undecoded_packed_data(tmp_path):
    paths = _packed_sources(tmp_path, [0, 1])
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time",
                           mask_and_scale=False)
    a = _open_vrt(_write(ds, paths, tmp_path / "o.vrt")).OpenMDArray("sst")
    assert a.GetScale() == pytest.approx(0.01)
    np.testing.assert_array_equal(a.ReadAsArray(), ds.sst.values)


def test_time_origin_with_time_of_day(tmp_path):
    paths = _packed_sources(tmp_path, [0, 1, 3, 7])
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    assert ds.time.dtype.kind == "M"
    t = _open_vrt(_write(ds, paths, tmp_path / "o.vrt")).OpenMDArray("time")
    np.testing.assert_allclose(t.ReadAsArray(), [0, 1, 3, 7])
    assert t.GetUnit() == "days since 1981-01-01 12:00:00"


def test_concat_coord_source_mode_reads_every_file(tmp_path):
    paths = _packed_sources(tmp_path, [0, 1, 3, 7])
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    out = _write(ds, paths, tmp_path / "o.vrt", coord_mode={"time": "source"})
    t = _open_vrt(out).OpenMDArray("time")
    np.testing.assert_allclose(t.ReadAsArray(), [0, 1, 3, 7])


def test_concat_coord_auto_never_points_at_one_file(tmp_path):
    paths = _packed_sources(tmp_path, [0, 1, 3, 7])
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    # threshold 0 would push every coord to source mode in auto
    out = _write(ds, paths, tmp_path / "o.vrt", inline_threshold=0)
    t = _open_vrt(out).OpenMDArray("time")
    np.testing.assert_allclose(t.ReadAsArray(), [0, 1, 3, 7])


def test_cftime_noleap_coord(tmp_path):
    paths = _packed_sources(tmp_path, [0, 59, 365], t_units="days since 2001-01-01")
    for p in paths:  # stamp a noleap calendar on each file
        with xr.open_dataset(p, decode_times=False) as d:
            d = d.load()
        d.time.attrs["calendar"] = "noleap"
        d.to_netcdf(p, mode="w")
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time",
                           decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    assert ds.time.dtype.kind == "O"
    out = _write(ds, paths, tmp_path / "o.vrt")
    t = _open_vrt(out).OpenMDArray("time")
    np.testing.assert_allclose(t.ReadAsArray(), [0, 59, 365])
    assert 'name="calendar"' in out.read_text()


@pytest.mark.parametrize("wkt, expected", [(WKT_LATLON, [2, 3]),
                                           (WKT1_NOAXIS, [3, 2])])
def test_axis_mapping_matches_gdal(tmp_path, wkt, expected):
    paths = _packed_sources(tmp_path, [0, 1])
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    out = _write(ds, paths, tmp_path / "o.vrt", crs=wkt)
    m = re.search(r'dataAxisToSRSAxisMapping="([^"]*)"', out.read_text())
    assert [int(v) for v in m.group(1).split(",")] == expected
    srs = _open_vrt(out).OpenMDArray("sst").GetSpatialRef()
    assert srs.GetDataAxisToSRSAxisMapping() == expected


def test_axis_mapping_agrees_with_netcdf_driver(tmp_path):
    """Whatever GDAL's netCDF driver reports for a CF grid_mapping source,
    xvrt should emit the same mapping for the same WKT."""
    paths = _packed_sources(tmp_path, [0])
    d = xr.open_dataset(paths[0], mask_and_scale=False, decode_times=False).load()
    d["crs"] = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude",
                                      "crs_wkt": WKT_LATLON})
    d["sst"].attrs["grid_mapping"] = "crs"
    ref = tmp_path / "gm.nc"
    d.to_netcdf(ref)
    r = gdal.OpenEx(str(ref), gdal.OF_MULTIDIM_RASTER).GetRootGroup().OpenMDArray("sst")
    ref_map = r.GetSpatialRef().GetDataAxisToSRSAxisMapping()

    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    out = _write(ds, paths, tmp_path / "o.vrt", crs=WKT_LATLON)
    got = _open_vrt(out).OpenMDArray("sst").GetSpatialRef().GetDataAxisToSRSAxisMapping()
    assert got == ref_map


def test_stack_of_2d_sources_values(tmp_path):
    paths = []
    for i in range(3):
        v = np.full((3, 4), float(i), dtype="f4")
        ds = xr.Dataset({"temp": (("lat", "lon"), v)},
                        coords={k: c for k, c in _coords(0, "days").items()
                                if k != "time"})
        p = tmp_path / f"s{i}.nc"
        ds.to_netcdf(p)
        paths.append(p)
    parts = [xr.open_dataset(p) for p in paths]
    ds = xr.concat(parts, dim="time").assign_coords(time=[10.0, 20.0, 30.0])
    out = _write(ds, paths, tmp_path / "o.vrt", composition="stack")
    a = _open_vrt(out).OpenMDArray("temp")
    np.testing.assert_array_equal(a.ReadAsArray(), ds.temp.values)


def test_stack_of_sources_with_length1_dim(tmp_path):
    """One-time-step-per-file sources already carry the stack dim; no
    newaxis view must be added (the source rank already matches)."""
    paths = _packed_sources(tmp_path, [0, 1, 2])
    pieces = [xr.open_dataset(p, decode_times=False) for p in paths]
    ds = xr.concat(pieces, dim="time")
    out = _write(ds, paths, tmp_path / "o.vrt", composition="stack")
    assert "SourceView" not in out.read_text()
    a = _open_vrt(out).OpenMDArray("sst")
    np.testing.assert_allclose(a.GetUnscaled().ReadAsArray(), ds.sst.values)


def test_stack_has_dim_override(tmp_path):
    paths = _packed_sources(tmp_path, [0, 1])
    pieces = [xr.open_dataset(p, decode_times=False) for p in paths]
    ds = xr.concat(pieces, dim="time")
    srcs = [{"path": str(p), "has_dim": True} for p in paths]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = write_mdim_vrt(ds, srcs, tmp_path / "o.vrt",
                             composition="stack", concat_dim="time")
    assert "SourceView" not in out.read_text()


# -- multi-variable ---------------------------------------------------------

OISST_VARS = ("sst", "anom", "err", "ice")


def _oisst_sources(tmp: Path, n_days=3, *, with_mask=False, with_crs=False):
    """OISST v2.1 file shape: one day per file, (time, zlev, lat, lon),
    four packed int16 variables, CF time with a 12:00 origin."""
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n_days):
        data_vars = {}
        for k, v in enumerate(OISST_VARS):
            raw = rng.integers(-500, 3000, (1, 1, 3, 4)).astype("int16")
            raw[0, 0, 0, k] = -999
            data_vars[v] = (("time", "zlev", "lat", "lon"), raw,
                            {"units": "degC" if v != "ice" else "%",
                             "long_name": f"{v} long name",
                             "scale_factor": 0.01, "add_offset": 0.0,
                             "_FillValue": np.int16(-999)})
        if with_mask:
            data_vars["mask"] = (("lat", "lon"),
                                 np.array([[1, 0, 1, 1]] * 3, dtype="int8"))
        if with_crs:
            data_vars["crs"] = ((), np.int32(0),
                                {"grid_mapping_name": "latitude_longitude",
                                 "crs_wkt": WKT_LATLON})
            for v in OISST_VARS:
                data_vars[v][2]["grid_mapping"] = "crs"
        coords = _coords(float(i), "days since 1978-01-01 12:00:00")
        coords["zlev"] = ("zlev", np.array([0.0], "f4"),
                          {"axis": "Z", "units": "meters"})
        ds = xr.Dataset(data_vars, coords=coords, attrs={"title": "OISST-like"})
        p = tmp / f"oisst-avhrr-v02r01.1978010{i}.nc"
        ds.to_netcdf(p, encoding={v: {"dtype": "int16"} for v in OISST_VARS})
        paths.append(p)
    return paths


def test_multivar_oisst_concat(tmp_path):
    paths = _oisst_sources(tmp_path, 3)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    out = _write(ds, paths, tmp_path / "o.vrt")
    g = _open_vrt(out)
    assert sorted(g.GetMDArrayNames()) == sorted(
        ["time", "zlev", "lat", "lon", *OISST_VARS])
    for v in OISST_VARS:
        a = g.OpenMDArray(v)
        assert [d.GetName() for d in a.GetDimensions()] == ["time", "zlev", "lat", "lon"]
        assert a.GetScale() == pytest.approx(0.01)
        got = a.GetUnscaled().ReadAsArray()
        want = ds[v].values
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=v)
        np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)],
                                   err_msg=v)
        assert a.GetAttribute("long_name").Read() == f"{v} long name"
    np.testing.assert_allclose(g.OpenMDArray("time").ReadAsArray(), [0, 1, 2])


def test_multivar_static_var_and_grid_mapping(tmp_path):
    paths = _oisst_sources(tmp_path, 2, with_mask=True, with_crs=True)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time",
                           data_vars="minimal", coords="minimal", compat="override")
    assert "time" not in ds["mask"].dims
    out = _write(ds, paths, tmp_path / "o.vrt")
    g = _open_vrt(out)
    names = g.GetMDArrayNames()
    assert "crs" not in names  # grid_mapping container -> <SRS>, not an array
    np.testing.assert_array_equal(g.OpenMDArray("mask").ReadAsArray(),
                                  ds["mask"].values)
    for v in OISST_VARS:
        srs = g.OpenMDArray(v).GetSpatialRef()
        assert srs is not None and srs.GetDataAxisToSRSAxisMapping() == [3, 4]
    assert g.OpenMDArray("mask").GetSpatialRef().GetDataAxisToSRSAxisMapping() == [1, 2]


def test_multivar_variables_subset_and_order(tmp_path):
    paths = _oisst_sources(tmp_path, 2)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    out = _write(ds, paths, tmp_path / "o.vrt", variables=["ice", "sst"])
    g = _open_vrt(out)
    assert [n for n in g.GetMDArrayNames() if n in OISST_VARS] == ["ice", "sst"]


def test_multivar_arrays_rename(tmp_path):
    """Source files call it 'sst' but the Dataset was renamed to 'temp'."""
    paths = _oisst_sources(tmp_path, 2)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    ds = ds[["sst", "ice"]].rename({"sst": "temp"})
    srcs = [{"path": str(p), "arrays": {"temp": "/sst"}} for p in paths]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = write_mdim_vrt(ds, srcs, tmp_path / "o.vrt",
                             composition="concat", concat_dim="time")
    a = _open_vrt(out).OpenMDArray("temp")
    np.testing.assert_allclose(np.nan_to_num(a.GetUnscaled().ReadAsArray()),
                               np.nan_to_num(ds["temp"].values))


def test_multivar_single_array_key_refused(tmp_path):
    from xvrt import VrtWriterError
    paths = _oisst_sources(tmp_path, 2)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    with pytest.raises(VrtWriterError, match="arrays"):
        write_mdim_vrt(ds, [{"path": str(p), "array": "/sst"} for p in paths],
                       tmp_path / "o.vrt", composition="concat", concat_dim="time")


def test_multivar_block_size_rank_mismatch(tmp_path):
    from xvrt import VrtWriterError
    paths = _oisst_sources(tmp_path, 2, with_mask=True)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time",
                           data_vars="minimal", coords="minimal", compat="override")
    with pytest.raises(VrtWriterError, match="rank"):
        _write(ds, paths, tmp_path / "o.vrt", block_size=[1, 1, 3, 4])
    out = _write(ds, paths, tmp_path / "o.vrt",
                 block_size={v: [1, 1, 3, 4] for v in OISST_VARS})
    assert "<BlockSize>1,1,3,4</BlockSize>" in out.read_text()


def test_multivar_stack_2d(tmp_path):
    paths = []
    for i in range(3):
        ds = xr.Dataset({"u": (("lat", "lon"), np.full((3, 4), i, "f4")),
                         "v": (("lat", "lon"), np.full((3, 4), -i, "f4"))},
                        coords={k: c for k, c in _coords(0, "d").items()
                                if k != "time"})
        p = tmp_path / f"s{i}.nc"
        ds.to_netcdf(p)
        paths.append(p)
    ds = xr.concat([xr.open_dataset(p) for p in paths], dim="time")
    ds = ds.assign_coords(time=[1.0, 2.0, 3.0])
    out = _write(ds, paths, tmp_path / "o.vrt", composition="stack")
    g = _open_vrt(out)
    for v in ("u", "v"):
        np.testing.assert_array_equal(g.OpenMDArray(v).ReadAsArray(), ds[v].values)


def test_accessor_forwards_variables(tmp_path):
    import xvrt.accessor  # noqa: F401
    paths = _oisst_sources(tmp_path, 2)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = ds.vrt.write_mdim(tmp_path / "o.vrt", sources=[str(p) for p in paths],
                                composition="concat", concat_dim="time",
                                variables=["err"])
    names = _open_vrt(out).GetMDArrayNames()
    assert "err" in names and "sst" not in names
