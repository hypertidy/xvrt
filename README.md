# xvrt

Write GDAL multidimensional VRT from xarray Datasets. Pure Python.

```python
## reticulate::use_python("~/.venv/bin/python3")
import xarray as xr
from xvrt import write_mdim_vrt

# Open N NetCDFs with numeric time (decode_times=False — VRT coords stay numeric).
#paths = sorted(glob.glob("bran/temp_*.nc"))
PREFIX = "https://thredds.nci.org.au/thredds/fileServer/gb6/BRAN/BRAN2020/daily"
VARS = ["atm_flux_diag", "ice_force", "ocean_eta_t", "ocean_force", "ocean_mld",
        "ocean_salt", "ocean_temp", "ocean_tx_trans_int_z", "ocean_ty_trans_int_z",
        "ocean_u", "ocean_v", "ocean_w"]
MONTHS = [f"{y}_{m:02d}" for y in range(2010, 2026) for m in range(1, 13)
          if (y, m) <= (2025, 4)]

urls = {v: [f"{PREFIX}/{v}_{ym}.nc" for ym in MONTHS] for v in VARS}
paths = urls["ocean_temp"][0:10]

ds = xr.open_mfdataset(paths, engine = "h5netcdf")

#pieces = [xr.open_dataset(p, decode_times=False) for p in paths]
#ds = xr.concat(pieces, dim="time")


# Emit a multidim VRT recipe.
write_mdim_vrt(
    ds[["temp"]],
    sources=[str(p) for p in paths],
    path="bran.vrt",
    composition="concat",
    concat_dim="Time",
    #crs = 'GEOGCRS[\"WGS 84\",\n    ENSEMBLE[\"World Geodetic System 1984 ensemble\",\n        MEMBER[\"World Geodetic System 1984 (Transit)\"],\n        MEMBER[\"World Geodetic System 1984 (G730)\"],\n        MEMBER[\"World Geodetic System 1984 (G873)\"],\n        MEMBER[\"World Geodetic System 1984 (G1150)\"],\n        MEMBER[\"World Geodetic System 1984 (G1674)\"],\n        MEMBER[\"World Geodetic System 1984 (G1762)\"],\n        MEMBER[\"World Geodetic System 1984 (G2139)\"],\n        MEMBER[\"World Geodetic System 1984 (G2296)\"],\n        ELLIPSOID[\"WGS 84\",6378137,298.257223563,\n            LENGTHUNIT[\"metre\",1]],\n        ENSEMBLEACCURACY[2.0]],\n    PRIMEM[\"Greenwich\",0,\n        ANGLEUNIT[\"degree\",0.0174532925199433]],\n    CS[ellipsoidal,2],\n        AXIS[\"geodetic latitude (Lat)\",north,\n            ORDER[1],\n            ANGLEUNIT[\"degree\",0.0174532925199433]],\n        AXIS[\"geodetic longitude (Lon)\",east,\n            ORDER[2],\n            ANGLEUNIT[\"degree\",0.0174532925199433]],\n    USAGE[\n        SCOPE[\"Horizontal component of 3D system.\"],\n        AREA[\"World.\"],\n        BBOX[-90,-180,90,180]],\n    ID[\"EPSG\",4326]]'
)




```

```
ds
<xarray.Dataset> Size: 335GB
Dimensions:         (Time: 304, nv: 2, st_ocean: 51, yt_ocean: 1500,
                     xt_ocean: 3600, st_edges_ocean: 52)
Coordinates:
  * Time            (Time) datetime64[ns] 2kB 2010-01-01T12:00:00 ... 2010-10...
  * nv              (nv) float64 16B 1.0 2.0
  * st_ocean        (st_ocean) float64 408B 2.5 7.5 12.5 ... 3.603e+03 4.509e+03
  * yt_ocean        (yt_ocean) float64 12kB -74.95 -74.85 -74.75 ... 74.85 74.95
  * xt_ocean        (xt_ocean) float64 29kB 0.05 0.15 0.25 ... 359.8 359.9 360.0
  * st_edges_ocean  (st_edges_ocean) float64 416B 0.0 5.0 ... 4.056e+03 5e+03
Data variables:
    average_T1      (Time) datetime64[ns] 2kB dask.array<chunksize=(1,), meta=np.ndarray>
    average_T2      (Time) datetime64[ns] 2kB dask.array<chunksize=(1,), meta=np.ndarray>
    average_DT      (Time) timedelta64[ns] 2kB dask.array<chunksize=(1,), meta=np.ndarray>
    Time_bounds     (Time, nv) timedelta64[ns] 5kB dask.array<chunksize=(1, 2), meta=np.ndarray>
    temp            (Time, st_ocean, yt_ocean, xt_ocean) float32 335GB dask.array<chunksize=(1, 1, 300, 300), meta=np.ndarray>
Attributes:
    filename:           TMP/ocean_temp_2010_01_01.nc.0000
```

```

```




## Why

VRT is a text spec, not a GDAL artefact. A pure-Python writer makes that
concrete. See [`mdim-vrt-writer-plan.md`](./mdim-vrt-writer-plan.md) for
the full rationale, rules (R1..R7), and scope.

## Relationship to other hypertidy packages

- **`vrtstack`** (R) — the parallel. `gdal_mdim_stack()` stacks classic 2D
  rasters using `<SourceBand>` + `<SourceTranspose>-1,0,1</SourceTranspose>`.
  xvrt stacks **mdim sources** (NetCDF/HDF5/Zarr/nested VRT) using
  `<SourceArray>/name</SourceArray>` and no transpose. The two writers
  target the same XSD but cover disjoint source-kind surfaces; they
  intersect via VRT-of-VRT composition.
- **`gdx`** — VRT → xarray reader. xvrt is the writing half.
- **`VirtualiZarr`** — architectural precedent: emit descriptive artefacts
  from a lazy xarray Dataset without calling the source-file libraries.

## Install

```
pip install -e .
pip install -e .[test]   # adds lxml + netCDF4 + pytest
pip install -e .[rio]    # adds rioxarray (extra CRS source)
```

## Run tests

```
pytest tests/
# then, where GDAL 3.12 is available:
python scripts/validate_roundtrip.py
# and for schema conformance (once schema/gdalvrt.xsd is in place):
python scripts/validate_xsd.py /path/to/output.vrt
```

## v0 scope (shipped)

- Compositions: **stack** (new dim, 1 element per source) and **concat**
  (existing dim, N elements per source).
- Coord carriers: **regular-spacing detection**, **inline values** below
  threshold, **source-ref** above. Per-coord override via
  `coord_mode={"lat": "source", ...}`.
- CRS: explicit WKT, or detected from `da.attrs["crs_wkt"]` /
  `da.rio.crs` / CF `grid_mapping` attr.
- Rule enforcement: R3 / R4 / R5 / R6 / R7 with clear error messages.
- Optional accessor: `import xvrt.accessor` then `ds.vrt.write_mdim(...)`.

## v1 (next)

- Mosaic composition with explicit overlap policy.
- Source-ref coord mode using a dedicated sidecar source (not just the
  first data source).
- Zarr and nested-VRT source demos.
- Byte-convergence test against `vrtstack::write_mdim_vrt()` in R.
