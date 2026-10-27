# xvrt

Write [GDAL multidimensional VRT](https://gdal.org/en/stable/drivers/raster/vrt_multidimensional.html)
from xarray Datasets and STAC catalogues. Pure Python: no GDAL needed to
write.

Documentation: <https://rgdal-dev.github.io/xvrt/>

A multidimensional VRT is a small XML file that describes a collection of
files as one n-dimensional dataset: which file supplies which slab of which
array, plus dimensions, coordinates, CRS, units and packing. It holds no data.
Any reader of GDAL's multidimensional API opens the whole collection as a
single cube and reads only the files a request touches.

## Install

```bash
pip install xvrt
pip install "xvrt[stac]"   # adds pyarrow and pyproj for from_stac()
```

Reading the result needs GDAL 3.x with multidimensional support (for example
GDAL's Python bindings from conda-forge).

## From xarray

Open files the way you normally would, then describe the result:

```python
import glob
import xarray as xr
from xvrt import write_mdim_vrt

paths = sorted(glob.glob("oisst-avhrr-v02r01.*.nc"))   # one day per file
ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time")

write_mdim_vrt(ds, paths, "oisst.vrt", composition="concat", concat_dim="time")
```

The same works over HTTP. Ten monthly files of one variable from a THREDDS
server (URLs become `/vsicurl/` sources):

```python
PREFIX = "https://thredds.nci.org.au/thredds/fileServer/gb6/BRAN/BRAN2020/daily"
months = [f"2010_{m:02d}" for m in range(1, 11)]
paths = [f"{PREFIX}/ocean_temp_{ym}.nc" for ym in months]

ds = xr.open_mfdataset(paths, engine="h5netcdf")
write_mdim_vrt(ds[["temp"]], paths, "bran.vrt",
               composition="concat", concat_dim="Time")
```

## From a STAC catalogue

`from_stac()` builds the VRT from catalogue fields alone (grid, CRS, item
datetimes, per-asset band storage) without opening a file:

```python
from xvrt import from_stac

from_stac("https://data.source.coop/ausantarctic/ghrsst-mur-v2/ghrsst-mur-v2.parquet",
          "mur.vrt", block_size=[1, 512, 512])
```

For the GHRSST MUR collection that is five arrays over 8913 days in a few
seconds.

## What it handles

- **Compositions:** `concat` (files already have the dimension) and `stack`
  (one file per position of a new dimension), with explicit positions and
  gaps.
- **Packed data:** storage dtype, scale, offset and nodata are recovered from
  xarray's `.encoding`, so GDAL decodes the stored bytes exactly as xarray did.
- **Time:** decoded times are re-encoded with their original CF units and
  calendar.
- **Multiple variables:** in the same files, or one source list per variable.
- **Raster sources:** GeoTIFF/COG bands via `<SourceBand>` and
  `<SourceTranspose>`.
- **Lazy reads:** every source states its extent (`<SourceSlab>`), so a read
  opens only the files it needs.
- **CRS:** `<SRS>` with the data-axis-to-SRS-axis mapping computed from the
  WKT axis order.

## Reading the result

```bash
gdal mdim info oisst.vrt
```

```python
from osgeo import gdal
ds = gdal.OpenEx("oisst.vrt", gdal.OF_MULTIDIM_RASTER)
sst = ds.GetRootGroup().OpenMDArray("sst").GetUnscaled()
```

or with an xarray backend for GDAL, such as
[gdalxarray](https://github.com/hypertidy/gdalxarray):
`xr.open_dataset("oisst.vrt", engine="gdalxarray")`.

## Development

```bash
pip install -e ".[test]"
pytest tests/
```

`tests/test_gdal_roundtrip.py` reads every written VRT back through GDAL and
validates it against GDAL's schema (`schema/gdalvrt.xsd`); it is skipped when
`osgeo.gdal` is not importable. With conda-forge:

```bash
micromamba create -n xvrt -c conda-forge python gdal libgdal-netcdf xarray netcdf4 h5netcdf dask lxml pyarrow pyproj pytest
```

The design notes are in [mdim-vrt-writer-plan.md](mdim-vrt-writer-plan.md);
the documentation site source is in [docs/](docs/).

## Related

- [hypertidy/vrtstack](https://github.com/hypertidy/vrtstack): the R
  counterpart, writing the same VRT format from raster collections and
  catalogues.
- [hypertidy/gdalxarray](https://github.com/hypertidy/gdalxarray): an xarray
  backend for GDAL that opens these VRTs.

## License

MIT
