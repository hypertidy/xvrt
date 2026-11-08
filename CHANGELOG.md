# Changelog

## 0.2.0

- `stack_rasters()`: stack 2D raster files (GeoTIFF/COG, local or URL)
  along a new dimension without constructing an xarray input -- the
  vrtstack mode. Dimension values come from a sequence, a filename regex
  (with optional `format=` date parsing), a callable, or `start=`/`step=`;
  the grid comes from the first file (rasterio or osgeo.gdal) or an
  explicit `grid=` so no file is opened.
- `open_raster_stack()`: the same, returning an xarray Dataset via a
  GDAL-backed engine (default `gdalxarray`).
- README: links that pointed at repository-relative paths are absolute,
  so they work on PyPI's project page.

## 0.1.0

First release.

- `write_mdim_vrt()`: GDAL multidimensional VRT from an xarray Dataset.
  - Compositions `concat` (files already carry the dimension) and `stack`
    (one file per position of a new dimension, with an optional `index` per
    source for gaps).
  - Every data variable becomes an array; sources may be shared or given per
    variable (`sources={"var": [...]}`); `variables=` selects and orders.
  - NetCDF/HDF5/Zarr array sources (`<SourceArray>`) and GeoTIFF/COG band
    sources (`<SourceBand>` + `<SourceTranspose>`).
  - Packed data: storage dtype, scale, offset and nodata from `.encoding`.
  - Time: decoded datetime and cftime coordinates re-encoded with their CF
    units and calendar.
  - `<SourceSlab>` extents on every source, so reads open only what they need.
  - `<SRS>` with `dataAxisToSRSAxisMapping` computed from the WKT axis order.
  - Coordinates as start/step when regular, inline values otherwise, or read
    from the sources.
- `from_stac()`: the same from STAC items or a stac-geoparquet file, using
  catalogue fields only (projection and raster extensions).
- `Dataset.vrt.write_mdim()` accessor (`import xvrt.accessor`).
- Output validates against GDAL's `gdalvrt.xsd`, except mdim `<BlockSize>`,
  written comma-separated as GDAL's driver requires.
