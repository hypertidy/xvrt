# Pinned schema

`gdalvrt.xsd` here is GDAL's own schema, copied from the GDAL 3.13.3
conda-forge build (`share/gdal/gdalvrt.xsd`). Upstream source:

    https://github.com/OSGeo/gdal/blob/master/frmts/vrt/data/gdalvrt.xsd

`tests/test_gdal_roundtrip.py` validates every VRT it writes against this
file; the writer itself never reads it.

Known disagreement: the schema types mdim `<BlockSize>` as a space-separated
`integerList`, but the VRT driver only parses comma-separated values. xvrt
follows the driver, and the tests ignore that one schema error.
