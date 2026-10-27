# xvrt documentation site

Quarto website, published to GitHub Pages by `.github/workflows/docs.yml`.

Pages execute Python (xvrt, xarray, GDAL, and for the GHRSST example network
access to source.coop). Results are frozen in `_freeze/` and committed, so CI
renders without executing. After changing a page, re-render it locally and
commit the updated `_freeze/`:

```bash
pip install -e .. quarto-cli jupyter matplotlib pandas pyarrow gdalxarray
quarto render          # whole site; or: quarto render examples/ghrsst-mur.qmd
quarto preview         # live preview
```

GDAL's Python bindings (`osgeo`) are easiest from conda-forge.
