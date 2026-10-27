#!/usr/bin/env python3
"""XSD validation against the pinned ``schema/gdalvrt.xsd``.

Reads every ``*.vrt`` emitted by the smoke tests and validates it against
the pinned schema (see schema/README.md).

Usage
-----
    # one-time: drop the XSD in place
    curl -o schema/gdalvrt.xsd \\
      https://raw.githubusercontent.com/OSGeo/gdal/refs/heads/release/3.12/frmts/vrt/data/gdalvrt.xsd

    # then run
    python scripts/validate_xsd.py path/to/output.vrt [more.vrt ...]
"""
from __future__ import annotations

import sys
from pathlib import Path


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "gdalvrt.xsd"


def main(argv: list[str]) -> int:
    try:
        from lxml import etree
    except ImportError:
        print("ERROR: lxml not installed. `pip install lxml`.")
        return 1

    if not SCHEMA_PATH.exists():
        print(f"ERROR: schema not found at {SCHEMA_PATH}")
        print(
            "Fetch it with:\n"
            "  curl -o schema/gdalvrt.xsd "
            "https://raw.githubusercontent.com/OSGeo/gdal/refs/heads/release/3.12/"
            "frmts/vrt/data/gdalvrt.xsd"
        )
        return 1

    schema_doc = etree.parse(str(SCHEMA_PATH))
    schema = etree.XMLSchema(schema_doc)

    if not argv:
        print("Usage: validate_xsd.py <file.vrt> [...]")
        return 1

    any_fail = False
    for path_str in argv:
        p = Path(path_str)
        doc = etree.parse(str(p))
        if schema.validate(doc):
            print(f"OK  {p}")
        else:
            any_fail = True
            print(f"FAIL {p}")
            for err in schema.error_log:
                print(f"   L{err.line}:{err.column}  {err.message}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
