"""numpy dtype -> GDAL VRT DataType string.

In the multidim XSD, ``<DataType>`` is ``xs:string`` -- not constrained to the
classic ``DataTypeType`` enumeration -- so technically we could emit anything.
In practice we stick to canonical GDAL names so the output round-trips through
``osgeo.gdal``'s mdim API without surprises.
"""
from __future__ import annotations

import numpy as np

from ._errors import VrtWriterError


# Mirrors ``_gdal_to_vrt_dtype`` in vrtstack/R. Kept aligned deliberately --
# the two writers should produce the same <DataType> string for the same
# logical dtype.
_NUMPY_TO_VRT = {
    "bool": "Byte",      # GDAL has no bool; promote.
    "int8": "Int8",
    "uint8": "Byte",
    "int16": "Int16",
    "uint16": "UInt16",
    "int32": "Int32",
    "uint32": "UInt32",
    "int64": "Int64",
    "uint64": "UInt64",
    "float16": "Float16",
    "float32": "Float32",
    "float64": "Float64",
    "complex64": "CFloat32",
    "complex128": "CFloat64",
}


def numpy_to_vrt_dtype(dtype: np.dtype | str) -> str:
    """Map a numpy dtype to a canonical GDAL VRT DataType name.

    Raises :class:`VrtWriterError` for datetime / timedelta / string / object
    dtypes. See :func:`xvrt.encode_datetime_coord` for the datetime escape
    hatch.
    """
    dt = np.dtype(dtype)
    name = dt.name

    if name in _NUMPY_TO_VRT:
        return _NUMPY_TO_VRT[name]

    kind = dt.kind
    if kind == "M":
        raise VrtWriterError(
            f"datetime64 dtype ({dt}) not supported in v0. Convert the coord "
            "to numeric first, e.g. via xvrt.encode_datetime_coord(da, "
            "unit='days', since='1970-01-01'), and attach CF units in attrs."
        )
    if kind == "m":
        raise VrtWriterError(f"timedelta64 dtype ({dt}) not supported in v0.")
    if kind in ("U", "S", "O"):
        raise VrtWriterError(
            f"string/object dtype ({dt}) not supported by the VRT mdim model "
            "used here. Convert categorical strings to integer codes first."
        )
    raise VrtWriterError(f"unsupported dtype: {dt}")
