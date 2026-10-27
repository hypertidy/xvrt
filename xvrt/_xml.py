"""XML emission helpers.

Thin wrapper over ``xml.etree.ElementTree`` that encodes the XSD's
child-ordering rules once. Build elements by whatever means is convenient,
then call :func:`sort_children` before serialising.

No ``lxml`` here. ``lxml`` appears only in the test suite, for XSD
validation.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Iterable, Sequence


# XSD-mandated child order inside <Array>. The value-carrier can be any of
# RegularlySpacedValues / ConstantValue / InlineValues / InlineValuesWithValueElement
# / Source (mutually exclusive in a given Array in practice).
_ARRAY_CHILD_ORDER: tuple[str, ...] = (
    "DataType",
    "Dimension",
    "DimensionRef",
    "BlockSize",
    "SRS",
    "Unit",
    "NoDataValue",
    "Offset",
    "Scale",
    "RegularlySpacedValues",
    "ConstantValue",
    "InlineValues",
    "InlineValuesWithValueElement",
    "Source",
    "Attribute",
)

# XSD order for <Group> children.
_GROUP_CHILD_ORDER: tuple[str, ...] = (
    "Dimension",
    "Attribute",
    "Array",
    "Group",
)

# XSD order for <Source> children.
_SOURCE_CHILD_ORDER: tuple[str, ...] = (
    "SourceFilename",
    "SourceArray",
    "SourceBand",
    "SourceTranspose",
    "SourceView",
    "SourceSlab",
    "DestSlab",
)


def _order_key(order: Sequence[str]):
    idx = {name: i for i, name in enumerate(order)}

    def key(el: ET.Element) -> int:
        try:
            return idx[el.tag]
        except KeyError as exc:
            raise ValueError(
                f"unexpected child <{el.tag}>; not in the ordering map"
            ) from exc
    return key


def sort_children(root: ET.Element) -> None:
    """Sort children of Array/Group/Source nodes in-place into XSD order.

    Recurses the whole tree. Idempotent. Safe to call twice.
    """
    for elem in root.iter():
        if elem.tag == "Array":
            elem[:] = sorted(elem, key=_order_key(_ARRAY_CHILD_ORDER))
        elif elem.tag == "Group":
            elem[:] = sorted(elem, key=_order_key(_GROUP_CHILD_ORDER))
        elif elem.tag == "Source":
            elem[:] = sorted(elem, key=_order_key(_SOURCE_CHILD_ORDER))


def indent(elem: ET.Element, level: int = 0, space: str = "  ") -> None:
    """Pretty-print an ElementTree in place.

    Equivalent to ``ET.indent()`` on 3.9+, but spelled out here for symmetry
    with older Pythons and to give us control over whitespace without pulling
    in ``lxml``.
    """
    # 3.9+ shortcut.
    try:
        ET.indent(elem, space=space, level=level)
        return
    except AttributeError:
        pass
    # Fallback.
    pad = "\n" + level * space
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + space
        for child in elem:
            indent(child, level + 1, space)
        if not elem.tail or not elem.tail.strip():
            elem[-1].tail = pad
    if not elem.tail or not elem.tail.strip():
        elem.tail = pad


def tostring(root: ET.Element) -> bytes:
    """Serialise to XML bytes with a standard declaration."""
    sort_children(root)
    indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def sub(parent: ET.Element, tag: str, text: str | None = None, **attrs) -> ET.Element:
    """Convenience: append a child with optional text and attrs."""
    # Filter None attrs so we don't emit e.g. type="None".
    clean = {k: _attrstr(v) for k, v in attrs.items() if v is not None}
    el = ET.SubElement(parent, tag, clean)
    if text is not None:
        el.text = str(text)
    return el


def _attrstr(v) -> str:
    # Attributes are always strings in XML; bool → "0"/"1", ints → str.
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def int_list(values: Iterable[int]) -> str:
    """Space-separated integer list, the XSD ``integerList`` shape.

    Currently unused: mdim ``<BlockSize>`` is typed ``integerList`` in the
    XSD, but GDAL's VRT driver only parses it comma-separated, so the writer
    uses :func:`comma_list` there and follows the driver.
    """
    return " ".join(str(int(v)) for v in values)


def comma_list(values: Iterable[int]) -> str:
    """Comma-separated integer list.

    Used for ``<DestSlab offset=...>`` — GDAL's mdim VRT driver parses the
    offset attribute as comma-separated per-dimension offsets.
    """
    return ",".join(str(int(v)) for v in values)


def repr_double(x: float) -> str:
    """Shortest round-trippable repr of a double.

    Python's ``repr(float)`` gives the shortest string that round-trips to the
    same IEEE-754 value. That's what we want for coord fidelity —
    ``%.17g``-style emission produces spurious trailing digits.
    """
    return repr(float(x))
