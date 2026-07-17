"""
ASCII file I/O handlers for IWFM model files.

This module provides functions for reading and writing IWFM ASCII input files
including node coordinates, element definitions, and stratigraphy data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.mesh import Element, Node
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.io.iwfm_reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.iwfm_reader import (
    strip_inline_comment as _strip_comment,
)

logger = logging.getLogger(__name__)


def read_nodes(filepath: Path | str) -> dict[int, Node]:
    """
    Read node coordinates from an IWFM node file.

    Expected format (C2VSimFG format):
        NNODES                    # Number of nodes
        FACT                      # Conversion factor (optional)
        ID  X  Y                  (one line per node)

    Args:
        filepath: Path to the node file

    Returns:
        Dictionary mapping node ID to Node object

    Raises:
        FileFormatError: If file format is invalid
    """
    filepath = Path(filepath)
    nodes: dict[int, Node] = {}

    with open(filepath) as f:
        # Skip comment lines and find NNODES
        n_nodes = None
        fact = 1.0  # Conversion factor, default 1.0
        line_num = 0

        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            # First non-comment line should be NNODES
            value_str, desc = _strip_comment(line)
            try:
                n_nodes = int(value_str)
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid NNODES value: '{value_str}'", line_number=line_num
                ) from e
            break

        if n_nodes is None:
            raise FileFormatError("Could not find NNODES in file")

        # Check for optional FACT (conversion factor) line
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, desc = _strip_comment(line)
            # Check if this is a conversion factor line (single float, not 3 values)
            parts = value_str.split()
            if len(parts) == 1:
                try:
                    fact = float(parts[0])
                    # It's a conversion factor, continue to node data
                    break
                except ValueError:
                    pass
            # If we got here with 3+ values, it's node data - process it
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    x = float(parts[1]) * fact
                    y = float(parts[2]) * fact
                    nodes[node_id] = Node(id=node_id, x=x, y=y)
                except ValueError as e:
                    raise FileFormatError(
                        f"Invalid node data: '{line.strip()}'", line_number=line_num
                    ) from e
            break

        # Read remaining node data
        nodes_read = len(nodes)
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, _ = _strip_comment(line)
            parts = value_str.split()
            if len(parts) < 3:
                raise FileFormatError(
                    f"Invalid node line format (expected ID X Y): '{line.strip()}'",
                    line_number=line_num,
                )

            try:
                node_id = int(parts[0])
                x = float(parts[1]) * fact
                y = float(parts[2]) * fact
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid node data: '{line.strip()}'", line_number=line_num
                ) from e

            nodes[node_id] = Node(id=node_id, x=x, y=y)
            nodes_read += 1

        if nodes_read != n_nodes:
            raise FileFormatError(
                f"Node count mismatch: expected {n_nodes} nodes, got {nodes_read}"
            )

    return nodes, fact


def read_elements(
    filepath: Path | str,
) -> tuple[dict[int, Element], int, dict[int, str]]:
    """
    Read element definitions from an IWFM element file.

    Expected format:
        NELEM                     / Number of elements
        NSUBREGION                / Number of subregions
        ID  NAME                  (one line per subregion)
        ID  V1  V2  V3  V4  SR   (one line per element, V4=0 for triangles)

    Args:
        filepath: Path to the element file

    Returns:
        Tuple of (elements dict, number of subregions, subregion names dict).
        The names dict maps subregion ID to name string (may be empty if no
        names could be parsed from the file).

    Raises:
        FileFormatError: If file format is invalid
    """
    filepath = Path(filepath)
    elements: dict[int, Element] = {}

    with open(filepath) as f:
        line_num = 0
        n_elem = None
        n_subregion = None

        # Read NELEM
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, _ = _strip_comment(line)
            try:
                n_elem = int(value_str)
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid NELEM value: '{value_str}'", line_number=line_num
                ) from e
            break

        # Read NSUBREGION
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, _ = _strip_comment(line)
            try:
                n_subregion = int(value_str)
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid NSUBREGION value: '{value_str}'", line_number=line_num
                ) from e
            break

        if n_elem is None:
            raise FileFormatError("Could not find NELEM in file")
        if n_subregion is None:
            raise FileFormatError("Could not find NSUBREGION in file")

        # Read subregion names (IWFM format has RNAME lines after NSUBREGION)
        subregion_names: dict[int, str] = {}
        subregion_names_read = 0
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, desc = _strip_comment(line)
            parts = value_str.split()

            # If we've already read all subregion names, this must be
            # element data — parse it and exit the subregion loop.
            if subregion_names_read >= n_subregion:
                if len(parts) >= 6:
                    try:
                        elem_id = int(parts[0])
                        v1 = int(parts[1])
                        v2 = int(parts[2])
                        v3 = int(parts[3])
                        v4 = int(parts[4])
                        subregion = int(parts[5])
                        verts: tuple[int, ...]
                        if v4 == 0:
                            verts = (v1, v2, v3)
                        else:
                            verts = (v1, v2, v3, v4)
                        elements[elem_id] = Element(
                            id=elem_id,
                            vertices=verts,
                            subregion=subregion,
                        )
                    except ValueError:
                        pass
                break

            # Check if this looks like element data (6 integers)
            # or subregion name (text that doesn't parse as 6 integers)
            if len(parts) >= 6:
                try:
                    # Try to parse as element data
                    elem_id = int(parts[0])
                    v1 = int(parts[1])
                    v2 = int(parts[2])
                    v3 = int(parts[3])
                    v4 = int(parts[4])
                    subregion = int(parts[5])

                    # It's valid element data - process it
                    elem_verts: tuple[int, ...]
                    if v4 == 0:
                        elem_verts = (v1, v2, v3)
                    else:
                        elem_verts = (v1, v2, v3, v4)
                    elements[elem_id] = Element(
                        id=elem_id, vertices=elem_verts, subregion=subregion
                    )
                    break
                except ValueError:
                    # Not element data, must be a subregion name
                    subregion_names_read += 1
                    # Try to extract ID and name: "ID  Name text"
                    try:
                        sr_id = int(parts[0])
                        sr_name = " ".join(parts[1:]).strip()
                        subregion_names[sr_id] = sr_name
                    except ValueError:
                        # Whole line is the name, use 1-based index
                        subregion_names[subregion_names_read] = value_str.strip()
            else:
                # Subregion name (less than 6 parts)
                subregion_names_read += 1
                # Try "ID  Name" format
                if parts:
                    try:
                        sr_id = int(parts[0])
                        sr_name = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
                        subregion_names[sr_id] = sr_name
                    except ValueError:
                        subregion_names[subregion_names_read] = value_str.strip()

        # Read remaining element data
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, _ = _strip_comment(line)
            parts = value_str.split()
            if len(parts) < 6:
                raise FileFormatError(
                    f"Invalid element line format: '{line.strip()}'",
                    line_number=line_num,
                )

            try:
                elem_id = int(parts[0])
                v1 = int(parts[1])
                v2 = int(parts[2])
                v3 = int(parts[3])
                v4 = int(parts[4])
                subregion = int(parts[5])
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid element data: '{line.strip()}'", line_number=line_num
                ) from e

            # Determine if triangle or quad
            raw_verts: tuple[int, ...]
            if v4 == 0:
                raw_verts = (v1, v2, v3)
            else:
                raw_verts = (v1, v2, v3, v4)

            elements[elem_id] = Element(id=elem_id, vertices=raw_verts, subregion=subregion)

    if len(elements) != n_elem:
        logger.warning(
            "Element file %s declares NELEM=%d but %d elements were read",
            filepath,
            n_elem,
            len(elements),
        )

    return elements, n_subregion, subregion_names


def read_stratigraphy(filepath: Path | str) -> Stratigraphy:
    """
    Read stratigraphy data from an IWFM stratigraphy file.

    IWFM format (matches Fortran Class_Stratigraphy.f90):
        NL                        # Number of layers
        FACT                      # Conversion factor
        ID  GS  W(1)  W(2) ...    (one line per node)

    Where W values are alternating aquitard/aquifer thicknesses:
        W(1) = aquitard thickness layer 1
        W(2) = aquifer thickness layer 1
        W(3) = aquitard thickness layer 2
        W(4) = aquifer thickness layer 2
        etc.

    Top and bottom elevations are computed from GS and cumulative thicknesses.

    Args:
        filepath: Path to the stratigraphy file

    Returns:
        Stratigraphy object

    Raises:
        FileFormatError: If file format is invalid
    """
    filepath = Path(filepath)

    with open(filepath) as f:
        line_num = 0
        n_layers = None
        fact = 1.0

        # Read NLAYERS (NL in file)
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, _ = _strip_comment(line)
            try:
                n_layers = int(value_str)
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid NLAYERS value: '{value_str}'", line_number=line_num
                ) from e
            break

        # Read FACT (conversion factor)
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, _ = _strip_comment(line)
            try:
                fact = float(value_str)
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid FACT value: '{value_str}'", line_number=line_num
                ) from e
            break

        if n_layers is None:
            raise FileFormatError("Could not find NLAYERS in file")

        # Collect node data first to determine n_nodes
        node_data: list[tuple[int, list[float]]] = []

        # Expected columns: ID, GS, then W(1), W(2), ... W(2*NL)
        expected_cols = 2 + n_layers * 2

        # Read node stratigraphy data
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value_str, _ = _strip_comment(line)
            parts = value_str.split()
            if len(parts) < expected_cols:
                raise FileFormatError(
                    f"Invalid stratigraphy line (expected {expected_cols} columns): "
                    f"'{line.strip()}'",
                    line_number=line_num,
                )

            try:
                node_id = int(parts[0])
                values = [float(p) * fact for p in parts[1:expected_cols]]
                node_data.append((node_id, values))
            except (ValueError, IndexError) as e:
                raise FileFormatError(
                    f"Invalid stratigraphy data: '{line.strip()}'", line_number=line_num
                ) from e

        # Determine n_nodes from data
        n_nodes = len(node_data)
        if n_nodes == 0:
            raise FileFormatError("No stratigraphy data found in file")

        # Node IDs in the stratigraphy file are not guaranteed to be
        # 1-based-contiguous, so we can't rely on ``idx = node_id - 1``.
        # Rows follow file order; duplicate IDs are an error.
        id_to_idx: dict[int, int] = {}
        for row_idx, (node_id, _values) in enumerate(node_data):
            if node_id in id_to_idx:
                raise FileFormatError(f"Duplicate node ID {node_id} in stratigraphy file")
            id_to_idx[node_id] = row_idx

        # Unpack file-order rows into a per-node arrays: ground surface,
        # alternating aquitard/aquifer thicknesses (W(2k) = aquitard k,
        # W(2k+1) = aquifer k). Delegate the thickness -> elevation math
        # to Stratigraphy.from_thicknesses so the loop lives in one place.
        gs_elev = np.zeros(n_nodes, dtype=np.float64)
        aquitard_thick = np.zeros((n_nodes, n_layers), dtype=np.float64)
        aquifer_thick = np.zeros((n_nodes, n_layers), dtype=np.float64)

        for node_id, values in node_data:
            idx = id_to_idx[node_id]
            gs_elev[idx] = values[0]
            for layer in range(n_layers):
                aquitard_thick[idx, layer] = values[1 + layer * 2]
                aquifer_thick[idx, layer] = values[1 + layer * 2 + 1]

    return Stratigraphy.from_thicknesses(
        gs_elev=gs_elev,
        aquitard_thicknesses=aquitard_thick,
        aquifer_thicknesses=aquifer_thick,
    )


def write_nodes(
    filepath: Path | str,
    nodes: dict[int, Node],
    header: str | None = None,
) -> None:
    """
    Write node coordinates to an IWFM node file.

    Args:
        filepath: Path to output file
        nodes: Dictionary mapping node ID to Node object
        header: Optional header comment (default generates one)
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        # Write header
        if header:
            for line in header.strip().split("\n"):
                f.write(f"C  {line}\n")
        else:
            f.write("C  Node data file generated by pyiwfm\n")
            f.write("C  ID             X              Y\n")

        # Write node count
        f.write(f"{len(nodes):<10}                    / NNODES\n")

        # Write nodes in ID order
        for node_id in sorted(nodes.keys()):
            node = nodes[node_id]
            f.write(f"{node.id:<5} {node.x:>14.6f} {node.y:>14.6f}\n")


def write_elements(
    filepath: Path | str,
    elements: dict[int, Element],
    n_subregions: int,
    header: str | None = None,
) -> None:
    """
    Write element definitions to an IWFM element file.

    Args:
        filepath: Path to output file
        elements: Dictionary mapping element ID to Element object
        n_subregions: Number of subregions
        header: Optional header comment
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        # Write header
        if header:
            for line in header.strip().split("\n"):
                f.write(f"C  {line}\n")
        else:
            f.write("C  Element data file generated by pyiwfm\n")
            f.write("C  ID  V1  V2  V3  V4  SR\n")

        # Write counts
        f.write(f"{len(elements):<10}                    / NELEM\n")
        f.write(f"{n_subregions:<10}                    / NSUBREGION\n")

        # Write elements in ID order
        for elem_id in sorted(elements.keys()):
            elem = elements[elem_id]
            v1, v2, v3 = elem.vertices[:3]
            v4 = elem.vertices[3] if elem.is_quad else 0
            f.write(f"{elem.id:<5} {v1:>5} {v2:>5} {v3:>5} {v4:>5} {elem.subregion:>3}\n")


def write_stratigraphy(
    filepath: Path | str,
    stratigraphy: Stratigraphy,
    header: str | None = None,
) -> None:
    """
    Write stratigraphy data to an IWFM stratigraphy file.

    IWFM format uses THICKNESSES, not elevations:
        NL                        / Number of layers
        FACT                      / Conversion factor
        ID  GS  W(1) W(2) ...     (one line per node)

    Where W values are alternating aquitard/aquifer thicknesses:
        W(1) = aquitard thickness layer 1 (gs - top_layer_1)
        W(2) = aquifer thickness layer 1 (top_layer_1 - bottom_layer_1)
        W(3) = aquitard thickness layer 2 (bottom_layer_1 - top_layer_2)
        W(4) = aquifer thickness layer 2 (top_layer_2 - bottom_layer_2)
        etc.

    Args:
        filepath: Path to output file
        stratigraphy: Stratigraphy object
        header: Optional header comment
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        # Write header
        if header:
            for line in header.strip().split("\n"):
                f.write(f"C  {line}\n")
        else:
            f.write("C  Stratigraphy data file generated by pyiwfm\n")
            layer_cols = "  ".join(
                [f"AQT{i + 1}  AQF{i + 1}" for i in range(stratigraphy.n_layers)]
            )
            f.write(f"C  ID  GS  {layer_cols}\n")

        # Write layer count and factor (IWFM format)
        f.write(f"{stratigraphy.n_layers:<10}                    / NLAYERS\n")
        f.write(f"{'1.0':>14}                          / FACTEL\n")

        # Precompute aquitard/aquifer thicknesses via the canonical helpers
        # so the math has a single owner.
        aquitard_thicks = stratigraphy.get_all_aquitard_thicknesses()
        aquifer_thicks = stratigraphy.top_elev - stratigraphy.bottom_elev

        # Write node data with thicknesses
        for idx in range(stratigraphy.n_nodes):
            node_id = idx + 1  # 1-based node ID
            gs = stratigraphy.gs_elev[idx]

            line = f"{node_id:<5} {gs:>10.4f}"
            for layer in range(stratigraphy.n_layers):
                line += f" {aquitard_thicks[idx, layer]:>10.4f} {aquifer_thicks[idx, layer]:>10.4f}"

            f.write(line + "\n")
