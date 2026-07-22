"""Unit tests for ASCII I/O handlers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.io.ascii import (
    read_elements,
    read_nodes,
    read_stratigraphy,
    write_elements,
    write_nodes,
    write_stratigraphy,
)


class TestReadNodes:
    """Tests for reading node files."""

    def test_read_nodes_basic(self, tmp_path: Path) -> None:
        """Test reading a basic node file."""
        node_file = tmp_path / "nodes.dat"
        node_file.write_text(
            """C  Node data file for testing
C  ID      X          Y
9                         / NNODES
1       0.0       0.0
2     100.0       0.0
3     200.0       0.0
4       0.0     100.0
5     100.0     100.0
6     200.0     100.0
7       0.0     200.0
8     100.0     200.0
9     200.0     200.0
"""
        )

        nodes, _ = read_nodes(node_file)

        assert len(nodes) == 9
        assert nodes[1].x == 0.0
        assert nodes[1].y == 0.0
        assert nodes[5].x == 100.0
        assert nodes[5].y == 100.0

    def test_read_nodes_with_comments(self, tmp_path: Path) -> None:
        """Test reading nodes file with various comment styles."""
        node_file = tmp_path / "nodes.dat"
        # Note: IWFM only recognizes C, c, * as comment characters in column 1
        node_file.write_text(
            """C  This is a comment
*  This is also a comment
c  Lowercase c also works
4                         / NNODES
1       0.0       0.0
2     100.0       0.0
3     100.0     100.0
4       0.0     100.0
"""
        )

        nodes, _ = read_nodes(node_file)
        assert len(nodes) == 4

    def test_read_nodes_invalid_count(self, tmp_path: Path) -> None:
        """Test error when node count doesn't match."""
        node_file = tmp_path / "nodes.dat"
        node_file.write_text(
            """4                         / NNODES
1       0.0       0.0
2     100.0       0.0
"""
        )

        with pytest.raises(FileFormatError, match="expected 4.*got 2"):
            read_nodes(node_file)

    def test_read_nodes_invalid_format(self, tmp_path: Path) -> None:
        """Test error on invalid line format."""
        node_file = tmp_path / "nodes.dat"
        node_file.write_text(
            """2                         / NNODES
1       0.0
2     100.0       0.0
"""
        )

        with pytest.raises(FileFormatError):
            read_nodes(node_file)


class TestReadElements:
    """Tests for reading element files."""

    def test_read_elements_quads(self, tmp_path: Path) -> None:
        """Test reading quadrilateral elements."""
        elem_file = tmp_path / "elements.dat"
        elem_file.write_text(
            """C  Element data file
4                         / NELEM
2                         / NSUBREGION
1  1  2  5  4  1
2  2  3  6  5  1
3  4  5  8  7  2
4  5  6  9  8  2
"""
        )

        elements, subregion_count, _ = read_elements(elem_file)

        assert len(elements) == 4
        assert subregion_count == 2
        assert elements[1].vertices == (1, 2, 5, 4)
        assert elements[1].subregion == 1
        assert elements[3].subregion == 2

    def test_read_elements_triangles(self, tmp_path: Path) -> None:
        """Test reading triangular elements."""
        elem_file = tmp_path / "elements.dat"
        elem_file.write_text(
            """2                         / NELEM
1                         / NSUBREGION
1  1  2  3  0  1
2  2  4  3  0  1
"""
        )

        elements, _, _ = read_elements(elem_file)

        assert len(elements) == 2
        assert elements[1].vertices == (1, 2, 3)  # Should trim trailing 0
        assert elements[1].is_triangle

    def test_read_elements_mixed(self, tmp_path: Path) -> None:
        """Test reading mixed triangles and quads."""
        elem_file = tmp_path / "elements.dat"
        elem_file.write_text(
            """2                         / NELEM
1                         / NSUBREGION
1  1  2  3  4  1
2  4  3  5  0  1
"""
        )

        elements, _, _ = read_elements(elem_file)

        assert elements[1].is_quad
        assert elements[2].is_triangle


class TestReadStratigraphy:
    """Tests for reading stratigraphy files."""

    def test_read_stratigraphy_basic(self, tmp_path: Path) -> None:
        """Test reading basic stratigraphy file.

        IWFM stratigraphy format (from Class_Stratigraphy.f90):
        - NL (number of layers)
        - FACT (conversion factor)
        - ID  GS  Aquitard1_thick  Aquifer1_thick  Aquitard2_thick  Aquifer2_thick  ...

        The elevations are computed from thicknesses:
        - L1_top = GS - Aquitard1
        - L1_bot = L1_top - Aquifer1
        - L2_top = L1_bot - Aquitard2
        - L2_bot = L2_top - Aquifer2
        """
        strat_file = tmp_path / "strat.dat"
        # Format: NL, FACT, then ID  GS  Aqt1  Aqu1  Aqt2  Aqu2 for each node
        # GS=100, Aqt1=0, Aqu1=50 -> L1_top=100, L1_bot=50
        # Aqt2=0, Aqu2=50 -> L2_top=50, L2_bot=0
        strat_file.write_text(
            """C  Stratigraphy data
2                         / NL (number of layers)
1.0                       / FACT
C  Node  GS     Aqt1   Aqu1   Aqt2   Aqu2
1    100.0     0.0    50.0     0.0    50.0
2    100.0     0.0    50.0     0.0    50.0
3    100.0     0.0    50.0     0.0    50.0
4    100.0     0.0    50.0     0.0    50.0
"""
        )

        strat = read_stratigraphy(strat_file)

        assert strat.n_nodes == 4
        assert strat.n_layers == 2
        np.testing.assert_allclose(strat.gs_elev, [100.0, 100.0, 100.0, 100.0])
        # Verify computed elevations
        np.testing.assert_allclose(strat.top_elev[:, 0], [100.0, 100.0, 100.0, 100.0])  # L1 top
        np.testing.assert_allclose(strat.bottom_elev[:, 0], [50.0, 50.0, 50.0, 50.0])  # L1 bottom
        np.testing.assert_allclose(strat.top_elev[:, 1], [50.0, 50.0, 50.0, 50.0])  # L2 top
        np.testing.assert_allclose(strat.bottom_elev[:, 1], [0.0, 0.0, 0.0, 0.0])  # L2 bottom

    def test_read_stratigraphy_non_sequential_node_ids(self, tmp_path: Path) -> None:
        """Node IDs may be non-sequential; rows must follow file order, not id-1."""
        strat_file = tmp_path / "strat.dat"
        # Three nodes with ids 101, 7, 42 and distinct ground-surface elevations.
        strat_file.write_text(
            """C  non-sequential stratigraphy
1                         / NL
1.0                       / FACT
C  ID  GS     Aqt1   Aqu1
101    500.0  5.0    45.0
7      600.0  0.0    60.0
42     700.0  2.0    58.0
"""
        )

        strat = read_stratigraphy(strat_file)

        assert strat.n_nodes == 3
        # Row order follows file order (row 0=id 101, row 1=id 7, row 2=id 42).
        np.testing.assert_allclose(strat.gs_elev, [500.0, 600.0, 700.0])
        np.testing.assert_allclose(strat.top_elev[:, 0], [495.0, 600.0, 698.0])
        np.testing.assert_allclose(strat.bottom_elev[:, 0], [450.0, 540.0, 640.0])

    def test_read_stratigraphy_duplicate_node_ids_raises(self, tmp_path: Path) -> None:
        """Duplicate node IDs in stratigraphy file must surface as FileFormatError."""
        strat_file = tmp_path / "strat.dat"
        strat_file.write_text(
            """1                         / NL
1.0                       / FACT
1    100.0  0.0  50.0
1    200.0  0.0  50.0
"""
        )

        with pytest.raises(FileFormatError, match="Duplicate node ID"):
            read_stratigraphy(strat_file)


class TestWriteNodes:
    """Tests for writing node files."""

    def test_write_nodes_basic(self, tmp_path: Path) -> None:
        """Test writing nodes to file."""
        nodes = {
            1: Node(id=1, x=0.0, y=0.0),
            2: Node(id=2, x=100.0, y=0.0),
            3: Node(id=3, x=100.0, y=100.0),
            4: Node(id=4, x=0.0, y=100.0),
        }

        out_file = tmp_path / "nodes_out.dat"
        write_nodes(out_file, nodes)

        # Read back and verify
        nodes_back, _ = read_nodes(out_file)
        assert len(nodes_back) == 4
        assert nodes_back[1].x == pytest.approx(0.0)
        assert nodes_back[2].x == pytest.approx(100.0)

    def test_write_nodes_roundtrip(self, tmp_path: Path, small_grid_nodes: list[dict]) -> None:
        """Test node write/read roundtrip preserves data."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}

        out_file = tmp_path / "nodes_roundtrip.dat"
        write_nodes(out_file, nodes)
        nodes_back, _ = read_nodes(out_file)

        assert len(nodes_back) == len(nodes)
        for nid in nodes:
            assert nodes_back[nid].x == pytest.approx(nodes[nid].x)
            assert nodes_back[nid].y == pytest.approx(nodes[nid].y)


class TestWriteElements:
    """Tests for writing element files."""

    def test_write_elements_quads(self, tmp_path: Path) -> None:
        """Test writing quadrilateral elements."""
        elements = {
            1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
            2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
        }

        out_file = tmp_path / "elements_out.dat"
        write_elements(out_file, elements, n_subregions=1)

        elements_back, n_sr, _ = read_elements(out_file)
        assert len(elements_back) == 2
        assert elements_back[1].vertices == (1, 2, 5, 4)

    def test_write_elements_triangles(self, tmp_path: Path) -> None:
        """Test writing triangular elements."""
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1),
            2: Element(id=2, vertices=(2, 4, 3), subregion=1),
        }

        out_file = tmp_path / "elements_out.dat"
        write_elements(out_file, elements, n_subregions=1)

        elements_back, _, _ = read_elements(out_file)
        assert elements_back[1].is_triangle
        assert elements_back[1].vertices == (1, 2, 3)


class TestWriteStratigraphy:
    """Tests for writing stratigraphy files."""

    def test_write_stratigraphy_roundtrip(
        self, tmp_path: Path, sample_stratigraphy_data: dict
    ) -> None:
        """Test stratigraphy write/read roundtrip."""
        strat = Stratigraphy(**sample_stratigraphy_data)

        out_file = tmp_path / "strat_out.dat"
        write_stratigraphy(out_file, strat)
        strat_back = read_stratigraphy(out_file)

        assert strat_back.n_nodes == strat.n_nodes
        assert strat_back.n_layers == strat.n_layers
        np.testing.assert_allclose(strat_back.gs_elev, strat.gs_elev)
        np.testing.assert_allclose(strat_back.top_elev, strat.top_elev)
        np.testing.assert_allclose(strat_back.bottom_elev, strat.bottom_elev)


class TestMeshRoundtrip:
    """Integration tests for complete mesh I/O roundtrip."""

    def test_full_mesh_roundtrip(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """Test complete mesh write/read roundtrip."""
        # Create original mesh
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        grid = AppGrid(nodes=nodes, elements=elements)

        # Write to files
        node_file = tmp_path / "nodes.dat"
        elem_file = tmp_path / "elements.dat"

        write_nodes(node_file, grid.nodes)
        write_elements(elem_file, grid.elements, n_subregions=2)

        # Read back
        nodes_back, _ = read_nodes(node_file)
        elements_back, n_sr, _ = read_elements(elem_file)

        # Verify
        assert len(nodes_back) == grid.n_nodes
        assert len(elements_back) == grid.n_elements
        assert n_sr == 2

        # Create new grid from read data
        grid_back = AppGrid(nodes=nodes_back, elements=elements_back)
        grid_back.compute_areas()

        # Areas should match
        assert grid_back.elements[1].area == pytest.approx(10000.0)


# =============================================================================
# Additional coverage tests
# =============================================================================

from pyiwfm.io.ascii import _is_comment_line, _strip_comment  # noqa: E402


class TestIsCommentLine:
    """Tests for _is_comment_line helper."""

    def test_empty_string(self) -> None:
        """Empty string is treated as a comment."""
        assert _is_comment_line("") is True

    def test_whitespace_only(self) -> None:
        """Whitespace-only lines are comments."""
        assert _is_comment_line("   ") is True
        assert _is_comment_line("\t\t") is True
        assert _is_comment_line("  \t  ") is True

    def test_uppercase_c(self) -> None:
        assert _is_comment_line("C  This is a comment") is True

    def test_lowercase_c(self) -> None:
        assert _is_comment_line("c  This is a comment") is True

    def test_asterisk(self) -> None:
        assert _is_comment_line("*  This is a comment") is True

    def test_data_line_not_comment(self) -> None:
        """A line starting with a digit is not a comment."""
        assert _is_comment_line("42 / NNODES") is False

    def test_indented_data_not_comment(self) -> None:
        """A line with leading whitespace followed by data is not a comment."""
        assert _is_comment_line("   100.0  200.0") is False

    def test_newline_only(self) -> None:
        """A line containing only a newline character is a comment."""
        assert _is_comment_line("\n") is True


class TestParseValueLine:
    """Tests for _strip_comment helper."""

    def test_slash_delimiter(self) -> None:
        """Traditional IWFM format with / delimiter."""
        value, desc = _strip_comment("42 / NNODES")
        assert value == "42"
        assert desc == "NNODES"

    def test_hash_delimiter_not_recognized(self) -> None:
        """Hash is not recognized as an inline comment delimiter."""
        value, desc = _strip_comment("42 # NNODES")
        assert value == "42 # NNODES"
        assert desc == ""

    def test_no_delimiter(self) -> None:
        """Bare value with no delimiter."""
        value, desc = _strip_comment("42")
        assert value == "42"
        assert desc == ""

    def test_slash_with_hash_in_description(self) -> None:
        """Only / is recognized as delimiter; # in description is preserved."""
        value, desc = _strip_comment("42 / NNODES # extra")
        assert value == "42"
        assert desc == "NNODES # extra"

    def test_whitespace_stripping(self) -> None:
        """Leading and trailing whitespace is stripped."""
        value, desc = _strip_comment("  42  /  description  ")
        assert value == "42"
        assert desc == "description"

    def test_bare_whitespace_value(self) -> None:
        """A bare value with trailing spaces."""
        value, desc = _strip_comment("  1.0  ")
        assert value == "1.0"
        assert desc == ""


class TestReadNodesAdditional:
    """Additional tests for read_nodes covering edge cases."""

    def test_read_nodes_with_conversion_factor(self, tmp_path: Path) -> None:
        """Test reading nodes with an explicit FACT conversion factor."""
        node_file = tmp_path / "nodes_fact.dat"
        node_file.write_text(
            """C  Test with conversion factor
2                         / NNODES
0.3048                    / FACT (feet to meters)
1       100.0       200.0
2       300.0       400.0
"""
        )
        nodes, fact = read_nodes(node_file)
        assert len(nodes) == 2
        assert nodes[1].x == pytest.approx(100.0 * 0.3048)
        assert nodes[1].y == pytest.approx(200.0 * 0.3048)
        assert nodes[2].x == pytest.approx(300.0 * 0.3048)
        assert fact == pytest.approx(0.3048)

    def test_read_nodes_all_comments_no_nnodes(self, tmp_path: Path) -> None:
        """File with only comments and no NNODES raises error."""
        node_file = tmp_path / "nodes_empty.dat"
        node_file.write_text(
            """C  Comment only file
C  No data here
"""
        )
        with pytest.raises(FileFormatError, match="Could not find NNODES"):
            read_nodes(node_file)

    def test_read_nodes_invalid_nnodes_text(self, tmp_path: Path) -> None:
        """Non-integer NNODES value raises FileFormatError."""
        node_file = tmp_path / "nodes_bad.dat"
        node_file.write_text("abc / NNODES\n")
        with pytest.raises(FileFormatError, match="Invalid NNODES"):
            read_nodes(node_file)

    def test_read_nodes_invalid_node_data_values(self, tmp_path: Path) -> None:
        """Invalid numeric values in node data line raise error."""
        node_file = tmp_path / "nodes_baddata.dat"
        node_file.write_text(
            """1                         / NNODES
1       abc       200.0
"""
        )
        with pytest.raises(FileFormatError, match="Invalid node data"):
            read_nodes(node_file)

    def test_read_nodes_too_few_columns(self, tmp_path: Path) -> None:
        """Node line with fewer than 3 columns raises error."""
        node_file = tmp_path / "nodes_short.dat"
        node_file.write_text(
            """2                         / NNODES
1.0                       / FACT
1       100.0       200.0
2       300.0
"""
        )
        with pytest.raises(FileFormatError, match="Invalid node line format"):
            read_nodes(node_file)

    def test_read_nodes_invalid_remaining_data(self, tmp_path: Path) -> None:
        """Invalid data in remaining node lines raises error."""
        node_file = tmp_path / "nodes_badremainder.dat"
        node_file.write_text(
            """2                         / NNODES
1.0                       / FACT
1       100.0       200.0
2       bad       400.0
"""
        )
        with pytest.raises(FileFormatError, match="Invalid node data"):
            read_nodes(node_file)

    def test_read_nodes_with_slash_inline_comments(self, tmp_path: Path) -> None:
        """Nodes file using / as inline comment delimiter."""
        node_file = tmp_path / "nodes_slash.dat"
        node_file.write_text(
            """C  Standard IWFM format
2                         / NNODES
1       10.0       20.0
2       30.0       40.0
"""
        )
        nodes, _ = read_nodes(node_file)
        assert len(nodes) == 2
        assert nodes[1].x == pytest.approx(10.0)

    def test_read_nodes_first_data_line_is_node_not_fact(self, tmp_path: Path) -> None:
        """When the line after NNODES has 3+ parts, it is node data (not FACT)."""
        node_file = tmp_path / "nodes_nofact.dat"
        node_file.write_text(
            """2                         / NNODES
1       100.0       200.0
2       300.0       400.0
"""
        )
        nodes, fact = read_nodes(node_file)
        assert len(nodes) == 2
        # With no FACT, default is 1.0
        assert nodes[1].x == pytest.approx(100.0)
        assert nodes[2].x == pytest.approx(300.0)
        assert fact == pytest.approx(1.0)

    def test_read_nodes_interleaved_comments(self, tmp_path: Path) -> None:
        """Comments between node data lines are skipped."""
        node_file = tmp_path / "nodes_interleavedcomments.dat"
        node_file.write_text(
            """C  Header
2                         / NNODES
1.0                       / FACT
C  first node follows
1       100.0       200.0
C  second node follows
2       300.0       400.0
"""
        )
        nodes, _ = read_nodes(node_file)
        assert len(nodes) == 2

    def test_read_nodes_fact_value_error_falls_through(self, tmp_path: Path) -> None:
        """When a single-token line doesn't parse as float, treat as node data or skip."""
        node_file = tmp_path / "nodes_nonfloat_fact.dat"
        # If single-part line is not a float, the code falls through to check
        # len(parts) >= 3. Since len is 1 and not >= 3, we break out with no node.
        # Then remaining nodes are read. With NNODES=1 and 1 node read = ok.
        node_file.write_text(
            """1                         / NNODES
1       50.0       60.0
"""
        )
        nodes, _ = read_nodes(node_file)
        assert len(nodes) == 1
        assert nodes[1].x == pytest.approx(50.0)

    def test_read_nodes_invalid_first_node_data(self, tmp_path: Path) -> None:
        """Invalid data in the first node line (before remaining loop) raises error."""
        node_file = tmp_path / "nodes_badfirstnode.dat"
        node_file.write_text(
            """1                         / NNODES
1.0                       / FACT
bad  100.0  200.0
"""
        )
        with pytest.raises(FileFormatError, match="Invalid node data"):
            read_nodes(node_file)


class TestReadElementsAdditional:
    """Additional tests for read_elements covering edge cases."""

    def test_read_elements_no_nelem(self, tmp_path: Path) -> None:
        """File with only comments and no NELEM raises error."""
        elem_file = tmp_path / "elems_empty.dat"
        elem_file.write_text("C  Empty file\n")
        with pytest.raises(FileFormatError, match="Could not find NELEM"):
            read_elements(elem_file)

    def test_read_elements_invalid_nelem(self, tmp_path: Path) -> None:
        """Non-integer NELEM raises error."""
        elem_file = tmp_path / "elems_badnelem.dat"
        elem_file.write_text("abc / NELEM\n1 / NSUBREGION\n")
        with pytest.raises(FileFormatError, match="Invalid NELEM"):
            read_elements(elem_file)

    def test_read_elements_no_nsubregion(self, tmp_path: Path) -> None:
        """Missing NSUBREGION raises error."""
        elem_file = tmp_path / "elems_nosub.dat"
        elem_file.write_text("2 / NELEM\nC  no subregion line\n")
        with pytest.raises(FileFormatError, match="Could not find NSUBREGION"):
            read_elements(elem_file)

    def test_read_elements_invalid_nsubregion(self, tmp_path: Path) -> None:
        """Non-integer NSUBREGION raises error."""
        elem_file = tmp_path / "elems_badsub.dat"
        elem_file.write_text("2 / NELEM\nabc / NSUBREGION\n")
        with pytest.raises(FileFormatError, match="Invalid NSUBREGION"):
            read_elements(elem_file)

    def test_read_elements_with_subregion_names(self, tmp_path: Path) -> None:
        """Elements file with subregion name lines before element data."""
        elem_file = tmp_path / "elems_subrnames.dat"
        elem_file.write_text(
            """2                         / NELEM
2                         / NSUBREGION
Region North
Region South
1  1  2  5  4  1  1
2  2  3  6  5  2  1
"""
        )
        elements, n_sr, _ = read_elements(elem_file)
        assert len(elements) == 2
        assert n_sr == 2

    def test_read_elements_invalid_remaining_data(self, tmp_path: Path) -> None:
        """Invalid data in remaining element lines raises error."""
        elem_file = tmp_path / "elems_baddata.dat"
        elem_file.write_text(
            """2                         / NELEM
1                         / NSUBREGION
1  1  2  5  4  1
2  bad  3  6  5  1
"""
        )
        with pytest.raises(FileFormatError, match="Invalid element data"):
            read_elements(elem_file)

    def test_read_elements_too_few_columns(self, tmp_path: Path) -> None:
        """Element line with fewer than 6 columns raises error."""
        elem_file = tmp_path / "elems_short.dat"
        elem_file.write_text(
            """2                         / NELEM
1                         / NSUBREGION
1  1  2  5  4  1
2  3  6
"""
        )
        with pytest.raises(FileFormatError, match="Invalid element line format"):
            read_elements(elem_file)

    def test_read_elements_subregion_names_short_line(self, tmp_path: Path) -> None:
        """Subregion name with less than 6 parts is recognized as a name."""
        elem_file = tmp_path / "elems_shortname.dat"
        elem_file.write_text(
            """1                         / NELEM
1                         / NSUBREGION
MyRegion
1  1  2  3  0  1
"""
        )
        elements, n_sr, _ = read_elements(elem_file)
        assert len(elements) == 1
        assert n_sr == 1

    def test_read_elements_with_comments_between(self, tmp_path: Path) -> None:
        """Comments interspersed with element data are skipped."""
        elem_file = tmp_path / "elems_comments.dat"
        elem_file.write_text(
            """C  Header
2                         / NELEM
1                         / NSUBREGION
C  Element data follows
1  1  2  5  4  1
C  Next element
2  2  3  6  5  1
"""
        )
        elements, n_sr, _ = read_elements(elem_file)
        assert len(elements) == 2


class TestReadStratigraphyAdditional:
    """Additional tests for read_stratigraphy covering edge cases."""

    def test_read_strat_invalid_nlayers(self, tmp_path: Path) -> None:
        """Non-integer NLAYERS raises error."""
        strat_file = tmp_path / "strat_bad.dat"
        strat_file.write_text("abc / NLAYERS\n1.0 / FACT\n")
        with pytest.raises(FileFormatError, match="Invalid NLAYERS"):
            read_stratigraphy(strat_file)

    def test_read_strat_invalid_fact(self, tmp_path: Path) -> None:
        """Non-numeric FACT raises error."""
        strat_file = tmp_path / "strat_badfact.dat"
        strat_file.write_text("1 / NLAYERS\nabc / FACT\n")
        with pytest.raises(FileFormatError, match="Invalid FACT"):
            read_stratigraphy(strat_file)

    def test_read_strat_no_nlayers(self, tmp_path: Path) -> None:
        """File with no NLAYERS raises error."""
        strat_file = tmp_path / "strat_empty.dat"
        strat_file.write_text("C  Empty file\n")
        with pytest.raises(FileFormatError, match="Could not find NLAYERS"):
            read_stratigraphy(strat_file)

    def test_read_strat_no_data(self, tmp_path: Path) -> None:
        """File with NLAYERS/FACT but no node data raises error."""
        strat_file = tmp_path / "strat_nodata.dat"
        strat_file.write_text("1 / NLAYERS\n1.0 / FACT\n")
        with pytest.raises(FileFormatError, match="No stratigraphy data found"):
            read_stratigraphy(strat_file)

    def test_read_strat_too_few_columns(self, tmp_path: Path) -> None:
        """Stratigraphy line with too few columns raises error."""
        strat_file = tmp_path / "strat_shortline.dat"
        strat_file.write_text(
            """1                         / NLAYERS
1.0                       / FACT
1  100.0  0.0
"""
        )
        # For 1 layer, expected_cols = 2 + 1*2 = 4, but line has 3 parts
        with pytest.raises(FileFormatError, match="Invalid stratigraphy line"):
            read_stratigraphy(strat_file)

    def test_read_strat_invalid_data(self, tmp_path: Path) -> None:
        """Invalid numeric data in stratigraphy raises error."""
        strat_file = tmp_path / "strat_badvals.dat"
        strat_file.write_text(
            """1                         / NLAYERS
1.0                       / FACT
1  100.0  bad  50.0
"""
        )
        with pytest.raises(FileFormatError, match="Invalid stratigraphy data"):
            read_stratigraphy(strat_file)

    def test_read_strat_with_conversion_factor(self, tmp_path: Path) -> None:
        """Stratigraphy with a non-unity FACT applies factor."""
        strat_file = tmp_path / "strat_fact.dat"
        strat_file.write_text(
            """1                         / NLAYERS
2.0                       / FACT
1  50.0  0.0  25.0
"""
        )
        strat = read_stratigraphy(strat_file)
        # All values multiplied by 2.0
        assert strat.gs_elev[0] == pytest.approx(100.0)  # 50*2
        assert strat.top_elev[0, 0] == pytest.approx(100.0)  # top = gs - aqt = 100 - 0 = 100
        assert strat.bottom_elev[0, 0] == pytest.approx(50.0)  # bot = 100 - 50 = 50

    def test_read_strat_inactive_layer(self, tmp_path: Path) -> None:
        """A layer with zero thickness is marked inactive."""
        strat_file = tmp_path / "strat_inactive.dat"
        strat_file.write_text(
            """1                         / NLAYERS
1.0                       / FACT
1  100.0  0.0  0.0
"""
        )
        strat = read_stratigraphy(strat_file)
        # Zero aquifer thickness means top == bottom, so inactive
        assert strat.active_node[0, 0] is np.bool_(False)

    def test_read_strat_with_comments(self, tmp_path: Path) -> None:
        """Comments interspersed in stratigraphy file are skipped."""
        strat_file = tmp_path / "strat_comments.dat"
        strat_file.write_text(
            """C  Stratigraphy
1                         / NLAYERS
C  Factor
1.0                       / FACT
C  Data
1  100.0  10.0  40.0
C  Next node
2  100.0  10.0  40.0
"""
        )
        strat = read_stratigraphy(strat_file)
        assert strat.n_nodes == 2
        np.testing.assert_allclose(strat.top_elev[:, 0], [90.0, 90.0])
        np.testing.assert_allclose(strat.bottom_elev[:, 0], [50.0, 50.0])


class TestWriteNodesAdditional:
    """Additional tests for write_nodes covering edge cases."""

    def test_write_nodes_custom_header(self, tmp_path: Path) -> None:
        """Test writing nodes with a custom header."""
        nodes = {1: Node(id=1, x=0.0, y=0.0)}
        out_file = tmp_path / "nodes_header.dat"
        write_nodes(out_file, nodes, header="Custom header\nSecond line")

        content = out_file.read_text()
        assert "C  Custom header" in content
        assert "C  Second line" in content

    def test_write_nodes_default_header(self, tmp_path: Path) -> None:
        """Test writing nodes with default header."""
        nodes = {1: Node(id=1, x=5.0, y=10.0)}
        out_file = tmp_path / "nodes_default.dat"
        write_nodes(out_file, nodes)

        content = out_file.read_text()
        assert "pyiwfm" in content


class TestWriteElementsAdditional:
    """Additional tests for write_elements covering edge cases."""

    def test_write_elements_custom_header(self, tmp_path: Path) -> None:
        """Test writing elements with a custom header."""
        elements = {1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1)}
        out_file = tmp_path / "elems_header.dat"
        write_elements(out_file, elements, n_subregions=1, header="My header")

        content = out_file.read_text()
        assert "C  My header" in content

    def test_write_elements_default_header(self, tmp_path: Path) -> None:
        """Test writing elements with default header."""
        elements = {1: Element(id=1, vertices=(1, 2, 3), subregion=1)}
        out_file = tmp_path / "elems_default.dat"
        write_elements(out_file, elements, n_subregions=1)

        content = out_file.read_text()
        assert "pyiwfm" in content

    def test_write_elements_mixed_tri_quad(self, tmp_path: Path) -> None:
        """Test writing mixed triangle and quad elements."""
        elements = {
            1: Element(id=1, vertices=(1, 2, 3), subregion=1),
            2: Element(id=2, vertices=(3, 4, 5, 6), subregion=1),
        }
        out_file = tmp_path / "elems_mixed.dat"
        write_elements(out_file, elements, n_subregions=1)

        elements_back, _, _ = read_elements(out_file)
        assert elements_back[1].is_triangle
        assert elements_back[2].is_quad


class TestWriteStratigraphyAdditional:
    """Additional tests for write_stratigraphy covering edge cases."""

    def test_write_stratigraphy_custom_header(self, tmp_path: Path) -> None:
        """Test writing stratigraphy with a custom header."""
        strat = Stratigraphy(
            n_layers=1,
            n_nodes=1,
            gs_elev=np.array([100.0]),
            top_elev=np.array([[100.0]]),
            bottom_elev=np.array([[50.0]]),
            active_node=np.array([[True]]),
        )
        out_file = tmp_path / "strat_header.dat"
        write_stratigraphy(out_file, strat, header="Custom strat header")

        content = out_file.read_text()
        assert "C  Custom strat header" in content

    def test_write_stratigraphy_default_header(self, tmp_path: Path) -> None:
        """Test writing stratigraphy with default header."""
        strat = Stratigraphy(
            n_layers=1,
            n_nodes=1,
            gs_elev=np.array([100.0]),
            top_elev=np.array([[100.0]]),
            bottom_elev=np.array([[50.0]]),
            active_node=np.array([[True]]),
        )
        out_file = tmp_path / "strat_default.dat"
        write_stratigraphy(out_file, strat)

        content = out_file.read_text()
        assert "pyiwfm" in content
        assert "AQT1" in content
        assert "AQF1" in content

    def test_write_stratigraphy_multilayer(self, tmp_path: Path) -> None:
        """Test writing and reading back multi-layer stratigraphy."""
        n_nodes = 2
        n_layers = 3
        gs = np.array([150.0, 150.0])
        top = np.zeros((n_nodes, n_layers))
        bot = np.zeros((n_nodes, n_layers))
        # Layer 1: top=150, bot=100; Layer 2: top=100, bot=50; Layer 3: top=50, bot=0
        for i in range(n_layers):
            top[:, i] = 150.0 - i * 50.0
            bot[:, i] = 100.0 - i * 50.0
        active = np.ones((n_nodes, n_layers), dtype=bool)

        strat = Stratigraphy(
            n_layers=n_layers,
            n_nodes=n_nodes,
            gs_elev=gs,
            top_elev=top,
            bottom_elev=bot,
            active_node=active,
        )
        out_file = tmp_path / "strat_multi.dat"
        write_stratigraphy(out_file, strat)
        strat_back = read_stratigraphy(out_file)

        assert strat_back.n_layers == 3
        assert strat_back.n_nodes == 2
        np.testing.assert_allclose(strat_back.gs_elev, gs)
        np.testing.assert_allclose(strat_back.top_elev, top)
        np.testing.assert_allclose(strat_back.bottom_elev, bot)
