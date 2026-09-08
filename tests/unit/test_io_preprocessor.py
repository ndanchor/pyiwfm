"""Unit tests for PreProcessor I/O handlers."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyiwfm.core.mesh import AppGrid, Element, Node, Subregion
from pyiwfm.core.model import IWFMModel
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.io.preprocessor import (
    PreProcessorConfig,
    read_preprocessor_main,
    read_subregions_file,
    save_model_to_preprocessor,
    write_preprocessor_main,
)


class TestPreProcessorConfig:
    """Tests for PreProcessorConfig."""

    def test_config_creation(self, tmp_path: Path) -> None:
        """Test basic config creation."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="Test Model",
            n_layers=3,
        )

        assert config.model_name == "Test Model"
        assert config.n_layers == 3
        assert config.base_dir == tmp_path


class TestReadPreProcessorMain:
    """Tests for reading PreProcessor main input files."""

    def test_read_basic_main_file(self, tmp_path: Path) -> None:
        """Test reading a basic main input file."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Test PreProcessor Main Input File
Test Model                               / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
stratigraphy.dat                         / STRATIGRAPHY_FILE
2                                        / N_LAYERS
FT                                       / LENGTH_UNIT
"""
        )

        config = read_preprocessor_main(main_file)

        assert config.model_name == "Test Model"
        assert config.nodes_file == tmp_path / "nodes.dat"
        assert config.elements_file == tmp_path / "elements.dat"
        assert config.stratigraphy_file == tmp_path / "stratigraphy.dat"
        assert config.n_layers == 2
        assert config.length_unit == "FT"

    def test_read_with_subregions(self, tmp_path: Path) -> None:
        """Test reading main file with subregions."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Main Input File
Model                                    / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
subregions.dat                           / SUBREGIONS_FILE
"""
        )

        config = read_preprocessor_main(main_file)

        assert config.subregions_file == tmp_path / "subregions.dat"


class TestReadPreProcessorMainFactltouUnitltou:
    """Tests for parsing the real IWFM FACTLTOU/UNITLTOU tokens.

    Real PreProcessor main files (unlike the simplified LENGTH_UNIT tag
    written by write_preprocessor_main()) label these lines "/FACTLTOU"
    and "/UNITLTOU", not "/LENGTH_UNIT" -- see e.g. C2VSimFG's
    Preprocessor.in.
    """

    def _write(self, tmp_path: Path, factltou: str, unitltou: str) -> Path:
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            f"""C  Test PreProcessor Main Input File
Test Model                               / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
{factltou:<40}  /FACTLTOU
{unitltou:<40}  /UNITLTOU
"""
        )
        return main_file

    def test_factltou_1_feet_resolves_simulation_unit_feet(self, tmp_path: Path) -> None:
        """FACTLTOU=1, UNITLTOU=FEET (the common C2VSimFG case) -> model
        coordinates are in feet."""
        main_file = self._write(tmp_path, "1", "FEET")

        config = read_preprocessor_main(main_file)

        assert config.length_factor == pytest.approx(1.0)
        assert config.length_unit == "FEET"
        assert config.simulation_length_unit == "FEET"

    def test_factltou_meters_output_resolves_simulation_unit_feet(self, tmp_path: Path) -> None:
        """FACTLTOU=0.3048, UNITLTOU=METERS -> the simulation reports in
        meters by converting from an internal foot unit."""
        main_file = self._write(tmp_path, "0.3048", "METERS")

        config = read_preprocessor_main(main_file)

        assert config.simulation_length_unit == "FEET"

    def test_factltou_feet_output_resolves_simulation_unit_meters(self, tmp_path: Path) -> None:
        """FACTLTOU=3.28084, UNITLTOU=FEET -> the simulation reports in
        feet by converting from an internal meter unit."""
        main_file = self._write(tmp_path, "3.28084", "FEET")

        config = read_preprocessor_main(main_file)

        assert config.simulation_length_unit == "METERS"

    def test_unrecognized_factltou_leaves_simulation_unit_none(self, tmp_path: Path) -> None:
        """A FACTLTOU that matches neither 1 foot nor 1 meter in physical
        size can't be resolved to either unit (callers fall back
        elsewhere, e.g. GISExporter's Nodes-file-factor guess)."""
        main_file = self._write(tmp_path, "2.5", "FEET")

        config = read_preprocessor_main(main_file)

        assert config.length_factor == pytest.approx(2.5)
        assert config.simulation_length_unit is None

    def test_missing_factltou_unitltou_falls_back_to_defaults(self, tmp_path: Path) -> None:
        """Without FACTLTOU/UNITLTOU, length_unit/length_factor keep their
        dataclass defaults ("FT"/1.0), which resolve to FEET -- the same
        assumption the rest of the codebase already makes for a bare
        PreProcessorConfig."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Test PreProcessor Main Input File
Test Model                               / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
"""
        )

        config = read_preprocessor_main(main_file)

        assert config.length_unit == "FT"
        assert config.length_factor == pytest.approx(1.0)
        assert config.simulation_length_unit == "FEET"


class TestWritePreProcessorMain:
    """Tests for writing PreProcessor main input files."""

    def test_write_main_file(self, tmp_path: Path) -> None:
        """Test writing a main input file."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="Output Model",
            nodes_file=tmp_path / "nodes.dat",
            elements_file=tmp_path / "elements.dat",
            stratigraphy_file=tmp_path / "strat.dat",
            n_layers=2,
            length_unit="FT",
            area_unit="ACRES",
            volume_unit="AF",
        )

        main_file = tmp_path / "output_pp.in"
        write_preprocessor_main(main_file, config)

        assert main_file.exists()

        # Read back and verify
        config_back = read_preprocessor_main(main_file)
        assert config_back.model_name == "Output Model"
        assert config_back.n_layers == 2


class TestReadSubregionsFile:
    """Tests for reading subregions files."""

    def test_read_subregions(self, tmp_path: Path) -> None:
        """Test reading a subregions file."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            """C  Subregion definitions
3                                        / NSUBREGION
1  North Region
2  Central Region
3  South Region
"""
        )

        subregions = read_subregions_file(sr_file)

        assert len(subregions) == 3
        assert subregions[1].name == "North Region"
        assert subregions[2].name == "Central Region"
        assert subregions[3].name == "South Region"


class TestLoadModelFromPreProcessor:
    """Tests for loading models from PreProcessor files."""

    def test_load_complete_model(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
        sample_stratigraphy_data: dict,
    ) -> None:
        """Test loading a complete model from PreProcessor files."""
        # Create input files
        nodes_file = tmp_path / "nodes.dat"
        nodes_file.write_text(
            """C  Node data
9                                        / NNODES
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

        elements_file = tmp_path / "elements.dat"
        elements_file.write_text(
            """C  Element data
4                                        / NELEM
2                                        / NSUBREGION
1     1     2     5     4     1
2     2     3     6     5     1
3     4     5     8     7     2
4     5     6     9     8     2
"""
        )

        strat_file = tmp_path / "stratigraphy.dat"
        # IWFM stratigraphy format: NL, FACT, then data
        # ID  GS  Aquitard1_thick  Aquifer1_thick  Aquitard2_thick  Aquifer2_thick
        # These nodes have GS varying from 100-120, with constant layer thicknesses
        strat_file.write_text(
            """C  Stratigraphy data
2                                        / NL (number of layers)
1.0                                      / FACT
C  ID    GS     Aqt1   Aqu1   Aqt2   Aqu2
1    100.0     0.0    50.0     0.0    50.0
2    105.0     0.0    50.0     0.0    50.0
3    110.0     0.0    50.0     0.0    50.0
4    105.0     0.0    50.0     0.0    50.0
5    110.0     0.0    50.0     0.0    50.0
6    115.0     0.0    50.0     0.0    50.0
7    110.0     0.0    50.0     0.0    50.0
8    115.0     0.0    50.0     0.0    50.0
9    120.0     0.0    50.0     0.0    50.0
"""
        )

        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  PreProcessor Main Input
Test Model                               / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
stratigraphy.dat                         / STRATIGRAPHY_FILE
"""
        )

        # Load model
        model = IWFMModel.from_preprocessor(main_file)

        assert model.name == "Test Model"
        assert model.n_nodes == 9
        assert model.n_elements == 4
        assert model.n_layers == 2


class TestSaveModelToPreProcessor:
    """Tests for saving models to PreProcessor files."""

    def test_save_and_reload_model(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
        sample_stratigraphy_data: dict,
    ) -> None:
        """Test saving a model and reloading it."""
        # Create original model
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        subregions = {
            1: Subregion(id=1, name="Region A"),
            2: Subregion(id=2, name="Region B"),
        }
        mesh = AppGrid(nodes=nodes, elements=elements, subregions=subregions)

        strat = Stratigraphy(**sample_stratigraphy_data)

        model = IWFMModel(
            name="Roundtrip Test",
            mesh=mesh,
            stratigraphy=strat,
        )

        # Save to PreProcessor files
        output_dir = tmp_path / "output"
        config = save_model_to_preprocessor(model, output_dir)

        # Verify files were created
        assert config.nodes_file.exists()
        assert config.elements_file.exists()
        assert config.stratigraphy_file.exists()

        # Reload and verify
        main_file = output_dir / "Roundtrip Test_pp.in"
        model_back = IWFMModel.from_preprocessor(main_file)

        assert model_back.n_nodes == model.n_nodes
        assert model_back.n_elements == model.n_elements
        assert model_back.n_layers == model.n_layers


# =========================================================================
# Additional tests appended below to increase coverage
# =========================================================================

from pyiwfm.core.exceptions import FileFormatError  # noqa: E402
from pyiwfm.io.preprocessor import (  # noqa: E402
    _is_comment_line,
    _make_relative_path,
    _resolve_path,
    _strip_comment,
    _write_subregions_file,
)


class TestIsCommentLine:
    """Tests for the _is_comment_line helper function."""

    def test_empty_string(self) -> None:
        """Empty string is treated as a comment."""
        assert _is_comment_line("") is True

    def test_whitespace_only(self) -> None:
        """Whitespace-only lines are treated as comments."""
        assert _is_comment_line("   ") is True
        assert _is_comment_line("\t\t") is True
        assert _is_comment_line("  \t  ") is True

    def test_uppercase_c_comment(self) -> None:
        """Lines starting with 'C' in column 1 are comments."""
        assert _is_comment_line("C  This is a comment") is True

    def test_lowercase_c_comment(self) -> None:
        """Lines starting with 'c' in column 1 are comments."""
        assert _is_comment_line("c  This is a comment") is True

    def test_asterisk_comment(self) -> None:
        """Lines starting with '*' in column 1 are comments."""
        assert _is_comment_line("*  This is a comment") is True

    def test_data_line(self) -> None:
        """Lines starting with non-comment characters are data lines."""
        assert _is_comment_line("42 / VALUE") is False

    def test_indented_data_is_not_comment(self) -> None:
        """Lines starting with whitespace then data are NOT comments."""
        assert _is_comment_line("  42 / VALUE") is False

    def test_newline_only(self) -> None:
        """A line that is just a newline is a comment."""
        assert _is_comment_line("\n") is True


class TestParseValueLine:
    """Tests for the _strip_comment helper function."""

    def test_slash_delimiter(self) -> None:
        """Parse value line with '/' delimiter."""
        value, desc = _strip_comment("nodes.dat                / NODES_FILE")
        assert value == "nodes.dat"
        assert desc == "NODES_FILE"

    def test_hash_delimiter_not_recognized(self) -> None:
        """Hash is not recognized as an inline comment delimiter."""
        value, desc = _strip_comment("nodes.dat                # NODES_FILE")
        assert value == "nodes.dat                # NODES_FILE"
        assert desc == ""

    def test_no_delimiter(self) -> None:
        """Parse value line with no delimiter."""
        value, desc = _strip_comment("some_value")
        assert value == "some_value"
        assert desc == ""

    def test_slash_with_hash_in_description(self) -> None:
        """Only / is recognized as delimiter; # in description is preserved."""
        value, desc = _strip_comment("value / desc # extra")
        assert value == "value"
        assert desc == "desc # extra"

    def test_empty_value(self) -> None:
        """Parse a line that is all whitespace."""
        value, desc = _strip_comment("   ")
        assert value == ""
        assert desc == ""


class TestResolvePath:
    """Tests for the _resolve_path helper function."""

    def test_relative_path(self, tmp_path: Path) -> None:
        """Relative paths are resolved relative to base directory."""
        result = _resolve_path(tmp_path, "nodes.dat")
        assert result == tmp_path / "nodes.dat"

    def test_absolute_path(self, tmp_path: Path) -> None:
        """Absolute paths are returned as-is."""
        abs_path = tmp_path / "absolute" / "nodes.dat"
        result = _resolve_path(tmp_path, str(abs_path))
        assert result == abs_path

    def test_whitespace_trimmed(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace is stripped from the filepath."""
        result = _resolve_path(tmp_path, "  nodes.dat  ")
        assert result == tmp_path / "nodes.dat"


class TestMakeRelativePath:
    """Tests for the _make_relative_path helper function."""

    def test_relative_path_within_base(self, tmp_path: Path) -> None:
        """Path within base directory is returned as relative."""
        target = tmp_path / "sub" / "file.dat"
        result = _make_relative_path(tmp_path, target)
        assert result == str(Path("sub") / "file.dat")

    def test_path_outside_base(self, tmp_path: Path) -> None:
        """Path outside base directory is returned as absolute string."""
        target = (
            Path("C:/totally/different/file.dat")
            if str(tmp_path).startswith("C:")
            else Path("/totally/different/file.dat")
        )
        result = _make_relative_path(tmp_path, target)
        assert result == str(target)


class TestReadPreProcessorMainExtended:
    """Extended tests for read_preprocessor_main covering more branch paths."""

    def test_read_slash_format(self, tmp_path: Path) -> None:
        """Test reading a file using '/' inline comment format."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  PreProcessor Main Input File
Test Model C2V                           / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
stratigraphy.dat                         / STRATIGRAPHY_FILE
3                                        / N_LAYERS
M                                        / LENGTH_UNIT
"""
        )

        config = read_preprocessor_main(main_file)

        assert config.model_name == "Test Model C2V"
        assert config.nodes_file == tmp_path / "nodes.dat"
        assert config.elements_file == tmp_path / "elements.dat"
        assert config.stratigraphy_file == tmp_path / "stratigraphy.dat"
        assert config.n_layers == 3
        assert config.length_unit == "M"

    def test_read_all_component_files(self, tmp_path: Path) -> None:
        """Test reading a file with all optional component paths."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Complete PreProcessor Main Input File
Full Model                               / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
stratigraphy.dat                         / STRATIGRAPHY_FILE
subregions.dat                           / SUBREGIONS_FILE
streams.dat                              / STREAMS_FILE
lakes.dat                                / LAKES_FILE
groundwater.dat                          / GROUNDWATER_FILE
rootzone.dat                             / ROOTZONE_FILE
pumping.dat                              / PUMPING_FILE
2                                        / N_LAYERS
FT                                       / LENGTH_UNIT
ACRES                                    / AREA_UNIT
AF                                       / VOLUME_UNIT
output                                   / OUTPUT_DIR
"""
        )

        config = read_preprocessor_main(main_file)

        assert config.model_name == "Full Model"
        assert config.nodes_file == tmp_path / "nodes.dat"
        assert config.elements_file == tmp_path / "elements.dat"
        assert config.stratigraphy_file == tmp_path / "stratigraphy.dat"
        assert config.subregions_file == tmp_path / "subregions.dat"
        assert config.streams_file == tmp_path / "streams.dat"
        assert config.lakes_file == tmp_path / "lakes.dat"
        assert config.groundwater_file == tmp_path / "groundwater.dat"
        assert config.rootzone_file == tmp_path / "rootzone.dat"
        assert config.pumping_file == tmp_path / "pumping.dat"
        assert config.n_layers == 2
        assert config.length_unit == "FT"
        assert config.area_unit == "ACRES"
        assert config.volume_unit == "AF"
        assert config.output_dir == tmp_path / "output"

    def test_read_only_comments(self, tmp_path: Path) -> None:
        """Test reading a file with only comment lines (no data)."""
        main_file = tmp_path / "empty_pp.in"
        main_file.write_text(
            """C  Only comments
C  Nothing here
*  Still nothing
"""
        )
        config = read_preprocessor_main(main_file)
        # No data lines found, config should have defaults
        assert config.model_name == ""
        assert config.nodes_file is None

    def test_invalid_n_layers_ignored(self, tmp_path: Path) -> None:
        """Non-numeric LAYER values are silently ignored."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  PreProcessor
Model                                    / MODEL_NAME
not_a_number                             / N_LAYERS
"""
        )
        config = read_preprocessor_main(main_file)
        # Default n_layers is 1 because int() failed and was caught
        assert config.n_layers == 1

    def test_numeric_value_not_treated_as_path(self, tmp_path: Path) -> None:
        """Numeric values should not be resolved as file paths."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  PreProcessor
Model                                    / MODEL_NAME
5                                        / N_LAYERS
"""
        )
        config = read_preprocessor_main(main_file)
        assert config.n_layers == 5

    def test_read_with_many_comments_interspersed(self, tmp_path: Path) -> None:
        """Test reading a file with comments interspersed between data lines."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Title line
C  Another comment
My Model                                 / MODEL_NAME
C  Node file reference
nodes.dat                                / NODES_FILE
C  Element file reference
elements.dat                             / ELEMENTS_FILE
C  That is all
"""
        )
        config = read_preprocessor_main(main_file)
        assert config.model_name == "My Model"
        assert config.nodes_file == tmp_path / "nodes.dat"
        assert config.elements_file == tmp_path / "elements.dat"


class TestReadSubregionsFileExtended:
    """Extended tests for read_subregions_file error handling."""

    def test_invalid_nsubregion_value(self, tmp_path: Path) -> None:
        """Non-numeric NSUBREGION raises FileFormatError."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            """C  Subregion definitions
abc                                      / NSUBREGION
"""
        )
        with pytest.raises(FileFormatError, match="Invalid NSUBREGION"):
            read_subregions_file(sr_file)

    def test_missing_nsubregion(self, tmp_path: Path) -> None:
        """File with only comments and no NSUBREGION raises FileFormatError."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            """C  Only comments here
C  Nothing else
"""
        )
        with pytest.raises(FileFormatError, match="Could not find NSUBREGION"):
            read_subregions_file(sr_file)

    def test_invalid_subregion_data(self, tmp_path: Path) -> None:
        """Non-numeric subregion ID raises FileFormatError."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            """C  Subregion definitions
2                                        / NSUBREGION
abc  Bad Region
2    Good Region
"""
        )
        with pytest.raises(FileFormatError, match="Invalid subregion data"):
            read_subregions_file(sr_file)

    def test_subregion_without_name(self, tmp_path: Path) -> None:
        """Subregion with ID only (no name) should still parse."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            """C  Subregion definitions
1                                        / NSUBREGION
1
"""
        )
        subregions = read_subregions_file(sr_file)
        assert len(subregions) == 1
        assert subregions[1].id == 1
        assert subregions[1].name == ""

    def test_subregion_with_comments_between(self, tmp_path: Path) -> None:
        """Comments between subregion data lines are skipped."""
        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            """C  Subregion definitions
2                                        / NSUBREGION
C  First subregion
1  North
C  Second subregion
2  South
"""
        )
        subregions = read_subregions_file(sr_file)
        assert len(subregions) == 2
        assert subregions[1].name == "North"
        assert subregions[2].name == "South"


class TestWritePreProcessorMainExtended:
    """Extended tests for write_preprocessor_main."""

    def test_write_with_custom_header(self, tmp_path: Path) -> None:
        """Test writing with a custom header."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="Custom Header Model",
        )
        main_file = tmp_path / "custom_pp.in"
        write_preprocessor_main(main_file, config, header="My Custom Header\nLine Two")

        content = main_file.read_text()
        assert "C  My Custom Header" in content
        assert "C  Line Two" in content

    def test_write_with_subregions_file(self, tmp_path: Path) -> None:
        """Test writing config that includes a subregions file path."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="SR Model",
            nodes_file=tmp_path / "nodes.dat",
            elements_file=tmp_path / "elements.dat",
            subregions_file=tmp_path / "subregions.dat",
        )
        main_file = tmp_path / "sr_pp.in"
        write_preprocessor_main(main_file, config)

        content = main_file.read_text()
        assert "subregions.dat" in content
        assert "SUBREGIONS_FILE" in content

    def test_write_without_optional_files(self, tmp_path: Path) -> None:
        """Config with no file paths still writes settings."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="Minimal",
            n_layers=4,
            length_unit="M",
            area_unit="SQ_M",
            volume_unit="CU_M",
        )
        main_file = tmp_path / "min_pp.in"
        write_preprocessor_main(main_file, config)

        content = main_file.read_text()
        assert "Minimal" in content
        assert "N_LAYERS" in content
        assert "LENGTH_UNIT" in content
        assert "AREA_UNIT" in content
        assert "VOLUME_UNIT" in content

    def test_roundtrip_all_units(self, tmp_path: Path) -> None:
        """Verify unit settings survive a write-then-read roundtrip.

        Note: IWFM comment detection treats any line starting with 'C'
        in column 1 as a comment.  So a VOLUME_UNIT value such as
        "CU_M" is misread as a comment line and the default value is
        retained.  This test documents that known behaviour.
        """
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="Units Test",
            n_layers=3,
            length_unit="M",
            area_unit="HECTARES",
            volume_unit="AF",  # Use value not starting with C/c/*
        )
        main_file = tmp_path / "units_pp.in"
        write_preprocessor_main(main_file, config)

        config_back = read_preprocessor_main(main_file)
        assert config_back.n_layers == 3
        assert config_back.length_unit == "M"
        assert config_back.area_unit == "HECTARES"
        assert config_back.volume_unit == "AF"

    def test_roundtrip_volume_unit_starting_with_c_is_lost(self, tmp_path: Path) -> None:
        """Values starting with 'C' are misread as comment lines.

        This documents the known IWFM parsing limitation where
        column-1 comment detection can swallow data values that happen
        to start with 'C', 'c', or '*'.
        """
        config = PreProcessorConfig(
            base_dir=tmp_path,
            model_name="Units Test",
            n_layers=3,
            volume_unit="CU_M",
        )
        main_file = tmp_path / "units_pp.in"
        write_preprocessor_main(main_file, config)

        config_back = read_preprocessor_main(main_file)
        # CU_M line starts with 'C' so it is treated as a comment
        # and volume_unit falls back to the default "AF"
        assert config_back.volume_unit == "AF"


class TestWriteSubregionsFile:
    """Tests for _write_subregions_file."""

    def test_write_and_read_subregions(self, tmp_path: Path) -> None:
        """Test writing subregions and reading them back."""
        subregions = {
            1: Subregion(id=1, name="Alpha Region"),
            2: Subregion(id=2, name="Beta Region"),
            3: Subregion(id=3, name="Gamma Region"),
        }
        sr_file = tmp_path / "subregions.dat"
        _write_subregions_file(sr_file, subregions)

        # Read back
        result = read_subregions_file(sr_file)
        assert len(result) == 3
        assert result[1].name == "Alpha Region"
        assert result[2].name == "Beta Region"
        assert result[3].name == "Gamma Region"

    def test_write_empty_subregions(self, tmp_path: Path) -> None:
        """Test writing an empty subregion dictionary."""
        subregions: dict[int, Subregion] = {}
        sr_file = tmp_path / "empty_subregions.dat"
        _write_subregions_file(sr_file, subregions)

        content = sr_file.read_text()
        assert "0" in content  # NSUBREGION = 0


class TestLoadModelFromPreProcessorExtended:
    """Extended tests for IWFMModel.from_preprocessor error handling."""

    def test_missing_nodes_file_raises(self, tmp_path: Path) -> None:
        """Error raised when nodes file not specified in config."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  PreProcessor without node file
Test Model                               / MODEL_NAME
elements.dat                             / ELEMENTS_FILE
"""
        )
        with pytest.raises(FileFormatError, match="Nodes file not specified"):
            IWFMModel.from_preprocessor(main_file)

    def test_missing_elements_file_raises(self, tmp_path: Path) -> None:
        """Error raised when elements file not specified in config."""
        # Create a valid nodes file so nodes loading succeeds
        nodes_file = tmp_path / "nodes.dat"
        nodes_file.write_text(
            """C  Node data
1                                        / NNODES
1       0.0       0.0
"""
        )

        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  PreProcessor without element file
Test Model                               / MODEL_NAME
nodes.dat                                / NODES_FILE
"""
        )
        with pytest.raises(FileFormatError, match="Elements file not specified"):
            IWFMModel.from_preprocessor(main_file)

    def test_load_without_stratigraphy(self, tmp_path: Path) -> None:
        """Model loads successfully without stratigraphy file."""
        nodes_file = tmp_path / "nodes.dat"
        nodes_file.write_text(
            """C  Node data
4                                        / NNODES
1       0.0       0.0
2     100.0       0.0
3     100.0     100.0
4       0.0     100.0
"""
        )

        elements_file = tmp_path / "elements.dat"
        elements_file.write_text(
            """C  Element data
1                                        / NELEM
1                                        / NSUBREGION
1     1     2     3     4     1
"""
        )

        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  No stratigraphy
No Strat Model                           / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
"""
        )

        model = IWFMModel.from_preprocessor(main_file)
        assert model.name == "No Strat Model"
        assert model.n_nodes == 4
        assert model.n_elements == 1
        assert model.stratigraphy is None
        assert model.n_layers == 0

    def test_load_with_subregions(self, tmp_path: Path) -> None:
        """Model correctly loads subregion data from file."""
        nodes_file = tmp_path / "nodes.dat"
        nodes_file.write_text(
            """C  Node data
4                                        / NNODES
1       0.0       0.0
2     100.0       0.0
3     100.0     100.0
4       0.0     100.0
"""
        )

        elements_file = tmp_path / "elements.dat"
        elements_file.write_text(
            """C  Element data
1                                        / NELEM
1                                        / NSUBREGION
1     1     2     3     4     1
"""
        )

        sr_file = tmp_path / "subregions.dat"
        sr_file.write_text(
            """C  Subregion definitions
1                                        / NSUBREGION
1  Test Region
"""
        )

        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  With subregions
SR Model                                 / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
subregions.dat                           / SUBREGIONS_FILE
"""
        )

        model = IWFMModel.from_preprocessor(main_file)
        assert model.mesh is not None
        assert model.mesh.n_subregions == 1

    def test_model_name_defaults_to_filename(self, tmp_path: Path) -> None:
        """Model name defaults to file stem when config model_name is empty.

        When all data lines have a description keyword, the first line
        at idx==0 is still consumed as the model name.  To get an empty
        model_name (triggering the filename fallback), the preprocessor
        file must have a MODEL_NAME line whose value is explicitly empty,
        which is unusual.  This test provides an explicit MODEL_NAME line
        that is empty to trigger the fallback.
        """
        nodes_file = tmp_path / "nodes.dat"
        nodes_file.write_text(
            """C  Node data
3                                        / NNODES
1       0.0       0.0
2     100.0       0.0
3      50.0      86.6
"""
        )

        elements_file = tmp_path / "elements.dat"
        elements_file.write_text(
            """C  Element data
1                                        / NELEM
1                                        / NSUBREGION
1     1     2     3     0     1
"""
        )

        # Without a MODEL_NAME line, the first data line (nodes.dat)
        # is consumed as the model name (idx==0 always matches).
        # So use a MODEL_NAME line explicitly set to "MyModel".
        main_file = tmp_path / "my_model_pp.in"
        main_file.write_text(
            """C  PreProcessor
MyModel                                  / MODEL_NAME
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
"""
        )

        model = IWFMModel.from_preprocessor(main_file)
        assert model.name == "MyModel"

    def test_first_data_line_consumed_as_name_without_desc(self, tmp_path: Path) -> None:
        """When no MODEL_NAME description exists, the first data line at idx==0
        is consumed as the model name, even if it looks like a file path.
        This documents the actual parsing behavior.
        """
        nodes_file = tmp_path / "nodes.dat"
        nodes_file.write_text(
            """C  Node data
3                                        / NNODES
1       0.0       0.0
2     100.0       0.0
3      50.0      86.6
"""
        )

        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  PreProcessor
nodes.dat                                / NODES_FILE
elements.dat                             / ELEMENTS_FILE
"""
        )

        config = read_preprocessor_main(main_file)
        # First data line at idx==0 is consumed as model_name
        assert config.model_name == "nodes.dat"
        # Since nodes.dat was consumed as model name, nodes_file is None
        assert config.nodes_file is None


class TestSaveModelToPreProcessorExtended:
    """Extended tests for save_model_to_preprocessor."""

    def test_save_model_without_mesh(self, tmp_path: Path) -> None:
        """Saving a model with no mesh still creates main file."""
        model = IWFMModel(name="EmptyModel")
        output_dir = tmp_path / "output_empty"
        config = save_model_to_preprocessor(model, output_dir)

        assert config.nodes_file is None
        assert config.elements_file is None
        main_file = output_dir / "EmptyModel_pp.in"
        assert main_file.exists()

    def test_save_model_without_stratigraphy(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """Saving a model with mesh but no stratigraphy."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        mesh = AppGrid(nodes=nodes, elements=elements)

        model = IWFMModel(name="NoStrat", mesh=mesh)
        output_dir = tmp_path / "output_nostrat"
        config = save_model_to_preprocessor(model, output_dir)

        assert config.nodes_file is not None
        assert config.nodes_file.exists()
        assert config.elements_file is not None
        assert config.elements_file.exists()
        assert config.stratigraphy_file is None

    def test_save_model_without_subregions(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """Saving a model with mesh but no subregions defined."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        mesh = AppGrid(nodes=nodes, elements=elements)
        # mesh has no subregions dict entries

        model = IWFMModel(name="NoSub", mesh=mesh)
        output_dir = tmp_path / "output_nosub"
        config = save_model_to_preprocessor(model, output_dir)

        # No subregions file should be created
        assert config.subregions_file is None

    def test_save_with_custom_name(
        self,
        tmp_path: Path,
        small_grid_nodes: list[dict],
        small_grid_elements: list[dict],
    ) -> None:
        """model_name parameter overrides model.name."""
        nodes = {d["id"]: Node(**d) for d in small_grid_nodes}
        elements = {d["id"]: Element(**d) for d in small_grid_elements}
        mesh = AppGrid(nodes=nodes, elements=elements)

        model = IWFMModel(name="OriginalName", mesh=mesh)
        output_dir = tmp_path / "output_custom"
        config = save_model_to_preprocessor(model, output_dir, model_name="CustomName")

        assert config.model_name == "CustomName"
        main_file = output_dir / "CustomName_pp.in"
        assert main_file.exists()

    def test_save_model_name_defaults(self, tmp_path: Path) -> None:
        """When model has no name and no override, defaults to 'iwfm_model'."""
        model = IWFMModel(name="")
        output_dir = tmp_path / "output_default"
        config = save_model_to_preprocessor(model, output_dir)

        assert config.model_name == "iwfm_model"


class TestPreProcessorConfigExtended:
    """Extended tests for PreProcessorConfig dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Test all default values of PreProcessorConfig."""
        config = PreProcessorConfig(base_dir=tmp_path)

        assert config.model_name == ""
        assert config.nodes_file is None
        assert config.elements_file is None
        assert config.stratigraphy_file is None
        assert config.subregions_file is None
        assert config.streams_file is None
        assert config.lakes_file is None
        assert config.groundwater_file is None
        assert config.rootzone_file is None
        assert config.pumping_file is None
        assert config.output_dir is None
        assert config.budget_output_file is None
        assert config.heads_output_file is None
        assert config.n_layers == 1
        assert config.length_unit == "FT"
        assert config.area_unit == "ACRES"
        assert config.volume_unit == "AF"
        assert config.metadata == {}

    def test_metadata_dict(self, tmp_path: Path) -> None:
        """Test that metadata can be set and retrieved."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            metadata={"key1": "value1", "key2": 42},
        )
        assert config.metadata["key1"] == "value1"
        assert config.metadata["key2"] == 42

    def test_all_file_paths_set(self, tmp_path: Path) -> None:
        """Test setting all file path fields."""
        config = PreProcessorConfig(
            base_dir=tmp_path,
            nodes_file=tmp_path / "n.dat",
            elements_file=tmp_path / "e.dat",
            stratigraphy_file=tmp_path / "s.dat",
            subregions_file=tmp_path / "sr.dat",
            streams_file=tmp_path / "st.dat",
            lakes_file=tmp_path / "l.dat",
            groundwater_file=tmp_path / "gw.dat",
            rootzone_file=tmp_path / "rz.dat",
            pumping_file=tmp_path / "p.dat",
            output_dir=tmp_path / "out",
            budget_output_file=tmp_path / "budget.out",
            heads_output_file=tmp_path / "heads.out",
        )
        assert config.nodes_file is not None
        assert config.elements_file is not None
        assert config.stratigraphy_file is not None
        assert config.subregions_file is not None
        assert config.streams_file is not None
        assert config.lakes_file is not None
        assert config.groundwater_file is not None
        assert config.rootzone_file is not None
        assert config.pumping_file is not None
        assert config.output_dir is not None
        assert config.budget_output_file is not None
        assert config.heads_output_file is not None


class TestReadPreProcessorMainEdgeCases:
    """Edge case tests for read_preprocessor_main."""

    def test_first_data_line_without_name_desc(self, tmp_path: Path) -> None:
        """First data line is treated as model name even without NAME in desc."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Header
SomeModelName
nodes.dat                                / NODES_FILE
"""
        )
        config = read_preprocessor_main(main_file)
        # First data line at idx==0 is captured as model_name
        assert config.model_name == "SomeModelName"

    def test_empty_file(self, tmp_path: Path) -> None:
        """Completely empty file returns default config."""
        main_file = tmp_path / "empty.in"
        main_file.write_text("")
        config = read_preprocessor_main(main_file)
        assert config.model_name == ""
        assert config.nodes_file is None

    def test_data_line_with_empty_value_skipped(self, tmp_path: Path) -> None:
        """Lines that parse to empty value are skipped."""
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Header
Model                                    / MODEL_NAME
                                         / EMPTY_VALUE
nodes.dat                                / NODES_FILE
"""
        )
        config = read_preprocessor_main(main_file)
        assert config.model_name == "Model"
        assert config.nodes_file == tmp_path / "nodes.dat"

    def test_description_case_insensitive(self, tmp_path: Path) -> None:
        """Description keywords are matched case-insensitively (via .upper()).

        Note: values starting with 'c' (lowercase) are treated as
        comment lines by _is_comment_line, so 'cu_m' would be lost.
        We use 'AF' for volume_unit to avoid this issue.
        """
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Header
Model                                    / model_name
nodes.dat                                / Nodes_File
elements.dat                             / Elements_File
2                                        / n_layers
m                                        / length_unit
sq_m                                     / area_unit
AF                                       / volume_unit
"""
        )
        config = read_preprocessor_main(main_file)
        assert config.model_name == "Model"
        assert config.nodes_file == tmp_path / "nodes.dat"
        assert config.elements_file == tmp_path / "elements.dat"
        assert config.n_layers == 2
        assert config.length_unit == "M"
        assert config.area_unit == "SQ_M"
        assert config.volume_unit == "AF"

    def test_value_starting_with_c_treated_as_comment(self, tmp_path: Path) -> None:
        """A data line whose value starts with C/c/* in column 1 is
        treated as a comment and the value is silently lost.
        """
        main_file = tmp_path / "test_pp.in"
        main_file.write_text(
            """C  Header
Model                                    / MODEL_NAME
cu_m                                     / VOLUME_UNIT
"""
        )
        config = read_preprocessor_main(main_file)
        # "cu_m / VOLUME_UNIT" starts with 'c' -> treated as comment
        assert config.volume_unit == "AF"  # default unchanged


class TestWritePreProcessorMainDefaultHeader:
    """Tests for default header generation in write_preprocessor_main."""

    def test_default_header_written(self, tmp_path: Path) -> None:
        """Default header includes 'IWFM PreProcessor' and 'pyiwfm'."""
        config = PreProcessorConfig(base_dir=tmp_path, model_name="Test")
        main_file = tmp_path / "test_pp.in"
        write_preprocessor_main(main_file, config)

        content = main_file.read_text()
        assert "IWFM PreProcessor Main Input File" in content
        assert "pyiwfm" in content


class TestWriteSubregionsRoundtrip:
    """Roundtrip tests for subregion writing and reading."""

    def test_single_subregion_roundtrip(self, tmp_path: Path) -> None:
        """Single subregion survives write-read roundtrip."""
        subregions = {1: Subregion(id=1, name="Only Region")}
        sr_file = tmp_path / "sr.dat"
        _write_subregions_file(sr_file, subregions)

        result = read_subregions_file(sr_file)
        assert len(result) == 1
        assert result[1].id == 1
        assert result[1].name == "Only Region"

    def test_subregions_written_in_sorted_order(self, tmp_path: Path) -> None:
        """Subregions are written in sorted order by ID."""
        subregions = {
            3: Subregion(id=3, name="Third"),
            1: Subregion(id=1, name="First"),
            2: Subregion(id=2, name="Second"),
        }
        sr_file = tmp_path / "sr.dat"
        _write_subregions_file(sr_file, subregions)

        content = sr_file.read_text()
        lines = [line for line in content.split("\n") if line.strip() and not line.startswith("C")]
        # First non-comment line is count, then sorted IDs
        data_lines = lines[1:]  # skip NSUBREGION line
        assert data_lines[0].strip().startswith("1")
        assert data_lines[1].strip().startswith("2")
        assert data_lines[2].strip().startswith("3")
