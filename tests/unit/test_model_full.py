"""
Comprehensive tests for pyiwfm.core.model module.

Tests the IWFMModel class which orchestrates all model components
including mesh, stratigraphy, groundwater, streams, lakes, and rootzone.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyiwfm.core.exceptions import ValidationError
from pyiwfm.core.model import IWFMModel

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_mesh():
    """Create a mock mesh/AppGrid."""
    mesh = MagicMock()
    mesh.n_nodes = 100
    mesh.n_elements = 80
    mesh.n_subregions = 3
    mesh.nodes = {i: MagicMock(id=i, x=float(i), y=float(i)) for i in range(1, 101)}
    mesh.elements = {i: MagicMock(id=i) for i in range(1, 81)}
    mesh.subregions = {1: MagicMock(name="Region 1"), 2: MagicMock(name="Region 2")}
    mesh.validate = MagicMock()
    mesh.compute_areas = MagicMock()
    mesh.compute_connectivity = MagicMock()
    return mesh


@pytest.fixture
def mock_stratigraphy():
    """Create a mock stratigraphy."""
    strat = MagicMock()
    strat.n_nodes = 100
    strat.n_layers = 3
    strat.validate = MagicMock(return_value=[])
    return strat


@pytest.fixture
def mock_groundwater():
    """Create a mock groundwater component."""
    gw = MagicMock()
    gw.n_wells = 50
    gw.n_boundary_conditions = 10
    gw.n_tile_drains = 5
    gw.aquifer_params = MagicMock()
    gw.wells = {i: MagicMock(id=i) for i in range(1, 51)}
    gw.validate = MagicMock()
    return gw


@pytest.fixture
def mock_streams():
    """Create a mock stream component."""
    streams = MagicMock()
    streams.n_nodes = 200
    streams.n_reaches = 15
    streams.n_diversions = 8
    streams.n_bypasses = 3
    streams.nodes = {i: MagicMock(id=i) for i in range(1, 201)}
    streams.validate = MagicMock()
    return streams


@pytest.fixture
def mock_lakes():
    """Create a mock lake component."""
    lakes = MagicMock()
    lakes.n_lakes = 4
    lakes.n_lake_elements = 20
    lakes.validate = MagicMock()
    return lakes


@pytest.fixture
def mock_rootzone():
    """Create a mock root zone component."""
    rz = MagicMock()
    rz.n_crop_types = 12
    rz.element_landuse = {i: MagicMock() for i in range(1, 81)}
    rz.soil_params = {i: MagicMock() for i in range(1, 6)}
    rz.validate = MagicMock()
    return rz


@pytest.fixture
def model_with_mesh(mock_mesh, mock_stratigraphy):
    """Create a model with mesh and stratigraphy."""
    return IWFMModel(
        name="TestModel",
        mesh=mock_mesh,
        stratigraphy=mock_stratigraphy,
    )


@pytest.fixture
def complete_model(
    mock_mesh, mock_stratigraphy, mock_groundwater, mock_streams, mock_lakes, mock_rootzone
):
    """Create a complete model with all components."""
    return IWFMModel(
        name="CompleteModel",
        mesh=mock_mesh,
        stratigraphy=mock_stratigraphy,
        groundwater=mock_groundwater,
        streams=mock_streams,
        lakes=mock_lakes,
        rootzone=mock_rootzone,
        metadata={"source": "test"},
    )


# =============================================================================
# Basic Model Creation Tests
# =============================================================================


class TestIWFMModelCreation:
    """Test IWFMModel basic creation."""

    def test_create_minimal_model(self):
        """Test creating model with just a name."""
        model = IWFMModel(name="TestModel")
        assert model.name == "TestModel"
        assert model.mesh is None
        assert model.stratigraphy is None
        assert model.groundwater is None
        assert model.streams is None
        assert model.lakes is None
        assert model.rootzone is None
        assert model.metadata == {}

    def test_create_model_with_mesh(self, mock_mesh):
        """Test creating model with mesh."""
        model = IWFMModel(name="TestModel", mesh=mock_mesh)
        assert model.mesh == mock_mesh
        assert model.n_nodes == 100
        assert model.n_elements == 80

    def test_create_model_with_metadata(self):
        """Test creating model with metadata."""
        metadata = {"source": "test", "version": "1.0"}
        model = IWFMModel(name="TestModel", metadata=metadata)
        assert model.metadata == metadata

    def test_create_complete_model(self, complete_model):
        """Test creating model with all components."""
        assert complete_model.name == "CompleteModel"
        assert complete_model.mesh is not None
        assert complete_model.stratigraphy is not None
        assert complete_model.groundwater is not None
        assert complete_model.streams is not None
        assert complete_model.lakes is not None
        assert complete_model.rootzone is not None


# =============================================================================
# Property Tests
# =============================================================================


class TestIWFMModelProperties:
    """Test IWFMModel properties."""

    def test_n_nodes_with_mesh(self, model_with_mesh):
        """Test n_nodes property with mesh."""
        assert model_with_mesh.n_nodes == 100

    def test_n_nodes_without_mesh(self):
        """Test n_nodes property without mesh."""
        model = IWFMModel(name="Empty")
        assert model.n_nodes == 0

    def test_n_elements_with_mesh(self, model_with_mesh):
        """Test n_elements property with mesh."""
        assert model_with_mesh.n_elements == 80

    def test_n_elements_without_mesh(self):
        """Test n_elements property without mesh."""
        model = IWFMModel(name="Empty")
        assert model.n_elements == 0

    def test_n_layers_with_stratigraphy(self, model_with_mesh):
        """Test n_layers property with stratigraphy."""
        assert model_with_mesh.n_layers == 3

    def test_n_layers_without_stratigraphy(self, mock_mesh):
        """Test n_layers property without stratigraphy."""
        model = IWFMModel(name="Test", mesh=mock_mesh)
        assert model.n_layers == 0

    def test_grid_alias(self, model_with_mesh):
        """Test grid property is alias for mesh."""
        assert model_with_mesh.grid is model_with_mesh.mesh

    def test_grid_setter(self, mock_mesh):
        """Test grid setter modifies mesh."""
        model = IWFMModel(name="Test")
        model.grid = mock_mesh
        assert model.mesh is mock_mesh


class TestIWFMModelComponentProperties:
    """Test component-related properties."""

    def test_n_wells_with_groundwater(self, complete_model):
        """Test n_wells property with groundwater component."""
        assert complete_model.n_wells == 50

    def test_n_wells_without_groundwater(self, model_with_mesh):
        """Test n_wells property without groundwater component."""
        assert model_with_mesh.n_wells == 0

    def test_n_stream_nodes_with_streams(self, complete_model):
        """Test n_stream_nodes property with stream component."""
        assert complete_model.n_stream_nodes == 200

    def test_n_stream_nodes_without_streams(self, model_with_mesh):
        """Test n_stream_nodes property without stream component."""
        assert model_with_mesh.n_stream_nodes == 0

    def test_n_stream_reaches_with_streams(self, complete_model):
        """Test n_stream_reaches property with stream component."""
        assert complete_model.n_stream_reaches == 15

    def test_n_stream_reaches_without_streams(self, model_with_mesh):
        """Test n_stream_reaches property without stream component."""
        assert model_with_mesh.n_stream_reaches == 0

    def test_n_diversions_with_streams(self, complete_model):
        """Test n_diversions property with stream component."""
        assert complete_model.n_diversions == 8

    def test_n_diversions_without_streams(self, model_with_mesh):
        """Test n_diversions property without stream component."""
        assert model_with_mesh.n_diversions == 0

    def test_n_lakes_with_lakes(self, complete_model):
        """Test n_lakes property with lake component."""
        assert complete_model.n_lakes == 4

    def test_n_lakes_without_lakes(self, model_with_mesh):
        """Test n_lakes property without lake component."""
        assert model_with_mesh.n_lakes == 0

    def test_n_crop_types_with_rootzone(self, complete_model):
        """Test n_crop_types property with rootzone component."""
        assert complete_model.n_crop_types == 12

    def test_n_crop_types_without_rootzone(self, model_with_mesh):
        """Test n_crop_types property without rootzone component."""
        assert model_with_mesh.n_crop_types == 0


class TestIWFMModelBooleanProperties:
    """Test boolean component properties."""

    def test_has_groundwater_true(self, complete_model):
        """Test has_groundwater is True when component exists."""
        assert complete_model.has_groundwater is True

    def test_has_groundwater_false(self, model_with_mesh):
        """Test has_groundwater is False when component is None."""
        assert model_with_mesh.has_groundwater is False

    def test_has_streams_true(self, complete_model):
        """Test has_streams is True when component exists."""
        assert complete_model.has_streams is True

    def test_has_streams_false(self, model_with_mesh):
        """Test has_streams is False when component is None."""
        assert model_with_mesh.has_streams is False

    def test_has_lakes_true(self, complete_model):
        """Test has_lakes is True when component exists."""
        assert complete_model.has_lakes is True

    def test_has_lakes_false(self, model_with_mesh):
        """Test has_lakes is False when component is None."""
        assert model_with_mesh.has_lakes is False

    def test_has_rootzone_true(self, complete_model):
        """Test has_rootzone is True when component exists."""
        assert complete_model.has_rootzone is True

    def test_has_rootzone_false(self, model_with_mesh):
        """Test has_rootzone is False when component is None."""
        assert model_with_mesh.has_rootzone is False


# =============================================================================
# Validation Tests
# =============================================================================


class TestIWFMModelValidation:
    """Test IWFMModel validation."""

    def test_validate_valid_model(self, model_with_mesh):
        """Test validation of valid model."""
        errors = model_with_mesh.validate()
        assert errors == []
        model_with_mesh.mesh.validate.assert_called_once()
        model_with_mesh.stratigraphy.validate.assert_called_once()

    def test_validate_no_mesh(self, mock_stratigraphy):
        """Test validation fails without mesh."""
        model = IWFMModel(name="Test", stratigraphy=mock_stratigraphy)
        with pytest.raises(ValidationError) as exc_info:
            model.validate()
        # Check the errors attribute for specific error message
        assert any("Model has no mesh" in e for e in exc_info.value.errors)

    def test_validate_no_stratigraphy(self, mock_mesh):
        """Test validation fails without stratigraphy."""
        model = IWFMModel(name="Test", mesh=mock_mesh)
        with pytest.raises(ValidationError) as exc_info:
            model.validate()
        assert any("Model has no stratigraphy" in e for e in exc_info.value.errors)

    def test_validate_node_count_mismatch(self, mock_mesh, mock_stratigraphy):
        """Test validation detects node count mismatch."""
        mock_stratigraphy.n_nodes = 50  # Different from mesh.n_nodes = 100
        model = IWFMModel(name="Test", mesh=mock_mesh, stratigraphy=mock_stratigraphy)
        with pytest.raises(ValidationError) as exc_info:
            model.validate()
        assert any("Node count mismatch" in e for e in exc_info.value.errors)

    def test_validate_mesh_validation_failure(self, mock_mesh, mock_stratigraphy):
        """Test validation handles mesh validation failure."""
        mock_mesh.validate.side_effect = Exception("Mesh error")
        model = IWFMModel(name="Test", mesh=mock_mesh, stratigraphy=mock_stratigraphy)
        with pytest.raises(ValidationError) as exc_info:
            model.validate()
        assert any("Mesh validation failed" in e for e in exc_info.value.errors)

    def test_validate_stratigraphy_validation_failure(self, mock_mesh, mock_stratigraphy):
        """Test validation handles stratigraphy validation failure."""
        mock_stratigraphy.validate.side_effect = Exception("Strat error")
        model = IWFMModel(name="Test", mesh=mock_mesh, stratigraphy=mock_stratigraphy)
        with pytest.raises(ValidationError) as exc_info:
            model.validate()
        assert any("Stratigraphy validation failed" in e for e in exc_info.value.errors)

    def test_validate_multiple_errors(self):
        """Test validation collects multiple errors."""
        model = IWFMModel(name="Test")  # No mesh, no stratigraphy
        with pytest.raises(ValidationError) as exc_info:
            model.validate()
        assert len(exc_info.value.errors) >= 2


class TestIWFMModelValidateComponents:
    """Test validate_components method."""

    def test_validate_components_all_valid(self, complete_model):
        """Test validate_components with all valid components."""
        warnings = complete_model.validate_components()
        assert warnings == []
        complete_model.groundwater.validate.assert_called_once()
        complete_model.streams.validate.assert_called_once()
        complete_model.lakes.validate.assert_called_once()
        complete_model.rootzone.validate.assert_called_once()

    def test_validate_components_groundwater_failure(self, complete_model):
        """Test validate_components handles groundwater validation failure."""
        complete_model.groundwater.validate.side_effect = Exception("GW error")
        warnings = complete_model.validate_components()
        assert len(warnings) == 1
        assert "Groundwater validation" in warnings[0]

    def test_validate_components_streams_failure(self, complete_model):
        """Test validate_components handles stream validation failure."""
        complete_model.streams.validate.side_effect = Exception("Stream error")
        warnings = complete_model.validate_components()
        assert len(warnings) == 1
        assert "Stream validation" in warnings[0]

    def test_validate_components_lakes_failure(self, complete_model):
        """Test validate_components handles lake validation failure."""
        complete_model.lakes.validate.side_effect = Exception("Lake error")
        warnings = complete_model.validate_components()
        assert len(warnings) == 1
        assert "Lake validation" in warnings[0]

    def test_validate_components_rootzone_failure(self, complete_model):
        """Test validate_components handles rootzone validation failure."""
        complete_model.rootzone.validate.side_effect = Exception("RZ error")
        warnings = complete_model.validate_components()
        assert len(warnings) == 1
        assert "Root zone validation" in warnings[0]

    def test_validate_components_no_components(self, model_with_mesh):
        """Test validate_components with no dynamic components."""
        warnings = model_with_mesh.validate_components()
        assert warnings == []

    def test_validate_components_multiple_failures(self, complete_model):
        """Test validate_components collects multiple failures."""
        complete_model.groundwater.validate.side_effect = Exception("GW error")
        complete_model.streams.validate.side_effect = Exception("Stream error")
        warnings = complete_model.validate_components()
        assert len(warnings) == 2


# =============================================================================
# Summary and Repr Tests
# =============================================================================


class TestIWFMModelSummary:
    """Test summary method."""

    def test_summary_minimal_model(self):
        """Test summary for minimal model."""
        model = IWFMModel(name="TestModel")
        summary = model.summary()
        assert "IWFM Model: TestModel" in summary
        assert "Nodes: 0" in summary
        assert "Elements: 0" in summary
        assert "Layers: 0" in summary

    def test_summary_with_mesh(self, model_with_mesh):
        """Test summary includes mesh info."""
        summary = model_with_mesh.summary()
        assert "Nodes: 100" in summary
        assert "Elements: 80" in summary
        assert "Layers: 3" in summary
        assert "Subregions: 3" in summary

    def test_summary_complete_model(self, complete_model):
        """Test summary for complete model."""
        summary = complete_model.summary()

        # Mesh info
        assert "Nodes: 100" in summary
        assert "Elements: 80" in summary
        assert "Layers: 3" in summary

        # Groundwater info
        assert "Groundwater Component:" in summary
        assert "Wells: 50" in summary
        assert "Boundary Conditions: 10" in summary
        assert "Tile Drains: 5" in summary
        assert "Aquifer Parameters: Loaded" in summary

        # Stream info
        assert "Stream Component:" in summary
        assert "Stream Nodes: 200" in summary
        assert "Reaches: 15" in summary
        assert "Diversions: 8" in summary
        assert "Bypasses: 3" in summary

        # Lake info
        assert "Lake Component:" in summary
        assert "Lakes: 4" in summary
        assert "Lake Elements: 20" in summary

        # Root zone info
        assert "Root Zone Component:" in summary
        assert "Crop Types: 12" in summary

        # Metadata
        assert "Source: test" in summary

    def test_summary_components_not_loaded(self, model_with_mesh):
        """Test summary shows 'Not loaded' for missing components."""
        summary = model_with_mesh.summary()
        assert "Not loaded" in summary

    def test_summary_no_aquifer_params(self, complete_model):
        """Test summary when aquifer params are None."""
        complete_model.groundwater.aquifer_params = None
        summary = complete_model.summary()
        assert "Aquifer Parameters: Not loaded" in summary


class TestIWFMModelRepr:
    """Test __repr__ method."""

    def test_repr_minimal_model(self):
        """Test repr for minimal model."""
        model = IWFMModel(name="TestModel")
        repr_str = repr(model)
        assert "IWFMModel(name='TestModel'" in repr_str
        assert "n_nodes=0" in repr_str
        assert "n_elements=0" in repr_str
        assert "n_layers=0" in repr_str

    def test_repr_complete_model(self, complete_model):
        """Test repr for complete model."""
        repr_str = repr(complete_model)
        assert "IWFMModel(name='CompleteModel'" in repr_str
        assert "n_nodes=100" in repr_str
        assert "n_elements=80" in repr_str
        assert "n_layers=3" in repr_str


# =============================================================================
# Class Method Loading Tests (using patches on source modules)
# =============================================================================


class TestIWFMModelFromPreprocessor:
    """Test from_preprocessor class method."""

    @patch("pyiwfm.io.preprocessor.read_preprocessor_main")
    @patch("pyiwfm.io.ascii.read_nodes")
    @patch("pyiwfm.io.ascii.read_elements")
    @patch("pyiwfm.core.mesh.AppGrid")
    def test_from_preprocessor_basic(
        self, mock_grid_cls, mock_read_elements, mock_read_nodes, mock_read_pp
    ):
        """Test basic preprocessor loading."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.model_name = "TestModel"
        mock_config.nodes_file = Path("nodes.dat")
        mock_config.elements_file = Path("elements.dat")
        mock_config.stratigraphy_file = None
        mock_config.streams_file = None
        mock_config.lakes_file = None
        mock_config.subregions_file = None
        mock_config.length_unit = "FT"
        mock_config.area_unit = "ACRE"
        mock_config.volume_unit = "AF"
        mock_read_pp.return_value = mock_config

        mock_read_nodes.return_value = ({1: MagicMock(), 2: MagicMock()}, 1.0)
        mock_read_elements.return_value = ({1: MagicMock()}, 1, {})

        mock_grid = MagicMock()
        mock_grid_cls.return_value = mock_grid

        model = IWFMModel.from_preprocessor("preprocessor.in")

        assert model.name == "TestModel"
        assert model.mesh == mock_grid
        mock_grid.compute_areas.assert_called_once()
        mock_grid.compute_connectivity.assert_called_once()

    @patch("pyiwfm.io.preprocessor.read_preprocessor_main")
    def test_from_preprocessor_no_nodes_file(self, mock_read_pp):
        """Test error when nodes file not specified."""
        mock_config = MagicMock()
        mock_config.nodes_file = None
        mock_read_pp.return_value = mock_config

        with pytest.raises(Exception) as exc_info:
            IWFMModel.from_preprocessor("preprocessor.in")
        assert "Nodes file not specified" in str(exc_info.value)

    @patch("pyiwfm.io.preprocessor.read_preprocessor_main")
    @patch("pyiwfm.io.ascii.read_nodes")
    def test_from_preprocessor_no_elements_file(self, mock_read_nodes, mock_read_pp):
        """Test error when elements file not specified."""
        mock_config = MagicMock()
        mock_config.nodes_file = Path("nodes.dat")
        mock_config.elements_file = None
        mock_read_pp.return_value = mock_config
        mock_read_nodes.return_value = ({}, 1.0)

        with pytest.raises(Exception) as exc_info:
            IWFMModel.from_preprocessor("preprocessor.in")
        assert "Elements file not specified" in str(exc_info.value)


class TestIWFMModelFromPreprocessorBinary:
    """Test from_preprocessor_binary class method."""

    @patch("pyiwfm.core.model._binary_data_to_model")
    @patch("pyiwfm.io.preprocessor_binary.PreprocessorBinaryReader.read")
    def test_from_preprocessor_binary_basic(self, mock_read, mock_to_model):
        """Test basic binary loading."""
        mock_data = MagicMock()
        mock_read.return_value = mock_data
        mock_model = MagicMock(spec=IWFMModel)
        mock_model.metadata = {}
        mock_model.streams = None
        mock_to_model.return_value = mock_model

        result = IWFMModel.from_preprocessor_binary("model.bin", name="TestModel")

        assert result is mock_model
        mock_read.assert_called_once()
        mock_to_model.assert_called_once_with(mock_data, name="TestModel")

    @patch("pyiwfm.core.model._binary_data_to_model")
    @patch("pyiwfm.io.preprocessor_binary.PreprocessorBinaryReader.read")
    def test_from_preprocessor_binary_default_name(self, mock_read, mock_to_model):
        """Test binary loading uses file stem as default name."""
        mock_data = MagicMock()
        mock_read.return_value = mock_data
        mock_model = MagicMock(spec=IWFMModel)
        mock_model.metadata = {}
        mock_model.streams = None
        mock_to_model.return_value = mock_model

        IWFMModel.from_preprocessor_binary("model.bin")

        mock_to_model.assert_called_once_with(mock_data, name="model")


class TestIWFMModelFromSimulation:
    """Test from_simulation class method."""

    @patch("pyiwfm.io.model_loader.load_complete_model")
    def test_from_simulation_basic(self, mock_load):
        """Test basic simulation loading."""
        mock_model = MagicMock(spec=IWFMModel)
        mock_load.return_value = mock_model

        result = IWFMModel.from_simulation("simulation.in")

        assert result == mock_model
        mock_load.assert_called_once_with("simulation.in")


class TestIWFMModelFromHDF5:
    """Test from_hdf5 class method."""

    @patch("pyiwfm.io.hdf5.read_model_hdf5")
    def test_from_hdf5_basic(self, mock_read):
        """Test basic HDF5 loading."""
        mock_model = MagicMock(spec=IWFMModel)
        mock_read.return_value = mock_model

        result = IWFMModel.from_hdf5("model.h5")

        assert result == mock_model
        mock_read.assert_called_once()


# =============================================================================
# Instance Method Saving Tests
# =============================================================================


class TestIWFMModelToPreprocessor:
    """Test to_preprocessor method."""

    @patch("pyiwfm.io.preprocessor.save_model_to_preprocessor")
    def test_to_preprocessor_basic(self, mock_save, complete_model, tmp_path):
        """Test basic preprocessor output."""
        mock_config = MagicMock()
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = tmp_path / "strat.dat"
        mock_config.subregions_file = tmp_path / "subregions.dat"
        mock_save.return_value = mock_config

        files = complete_model.to_preprocessor(tmp_path)

        assert "nodes" in files
        assert "elements" in files
        assert "stratigraphy" in files
        assert "subregions" in files
        mock_save.assert_called_once_with(complete_model, tmp_path, "CompleteModel")

    @patch("pyiwfm.io.preprocessor.save_model_to_preprocessor")
    def test_to_preprocessor_partial_output(self, mock_save, model_with_mesh, tmp_path):
        """Test preprocessor output with some files None."""
        mock_config = MagicMock()
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = None
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None
        mock_save.return_value = mock_config

        files = model_with_mesh.to_preprocessor(tmp_path)

        assert "nodes" in files
        assert "elements" not in files


class TestIWFMModelToSimulation:
    """Test to_simulation method."""

    @patch("pyiwfm.io.preprocessor.save_complete_model")
    def test_to_simulation_basic(self, mock_save, complete_model, tmp_path):
        """Test basic simulation output."""
        expected_files = {"main": tmp_path / "simulation.in"}
        mock_save.return_value = expected_files

        files = complete_model.to_simulation(tmp_path)

        assert files == expected_files
        mock_save.assert_called_once_with(
            complete_model,
            tmp_path,
            timeseries_format="text",
            file_paths=None,
        )


class TestIWFMModelToHDF5:
    """Test to_hdf5 method."""

    @patch("pyiwfm.io.hdf5.write_model_hdf5")
    def test_to_hdf5_basic(self, mock_write, complete_model, tmp_path):
        """Test basic HDF5 output."""
        output_file = tmp_path / "model.h5"

        complete_model.to_hdf5(output_file)

        mock_write.assert_called_once_with(output_file, complete_model)


class TestIWFMModelToBinary:
    """Test to_binary method."""

    @patch("pyiwfm.io.binary.write_binary_mesh")
    @patch("pyiwfm.io.binary.write_binary_stratigraphy")
    def test_to_binary_with_mesh_and_strat(
        self, mock_write_strat, mock_write_mesh, complete_model, tmp_path
    ):
        """Test binary output with mesh and stratigraphy."""
        output_file = tmp_path / "model.bin"

        complete_model.to_binary(output_file)

        mock_write_mesh.assert_called_once_with(output_file, complete_model.mesh)
        mock_write_strat.assert_called_once()

    @patch("pyiwfm.io.binary.write_binary_mesh")
    @patch("pyiwfm.io.binary.write_binary_stratigraphy")
    def test_to_binary_mesh_only(self, mock_write_strat, mock_write_mesh, mock_mesh, tmp_path):
        """Test binary output with only mesh."""
        model = IWFMModel(name="Test", mesh=mock_mesh)
        output_file = tmp_path / "model.bin"

        model.to_binary(output_file)

        mock_write_mesh.assert_called_once()
        mock_write_strat.assert_not_called()

    @patch("pyiwfm.io.binary.write_binary_mesh")
    @patch("pyiwfm.io.binary.write_binary_stratigraphy")
    def test_to_binary_empty_model(self, mock_write_strat, mock_write_mesh, tmp_path):
        """Test binary output with empty model."""
        model = IWFMModel(name="Empty")
        output_file = tmp_path / "model.bin"

        model.to_binary(output_file)

        mock_write_mesh.assert_not_called()
        mock_write_strat.assert_not_called()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIWFMModelIntegration:
    """Integration tests for IWFMModel."""

    def test_model_workflow(self, mock_mesh, mock_stratigraphy):
        """Test typical model workflow."""
        # Create model
        model = IWFMModel(
            name="IntegrationTest",
            mesh=mock_mesh,
            stratigraphy=mock_stratigraphy,
            metadata={"source": "integration_test"},
        )

        # Access properties
        assert model.n_nodes == 100
        assert model.n_elements == 80
        assert model.n_layers == 3

        # Validate
        errors = model.validate()
        assert errors == []

        # Get summary
        summary = model.summary()
        assert "IntegrationTest" in summary

        # Check repr
        repr_str = repr(model)
        assert "IntegrationTest" in repr_str

    def test_complete_model_workflow(self, complete_model):
        """Test workflow with complete model."""
        # Access all component properties
        assert complete_model.n_wells == 50
        assert complete_model.n_stream_nodes == 200
        assert complete_model.n_lakes == 4
        assert complete_model.n_crop_types == 12

        # Check boolean properties
        assert complete_model.has_groundwater
        assert complete_model.has_streams
        assert complete_model.has_lakes
        assert complete_model.has_rootzone

        # Validate components
        warnings = complete_model.validate_components()
        assert warnings == []

        # Get complete summary
        summary = complete_model.summary()
        assert "Wells: 50" in summary
        assert "Stream Nodes: 200" in summary
        assert "Lakes: 4" in summary
        assert "Crop Types: 12" in summary

    def test_model_metadata_updates(self, model_with_mesh):
        """Test metadata can be updated."""
        model_with_mesh.metadata["custom_key"] = "custom_value"
        model_with_mesh.metadata["version"] = "2.0"

        assert model_with_mesh.metadata["custom_key"] == "custom_value"
        assert model_with_mesh.metadata["version"] == "2.0"

    def test_model_component_assignment(self, model_with_mesh, mock_groundwater):
        """Test components can be assigned after creation."""
        assert model_with_mesh.groundwater is None
        assert model_with_mesh.has_groundwater is False

        model_with_mesh.groundwater = mock_groundwater

        assert model_with_mesh.has_groundwater is True
        assert model_with_mesh.n_wells == 50
