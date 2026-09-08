"""Comprehensive tests for pyiwfm.core.model module.

Covers:
- _apply_kh_anomalies() helper function
- _apply_parametric_grids() helper function
- IWFMModel.from_preprocessor() class method
- IWFMModel.from_simulation_with_preprocessor() class method
- IWFMModel.summary() and validate_components() with all component types
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyiwfm.components.groundwater import AppGW, AquiferParameters
from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.model import IWFMModel, _apply_kh_anomalies, _apply_parametric_grids
from pyiwfm.io.groundwater import KhAnomalyEntry

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def quad_mesh() -> AppGrid:
    """A small 4-node quad mesh with IDs 1-4."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=100.0, y=100.0),
        4: Node(id=4, x=0.0, y=100.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
    }
    return AppGrid(nodes=nodes, elements=elements)


@pytest.fixture
def six_node_mesh() -> AppGrid:
    """A 6-node mesh with two quads sharing an edge."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=200.0, y=0.0),
        4: Node(id=4, x=0.0, y=100.0),
        5: Node(id=5, x=100.0, y=100.0),
        6: Node(id=6, x=200.0, y=100.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
    }
    return AppGrid(nodes=nodes, elements=elements)


@pytest.fixture
def mock_pp_config(tmp_path: Path) -> MagicMock:
    """Mock preprocessor config with required file references."""
    cfg = MagicMock()
    cfg.model_name = "TestModel"
    cfg.nodes_file = tmp_path / "nodes.dat"
    cfg.elements_file = tmp_path / "elements.dat"
    cfg.stratigraphy_file = None
    cfg.subregions_file = None
    cfg.streams_file = None
    cfg.lakes_file = None
    cfg.length_unit = "FT"
    cfg.area_unit = "SQFT"
    cfg.volume_unit = "CUFT"
    return cfg


def _make_mock_mesh() -> MagicMock:
    """Return a MagicMock that behaves like an AppGrid for patching."""
    m = MagicMock()
    m.n_nodes = 4
    m.n_elements = 1
    m.n_subregions = 0
    m.nodes = {
        1: MagicMock(id=1, x=0.0, y=0.0),
        2: MagicMock(id=2, x=100.0, y=0.0),
        3: MagicMock(id=3, x=100.0, y=100.0),
        4: MagicMock(id=4, x=0.0, y=100.0),
    }
    m.elements = {1: MagicMock(id=1)}
    return m


def _make_sim_config(base_dir: Path) -> MagicMock:
    """Return a MagicMock that looks like SimulationConfig for the method."""
    from pyiwfm.core.timeseries import TimeUnit

    cfg = MagicMock()
    cfg.start_date = datetime(2000, 1, 1)
    cfg.end_date = datetime(2000, 12, 31)
    cfg.time_step_length = 1
    cfg.time_step_unit = TimeUnit.DAY
    cfg.matrix_solver = 2
    cfg.relaxation = 1.0
    cfg.max_iterations = 50
    cfg.max_supply_iterations = 50
    cfg.convergence_tolerance = 1e-6
    cfg.convergence_volume = 0.0
    cfg.convergence_supply = 0.001
    cfg.supply_adjust_option = 0
    cfg.debug_flag = 0
    cfg.cache_size = 500000
    cfg.binary_preprocessor_file = None
    cfg.irrigation_fractions_file = None
    cfg.supply_adjust_file = None
    cfg.precipitation_file = None
    cfg.et_file = None
    cfg.title_lines = []
    cfg.groundwater_file = None
    cfg.streams_file = None
    cfg.lakes_file = None
    cfg.rootzone_file = None
    cfg.small_watershed_file = None
    cfg.unsaturated_zone_file = None
    return cfg


# ===========================================================================
# 1. _apply_kh_anomalies() tests
# ===========================================================================


class TestApplyKhAnomalies:
    """Tests for the _apply_kh_anomalies helper function."""

    def test_kh_is_none_returns_zero(self, quad_mesh: AppGrid) -> None:
        """When params.kh is None, no anomalies can be applied."""
        params = AquiferParameters(n_nodes=4, n_layers=2, kh=None)
        anomalies = [KhAnomalyEntry(element_id=1, kh_per_layer=[10.0, 20.0])]
        result = _apply_kh_anomalies(params, anomalies, quad_mesh)
        assert result == 0

    def test_element_not_found_skips(self, quad_mesh: AppGrid) -> None:
        """Anomaly referencing nonexistent element is silently skipped."""
        kh = np.ones((4, 2), dtype=np.float64)
        params = AquiferParameters(n_nodes=4, n_layers=2, kh=kh)
        anomalies = [KhAnomalyEntry(element_id=99, kh_per_layer=[10.0, 20.0])]
        result = _apply_kh_anomalies(params, anomalies, quad_mesh)
        assert result == 0
        # kh should be unchanged
        np.testing.assert_array_equal(params.kh, np.ones((4, 2)))

    def test_node_id_not_in_lookup_skips(self) -> None:
        """If an element vertex is not in the mesh's node dict, that vertex is skipped."""
        # Build a mesh where element references node 99 that is NOT in nodes dict
        nodes = {1: Node(id=1, x=0.0, y=0.0), 2: Node(id=2, x=1.0, y=0.0)}
        # We'll create an element with a hand-built vertex list including 99
        # Element validates 3 or 4 vertices, so we use a triangle
        elem = Element(id=1, vertices=(1, 2, 99))
        elements = {1: elem}
        # Note: AppGrid won't complain at construction, only at validate()
        mesh = AppGrid(nodes=nodes, elements=elements)

        kh = np.ones((2, 1), dtype=np.float64)
        params = AquiferParameters(n_nodes=2, n_layers=1, kh=kh)
        anomalies = [KhAnomalyEntry(element_id=1, kh_per_layer=[42.0])]
        result = _apply_kh_anomalies(params, anomalies, mesh)
        # Still counts as applied (element found) even though one vertex was skipped
        assert result == 1
        # Nodes 1 and 2 (indices 0 and 1) should be overwritten
        assert params.kh[0, 0] == 42.0
        assert params.kh[1, 0] == 42.0

    def test_successful_single_anomaly_quad(self, quad_mesh: AppGrid) -> None:
        """A single anomaly on a 4-node quad overwrites all 4 vertex nodes."""
        kh = np.zeros((4, 2), dtype=np.float64)
        params = AquiferParameters(n_nodes=4, n_layers=2, kh=kh)
        anomalies = [KhAnomalyEntry(element_id=1, kh_per_layer=[5.0, 15.0])]
        result = _apply_kh_anomalies(params, anomalies, quad_mesh)
        assert result == 1
        for i in range(4):
            assert params.kh[i, 0] == 5.0
            assert params.kh[i, 1] == 15.0

    def test_multiple_anomalies_with_layer_clamp(self, six_node_mesh: AppGrid) -> None:
        """Multiple anomalies with more kh_per_layer entries than n_layers are clamped."""
        kh = np.zeros((6, 2), dtype=np.float64)
        params = AquiferParameters(n_nodes=6, n_layers=2, kh=kh)
        anomalies = [
            KhAnomalyEntry(element_id=1, kh_per_layer=[3.0, 6.0, 99.0]),
            KhAnomalyEntry(element_id=2, kh_per_layer=[7.0, 14.0]),
        ]
        result = _apply_kh_anomalies(params, anomalies, six_node_mesh)
        assert result == 2
        # Element 1 vertices: (1, 2, 5, 4) -> indices 0,1,4,3
        # Element 2 vertices: (2, 3, 6, 5) -> indices 1,2,5,4
        # Sorted node IDs [1,2,3,4,5,6], so node_id_to_idx: {1:0,2:1,3:2,4:3,5:4,6:5}
        # Node 2 (idx 1) is in both; second anomaly writes last
        assert params.kh[0, 0] == 3.0  # node 1 from elem 1
        assert params.kh[0, 1] == 6.0
        assert params.kh[1, 0] == 7.0  # node 2 overwritten by elem 2
        assert params.kh[1, 1] == 14.0
        assert params.kh[2, 0] == 7.0  # node 3 from elem 2
        assert params.kh[3, 0] == 3.0  # node 4 from elem 1
        assert params.kh[4, 0] == 7.0  # node 5 overwritten by elem 2

    def test_return_count_accuracy(self, quad_mesh: AppGrid) -> None:
        """Return count matches the number of anomalies whose elements were found."""
        kh = np.ones((4, 1), dtype=np.float64)
        params = AquiferParameters(n_nodes=4, n_layers=1, kh=kh)
        anomalies = [
            KhAnomalyEntry(element_id=1, kh_per_layer=[10.0]),
            KhAnomalyEntry(element_id=999, kh_per_layer=[20.0]),
            KhAnomalyEntry(element_id=1, kh_per_layer=[30.0]),
        ]
        result = _apply_kh_anomalies(params, anomalies, quad_mesh)
        assert result == 2  # elements 1 found twice, 999 not found

    def test_anomaly_with_empty_kh_per_layer(self, quad_mesh: AppGrid) -> None:
        """An anomaly with empty kh_per_layer list applies nothing but still counts."""
        kh = np.ones((4, 2), dtype=np.float64)
        params = AquiferParameters(n_nodes=4, n_layers=2, kh=kh)
        anomalies = [KhAnomalyEntry(element_id=1, kh_per_layer=[])]
        result = _apply_kh_anomalies(params, anomalies, quad_mesh)
        # Element was found, so it counts as applied
        assert result == 1
        # But kh remains unchanged because min(0, 2) == 0 layers
        np.testing.assert_array_equal(params.kh, np.ones((4, 2)))


# ===========================================================================
# 2. _apply_parametric_grids() tests
# ===========================================================================


class TestApplyParametricGrids:
    """Tests for the _apply_parametric_grids helper function."""

    def test_empty_mesh_returns_false(self) -> None:
        """If mesh has zero nodes, returns False immediately."""
        empty_mesh = AppGrid(nodes={}, elements={})
        gw = MagicMock(spec=AppGW)
        result = _apply_parametric_grids(gw, [], empty_mesh)
        assert result is False

    def test_interpolation_returning_none_leaves_zeros(self, quad_mesh: AppGrid) -> None:
        """Nodes outside the parametric grid get None and keep zero values."""
        grid_data = MagicMock()
        grid_data.node_values = MagicMock()
        grid_data.node_values.shape = (3, 1)  # shape[1] -> n_layers = 1
        grid_data.n_nodes = 3
        grid_data.node_coords = np.array([[0, 0], [1, 0], [0.5, 1.0]])
        grid_data.elements = [(0, 1, 2)]

        with patch(
            "pyiwfm.io.parametric_grid.ParametricGrid.interpolate",
            return_value=None,
        ):
            gw = MagicMock(spec=AppGW)
            result = _apply_parametric_grids(gw, [grid_data], quad_mesh)

        assert result is True
        # set_aquifer_parameters or aquifer_params should have been called
        # with all-zero arrays since interpolation returned None for every node
        call_args = gw.set_aquifer_parameters.call_args
        params = call_args[0][0]
        assert isinstance(params, AquiferParameters)
        np.testing.assert_array_equal(params.kh, np.zeros((4, 1)))

    def test_successful_interpolation_assigns_positive(self, quad_mesh: AppGrid) -> None:
        """Positive interpolated values are assigned to parameter arrays."""
        grid_data = MagicMock()
        grid_data.node_values = MagicMock()
        grid_data.node_values.shape = (3, 1)  # n_layers = 1
        grid_data.n_nodes = 3
        grid_data.node_coords = np.array([[0, 0], [200, 0], [100, 200]])
        grid_data.elements = [(0, 1, 2)]

        # Return values: shape (n_layers, 5) = (1, 5)
        interp_result = np.array([[10.0, 0.5, 0.2, 0.01, 3.0]])

        with patch(
            "pyiwfm.io.parametric_grid.ParametricGrid.interpolate",
            return_value=interp_result,
        ):
            gw = MagicMock(spec=AppGW)
            result = _apply_parametric_grids(gw, [grid_data], quad_mesh)

        assert result is True
        call_args = gw.set_aquifer_parameters.call_args
        params = call_args[0][0]
        assert params.kh[0, 0] == 10.0
        assert params.specific_storage[0, 0] == 0.5
        assert params.specific_yield[0, 0] == 0.2
        assert params.aquitard_kv[0, 0] == 0.01
        assert params.kv[0, 0] == 3.0

    def test_only_positive_values_assigned(self, quad_mesh: AppGrid) -> None:
        """Zero and negative interpolated values are NOT assigned."""
        grid_data = MagicMock()
        grid_data.node_values = MagicMock()
        grid_data.node_values.shape = (3, 1)
        grid_data.n_nodes = 3
        grid_data.node_coords = np.array([[0, 0], [200, 0], [100, 200]])
        grid_data.elements = [(0, 1, 2)]

        # Kh=0 (not assigned), Ss=-1 (not assigned), Sy=0.1 (assigned)
        interp_result = np.array([[0.0, -1.0, 0.1, 0.0, -5.0]])

        with patch(
            "pyiwfm.io.parametric_grid.ParametricGrid.interpolate",
            return_value=interp_result,
        ):
            gw = MagicMock(spec=AppGW)
            result = _apply_parametric_grids(gw, [grid_data], quad_mesh)

        assert result is True
        params = gw.set_aquifer_parameters.call_args[0][0]
        # Only Sy should be non-zero
        assert params.kh[0, 0] == 0.0
        assert params.specific_storage[0, 0] == 0.0
        assert params.specific_yield[0, 0] == 0.1
        assert params.aquitard_kv[0, 0] == 0.0
        assert params.kv[0, 0] == 0.0

    def test_set_aquifer_parameters_called(self, quad_mesh: AppGrid) -> None:
        """set_aquifer_parameters is called with the constructed AquiferParameters."""
        grid_data = MagicMock()
        grid_data.node_values = MagicMock()
        grid_data.node_values.shape = (3, 1)
        grid_data.n_nodes = 3
        grid_data.node_coords = np.array([[0, 0], [200, 0], [100, 200]])
        grid_data.elements = [(0, 1, 2)]

        with patch(
            "pyiwfm.io.parametric_grid.ParametricGrid.interpolate",
            return_value=np.array([[5.0, 1.0, 0.3, 0.02, 2.0]]),
        ):
            gw = MagicMock(spec=AppGW)
            _apply_parametric_grids(gw, [grid_data], quad_mesh)

        gw.set_aquifer_parameters.assert_called_once()

    def test_value_error_fallback_to_direct_assignment(self, quad_mesh: AppGrid) -> None:
        """When set_aquifer_parameters raises ValueError, falls back to direct assignment."""
        grid_data = MagicMock()
        grid_data.node_values = MagicMock()
        grid_data.node_values.shape = (3, 1)
        grid_data.n_nodes = 3
        grid_data.node_coords = np.array([[0, 0], [200, 0], [100, 200]])
        grid_data.elements = [(0, 1, 2)]

        with patch(
            "pyiwfm.io.parametric_grid.ParametricGrid.interpolate",
            return_value=np.array([[5.0, 1.0, 0.3, 0.02, 2.0]]),
        ):
            gw = MagicMock(spec=AppGW)
            gw.set_aquifer_parameters.side_effect = ValueError("mismatch")
            result = _apply_parametric_grids(gw, [grid_data], quad_mesh)

        assert result is True
        # aquifer_params should be set directly
        assert gw.aquifer_params is not None

    def test_returns_true_on_success(self, quad_mesh: AppGrid) -> None:
        """Returns True when interpolation completes."""
        grid_data = MagicMock()
        grid_data.node_values = MagicMock()
        grid_data.node_values.shape = (2, 1)
        grid_data.n_nodes = 2
        grid_data.node_coords = np.array([[0, 0], [200, 200]])
        grid_data.elements = [(0, 1)]

        with (
            patch(
                "pyiwfm.io.parametric_grid.ParametricGrid.interpolate",
                return_value=None,
            ),
            patch(
                "pyiwfm.io.parametric_grid.ParamElement",
            ),
        ):
            gw = MagicMock(spec=AppGW)
            result = _apply_parametric_grids(gw, [grid_data], quad_mesh)

        assert result is True


# ===========================================================================
# 3. IWFMModel.from_preprocessor() tests
# ===========================================================================


class TestFromPreprocessor:
    """Tests for IWFMModel.from_preprocessor() class method."""

    def _patch_preprocessor(
        self,
        pp_config: MagicMock,
        nodes: dict | None = None,
        elements: tuple | None = None,
        mesh_mock: MagicMock | None = None,
    ):
        """Return a context manager patching all imports used by from_preprocessor."""
        if nodes is None:
            nodes = {1: MagicMock(id=1, x=0.0, y=0.0)}
        if elements is None:
            elements = (
                {1: MagicMock(id=1, vertices=(1, 1, 1), subregion=1)},
                1,
                {1: "Subregion 1"},
            )
        if mesh_mock is None:
            mesh_mock = _make_mock_mesh()

        return (
            patch(
                "pyiwfm.io.preprocessor.read_preprocessor_main",
                return_value=pp_config,
            ),
            patch("pyiwfm.io.ascii.read_nodes", return_value=(nodes, 1.0)),
            patch("pyiwfm.io.ascii.read_elements", return_value=elements),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mesh_mock),
        )

    def test_successful_load_with_mesh(self, tmp_path: Path, mock_pp_config: MagicMock) -> None:
        """Successful load creates model with mesh and correct metadata."""
        patches = self._patch_preprocessor(mock_pp_config)
        with patches[0], patches[1], patches[2], patches[3]:
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")
        assert model.name == "TestModel"
        assert model.metadata["source"] == "preprocessor"
        assert model.metadata["length_unit"] == "FT"

    def test_missing_nodes_file_raises(self, tmp_path: Path, mock_pp_config: MagicMock) -> None:
        """FileFormatError when nodes_file is None."""
        mock_pp_config.nodes_file = None
        with patch(
            "pyiwfm.io.preprocessor.read_preprocessor_main",
            return_value=mock_pp_config,
        ):
            with pytest.raises(FileFormatError, match="Nodes file"):
                IWFMModel.from_preprocessor(tmp_path / "pp.in")

    def test_missing_elements_file_raises(self, tmp_path: Path, mock_pp_config: MagicMock) -> None:
        """FileFormatError when elements_file is None."""
        mock_pp_config.elements_file = None
        with (
            patch(
                "pyiwfm.io.preprocessor.read_preprocessor_main",
                return_value=mock_pp_config,
            ),
            patch(
                "pyiwfm.io.ascii.read_nodes",
                return_value=({1: MagicMock()}, 1.0),
            ),
        ):
            with pytest.raises(FileFormatError, match="Elements file"):
                IWFMModel.from_preprocessor(tmp_path / "pp.in")

    def test_stream_loading_success(self, tmp_path: Path, mock_pp_config: MagicMock) -> None:
        """Streams loaded when load_streams=True and file exists."""
        streams_path = tmp_path / "streams.dat"
        streams_path.write_text("fake")
        mock_pp_config.streams_file = streams_path

        mock_stream_reader = MagicMock()
        mock_stream_reader.read_stream_nodes.return_value = {
            1: MagicMock(id=1),
        }

        patches = self._patch_preprocessor(mock_pp_config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch("pyiwfm.io.streams.StreamReader", return_value=mock_stream_reader),
            patch("pyiwfm.components.stream.AppStream") as mock_app_stream_cls,
        ):
            mock_stream_inst = MagicMock()
            mock_app_stream_cls.return_value = mock_stream_inst
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")
        assert model.streams is mock_stream_inst

    def test_stream_loading_exception_stores_error(
        self, tmp_path: Path, mock_pp_config: MagicMock
    ) -> None:
        """Stream loading exception stored in metadata."""
        streams_path = tmp_path / "streams.dat"
        streams_path.write_text("fake")
        mock_pp_config.streams_file = streams_path

        patches = self._patch_preprocessor(mock_pp_config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "pyiwfm.io.streams.StreamReader",
                side_effect=ValueError("stream parse fail"),
            ),
        ):
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")
        assert "streams_load_error" in model.metadata
        assert "stream parse fail" in model.metadata["streams_load_error"]

    def test_lake_loading_success(self, tmp_path: Path, mock_pp_config: MagicMock) -> None:
        """Lakes loaded when load_lakes=True and file exists."""
        lake_path = tmp_path / "lakes.dat"
        lake_path.write_text("fake")
        mock_pp_config.lakes_file = lake_path

        mock_lake_reader = MagicMock()
        mock_lake_reader.read_lake_definitions.return_value = {
            1: MagicMock(id=1),
        }

        patches = self._patch_preprocessor(mock_pp_config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch("pyiwfm.io.lakes.LakeReader", return_value=mock_lake_reader),
            patch("pyiwfm.components.lake.AppLake") as mock_app_lake_cls,
        ):
            mock_lake_inst = MagicMock()
            mock_app_lake_cls.return_value = mock_lake_inst
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")
        assert model.lakes is mock_lake_inst

    def test_lake_loading_exception_stores_error(
        self, tmp_path: Path, mock_pp_config: MagicMock
    ) -> None:
        """Lake loading exception stored in metadata."""
        lake_path = tmp_path / "lakes.dat"
        lake_path.write_text("fake")
        mock_pp_config.lakes_file = lake_path

        patches = self._patch_preprocessor(mock_pp_config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "pyiwfm.io.lakes.LakeReader",
                side_effect=ValueError("lake parse fail"),
            ),
        ):
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")
        assert "lakes_load_error" in model.metadata

    def test_load_streams_false_skips(self, tmp_path: Path, mock_pp_config: MagicMock) -> None:
        """When load_streams=False, streams are not loaded even if file exists."""
        streams_path = tmp_path / "streams.dat"
        streams_path.write_text("fake")
        mock_pp_config.streams_file = streams_path

        patches = self._patch_preprocessor(mock_pp_config)
        with patches[0], patches[1], patches[2], patches[3]:
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in", load_streams=False)
        assert model.streams is None

    def test_subregions_loaded_when_file_exists(
        self, tmp_path: Path, mock_pp_config: MagicMock
    ) -> None:
        """Subregions file is read when it exists."""
        sr_path = tmp_path / "subregions.dat"
        sr_path.write_text("fake")
        mock_pp_config.subregions_file = sr_path

        mock_subregions = {1: MagicMock(id=1, name="SR1")}

        patches = self._patch_preprocessor(mock_pp_config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "pyiwfm.io.preprocessor.read_subregions_file",
                return_value=mock_subregions,
            ) as mock_read_sr,
        ):
            IWFMModel.from_preprocessor(tmp_path / "pp.in")
        mock_read_sr.assert_called_once_with(sr_path)

    def test_stratigraphy_loaded_when_file_exists(
        self, tmp_path: Path, mock_pp_config: MagicMock
    ) -> None:
        """Stratigraphy read when file exists."""
        strat_path = tmp_path / "strat.dat"
        strat_path.write_text("fake")
        mock_pp_config.stratigraphy_file = strat_path
        mock_strat = MagicMock()

        patches = self._patch_preprocessor(mock_pp_config)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patch(
                "pyiwfm.io.ascii.read_stratigraphy",
                return_value=mock_strat,
            ),
        ):
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")
        assert model.stratigraphy is mock_strat


# ===========================================================================
# 4. IWFMModel.from_simulation_with_preprocessor() tests
# ===========================================================================


def _base_sim_setup(
    tmp_path: Path,
    sim_config_overrides: dict[str, Any] | None = None,
) -> tuple[Path, Path, MagicMock]:
    """Set up simulation and preprocessor files and return a sim_config mock."""
    sim_file = tmp_path / "Simulation" / "Simulation.in"
    sim_file.parent.mkdir(parents=True, exist_ok=True)
    sim_file.write_text("fake")
    pp_file = tmp_path / "Preprocessor" / "Preprocessor.in"
    pp_file.parent.mkdir(parents=True, exist_ok=True)
    pp_file.write_text("fake")

    sim_config = _make_sim_config(sim_file.parent)
    if sim_config_overrides:
        for k, v in sim_config_overrides.items():
            setattr(sim_config, k, v)
    return sim_file, pp_file, sim_config


class TestFromSimWithPP_Metadata:
    """Test simulation metadata populated by from_simulation_with_preprocessor."""

    def test_metadata_keys_populated(self, tmp_path: Path) -> None:
        """Core metadata keys (start_date, end_date, etc.) are populated."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata["start_date"] == "2000-01-01T00:00:00"
        assert model.metadata["end_date"] == "2000-12-31T00:00:00"
        assert model.metadata["time_step_length"] == 1
        assert model.metadata["matrix_solver"] == 2
        assert model.metadata["convergence_tolerance"] == 1e-6

    def test_supply_adjust_file_stored(self, tmp_path: Path) -> None:
        """supply_adjust_file path is stored in metadata when present."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        sim_config.supply_adjust_file = "supply.dat"

        # Create the file so sa_path.exists() passes
        sa_path = sim_file.parent / "supply.dat"
        sa_path.write_text("fake")

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.supply_adjust.read_supply_adjustment",
                return_value=MagicMock(),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "supply_adjust_file" in model.metadata
        assert model.supply_adjustment is not None

    def test_supply_adjust_file_load_error(self, tmp_path: Path) -> None:
        """supply_adjust load error is logged but does not raise."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        sim_config.supply_adjust_file = "supply_bad.dat"

        sa_path = sim_file.parent / "supply_bad.dat"
        sa_path.write_text("fake")

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.supply_adjust.read_supply_adjustment",
                side_effect=ValueError("bad file"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.supply_adjustment is None

    def test_optional_files_stored(self, tmp_path: Path) -> None:
        """Precipitation, ET, and irrigation files stored when present."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        sim_config.precipitation_file = "precip.dat"
        sim_config.et_file = "et.dat"
        sim_config.irrigation_fractions_file = "irig.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "precipitation_file" in model.metadata
        assert "et_file" in model.metadata
        assert "irrigation_fractions_file" in model.metadata

    def test_binary_preprocessor_file_stored(self, tmp_path: Path) -> None:
        """binary_preprocessor_file stored when present."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        sim_config.binary_preprocessor_file = "pp.bin"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "binary_preprocessor_file" in model.metadata


class TestFromSimWithPP_Groundwater:
    """Tests for groundwater loading in from_simulation_with_preprocessor."""

    def _setup_gw(
        self,
        tmp_path: Path,
        gw_config_overrides: dict[str, Any] | None = None,
    ) -> tuple[Path, Path, MagicMock, MagicMock]:
        """Setup with a GW main file that exists."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        gw_path = sim_file.parent / "gw.dat"
        gw_path.write_text("fake")
        sim_config.groundwater_file = "gw.dat"

        mock_mesh = _make_mock_mesh()
        mock_model = IWFMModel(name="Test", mesh=mock_mesh)
        mock_model.stratigraphy = MagicMock(n_layers=2)

        return sim_file, pp_file, sim_config, mock_model

    def test_gw_main_file_reader_success(self, tmp_path: Path) -> None:
        """GWMainFileReader successfully reads the GW main file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = "gw_budget.hdf"
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = "gw_heads.hdf"
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.groundwater is not None
        assert model.metadata.get("gw_version") == "4.0"
        assert model.metadata.get("gw_budget_file") == "gw_budget.hdf"

    def test_gw_boundary_conditions_loaded(self, tmp_path: Path) -> None:
        """Boundary conditions loaded from BC sub-file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        bc_path = sim_file.parent / "bc.dat"
        bc_path.write_text("fake")

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = bc_path
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        bc_config = MagicMock()
        bc_config.n_specified_flow = 1
        bc_config.n_specified_head = 2
        bc_config.n_general_head = 1
        bc_config.n_constrained_gh = 0
        bc_config.n_bc_output_nodes = 0
        bc_config.ts_data_file = None
        bc_config.specified_head_bcs = [
            MagicMock(node_id=1, head_value=100.0, layer=1),
            MagicMock(node_id=2, head_value=95.0, layer=1),
        ]
        bc_config.specified_flow_bcs = [
            MagicMock(node_id=3, base_flow=-10.0, layer=1),
        ]
        bc_config.general_head_bcs = [
            MagicMock(node_id=4, external_head=90.0, layer=1, conductance=0.5),
        ]

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
            patch(
                "pyiwfm.io.gw_boundary.GWBoundaryReader.read",
                return_value=bc_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("gw_n_specified_head_bc") == 2
        assert model.metadata.get("gw_n_specified_flow_bc") == 1
        assert model.metadata.get("gw_n_general_head_bc") == 1

    def test_gw_pumping_loaded(self, tmp_path: Path) -> None:
        """Pumping loaded from sub-file and well specs converted."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        pump_path = sim_file.parent / "pump.dat"
        pump_path.write_text("fake")

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = pump_path
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        pump_config = MagicMock()
        pump_config.n_wells = 2
        pump_config.n_elem_pumping = 0
        pump_config.ts_data_file = None
        pump_config.well_specs = [
            MagicMock(id=1, x=10.0, y=20.0, perf_top=50.0, perf_bottom=10.0),
            MagicMock(id=2, x=30.0, y=40.0, perf_top=60.0, perf_bottom=20.0),
        ]

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
            patch(
                "pyiwfm.io.gw_pumping.PumpingReader.read",
                return_value=pump_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("gw_n_wells") == 2
        assert model.groundwater is not None

    def test_gw_pumping_fallback_to_groundwater_reader(self, tmp_path: Path) -> None:
        """When PumpingReader fails, falls back to GroundwaterReader."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        pump_path = sim_file.parent / "pump.dat"
        pump_path.write_text("fake")

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = pump_path
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        mock_gw_reader = MagicMock()
        mock_gw_reader.read_wells.return_value = {1: MagicMock(id=1)}

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
            patch(
                "pyiwfm.io.gw_pumping.PumpingReader.read",
                side_effect=ValueError("bad pump file"),
            ),
            patch(
                "pyiwfm.io.groundwater.GroundwaterReader",
                return_value=mock_gw_reader,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.groundwater is not None

    def test_gw_tile_drains_loaded(self, tmp_path: Path) -> None:
        """Tile drains loaded with conversion factors."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        td_path = sim_file.parent / "td.dat"
        td_path.write_text("fake")

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = td_path
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        td_config = MagicMock()
        td_config.n_drains = 1
        td_config.n_sub_irrigation = 0
        td_config.drain_height_factor = 1.0
        td_config.drain_conductance_factor = 2.0
        td_config.drain_time_unit = "1DAY"
        td_config.subirig_height_factor = 1.0
        td_config.subirig_conductance_factor = 1.0
        td_config.subirig_time_unit = "1MON"
        td_config.tile_drains = [
            MagicMock(id=1, gw_node=5, elevation=100.0, conductance=0.5, dest_type=2, dest_id=10),
        ]

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
            patch(
                "pyiwfm.io.gw_tiledrain.TileDrainReader.read",
                return_value=td_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("gw_n_tile_drains") == 1
        assert model.groundwater.td_cond_factor == 2.0

    def test_gw_subsidence_loaded(self, tmp_path: Path) -> None:
        """Subsidence loaded from sub-file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        subs_path = sim_file.parent / "subs.dat"
        subs_path.write_text("fake")

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = subs_path
        gw_config.aquifer_params = None
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        subs_config = MagicMock()
        subs_config.version = "1.0"
        subs_config.node_params = [MagicMock(), MagicMock()]

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
            patch(
                "pyiwfm.io.gw_subsidence.SubsidenceReader.read",
                return_value=subs_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("gw_subsidence_version") == "1.0"
        assert model.metadata.get("gw_subsidence_n_nodes") == 2

    def test_aquifer_params_inline_loaded(self, tmp_path: Path) -> None:
        """Inline aquifer parameters loaded and set on GW component."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        aq_params = AquiferParameters(
            n_nodes=4,
            n_layers=2,
            kh=np.ones((4, 2)),
        )

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = aq_params
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("gw_aquifer_params_loaded") is True
        assert model.metadata.get("gw_aquifer_n_nodes") == 4

    def test_aquifer_params_value_error_fallback(self, tmp_path: Path) -> None:
        """When set_aquifer_parameters raises ValueError (mismatch), params stored anyway."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        # Mismatched n_nodes to trigger ValueError
        aq_params = AquiferParameters(
            n_nodes=999,
            n_layers=2,
            kh=np.ones((999, 2)),
        )

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = aq_params
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("gw_aquifer_params_loaded") is True
        assert model.metadata.get("gw_aquifer_params_mismatch") is True

    def test_parametric_grids_branch(self, tmp_path: Path) -> None:
        """Parametric grids used when aquifer_params is None."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        grid_data = MagicMock()
        grid_data.node_values = MagicMock()
        grid_data.node_values.shape = (3, 2)
        grid_data.n_nodes = 3
        grid_data.node_coords = np.array([[0, 0], [200, 0], [100, 200]])
        grid_data.elements = [(0, 1, 2)]

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = [grid_data]
        gw_config.kh_anomalies = []
        gw_config.initial_heads = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
            patch(
                "pyiwfm.core.model._apply_parametric_grids",
                return_value=True,
            ) as mock_apply,
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        mock_apply.assert_called_once()
        assert model.metadata.get("gw_aquifer_params_loaded") is True

    def test_kh_anomaly_application(self, tmp_path: Path) -> None:
        """Kh anomalies applied when present and aquifer_params exist."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        aq_params = AquiferParameters(
            n_nodes=4,
            n_layers=2,
            kh=np.ones((4, 2)),
        )
        anomalies = [KhAnomalyEntry(element_id=1, kh_per_layer=[42.0, 43.0])]

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = aq_params
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = anomalies
        gw_config.initial_heads = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
            patch(
                "pyiwfm.core.model._apply_kh_anomalies",
                return_value=1,
            ) as mock_apply_kh,
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        mock_apply_kh.assert_called_once()
        assert model.metadata.get("gw_kh_anomaly_count") == 1
        assert model.metadata.get("gw_kh_anomaly_applied") == 1

    def test_initial_heads_loaded(self, tmp_path: Path) -> None:
        """Initial heads loaded from GW main file config."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        init_heads = np.full((4, 2), 100.0, dtype=np.float64)

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = init_heads

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("gw_initial_heads_loaded") is True

    def test_initial_heads_value_error_fallback(self, tmp_path: Path) -> None:
        """Shape mismatch in initial heads stored in metadata."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        # Wrong shape to trigger ValueError in set_heads
        init_heads = np.full((999, 999), 100.0, dtype=np.float64)

        gw_config = MagicMock()
        gw_config.hydrograph_locations = []
        gw_config.version = "4.0"
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = []
        gw_config.kh_anomalies = []
        gw_config.initial_heads = init_heads

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                return_value=gw_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "gw_initial_heads_shape" in model.metadata

    def test_gw_main_file_reader_exception_fallback(self, tmp_path: Path) -> None:
        """When GWMainFileReader raises, falls back to GroundwaterReader."""
        sim_file, pp_file, sim_config, mock_model = self._setup_gw(tmp_path)

        mock_gw_reader = MagicMock()
        mock_gw_reader.read_wells.return_value = {1: MagicMock(id=1)}

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.groundwater.GWMainFileReader.read",
                side_effect=ValueError("bad gw main file"),
            ),
            patch(
                "pyiwfm.io.groundwater.GroundwaterReader",
                return_value=mock_gw_reader,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.groundwater is not None
        mock_gw_reader.read_wells.assert_called_once()


class TestFromSimWithPP_Streams:
    """Tests for stream loading in from_simulation_with_preprocessor."""

    def _setup_stream(self, tmp_path: Path) -> tuple[Path, Path, MagicMock, MagicMock]:
        """Set up with stream file that exists."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        stream_path = sim_file.parent / "stream.dat"
        stream_path.write_text("fake")
        sim_config.streams_file = "stream.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)
        mock_model.streams = None  # ensure streams is None so loading is attempted

        return sim_file, pp_file, sim_config, mock_model

    def test_stream_main_file_reader_success(self, tmp_path: Path) -> None:
        """StreamMainFileReader successfully reads stream main file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_stream(tmp_path)

        stream_config = MagicMock()
        stream_config.version = "5.0"
        stream_config.hydrograph_count = 3
        stream_config.hydrograph_output_type = 0
        stream_config.budget_output_file = "stream_budget.hdf"
        stream_config.diversion_budget_file = None
        stream_config.hydrograph_output_file = "stream_hydro.out"
        stream_config.hydrograph_specs = [(1, "Node1"), (2, "Node2")]
        stream_config.inflow_file = None
        stream_config.diversion_spec_file = None
        stream_config.bypass_spec_file = None
        stream_config.diversion_file = None
        stream_config.bed_params = []
        stream_config.interaction_type = None
        stream_config.evap_area_file = None
        stream_config.evap_node_specs = None
        stream_config.cross_section_data = None
        stream_config.initial_conditions = None
        stream_config.node_budget_count = 0
        stream_config.node_budget_ids = []
        stream_config.node_budget_output_file = None
        stream_config.final_flow_file = None
        stream_config.conductivity_factor = 1.0
        stream_config.conductivity_time_unit = "1DAY"
        stream_config.length_factor = 1.0

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None
        assert model.metadata.get("stream_version") == "5.0"
        assert model.metadata.get("stream_hydrograph_count") == 3

    def test_diversions_loaded(self, tmp_path: Path) -> None:
        """Diversions loaded from diversion spec sub-file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_stream(tmp_path)

        div_path = sim_file.parent / "div.dat"
        div_path.write_text("fake")

        stream_config = MagicMock()
        stream_config.version = "4.0"
        stream_config.hydrograph_count = 0
        stream_config.hydrograph_output_type = 0
        stream_config.budget_output_file = None
        stream_config.diversion_budget_file = None
        stream_config.hydrograph_output_file = None
        stream_config.hydrograph_specs = []
        stream_config.inflow_file = None
        stream_config.diversion_spec_file = div_path
        stream_config.bypass_spec_file = None
        stream_config.diversion_file = None
        stream_config.bed_params = []
        stream_config.interaction_type = None
        stream_config.evap_area_file = None
        stream_config.evap_node_specs = None
        stream_config.cross_section_data = None
        stream_config.initial_conditions = None
        stream_config.node_budget_count = 0
        stream_config.node_budget_ids = []
        stream_config.node_budget_output_file = None
        stream_config.final_flow_file = None
        stream_config.conductivity_factor = 1.0
        stream_config.conductivity_time_unit = "1DAY"
        stream_config.length_factor = 1.0

        div_config = MagicMock()
        div_config.n_diversions = 2
        div_config.n_element_groups = 1
        div_config.diversions = [
            MagicMock(
                id=1,
                stream_node=5,
                dest_type=0,
                dest_id=0,
                name="Div1",
                max_diver_col=1,
                frac_max_diver=1.0,
                recv_loss_col=0,
                frac_recv_loss=0.0,
                non_recv_loss_col=0,
                frac_non_recv_loss=0.0,
                spill_col=0,
                frac_spill=0.0,
                delivery_col=0,
                frac_delivery=0.0,
                irrig_frac_col=0,
                adjustment_col=0,
            ),
        ]
        div_config.element_groups = []
        div_config.recharge_zones = []
        div_config.spill_zones = []
        div_config.has_spills = False

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_diversion.DiversionSpecReader.read",
                return_value=div_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("stream_n_diversions") == 2

    def test_bypasses_loaded(self, tmp_path: Path) -> None:
        """Bypasses loaded from bypass spec sub-file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_stream(tmp_path)

        byp_path = sim_file.parent / "bypass.dat"
        byp_path.write_text("fake")

        stream_config = MagicMock()
        stream_config.version = "4.0"
        stream_config.hydrograph_count = 0
        stream_config.hydrograph_output_type = 0
        stream_config.budget_output_file = None
        stream_config.diversion_budget_file = None
        stream_config.hydrograph_output_file = None
        stream_config.hydrograph_specs = []
        stream_config.inflow_file = None
        stream_config.diversion_spec_file = None
        stream_config.bypass_spec_file = byp_path
        stream_config.diversion_file = None
        stream_config.bed_params = []
        stream_config.interaction_type = None
        stream_config.evap_area_file = None
        stream_config.evap_node_specs = None
        stream_config.cross_section_data = None
        stream_config.initial_conditions = None
        stream_config.node_budget_count = 0
        stream_config.node_budget_ids = []
        stream_config.node_budget_output_file = None
        stream_config.final_flow_file = None
        stream_config.conductivity_factor = 1.0
        stream_config.conductivity_time_unit = "1DAY"
        stream_config.length_factor = 1.0

        byp_config = MagicMock()
        byp_config.n_bypasses = 1
        byp_config.flow_factor = 1.0
        byp_config.flow_time_unit = "1DAY"
        byp_config.bypass_factor = 1.0
        byp_config.bypass_time_unit = "1DAY"
        byp_config.bypasses = [
            MagicMock(
                id=1,
                export_stream_node=3,
                dest_id=10,
                dest_type=1,
                name="Bypass1",
                rating_table_col=0,
                frac_recoverable=0.5,
                frac_non_recoverable=0.1,
                inline_rating=None,
            ),
        ]
        byp_config.seepage_zones = []

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_bypass.BypassSpecReader.read",
                return_value=byp_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("stream_n_bypasses") == 1

    def test_bed_params_populated(self, tmp_path: Path) -> None:
        """Bed parameters populated on stream nodes from main file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_stream(tmp_path)

        bp = MagicMock()
        bp.node_id = 1
        bp.conductivity = 5.0
        bp.bed_thickness = 1.0
        bp.wetted_perimeter = 10.0
        bp.gw_node = 42

        stream_config = MagicMock()
        stream_config.version = "4.0"
        stream_config.hydrograph_count = 0
        stream_config.hydrograph_output_type = 0
        stream_config.budget_output_file = None
        stream_config.diversion_budget_file = None
        stream_config.hydrograph_output_file = None
        stream_config.hydrograph_specs = []
        stream_config.inflow_file = None
        stream_config.diversion_spec_file = None
        stream_config.bypass_spec_file = None
        stream_config.diversion_file = None
        stream_config.bed_params = [bp]
        stream_config.interaction_type = None
        stream_config.evap_area_file = None
        stream_config.evap_node_specs = None
        stream_config.cross_section_data = None
        stream_config.initial_conditions = None
        stream_config.node_budget_count = 0
        stream_config.node_budget_ids = []
        stream_config.node_budget_output_file = None
        stream_config.final_flow_file = None
        stream_config.conductivity_factor = 1.0
        stream_config.conductivity_time_unit = "1DAY"
        stream_config.length_factor = 1.0

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None

    def test_stream_loading_exception_fallback(self, tmp_path: Path) -> None:
        """Stream main file reader failure falls back to StreamReader."""
        sim_file, pp_file, sim_config, mock_model = self._setup_stream(tmp_path)

        mock_stream_reader = MagicMock()
        # reach_id needs an explicit int value — _build_reaches_from_node_reach_ids
        # compares it to 0; without this the auto-MagicMock attribute would TypeError.
        mock_stream_reader.read_stream_nodes.return_value = {
            1: MagicMock(id=1, reach_id=0),
        }

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                side_effect=ValueError("bad stream main"),
            ),
            patch(
                "pyiwfm.io.streams.StreamReader",
                return_value=mock_stream_reader,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None
        mock_stream_reader.read_stream_nodes.assert_called_once()

    def test_stream_inflow_loaded(self, tmp_path: Path) -> None:
        """Inflow info loaded from inflow sub-file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_stream(tmp_path)

        inflow_path = sim_file.parent / "inflow.dat"
        inflow_path.write_text("fake")

        stream_config = MagicMock()
        stream_config.version = "4.0"
        stream_config.hydrograph_count = 0
        stream_config.hydrograph_output_type = 0
        stream_config.budget_output_file = None
        stream_config.diversion_budget_file = None
        stream_config.hydrograph_output_file = None
        stream_config.hydrograph_specs = []
        stream_config.inflow_file = inflow_path
        stream_config.diversion_spec_file = None
        stream_config.bypass_spec_file = None
        stream_config.diversion_file = None
        stream_config.bed_params = []
        stream_config.interaction_type = None
        stream_config.evap_area_file = None
        stream_config.evap_node_specs = None
        stream_config.cross_section_data = None
        stream_config.initial_conditions = None
        stream_config.node_budget_count = 0
        stream_config.node_budget_ids = []
        stream_config.node_budget_output_file = None
        stream_config.final_flow_file = None
        stream_config.conductivity_factor = 1.0
        stream_config.conductivity_time_unit = "1DAY"
        stream_config.length_factor = 1.0

        inflow_config = MagicMock()
        inflow_config.n_inflows = 5
        inflow_config.inflow_nodes = [1, 2, 3, 4, 5]

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_inflow.InflowReader.read",
                return_value=inflow_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("stream_n_inflows") == 5

    def test_stream_hydrograph_specs_stored(self, tmp_path: Path) -> None:
        """Hydrograph specs stored in metadata as list of dicts."""
        sim_file, pp_file, sim_config, mock_model = self._setup_stream(tmp_path)

        stream_config = MagicMock()
        stream_config.version = "4.0"
        stream_config.hydrograph_count = 2
        stream_config.hydrograph_output_type = 0
        stream_config.budget_output_file = None
        stream_config.diversion_budget_file = None
        stream_config.hydrograph_output_file = "hydro.out"
        stream_config.hydrograph_specs = [(1, "Node_A"), (2, "Node_B")]
        stream_config.inflow_file = None
        stream_config.diversion_spec_file = None
        stream_config.bypass_spec_file = None
        stream_config.diversion_file = None
        stream_config.bed_params = []
        stream_config.interaction_type = None
        stream_config.evap_area_file = None
        stream_config.evap_node_specs = None
        stream_config.cross_section_data = None
        stream_config.initial_conditions = None
        stream_config.node_budget_count = 0
        stream_config.node_budget_ids = []
        stream_config.node_budget_output_file = None
        stream_config.final_flow_file = None
        stream_config.conductivity_factor = 1.0
        stream_config.conductivity_time_unit = "1DAY"
        stream_config.length_factor = 1.0

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        specs = model.metadata.get("stream_hydrograph_specs")
        assert specs is not None
        assert len(specs) == 2
        assert specs[0]["node_id"] == 1
        assert specs[0]["name"] == "Node_A"


class TestFromSimWithPP_Lakes:
    """Tests for lake loading in from_simulation_with_preprocessor."""

    def _setup_lake(self, tmp_path: Path) -> tuple[Path, Path, MagicMock, MagicMock]:
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        lake_path = sim_file.parent / "lake.dat"
        lake_path.write_text("fake")
        sim_config.lakes_file = "lake.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)
        mock_model.lakes = None

        return sim_file, pp_file, sim_config, mock_model

    def test_lake_main_file_reader_success(self, tmp_path: Path) -> None:
        """LakeMainFileReader successfully reads lake main file."""
        sim_file, pp_file, sim_config, mock_model = self._setup_lake(tmp_path)

        lake_config = MagicMock()
        lake_config.version = "4.0"
        lake_config.lake_params = [MagicMock(lake_id=1, name="Lake1")]
        lake_config.max_elev_file = None
        lake_config.budget_output_file = "lake_budget.hdf"
        lake_config.conductance_factor = 1.0
        lake_config.depth_factor = 1.0
        lake_config.outflow_ratings = []

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.lakes.LakeMainFileReader.read",
                return_value=lake_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.lakes is not None
        assert model.metadata.get("lake_version") == "4.0"
        assert model.metadata.get("lake_n_lakes") == 1

    def test_lake_fallback_to_lake_reader(self, tmp_path: Path) -> None:
        """When LakeMainFileReader fails, falls back to LakeReader."""
        sim_file, pp_file, sim_config, mock_model = self._setup_lake(tmp_path)

        mock_lake_reader = MagicMock()
        mock_lake_reader.read_lake_definitions.return_value = {
            1: MagicMock(id=1),
        }

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.lakes.LakeMainFileReader.read",
                side_effect=ValueError("bad lake main"),
            ),
            patch(
                "pyiwfm.io.lakes.LakeReader",
                return_value=mock_lake_reader,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.lakes is not None
        mock_lake_reader.read_lake_definitions.assert_called_once()

    def test_lake_complete_failure_stores_error(self, tmp_path: Path) -> None:
        """Complete lake loading failure stores error in metadata."""
        sim_file, pp_file, sim_config, mock_model = self._setup_lake(tmp_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.components.lake.AppLake",
                side_effect=ValueError("cannot create AppLake"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "lakes_load_error" in model.metadata


class TestFromSimWithPP_RootZone:
    """Tests for root zone loading in from_simulation_with_preprocessor."""

    def _setup_rz(self, tmp_path: Path) -> tuple[Path, Path, MagicMock, MagicMock]:
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        rz_path = sim_file.parent / "rootzone.dat"
        rz_path.write_text("fake")
        sim_config.rootzone_file = "rootzone.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        return sim_file, pp_file, sim_config, mock_model

    def test_rootzone_main_file_reader_success(self, tmp_path: Path) -> None:
        """RootZoneMainFileReader reads root zone main file with soil params."""
        sim_file, pp_file, sim_config, mock_model = self._setup_rz(tmp_path)

        rz_config = MagicMock()
        rz_config.version = "4.2"
        rz_config.gw_uptake_enabled = True
        rz_config.nonponded_crop_file = None
        rz_config.ponded_crop_file = None
        rz_config.urban_file = None
        rz_config.native_veg_file = None
        rz_config.return_flow_file = None
        rz_config.reuse_file = None
        rz_config.irrigation_period_file = None
        rz_config.ag_water_demand_file = None
        rz_config.surface_flow_dest_file = None
        rz_config.lwu_budget_file = None
        rz_config.rz_budget_file = None
        rz_config.k_factor = 1.0
        rz_config.k_exdth_factor = 1.0
        rz_config.element_soil_params = []  # empty for simplicity

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.rootzone.RootZoneMainFileReader.read",
                return_value=rz_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.rootzone is not None
        assert model.metadata.get("rootzone_version") == "4.2"
        assert model.metadata.get("rootzone_gw_uptake") is True

    def test_rootzone_fallback_to_rootzone_reader(self, tmp_path: Path) -> None:
        """When RootZoneMainFileReader fails, falls back to RootZoneReader."""
        sim_file, pp_file, sim_config, mock_model = self._setup_rz(tmp_path)

        mock_rz_reader = MagicMock()
        mock_rz_reader.read_crop_types.return_value = {1: MagicMock(id=1)}

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.rootzone.RootZoneMainFileReader.read",
                side_effect=ValueError("bad rz main"),
            ),
            patch(
                "pyiwfm.io.rootzone.RootZoneReader",
                return_value=mock_rz_reader,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.rootzone is not None
        mock_rz_reader.read_crop_types.assert_called_once()

    def test_rootzone_complete_failure_stores_error(self, tmp_path: Path) -> None:
        """Complete root zone failure stores error in metadata."""
        sim_file, pp_file, sim_config, mock_model = self._setup_rz(tmp_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.components.rootzone.RootZone",
                side_effect=ValueError("cannot create RootZone"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "rootzone_load_error" in model.metadata


class TestFromSimWithPP_SmallWatershedUnsatZone:
    """Tests for small watershed and unsaturated zone loading."""

    def test_small_watershed_loaded(self, tmp_path: Path) -> None:
        """Small watershed loaded from sub-file when n_watersheds > 0."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        sw_path = sim_file.parent / "swshed.dat"
        sw_path.write_text("fake")
        sim_config.small_watershed_file = "swshed.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        sw_config = MagicMock()
        sw_config.version = "1.0"
        sw_config.n_watersheds = 3
        sw_config.budget_output_file = "sw_budget.hdf"

        mock_sw_component = MagicMock()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.small_watershed.SmallWatershedMainReader.read",
                return_value=sw_config,
            ),
            patch(
                "pyiwfm.components.small_watershed.AppSmallWatershed.from_config",
                return_value=mock_sw_component,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.small_watersheds is mock_sw_component
        assert model.metadata.get("small_watershed_version") == "1.0"
        assert model.metadata.get("small_watershed_count") == 3

    def test_unsaturated_zone_loaded(self, tmp_path: Path) -> None:
        """Unsaturated zone loaded from sub-file when n_layers > 0."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        uz_path = sim_file.parent / "unsatzone.dat"
        uz_path.write_text("fake")
        sim_config.unsaturated_zone_file = "unsatzone.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        uz_config = MagicMock()
        uz_config.version = "2.0"
        uz_config.n_layers = 3
        uz_config.budget_file = "uz_budget.hdf"

        mock_uz_component = MagicMock()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.unsaturated_zone.UnsatZoneMainReader.read",
                return_value=uz_config,
            ),
            patch(
                "pyiwfm.components.unsaturated_zone.AppUnsatZone.from_config",
                return_value=mock_uz_component,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.unsaturated_zone is mock_uz_component
        assert model.metadata.get("unsat_zone_version") == "2.0"
        assert model.metadata.get("unsat_zone_n_layers") == 3


# ===========================================================================
# 5. summary() and validate_components() tests
# ===========================================================================


class TestSummaryAndValidation:
    """Tests for summary() and validate_components() with all component types."""

    def test_summary_with_small_watersheds(self) -> None:
        """Summary includes small watershed line when component is present."""
        sw = MagicMock()
        sw.n_watersheds = 5
        model = IWFMModel(name="Test", small_watersheds=sw)
        text = model.summary()
        assert "Watersheds: 5" in text

    def test_summary_with_unsaturated_zone(self) -> None:
        """Summary includes unsaturated zone line when component is present."""
        uz = MagicMock()
        uz.n_layers = 3
        uz.n_elements = 100
        model = IWFMModel(name="Test", unsaturated_zone=uz)
        text = model.summary()
        assert "Layers: 3" in text
        assert "Elements: 100" in text

    def test_summary_without_optional_components(self) -> None:
        """Summary shows 'Not loaded' for absent optional components."""
        model = IWFMModel(name="Test")
        text = model.summary()
        assert "Not loaded" in text

    def test_summary_with_all_components(self) -> None:
        """Summary includes all component sections when all are populated."""
        mesh = MagicMock()
        mesh.n_nodes = 10
        mesh.n_elements = 5
        mesh.n_subregions = 2

        strat = MagicMock()
        strat.n_layers = 3

        gw = MagicMock()
        gw.n_wells = 2
        gw.n_hydrograph_locations = 1
        gw.n_boundary_conditions = 3
        gw.n_tile_drains = 1
        gw.aquifer_params = MagicMock()

        streams = MagicMock()
        streams.n_nodes = 20
        streams.n_reaches = 5
        streams.n_diversions = 2
        streams.n_bypasses = 1

        lakes = MagicMock()
        lakes.n_lakes = 1
        lakes.n_lake_elements = 4

        rz = MagicMock()
        rz.n_crop_types = 7
        rz.element_landuse = {1: "ag", 2: "urban"}
        rz.soil_params = {1: MagicMock()}

        sw = MagicMock()
        sw.n_watersheds = 2

        uz = MagicMock()
        uz.n_layers = 3
        uz.n_elements = 50

        model = IWFMModel(
            name="FullModel",
            mesh=mesh,
            stratigraphy=strat,
            groundwater=gw,
            streams=streams,
            lakes=lakes,
            rootzone=rz,
            small_watersheds=sw,
            unsaturated_zone=uz,
            metadata={"source": "test"},
        )
        text = model.summary()
        assert "FullModel" in text
        assert "Wells: 2" in text
        assert "Stream Nodes: 20" in text
        assert "Lakes: 1" in text
        assert "Crop Types: 7" in text
        assert "Watersheds: 2" in text

    def test_validate_components_with_small_watersheds(self) -> None:
        """validate_components calls validate on small_watersheds."""
        sw = MagicMock()
        sw.validate.side_effect = ValueError("sw invalid")
        model = IWFMModel(name="Test", small_watersheds=sw)
        warnings = model.validate_components()
        assert any("Small watershed" in w for w in warnings)

    def test_validate_components_with_unsaturated_zone(self) -> None:
        """validate_components calls validate on unsaturated_zone."""
        uz = MagicMock()
        uz.validate.side_effect = ValueError("uz invalid")
        model = IWFMModel(name="Test", unsaturated_zone=uz)
        warnings = model.validate_components()
        assert any("Unsaturated zone" in w for w in warnings)

    def test_validate_components_no_warnings_when_all_valid(self) -> None:
        """No warnings when all components validate successfully."""
        gw = MagicMock()
        gw.validate.return_value = None
        streams = MagicMock()
        streams.validate.return_value = None
        lakes = MagicMock()
        lakes.validate.return_value = None
        rz = MagicMock()
        rz.validate.return_value = None
        sw = MagicMock()
        sw.validate.return_value = None
        uz = MagicMock()
        uz.validate.return_value = None

        model = IWFMModel(
            name="Test",
            groundwater=gw,
            streams=streams,
            lakes=lakes,
            rootzone=rz,
            small_watersheds=sw,
            unsaturated_zone=uz,
        )
        warnings = model.validate_components()
        assert warnings == []

    def test_validate_components_multiple_failures(self) -> None:
        """Multiple component validation failures all reported."""
        gw = MagicMock()
        gw.validate.side_effect = ValueError("gw fail")
        streams = MagicMock()
        streams.validate.side_effect = ValueError("stream fail")
        model = IWFMModel(name="Test", groundwater=gw, streams=streams)
        warnings = model.validate_components()
        assert len(warnings) == 2
        assert any("Groundwater" in w for w in warnings)
        assert any("Stream" in w for w in warnings)

    def test_validate_components_empty_model(self) -> None:
        """Empty model (no components) returns no warnings."""
        model = IWFMModel(name="Empty")
        warnings = model.validate_components()
        assert warnings == []


# ===========================================================================
# 6. Deep stream loading branches
# ===========================================================================


def _make_base_stream_config(**overrides: Any) -> MagicMock:
    """Return a fully-populated stream_config mock with sensible defaults."""
    cfg = MagicMock()
    cfg.version = "4.0"
    cfg.hydrograph_count = 0
    cfg.hydrograph_output_type = 0
    cfg.budget_output_file = None
    cfg.diversion_budget_file = None
    cfg.hydrograph_output_file = None
    cfg.hydrograph_specs = []
    cfg.inflow_file = None
    cfg.diversion_spec_file = None
    cfg.bypass_spec_file = None
    cfg.diversion_file = None
    cfg.bed_params = []
    cfg.interaction_type = None
    cfg.evap_area_file = None
    cfg.evap_node_specs = None
    cfg.cross_section_data = None
    cfg.initial_conditions = None
    cfg.node_budget_count = 0
    cfg.node_budget_ids = []
    cfg.node_budget_output_file = None
    cfg.final_flow_file = None
    cfg.conductivity_factor = 1.0
    cfg.conductivity_time_unit = "1DAY"
    cfg.length_factor = 1.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestStreamDeepLoading:
    """Cover uncovered deep-loading branches within stream component loading."""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path, MagicMock, MagicMock]:
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        stream_path = sim_file.parent / "stream.dat"
        stream_path.write_text("fake")
        sim_config.streams_file = "stream.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)
        mock_model.streams = None
        return sim_file, pp_file, sim_config, mock_model

    def _run(
        self,
        sim_file: Path,
        pp_file: Path,
        sim_config: MagicMock,
        mock_model: MagicMock,
        stream_config: MagicMock,
        extra_patches: dict[str, Any] | None = None,
    ) -> IWFMModel:
        patches = {
            "pyiwfm.io.simulation.SimulationReader.read": sim_config,
            "pyiwfm.io.preprocessor._resolve_path": lambda base, p: Path(base) / p,
            "pyiwfm.io.streams.StreamMainFileReader.read": stream_config,
        }
        if extra_patches:
            patches.update(extra_patches)

        ctx_managers = [patch.object(IWFMModel, "from_preprocessor", return_value=mock_model)]
        for target, val in patches.items():
            if callable(val) and not isinstance(val, MagicMock):
                ctx_managers.append(patch(target, side_effect=val))
            else:
                ctx_managers.append(patch(target, return_value=val))

        # Stack all context managers
        with ctx_managers[0]:
            with ctx_managers[1]:
                with ctx_managers[2]:
                    with ctx_managers[3]:
                        if len(ctx_managers) > 4:
                            with ctx_managers[4]:
                                return IWFMModel.from_simulation_with_preprocessor(
                                    sim_file, pp_file
                                )
                        return IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

    def test_diversion_budget_file_metadata(self, tmp_path: Path) -> None:
        """Diversion budget file path stored in metadata (line 847)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)
        stream_config = _make_base_stream_config(
            diversion_budget_file="div_budget.hdf",
        )
        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("stream_diversion_budget_file") == "div_budget.hdf"

    def test_diversion_ts_file_source(self, tmp_path: Path) -> None:
        """Diversion time series file stored in source_files (line 870)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)
        stream_config = _make_base_stream_config(
            diversion_file=Path("div_ts.dat"),
        )
        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.source_files.get("stream_diversion_ts") == Path("div_ts.dat")

    def test_bypass_with_rating_table_flow_factor_not_one(self, tmp_path: Path) -> None:
        """Bypass rating table flows undone by flow_factor != 1.0 (lines 938-943)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        byp_path = sim_file.parent / "bypass.dat"
        byp_path.write_text("fake")

        inline_rating = MagicMock()
        inline_rating.flows = np.array([10.0, 20.0])
        inline_rating.fractions = np.array([0.5, 0.8])

        byp_spec = MagicMock()
        byp_spec.id = 1
        byp_spec.export_stream_node = 3
        byp_spec.dest_id = 10
        byp_spec.dest_type = 1
        byp_spec.name = "Bypass_RT"
        byp_spec.rating_table_col = 0
        byp_spec.frac_recoverable = 0.5
        byp_spec.frac_non_recoverable = 0.1
        byp_spec.inline_rating = inline_rating

        byp_config = MagicMock()
        byp_config.n_bypasses = 1
        byp_config.flow_factor = 2.0  # non-unity flow factor
        byp_config.flow_time_unit = "1DAY"
        byp_config.bypass_factor = 1.0
        byp_config.bypass_time_unit = "1DAY"
        byp_config.bypasses = [byp_spec]
        byp_config.seepage_zones = []

        stream_config = _make_base_stream_config(bypass_spec_file=byp_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_bypass.BypassSpecReader.read",
                return_value=byp_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("stream_n_bypasses") == 1
        assert model.streams is not None

    def test_bypass_with_rating_table_flow_factor_one(self, tmp_path: Path) -> None:
        """Bypass rating table with flow_factor == 1.0 (else branch lines 941-943)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        byp_path = sim_file.parent / "bypass.dat"
        byp_path.write_text("fake")

        inline_rating = MagicMock()
        inline_rating.flows = np.array([5.0, 15.0])
        inline_rating.fractions = np.array([0.3, 0.7])

        byp_spec = MagicMock()
        byp_spec.id = 1
        byp_spec.export_stream_node = 3
        byp_spec.dest_id = 10
        byp_spec.dest_type = 1
        byp_spec.name = "Bypass_FF1"
        byp_spec.rating_table_col = 0
        byp_spec.frac_recoverable = 0.5
        byp_spec.frac_non_recoverable = 0.1
        byp_spec.inline_rating = inline_rating

        byp_config = MagicMock()
        byp_config.n_bypasses = 1
        byp_config.flow_factor = 1.0  # unity flow factor
        byp_config.flow_time_unit = "1DAY"
        byp_config.bypass_factor = 1.0
        byp_config.bypass_time_unit = "1DAY"
        byp_config.bypasses = [byp_spec]
        byp_config.seepage_zones = []

        stream_config = _make_base_stream_config(bypass_spec_file=byp_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_bypass.BypassSpecReader.read",
                return_value=byp_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("stream_n_bypasses") == 1

    def test_bypass_seepage_zones_mapped(self, tmp_path: Path) -> None:
        """Seepage zones mapped to bypass objects (lines 964-967)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        byp_path = sim_file.parent / "bypass.dat"
        byp_path.write_text("fake")

        byp_spec = MagicMock()
        byp_spec.id = 1
        byp_spec.export_stream_node = 3
        byp_spec.dest_id = 10
        byp_spec.dest_type = 1
        byp_spec.name = "Bypass_SZ"
        byp_spec.rating_table_col = 0
        byp_spec.frac_recoverable = 0.5
        byp_spec.frac_non_recoverable = 0.1
        byp_spec.inline_rating = None

        seepage_zone = MagicMock()
        seepage_zone.bypass_id = 1

        byp_config = MagicMock()
        byp_config.n_bypasses = 1
        byp_config.flow_factor = 1.0
        byp_config.flow_time_unit = "1DAY"
        byp_config.bypass_factor = 1.0
        byp_config.bypass_time_unit = "1DAY"
        byp_config.bypasses = [byp_spec]
        byp_config.seepage_zones = [seepage_zone]

        stream_config = _make_base_stream_config(bypass_spec_file=byp_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_bypass.BypassSpecReader.read",
                return_value=byp_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None
        # bypass 1 should exist and have the seepage zone appended
        assert 1 in model.streams.bypasses

    def test_diversion_loading_exception_pass(self, tmp_path: Path) -> None:
        """Diversion reader exception silently passes (lines 919-920)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        div_path = sim_file.parent / "div.dat"
        div_path.write_text("fake")

        stream_config = _make_base_stream_config(diversion_spec_file=div_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_diversion.DiversionSpecReader.read",
                side_effect=ValueError("bad div file"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        # Stream still loads despite diversion failure
        assert model.streams is not None
        # No diversion metadata should be set
        assert "stream_n_diversions" not in model.metadata

    def test_bypass_loading_exception_pass(self, tmp_path: Path) -> None:
        """Bypass reader exception silently passes (lines 964-967 exception)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        byp_path = sim_file.parent / "bypass.dat"
        byp_path.write_text("fake")

        stream_config = _make_base_stream_config(bypass_spec_file=byp_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_bypass.BypassSpecReader.read",
                side_effect=ValueError("bad bypass file"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None
        assert "stream_n_bypasses" not in model.metadata

    def test_inflow_loading_exception_pass(self, tmp_path: Path) -> None:
        """Inflow reader exception silently passes (lines 979-980)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        inflow_path = sim_file.parent / "inflow.dat"
        inflow_path.write_text("fake")

        stream_config = _make_base_stream_config(inflow_file=inflow_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
            patch(
                "pyiwfm.io.stream_inflow.InflowReader.read",
                side_effect=ValueError("bad inflow file"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None
        assert "stream_n_inflows" not in model.metadata

    def test_bed_params_create_new_nodes(self, tmp_path: Path) -> None:
        """Bed params create new StrmNode objects with full attributes (lines 985-996)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        bp1 = MagicMock()
        bp1.node_id = 10
        bp1.conductivity = 5.0
        bp1.bed_thickness = 1.5
        bp1.wetted_perimeter = 12.0
        bp1.gw_node = 42

        bp2 = MagicMock()
        bp2.node_id = 20
        bp2.conductivity = 3.0
        bp2.bed_thickness = 0.8
        bp2.wetted_perimeter = None  # None branch
        bp2.gw_node = 0  # gw_node <= 0 branch

        stream_config = _make_base_stream_config(bed_params=[bp1, bp2])

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None
        # Nodes should be created
        assert 10 in model.streams.nodes
        assert 20 in model.streams.nodes
        node10 = model.streams.nodes[10]
        assert node10.conductivity == 5.0
        assert node10.bed_thickness == 1.5
        assert node10.gw_node == 42

    def test_interaction_type_set(self, tmp_path: Path) -> None:
        """Interaction type set on stream (line 1003)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)
        stream_config = _make_base_stream_config(interaction_type=2)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams.interaction_type == 2

    def test_evap_area_file_and_node_specs(self, tmp_path: Path) -> None:
        """Stream evaporation area file and node specs loaded (lines 1006-1017)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)
        stream_config = _make_base_stream_config(
            evap_area_file="evap_area.dat",
            evap_node_specs=[(1, 3, 5), (2, 4, 6)],
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams.evap_area_file == "evap_area.dat"
        assert len(model.streams.evap_node_specs) == 2

    def test_cross_section_data_loaded(self, tmp_path: Path) -> None:
        """Cross-section data assigned to existing stream nodes (lines 1021-1030)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        bp = MagicMock()
        bp.node_id = 1
        bp.conductivity = 5.0
        bp.bed_thickness = 1.0
        bp.wetted_perimeter = 10.0
        bp.gw_node = 2

        cs = MagicMock()
        cs.node_id = 1
        cs.bottom_elev = 100.0
        cs.B0 = 50.0
        cs.s = 2.0
        cs.n = 0.03
        cs.max_flow_depth = 10.0

        stream_config = _make_base_stream_config(
            bed_params=[bp],
            cross_section_data=[cs],
            roughness_factor=1.0,
            cross_section_length_factor=1.0,
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None
        node = model.streams.nodes[1]
        assert node.cross_section is not None
        assert model.streams.roughness_factor == 1.0

    def test_initial_conditions_loaded(self, tmp_path: Path) -> None:
        """Initial conditions assigned to stream nodes (lines 1033-1038)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        bp = MagicMock()
        bp.node_id = 1
        bp.conductivity = 5.0
        bp.bed_thickness = 1.0
        bp.wetted_perimeter = 10.0
        bp.gw_node = 2

        ic_row = MagicMock()
        ic_row.node_id = 1
        ic_row.value = 50.0

        stream_config = _make_base_stream_config(
            bed_params=[bp],
            initial_conditions=[ic_row],
            ic_type=1,
            ic_factor=1.0,
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams is not None
        assert model.streams.nodes[1].initial_condition == 50.0
        assert model.streams.ic_type == 1
        assert model.streams.ic_factor == 1.0

    def test_budget_nodes_loaded(self, tmp_path: Path) -> None:
        """Budget node data stored on stream component (lines 1041-1047)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)
        stream_config = _make_base_stream_config(
            node_budget_count=3,
            node_budget_ids=[1, 5, 10],
            node_budget_output_file="node_budget.hdf",
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams.budget_node_count == 3
        assert model.streams.budget_node_ids == [1, 5, 10]
        assert model.streams.budget_output_file == "node_budget.hdf"

    def test_final_flow_file_stored(self, tmp_path: Path) -> None:
        """Final flow file stored on stream (line 1051)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)
        stream_config = _make_base_stream_config(
            final_flow_file="final_flows.dat",
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.streams.StreamMainFileReader.read",
                return_value=stream_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.streams.final_flow_file == "final_flows.dat"

    def test_outer_exception_stores_error(self, tmp_path: Path) -> None:
        """Complete stream loading failure stores error in metadata (lines 1064-1065)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.components.stream.AppStream",
                side_effect=ValueError("cannot create AppStream"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "streams_load_error" in model.metadata
        assert "cannot create AppStream" in model.metadata["streams_load_error"]


# ===========================================================================
# 7. Deep lake loading branches
# ===========================================================================


class TestLakeDeepLoading:
    """Cover uncovered lake loading branches."""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path, MagicMock, MagicMock]:
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        lake_path = sim_file.parent / "lake.dat"
        lake_path.write_text("fake")
        sim_config.lakes_file = "lake.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)
        mock_model.lakes = None
        return sim_file, pp_file, sim_config, mock_model

    def test_max_elev_file_stored(self, tmp_path: Path) -> None:
        """Max elevation file stored in source_files (line 1087)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        lake_config = MagicMock()
        lake_config.version = "4.0"
        lake_config.lake_params = [MagicMock(lake_id=1, name="Lake1")]
        lake_config.max_elev_file = Path("max_elev.dat")
        lake_config.budget_output_file = None
        lake_config.conductance_factor = 1.0
        lake_config.depth_factor = 1.0
        lake_config.outflow_ratings = []

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.lakes.LakeMainFileReader.read",
                return_value=lake_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.source_files.get("lake_max_elev_ts") == Path("max_elev.dat")

    def test_outflow_ratings_counted(self, tmp_path: Path) -> None:
        """Outflow ratings count stored in metadata (line 1110)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        lake_config = MagicMock()
        lake_config.version = "5.0"
        lake_config.lake_params = [MagicMock(lake_id=1, name="Lake1")]
        lake_config.max_elev_file = None
        lake_config.budget_output_file = None
        lake_config.conductance_factor = 1.0
        lake_config.depth_factor = 1.0
        lake_config.outflow_ratings = [MagicMock(), MagicMock(), MagicMock()]

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.lakes.LakeMainFileReader.read",
                return_value=lake_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("lake_n_outflow_ratings") == 3


# ===========================================================================
# 8. Deep rootzone loading branches
# ===========================================================================


class TestRootZoneDeepLoading:
    """Cover uncovered rootzone loading branches."""

    def _setup(self, tmp_path: Path) -> tuple[Path, Path, MagicMock, MagicMock]:
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        rz_path = sim_file.parent / "rootzone.dat"
        rz_path.write_text("fake")
        sim_config.rootzone_file = "rootzone.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)
        return sim_file, pp_file, sim_config, mock_model

    def _make_rz_config(
        self,
        tmp_path: Path,
        version: str = "4.2",
        soil_params: list | None = None,
        **overrides: Any,
    ) -> MagicMock:
        """Create a rootzone config mock with optional soil params."""
        cfg = MagicMock()
        cfg.version = version
        cfg.gw_uptake_enabled = True
        cfg.nonponded_crop_file = overrides.get("nonponded_crop_file", None)
        cfg.ponded_crop_file = overrides.get("ponded_crop_file", None)
        cfg.urban_file = overrides.get("urban_file", None)
        cfg.native_veg_file = overrides.get("native_veg_file", None)
        cfg.return_flow_file = overrides.get("return_flow_file", None)
        cfg.reuse_file = overrides.get("reuse_file", None)
        cfg.irrigation_period_file = overrides.get("irrigation_period_file", None)
        cfg.ag_water_demand_file = overrides.get("ag_water_demand_file", None)
        cfg.surface_flow_dest_file = overrides.get("surface_flow_dest_file", None)
        cfg.lwu_budget_file = overrides.get("lwu_budget_file", None)
        cfg.rz_budget_file = overrides.get("rz_budget_file", None)
        cfg.k_factor = overrides.get("k_factor", 1.0)
        cfg.k_exdth_factor = overrides.get("k_exdth_factor", 1.0)
        cfg.element_soil_params = soil_params if soil_params is not None else []
        return cfg

    def test_soil_params_populated_pre_v412(self, tmp_path: Path) -> None:
        """Soil params populated with pre-v4.12 surface flow destinations (lines 1200-1240)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        row = MagicMock()
        row.element_id = 1
        row.total_porosity = 0.4
        row.field_capacity = 0.25
        row.wilting_point = 0.1
        row.hydraulic_conductivity = 0.5
        row.lambda_param = 2.0
        row.kunsat_method = 1
        row.k_ponded = 0.01
        row.capillary_rise = 0.3
        row.precip_column = 1
        row.precip_factor = 1.0
        row.generic_moisture_column = 0
        row.surface_flow_dest_type = 2
        row.surface_flow_dest_id = 5

        rz_config = self._make_rz_config(
            tmp_path,
            version="4.2",
            soil_params=[row],
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.rootzone.RootZoneMainFileReader.read",
                return_value=rz_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.rootzone is not None
        sp = model.rootzone.soil_params.get(1)
        assert sp is not None
        assert sp.porosity == 0.4
        assert sp.field_capacity == 0.25
        assert sp.wilting_point == 0.1
        assert sp.saturated_kv == 0.5  # kh * k_factor(1.0)
        assert sp.lambda_param == 2.0
        # Pre-v4.12 surface flow dest
        assert model.rootzone.surface_flow_destinations.get(1) == (2, 5)

    def test_soil_params_populated_v412_plus(self, tmp_path: Path) -> None:
        """Soil params populated with v4.12+ per-landuse surface flow dests (lines 1223-1235)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        row = MagicMock()
        row.element_id = 1
        row.total_porosity = 0.35
        row.field_capacity = 0.2
        row.wilting_point = 0.08
        row.hydraulic_conductivity = 0.3
        row.lambda_param = 1.5
        row.kunsat_method = 2
        row.k_ponded = 0.02
        row.capillary_rise = 0.5
        row.precip_column = 2
        row.precip_factor = 1.2
        row.generic_moisture_column = 1
        row.dest_ag = 10
        row.dest_urban_in = 20
        row.dest_urban_out = 30
        row.dest_nvrv = 40

        rz_config = self._make_rz_config(
            tmp_path,
            version="4.12",
            soil_params=[row],
            k_factor=2.0,
            k_exdth_factor=1.5,
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.rootzone.RootZoneMainFileReader.read",
                return_value=rz_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.rootzone is not None
        sp = model.rootzone.soil_params.get(1)
        assert sp is not None
        assert sp.saturated_kv == 0.3 * 2.0  # kh * k_factor
        assert sp.capillary_rise == 0.5 * 1.5  # capillary_rise * k_exdth_factor
        # v4.12+ per-landuse dests: (raw_value, abs(raw_value))
        assert model.rootzone.surface_flow_dest_ag.get(1) == (10, 10)
        assert model.rootzone.surface_flow_dest_urban_in.get(1) == (20, 20)
        assert model.rootzone.surface_flow_dest_urban_out.get(1) == (30, 30)
        assert model.rootzone.surface_flow_dest_nvrv.get(1) == (40, 40)

    def test_rootzone_sub_file_paths_stored(self, tmp_path: Path) -> None:
        """All rootzone sub-file paths stored in source_files and metadata (lines 1152-1197)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        rz_config = self._make_rz_config(
            tmp_path,
            nonponded_crop_file=Path("nonponded.dat"),
            ponded_crop_file=Path("ponded.dat"),
            urban_file=Path("urban.dat"),
            native_veg_file=Path("native.dat"),
            return_flow_file=Path("return_flow.dat"),
            reuse_file=Path("reuse.dat"),
            irrigation_period_file=Path("irrig.dat"),
            ag_water_demand_file=Path("ag_demand.dat"),
            surface_flow_dest_file=Path("sfd.dat"),
            lwu_budget_file="lwu_budget.hdf",
            rz_budget_file="rz_budget.hdf",
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.rootzone.RootZoneMainFileReader.read",
                return_value=rz_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        # Source files
        assert model.source_files.get("rootzone_nonponded") == Path("nonponded.dat")
        assert model.source_files.get("rootzone_ponded") == Path("ponded.dat")
        assert model.source_files.get("rootzone_urban") == Path("urban.dat")
        assert model.source_files.get("rootzone_native") == Path("native.dat")
        assert model.source_files.get("rootzone_return_flow_ts") == Path("return_flow.dat")
        assert model.source_files.get("rootzone_reuse_ts") == Path("reuse.dat")
        assert model.source_files.get("rootzone_irig_period_ts") == Path("irrig.dat")
        assert model.source_files.get("rootzone_ag_demand_ts") == Path("ag_demand.dat")
        assert model.source_files.get("rootzone_surface_flow_dest") == Path("sfd.dat")
        # Metadata
        assert "rootzone_nonponded_file" in model.metadata
        assert "rootzone_ponded_file" in model.metadata
        assert "rootzone_urban_file" in model.metadata
        assert "rootzone_native_veg_file" in model.metadata
        assert "rootzone_lwu_budget_file" in model.metadata
        assert "rootzone_rz_budget_file" in model.metadata

    def test_v4x_sub_files_loaded(self, tmp_path: Path) -> None:
        """v4.x sub-files (nonponded, ponded, urban, native) are read (lines 1255-1295)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        # Create dummy sub-files so .exists() returns True
        np_path = sim_file.parent / "nonponded.dat"
        np_path.write_text("fake")
        pd_path = sim_file.parent / "ponded.dat"
        pd_path.write_text("fake")
        ur_path = sim_file.parent / "urban.dat"
        ur_path.write_text("fake")
        nv_path = sim_file.parent / "native.dat"
        nv_path.write_text("fake")

        rz_config = self._make_rz_config(
            tmp_path,
            nonponded_crop_file=np_path,
            ponded_crop_file=pd_path,
            urban_file=ur_path,
            native_veg_file=nv_path,
        )

        mock_np_config = MagicMock(name="np_config")
        mock_pd_config = MagicMock(name="pd_config")
        mock_ur_config = MagicMock(name="ur_config")
        mock_nv_config = MagicMock(name="nv_config")

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.rootzone.RootZoneMainFileReader.read",
                return_value=rz_config,
            ),
            patch(
                "pyiwfm.io.rootzone_v4x.NonPondedCropReaderV4x.read",
                return_value=mock_np_config,
            ),
            patch(
                "pyiwfm.io.rootzone_v4x.PondedCropReaderV4x.read",
                return_value=mock_pd_config,
            ),
            patch(
                "pyiwfm.io.rootzone_v4x.UrbanReaderV4x.read",
                return_value=mock_ur_config,
            ),
            patch(
                "pyiwfm.io.rootzone_v4x.NativeRiparianReaderV4x.read",
                return_value=mock_nv_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.rootzone is not None
        assert model.rootzone.nonponded_config is mock_np_config
        assert model.rootzone.ponded_config is mock_pd_config
        assert model.rootzone.urban_config is mock_ur_config
        assert model.rootzone.native_riparian_config is mock_nv_config

    def test_v4x_sub_file_exception_passes(self, tmp_path: Path) -> None:
        """v4.x sub-file reader exception silently passes (line 1294-1295)."""
        sim_file, pp_file, sim_config, mock_model = self._setup(tmp_path)

        np_path = sim_file.parent / "nonponded.dat"
        np_path.write_text("fake")

        rz_config = self._make_rz_config(
            tmp_path,
            nonponded_crop_file=np_path,
        )

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.rootzone.RootZoneMainFileReader.read",
                return_value=rz_config,
            ),
            patch(
                "pyiwfm.io.rootzone_v4x.NonPondedCropReaderV4x",
                side_effect=ImportError("rootzone_v4x not available"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        # Rootzone still loads despite sub-file failure
        assert model.rootzone is not None


# ===========================================================================
# 9. Deep small watershed and unsaturated zone loading branches
# ===========================================================================


class TestSmallWatershedDeepLoading:
    """Cover uncovered small watershed loading branches."""

    def test_budget_file_stored(self, tmp_path: Path) -> None:
        """Small watershed budget file stored in metadata (lines 1324-1327)."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        sw_path = sim_file.parent / "swshed.dat"
        sw_path.write_text("fake")
        sim_config.small_watershed_file = "swshed.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        sw_config = MagicMock()
        sw_config.version = "1.0"
        sw_config.n_watersheds = 2
        sw_config.budget_output_file = "sw_budget.hdf"

        mock_sw_component = MagicMock()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.small_watershed.SmallWatershedMainReader.read",
                return_value=sw_config,
            ),
            patch(
                "pyiwfm.components.small_watershed.AppSmallWatershed.from_config",
                return_value=mock_sw_component,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("small_watershed_budget_file") == "sw_budget.hdf"

    def test_no_budget_file_when_absent(self, tmp_path: Path) -> None:
        """No budget metadata when budget_output_file is falsy (else branch)."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        sw_path = sim_file.parent / "swshed.dat"
        sw_path.write_text("fake")
        sim_config.small_watershed_file = "swshed.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        sw_config = MagicMock()
        sw_config.version = "1.0"
        sw_config.n_watersheds = 0
        sw_config.budget_output_file = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.small_watershed.SmallWatershedMainReader.read",
                return_value=sw_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "small_watershed_budget_file" not in model.metadata
        assert model.small_watersheds is None  # n_watersheds == 0

    def test_load_error_stored(self, tmp_path: Path) -> None:
        """Small watershed loading error stored in metadata (lines 1334-1335)."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        sw_path = sim_file.parent / "swshed.dat"
        sw_path.write_text("fake")
        sim_config.small_watershed_file = "swshed.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.small_watershed.SmallWatershedMainReader.read",
                side_effect=ValueError("bad sw file"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "small_watershed_load_error" in model.metadata
        assert "bad sw file" in model.metadata["small_watershed_load_error"]


class TestUnsatZoneDeepLoading:
    """Cover uncovered unsaturated zone loading branches."""

    def test_budget_file_stored(self, tmp_path: Path) -> None:
        """UZ budget file stored in metadata (lines 1350-1353)."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        uz_path = sim_file.parent / "unsatzone.dat"
        uz_path.write_text("fake")
        sim_config.unsaturated_zone_file = "unsatzone.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        uz_config = MagicMock()
        uz_config.version = "2.0"
        uz_config.n_layers = 3
        uz_config.budget_file = "uz_budget.hdf"

        mock_uz_component = MagicMock()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.unsaturated_zone.UnsatZoneMainReader.read",
                return_value=uz_config,
            ),
            patch(
                "pyiwfm.components.unsaturated_zone.AppUnsatZone.from_config",
                return_value=mock_uz_component,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert model.metadata.get("unsat_zone_budget_file") == "uz_budget.hdf"

    def test_no_budget_file_when_absent(self, tmp_path: Path) -> None:
        """No UZ budget metadata when budget_file is falsy."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        uz_path = sim_file.parent / "unsatzone.dat"
        uz_path.write_text("fake")
        sim_config.unsaturated_zone_file = "unsatzone.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        uz_config = MagicMock()
        uz_config.version = "2.0"
        uz_config.n_layers = 0
        uz_config.budget_file = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.unsaturated_zone.UnsatZoneMainReader.read",
                return_value=uz_config,
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "unsat_zone_budget_file" not in model.metadata
        assert model.unsaturated_zone is None  # n_layers == 0

    def test_load_error_stored(self, tmp_path: Path) -> None:
        """UZ loading error stored in metadata (lines 1360-1361)."""
        sim_file, pp_file, sim_config = _base_sim_setup(tmp_path)
        uz_path = sim_file.parent / "unsatzone.dat"
        uz_path.write_text("fake")
        sim_config.unsaturated_zone_file = "unsatzone.dat"

        mock_model = IWFMModel(name="Test", mesh=_make_mock_mesh())
        mock_model.stratigraphy = MagicMock(n_layers=2)

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch(
                "pyiwfm.io.simulation.SimulationReader.read",
                return_value=sim_config,
            ),
            patch(
                "pyiwfm.io.preprocessor._resolve_path",
                side_effect=lambda base, p: Path(base) / p,
            ),
            patch(
                "pyiwfm.io.unsaturated_zone.UnsatZoneMainReader.read",
                side_effect=ValueError("bad uz file"),
            ),
        ):
            model = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert "unsat_zone_load_error" in model.metadata
        assert "bad uz file" in model.metadata["unsat_zone_load_error"]


# ===========================================================================
# 10. to_preprocessor and property coverage
# ===========================================================================


class TestToPreprocessorAndProperties:
    """Cover to_preprocessor file-check branches and has_* properties."""

    def test_to_preprocessor_with_files(self, tmp_path: Path) -> None:
        """to_preprocessor returns dict with nodes/elements keys (lines 1408-1410)."""
        mock_config = MagicMock()
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None

        model = IWFMModel(name="Test")
        with patch(
            "pyiwfm.io.preprocessor.save_model_to_preprocessor",
            return_value=mock_config,
        ):
            files = model.to_preprocessor(tmp_path)

        assert "nodes" in files
        assert "elements" in files
        assert "stratigraphy" not in files

    def test_has_small_watersheds_property(self) -> None:
        """has_small_watersheds returns True when component present (line 1638)."""
        model = IWFMModel(name="Test", small_watersheds=MagicMock())
        assert model.has_small_watersheds is True

        model2 = IWFMModel(name="Test")
        assert model2.has_small_watersheds is False

    def test_has_unsaturated_zone_property(self) -> None:
        """has_unsaturated_zone returns True when component present (line 1643)."""
        model = IWFMModel(name="Test", unsaturated_zone=MagicMock())
        assert model.has_unsaturated_zone is True

        model2 = IWFMModel(name="Test")
        assert model2.has_unsaturated_zone is False

    def test_summary_with_all_loaded_includes_all_sections(self) -> None:
        """summary() includes all component sections for complete model (line 1638+1643)."""
        mesh = MagicMock()
        mesh.n_nodes = 10
        mesh.n_elements = 5
        mesh.n_subregions = 2

        strat = MagicMock()
        strat.n_layers = 3

        gw = MagicMock()
        gw.n_wells = 2
        gw.n_hydrograph_locations = 1
        gw.n_boundary_conditions = 3
        gw.n_tile_drains = 1
        gw.aquifer_params = MagicMock()

        streams = MagicMock()
        streams.n_nodes = 20
        streams.n_reaches = 5
        streams.n_diversions = 2
        streams.n_bypasses = 1

        lakes = MagicMock()
        lakes.n_lakes = 1
        lakes.n_lake_elements = 4

        rz = MagicMock()
        rz.n_crop_types = 7
        rz.element_landuse = {1: "ag"}
        rz.soil_params = {1: MagicMock()}

        sw = MagicMock()
        sw.n_watersheds = 2

        uz = MagicMock()
        uz.n_layers = 3
        uz.n_elements = 50

        model = IWFMModel(
            name="CompleteModel",
            mesh=mesh,
            stratigraphy=strat,
            groundwater=gw,
            streams=streams,
            lakes=lakes,
            rootzone=rz,
            small_watersheds=sw,
            unsaturated_zone=uz,
        )
        text = model.summary()
        assert "CompleteModel" in text
        assert model.has_small_watersheds is True
        assert model.has_unsaturated_zone is True
