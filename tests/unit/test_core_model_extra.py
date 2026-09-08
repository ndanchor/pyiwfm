"""Extra tests for pyiwfm.core.model to increase coverage to 95%+.

Targets uncovered lines: 54, 57-58, 349-358, 367->371, 451-452,
663->1014, 680, 687, 691, 694->696, 696->700, 714, 744, 758-760,
776, 797-807, 811, 830, 838-839, 867, 880-881, 921, 932->936, 939-940,
960->969, 965-966, 982-983, 1007-1008, 1017->1324, 1163->1162,
1184->1189, 1222->1221, 1234->1233, 1243->1252, 1278-1310,
1327->1385, 1388->1769, 1524-1601, 1643-1644, 1660-1661, 1680-1681,
1691-1697, 1709-1715, 1752->1738, 1759-1760, 1865->1867
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from pyiwfm.core.model import (
    IWFMModel,
    _build_reaches_from_node_reach_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_config(base_dir: Path, **overrides):
    """Build a minimal SimulationConfig-like mock."""
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
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _pp_model(mesh_n_nodes=4, mesh_n_elements=1, n_layers=2):
    """Return a model as if from_preprocessor succeeded."""
    mock_mesh = MagicMock()
    mock_mesh.n_nodes = mesh_n_nodes
    mock_mesh.n_elements = mesh_n_elements
    mock_mesh.nodes = {
        i: MagicMock(id=i, x=float(i), y=float(i)) for i in range(1, mesh_n_nodes + 1)
    }
    mock_mesh.elements = {i: MagicMock(id=i) for i in range(1, mesh_n_elements + 1)}
    mock_strat = MagicMock()
    mock_strat.n_layers = n_layers
    mock_strat.n_nodes = mesh_n_nodes
    return IWFMModel(
        name="TestModel",
        mesh=mock_mesh,
        stratigraphy=mock_strat,
        metadata={"source": "preprocessor"},
    )


# ===========================================================================
# 1. _build_reaches_from_node_reach_ids  (lines 54, 57-58)
# ===========================================================================


class TestBuildReachesFromNodeReachIds:
    """Tests for _build_reaches_from_node_reach_ids helper."""

    def test_builds_reaches_from_node_reach_ids(self):
        """Nodes with reach_id > 0 should be grouped into reaches."""
        from pyiwfm.components.stream import AppStream, StrmNode

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0, reach_id=1))
        stream.add_node(StrmNode(id=3, x=2.0, y=0.0, reach_id=2))

        _build_reaches_from_node_reach_ids(stream)

        assert len(stream.reaches) == 2
        reach1 = stream.reaches[1]
        assert reach1.upstream_node == 1
        assert reach1.downstream_node == 2
        assert reach1.nodes == [1, 2]
        reach2 = stream.reaches[2]
        assert reach2.upstream_node == 3
        assert reach2.downstream_node == 3

    def test_skips_when_reaches_already_populated(self):
        """If reaches already exist, the function does nothing."""
        from pyiwfm.components.stream import AppStream, StrmNode, StrmReach

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1))
        stream.add_reach(StrmReach(id=99, upstream_node=1, downstream_node=1, nodes=[1]))

        _build_reaches_from_node_reach_ids(stream)

        # Should still have the original reach, not build new ones
        assert len(stream.reaches) == 1
        assert 99 in stream.reaches

    def test_no_reach_ids_means_no_reaches(self):
        """Nodes without reach_id (or reach_id=0) produce no reaches."""
        from pyiwfm.components.stream import AppStream, StrmNode

        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0))

        _build_reaches_from_node_reach_ids(stream)

        assert len(stream.reaches) == 0


# ===========================================================================
# 2. from_preprocessor – stream spec reader fallback (lines 349-358)
# ===========================================================================


class TestFromPreprocessorStreamFallback:
    """Test the StreamSpec fallback path in from_preprocessor."""

    def test_stream_spec_reader_fallback(self, tmp_path: Path):
        """When StreamReader fails, falls back to StreamSpecReader."""
        streams_path = tmp_path / "streams.dat"
        streams_path.write_text("fake")

        mock_pp_config = MagicMock()
        mock_pp_config.model_name = "Test"
        mock_pp_config.nodes_file = tmp_path / "nodes.dat"
        mock_pp_config.elements_file = tmp_path / "elements.dat"
        mock_pp_config.stratigraphy_file = None
        mock_pp_config.subregions_file = None
        mock_pp_config.streams_file = streams_path
        mock_pp_config.lakes_file = None
        mock_pp_config.length_unit = "FT"
        mock_pp_config.area_unit = "SQFT"
        mock_pp_config.volume_unit = "CUFT"

        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 2
        mock_mesh.n_elements = 1
        mock_mesh.nodes = {1: MagicMock(), 2: MagicMock()}
        mock_mesh.elements = {1: MagicMock()}

        # Reach spec data
        rs = MagicMock()
        rs.id = 1
        rs.node_ids = [10, 11]
        rs.node_to_gw_node = {10: 1, 11: 2}
        rs.name = "Reach1"

        mock_stream_reader = MagicMock()
        mock_stream_reader.read_stream_nodes.side_effect = ValueError("fail")

        mock_spec_reader = MagicMock()
        mock_spec_reader.read.return_value = (1, 0, [rs])

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_pp_config),
            patch(
                "pyiwfm.io.ascii.read_nodes", return_value=({1: MagicMock(), 2: MagicMock()}, 1.0)
            ),
            patch("pyiwfm.io.ascii.read_elements", return_value=({1: MagicMock()}, 1, {})),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mock_mesh),
            patch("pyiwfm.io.streams.StreamReader", return_value=mock_stream_reader),
            patch("pyiwfm.io.streams.StreamSpecReader", return_value=mock_spec_reader),
        ):
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")

        assert model.streams is not None
        assert len(model.streams.nodes) >= 2

    def test_both_stream_readers_fail_reraises_first(self, tmp_path: Path):
        """When both StreamReader and StreamSpecReader fail, the first error is re-raised."""
        streams_path = tmp_path / "streams.dat"
        streams_path.write_text("fake")

        mock_pp_config = MagicMock()
        mock_pp_config.model_name = "Test"
        mock_pp_config.nodes_file = tmp_path / "nodes.dat"
        mock_pp_config.elements_file = tmp_path / "elements.dat"
        mock_pp_config.stratigraphy_file = None
        mock_pp_config.subregions_file = None
        mock_pp_config.streams_file = streams_path
        mock_pp_config.lakes_file = None
        mock_pp_config.length_unit = "FT"
        mock_pp_config.area_unit = "SQFT"
        mock_pp_config.volume_unit = "CUFT"

        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 2
        mock_mesh.n_elements = 1
        mock_mesh.nodes = {1: MagicMock(), 2: MagicMock()}
        mock_mesh.elements = {1: MagicMock()}

        # Real parsers raise ValueError on bad data; the outer try/except in
        # from_preprocessor narrows its catch to expected parser exceptions.
        mock_stream_reader = MagicMock()
        mock_stream_reader.read_stream_nodes.side_effect = ValueError("first fail")

        mock_spec_reader = MagicMock()
        mock_spec_reader.read.side_effect = ValueError("second fail")

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_pp_config),
            patch(
                "pyiwfm.io.ascii.read_nodes", return_value=({1: MagicMock(), 2: MagicMock()}, 1.0)
            ),
            patch("pyiwfm.io.ascii.read_elements", return_value=({1: MagicMock()}, 1, {})),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mock_mesh),
            patch("pyiwfm.io.streams.StreamReader", return_value=mock_stream_reader),
            patch("pyiwfm.io.streams.StreamSpecReader", return_value=mock_spec_reader),
        ):
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")

        # The error is caught by the outer try/except and stored in metadata
        assert "streams_load_error" in model.metadata
        assert "first fail" in model.metadata["streams_load_error"]


# ===========================================================================
# 3. from_simulation_with_preprocessor – GW loading paths
# ===========================================================================


class TestFromSimWithPPGroundwater:
    """Test groundwater loading in from_simulation_with_preprocessor."""

    def _run_sim_with_pp(self, tmp_path, sim_cfg, pp_model_instance=None):
        """Run from_simulation_with_preprocessor with mocked deps."""
        sim_file = tmp_path / "sim.in"
        pp_file = tmp_path / "pp.in"
        sim_file.write_text("fake")
        pp_file.write_text("fake")

        if pp_model_instance is None:
            pp_model_instance = _pp_model()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=pp_model_instance),
            patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda bd, p: Path(p)),
        ):
            mock_sim_reader_cls.return_value.read.return_value = sim_cfg
            return IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

    def test_gw_main_file_reader_with_hydrographs(self, tmp_path: Path):
        """Test GW main file reader path – hydrograph locations and output files."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = [MagicMock()]
        gw_config.budget_output_file = "gw_budget.hdf"
        gw_config.zbudget_output_file = "gw_zbudget.hdf"
        gw_config.head_all_output_file = "head_all.out"
        gw_config.hydrograph_output_file = "gw_hydro.out"
        gw_config.volume_output_unit = "AF"
        gw_config.head_output_unit = "FT"
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.groundwater is not None
        assert model.metadata.get("gw_version") == "4.0"
        assert model.metadata.get("gw_budget_file") == "gw_budget.hdf"
        assert model.metadata.get("gw_zbudget_file") == "gw_zbudget.hdf"
        assert model.metadata.get("gw_head_all_file") == "head_all.out"
        assert model.metadata.get("gw_hydrograph_file") == "gw_hydro.out"
        assert model.metadata.get("gw_volume_output_unit") == "AF"
        assert model.metadata.get("gw_length_output_unit") == "FT"

    def test_gw_bc_reader_with_constrained_gh(self, tmp_path: Path):
        """Test GW boundary condition reader with constrained GH BCs and TS file."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")
        bc_file = tmp_path / "gw_bc.dat"
        bc_file.write_text("fake")

        bc_config = MagicMock()
        bc_config.n_specified_flow = 0
        bc_config.n_specified_head = 0
        bc_config.n_general_head = 0
        bc_config.n_constrained_gh = 1
        # n_bc_output_nodes is compared to int (> 0); set explicitly so the
        # comparison doesn't TypeError when the helper formerly hidden by
        # the over-broad except now reaches it.
        bc_config.n_bc_output_nodes = 0
        bc_config.specified_head_bcs = []
        bc_config.specified_flow_bcs = []
        bc_config.general_head_bcs = []
        cgh_bc = MagicMock()
        cgh_bc.node_id = 1
        cgh_bc.external_head = 100.0
        cgh_bc.layer = 1
        cgh_bc.conductance = 0.5
        cgh_bc.constraining_head = 90.0
        cgh_bc.max_flow = 10.0
        cgh_bc.ts_column = 1
        cgh_bc.max_flow_ts_column = 2
        bc_config.constrained_gh_bcs = [cgh_bc]
        bc_config.ts_data_file = tmp_path / "bc_ts.dat"

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = bc_file
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch("pyiwfm.io.gw_boundary.GWBoundaryReader") as mock_bc_cls,
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            mock_bc_cls.return_value.read.return_value = bc_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.groundwater is not None
        assert model.metadata.get("gw_n_constrained_gh_bc") == 1

    def test_gw_pumping_with_well_specs(self, tmp_path: Path):
        """Test GW pumping reader merges well pumping specs onto Well objects."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")
        pump_file = tmp_path / "gw_pump.dat"
        pump_file.write_text("fake")

        ws = MagicMock()
        ws.id = 1
        ws.x = 10.0
        ws.y = 20.0
        ws.name = "Well1"
        ws.perf_top = 50.0
        ws.perf_bottom = 10.0
        ws.radius = 0.5

        wps = MagicMock()
        wps.well_id = 1
        wps.pump_column = 1
        wps.pump_fraction = 1.0
        wps.dist_method = 1
        wps.dest_type = 0
        wps.dest_id = 0
        wps.irig_frac_column = 0
        wps.adjust_column = 0
        wps.pump_max_column = 0
        wps.pump_max_fraction = 0.0

        eps = MagicMock()
        eps.element_id = 1
        eps.pump_column = 2
        eps.pump_fraction = 0.5
        eps.dist_method = 0
        eps.layer_factors = [1.0, 0.0]
        eps.dest_type = 0
        eps.dest_id = 0
        eps.irig_frac_column = 0
        eps.adjust_column = 0
        eps.pump_max_column = 0
        eps.pump_max_fraction = 0.0

        pump_config = MagicMock()
        pump_config.n_wells = 1
        pump_config.n_elem_pumping = 1
        pump_config.ts_data_file = tmp_path / "pump_ts.dat"
        pump_config.well_specs = [ws]
        pump_config.well_pumping_specs = [wps]
        pump_config.elem_pumping_specs = [eps]

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = None
        gw_config.pumping_file = pump_file
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch("pyiwfm.io.gw_pumping.PumpingReader") as mock_pump_cls,
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            mock_pump_cls.return_value.read.return_value = pump_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.groundwater is not None
        assert model.metadata.get("gw_n_wells") == 1
        assert model.metadata.get("gw_n_elem_pumping") == 1

    def test_gw_pumping_fallback_to_simple_well_reader(self, tmp_path: Path):
        """When PumpingReader raises, falls back to GroundwaterReader.read_wells."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")
        pump_file = tmp_path / "gw_pump.dat"
        pump_file.write_text("fake")

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = None
        gw_config.pumping_file = pump_file
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        mock_well = MagicMock()

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch("pyiwfm.io.gw_pumping.PumpingReader") as mock_pump_cls,
            patch("pyiwfm.io.groundwater.GroundwaterReader") as mock_gw_reader_cls,
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            mock_pump_cls.return_value.read.side_effect = ValueError("pump fail")
            mock_gw_reader_cls.return_value.read_wells.return_value = {1: mock_well}
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.groundwater is not None

    def test_gw_tile_drain_with_sub_irrigation(self, tmp_path: Path):
        """Test tile drain loading with sub-irrigation."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")
        td_file = tmp_path / "gw_td.dat"
        td_file.write_text("fake")

        td = MagicMock()
        td.id = 1
        td.gw_node = 1
        td.elevation = 100.0
        td.conductance = 0.5
        td.dest_type = 2
        td.dest_id = 3

        si = MagicMock()
        si.id = 1
        si.gw_node = 2
        si.elevation = 95.0
        si.conductance = 0.3

        td_config = MagicMock()
        td_config.n_drains = 1
        td_config.n_sub_irrigation = 1
        td_config.tile_drains = [td]
        td_config.sub_irrigations = [si]
        td_config.drain_height_factor = 1.0
        td_config.drain_conductance_factor = 1.0
        td_config.drain_time_unit = "1MIN"
        td_config.subirig_height_factor = 1.0
        td_config.subirig_conductance_factor = 1.0
        td_config.subirig_time_unit = "1MIN"

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = td_file
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch("pyiwfm.io.gw_tiledrain.TileDrainReader") as mock_td_cls,
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            mock_td_cls.return_value.read.return_value = td_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.metadata.get("gw_n_tile_drains") == 1
        assert model.metadata.get("gw_n_sub_irrigation") == 1

    def test_gw_tile_drain_exception_passes(self, tmp_path: Path):
        """Tile drain reader exception is silently caught."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")
        td_file = tmp_path / "gw_td.dat"
        td_file.write_text("fake")

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = td_file
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch("pyiwfm.io.gw_tiledrain.TileDrainReader") as mock_td_cls,
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            mock_td_cls.return_value.read.side_effect = ValueError("td fail")
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.groundwater is not None
        # No crash; tile drain error silently passed

    def test_gw_subsidence_loading(self, tmp_path: Path):
        """Test subsidence reader loading – NodeSubsidence + legacy Subsidence."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")
        subs_file = tmp_path / "gw_subs.dat"
        subs_file.write_text("fake")

        sp = MagicMock()
        sp.node_id = 1
        sp.elastic_sc = [0.001, 0.002]
        sp.inelastic_sc = [0.01, 0.02]
        sp.interbed_thick = [10.0, 20.0]
        sp.interbed_thick_min = [1.0, 2.0]
        sp.precompact_head = [50.0, 40.0]
        sp.kv_sub = [0.1, 0.2]
        sp.n_eq = [1, 1]

        subs_config = MagicMock()
        subs_config.version = "4.0"
        subs_config.node_params = [sp]
        subs_config.n_hydrograph_outputs = 0
        subs_config.hydrograph_output_file = "subs_hydro.out"

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = subs_file
        gw_config.aquifer_params = None
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch("pyiwfm.io.gw_subsidence.SubsidenceReader") as mock_subs_cls,
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            mock_subs_cls.return_value.read.return_value = subs_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.metadata.get("gw_subsidence_version") == "4.0"
        assert model.metadata.get("subsidence_hydrograph_file") == "subs_hydro.out"

    def test_gw_subsidence_exception_passes(self, tmp_path: Path):
        """Subsidence reader exception is silently caught."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")
        subs_file = tmp_path / "gw_subs.dat"
        subs_file.write_text("fake")

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = subs_file
        gw_config.aquifer_params = None
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch("pyiwfm.io.gw_subsidence.SubsidenceReader") as mock_subs_cls,
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            mock_subs_cls.return_value.read.side_effect = ValueError("subs fail")
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.groundwater is not None

    def test_gw_parametric_grids_exception(self, tmp_path: Path):
        """Parametric grid exception is silently caught."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = None
        gw_config.parametric_grids = [MagicMock()]  # non-None
        gw_config.kh_anomalies = None
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch(
                "pyiwfm.core.model._apply_parametric_grids",
                side_effect=ValueError("pgrid fail"),
            ),
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.groundwater is not None

    def test_gw_kh_anomaly_exception(self, tmp_path: Path):
        """Kh anomaly exception is silently caught."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")

        mock_aq_params = MagicMock()

        gw_config = MagicMock()
        gw_config.version = "4.0"
        gw_config.hydrograph_locations = []
        gw_config.budget_output_file = None
        gw_config.zbudget_output_file = None
        gw_config.head_all_output_file = None
        gw_config.hydrograph_output_file = None
        gw_config.volume_output_unit = None
        gw_config.head_output_unit = None
        gw_config.bc_file = None
        gw_config.pumping_file = None
        gw_config.tile_drain_file = None
        gw_config.subsidence_file = None
        gw_config.aquifer_params = mock_aq_params
        gw_config.parametric_grids = None
        gw_config.kh_anomalies = [MagicMock()]  # non-empty
        gw_config.initial_heads = None

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch("pyiwfm.io.groundwater.GWMainFileReader") as mock_gw_cls,
            patch(
                "pyiwfm.core.model._apply_kh_anomalies",
                side_effect=ValueError("anom fail"),
            ),
        ):
            mock_gw_cls.return_value.read.return_value = gw_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.groundwater is not None

    def test_gw_outer_exception_stores_error(self, tmp_path: Path):
        """When AppGW creation itself raises, error goes to metadata."""
        gw_file = tmp_path / "gw.dat"
        gw_file.write_text("fake")

        sim_cfg = _sim_config(tmp_path, groundwater_file=str(gw_file))

        with (
            patch(
                "pyiwfm.components.groundwater.AppGW",
                side_effect=ValueError("gw init fail"),
            ),
        ):
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert "groundwater_load_error" in model.metadata
        assert "gw init fail" in model.metadata["groundwater_load_error"]


# ===========================================================================
# 4. from_simulation_with_preprocessor – Stream reach enrichment
# ===========================================================================


class TestFromSimWithPPStreamEnrichment:
    """Test stream reach enrichment from preprocessor (lines 1278-1310)."""

    def _run_sim_with_pp(self, tmp_path, sim_cfg, pp_model_instance=None):
        sim_file = tmp_path / "sim.in"
        pp_file = tmp_path / "pp.in"
        sim_file.write_text("fake")
        pp_file.write_text("fake")

        if pp_model_instance is None:
            pp_model_instance = _pp_model()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=pp_model_instance),
            patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda bd, p: Path(p)),
        ):
            mock_sim_reader_cls.return_value.read.return_value = sim_cfg
            return IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

    def test_stream_reach_enrichment_from_pp(self, tmp_path: Path):
        """When stream nodes exist but no reaches, enriches from preprocessor."""
        from pyiwfm.components.stream import AppStream, StrmNode

        stream_file = tmp_path / "stream.dat"
        stream_file.write_text("fake")

        pp_model = _pp_model()
        # Put a stream with nodes but no reaches on the preprocessor model
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0))
        stream.add_node(StrmNode(id=2, x=1.0, y=0.0))
        pp_model.streams = stream

        # Make the stream main file reader fail so it uses existing stream
        sim_cfg = _sim_config(tmp_path, streams_file=str(stream_file))

        # StreamMainFileReader will be imported and used
        stream_config = MagicMock()
        stream_config.version = "4.0"
        stream_config.hydrograph_count = 0
        stream_config.hydrograph_output_type = None
        stream_config.budget_output_file = None
        stream_config.diversion_budget_file = None
        stream_config.hydrograph_output_file = None
        stream_config.hydrograph_specs = []
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
        stream_config.conductivity_time_unit = "1MIN"
        stream_config.length_factor = 1.0
        stream_config.roughness_factor = 1.0
        stream_config.cross_section_length_factor = 1.0

        # PP config for enrichment
        pp_config = MagicMock()
        mock_streams_file = MagicMock()
        mock_streams_file.exists.return_value = True
        pp_config.streams_file = mock_streams_file

        rs = MagicMock()
        rs.id = 1
        rs.node_ids = [1, 2]
        rs.node_to_gw_node = {1: 10, 2: 20}
        rs.name = "Reach1"

        with (
            patch("pyiwfm.io.streams.StreamMainFileReader") as mock_stream_main_cls,
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=pp_config),
            patch("pyiwfm.io.streams.StreamSpecReader") as mock_spec_cls,
        ):
            mock_stream_main_cls.return_value.read.return_value = stream_config
            mock_spec_cls.return_value.read.return_value = (1, 0, [rs])
            model = self._run_sim_with_pp(tmp_path, sim_cfg, pp_model)

        assert model.streams is not None
        assert len(model.streams.reaches) >= 1


# ===========================================================================
# 4b. from_simulation_with_preprocessor – Stream loading details
# ===========================================================================


class TestFromSimWithPPStreamDetails:
    """Test stream loading: bed params, cross-sections, ICs, budget nodes, bypasses."""

    def _run_sim_with_pp(self, tmp_path, sim_cfg, pp_model_instance=None):
        sim_file = tmp_path / "sim.in"
        pp_file = tmp_path / "pp.in"
        sim_file.write_text("fake")
        pp_file.write_text("fake")

        if pp_model_instance is None:
            pp_model_instance = _pp_model()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=pp_model_instance),
            patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda bd, p: Path(p)),
        ):
            mock_sim_reader_cls.return_value.read.return_value = sim_cfg
            return IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

    def _make_stream_config(self, tmp_path):
        """Build a complete stream config with bed params, etc."""
        sc = MagicMock()
        sc.version = "5.0"
        sc.hydrograph_count = 0
        sc.hydrograph_output_type = None
        sc.budget_output_file = "stream_budget.hdf"
        sc.diversion_budget_file = "div_budget.hdf"
        sc.hydrograph_output_file = "stream_hydro.out"
        sc.hydrograph_specs = [(1, "Node1")]

        # Bed params with a node not in stream.nodes (forces node creation)
        bp = MagicMock()
        bp.node_id = 99
        bp.conductivity = 0.5
        bp.bed_thickness = 1.0
        bp.wetted_perimeter = 3.0
        bp.gw_node = 5
        sc.bed_params = [bp]

        sc.interaction_type = 2

        # Cross-section data
        cs = MagicMock()
        cs.node_id = 99  # match the bed param node
        cs.bottom_elev = 10.0
        cs.B0 = 5.0
        cs.s = 1.5
        cs.n = 0.035
        cs.max_flow_depth = 20.0
        sc.cross_section_data = [cs]
        sc.roughness_factor = 1.0
        sc.cross_section_length_factor = 1.0

        # Initial conditions
        ic = MagicMock()
        ic.node_id = 99
        ic.value = 15.0
        sc.initial_conditions = [ic]
        sc.ic_type = 1
        sc.ic_factor = 1.0

        # Budget nodes
        sc.node_budget_count = 1
        sc.node_budget_ids = [99]
        sc.node_budget_output_file = "node_budget.hdf"

        # Final flow file
        sc.final_flow_file = "final_flow.out"

        sc.conductivity_factor = 1.0
        sc.conductivity_time_unit = "1MIN"
        sc.length_factor = 1.0

        # Evaporation
        sc.evap_area_file = "evap_area.dat"
        sc.evap_node_specs = [(99, 1, 2)]

        # Sub-files
        sc.inflow_file = None
        sc.diversion_spec_file = None
        sc.bypass_spec_file = None
        sc.diversion_file = None

        return sc

    def test_stream_bed_params_create_node(self, tmp_path: Path):
        """Bed params referencing unknown node_id create new StrmNode."""
        stream_file = tmp_path / "stream.dat"
        stream_file.write_text("fake")

        stream_config = self._make_stream_config(tmp_path)

        sim_cfg = _sim_config(tmp_path, streams_file=str(stream_file))

        with patch("pyiwfm.io.streams.StreamMainFileReader") as mock_sc_cls:
            mock_sc_cls.return_value.read.return_value = stream_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.streams is not None
        assert 99 in model.streams.nodes
        assert model.metadata.get("stream_node_budget_file") == "node_budget.hdf"

    def test_stream_bypass_loading(self, tmp_path: Path):
        """Test bypass loading with rating table and seepage zones."""
        stream_file = tmp_path / "stream.dat"
        stream_file.write_text("fake")
        bypass_file = tmp_path / "bypass.dat"
        bypass_file.write_text("fake")

        stream_config = self._make_stream_config(tmp_path)
        stream_config.bypass_spec_file = bypass_file

        bs = MagicMock()
        bs.id = 1
        bs.export_stream_node = 1
        bs.dest_id = 2
        bs.dest_type = 0
        bs.name = "Bypass1"
        bs.rating_table_col = 0
        bs.frac_recoverable = 0.1
        bs.frac_non_recoverable = 0.05
        bs.inline_rating = MagicMock()
        bs.inline_rating.flows = np.array([0.0, 100.0, 200.0])
        bs.inline_rating.fractions = np.array([0.0, 0.5, 1.0])

        sz = MagicMock()
        sz.bypass_id = 1

        byp_config = MagicMock()
        byp_config.n_bypasses = 1
        byp_config.bypasses = [bs]
        byp_config.flow_factor = 2.0
        byp_config.flow_time_unit = "1DAY"
        byp_config.bypass_factor = 1.0
        byp_config.bypass_time_unit = "1DAY"
        byp_config.seepage_zones = [sz]

        sim_cfg = _sim_config(tmp_path, streams_file=str(stream_file))

        with (
            patch("pyiwfm.io.streams.StreamMainFileReader") as mock_sc_cls,
            patch("pyiwfm.io.stream_bypass.BypassSpecReader") as mock_byp_cls,
        ):
            mock_sc_cls.return_value.read.return_value = stream_config
            mock_byp_cls.return_value.read.return_value = byp_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.streams is not None
        assert model.metadata.get("stream_n_bypasses") == 1

    def test_stream_diversion_loading(self, tmp_path: Path):
        """Test diversion loading from sub-file."""
        stream_file = tmp_path / "stream.dat"
        stream_file.write_text("fake")
        div_file = tmp_path / "div.dat"
        div_file.write_text("fake")

        stream_config = self._make_stream_config(tmp_path)
        stream_config.diversion_spec_file = div_file

        ds = MagicMock()
        ds.id = 1
        ds.stream_node = 1
        ds.dest_type = 1
        ds.dest_id = 5
        ds.name = "Div1"
        ds.max_diver_col = 1
        ds.frac_max_diver = 1.0
        ds.recv_loss_col = 0
        ds.frac_recv_loss = 0.0
        ds.non_recv_loss_col = 0
        ds.frac_non_recv_loss = 0.0
        ds.spill_col = 0
        ds.frac_spill = 0.0
        ds.delivery_col = 0
        ds.frac_delivery = 1.0
        ds.irrig_frac_col = 0
        ds.adjustment_col = 0

        div_config = MagicMock()
        div_config.n_diversions = 1
        div_config.n_element_groups = 0
        div_config.diversions = [ds]
        div_config.element_groups = []
        div_config.recharge_zones = []
        div_config.spill_zones = []
        div_config.has_spills = False

        sim_cfg = _sim_config(tmp_path, streams_file=str(stream_file))

        with (
            patch("pyiwfm.io.streams.StreamMainFileReader") as mock_sc_cls,
            patch("pyiwfm.io.stream_diversion.DiversionSpecReader") as mock_div_cls,
        ):
            mock_sc_cls.return_value.read.return_value = stream_config
            mock_div_cls.return_value.read.return_value = div_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.streams is not None
        assert model.metadata.get("stream_n_diversions") == 1

    def test_stream_inflow_loading(self, tmp_path: Path):
        """Test inflow loading from sub-file."""
        stream_file = tmp_path / "stream.dat"
        stream_file.write_text("fake")
        inflow_file = tmp_path / "inflow.dat"
        inflow_file.write_text("fake")

        stream_config = self._make_stream_config(tmp_path)
        stream_config.inflow_file = inflow_file

        inflow_config = MagicMock()
        inflow_config.n_inflows = 2
        inflow_config.inflow_nodes = [1, 2]

        sim_cfg = _sim_config(tmp_path, streams_file=str(stream_file))

        with (
            patch("pyiwfm.io.streams.StreamMainFileReader") as mock_sc_cls,
            patch("pyiwfm.io.stream_inflow.InflowReader") as mock_inf_cls,
        ):
            mock_sc_cls.return_value.read.return_value = stream_config
            mock_inf_cls.return_value.read.return_value = inflow_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.metadata.get("stream_n_inflows") == 2

    def test_stream_main_reader_exception_falls_back(self, tmp_path: Path):
        """When StreamMainFileReader fails, falls back to StreamReader."""
        stream_file = tmp_path / "stream.dat"
        stream_file.write_text("fake")

        # reach_id needs an explicit int — _build_reaches_from_node_reach_ids
        # compares it to 0 and the auto-MagicMock attribute would TypeError.
        mock_node = MagicMock(reach_id=0, id=1)

        sim_cfg = _sim_config(tmp_path, streams_file=str(stream_file))

        with (
            patch("pyiwfm.io.streams.StreamMainFileReader") as mock_sc_cls,
            patch("pyiwfm.io.streams.StreamReader") as mock_sr_cls,
        ):
            mock_sc_cls.return_value.read.side_effect = ValueError("main fail")
            mock_sr_cls.return_value.read_stream_nodes.return_value = {1: mock_node}
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.streams is not None

    def test_stream_load_exception_stores_error(self, tmp_path: Path):
        """Outer stream loading exception stored in metadata."""
        stream_file = tmp_path / "stream.dat"
        stream_file.write_text("fake")

        sim_cfg = _sim_config(tmp_path, streams_file=str(stream_file))

        with (
            patch(
                "pyiwfm.components.stream.AppStream",
                side_effect=ValueError("stream init fail"),
            ),
        ):
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert "streams_load_error" in model.metadata

    def test_stream_lakes_not_loaded_when_already_present(self, tmp_path: Path):
        """Lakes are not loaded when model.lakes is already set."""
        lake_file = tmp_path / "lake.dat"
        lake_file.write_text("fake")

        pp_model = _pp_model()
        pp_model.lakes = MagicMock()  # Already set

        sim_cfg = _sim_config(tmp_path, lakes_file=str(lake_file))

        model = self._run_sim_with_pp(tmp_path, sim_cfg, pp_model)

        # Lakes should remain the preprocessor-loaded one
        assert model.lakes is pp_model.lakes

    def test_lake_loading_with_hierarchical_reader(self, tmp_path: Path):
        """Test lake loading via LakeMainFileReader."""
        lake_file = tmp_path / "lake.dat"
        lake_file.write_text("fake")

        lake_config = MagicMock()
        lake_config.version = "4.0"
        lake_config.lake_params = [MagicMock(lake_id=1, name="Lake1")]
        lake_config.max_elev_file = tmp_path / "max_elev.dat"
        lake_config.budget_output_file = "lake_budget.hdf"
        lake_config.conductance_factor = 1.0
        lake_config.depth_factor = 1.0
        lake_config.outflow_ratings = None

        sim_cfg = _sim_config(tmp_path, lakes_file=str(lake_file))

        with patch("pyiwfm.io.lakes.LakeMainFileReader") as mock_lk_cls:
            mock_lk_cls.return_value.read.return_value = lake_config
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.lakes is not None
        assert model.metadata.get("lake_version") == "4.0"


# ===========================================================================
# 5. from_simulation_with_preprocessor – Root zone loading
# ===========================================================================


class TestFromSimWithPPRootZone:
    """Test rootzone loading in from_simulation_with_preprocessor."""

    def _run_sim_with_pp(self, tmp_path, sim_cfg, pp_model_instance=None):
        sim_file = tmp_path / "sim.in"
        pp_file = tmp_path / "pp.in"
        sim_file.write_text("fake")
        pp_file.write_text("fake")

        if pp_model_instance is None:
            pp_model_instance = _pp_model()

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=pp_model_instance),
            patch("pyiwfm.io.simulation.SimulationReader") as mock_sim_reader_cls,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda bd, p: Path(p)),
        ):
            mock_sim_reader_cls.return_value.read.return_value = sim_cfg
            return IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

    def _make_rz_config(self, tmp_path, version="4.0"):
        """Build a mock rootzone config."""
        rz_config = MagicMock()
        rz_config.version = version
        rz_config.gw_uptake_enabled = True

        # Soil params
        row = MagicMock()
        row.element_id = 1
        row.total_porosity = 0.4
        row.field_capacity = 0.25
        row.wilting_point = 0.1
        row.hydraulic_conductivity = 1.0
        row.lambda_param = 0.5
        row.kunsat_method = 1
        row.k_ponded = 0.0
        row.capillary_rise = 0.5
        row.precip_column = 1
        row.precip_factor = 1.0
        row.generic_moisture_column = 0
        row.surface_flow_dest_type = 0
        row.surface_flow_dest_id = 0
        row.dest_ag = 0
        row.dest_urban_in = 0
        row.dest_urban_out = 0
        row.dest_nvrv = 0
        rz_config.element_soil_params = [row]
        rz_config.k_factor = 1.0
        rz_config.k_exdth_factor = 1.0

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
        return rz_config

    def test_rz_v4x_nonponded_crop_types(self, tmp_path: Path):
        """Test v4.x root zone: nonponded crop types extracted from config."""
        rz_file = tmp_path / "rz.dat"
        rz_file.write_text("fake")

        rz_config = self._make_rz_config(tmp_path, version="4.0")

        # Add nonponded crop config
        np_file = tmp_path / "np.dat"
        np_file.write_text("fake")
        rz_config.nonponded_crop_file = np_file

        np_cfg = MagicMock()
        np_cfg.crop_codes = ["CORN", "WHEAT"]
        rd1 = MagicMock()
        rd1.max_root_depth = 3.0
        rd2 = MagicMock()
        rd2.max_root_depth = 2.5
        np_cfg.root_depth_data = [rd1, rd2]
        np_cfg.area_data_file = None
        np_cfg.elemental_area_file = None

        sim_cfg = _sim_config(tmp_path, rootzone_file=str(rz_file))

        with (
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as mock_rz_cls,
            patch("pyiwfm.io.rootzone_v4x.NonPondedCropReaderV4x") as mock_np_cls,
            patch("pyiwfm.io.rootzone_v4x.PondedCropReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.UrbanReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.NativeRiparianReaderV4x"),
        ):
            mock_rz_cls.return_value.read.return_value = rz_config
            mock_np_cls.return_value.read.return_value = np_cfg
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.rootzone is not None
        assert model.rootzone.n_crop_types >= 2

    def test_rz_v4x_ponded_crop_types(self, tmp_path: Path):
        """Test v4.x root zone: ponded crop types extracted from config."""
        rz_file = tmp_path / "rz.dat"
        rz_file.write_text("fake")

        rz_config = self._make_rz_config(tmp_path, version="4.0")

        p_file = tmp_path / "p.dat"
        p_file.write_text("fake")
        rz_config.ponded_crop_file = p_file

        p_cfg = MagicMock()
        p_cfg.root_depths = [1.5, 2.0, 1.0, 0.8, 0.5]
        p_cfg.area_data_file = None
        p_cfg.elemental_area_file = None

        sim_cfg = _sim_config(tmp_path, rootzone_file=str(rz_file))

        with (
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as mock_rz_cls,
            patch("pyiwfm.io.rootzone_v4x.NonPondedCropReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.PondedCropReaderV4x") as mock_p_cls,
            patch("pyiwfm.io.rootzone_v4x.UrbanReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.NativeRiparianReaderV4x"),
        ):
            mock_rz_cls.return_value.read.return_value = rz_config
            mock_p_cls.return_value.read.return_value = p_cfg
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.rootzone is not None
        assert model.rootzone.n_crop_types >= 5

    def test_rz_v4x_ponded_reader_exception(self, tmp_path: Path):
        """v4.x ponded reader exception is logged but does not crash."""
        rz_file = tmp_path / "rz.dat"
        rz_file.write_text("fake")

        rz_config = self._make_rz_config(tmp_path, version="4.0")
        p_file = tmp_path / "p.dat"
        p_file.write_text("fake")
        rz_config.ponded_crop_file = p_file

        sim_cfg = _sim_config(tmp_path, rootzone_file=str(rz_file))

        with (
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as mock_rz_cls,
            patch("pyiwfm.io.rootzone_v4x.NonPondedCropReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.PondedCropReaderV4x") as mock_p_cls,
            patch("pyiwfm.io.rootzone_v4x.UrbanReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.NativeRiparianReaderV4x"),
        ):
            mock_rz_cls.return_value.read.return_value = rz_config
            mock_p_cls.return_value.read.side_effect = ValueError("ponded fail")
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.rootzone is not None

    def test_rz_v4x_urban_reader_exception(self, tmp_path: Path):
        """v4.x urban reader exception is logged but does not crash."""
        rz_file = tmp_path / "rz.dat"
        rz_file.write_text("fake")

        rz_config = self._make_rz_config(tmp_path, version="4.0")
        u_file = tmp_path / "u.dat"
        u_file.write_text("fake")
        rz_config.urban_file = u_file

        sim_cfg = _sim_config(tmp_path, rootzone_file=str(rz_file))

        with (
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as mock_rz_cls,
            patch("pyiwfm.io.rootzone_v4x.NonPondedCropReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.PondedCropReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.UrbanReaderV4x") as mock_u_cls,
            patch("pyiwfm.io.rootzone_v4x.NativeRiparianReaderV4x"),
        ):
            mock_rz_cls.return_value.read.return_value = rz_config
            mock_u_cls.return_value.read.side_effect = ValueError("urban fail")
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.rootzone is not None

    def test_rz_v4x_native_reader_exception(self, tmp_path: Path):
        """v4.x native/riparian reader exception is logged but does not crash."""
        rz_file = tmp_path / "rz.dat"
        rz_file.write_text("fake")

        rz_config = self._make_rz_config(tmp_path, version="4.0")
        n_file = tmp_path / "nv.dat"
        n_file.write_text("fake")
        rz_config.native_veg_file = n_file

        sim_cfg = _sim_config(tmp_path, rootzone_file=str(rz_file))

        with (
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as mock_rz_cls,
            patch("pyiwfm.io.rootzone_v4x.NonPondedCropReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.PondedCropReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.UrbanReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.NativeRiparianReaderV4x") as mock_n_cls,
        ):
            mock_rz_cls.return_value.read.return_value = rz_config
            mock_n_cls.return_value.read.side_effect = ValueError("native fail")
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.rootzone is not None

    def test_rz_v5_sub_file_loading(self, tmp_path: Path):
        """Test v5.0 root zone sub-file loading (lines 1524-1601)."""
        rz_file = tmp_path / "rz.dat"
        rz_file.write_text("fake")

        rz_config = self._make_rz_config(tmp_path, version="5.0")

        np_file = tmp_path / "np.dat"
        np_file.write_text("fake")
        p_file = tmp_path / "p.dat"
        p_file.write_text("fake")
        u_file = tmp_path / "u.dat"
        u_file.write_text("fake")
        nv_file = tmp_path / "nv.dat"
        nv_file.write_text("fake")

        rz_config.nonponded_crop_file = np_file
        rz_config.ponded_crop_file = p_file
        rz_config.urban_file = u_file
        rz_config.native_veg_file = nv_file

        np_cfg = MagicMock()
        np_cfg.crop_codes = ["CORN"]
        np_cfg.root_depth_data = [MagicMock(max_root_depth=3.0)]
        np_cfg.area_data_file = None
        np_cfg.elemental_area_file = None

        p_cfg = MagicMock()
        p_cfg.root_depths = [1.5]
        p_cfg.area_data_file = None
        p_cfg.elemental_area_file = None

        u_cfg = MagicMock()
        u_cfg.area_data_file = None
        u_cfg.elemental_area_file = None

        nr_cfg = MagicMock()
        nr_cfg.area_data_file = None
        nr_cfg.elemental_area_file = None

        sim_cfg = _sim_config(tmp_path, rootzone_file=str(rz_file))

        with (
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as mock_rz_cls,
            patch("pyiwfm.io.rootzone_nonponded.NonPondedCropReader") as mock_np_cls,
            patch("pyiwfm.io.rootzone_ponded.PondedCropReader") as mock_p_cls,
            patch("pyiwfm.io.rootzone_urban.UrbanLandUseReader") as mock_u_cls,
            patch("pyiwfm.io.rootzone_native.NativeRiparianReader") as mock_nr_cls,
        ):
            mock_rz_cls.return_value.read.return_value = rz_config
            mock_np_cls.return_value.read.return_value = np_cfg
            mock_p_cls.return_value.read.return_value = p_cfg
            mock_u_cls.return_value.read.return_value = u_cfg
            mock_nr_cls.return_value.read.return_value = nr_cfg
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.rootzone is not None
        assert model.metadata.get("rootzone_version") == "5.0"

    def test_rz_v5_sub_file_exceptions(self, tmp_path: Path):
        """v5 sub-file readers that throw exceptions are caught gracefully."""
        rz_file = tmp_path / "rz.dat"
        rz_file.write_text("fake")

        rz_config = self._make_rz_config(tmp_path, version="5.0")

        np_file = tmp_path / "np.dat"
        np_file.write_text("fake")
        p_file = tmp_path / "p.dat"
        p_file.write_text("fake")
        u_file = tmp_path / "u.dat"
        u_file.write_text("fake")
        nv_file = tmp_path / "nv.dat"
        nv_file.write_text("fake")

        rz_config.nonponded_crop_file = np_file
        rz_config.ponded_crop_file = p_file
        rz_config.urban_file = u_file
        rz_config.native_veg_file = nv_file

        sim_cfg = _sim_config(tmp_path, rootzone_file=str(rz_file))

        with (
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as mock_rz_cls,
            patch("pyiwfm.io.rootzone_nonponded.NonPondedCropReader") as mock_np_cls,
            patch("pyiwfm.io.rootzone_ponded.PondedCropReader") as mock_p_cls,
            patch("pyiwfm.io.rootzone_urban.UrbanLandUseReader") as mock_u_cls,
            patch("pyiwfm.io.rootzone_native.NativeRiparianReader") as mock_nr_cls,
        ):
            mock_rz_cls.return_value.read.return_value = rz_config
            mock_np_cls.return_value.read.side_effect = ValueError("np fail")
            mock_p_cls.return_value.read.side_effect = ValueError("p fail")
            mock_u_cls.return_value.read.side_effect = ValueError("u fail")
            mock_nr_cls.return_value.read.side_effect = ValueError("nr fail")
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        assert model.rootzone is not None

    def test_rz_area_file_wiring_exception(self, tmp_path: Path):
        """Area file wiring exception is caught (lines 1759-1760)."""
        rz_file = tmp_path / "rz.dat"
        rz_file.write_text("fake")

        rz_config = self._make_rz_config(tmp_path, version="4.0")

        np_file = tmp_path / "np.dat"
        np_file.write_text("fake")
        rz_config.nonponded_crop_file = np_file

        # Make nonponded config that will cause area file wiring to fail
        np_cfg = MagicMock()
        np_cfg.crop_codes = ["CORN"]
        np_cfg.root_depth_data = [MagicMock(max_root_depth=3.0)]
        # Set area_data_file to something that will cause an error when is_absolute called
        area_file = MagicMock()
        area_file.is_absolute.side_effect = ValueError("path error")
        np_cfg.area_data_file = area_file
        np_cfg.elemental_area_file = None

        sim_cfg = _sim_config(tmp_path, rootzone_file=str(rz_file))

        with (
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as mock_rz_cls,
            patch("pyiwfm.io.rootzone_v4x.NonPondedCropReaderV4x") as mock_np_cls,
            patch("pyiwfm.io.rootzone_v4x.PondedCropReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.UrbanReaderV4x"),
            patch("pyiwfm.io.rootzone_v4x.NativeRiparianReaderV4x"),
        ):
            mock_rz_cls.return_value.read.return_value = rz_config
            mock_np_cls.return_value.read.return_value = np_cfg
            model = self._run_sim_with_pp(tmp_path, sim_cfg)

        # Should not crash
        assert model.rootzone is not None


# ===========================================================================
# 6. to_preprocessor – nodes_file None branch (line 1865)
# ===========================================================================


class TestToPreprocessorEdgeCases:
    """Test to_preprocessor edge cases."""

    def test_to_preprocessor_no_nodes_file(self, tmp_path: Path):
        """When save returns config with nodes_file=None, it is excluded."""
        model = IWFMModel(name="TestExport")

        mock_config = MagicMock()
        mock_config.nodes_file = None
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None

        with patch(
            "pyiwfm.io.preprocessor.save_model_to_preprocessor",
            return_value=mock_config,
        ):
            files = model.to_preprocessor(tmp_path)

        assert "nodes" not in files
        assert "elements" in files
        assert "stratigraphy" not in files


# ===========================================================================
# 7. validate_components – small_watersheds and unsaturated_zone
# ===========================================================================


class TestValidateComponentsSWUZ:
    """Test validate_components for small watershed and unsaturated zone."""

    def test_validate_small_watershed_failure(self):
        """Small watershed validation failure is captured."""
        model = IWFMModel(name="Test")
        mock_sw = MagicMock()
        mock_sw.validate.side_effect = ValueError("SW error")
        model.small_watersheds = mock_sw

        warnings = model.validate_components()
        assert len(warnings) == 1
        assert "Small watershed" in warnings[0]

    def test_validate_unsaturated_zone_failure(self):
        """Unsaturated zone validation failure is captured."""
        model = IWFMModel(name="Test")
        mock_uz = MagicMock()
        mock_uz.validate.side_effect = ValueError("UZ error")
        model.unsaturated_zone = mock_uz

        warnings = model.validate_components()
        assert len(warnings) == 1
        assert "Unsaturated zone" in warnings[0]

    def test_validate_both_sw_and_uz_pass(self):
        """Both small watershed and unsaturated zone pass validation."""
        model = IWFMModel(name="Test")
        mock_sw = MagicMock()
        mock_sw.validate.return_value = None
        model.small_watersheds = mock_sw

        mock_uz = MagicMock()
        mock_uz.validate.return_value = None
        model.unsaturated_zone = mock_uz

        warnings = model.validate_components()
        assert len(warnings) == 0


# ===========================================================================
# 8. has_small_watersheds and has_unsaturated_zone
# ===========================================================================


class TestHasComponentsSWUZ:
    """Test has_small_watersheds and has_unsaturated_zone properties."""

    def test_has_small_watersheds_false(self):
        model = IWFMModel(name="Test")
        assert model.has_small_watersheds is False

    def test_has_small_watersheds_true(self):
        model = IWFMModel(name="Test")
        model.small_watersheds = MagicMock()
        assert model.has_small_watersheds is True

    def test_has_unsaturated_zone_false(self):
        model = IWFMModel(name="Test")
        assert model.has_unsaturated_zone is False

    def test_has_unsaturated_zone_true(self):
        model = IWFMModel(name="Test")
        model.unsaturated_zone = MagicMock()
        assert model.has_unsaturated_zone is True


# ===========================================================================
# 9. Summary with small_watersheds and unsaturated_zone populated
# ===========================================================================


class TestSummaryWithSWUZ:
    """Test summary output with SW and UZ components."""

    def test_summary_includes_small_watersheds(self):
        model = IWFMModel(name="SW")
        mock_sw = MagicMock()
        mock_sw.n_watersheds = 3
        model.small_watersheds = mock_sw
        summary = model.summary()
        assert "Watersheds: 3" in summary

    def test_summary_includes_unsaturated_zone(self):
        model = IWFMModel(name="UZ")
        mock_uz = MagicMock()
        mock_uz.n_layers = 5
        mock_uz.n_elements = 100
        model.unsaturated_zone = mock_uz
        summary = model.summary()
        assert "Layers: 5" in summary
        assert "Elements: 100" in summary
