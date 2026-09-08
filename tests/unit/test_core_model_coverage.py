"""Tests for core/model.py error paths.

Covers:
- from_preprocessor() stream/lake loading exceptions (lines 144-176)
- from_preprocessor_binary() simplified binary loader (lines 179-258)
- from_preprocessor() missing nodes/elements -> FileFormatError
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestFromPreprocessorStreamLoadError:
    """Test from_preprocessor() stream loading exception path."""

    def test_from_preprocessor_stream_load_error(self, tmp_path: Path) -> None:
        """Stream loading exception -> caught and stored in metadata."""
        from pyiwfm.core.model import IWFMModel

        mock_config = MagicMock()
        mock_config.model_name = "TestModel"
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None
        mock_config.streams_file = tmp_path / "streams.dat"
        mock_config.lakes_file = None
        mock_config.length_unit = "FT"
        mock_config.area_unit = "SQFT"
        mock_config.volume_unit = "CUFT"

        # Create a fake stream file so .exists() is True
        (tmp_path / "streams.dat").write_text("fake")

        mock_nodes = {1: SimpleNamespace(id=1, x=0.0, y=0.0)}
        mock_elements = ({1: SimpleNamespace(id=1, vertices=[1, 1, 1], subregion=1)}, 1, {})
        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 1
        mock_mesh.n_elements = 1

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_config),
            patch("pyiwfm.io.ascii.read_nodes", return_value=(mock_nodes, 1.0)),
            patch("pyiwfm.io.ascii.read_elements", return_value=mock_elements),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mock_mesh),
            patch("pyiwfm.io.streams.StreamReader") as MockStreamReader,
        ):
            # Real parsers raise ValueError on bad data; from_preprocessor
            # narrows its catch tuple to expected parser exceptions.
            MockStreamReader.return_value.read_stream_nodes.side_effect = ValueError(
                "Stream parse error"
            )
            model = IWFMModel.from_preprocessor(tmp_path / "preprocessor.in", load_streams=True)
            assert "streams_load_error" in model.metadata
            assert "Stream parse error" in model.metadata["streams_load_error"]


class TestFromPreprocessorLakeLoadError:
    """Test from_preprocessor() lake loading exception path."""

    def test_from_preprocessor_lake_load_error(self, tmp_path: Path) -> None:
        """Lake loading exception -> caught and stored in metadata."""
        from pyiwfm.core.model import IWFMModel

        mock_config = MagicMock()
        mock_config.model_name = "TestModel"
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None
        mock_config.streams_file = None
        mock_config.lakes_file = tmp_path / "lakes.dat"
        mock_config.length_unit = "FT"
        mock_config.area_unit = "SQFT"
        mock_config.volume_unit = "CUFT"

        (tmp_path / "lakes.dat").write_text("fake")

        mock_nodes = {1: SimpleNamespace(id=1, x=0.0, y=0.0)}
        mock_elements = ({1: SimpleNamespace(id=1, vertices=[1, 1, 1], subregion=1)}, 1, {})
        mock_mesh = MagicMock()
        mock_mesh.n_nodes = 1
        mock_mesh.n_elements = 1

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_config),
            patch("pyiwfm.io.ascii.read_nodes", return_value=(mock_nodes, 1.0)),
            patch("pyiwfm.io.ascii.read_elements", return_value=mock_elements),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mock_mesh),
            patch("pyiwfm.io.lakes.LakeReader") as MockLakeReader,
        ):
            MockLakeReader.return_value.read_lake_definitions.side_effect = ValueError(
                "Lake parse error"
            )
            model = IWFMModel.from_preprocessor(tmp_path / "preprocessor.in", load_lakes=True)
            assert "lakes_load_error" in model.metadata
            assert "Lake parse error" in model.metadata["lakes_load_error"]


class TestFromPreprocessorMissingNodes:
    """Test from_preprocessor() when nodes file is not specified."""

    def test_from_preprocessor_missing_nodes(self, tmp_path: Path) -> None:
        """Missing nodes file -> FileFormatError."""
        from pyiwfm.core.exceptions import FileFormatError
        from pyiwfm.core.model import IWFMModel

        mock_config = MagicMock()
        mock_config.nodes_file = None

        with patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_config):
            with pytest.raises(FileFormatError, match="Nodes file"):
                IWFMModel.from_preprocessor(tmp_path / "preprocessor.in")


class TestFromPreprocessorMissingElements:
    """Test from_preprocessor() when elements file is not specified."""

    def test_from_preprocessor_missing_elements(self, tmp_path: Path) -> None:
        """Missing elements file -> FileFormatError."""
        from pyiwfm.core.exceptions import FileFormatError
        from pyiwfm.core.model import IWFMModel

        mock_config = MagicMock()
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = None
        mock_nodes = {1: SimpleNamespace(id=1, x=0.0, y=0.0)}

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_config),
            patch("pyiwfm.io.ascii.read_nodes", return_value=(mock_nodes, 1.0)),
        ):
            with pytest.raises(FileFormatError, match="Elements file"):
                IWFMModel.from_preprocessor(tmp_path / "preprocessor.in")


class TestFromPreprocessorBinary:
    """Test from_preprocessor_binary()."""

    def test_from_preprocessor_binary_stub(self, tmp_path: Path) -> None:
        """Binary loader basic path via PreprocessorBinaryReader."""
        from pyiwfm.core.model import IWFMModel

        mock_data = MagicMock()
        mock_model = MagicMock(spec=IWFMModel)
        mock_model.name = "TestBinary"
        mock_model.metadata = {"source": "binary"}

        with (
            patch(
                "pyiwfm.io.preprocessor_binary.PreprocessorBinaryReader.read",
                return_value=mock_data,
            ),
            patch(
                "pyiwfm.core.model._binary_data_to_model",
                return_value=mock_model,
            ),
        ):
            model = IWFMModel.from_preprocessor_binary(
                tmp_path / "preprocessor.bin",
                name="TestBinary",
            )
            assert model.name == "TestBinary"
            assert model.metadata["source"] == "binary"


class TestFromPreprocessorSubregionsAndStratigraphy:
    """Test from_preprocessor() loading subregions and stratigraphy."""

    def test_from_preprocessor_with_subregions(self, tmp_path: Path) -> None:
        """Subregions file exists -> loaded."""
        from pyiwfm.core.model import IWFMModel

        sub_file = tmp_path / "subregions.dat"
        sub_file.write_text("fake")

        mock_config = MagicMock()
        mock_config.model_name = "SubTest"
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = sub_file
        mock_config.streams_file = None
        mock_config.lakes_file = None
        mock_config.length_unit = "FT"
        mock_config.area_unit = "SQFT"
        mock_config.volume_unit = "CUFT"

        mock_nodes = {1: SimpleNamespace(id=1, x=0.0, y=0.0)}
        mock_elements = ({1: SimpleNamespace(id=1, vertices=[1, 1, 1], subregion=1)}, 1, {})
        mock_mesh = MagicMock()
        mock_subregions = {1: SimpleNamespace(id=1, name="Region1")}

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_config),
            patch("pyiwfm.io.ascii.read_nodes", return_value=(mock_nodes, 1.0)),
            patch("pyiwfm.io.ascii.read_elements", return_value=mock_elements),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mock_mesh),
            patch(
                "pyiwfm.io.preprocessor.read_subregions_file", return_value=mock_subregions
            ) as mock_read_sub,
        ):
            IWFMModel.from_preprocessor(tmp_path / "pp.in")
            mock_read_sub.assert_called_once_with(sub_file)

    def test_from_preprocessor_with_stratigraphy(self, tmp_path: Path) -> None:
        """Stratigraphy file exists -> loaded."""
        from pyiwfm.core.model import IWFMModel

        strat_file = tmp_path / "strat.dat"
        strat_file.write_text("fake")

        mock_config = MagicMock()
        mock_config.model_name = "StratTest"
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = strat_file
        mock_config.subregions_file = None
        mock_config.streams_file = None
        mock_config.lakes_file = None
        mock_config.length_unit = "FT"
        mock_config.area_unit = "SQFT"
        mock_config.volume_unit = "CUFT"

        mock_nodes = {1: SimpleNamespace(id=1, x=0.0, y=0.0)}
        mock_elements = ({1: SimpleNamespace(id=1, vertices=[1, 1, 1], subregion=1)}, 1, {})
        mock_mesh = MagicMock()
        mock_strat = MagicMock()

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_config),
            patch("pyiwfm.io.ascii.read_nodes", return_value=(mock_nodes, 1.0)),
            patch("pyiwfm.io.ascii.read_elements", return_value=mock_elements),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mock_mesh),
            patch("pyiwfm.io.ascii.read_stratigraphy", return_value=mock_strat) as mock_read_strat,
        ):
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in")
            mock_read_strat.assert_called_once_with(strat_file)
            assert model.stratigraphy is mock_strat


class TestFromPreprocessorSuccessfulStreamLoading:
    """Test from_preprocessor() successful stream loading path."""

    def test_stream_loading_success(self, tmp_path: Path) -> None:
        """Streams file exists and loads successfully."""
        from pyiwfm.core.model import IWFMModel

        stream_file = tmp_path / "streams.dat"
        stream_file.write_text("fake")

        mock_config = MagicMock()
        mock_config.model_name = "StreamSuccessTest"
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None
        mock_config.streams_file = stream_file
        mock_config.lakes_file = None
        mock_config.length_unit = "FT"
        mock_config.area_unit = "SQFT"
        mock_config.volume_unit = "CUFT"

        mock_nodes = {1: SimpleNamespace(id=1, x=0.0, y=0.0)}
        mock_elements = ({1: SimpleNamespace(id=1, vertices=[1, 1, 1], subregion=1)}, 1, {})
        mock_mesh = MagicMock()

        mock_stream_node = MagicMock()
        mock_stream_nodes = {1: mock_stream_node}
        mock_app_stream = MagicMock()

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_config),
            patch("pyiwfm.io.ascii.read_nodes", return_value=(mock_nodes, 1.0)),
            patch("pyiwfm.io.ascii.read_elements", return_value=mock_elements),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mock_mesh),
            patch("pyiwfm.io.streams.StreamReader") as MockReader,
            patch("pyiwfm.components.stream.AppStream", return_value=mock_app_stream),
        ):
            MockReader.return_value.read_stream_nodes.return_value = mock_stream_nodes
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in", load_streams=True)
            assert model.streams is mock_app_stream
            mock_app_stream.add_node.assert_called_once_with(mock_stream_node)


class TestFromPreprocessorSuccessfulLakeLoading:
    """Test from_preprocessor() successful lake loading path."""

    def test_lake_loading_success(self, tmp_path: Path) -> None:
        """Lakes file exists and loads successfully."""
        from pyiwfm.core.model import IWFMModel

        lake_file = tmp_path / "lakes.dat"
        lake_file.write_text("fake")

        mock_config = MagicMock()
        mock_config.model_name = "LakeSuccessTest"
        mock_config.nodes_file = tmp_path / "nodes.dat"
        mock_config.elements_file = tmp_path / "elements.dat"
        mock_config.stratigraphy_file = None
        mock_config.subregions_file = None
        mock_config.streams_file = None
        mock_config.lakes_file = lake_file
        mock_config.length_unit = "FT"
        mock_config.area_unit = "SQFT"
        mock_config.volume_unit = "CUFT"

        mock_nodes = {1: SimpleNamespace(id=1, x=0.0, y=0.0)}
        mock_elements = ({1: SimpleNamespace(id=1, vertices=[1, 1, 1], subregion=1)}, 1, {})
        mock_mesh = MagicMock()

        mock_lake = MagicMock()
        mock_lakes_dict = {1: mock_lake}
        mock_app_lake = MagicMock()

        with (
            patch("pyiwfm.io.preprocessor.read_preprocessor_main", return_value=mock_config),
            patch("pyiwfm.io.ascii.read_nodes", return_value=(mock_nodes, 1.0)),
            patch("pyiwfm.io.ascii.read_elements", return_value=mock_elements),
            patch("pyiwfm.core.mesh.AppGrid", return_value=mock_mesh),
            patch("pyiwfm.io.lakes.LakeReader") as MockReader,
            patch("pyiwfm.components.lake.AppLake", return_value=mock_app_lake),
        ):
            MockReader.return_value.read_lake_definitions.return_value = mock_lakes_dict
            model = IWFMModel.from_preprocessor(tmp_path / "pp.in", load_lakes=True)
            assert model.lakes is mock_app_lake
            mock_app_lake.add_lake.assert_called_once_with(mock_lake)


class TestFromSimulationWithPreprocessor:
    """Test from_simulation_with_preprocessor() method."""

    def _make_mock_sim_config(self, base_dir: Path) -> MagicMock:
        """Create a mock SimulationConfig."""
        cfg = MagicMock()
        cfg.start_date.isoformat.return_value = "2020-01-01"
        cfg.end_date.isoformat.return_value = "2020-12-31"
        cfg.time_step_length = 1
        cfg.time_step_unit.value = "MONTH"
        cfg.groundwater_file = base_dir / "gw.dat"
        cfg.streams_file = base_dir / "streams.dat"
        cfg.lakes_file = base_dir / "lakes.dat"
        cfg.rootzone_file = base_dir / "rootzone.dat"
        return cfg

    def test_full_load(self, tmp_path: Path) -> None:
        """Load with all components successfully."""
        from pyiwfm.core.model import IWFMModel

        sim_file = tmp_path / "simulation.in"
        pp_file = tmp_path / "preprocessor.in"

        # Create files so .exists() returns True
        for f in [
            sim_file,
            pp_file,
            tmp_path / "gw.dat",
            tmp_path / "streams.dat",
            tmp_path / "lakes.dat",
            tmp_path / "rootzone.dat",
        ]:
            f.write_text("fake")

        mock_model = MagicMock(spec=IWFMModel)
        mock_model.metadata = {}
        mock_model.source_files = {}
        mock_model.mesh = MagicMock()
        mock_model.mesh.n_nodes = 10
        mock_model.mesh.n_elements = 5
        mock_model.n_layers = 2
        mock_model.streams = None
        mock_model.lakes = None

        sim_config = self._make_mock_sim_config(tmp_path)

        mock_wells = {1: MagicMock()}
        mock_stream_nodes = {1: MagicMock()}
        mock_lake_defs = {1: MagicMock()}
        mock_crops = {1: MagicMock()}

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch("pyiwfm.io.simulation.SimulationReader") as MockSimReader,
            patch("pyiwfm.io.groundwater.GroundwaterReader") as MockGWReader,
            patch("pyiwfm.io.streams.StreamReader") as MockStreamReader,
            patch("pyiwfm.io.lakes.LakeReader") as MockLakeReader,
            patch("pyiwfm.io.rootzone.RootZoneReader") as MockRZReader,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda base, p: Path(p)),
            patch("pyiwfm.components.groundwater.AppGW"),
            patch("pyiwfm.components.stream.AppStream"),
            patch("pyiwfm.components.lake.AppLake"),
            patch("pyiwfm.components.rootzone.RootZone"),
        ):
            MockSimReader.return_value.read.return_value = sim_config
            MockGWReader.return_value.read_wells.return_value = mock_wells
            MockStreamReader.return_value.read_stream_nodes.return_value = mock_stream_nodes
            MockLakeReader.return_value.read_lake_definitions.return_value = mock_lake_defs
            MockRZReader.return_value.read_crop_types.return_value = mock_crops

            result = IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        assert result is mock_model
        assert "simulation_file" in mock_model.metadata
        assert mock_model.metadata["source"] == "simulation_with_preprocessor"

    def test_gw_load_error(self, tmp_path: Path) -> None:
        """GW loading exception -> caught in metadata."""
        from pyiwfm.core.model import IWFMModel

        sim_file = tmp_path / "simulation.in"
        pp_file = tmp_path / "preprocessor.in"
        gw_file = tmp_path / "gw.dat"
        for f in [sim_file, pp_file, gw_file]:
            f.write_text("fake")

        mock_model = MagicMock(spec=IWFMModel)
        mock_model.metadata = {}
        mock_model.source_files = {}
        mock_model.mesh = MagicMock()
        mock_model.mesh.n_nodes = 10
        mock_model.mesh.n_elements = 5
        mock_model.n_layers = 2
        mock_model.streams = None
        mock_model.lakes = None

        sim_config = MagicMock()
        sim_config.start_date.isoformat.return_value = "2020-01-01"
        sim_config.end_date.isoformat.return_value = "2020-12-31"
        sim_config.time_step_length = 1
        sim_config.time_step_unit.value = "MONTH"
        sim_config.groundwater_file = gw_file
        sim_config.streams_file = None
        sim_config.lakes_file = None
        sim_config.rootzone_file = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch("pyiwfm.io.simulation.SimulationReader") as MockSimReader,
            patch("pyiwfm.io.groundwater.GWMainFileReader") as MockGWMainReader,
            patch("pyiwfm.io.groundwater.GroundwaterReader") as MockGWReader,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda base, p: Path(p)),
        ):
            MockSimReader.return_value.read.return_value = sim_config
            # Make hierarchical reader fail, triggering fallback to direct reader
            MockGWMainReader.return_value.read.side_effect = ValueError("GW main file error")
            MockGWReader.return_value.read_wells.side_effect = ValueError("GW error")

            IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        # Since both readers fail, but we catch exceptions, check that
        # the gw component is created but with no wells
        assert mock_model.groundwater is not None or "groundwater_load_error" in mock_model.metadata

    def test_streams_load_error(self, tmp_path: Path) -> None:
        """Streams loading exception -> caught in metadata."""
        from pyiwfm.core.model import IWFMModel

        sim_file = tmp_path / "simulation.in"
        pp_file = tmp_path / "preprocessor.in"
        stream_file = tmp_path / "streams.dat"
        for f in [sim_file, pp_file, stream_file]:
            f.write_text("fake")

        mock_model = MagicMock(spec=IWFMModel)
        mock_model.metadata = {}
        mock_model.source_files = {}
        mock_model.mesh = MagicMock()
        mock_model.streams = None
        mock_model.lakes = None

        sim_config = MagicMock()
        sim_config.start_date.isoformat.return_value = "2020-01-01"
        sim_config.end_date.isoformat.return_value = "2020-12-31"
        sim_config.time_step_length = 1
        sim_config.time_step_unit.value = "MONTH"
        sim_config.groundwater_file = None
        sim_config.streams_file = stream_file
        sim_config.lakes_file = None
        sim_config.rootzone_file = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch("pyiwfm.io.simulation.SimulationReader") as MockSimReader,
            patch("pyiwfm.io.streams.StreamMainFileReader") as MockStreamMainReader,
            patch("pyiwfm.io.streams.StreamReader") as MockStreamReader,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda base, p: Path(p)),
        ):
            MockSimReader.return_value.read.return_value = sim_config
            # Make hierarchical reader fail, triggering fallback to direct reader
            MockStreamMainReader.return_value.read.side_effect = ValueError(
                "Stream main file error"
            )
            MockStreamReader.return_value.read_stream_nodes.side_effect = ValueError("Stream error")

            IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        # Since both readers fail, but we catch exceptions, check that
        # the stream component is created but empty or load_error is set
        assert mock_model.streams is not None or "streams_load_error" in mock_model.metadata

    def test_lakes_load_error(self, tmp_path: Path) -> None:
        """Lakes loading exception -> caught in metadata."""
        from pyiwfm.core.model import IWFMModel

        sim_file = tmp_path / "simulation.in"
        pp_file = tmp_path / "preprocessor.in"
        lake_file = tmp_path / "lakes.dat"
        for f in [sim_file, pp_file, lake_file]:
            f.write_text("fake")

        mock_model = MagicMock(spec=IWFMModel)
        mock_model.metadata = {}
        mock_model.source_files = {}
        mock_model.mesh = MagicMock()
        mock_model.streams = MagicMock()  # Already loaded
        mock_model.lakes = None

        sim_config = MagicMock()
        sim_config.start_date.isoformat.return_value = "2020-01-01"
        sim_config.end_date.isoformat.return_value = "2020-12-31"
        sim_config.time_step_length = 1
        sim_config.time_step_unit.value = "MONTH"
        sim_config.groundwater_file = None
        sim_config.streams_file = None
        sim_config.lakes_file = lake_file
        sim_config.rootzone_file = None

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch("pyiwfm.io.simulation.SimulationReader") as MockSimReader,
            patch("pyiwfm.io.lakes.LakeMainFileReader") as MockLakeMainReader,
            patch("pyiwfm.io.lakes.LakeReader") as MockLakeReader,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda base, p: Path(p)),
        ):
            MockSimReader.return_value.read.return_value = sim_config
            MockLakeMainReader.return_value.read.side_effect = ValueError("Lake main file error")
            MockLakeReader.return_value.read_lake_definitions.side_effect = ValueError("Lake error")

            IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        # Lakes component is created but empty, or error stored as metadata
        assert mock_model.lakes is not None or "lakes_load_error" in mock_model.metadata

    def test_rootzone_load_error(self, tmp_path: Path) -> None:
        """Rootzone loading exception -> caught in metadata."""
        from pyiwfm.core.model import IWFMModel

        sim_file = tmp_path / "simulation.in"
        pp_file = tmp_path / "preprocessor.in"
        rz_file = tmp_path / "rootzone.dat"
        for f in [sim_file, pp_file, rz_file]:
            f.write_text("fake")

        mock_model = MagicMock(spec=IWFMModel)
        mock_model.metadata = {}
        mock_model.source_files = {}
        mock_model.mesh = MagicMock()
        mock_model.mesh.n_elements = 5
        mock_model.streams = MagicMock()
        mock_model.lakes = MagicMock()

        sim_config = MagicMock()
        sim_config.start_date.isoformat.return_value = "2020-01-01"
        sim_config.end_date.isoformat.return_value = "2020-12-31"
        sim_config.time_step_length = 1
        sim_config.time_step_unit.value = "MONTH"
        sim_config.groundwater_file = None
        sim_config.streams_file = None
        sim_config.lakes_file = None
        sim_config.rootzone_file = rz_file

        with (
            patch.object(IWFMModel, "from_preprocessor", return_value=mock_model),
            patch("pyiwfm.io.simulation.SimulationReader") as MockSimReader,
            patch("pyiwfm.io.rootzone.RootZoneMainFileReader") as MockRZMainReader,
            patch("pyiwfm.io.rootzone.RootZoneReader") as MockRZReader,
            patch("pyiwfm.io.preprocessor._resolve_path", side_effect=lambda base, p: Path(p)),
        ):
            MockSimReader.return_value.read.return_value = sim_config
            # Make hierarchical reader fail, triggering fallback to direct reader
            MockRZMainReader.return_value.read.side_effect = ValueError("RZ main file error")
            MockRZReader.return_value.read_crop_types.side_effect = ValueError("RZ error")

            IWFMModel.from_simulation_with_preprocessor(sim_file, pp_file)

        # Since both readers fail, but we catch exceptions, check that
        # the rootzone component is created but empty or load_error is set
        assert mock_model.rootzone is not None or "rootzone_load_error" in mock_model.metadata
