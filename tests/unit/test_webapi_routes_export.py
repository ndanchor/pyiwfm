"""Comprehensive tests for the FastAPI export routes.

Covers all four endpoints in routes/export.py:
  - GET /api/export/heads-csv
  - GET /api/export/mesh-geojson
  - GET /api/export/budget-csv
  - GET /api/export/hydrograph-csv

Tests every branch including success, error, and edge cases for 95%+ coverage.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not available")
pydantic = pytest.importorskip("pydantic", reason="Pydantic not available")

from fastapi.testclient import TestClient  # noqa: E402

from pyiwfm.core.mesh import AppGrid, Element, Node  # noqa: E402
from pyiwfm.visualization.webapi.config import model_state  # noqa: E402
from pyiwfm.visualization.webapi.server import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid():
    """Create a simple 4-node quad grid for testing."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0),
        2: Node(id=2, x=100.0, y=0.0),
        3: Node(id=3, x=100.0, y=100.0),
        4: Node(id=4, x=0.0, y=100.0),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 3, 4), subregion=1),
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    grid.compute_areas()
    return grid


def _make_mock_model(**kwargs):
    """Create a minimal mock IWFMModel."""
    model = MagicMock()
    model.name = "TestModel"
    model.grid = _make_grid()
    model.metadata = kwargs.get("metadata", {})
    model.has_streams = kwargs.get("with_streams", False)
    model.has_lakes = False
    model.n_nodes = 4
    model.n_elements = 1
    model.n_layers = 2
    model.n_lakes = 0
    model.n_stream_nodes = 0

    # Stratigraphy
    if kwargs.get("with_stratigraphy", True):
        strat = MagicMock()
        strat.n_layers = 2
        strat.n_nodes = 4
        strat.gs_elev = np.array([100.0, 100.0, 100.0, 100.0])
        strat.top_elev = np.full((4, 2), 100.0)
        strat.top_elev[:, 1] = 50.0
        strat.bottom_elev = np.zeros((4, 2))
        strat.bottom_elev[:, 0] = 50.0
        strat.bottom_elev[:, 1] = 0.0
        model.stratigraphy = strat
    else:
        model.stratigraphy = None

    # Streams
    if kwargs.get("with_streams", False):
        streams = MagicMock()
        streams.n_nodes = 2
        sn1 = MagicMock()
        sn1.id = 1
        sn1.groundwater_node = 1
        sn2 = MagicMock()
        sn2.id = 2
        sn2.groundwater_node = 2
        reach = MagicMock()
        reach.id = 1
        reach.stream_nodes = [sn1, sn2]
        streams.reaches = [reach]
        model.streams = streams
        model.has_streams = True
        model.n_stream_nodes = 2
    else:
        model.streams = None

    return model


def _make_mock_head_loader(n_frames=5, n_nodes=4, n_layers=2):
    """Create a mock head data loader."""
    loader = MagicMock()
    loader.n_frames = n_frames
    base_time = datetime(2020, 1, 1)
    loader.times = [base_time + timedelta(days=30 * i) for i in range(n_frames)]

    def get_frame(timestep):
        return np.random.default_rng(42).random((n_nodes, n_layers)) * 100.0

    loader.get_frame = get_frame
    return loader


def _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=10):
    """Create a mock GW hydrograph reader."""
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    base_time = datetime(2020, 1, 1)

    def get_time_series(col_idx):
        times_list = [(base_time + timedelta(days=30 * i)).isoformat() for i in range(n_timesteps)]
        values_list = (np.arange(n_timesteps, dtype=float) + col_idx).tolist()
        return times_list, values_list

    reader.get_time_series = get_time_series
    return reader


def _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=10):
    """Create a mock stream hydrograph reader."""
    reader = MagicMock()
    reader.n_columns = n_columns
    reader.n_timesteps = n_timesteps
    reader.hydrograph_ids = [1, 2, 3]
    base_time = datetime(2020, 1, 1)

    def find_column_by_node_id(node_id):
        if node_id in reader.hydrograph_ids:
            return reader.hydrograph_ids.index(node_id)
        return None

    reader.find_column_by_node_id = find_column_by_node_id

    def get_time_series(col_idx):
        times_list = [(base_time + timedelta(days=30 * i)).isoformat() for i in range(n_timesteps)]
        values_list = (np.arange(n_timesteps, dtype=float) * 10 + col_idx).tolist()
        return times_list, values_list

    reader.get_time_series = get_time_series
    return reader


def _make_mock_budget_reader(
    locations=None, columns=None, n_timesteps=5, use_months=True, has_start_dt=True
):
    """Create a mock budget reader."""
    reader = MagicMock()

    if locations is None:
        locations = ["Region 1", "Region 2"]
    if columns is None:
        columns = ["Deep Percolation", "Gain from Stream", "Net Change"]

    reader.locations = locations
    reader.descriptor = "GW Budget"

    def get_column_headers(loc):
        return columns

    reader.get_column_headers = get_column_headers

    def get_values(loc, col_indices=None):
        n_cols = len(columns)
        if col_indices is not None:
            n_cols = len(col_indices)
        times_arr = np.arange(n_timesteps)
        values_arr = np.random.default_rng(42).random((n_timesteps, n_cols)) * 1000
        return times_arr, values_arr

    reader.get_values = get_values

    # Mock header for timestamp info
    header = MagicMock()
    ts = MagicMock()
    if has_start_dt:
        ts.start_datetime = datetime(2020, 1, 1)
    else:
        ts.start_datetime = None
    ts.delta_t_minutes = 43200  # ~30 days in minutes
    ts.unit = "1MON" if use_months else "1DAY"
    header.timestep = ts
    reader.header = header

    return reader


def _reset_model_state():
    """Reset the global model_state to a clean state."""
    model_state._model = None
    model_state._mesh_3d = None
    model_state._mesh_surface = None
    model_state._surface_json_data = None
    model_state._bounds = None
    model_state._pv_mesh_3d = None
    model_state._layer_surface_cache = {}
    model_state._crs = "+proj=utm +zone=10 +datum=NAD83 +units=us-ft +no_defs"
    model_state._transformer = None
    model_state._geojson_cache = {}
    model_state._head_loader = None
    model_state._gw_hydrograph_reader = None
    model_state._stream_hydrograph_reader = None
    model_state._budget_readers = {}
    model_state._observations = {}
    model_state._results_dir = None
    # Restore any monkey-patched methods back to the class originals
    for attr in (
        "get_budget_reader",
        "get_available_budgets",
        "reproject_coords",
        "get_stream_reach_boundaries",
        "get_head_loader",
        "get_gw_hydrograph_reader",
        "get_stream_hydrograph_reader",
        "get_area_manager",
        "get_subsidence_reader",
    ):
        if attr in model_state.__dict__:
            del model_state.__dict__[attr]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded."""
    _reset_model_state()
    app = create_app()
    yield TestClient(app)
    _reset_model_state()


@pytest.fixture()
def client_with_model():
    """TestClient with a basic mock model loaded."""
    _reset_model_state()
    model = _make_mock_model(with_streams=True, with_stratigraphy=True)
    model_state._model = model
    app = create_app()
    yield TestClient(app), model
    _reset_model_state()


# ===========================================================================
# 1. GET /api/export/heads-csv
# ===========================================================================


class TestExportHeadsCsv:
    """Tests for GET /api/export/heads-csv."""

    def test_success_default_params(self, client_with_model):
        """Export heads with default timestep=0, layer=1 returns valid CSV."""
        client, _ = client_with_model
        loader = _make_mock_head_loader(n_frames=5, n_nodes=4, n_layers=2)
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment; filename=" in resp.headers["content-disposition"]
        assert "heads_ts0_layer1" in resp.headers["content-disposition"]
        # Verify date is in filename (20200101 for timestep 0)
        assert "20200101" in resp.headers["content-disposition"]
        # Parse CSV content
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        assert rows[0] == ["node_id", "head_ft"]
        assert len(rows) == 5  # header + 4 nodes
        # Check node IDs are 1-based
        assert rows[1][0] == "1"
        assert rows[4][0] == "4"

    def test_success_explicit_timestep_and_layer(self, client_with_model):
        """Export heads with explicit timestep=2, layer=2."""
        client, _ = client_with_model
        loader = _make_mock_head_loader(n_frames=5, n_nodes=4, n_layers=2)
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv?timestep=2&layer=2")
        assert resp.status_code == 200
        assert "heads_ts2_layer2" in resp.headers["content-disposition"]
        # timestep 2 -> 2020-01-01 + 60 days -> 2020-03-01
        assert "20200301" in resp.headers["content-disposition"]

    def test_no_head_data_returns_404(self, client_with_model):
        """Return 404 when head loader is None."""
        client, _ = client_with_model
        with patch.object(model_state, "get_head_loader", return_value=None):
            resp = client.get("/api/export/heads-csv")
        assert resp.status_code == 404
        assert "No head data available" in resp.json()["detail"]

    def test_timestep_out_of_range_returns_400(self, client_with_model):
        """Return 400 when timestep >= n_frames."""
        client, _ = client_with_model
        loader = _make_mock_head_loader(n_frames=3)
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv?timestep=5")
        assert resp.status_code == 400
        assert "Timestep 5 out of range" in resp.json()["detail"]

    def test_layer_out_of_range_returns_400(self, client_with_model):
        """Return 400 when layer > n_layers in the frame."""
        client, _ = client_with_model
        loader = _make_mock_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv?layer=5")
        assert resp.status_code == 400
        assert "Layer 5 out of range" in resp.json()["detail"]

    def test_timestep_at_boundary(self, client_with_model):
        """Timestep exactly at n_frames returns 400."""
        client, _ = client_with_model
        loader = _make_mock_head_loader(n_frames=3)
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv?timestep=3")
        assert resp.status_code == 400

    def test_layer_at_max(self, client_with_model):
        """Layer exactly at n_layers succeeds (1-based, so layer=2 -> index=1)."""
        client, _ = client_with_model
        loader = _make_mock_head_loader(n_frames=3, n_nodes=4, n_layers=2)
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv?layer=2")
        assert resp.status_code == 200
        assert "layer2" in resp.headers["content-disposition"]

    def test_csv_values_are_rounded(self, client_with_model):
        """Head values in CSV are rounded to 3 decimal places."""
        client, _ = client_with_model
        loader = _make_mock_head_loader(n_frames=1, n_nodes=2, n_layers=1)
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv")
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        for row in rows[1:]:
            val_str = row[1]
            # Verify at most 3 decimal places
            if "." in val_str:
                decimals = val_str.split(".")[1]
                assert len(decimals) <= 3

    def test_filename_without_date_when_times_short(self, client_with_model):
        """When timestep >= len(loader.times), dt is None and no date in filename."""
        client, _ = client_with_model
        loader = _make_mock_head_loader(n_frames=5, n_nodes=4, n_layers=2)
        # Make times list shorter than n_frames
        loader.times = []
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv?timestep=0")
        assert resp.status_code == 200
        disposition = resp.headers["content-disposition"]
        # Filename should be heads_ts0_layer1.csv with no date
        assert "heads_ts0_layer1.csv" in disposition

    def test_negative_timestep_rejected_by_query_validation(self, client_with_model):
        """FastAPI validates timestep >= 0 via Query(ge=0)."""
        client, _ = client_with_model
        loader = _make_mock_head_loader()
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv?timestep=-1")
        assert resp.status_code == 422  # Validation error

    def test_layer_zero_rejected_by_query_validation(self, client_with_model):
        """FastAPI validates layer >= 1 via Query(ge=1)."""
        client, _ = client_with_model
        loader = _make_mock_head_loader()
        with patch.object(model_state, "get_head_loader", return_value=loader):
            resp = client.get("/api/export/heads-csv?layer=0")
        assert resp.status_code == 422


# ===========================================================================
# 2. GET /api/export/mesh-geojson
# ===========================================================================


class TestExportMeshGeojson:
    """Tests for GET /api/export/mesh-geojson."""

    def test_no_model_returns_404(self, client_no_model):
        """Return 404 when no model is loaded."""
        resp = client_no_model.get("/api/export/mesh-geojson")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    def test_success_returns_geojson(self, client_with_model):
        """Returns valid GeoJSON with correct content type and disposition."""
        client, _ = client_with_model
        mock_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"element_id": 1},
                }
            ],
        }
        with patch(
            "pyiwfm.visualization.webapi.routes.mesh.get_mesh_geojson",
            return_value=mock_geojson,
        ):
            resp = client.get("/api/export/mesh-geojson")
        assert resp.status_code == 200
        assert "application/geo+json" in resp.headers["content-type"]
        assert "mesh_layer1.geojson" in resp.headers["content-disposition"]
        data = json.loads(resp.text)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1

    def test_explicit_layer_in_filename(self, client_with_model):
        """Layer parameter appears in the download filename."""
        client, _ = client_with_model
        mock_geojson = {"type": "FeatureCollection", "features": []}
        with patch(
            "pyiwfm.visualization.webapi.routes.mesh.get_mesh_geojson",
            return_value=mock_geojson,
        ):
            resp = client.get("/api/export/mesh-geojson?layer=3")
        assert resp.status_code == 200
        assert "mesh_layer3.geojson" in resp.headers["content-disposition"]

    def test_get_mesh_geojson_raises_returns_500(self, client_with_model):
        """Return 500 when get_mesh_geojson raises an exception."""
        client, _ = client_with_model
        with patch(
            "pyiwfm.visualization.webapi.routes.mesh.get_mesh_geojson",
            side_effect=ValueError("Mesh computation failed"),
        ):
            resp = client.get("/api/export/mesh-geojson")
        assert resp.status_code == 500
        assert "Mesh computation failed" in resp.json()["detail"]

    def test_layer_zero_rejected_by_query_validation(self, client_with_model):
        """FastAPI validates layer >= 1."""
        client, _ = client_with_model
        resp = client.get("/api/export/mesh-geojson?layer=0")
        assert resp.status_code == 422


# ===========================================================================
# 2b. GET /api/export/geopackage
# ===========================================================================


class TestExportGeopackage:
    """Tests for GET /api/export/geopackage."""

    def test_no_model_returns_404(self, client_no_model):
        """Return 404 when no model is loaded."""
        resp = client_no_model.get("/api/export/geopackage")
        assert resp.status_code == 404

    def test_defaults_to_model_metadata_length_unit(self, client_with_model):
        """The model's own resolved simulation_length_unit is used when
        the model_length_unit query param is omitted."""
        client, model = client_with_model
        model.metadata["simulation_length_unit"] = "FEET"

        mock_exporter = MagicMock()
        mock_exporter.resolved_model_length_unit = "FEET"
        mock_exporter.export_geopackage.side_effect = lambda path, **kw: (
            __import__("pathlib").Path(path).write_bytes(b"fake-gpkg")
        )

        with patch(
            "pyiwfm.visualization.gis_export.GISExporter", return_value=mock_exporter
        ) as mock_cls:
            resp = client.get("/api/export/geopackage")

        assert resp.status_code == 200
        assert resp.headers["X-Model-Length-Unit"] == "FEET"
        assert "X-Model-Length-Unit-Ambiguous" not in resp.headers
        _, kwargs = mock_cls.call_args
        assert kwargs["model_length_unit"] == "FEET"

    def test_query_param_overrides_model_metadata(self, client_with_model):
        """An explicit model_length_unit query param takes priority over
        the model's own metadata."""
        client, model = client_with_model
        model.metadata["simulation_length_unit"] = "FEET"

        mock_exporter = MagicMock()
        mock_exporter.resolved_model_length_unit = "METERS"
        mock_exporter.export_geopackage.side_effect = lambda path, **kw: (
            __import__("pathlib").Path(path).write_bytes(b"fake-gpkg")
        )

        with patch(
            "pyiwfm.visualization.gis_export.GISExporter", return_value=mock_exporter
        ) as mock_cls:
            resp = client.get("/api/export/geopackage?model_length_unit=METERS")

        assert resp.status_code == 200
        assert resp.headers["X-Model-Length-Unit"] == "METERS"
        _, kwargs = mock_cls.call_args
        assert kwargs["model_length_unit"] == "METERS"

    def test_ambiguous_unit_flagged_in_response_header(self, client_with_model):
        """When GISExporter can't resolve the model's unit, the response
        still succeeds but flags the ambiguity via a response header
        rather than silently guessing."""
        client, _ = client_with_model

        mock_exporter = MagicMock()
        mock_exporter.resolved_model_length_unit = None
        mock_exporter.export_geopackage.side_effect = lambda path, **kw: (
            __import__("pathlib").Path(path).write_bytes(b"fake-gpkg")
        )

        with patch("pyiwfm.visualization.gis_export.GISExporter", return_value=mock_exporter):
            resp = client.get("/api/export/geopackage")

        assert resp.status_code == 200
        assert resp.headers["X-Model-Length-Unit-Ambiguous"] == "true"
        assert "X-Model-Length-Unit" not in resp.headers

    def test_export_failure_returns_500(self, client_with_model):
        """Return 500 when the underlying export raises."""
        client, _ = client_with_model

        mock_exporter = MagicMock()
        mock_exporter.resolved_model_length_unit = "FEET"
        mock_exporter.export_geopackage.side_effect = RuntimeError("disk full")

        with patch("pyiwfm.visualization.gis_export.GISExporter", return_value=mock_exporter):
            resp = client.get("/api/export/geopackage")

        assert resp.status_code == 500
        assert "disk full" in resp.json()["detail"]


# ===========================================================================
# 3. GET /api/export/budget-csv
# ===========================================================================


class TestExportBudgetCsv:
    """Tests for GET /api/export/budget-csv."""

    def test_no_budget_reader_returns_404(self, client_with_model):
        """Return 404 when budget reader is None."""
        client, _ = client_with_model
        with patch.object(model_state, "get_budget_reader", return_value=None):
            resp = client.get("/api/export/budget-csv?budget_type=gw")
        assert resp.status_code == 404
        assert "Budget type 'gw' not available" in resp.json()["detail"]

    def test_missing_budget_type_returns_422(self, client_with_model):
        """budget_type is required; omitting it returns 422."""
        client, _ = client_with_model
        resp = client.get("/api/export/budget-csv")
        assert resp.status_code == 422

    def test_success_with_monthly_timestep(self, client_with_model):
        """Export budget CSV with monthly (MON) timestep uses relativedelta."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(use_months=True, has_start_dt=True, n_timesteps=3)
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        assert rows[0][0] == "datetime"
        assert rows[0][1:] == ["Deep Percolation", "Gain from Stream", "Net Change"]
        assert len(rows) == 4  # header + 3 timesteps
        # Monthly: 2020-01-01, 2020-02-01, 2020-03-01
        assert rows[1][0] == "2020-01-01T00:00:00"
        assert rows[2][0] == "2020-02-01T00:00:00"
        assert rows[3][0] == "2020-03-01T00:00:00"

    def test_success_with_non_monthly_timestep(self, client_with_model):
        """Export budget CSV with non-MON timestep uses timedelta."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(use_months=False, has_start_dt=True, n_timesteps=3)
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=lwu")
        assert resp.status_code == 200
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        # Non-monthly: uses timedelta with delta_t_minutes=43200 (~30 days)
        assert rows[1][0] == "2020-01-01T00:00:00"
        # Second timestep: 2020-01-01 + 43200 minutes = 2020-01-31T00:00:00
        assert "2020-01-31" in rows[2][0]

    def test_success_without_start_datetime(self, client_with_model):
        """When start_datetime is None, use raw time array values as strings."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(has_start_dt=False, n_timesteps=3)
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=rz")
        assert resp.status_code == 200
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        # Without start_datetime, times come from times_arr as strings
        assert rows[1][0] == "0"
        assert rows[2][0] == "1"
        assert rows[3][0] == "2"

    def test_success_with_location_param(self, client_with_model):
        """Export budget CSV with explicit location parameter."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(locations=["Region 1", "Region 2"], n_timesteps=2)
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw&location=Region%201")
        assert resp.status_code == 200
        assert "budget_gw_Region_1.csv" in resp.headers["content-disposition"]

    def test_default_location_uses_first(self, client_with_model):
        """When location is empty, filename uses reader.locations[0]."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(locations=["My Region/A", "Other"], n_timesteps=2)
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw")
        assert resp.status_code == 200
        # Spaces and slashes are replaced with underscores
        assert "budget_gw_My_Region_A.csv" in resp.headers["content-disposition"]

    def test_get_values_raises_key_error_returns_400(self, client_with_model):
        """Return 400 when reader.get_values raises KeyError."""
        client, _ = client_with_model
        budget_reader = MagicMock()
        budget_reader.get_values.side_effect = KeyError("Location not found")
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw&location=bad")
        assert resp.status_code == 400

    def test_get_values_raises_index_error_returns_400(self, client_with_model):
        """Return 400 when reader.get_values raises IndexError."""
        client, _ = client_with_model
        budget_reader = MagicMock()
        budget_reader.get_values.side_effect = IndexError("Index out of range")
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw&location=99")
        assert resp.status_code == 400

    def test_csv_values_are_rounded_to_4_decimals(self, client_with_model):
        """Budget values in CSV are rounded to 4 decimal places."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(n_timesteps=2)
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw")
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        for row in rows[1:]:
            for val_str in row[1:]:  # skip datetime column
                if "." in val_str:
                    decimals = val_str.split(".")[1]
                    assert len(decimals) <= 4

    def test_location_with_special_chars_sanitized(self, client_with_model):
        """Spaces and slashes in location name are replaced in filename."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(n_timesteps=1)
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw&location=My%20Region/Area")
        assert resp.status_code == 200
        assert "My_Region_Area" in resp.headers["content-disposition"]

    def test_unit_none_uses_non_monthly_path(self, client_with_model):
        """When ts.unit is None, use_months is False (no MON check)."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(n_timesteps=2, has_start_dt=True)
        # Set unit to None
        budget_reader.header.timestep.unit = None
        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw")
        assert resp.status_code == 200
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        # Should use timedelta path, not relativedelta
        assert rows[1][0] == "2020-01-01T00:00:00"

    def test_empty_location_passes_zero_to_reader(self, client_with_model):
        """When location param is empty string, loc=0 is passed to get_values."""
        client, _ = client_with_model
        budget_reader = _make_mock_budget_reader(n_timesteps=1)

        call_args_capture = []
        original_get_values = budget_reader.get_values

        def capturing_get_values(loc, **kwargs):
            call_args_capture.append(loc)
            return original_get_values(loc, **kwargs)

        budget_reader.get_values = capturing_get_values

        with patch.object(model_state, "get_budget_reader", return_value=budget_reader):
            resp = client.get("/api/export/budget-csv?budget_type=gw")
        assert resp.status_code == 200
        assert call_args_capture[0] == 0


# ===========================================================================
# 4. GET /api/export/hydrograph-csv
# ===========================================================================


class TestExportHydrographCsv:
    """Tests for GET /api/export/hydrograph-csv."""

    # --- No model loaded ---

    def test_no_model_returns_404(self, client_no_model):
        """Return 404 when no model is loaded."""
        resp = client_no_model.get("/api/export/hydrograph-csv?type=gw&location_id=1")
        assert resp.status_code == 404
        assert "No model loaded" in resp.json()["detail"]

    # --- Missing required params ---

    def test_missing_type_returns_422(self, client_with_model):
        """type is required; omitting it returns 422."""
        client, _ = client_with_model
        resp = client.get("/api/export/hydrograph-csv?location_id=1")
        assert resp.status_code == 422

    def test_missing_location_id_returns_422(self, client_with_model):
        """location_id is required; omitting it returns 422."""
        client, _ = client_with_model
        resp = client.get("/api/export/hydrograph-csv?type=gw")
        assert resp.status_code == 422

    # --- Unknown type ---

    def test_unknown_type_returns_400(self, client_with_model):
        """Return 400 for unknown hydrograph type."""
        client, _ = client_with_model
        resp = client.get("/api/export/hydrograph-csv?type=unknown&location_id=1")
        assert resp.status_code == 400
        assert "Unknown type: unknown" in resp.json()["detail"]

    # --- GW hydrograph tests ---

    def test_gw_no_reader_returns_404(self, client_with_model):
        """Return 404 when GW hydrograph reader is None."""
        client, _ = client_with_model
        with patch.object(model_state, "get_gw_hydrograph_reader", return_value=None):
            resp = client.get("/api/export/hydrograph-csv?type=gw&location_id=1")
        assert resp.status_code == 404
        assert "No GW hydrograph data available" in resp.json()["detail"]

    def test_gw_zero_timesteps_returns_404(self, client_with_model):
        """Return 404 when GW reader has n_timesteps=0."""
        client, _ = client_with_model
        reader = _make_mock_gw_hydrograph_reader(n_timesteps=0)
        reader.n_timesteps = 0
        with patch.object(model_state, "get_gw_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=gw&location_id=1")
        assert resp.status_code == 404
        assert "No GW hydrograph data available" in resp.json()["detail"]

    def test_gw_location_out_of_range_returns_404(self, client_with_model):
        """Return 404 when location_id is out of column range."""
        client, _ = client_with_model
        reader = _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=5)
        with patch.object(model_state, "get_gw_hydrograph_reader", return_value=reader):
            # location_id=5 -> column_index=4, but n_columns=2
            resp = client.get("/api/export/hydrograph-csv?type=gw&location_id=5")
        assert resp.status_code == 404
        assert "GW hydrograph 5 out of range" in resp.json()["detail"]

    def test_gw_location_zero_out_of_range_returns_404(self, client_with_model):
        """location_id=0 means column_index=-1, which is out of range."""
        client, _ = client_with_model
        reader = _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=5)
        with patch.object(model_state, "get_gw_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=gw&location_id=0")
        assert resp.status_code == 404
        assert "GW hydrograph 0 out of range" in resp.json()["detail"]

    def test_gw_success(self, client_with_model):
        """Successful GW hydrograph CSV export."""
        client, _ = client_with_model
        reader = _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=5)
        with patch.object(model_state, "get_gw_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=gw&location_id=1")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "hydrograph_gw_1.csv" in resp.headers["content-disposition"]
        csv_reader = csv.reader(io.StringIO(resp.text))
        rows = list(csv_reader)
        assert rows[0] == ["datetime", "head_ft"]
        assert len(rows) == 6  # header + 5 timesteps

    def test_gw_second_column(self, client_with_model):
        """GW export for location_id=2 uses column_index=1."""
        client, _ = client_with_model
        reader = _make_mock_gw_hydrograph_reader(n_columns=3, n_timesteps=3)
        with patch.object(model_state, "get_gw_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=gw&location_id=2")
        assert resp.status_code == 200
        assert "hydrograph_gw_2.csv" in resp.headers["content-disposition"]

    def test_gw_values_rounded_to_3_decimals(self, client_with_model):
        """GW hydrograph values are rounded to 3 decimal places."""
        client, _ = client_with_model
        reader = _make_mock_gw_hydrograph_reader(n_columns=2, n_timesteps=3)
        with patch.object(model_state, "get_gw_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=gw&location_id=1")
        csv_reader = csv.reader(io.StringIO(resp.text))
        rows = list(csv_reader)
        for row in rows[1:]:
            val_str = row[1]
            if "." in val_str:
                decimals = val_str.split(".")[1]
                assert len(decimals) <= 3

    # --- Stream hydrograph tests ---

    def test_stream_no_reader_returns_404(self, client_with_model):
        """Return 404 when stream hydrograph reader is None."""
        client, _ = client_with_model
        with patch.object(model_state, "get_stream_hydrograph_reader", return_value=None):
            resp = client.get("/api/export/hydrograph-csv?type=stream&location_id=1")
        assert resp.status_code == 404
        assert "No stream hydrograph data available" in resp.json()["detail"]

    def test_stream_zero_timesteps_returns_404(self, client_with_model):
        """Return 404 when stream reader has n_timesteps=0."""
        client, _ = client_with_model
        reader = _make_mock_stream_hydrograph_reader(n_timesteps=0)
        reader.n_timesteps = 0
        with patch.object(model_state, "get_stream_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=stream&location_id=1")
        assert resp.status_code == 404

    def test_stream_node_not_found_returns_404(self, client_with_model):
        """Return 404 when stream node ID is not in hydrograph_ids."""
        client, _ = client_with_model
        reader = _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=5)
        with patch.object(model_state, "get_stream_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=stream&location_id=999")
        assert resp.status_code == 404
        assert "Stream node 999 not found" in resp.json()["detail"]

    def test_stream_success_found_by_find_column(self, client_with_model):
        """Successful stream hydrograph export when find_column_by_node_id finds it."""
        client, _ = client_with_model
        reader = _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=5)
        with patch.object(model_state, "get_stream_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=stream&location_id=2")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "hydrograph_stream_2.csv" in resp.headers["content-disposition"]
        csv_reader = csv.reader(io.StringIO(resp.text))
        rows = list(csv_reader)
        assert rows[0] == ["datetime", "flow_cfs"]
        assert len(rows) == 6  # header + 5 timesteps

    def test_stream_fallback_to_hydrograph_ids_index(self, client_with_model):
        """When find_column_by_node_id returns None but id is in hydrograph_ids,
        fall back to index lookup."""
        client, _ = client_with_model
        reader = _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=5)
        # Override find_column_by_node_id to always return None
        reader.find_column_by_node_id = MagicMock(return_value=None)
        # hydrograph_ids = [1, 2, 3], so location_id=2 should be found at index 1
        with patch.object(model_state, "get_stream_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=stream&location_id=2")
        assert resp.status_code == 200
        assert "hydrograph_stream_2.csv" in resp.headers["content-disposition"]

    def test_stream_both_lookups_fail_returns_404(self, client_with_model):
        """When both find_column_by_node_id and hydrograph_ids lookup fail, return 404."""
        client, _ = client_with_model
        reader = _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=5)
        reader.find_column_by_node_id = MagicMock(return_value=None)
        reader.hydrograph_ids = [10, 20, 30]  # location_id=5 not in this list
        with patch.object(model_state, "get_stream_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=stream&location_id=5")
        assert resp.status_code == 404
        assert "Stream node 5 not found" in resp.json()["detail"]

    def test_stream_values_rounded_to_3_decimals(self, client_with_model):
        """Stream hydrograph values are rounded to 3 decimal places."""
        client, _ = client_with_model
        reader = _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=3)
        with patch.object(model_state, "get_stream_hydrograph_reader", return_value=reader):
            resp = client.get("/api/export/hydrograph-csv?type=stream&location_id=1")
        csv_reader = csv.reader(io.StringIO(resp.text))
        rows = list(csv_reader)
        for row in rows[1:]:
            val_str = row[1]
            if "." in val_str:
                decimals = val_str.split(".")[1]
                assert len(decimals) <= 3

    def test_stream_all_three_ids(self, client_with_model):
        """Successfully export stream hydrograph for each of the 3 known IDs."""
        client, _ = client_with_model
        reader = _make_mock_stream_hydrograph_reader(n_columns=3, n_timesteps=3)
        for loc_id in [1, 2, 3]:
            with patch.object(model_state, "get_stream_hydrograph_reader", return_value=reader):
                resp = client.get(f"/api/export/hydrograph-csv?type=stream&location_id={loc_id}")
            assert resp.status_code == 200
            assert f"hydrograph_stream_{loc_id}.csv" in resp.headers["content-disposition"]
