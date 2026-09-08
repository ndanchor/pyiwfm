"""Unit tests for GIS export functionality."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas
import numpy as np
import pytest

from pyiwfm.components.stream import AppStream, StrmNode, StrmReach
from pyiwfm.core.mesh import AppGrid, Element, Node
from pyiwfm.core.stratigraphy import Stratigraphy
from pyiwfm.visualization.gis_export import GISExporter, SpatialUnitWarning

# Default CRS for tests (UTM Zone 10N - California), meters
TEST_CRS = "EPSG:26910"
# Default CRS for tests (California Teale Albers), feet
TEST_CRS_FT = "ESRI:102600"


@pytest.fixture
def simple_grid() -> AppGrid:
    """Create a simple 2x2 quad mesh for testing."""
    nodes = {
        1: Node(id=1, x=0.0, y=0.0, is_boundary=True),
        2: Node(id=2, x=100.0, y=0.0, is_boundary=True),
        3: Node(id=3, x=200.0, y=0.0, is_boundary=True),
        4: Node(id=4, x=0.0, y=100.0, is_boundary=True),
        5: Node(id=5, x=100.0, y=100.0, is_boundary=False),
        6: Node(id=6, x=200.0, y=100.0, is_boundary=True),
        7: Node(id=7, x=0.0, y=200.0, is_boundary=True),
        8: Node(id=8, x=100.0, y=200.0, is_boundary=True),
        9: Node(id=9, x=200.0, y=200.0, is_boundary=True),
    }
    elements = {
        1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
        2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
        3: Element(id=3, vertices=(4, 5, 8, 7), subregion=2),
        4: Element(id=4, vertices=(5, 6, 9, 8), subregion=2),
    }
    grid = AppGrid(nodes=nodes, elements=elements)
    grid.compute_connectivity()
    return grid


@pytest.fixture
def simple_stratigraphy() -> Stratigraphy:
    """Create simple stratigraphy for testing."""
    n_nodes = 9
    n_layers = 2
    gs_elev = np.full(n_nodes, 100.0)
    top_elev = np.column_stack(
        [
            np.full(n_nodes, 100.0),
            np.full(n_nodes, 50.0),
        ]
    )
    bottom_elev = np.column_stack(
        [
            np.full(n_nodes, 50.0),
            np.full(n_nodes, 0.0),
        ]
    )
    active_node = np.ones((n_nodes, n_layers), dtype=bool)
    return Stratigraphy(
        n_layers=n_layers,
        n_nodes=n_nodes,
        gs_elev=gs_elev,
        top_elev=top_elev,
        bottom_elev=bottom_elev,
        active_node=active_node,
    )


@pytest.fixture
def simple_stream() -> AppStream:
    """Create simple stream network for testing."""
    stream = AppStream()

    # Add stream nodes
    for i in range(1, 6):
        stream.add_node(
            StrmNode(
                id=i,
                x=float(i * 40),
                y=100.0,
                reach_id=1,
            )
        )

    # Add reach
    stream.add_reach(
        StrmReach(
            id=1,
            name="Test Stream",
            upstream_node=1,
            downstream_node=5,
            nodes=[1, 2, 3, 4, 5],
        )
    )

    return stream


class TestGISExporter:
    """Tests for GIS exporter."""

    def test_exporter_creation(self, simple_grid: AppGrid) -> None:
        """Test exporter creation."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        assert exporter is not None

    def test_exporter_with_crs(self, simple_grid: AppGrid) -> None:
        """Test exporter with CRS specification."""
        exporter = GISExporter(grid=simple_grid, crs="EPSG:26910")
        assert exporter.crs == "EPSG:26910"

    def test_nodes_to_geodataframe(self, simple_grid: AppGrid) -> None:
        """Test converting nodes to GeoDataFrame."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        gdf = exporter.nodes_to_geodataframe()

        assert len(gdf) == 9
        assert "geometry" in gdf.columns
        assert "node_id" in gdf.columns
        assert "x" in gdf.columns
        assert "y" in gdf.columns
        assert "is_boundary" in gdf.columns

    def test_elements_to_geodataframe(self, simple_grid: AppGrid) -> None:
        """Test converting elements to GeoDataFrame."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        gdf = exporter.elements_to_geodataframe()

        assert len(gdf) == 4
        assert "geometry" in gdf.columns
        assert "element_id" in gdf.columns
        assert "subregion" in gdf.columns
        assert "n_vertices" in gdf.columns

    def test_elements_geometry_type(self, simple_grid: AppGrid) -> None:
        """Test that element geometries are polygons."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        gdf = exporter.elements_to_geodataframe()

        for geom in gdf.geometry:
            assert geom.geom_type == "Polygon"

    def test_stream_to_geodataframe(self, simple_grid: AppGrid, simple_stream: AppStream) -> None:
        """Test converting stream network to GeoDataFrame."""
        exporter = GISExporter(grid=simple_grid, streams=simple_stream, crs=TEST_CRS)
        gdf = exporter.streams_to_geodataframe()

        assert len(gdf) == 1  # One reach
        assert "geometry" in gdf.columns
        assert "reach_id" in gdf.columns
        assert gdf.iloc[0].geometry.geom_type == "LineString"

    def test_export_to_geopackage(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting to GeoPackage."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)

        output_file = tmp_path / "test.gpkg"
        exporter.export_geopackage(output_file)

        assert output_file.exists()

        # Read back and verify
        nodes_gdf = geopandas.read_file(output_file, layer="nodes")
        elements_gdf = geopandas.read_file(output_file, layer="elements")

        assert len(nodes_gdf) == 9
        assert len(elements_gdf) == 4

    def test_export_to_shapefile(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting to Shapefile."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)

        output_dir = tmp_path / "shapefiles"
        exporter.export_shapefiles(output_dir)

        assert (output_dir / "nodes.shp").exists()
        assert (output_dir / "elements.shp").exists()

    def test_export_to_geojson(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting to GeoJSON."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)

        nodes_file = tmp_path / "nodes.geojson"
        elements_file = tmp_path / "elements.geojson"

        exporter.export_geojson(nodes_file, layer="nodes")
        exporter.export_geojson(elements_file, layer="elements")

        assert nodes_file.exists()
        assert elements_file.exists()

    def test_export_with_stratigraphy(
        self,
        simple_grid: AppGrid,
        simple_stratigraphy: Stratigraphy,
        tmp_path: Path,
    ) -> None:
        """Test exporting with stratigraphy data."""
        exporter = GISExporter(
            grid=simple_grid,
            stratigraphy=simple_stratigraphy,
            crs=TEST_CRS,
        )

        gdf = exporter.nodes_to_geodataframe()

        # Should have stratigraphy columns
        assert "gs_elev" in gdf.columns
        assert "layer_1_top" in gdf.columns
        assert "layer_1_bottom" in gdf.columns

    def test_export_with_streams(
        self,
        simple_grid: AppGrid,
        simple_stream: AppStream,
        tmp_path: Path,
    ) -> None:
        """Test exporting with stream network."""
        exporter = GISExporter(grid=simple_grid, streams=simple_stream, crs=TEST_CRS)

        output_file = tmp_path / "model.gpkg"
        exporter.export_geopackage(output_file, include_streams=True)

        assert output_file.exists()

        # Read back streams layer
        streams_gdf = geopandas.read_file(output_file, layer="streams")
        assert len(streams_gdf) == 1

    def test_subregions_to_geodataframe(self, simple_grid: AppGrid) -> None:
        """Test converting subregions to GeoDataFrame."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        gdf = exporter.subregions_to_geodataframe()

        # Should have 2 subregions (dissolved from elements)
        assert len(gdf) == 2
        assert "subregion_id" in gdf.columns

    def test_boundary_to_geodataframe(self, simple_grid: AppGrid) -> None:
        """Test extracting model boundary."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        gdf = exporter.boundary_to_geodataframe()

        assert len(gdf) == 1
        assert gdf.iloc[0].geometry.geom_type == "Polygon"

    def test_model_nodes_factor_storage(self, simple_grid: AppGrid) -> None:
        """
        Test storing of nodes_factor and successful generation of AppGrid
        given different user-defined values
        """
        fact_m2ft_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=3.28084
        )

        fact_ft2m_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=0.3048
        )

        fact_nochg_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=1.0
        )

        assert fact_m2ft_grid.nodes_factor == pytest.approx(3.28084)
        assert fact_ft2m_grid.nodes_factor == pytest.approx(0.3048)
        assert fact_nochg_grid.nodes_factor == pytest.approx(1.0)

    def test_exporter_adjustment_factor(self, simple_grid: AppGrid) -> None:
        """
        Test storing of nodes_factor and successful generation of AppGrid
        given different user-defined values
        """
        fact_neg_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=-1.0
        )

        fact_zero_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=0.0
        )

        neg_exporter = GISExporter(grid=fact_neg_grid, crs=TEST_CRS)

        zero_exporter = GISExporter(grid=fact_zero_grid, crs=TEST_CRS)

        assert neg_exporter.adjustment_factor == pytest.approx(1.0)
        assert zero_exporter.adjustment_factor == pytest.approx(1.0)

    def test_model_m2ft_crs_ft_xy_conversion(self, simple_grid: AppGrid) -> None:
        """
        Test converting model xy coordinate units to user-specified
            crs units.
            XY nodes: meters
            Model units: feet
            CRS units: feet
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=3.28084
        )
        fact_grid.compute_connectivity()
        exporter = GISExporter(grid=fact_grid, crs=TEST_CRS_FT)

        assert exporter.adjustment_factor == pytest.approx(1.0)

    def test_model_ft2m_crs_ft_xy_conversion(self, simple_grid: AppGrid) -> None:
        """
        Test converting model xy coordinate units to user-specified
            crs units.
            XY nodes: feet
            Model units: meters
            CRS units: feet
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=0.3048
        )
        fact_grid.compute_connectivity()
        exporter = GISExporter(grid=fact_grid, crs=TEST_CRS_FT)

        assert exporter.adjustment_factor == pytest.approx(3.28084)

    def test_model_m2ft_crs_m_xy_conversion(self, simple_grid: AppGrid) -> None:
        """
        Test converting model xy coordinate units to user-specified
            crs units.
            XY nodes: meters
            Model units: feet
            CRS units: meters
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=3.28084
        )
        fact_grid.compute_connectivity()
        exporter = GISExporter(grid=fact_grid, crs=TEST_CRS)

        assert exporter.adjustment_factor == pytest.approx(0.3048)

    def test_model_ft2m_crs_m_xy_conversion(self, simple_grid: AppGrid) -> None:
        """
        Test converting model xy coordinate units to user-specified
            crs units.
            XY nodes: feet
            Model units: meters
            CRS units: feet
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=0.3048
        )
        fact_grid.compute_connectivity()
        exporter = GISExporter(grid=fact_grid, crs=TEST_CRS)

        assert exporter.adjustment_factor == pytest.approx(1.0)

    def test_model_ft2ft_crs_m_xy_conversion(self, simple_grid: AppGrid) -> None:
        """
        Test converting model xy coordinate units to user-specified
            crs units.
            XY nodes: feet
            Model units: feet
            CRS units: meters

        A Nodes-file factor of 1.0 alone is ambiguous (see
        test_ambiguous_factor_with_crs_warns_and_skips_conversion), so the
        model's native unit is disambiguated explicitly here via
        model_length_unit -- exactly what the ambiguous case calls for.
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=1.0
        )
        fact_grid.compute_connectivity()
        exporter = GISExporter(grid=fact_grid, crs=TEST_CRS, model_length_unit="FEET")

        assert exporter.resolved_model_length_unit == "FEET"
        assert exporter.adjustment_factor == pytest.approx(0.3048)

    def test_model_ft2ft_crs_ft_xy_conversion(self, simple_grid: AppGrid) -> None:
        """
        Test converting model xy coordinate units to user-specified
            crs units.
            XY nodes: feet
            Model units: feet
            CRS units: feet
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=1.0
        )
        fact_grid.compute_connectivity()
        exporter = GISExporter(grid=fact_grid, crs=TEST_CRS_FT)

        assert exporter.adjustment_factor == pytest.approx(1.0)

    def test_model_m2m_crs_ft_xy_conversion(self, simple_grid: AppGrid) -> None:
        """
        Test converting model xy coordinate units to user-specified
            crs units.
            XY nodes: m
            Model units: m
            CRS units: feet

        A Nodes-file factor of 1.0 alone is ambiguous (see
        test_ambiguous_factor_with_crs_warns_and_skips_conversion), so the
        model's native unit is disambiguated explicitly here via
        model_length_unit -- exactly what the ambiguous case calls for.
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=1.0
        )
        fact_grid.compute_connectivity()
        exporter = GISExporter(grid=fact_grid, crs=TEST_CRS_FT, model_length_unit="METERS")

        assert exporter.resolved_model_length_unit == "METERS"
        assert exporter.adjustment_factor == pytest.approx(3.28084)

    def test_ambiguous_factor_with_crs_warns_and_skips_conversion(
        self, simple_grid: AppGrid
    ) -> None:
        """
        A Nodes-file factor of 1.0 alone can't say what unit the model's
        (matching) input and internal coordinates are in -- it's genuinely
        ambiguous regardless of the target CRS's unit. GISExporter should
        warn rather than crash or silently guess, and fall back to no
        conversion.
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes, elements=simple_grid.elements, nodes_factor=1.0
        )
        fact_grid.compute_connectivity()

        with pytest.warns(SpatialUnitWarning):
            exporter = GISExporter(grid=fact_grid, crs=TEST_CRS_FT)

        assert exporter.resolved_model_length_unit is None
        assert exporter.adjustment_factor == pytest.approx(1.0)

    def test_grid_length_unit_resolves_ambiguous_factor(self, simple_grid: AppGrid) -> None:
        """
        grid.length_unit (as set by IWFMModel.from_preprocessor() from the
        PreProcessor main file's FACTLTOU/UNITLTOU) disambiguates a Nodes
        factor of 1.0 without needing an explicit model_length_unit
        override, and without any warning.
        """
        fact_grid = AppGrid(
            nodes=simple_grid.nodes,
            elements=simple_grid.elements,
            nodes_factor=1.0,
            length_unit="FEET",
        )
        fact_grid.compute_connectivity()

        with warnings.catch_warnings():
            warnings.simplefilter("error", SpatialUnitWarning)
            exporter = GISExporter(grid=fact_grid, crs=TEST_CRS)

        assert exporter.resolved_model_length_unit == "FEET"
        assert exporter.adjustment_factor == pytest.approx(0.3048)

    def test_no_crs_skips_unit_resolution_entirely(self, simple_grid: AppGrid) -> None:
        """GISExporter(crs=None) must not crash, even with an ambiguous
        or missing nodes_factor -- there's nothing to align against."""
        exporter = GISExporter(grid=simple_grid)

        assert exporter.crs is None
        assert exporter.resolved_crs_length_unit is None
        assert exporter.adjustment_factor == pytest.approx(1.0)

        gdf = exporter.nodes_to_geodataframe()
        assert gdf.crs is None


class TestGISExporterAttributes:
    """Tests for GIS export with additional attributes."""

    def test_nodes_with_custom_attributes(self, simple_grid: AppGrid) -> None:
        """Test adding custom attributes to nodes."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)

        # Add custom attribute data
        node_ids = list(simple_grid.nodes.keys())
        heads = {nid: 50.0 + nid for nid in node_ids}

        gdf = exporter.nodes_to_geodataframe(attributes={"head": heads})

        assert "head" in gdf.columns
        assert gdf[gdf["node_id"] == 1]["head"].values[0] == 51.0

    def test_elements_with_custom_attributes(self, simple_grid: AppGrid) -> None:
        """Test adding custom attributes to elements."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)

        # Add custom attribute data
        elem_ids = list(simple_grid.elements.keys())
        kh = {eid: 10.0 * eid for eid in elem_ids}

        gdf = exporter.elements_to_geodataframe(attributes={"kh": kh})

        assert "kh" in gdf.columns
        assert gdf[gdf["element_id"] == 2]["kh"].values[0] == 20.0


# ── Additional tests for increased coverage ──────────────────────────


class TestGISExporterExportGeojsonLayers:
    """Tests for GeoJSON export with various layer types."""

    def test_export_geojson_streams_layer(
        self, simple_grid: AppGrid, simple_stream: AppStream, tmp_path: Path
    ) -> None:
        """Test exporting streams layer to GeoJSON."""
        exporter = GISExporter(grid=simple_grid, streams=simple_stream, crs=TEST_CRS)
        output_file = tmp_path / "streams.geojson"
        exporter.export_geojson(output_file, layer="streams")
        assert output_file.exists()

    def test_export_geojson_subregions_layer(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting subregions layer to GeoJSON."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        output_file = tmp_path / "subregions.geojson"
        exporter.export_geojson(output_file, layer="subregions")
        assert output_file.exists()

    def test_export_geojson_boundary_layer(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting boundary layer to GeoJSON."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        output_file = tmp_path / "boundary.geojson"
        exporter.export_geojson(output_file, layer="boundary")
        assert output_file.exists()

    def test_export_geojson_invalid_layer(self, simple_grid: AppGrid, tmp_path: Path) -> None:
        """Test exporting with an invalid layer name raises ValueError."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        output_file = tmp_path / "invalid.geojson"
        with pytest.raises(ValueError, match="Unknown layer"):
            exporter.export_geojson(output_file, layer="not_a_layer")


class TestGISExporterStreamsEdgeCases:
    """Tests for stream-related edge cases."""

    def test_streams_to_geodataframe_no_streams(self, simple_grid: AppGrid) -> None:
        """Test streams_to_geodataframe returns empty GDF when no streams."""
        exporter = GISExporter(grid=simple_grid, streams=None, crs=TEST_CRS)
        gdf = exporter.streams_to_geodataframe()
        assert len(gdf) == 0
        assert "reach_id" in gdf.columns

    def test_export_shapefiles_with_streams(
        self, simple_grid: AppGrid, simple_stream: AppStream, tmp_path: Path
    ) -> None:
        """Test exporting shapefiles with stream network included."""
        exporter = GISExporter(grid=simple_grid, streams=simple_stream, crs=TEST_CRS)
        output_dir = tmp_path / "shp_with_streams"
        exporter.export_shapefiles(output_dir, include_streams=True)

        assert (output_dir / "nodes.shp").exists()
        assert (output_dir / "elements.shp").exists()
        assert (output_dir / "streams.shp").exists()

    def test_export_geopackage_no_streams_flag(
        self, simple_grid: AppGrid, simple_stream: AppStream, tmp_path: Path
    ) -> None:
        """Test geopackage export with include_streams=False."""
        exporter = GISExporter(grid=simple_grid, streams=simple_stream, crs=TEST_CRS)
        output_file = tmp_path / "no_streams.gpkg"
        exporter.export_geopackage(
            output_file,
            include_streams=False,
            include_subregions=False,
            include_boundary=False,
        )
        assert output_file.exists()
        # Verify only nodes and elements layers exist
        layers = geopandas.list_layers(output_file)
        layer_names = list(layers["name"])
        assert "nodes" in layer_names
        assert "elements" in layer_names
        assert "streams" not in layer_names

    def test_export_shapefiles_no_optional_layers(
        self, simple_grid: AppGrid, tmp_path: Path
    ) -> None:
        """Test shapefile export with all optional layers disabled."""
        exporter = GISExporter(grid=simple_grid, crs=TEST_CRS)
        output_dir = tmp_path / "minimal_shp"
        exporter.export_shapefiles(
            output_dir,
            include_streams=False,
            include_subregions=False,
            include_boundary=False,
        )
        assert (output_dir / "nodes.shp").exists()
        assert (output_dir / "elements.shp").exists()
        assert not (output_dir / "subregions.shp").exists()
        assert not (output_dir / "boundary.shp").exists()


class TestGISExporterStreamCoordinateResolution:
    """Tests for stream coordinate resolution via gw_node → grid.nodes."""

    def test_resolves_coords_via_gw_node(self, simple_grid: AppGrid) -> None:
        """Coordinates should be resolved via gw_node → grid.nodes, not stream node x/y."""
        stream = AppStream()
        # Stream nodes with x=0, y=0 but valid gw_node pointing to grid nodes 2,3,5
        # (grid node 1 is at (0,0) so we use nodes with non-zero coords)
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1, gw_node=2))
        stream.add_node(StrmNode(id=2, x=0.0, y=0.0, reach_id=1, gw_node=3))
        stream.add_node(StrmNode(id=3, x=0.0, y=0.0, reach_id=1, gw_node=5))
        stream.add_reach(
            StrmReach(
                id=1,
                upstream_node=1,
                downstream_node=3,
                nodes=[1, 2, 3],
                name="R1",
            )
        )

        exporter = GISExporter(grid=simple_grid, streams=stream, crs=TEST_CRS)
        gdf = exporter.streams_to_geodataframe()

        assert len(gdf) == 1
        coords = list(gdf.iloc[0].geometry.coords)
        # Grid nodes 2(100,0), 3(200,0), 5(100,100) — all non-zero
        assert coords[0] == (100.0, 0.0)  # grid node 2
        assert coords[1] == (200.0, 0.0)  # grid node 3
        assert coords[2] == (100.0, 100.0)  # grid node 5

    def test_zero_coords_no_gw_node_skipped(self, simple_grid: AppGrid) -> None:
        """Stream nodes with (0,0) and no gw_node should be skipped."""
        stream = AppStream()
        # Node 1 has valid gw_node, node 2 has (0,0) and no gw_node
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1, gw_node=2))
        stream.add_node(StrmNode(id=2, x=0.0, y=0.0, reach_id=1, gw_node=None))
        stream.add_reach(
            StrmReach(
                id=1,
                upstream_node=1,
                downstream_node=2,
                nodes=[1, 2],
                name="R1",
            )
        )

        # No CRS to avoid empty-GDF CRS assignment error
        exporter = GISExporter(grid=simple_grid, streams=stream)
        gdf = exporter.streams_to_geodataframe()

        # Only 1 valid coordinate, so LineString needs 2+ points — reach should be dropped
        assert len(gdf) == 0

    def test_nonzero_stream_coords_used_as_fallback(self, simple_grid: AppGrid) -> None:
        """Non-zero stream node coordinates used when no gw_node."""
        stream = AppStream()
        stream.add_node(StrmNode(id=1, x=50.0, y=50.0, reach_id=1, gw_node=None))
        stream.add_node(StrmNode(id=2, x=150.0, y=50.0, reach_id=1, gw_node=None))
        stream.add_reach(
            StrmReach(
                id=1,
                upstream_node=1,
                downstream_node=2,
                nodes=[1, 2],
                name="R1",
            )
        )

        exporter = GISExporter(grid=simple_grid, streams=stream, crs=TEST_CRS)
        gdf = exporter.streams_to_geodataframe()

        assert len(gdf) == 1
        coords = list(gdf.iloc[0].geometry.coords)
        assert coords[0] == (50.0, 50.0)
        assert coords[1] == (150.0, 50.0)

    def test_mixed_gw_node_and_direct_coords(self, simple_grid: AppGrid) -> None:
        """Mix of gw_node resolved and direct non-zero coordinates."""
        stream = AppStream()
        # Use grid node 2 (100,0) so we can distinguish from direct coords
        stream.add_node(StrmNode(id=1, x=0.0, y=0.0, reach_id=1, gw_node=2))
        stream.add_node(StrmNode(id=2, x=150.0, y=150.0, reach_id=1, gw_node=None))
        stream.add_reach(
            StrmReach(
                id=1,
                upstream_node=1,
                downstream_node=2,
                nodes=[1, 2],
                name="R1",
            )
        )

        exporter = GISExporter(grid=simple_grid, streams=stream, crs=TEST_CRS)
        gdf = exporter.streams_to_geodataframe()

        assert len(gdf) == 1
        coords = list(gdf.iloc[0].geometry.coords)
        assert len(coords) == 2
        # First from grid node 2 (100, 0), second from direct coords
        assert coords[0] == (100.0, 0.0)
        assert coords[1] == (150.0, 150.0)
