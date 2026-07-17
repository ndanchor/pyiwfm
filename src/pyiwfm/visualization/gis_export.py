"""
GIS export functionality for IWFM models.

This module provides the :class:`GISExporter` class for exporting IWFM model
data to various GIS formats including GeoPackage, Shapefile, and GeoJSON.

Supported Formats
-----------------
- **GeoPackage** (.gpkg): Recommended format, single file with multiple layers
- **Shapefile** (.shp): Widely compatible but limited to 10-char field names
- **GeoJSON** (.geojson): Text-based, good for web applications

Example
-------
Export a mesh to GeoPackage:

>>> from pyiwfm.core.mesh import AppGrid, Node, Element
>>> from pyiwfm.visualization.gis_export import GISExporter
>>>
>>> # Create simple mesh
>>> nodes = {1: Node(id=1, x=0.0, y=0.0), 2: Node(id=2, x=100.0, y=0.0),
...          3: Node(id=3, x=50.0, y=100.0)}
>>> elements = {1: Element(id=1, vertices=(1, 2, 3))}
>>> grid = AppGrid(nodes=nodes, elements=elements)
>>> grid.compute_connectivity()
>>>
>>> # Export to GeoPackage
>>> exporter = GISExporter(grid=grid, crs="EPSG:26910")
>>> exporter.export_geopackage("model.gpkg")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
import math

import geopandas as gpd
import pyproj
from shapely.geometry import LineString, Point, Polygon

if TYPE_CHECKING:
    from pyiwfm.components.stream import AppStream
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy


class GISExporter:
    """
    Export IWFM model data to GIS formats.

    This class converts model meshes, stratigraphy, and stream networks
    to GeoDataFrames that can be exported to various GIS formats.

    Parameters
    ----------
    grid : AppGrid
        Model mesh to export.
    stratigraphy : Stratigraphy, optional
        Model stratigraphy. If provided, layer elevations are added
        as attributes to the nodes GeoDataFrame.
    streams : AppStream, optional
        Stream network. If provided, enables stream layer export.
    crs : str, optional
        Coordinate reference system (e.g., 'EPSG:26910', 'EPSG:2227').
        If None, output files will have no CRS defined.

    Raises
    ------
    ImportError
        If geopandas or shapely are not installed.

    Examples
    --------
    Basic export to GeoPackage:

    >>> exporter = GISExporter(grid=grid, crs="EPSG:26910")
    >>> exporter.export_geopackage("model.gpkg")

    Export with stratigraphy data:

    >>> exporter = GISExporter(grid=grid, stratigraphy=strat, crs="EPSG:26910")
    >>> gdf = exporter.nodes_to_geodataframe()
    >>> # GeoDataFrame includes gs_elev, layer_1_top, layer_1_bottom, etc.

    Export with custom attributes:

    >>> head_data = {1: 50.0, 2: 52.0, 3: 48.0}  # node_id -> value
    >>> gdf = exporter.nodes_to_geodataframe(attributes={"head_ft": head_data})
    >>> gdf.to_file("nodes_with_heads.gpkg", driver="GPKG")

    Export to multiple formats:

    >>> exporter.export_geopackage("model.gpkg")  # GeoPackage
    >>> exporter.export_shapefiles("shapefiles/")  # Shapefiles
    >>> exporter.export_geojson("elements.geojson", layer="elements")
    """

    def __init__(
        self,
        grid: AppGrid,
        stratigraphy: Stratigraphy | None = None,
        streams: AppStream | None = None,
        crs: str | None = None,
        adjustment_factor: float | None = None
    ) -> None:


        """
        Initialize the GIS exporter.

        Args:
            grid: Model mesh
            stratigraphy: Model stratigraphy (optional)
            streams: Stream network (optional)
            crs: Coordinate reference system (e.g., 'EPSG:26910')
        """
        self.grid = grid
        self.stratigraphy = stratigraphy
        self.streams = streams
        self.crs = pyproj.CRS.from_user_input(crs)
        self.adjustment_factor = self._calculate_adjustment()

    def _calculate_adjustment(self):
        """
        Compares the target CRS units against the model mesh unit factor
        to guess whether model units are meters or feet and prevent
        misaligned GIS exports.
        """
        # 1. Safely extract target CRS units (e.g., 'metre', 'us survey foot')
        try:
            crs_unit = self.crs.axis_info[0].unit_name.lower()
        except (AttributeError, IndexError):
            # Fallback if pyproj axis_info structure is not accessible
            crs_unit = 'unknown'

        # Get the raw preprocessor/mesh factor passed down
        # (By default, 1.0 means no raw preprocessing transformation occurred)
        model_factor = getattr(self.grid, 'nodes_factor', 1.0)

        # 2. Establish defaults
        crs_is_feet = 'foot' in crs_unit or 'ft' in crs_unit
        crs_is_meters = 'metr' in crs_unit or 'm' == crs_unit

        # 3. Guardrails & Unit Deduction
        # Standard conversion constants for evaluation
        FT_TO_M = 0.3048
        M_TO_FT = 1.0 / FT_TO_M

        # Case A: Target CRS is in FEET
        if crs_is_feet:
            # If the factor is close to 0.3048, the raw files were in feet,
            # but the model internally converted them to meters.
            if math.isclose(model_factor, FT_TO_M, rel_tol=1e-3):
                # Model is internally in meters, but target CRS wants feet.
                # Convert internal meters back to feet:
                return 1.0 / model_factor

            # If the factor is close to 3.28084, the raw files were in meters,
            # and the model internally converted them to feet.
            elif math.isclose(model_factor, M_TO_FT, rel_tol=1e-3):
                # Model is internally in feet, target CRS wants feet.
                return 1.0

            else:
                # Unknown units/conversion factor, do nothing.
                return 1.0

        # Case B: Target CRS is in METERS (Metric)
        elif crs_is_meters:
            # If the factor is close to 3.28084, the raw files were in meters,
            # but the model internally converted them to feet.
            if math.isclose(model_factor, M_TO_FT, rel_tol=1e-3):
                # Model is internally in feet, but target CRS wants meters.
                # Convert internal feet back to meters:
                return 1.0 / model_factor

            # If the factor is close to 0.3048, the raw files were in feet,
            # but the model internally converted them to meters.
            elif math.isclose(model_factor, FT_TO_M, rel_tol=1e-3):
                # Model is internally in meters, target CRS wants meters.
                return 1.0

            else:
                # Unknown units/conversion factor, do nothing.
                return 1.0

        # Case C: Unknown CRS unit type or unhandled unit
        else:
            # Fall back to strictly reversing the preprocessor conversion factor
            # on the assumption that the input files matched the target CRS projection.
            return 1.0 / model_factor if model_factor != 0 else 1.0

    def nodes_to_geodataframe(
        self,
        attributes: dict[str, dict[int, Any]] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Convert mesh nodes to a GeoDataFrame.

        Args:
            attributes: Optional dict of attribute_name -> {node_id: value}

        Returns:
            GeoDataFrame with node points
        """
        data = []

        for node in self.grid.iter_nodes():

            converted_x = node.x * self.adjustment_factor
            converted_y = node.y * self.adjustment_factor

            row = {

                "node_id": node.id,
                "x": converted_x,
                "y": converted_y,
                "is_boundary": node.is_boundary,
                "area": node.area,
                "geometry": Point(converted_x, converted_y),
            }

            # Add stratigraphy data if available
            if self.stratigraphy is not None:
                idx = node.id - 1  # Convert to 0-indexed
                if 0 <= idx < self.stratigraphy.n_nodes:
                    row["gs_elev"] = float(self.stratigraphy.gs_elev[idx])
                    for layer in range(self.stratigraphy.n_layers):
                        row[f"layer_{layer + 1}_top"] = float(
                            self.stratigraphy.top_elev[idx, layer]
                        )
                        row[f"layer_{layer + 1}_bottom"] = float(
                            self.stratigraphy.bottom_elev[idx, layer]
                        )

            # Add custom attributes
            if attributes:
                for attr_name, attr_values in attributes.items():
                    if node.id in attr_values:
                        row[attr_name] = attr_values[node.id]

            data.append(row)

        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        return gdf

    def elements_to_geodataframe(
        self,
        attributes: dict[str, dict[int, Any]] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Convert mesh elements to a GeoDataFrame.

        Args:
            attributes: Optional dict of attribute_name -> {element_id: value}

        Returns:
            GeoDataFrame with element polygons
        """
        data = []

        for elem in self.grid.iter_elements():
            # Get vertex coordinates
            coords = []
            for vid in elem.vertices:
                node = self.grid.nodes[vid]

                converted_x = node.x * self.adjustment_factor
                converted_y = node.y * self.adjustment_factor

                coords.append((converted_x, converted_y))
            # Close the polygon
            coords.append(coords[0])

            row = {
                "element_id": elem.id,
                "subregion": elem.subregion,
                "n_vertices": elem.n_vertices,
                "area": elem.area,
                "geometry": Polygon(coords),
            }

            # Add custom attributes
            if attributes:
                for attr_name, attr_values in attributes.items():
                    if elem.id in attr_values:
                        row[attr_name] = attr_values[elem.id]

            data.append(row)

        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        return gdf

    def streams_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert stream network to a GeoDataFrame.

        Returns:
            GeoDataFrame with stream reach linestrings
        """
        if self.streams is None:
            return gpd.GeoDataFrame(columns=["reach_id", "name", "geometry"], crs=self.crs)

        data = []

        for reach in self.streams.iter_reaches():
            # Get node coordinates for this reach, resolving via gw_node
            coords = []
            for nid in reach.nodes:
                if nid in self.streams.nodes:
                    sn = self.streams.nodes[nid]
                    gw = getattr(sn, "gw_node", None)
                    if gw is not None and gw in self.grid.nodes:
                        gn = self.grid.nodes[gw]

                        converted_gn_x = gn.x * self.adjustment_factor
                        converted_gn_y = gn.y * self.adjustment_factor

                        coords.append((converted_gn_x, converted_gn_y))
                    elif sn.x != 0.0 or sn.y != 0.0:
                        converted_sn_x = sn.x * self.adjustment_factor
                        converted_sn_y = sn.y * self.adjustment_factor

                        coords.append((converted_sn_x, converted_sn_y))

            if len(coords) >= 2:
                row = {
                    "reach_id": reach.id,
                    "name": reach.name,
                    "n_nodes": reach.n_nodes,
                    "geometry": LineString(coords),
                }
                data.append(row)

        gdf = gpd.GeoDataFrame(data, crs=self.crs)
        return gdf

    def subregions_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert subregions to a GeoDataFrame (dissolved elements).

        Returns:
            GeoDataFrame with subregion polygons
        """

        # Get elements GeoDataFrame
        elements_gdf = self.elements_to_geodataframe()

        # Dissolve by subregion
        subregions_gdf = elements_gdf.dissolve(by="subregion", as_index=False)
        subregions_gdf = subregions_gdf.rename(columns={"subregion": "subregion_id"})
        subregions_gdf = subregions_gdf[["subregion_id", "geometry"]]

        return subregions_gdf

    def boundary_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Extract model boundary as a GeoDataFrame.

        Returns:
            GeoDataFrame with model boundary polygon
        """
        from shapely.ops import unary_union

        # Get elements and dissolve all to get boundary
        elements_gdf = self.elements_to_geodataframe()
        boundary_geom = unary_union(elements_gdf.geometry)

        gdf = gpd.GeoDataFrame(
            [{"boundary_id": 1, "geometry": boundary_geom}],
            crs=self.crs,
        )
        return gdf

    def export_geopackage(
        self,
        output_path: Path | str,
        include_streams: bool = True,
        include_subregions: bool = True,
        include_boundary: bool = True,
    ) -> None:
        """
        Export model to GeoPackage format.

        Args:
            output_path: Output file path (.gpkg)
            include_streams: Include stream network layer
            include_subregions: Include subregions layer
            include_boundary: Include boundary layer
        """
        output_path = Path(output_path)

        # Export nodes
        nodes_gdf = self.nodes_to_geodataframe()
        nodes_gdf.to_file(output_path, layer="nodes", driver="GPKG")

        # Export elements
        elements_gdf = self.elements_to_geodataframe()
        elements_gdf.to_file(output_path, layer="elements", driver="GPKG")

        # Export streams if available
        if include_streams and self.streams is not None:
            streams_gdf = self.streams_to_geodataframe()
            if len(streams_gdf) > 0:
                streams_gdf.to_file(output_path, layer="streams", driver="GPKG")

        # Export subregions
        if include_subregions:
            subregions_gdf = self.subregions_to_geodataframe()
            if len(subregions_gdf) > 0:
                subregions_gdf.to_file(output_path, layer="subregions", driver="GPKG")

        # Export boundary
        if include_boundary:
            boundary_gdf = self.boundary_to_geodataframe()
            boundary_gdf.to_file(output_path, layer="boundary", driver="GPKG")

    def _shorten_columns_for_shapefile(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Shorten column names to fit Shapefile's 10-character limit.

        Args:
            gdf: GeoDataFrame to modify

        Returns:
            GeoDataFrame with shortened column names
        """
        # Mapping of long column names to short versions (max 10 chars)
        column_map = {
            "is_boundary": "is_bndry",
            "subregion_id": "subreg_id",
            "boundary_id": "bndry_id",
        }
        rename_dict = {k: v for k, v in column_map.items() if k in gdf.columns}
        if rename_dict:
            gdf = gdf.rename(columns=rename_dict)
        return gdf

    def export_shapefiles(
        self,
        output_dir: Path | str,
        include_streams: bool = True,
        include_subregions: bool = True,
        include_boundary: bool = True,
    ) -> None:
        """
        Export model to Shapefile format.

        Creates separate shapefiles for nodes, elements, etc.

        Args:
            output_dir: Output directory
            include_streams: Include stream network shapefile
            include_subregions: Include subregions shapefile
            include_boundary: Include boundary shapefile
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export nodes
        nodes_gdf = self.nodes_to_geodataframe()
        nodes_gdf = self._shorten_columns_for_shapefile(nodes_gdf)
        nodes_gdf.to_file(output_dir / "nodes.shp")

        # Export elements
        elements_gdf = self.elements_to_geodataframe()
        elements_gdf = self._shorten_columns_for_shapefile(elements_gdf)
        elements_gdf.to_file(output_dir / "elements.shp")

        # Export streams if available
        if include_streams and self.streams is not None:
            streams_gdf = self.streams_to_geodataframe()
            if len(streams_gdf) > 0:
                streams_gdf = self._shorten_columns_for_shapefile(streams_gdf)
                streams_gdf.to_file(output_dir / "streams.shp")

        # Export subregions
        if include_subregions:
            subregions_gdf = self.subregions_to_geodataframe()
            if len(subregions_gdf) > 0:
                subregions_gdf = self._shorten_columns_for_shapefile(subregions_gdf)
                subregions_gdf.to_file(output_dir / "subregions.shp")

        # Export boundary
        if include_boundary:
            boundary_gdf = self.boundary_to_geodataframe()
            boundary_gdf = self._shorten_columns_for_shapefile(boundary_gdf)
            boundary_gdf.to_file(output_dir / "boundary.shp")

    def export_geojson(
        self,
        output_path: Path | str,
        layer: str = "elements",
    ) -> None:
        """
        Export a single layer to GeoJSON format.

        Args:
            output_path: Output file path (.geojson)
            layer: Layer to export ('nodes', 'elements', 'streams',
                   'subregions', 'boundary')
        """
        output_path = Path(output_path)

        if layer == "nodes":
            gdf = self.nodes_to_geodataframe()
        elif layer == "elements":
            gdf = self.elements_to_geodataframe()
        elif layer == "streams":
            gdf = self.streams_to_geodataframe()
        elif layer == "subregions":
            gdf = self.subregions_to_geodataframe()
        elif layer == "boundary":
            gdf = self.boundary_to_geodataframe()
        else:
            raise ValueError(f"Unknown layer: {layer}")

        gdf.to_file(output_path, driver="GeoJSON")
