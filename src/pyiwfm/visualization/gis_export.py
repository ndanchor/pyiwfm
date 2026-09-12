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
>>> exporter = GISExporter(grid=grid, input_crs="EPSG:26910", output_crs="EPSG:4326")
>>> exporter.export_geopackage("model.gpkg")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    input_crs : str or pyproj.CRS, optional
        Coordinate reference system of the original input node coordinates
        (e.g., 'EPSG:26910', 'EPSG:2227'). Defaults to 'EPSG:26910'.
    output_crs : str or pyproj.CRS, optional
        Target coordinate reference system for exported GIS files. If None,
        defaults to matching *input_crs*.

    Raises
    ------
    ImportError
        If geopandas or shapely are not installed.

    Examples
    --------
    Basic export to GeoPackage:

    >>> exporter = GISExporter(grid=grid, input_crs="EPSG:26910")
    >>> exporter.export_geopackage("model.gpkg")

    Export with stratigraphy data:

    >>> exporter = GISExporter(grid=grid, stratigraphy=strat, input_crs="EPSG:26910")
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
        input_crs: str | pyproj.CRS | None = None,
        output_crs: str | pyproj.CRS | None = None,
    ) -> None:
        """
        Initialize the GIS exporter.

        Args:
            grid: Model mesh
            stratigraphy: Model stratigraphy (optional)
            streams: Stream network (optional)
            input_crs: Input coordinate reference system (e.g., 'EPSG:26910'). If
                    None, no CRS is assigned and coordinates are exported
                    unconverted.
            output_crs: Output coordinate reference system (e.g., 'EPSG:26910'). If
                    None, the input CRS is utilized.
        """
        self.grid = grid
        self.stratigraphy = stratigraphy
        self.streams = streams

        # 1. Define Input CRS (default to EPSG:26910 if user provides None)
        self.input_crs = (
            pyproj.CRS.from_user_input(input_crs)
            if input_crs is not None
            else pyproj.CRS.from_user_input("EPSG:26910")
        )

        # 2. Define Output CRS (default to match input_crs if not explicitly set)
        self.output_crs = (
            pyproj.CRS.from_user_input(output_crs)
            if output_crs is not None
            else self.input_crs
        )


    @property
    def _unscale_factor(self) -> float:
        """
        Factor to convert internal model coordinates back to original node file coordinates.
        If grid.nodes_factor is 1.0 or None, no scaling is applied.
        """
        nodes_factor = getattr(self.grid, "nodes_factor", 1.0)
        if nodes_factor is None or nodes_factor <= 0:
            return 1.0
        return 1.0 / nodes_factor

    def _finalize_gdf(self, data: list[dict], geometry_col: str = "geometry") -> gpd.GeoDataFrame:
        """
        Creates a GeoDataFrame in input_crs and reprojects to output_crs if needed.
        """
        gdf = gpd.GeoDataFrame(data, crs=self.input_crs)

        if self.output_crs != self.input_crs:
            gdf = gdf.to_crs(self.output_crs)

        return gdf

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
        scale = self._unscale_factor

        for node in self.grid.iter_nodes():
            unscaled_x = node.x * scale
            unscaled_y = node.y * scale

            row = {
                "node_id": node.id,
                "x": unscaled_x,
                "y": unscaled_y,
                "is_boundary": node.is_boundary,
                "area": node.area,
                "geometry": Point(unscaled_x, unscaled_y),
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

        return self._finalize_gdf(data)

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
        scale = self._unscale_factor

        for elem in self.grid.iter_elements():
            # Get vertex coordinates
            coords = []
            for vid in elem.vertices:
                node = self.grid.nodes[vid]
                unscaled_x = node.x * scale
                unscaled_y = node.y * scale
                coords.append((unscaled_x, unscaled_y))

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

        return self._finalize_gdf(data)

    def streams_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert stream network to a GeoDataFrame.

        Returns:
            GeoDataFrame with stream reach linestrings
        """
        if self.streams is None:
            return gpd.GeoDataFrame(columns=["reach_id", "name", "geometry"], crs=self.output_crs)

        data = []
        scale = self._unscale_factor

        for reach in self.streams.iter_reaches():
            coords = []
            for nid in reach.nodes:
                if nid in self.streams.nodes:
                    sn = self.streams.nodes[nid]
                    gw = getattr(sn, "gw_node", None)
                    if gw is not None and gw in self.grid.nodes:
                        gn = self.grid.nodes[gw]
                        unscaled_gn_x = gn.x * scale
                        unscaled_gn_y = gn.y * scale
                        coords.append((unscaled_gn_x, unscaled_gn_y))
                    elif sn.x != 0.0 or sn.y != 0.0:
                        unscaled_sn_x = sn.x * scale
                        unscaled_sn_y = sn.y * scale
                        coords.append((unscaled_sn_x, unscaled_sn_y))

            if len(coords) >= 2:
                row = {
                    "reach_id": reach.id,
                    "name": reach.name,
                    "n_nodes": reach.n_nodes,
                    "geometry": LineString(coords),
                }
                data.append(row)

        return self._finalize_gdf(data)

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
        # Get elements (already converted & reprojected to output_crs)
        elements_gdf = self.elements_to_geodataframe()

        # Dissolve all elements into a single boundary polygon
        boundary_gdf = elements_gdf.dissolve()
        boundary_gdf["boundary_id"] = 1

        # Clean up columns to keep only the boundary geometry and ID
        boundary_gdf = boundary_gdf[["boundary_id", "geometry"]]

        return boundary_gdf

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
