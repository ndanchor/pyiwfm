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

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pyproj
from shapely.geometry import LineString, Point, Polygon

from pyiwfm.core.units import (
    FEET_PER_METER,
    METERS_PER_FOOT,
    normalize_length_unit_name,
)

if TYPE_CHECKING:
    from pyiwfm.components.stream import AppStream
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy


class SpatialUnitWarning(UserWarning):
    """Warned when a GIS export's coordinate unit conversion is uncertain."""


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
        If None, output files will have no CRS defined and node/element
        coordinates are written unconverted.
    model_length_unit : str, optional
        The model's native coordinate length unit ('FEET' or 'METERS'),
        used together with *crs* to convert node/element/stream
        coordinates (which are stored in this unit) into whatever unit
        *crs* expects. If not given, falls back to ``grid.length_unit``
        (set by :meth:`IWFMModel.from_preprocessor` from the
        PreProcessor main file's FACTLTOU/UNITLTOU) and then to a
        best-effort guess from ``grid.nodes_factor``. Pass this
        explicitly when that can't be resolved -- see
        :attr:`resolved_model_length_unit`.

    Attributes
    ----------
    resolved_model_length_unit : str or None
        The model length unit actually used ('FEET', 'METERS', or None
        if it could not be determined).
    resolved_crs_length_unit : str or None
        The target CRS's length unit actually used, or None if *crs* is
        unset or its unit could not be determined.
    adjustment_factor : float
        The multiplicative factor applied to node/element/stream
        coordinates to convert them from ``resolved_model_length_unit``
        to ``resolved_crs_length_unit``. 1.0 when no conversion is
        needed or possible.

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
        crs: str | pyproj.CRS | None = None,
        model_length_unit: str | None = None,
    ) -> None:
        """
        Initialize the GIS exporter.

        Args:
            grid: Model mesh
            stratigraphy: Model stratigraphy (optional)
            streams: Stream network (optional)
            crs: Coordinate reference system (e.g., 'EPSG:26910'). If
                None, no CRS is assigned and coordinates are exported
                unconverted.
            model_length_unit: Explicit override for the model's native
                coordinate length unit ('FEET' or 'METERS'). See the
                class docstring for the fallback order used when this
                is omitted.
        """
        self.grid = grid
        self.stratigraphy = stratigraphy
        self.streams = streams
        self.crs = pyproj.CRS.from_user_input(crs) if crs is not None else None
        self._model_length_unit_override = model_length_unit

        self.resolved_crs_length_unit = self._crs_length_unit()
        self.resolved_model_length_unit = self._model_length_unit()
        self.adjustment_factor = self._calculate_adjustment()

    def _crs_length_unit(self) -> str | None:
        """Return the target CRS's length unit ('FEET'/'METERS'/None)."""
        if self.crs is None:
            return None
        try:
            unit_name = self.crs.axis_info[0].unit_name
        except (AttributeError, IndexError):
            return None
        return normalize_length_unit_name(unit_name)

    def _model_length_unit(self) -> str | None:
        """
        Resolve the model's native coordinate length unit, in priority
        order:

        1. An explicit ``model_length_unit`` passed to the constructor.
        2. ``grid.length_unit``, set by
           :meth:`IWFMModel.from_preprocessor` from the PreProcessor
           main file's FACTLTOU/UNITLTOU pair -- authoritative.
        3. A best-effort guess from the raw Nodes file conversion factor
           (``grid.nodes_factor``) against the feet<->meters conversion
           constants. This only resolves the ambiguity when the factor
           isn't ~1.0 -- a factor of 1.0 genuinely doesn't say what unit
           the (matching) input and internal coordinates are in.

        Returns None if none of the above resolve it.
        """
        override = normalize_length_unit_name(self._model_length_unit_override)
        if override is not None:
            return override

        grid_unit = normalize_length_unit_name(getattr(self.grid, "length_unit", None))
        if grid_unit is not None:
            return grid_unit

        # Fallback: guess from the Nodes file FACT value alone.
        model_factor = getattr(self.grid, "nodes_factor", None)
        if model_factor is None or model_factor <= 0:
            return None
        if math.isclose(model_factor, FEET_PER_METER, rel_tol=1e-3):
            # Raw coordinates were in meters, converted to internal feet.
            return "FEET"
        if math.isclose(model_factor, METERS_PER_FOOT, rel_tol=1e-3):
            # Raw coordinates were in feet, converted to internal meters.
            return "METERS"
        return None

    def _calculate_adjustment(self) -> float:
        """
        Determine the multiplicative factor that converts node/element/
        stream coordinates (stored in the model's native length unit)
        into the units expected by the target CRS.

        Returns 1.0 (no conversion) when no CRS is set. Otherwise, if
        either the CRS's unit or the model's native unit can't be
        determined, also returns 1.0 but emits a :class:`SpatialUnitWarning`
        -- silently assuming no conversion is needed can misplace the
        exported geometry, so callers should heed the warning (or pass
        ``model_length_unit`` explicitly to resolve it).
        """
        if self.crs is None:
            return 1.0

        crs_unit = self.resolved_crs_length_unit
        if crs_unit is None:
            warnings.warn(
                "Could not determine the target CRS's length unit; "
                "assuming it already matches the model and applying no "
                "coordinate conversion. Please verify the specified CRS: "
                "https://spatialreference.org/",
                category=SpatialUnitWarning,
                stacklevel=3,
            )
            return 1.0

        model_unit = self.resolved_model_length_unit
        if model_unit is None:
            warnings.warn(
                "Could not determine the model's native coordinate length "
                "unit: the PreProcessor main file's FACTLTOU/UNITLTOU are "
                "unavailable and the Nodes file conversion factor "
                f"({getattr(self.grid, 'nodes_factor', None)!r}) is "
                "ambiguous (e.g. 1.0). Assuming it already matches the "
                "target CRS and applying no coordinate conversion. Pass "
                "GISExporter(..., model_length_unit='FEET' or 'METERS') "
                "to resolve this explicitly.",
                category=SpatialUnitWarning,
                stacklevel=3,
            )
            return 1.0

        if model_unit == crs_unit:
            return 1.0
        if model_unit == "FEET" and crs_unit == "METERS":
            return METERS_PER_FOOT
        return FEET_PER_METER  # model_unit == "METERS" and crs_unit == "FEET"

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
