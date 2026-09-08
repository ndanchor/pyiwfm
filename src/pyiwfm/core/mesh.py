"""
Mesh classes for IWFM model representation.

This module provides the core mesh data structures including:

- :class:`Node`: Mesh vertices with coordinates and connectivity
- :class:`Element`: Triangular or quadrilateral finite elements
- :class:`Face`: Element edges for flux calculations
- :class:`Subregion`: Named groups of elements
- :class:`AppGrid`: The complete mesh container (mirrors IWFM's Class_AppGrid)

Example
-------
Create a simple triangular mesh:

>>> from pyiwfm.core.mesh import AppGrid, Node, Element
>>> nodes = {
...     1: Node(id=1, x=0.0, y=0.0),
...     2: Node(id=2, x=100.0, y=0.0),
...     3: Node(id=3, x=50.0, y=100.0),
... }
>>> elements = {
...     1: Element(id=1, vertices=(1, 2, 3), subregion=1),
... }
>>> grid = AppGrid(nodes=nodes, elements=elements)
>>> grid.compute_connectivity()
>>> print(f"Mesh: {grid.n_nodes} nodes, {grid.n_elements} elements")
Mesh: 3 nodes, 1 elements
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pyiwfm.core.exceptions import MeshError


@dataclass
class Node:
    """
    A mesh node (vertex) with coordinates and connectivity information.

    Parameters
    ----------
    id : int
        Unique node identifier (1-based in IWFM).
    x : float
        X coordinate in model units.
    y : float
        Y coordinate in model units.
    area : float, optional
        Node area for water budget calculations (Voronoi-like area).
        Default is 0.0, computed by :meth:`AppGrid.compute_areas`.
    is_boundary : bool, optional
        True if node is on the model boundary. Default is False,
        computed by :meth:`AppGrid.compute_connectivity`.
    connected_nodes : list of int, optional
        IDs of directly connected nodes. Computed automatically.
    surrounding_elements : list of int, optional
        IDs of elements containing this node. Computed automatically.

    Examples
    --------
    Create a simple node:

    >>> node = Node(id=1, x=100.0, y=200.0)
    >>> print(f"Node {node.id} at ({node.x}, {node.y})")
    Node 1 at (100.0, 200.0)

    Create a boundary node:

    >>> boundary_node = Node(id=2, x=0.0, y=0.0, is_boundary=True)
    >>> boundary_node.is_boundary
    True

    Calculate distance between nodes:

    >>> n1 = Node(id=1, x=0.0, y=0.0)
    >>> n2 = Node(id=2, x=3.0, y=4.0)
    >>> n1.distance_to(n2)
    5.0
    """

    id: int
    x: float
    y: float
    area: float = 0.0
    is_boundary: bool = False
    connected_nodes: list[int] = field(default_factory=list)
    surrounding_elements: list[int] = field(default_factory=list)

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return (x, y) coordinate tuple."""
        return (self.x, self.y)

    def distance_to(self, other: Node) -> float:
        """Calculate Euclidean distance to another node."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.id, self.x, self.y))

    def __repr__(self) -> str:
        return f"Node(id={self.id}, x={self.x}, y={self.y})"


@dataclass
class Element:
    """
    A finite element (triangle or quadrilateral).

    IWFM supports both triangular (3 vertices) and quadrilateral (4 vertices)
    elements. Vertices must be ordered counter-clockwise.

    Parameters
    ----------
    id : int
        Unique element identifier (1-based in IWFM).
    vertices : tuple of int
        Node IDs defining the element (3 or 4 nodes, counter-clockwise order).
    subregion : int, optional
        Subregion ID this element belongs to. Default is 0 (unassigned).
    area : float, optional
        Element area in coordinate units squared. Default is 0.0,
        computed by :meth:`AppGrid.compute_areas`.

    Raises
    ------
    MeshError
        If number of vertices is not 3 or 4.

    Examples
    --------
    Create a triangular element:

    >>> tri = Element(id=1, vertices=(1, 2, 3), subregion=1)
    >>> tri.is_triangle
    True
    >>> tri.n_vertices
    3

    Create a quadrilateral element:

    >>> quad = Element(id=2, vertices=(1, 2, 5, 4), subregion=1)
    >>> quad.is_quad
    True
    >>> quad.n_vertices
    4

    Get element edges:

    >>> tri = Element(id=1, vertices=(1, 2, 3))
    >>> tri.edges
    [(1, 2), (2, 3), (3, 1)]
    """

    id: int
    vertices: tuple[int, ...]
    subregion: int = 0
    area: float = 0.0

    def __post_init__(self) -> None:
        """Validate element after initialization."""
        n = len(self.vertices)
        if n < 3:
            raise MeshError(f"Element {self.id}: too few vertices ({n}), minimum is 3")
        if n > 4:
            raise MeshError(f"Element {self.id}: too many vertices ({n}), maximum is 4")

    @property
    def is_triangle(self) -> bool:
        """Return True if element is a triangle."""
        return len(self.vertices) == 3

    @property
    def is_quad(self) -> bool:
        """Return True if element is a quadrilateral."""
        return len(self.vertices) == 4

    @property
    def n_vertices(self) -> int:
        """Return number of vertices."""
        return len(self.vertices)

    @property
    def edges(self) -> list[tuple[int, int]]:
        """
        Return list of edge tuples (node1_id, node2_id).

        Edges are returned in order around the element, closing back to first vertex.
        """
        n = len(self.vertices)
        return [(self.vertices[i], self.vertices[(i + 1) % n]) for i in range(n)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Element):
            return NotImplemented
        return self.id == other.id and self.vertices == other.vertices

    def __hash__(self) -> int:
        return hash((self.id, self.vertices))

    def __repr__(self) -> str:
        return f"Element(id={self.id}, vertices={self.vertices})"


@dataclass
class Face:
    """
    An element face (edge) shared between elements.

    Faces are used for flux calculations between elements.

    Attributes:
        id: Unique face identifier
        nodes: Tuple of two node IDs defining the face
        elements: Tuple of (element1_id, element2_id). Second element is None for boundary.
    """

    id: int
    nodes: tuple[int, int]
    elements: tuple[int, int | None]

    @property
    def is_boundary(self) -> bool:
        """Return True if this face is on the model boundary."""
        return self.elements[1] is None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Face):
            return NotImplemented
        return self.id == other.id and self.nodes == other.nodes

    def __hash__(self) -> int:
        return hash((self.id, self.nodes))

    def __repr__(self) -> str:
        return f"Face(id={self.id}, nodes={self.nodes})"


@dataclass
class Subregion:
    """
    A named group of elements for reporting purposes.

    Attributes:
        id: Unique subregion identifier (1-based)
        name: Descriptive name for the subregion
    """

    id: int
    name: str = ""

    def __repr__(self) -> str:
        return f"Subregion(id={self.id}, name='{self.name}')"


@dataclass
class AppGrid:
    """
    The complete finite element mesh for an IWFM model.

    This class mirrors IWFM's Class_AppGrid and contains all mesh geometry,
    topology, and connectivity information.

    Parameters
    ----------
    nodes : dict
        Dictionary mapping node ID to :class:`Node` object.
    nodes_factor: float, optional
        Length conversion factor read from the preprocessor Nodes file
        header (``FACT``). Already applied to ``Node.x``/``Node.y`` by
        the reader; kept here only as a fallback signal for guessing the
        model's native length unit when ``length_unit`` isn't set.
    length_unit : str, optional
        The model's native (simulation) coordinate length unit,
        ``"FEET"`` or ``"METERS"``, resolved from the PreProcessor main
        file's ``FACTLTOU``/``UNITLTOU`` pair (see
        :func:`pyiwfm.core.units.resolve_model_length_unit`). ``None``
        when unknown. This is the authoritative unit for ``Node.x``/
        ``Node.y`` and is used by :class:`~pyiwfm.visualization.gis_export.GISExporter`
        to align exports with a target CRS.
    elements : dict, optional
        Dictionary mapping element ID to :class:`Element` object.
    faces : dict, optional
        Dictionary mapping face ID to :class:`Face` object.
        Built automatically by :meth:`build_faces`.
    subregions : dict, optional
        Dictionary mapping subregion ID to :class:`Subregion` object.

    Examples
    --------
    Create a 2x2 quad mesh:

    >>> from pyiwfm.core.mesh import AppGrid, Node, Element
    >>> nodes = {
    ...     1: Node(id=1, x=0.0, y=0.0),
    ...     2: Node(id=2, x=100.0, y=0.0),
    ...     3: Node(id=3, x=200.0, y=0.0),
    ...     4: Node(id=4, x=0.0, y=100.0),
    ...     5: Node(id=5, x=100.0, y=100.0),
    ...     6: Node(id=6, x=200.0, y=100.0),
    ... }
    >>> elements = {
    ...     1: Element(id=1, vertices=(1, 2, 5, 4), subregion=1),
    ...     2: Element(id=2, vertices=(2, 3, 6, 5), subregion=1),
    ... }
    >>> grid = AppGrid(nodes=nodes, elements=elements)
    >>> grid.compute_connectivity()
    >>> grid.n_nodes
    6
    >>> grid.n_elements
    2

    Iterate over nodes:

    >>> for node in grid.iter_nodes():
    ...     print(f"Node {node.id}: ({node.x}, {node.y})")
    Node 1: (0.0, 0.0)
    ...

    Get bounding box:

    >>> xmin, ymin, xmax, ymax = grid.bounding_box
    >>> print(f"Extent: ({xmin}, {ymin}) to ({xmax}, {ymax})")
    Extent: (0.0, 0.0) to (200.0, 100.0)
    """

    nodes: dict[int, Node]
    nodes_factor: float | None = None
    length_unit: str | None = None
    elements: dict[int, Element] = field(default_factory=dict)
    faces: dict[int, Face] = field(default_factory=dict)
    subregions: dict[int, Subregion] = field(default_factory=dict)

    # Cached numpy arrays for efficient computation
    _x_cache: NDArray[np.float64] | None = field(default=None, repr=False)
    _y_cache: NDArray[np.float64] | None = field(default=None, repr=False)
    _vertex_cache: NDArray[np.int32] | None = field(default=None, repr=False)
    _node_id_to_idx: dict[int, int] | None = field(default=None, repr=False)
    _centroid_cache: NDArray[np.float64] | None = field(default=None, repr=False)
    _element_id_to_idx: dict[int, int] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal mappings."""
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate cached arrays when mesh changes."""
        self._x_cache = None
        self._y_cache = None
        self._vertex_cache = None
        self._node_id_to_idx = None
        self._centroid_cache = None
        self._element_id_to_idx = None

    def _build_node_mapping(self) -> None:
        """Build mapping from node ID to array index."""
        if self._node_id_to_idx is None:
            sorted_ids = sorted(self.nodes.keys())
            self._node_id_to_idx = {nid: idx for idx, nid in enumerate(sorted_ids)}

    @property
    def n_nodes(self) -> int:
        """Return number of nodes in the mesh."""
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        """Return number of elements in the mesh."""
        return len(self.elements)

    @property
    def n_faces(self) -> int:
        """Return number of faces in the mesh."""
        return len(self.faces)

    @property
    def n_subregions(self) -> int:
        """Return number of subregions."""
        return len(self.subregions)

    @property
    def x(self) -> NDArray[np.float64]:
        """Return x coordinates as numpy array, sorted by node ID."""
        if self._x_cache is None:
            sorted_ids = sorted(self.nodes.keys())
            self._x_cache = np.array([self.nodes[nid].x for nid in sorted_ids])
        return self._x_cache

    @property
    def y(self) -> NDArray[np.float64]:
        """Return y coordinates as numpy array, sorted by node ID."""
        if self._y_cache is None:
            sorted_ids = sorted(self.nodes.keys())
            self._y_cache = np.array([self.nodes[nid].y for nid in sorted_ids])
        return self._y_cache

    @property
    def vertex(self) -> NDArray[np.int32]:
        """
        Return vertex connectivity array.

        Returns array of shape (n_elements, max_vertices) where max_vertices=4.
        For triangles, the 4th vertex is 0 (IWFM convention).
        Values are 0-based indices into the node arrays.
        """
        if self._vertex_cache is None:
            self._build_node_mapping()
            assert self._node_id_to_idx is not None
            sorted_elem_ids = sorted(self.elements.keys())
            n_elem = len(sorted_elem_ids)
            self._vertex_cache = np.zeros((n_elem, 4), dtype=np.int32)

            for i, eid in enumerate(sorted_elem_ids):
                elem = self.elements[eid]
                for j, vid in enumerate(elem.vertices):
                    # Convert node ID to 0-based index
                    self._vertex_cache[i, j] = self._node_id_to_idx[vid]
                # For triangles, 4th vertex remains 0

        return self._vertex_cache

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Return bounding box as (xmin, ymin, xmax, ymax)."""
        x = self.x
        y = self.y
        return (float(x.min()), float(y.min()), float(x.max()), float(y.max()))

    def get_node(self, node_id: int) -> Node:
        """Get a node by ID. Raises KeyError if not found."""
        return self.nodes[node_id]

    def get_element(self, element_id: int) -> Element:
        """Get an element by ID. Raises KeyError if not found."""
        return self.elements[element_id]

    def get_face(self, face_id: int) -> Face:
        """Get a face by ID. Raises KeyError if not found."""
        return self.faces[face_id]

    @property
    def element_centroids(self) -> NDArray[np.float64]:
        """
        Return element centroids as an ``(n_elements, 2)`` array.

        Centroids are computed once (vectorized) on first access and cached
        until the mesh is mutated (see :meth:`_invalidate_cache`). Row ordering
        follows the sorted-by-ID order used by :attr:`vertex`.
        """
        if self._centroid_cache is None:
            self._build_centroid_cache()
        assert self._centroid_cache is not None
        return self._centroid_cache

    def _build_centroid_cache(self) -> None:
        """
        Build the element centroid cache in a single vectorized pass.

        Uses :attr:`vertex` (0-based indices into :attr:`x`/:attr:`y`, with
        the 4th column zero-padded for triangles) and a per-element
        ``n_vertices`` count to mask out triangle padding before averaging.
        """
        sorted_elem_ids = sorted(self.elements.keys())
        n_elem = len(sorted_elem_ids)

        if n_elem == 0:
            self._centroid_cache = np.zeros((0, 2), dtype=np.float64)
            self._element_id_to_idx = {}
            return

        v = self.vertex  # (n_elem, 4)
        x = self.x  # (n_nodes,)
        y = self.y  # (n_nodes,)

        n_vertices = np.fromiter(
            (len(self.elements[eid].vertices) for eid in sorted_elem_ids),
            dtype=np.int32,
            count=n_elem,
        )

        # Mask triangle padding: column j is valid iff j < n_vertices[i]
        col = np.arange(v.shape[1], dtype=np.int32)
        mask = col[np.newaxis, :] < n_vertices[:, np.newaxis]

        vx = x[v]
        vy = y[v]

        centroids = np.empty((n_elem, 2), dtype=np.float64)
        counts = n_vertices.astype(np.float64)
        centroids[:, 0] = np.where(mask, vx, 0.0).sum(axis=1) / counts
        centroids[:, 1] = np.where(mask, vy, 0.0).sum(axis=1) / counts

        self._centroid_cache = centroids
        self._element_id_to_idx = {eid: i for i, eid in enumerate(sorted_elem_ids)}

    def get_element_centroid(self, element_id: int) -> tuple[float, float]:
        """Return the centroid ``(x, y)`` of an element.

        Backed by a cached ``(n_elements, 2)`` array built on first use; scalar
        lookups are O(1) dict + array-index after the first call.
        """
        if self._centroid_cache is None:
            self._build_centroid_cache()
        assert self._element_id_to_idx is not None
        idx = self._element_id_to_idx[element_id]
        cx, cy = self.element_centroids[idx]
        return (float(cx), float(cy))

    def get_elements_in_subregion(self, subregion_id: int) -> list[int]:
        """Return list of element IDs in a subregion."""
        return [eid for eid, elem in self.elements.items() if elem.subregion == subregion_id]

    def get_elements_by_subregion(self) -> dict[int, list[int]]:
        """
        Group all elements by their subregion.

        Returns
        -------
        dict[int, list[int]]
            Dictionary mapping subregion ID to list of element IDs.
            Elements with subregion=0 are grouped under key 0.
        """
        result: dict[int, list[int]] = {}
        for eid, elem in self.elements.items():
            sr_id = elem.subregion
            if sr_id not in result:
                result[sr_id] = []
            result[sr_id].append(eid)
        return result

    def get_subregion_areas(self) -> dict[int, float]:
        """
        Compute total area for each subregion.

        Returns
        -------
        dict[int, float]
            Dictionary mapping subregion ID to total area.
        """
        areas: dict[int, float] = {}
        for elem in self.elements.values():
            sr_id = elem.subregion
            if sr_id not in areas:
                areas[sr_id] = 0.0
            areas[sr_id] += elem.area
        return areas

    def get_element_areas_array(self) -> NDArray[np.float64]:
        """
        Get array of element areas in element ID order.

        Returns
        -------
        NDArray
            Array of areas where index i corresponds to element (i+1).
        """
        import numpy as np

        max_id = max(self.elements.keys()) if self.elements else 0
        areas = np.zeros(max_id)
        for eid, elem in self.elements.items():
            areas[eid - 1] = elem.area
        return areas

    def get_boundary_node_ids(self) -> list[int]:
        """Return list of boundary node IDs."""
        return [nid for nid, node in self.nodes.items() if node.is_boundary]

    def iter_nodes(self) -> Iterator[Node]:
        """Iterate over nodes in ID order."""
        for nid in sorted(self.nodes.keys()):
            yield self.nodes[nid]

    def iter_elements(self) -> Iterator[Element]:
        """Iterate over elements in ID order."""
        for eid in sorted(self.elements.keys()):
            yield self.elements[eid]

    def compute_connectivity(self) -> None:
        """
        Compute node-to-element and node-to-node connectivity.

        This populates:
        - Node.surrounding_elements: Elements containing each node
        - Node.connected_nodes: Nodes directly connected via element edges
        - Node.is_boundary: True if node is on boundary
        """
        # Clear existing connectivity
        for node in self.nodes.values():
            node.surrounding_elements = []
            node.connected_nodes = []
            node.is_boundary = False

        # Use sets for O(1) membership checks during construction
        surround_sets: dict[int, set[int]] = {nid: set() for nid in self.nodes}
        connect_sets: dict[int, set[int]] = {nid: set() for nid in self.nodes}

        # Build node-to-element and node-to-node mappings in a single pass
        for eid, elem in self.elements.items():
            verts = elem.vertices
            for vid in verts:
                surround_sets[vid].add(eid)
            n = len(verts)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        connect_sets[verts[i]].add(verts[j])

        # Convert sets to sorted lists for stable ordering
        for nid, node in self.nodes.items():
            node.surrounding_elements = sorted(surround_sets[nid])
            node.connected_nodes = sorted(connect_sets[nid])

        # Identify boundary nodes by counting edge occurrences
        edge_count: dict[tuple[int, int], int] = {}
        for elem in self.elements.values():
            for n1, n2 in elem.edges:
                # Normalize edge to (min, max) for counting
                edge = (min(n1, n2), max(n1, n2))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear only once
        for edge, count in edge_count.items():
            if count == 1:
                self.nodes[edge[0]].is_boundary = True
                self.nodes[edge[1]].is_boundary = True

    def build_faces(self) -> None:
        """
        Build face (edge) data structure from elements.

        Each unique edge becomes a Face with references to adjacent elements.
        """
        self.faces.clear()

        # Map edges to elements
        edge_to_elements: dict[tuple[int, int], list[int]] = {}

        for eid, elem in self.elements.items():
            for n1, n2 in elem.edges:
                # Normalize edge key
                edge_key = (min(n1, n2), max(n1, n2))
                if edge_key not in edge_to_elements:
                    edge_to_elements[edge_key] = []
                edge_to_elements[edge_key].append(eid)

        # Create faces
        face_id = 1
        for (n1, n2), elems in edge_to_elements.items():
            if len(elems) == 1:
                # Boundary face
                self.faces[face_id] = Face(id=face_id, nodes=(n1, n2), elements=(elems[0], None))
            else:
                # Interior face
                self.faces[face_id] = Face(
                    id=face_id, nodes=(n1, n2), elements=(elems[0], elems[1])
                )
            face_id += 1

    def compute_areas(self) -> None:
        """
        Compute element areas and node areas.

        Element areas are computed using the shoelace formula.
        Node areas are computed by distributing element area equally to vertices.
        """
        # Reset areas
        for node in self.nodes.values():
            node.area = 0.0
        for elem in self.elements.values():
            elem.area = 0.0

        # Compute element areas using shoelace formula
        for _eid, elem in self.elements.items():
            coords = [(self.nodes[vid].x, self.nodes[vid].y) for vid in elem.vertices]
            area = self._polygon_area(coords)
            elem.area = area

            # Distribute area to nodes (equal share)
            n = len(elem.vertices)
            node_share = area / n
            for vid in elem.vertices:
                self.nodes[vid].area += node_share

    @staticmethod
    def _polygon_area(coords: list[tuple[float, float]]) -> float:
        """
        Compute polygon area using shoelace formula.

        Assumes vertices are in counter-clockwise order.
        """
        n = len(coords)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        return abs(area) / 2.0

    def validate(self) -> None:
        """
        Validate mesh integrity.

        Raises:
            MeshError: If mesh is invalid
        """
        if not self.nodes:
            raise MeshError("Mesh has no nodes")

        if not self.elements:
            raise MeshError("Mesh has no elements")

        # Check all vertex references are valid
        node_ids = set(self.nodes.keys())
        for eid, elem in self.elements.items():
            for vid in elem.vertices:
                if vid not in node_ids:
                    raise MeshError(f"Element {eid} has invalid vertex reference: {vid}")

            # Check for duplicate vertices
            if len(set(elem.vertices)) != len(elem.vertices):
                raise MeshError(f"Element {eid} has duplicate vertices")

    def __repr__(self) -> str:
        return (
            f"AppGrid(n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
            f"n_subregions={self.n_subregions})"
        )
