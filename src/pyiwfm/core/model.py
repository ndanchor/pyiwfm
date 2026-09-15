"""
IWFMModel class - main orchestrator for IWFM model components.

This module provides the central IWFMModel class that orchestrates all
model components including mesh, stratigraphy, groundwater, streams,
lakes, and root zone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pyiwfm.core.exceptions import (
    ComponentError,
    ComponentLoadError,
    IWFMIOError,
    ValidationError,
)
from pyiwfm.core.model_factory import (
    apply_kh_anomalies as _apply_kh_anomalies,
)
from pyiwfm.core.model_factory import (
    apply_parametric_grids as _apply_parametric_grids,
)
from pyiwfm.core.model_factory import (
    apply_parametric_subsidence as _apply_parametric_subsidence,
)
from pyiwfm.core.model_factory import (
    binary_data_to_model as _binary_data_to_model,
)
from pyiwfm.core.model_factory import (
    build_reaches_from_node_reach_ids as _build_reaches_from_node_reach_ids,
)
from pyiwfm.core.model_factory import (
    resolve_stream_node_coordinates as _resolve_stream_node_coordinates,
)

logger = logging.getLogger(__name__)

# Exceptions we expect during component-file parsing. Anything outside this
# tuple is a programmer error (TypeError, AttributeError, NameError, etc.)
# and should bubble up rather than be swallowed.
#
# Includes:
#   - Python builtins for I/O / parsing failures (OSError covers
#     FileNotFoundError + permission errors; ValueError covers bad numeric
#     conversions; KeyError / IndexError cover malformed dicts/arrays).
#   - ImportError: pyiwfm has many optional dependencies (triangle, gmsh,
#     dss, vtk, etc.) and component readers may import them lazily. A
#     missing optional dep is a legitimate "feature unavailable" condition,
#     not a programmer error.
#   - pyiwfm's own IWFMIOError (parent of FileFormatError) and
#     ComponentError, raised by domain readers for format-specific issues.
_COMPONENT_LOAD_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    KeyError,
    IndexError,
    UnicodeDecodeError,
    ImportError,
    IWFMIOError,
    ComponentError,
)


def _record_component_failure(
    model: IWFMModel,
    component_name: str,
    source_file: Path | None,
    exc: BaseException,
    *,
    strict: bool,
) -> None:
    """Common handling for a failed component load.

    With ``strict=False`` (default): logs a structured warning with full
    traceback and stores the error in ``model.metadata`` for backward
    compatibility. The model retains whatever components loaded
    successfully.

    With ``strict=True``: logs the warning, then raises
    :class:`ComponentLoadError` chained from ``exc``. Callers that need a
    complete model (calibration, analysis pipelines) should pass this.
    """
    logger.warning(
        "Failed to load %s component from %s: %s: %s",
        component_name,
        source_file,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    model.metadata[f"{component_name}_load_error"] = f"{type(exc).__name__}: {exc}"
    if strict:
        raise ComponentLoadError(component_name, source_file, exc) from exc


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyiwfm.components.groundwater import AppGW
    from pyiwfm.components.lake import AppLake
    from pyiwfm.components.rootzone import RootZone
    from pyiwfm.components.small_watershed import AppSmallWatershed
    from pyiwfm.components.stream import AppStream
    from pyiwfm.components.unsaturated_zone import AppUnsatZone
    from pyiwfm.core.mesh import AppGrid
    from pyiwfm.core.stratigraphy import Stratigraphy
    from pyiwfm.io.supply_adjust import SupplyAdjustment


@dataclass
class IWFMModel:
    """
    The main IWFM model container class.

    This class orchestrates all model components and provides methods for
    reading, writing, and validating IWFM models. It mirrors the structure
    of IWFM's Package_Model.

    Attributes:
        name: Model name/identifier
        mesh: Finite element mesh (AppGrid)
        stratigraphy: Vertical layering structure
        groundwater: Groundwater component (AppGW) - wells, pumping, BCs, aquifer params
        streams: Stream network component (AppStream) - nodes, reaches, diversions, bypasses
        lakes: Lake component (AppLake) - lake definitions, elements, rating curves
        rootzone: Root zone component (RootZone) - crop types, soil params, land use
        small_watersheds: Small Watershed component (AppSmallWatershed)
        unsaturated_zone: Unsaturated Zone component (AppUnsatZone)
        supply_adjustment: Parsed supply adjustment specification data
        metadata: Additional model metadata
    """

    name: str
    mesh: AppGrid | None = None
    stratigraphy: Stratigraphy | None = None
    groundwater: AppGW | None = None
    streams: AppStream | None = None
    lakes: AppLake | None = None
    rootzone: RootZone | None = None
    small_watersheds: AppSmallWatershed | None = None
    unsaturated_zone: AppUnsatZone | None = None
    supply_adjustment: SupplyAdjustment | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_files: dict[str, Path] = field(default_factory=dict)

    # Components mutated since the last load/save. Populated by the
    # mutation helpers (set_aquifer_parameter, set_stratigraphy_*,
    # add_observation_well, etc.) so save_complete_model can warn or
    # log if a dirty component fails to write. Not user-facing state —
    # tests should reach into it directly.
    _dirty: set[str] = field(default_factory=set, repr=False)

    # ========================================================================
    # Class Methods for Loading Models
    # ========================================================================

    @classmethod
    def from_preprocessor(
        cls,
        pp_file: Path | str,
        load_streams: bool = True,
        load_lakes: bool = True,
        *,
        strict: bool = False,
    ) -> IWFMModel:
        """
        Load a model from PreProcessor input files.

        This loads the model structure (mesh, stratigraphy) and optionally
        the stream and lake geometry from the preprocessor input file and
        all files referenced by it.

        Note: This creates a "partial" model with only the static geometry
        defined in the preprocessor. It does not include dynamic components
        like groundwater parameters, pumping, or root zone data which are
        defined in the simulation input files.

        Args:
            pp_file: Path to the main PreProcessor input file
            load_streams: If True, load stream network geometry
            load_lakes: If True, load lake geometry
            strict: If True, raise :class:`~pyiwfm.core.exceptions.ComponentLoadError`
                when an optional component file cannot be parsed. The default
                (``False``) preserves the historical lenient behavior: log a
                warning, record the error in ``model.metadata``, and continue
                with whatever components loaded. Use ``strict=True`` from
                calibration / analysis pipelines that require a complete model.

        Returns:
            IWFMModel instance with mesh, stratigraphy, and optionally
            streams/lakes geometry loaded

        Example:
            >>> model = IWFMModel.from_preprocessor("Preprocessor/Preprocessor.in")
            >>> print(f"Loaded {model.n_nodes} nodes, {model.n_elements} elements")
        """
        from pyiwfm.core.mesh import AppGrid, Subregion
        from pyiwfm.io.ascii import read_elements, read_nodes, read_stratigraphy
        from pyiwfm.io.preprocessor import (
            read_preprocessor_main,
            read_subregions_file,
        )

        pp_file = Path(pp_file)
        config = read_preprocessor_main(pp_file)

        # Read nodes
        if config.nodes_file is None:
            from pyiwfm.core.exceptions import FileFormatError

            raise FileFormatError("Nodes file not specified in PreProcessor file")
        nodes, nodes_factor = read_nodes(config.nodes_file)

        # Read elements
        if config.elements_file is None:
            from pyiwfm.core.exceptions import FileFormatError

            raise FileFormatError("Elements file not specified in PreProcessor file")
        elements, n_subregions, subregion_names = read_elements(config.elements_file)

        # Read subregions: prefer separate file, fall back to names from element file
        subregions: dict[int, Subregion] = {}
        if config.subregions_file and config.subregions_file.exists():
            subregions = read_subregions_file(config.subregions_file)
        elif subregion_names:
            subregions = {
                sr_id: Subregion(id=sr_id, name=name) for sr_id, name in subregion_names.items()
            }

        # Create mesh
        mesh = AppGrid(
            nodes=nodes,
            elements=elements,
            subregions=subregions,
            nodes_factor=nodes_factor,
            length_unit=config.simulation_length_unit,
        )
        mesh.compute_areas()
        mesh.compute_connectivity()

        # Read stratigraphy
        stratigraphy = None
        if config.stratigraphy_file and config.stratigraphy_file.exists():
            stratigraphy = read_stratigraphy(config.stratigraphy_file)

        # Create model
        model = cls(
            name=config.model_name or pp_file.stem,
            mesh=mesh,
            stratigraphy=stratigraphy,
            metadata={
                "source": "preprocessor",
                "preprocessor_file": str(pp_file),
                "length_unit": config.length_unit,
                "simulation_length_unit": config.simulation_length_unit,
                "area_unit": config.area_unit,
                "volume_unit": config.volume_unit,
            },
        )

        # Load stream geometry if requested
        if load_streams and config.streams_file and config.streams_file.exists():
            try:
                from pyiwfm.components.stream import AppStream, StrmNode, StrmReach
                from pyiwfm.io.streams import StreamReader, StreamSpecReader

                stream = AppStream()
                first_error: Exception | None = None

                # Try simple stream-nodes format first
                try:
                    reader = StreamReader()
                    nodes_dict = reader.read_stream_nodes(config.streams_file)
                    for node in nodes_dict.values():
                        stream.add_node(node)
                except Exception as exc:
                    first_error = exc

                # Fallback: parse as StreamsSpec (reach-based format)
                if not stream.nodes:
                    try:
                        spec_reader = StreamSpecReader()
                        n_reaches, _n_rt, reach_specs = spec_reader.read(config.streams_file)
                        for rs in reach_specs:
                            for sn_id in rs.node_ids:
                                if sn_id not in stream.nodes:
                                    gw = rs.node_to_gw_node.get(sn_id)
                                    node = StrmNode(
                                        id=sn_id,
                                        x=0.0,
                                        y=0.0,
                                        reach_id=rs.id,
                                        gw_node=gw if gw and gw > 0 else None,
                                    )
                                    # Transfer bottom elevation from rating table section
                                    if sn_id in rs.node_bottom_elevations:
                                        node.bottom_elev = rs.node_bottom_elevations[sn_id]
                                    # Transfer rating table
                                    if sn_id in rs.node_rating_tables:
                                        import numpy as np

                                        from pyiwfm.components.stream import StreamRating

                                        stages, flows = rs.node_rating_tables[sn_id]
                                        node.rating = StreamRating(
                                            stages=np.array(stages, dtype=np.float64),
                                            flows=np.array(flows, dtype=np.float64),
                                        )
                                    stream.add_node(node)
                            stream.add_reach(
                                StrmReach(
                                    id=rs.id,
                                    upstream_node=rs.node_ids[0] if rs.node_ids else 0,
                                    downstream_node=rs.node_ids[-1] if rs.node_ids else 0,
                                    nodes=list(rs.node_ids),
                                    name=rs.name,
                                )
                            )
                    except Exception:
                        # Both paths failed — re-raise original error
                        if first_error is not None:
                            raise first_error from None

                # Safety net: build reaches from node reach_ids
                _build_reaches_from_node_reach_ids(stream)

                model.streams = stream
            except _COMPONENT_LOAD_EXCEPTIONS as e:
                _record_component_failure(model, "streams", config.streams_file, e, strict=strict)

        # Load lake geometry if requested
        if load_lakes and config.lakes_file and config.lakes_file.exists():
            try:
                from pyiwfm.components.lake import AppLake, LakeElement
                from pyiwfm.io.lakes import LakeReader

                lake_reader = LakeReader()
                lakes_dict = lake_reader.read_lake_definitions(config.lakes_file)

                lakes = AppLake()
                for lake in lakes_dict.values():
                    lakes.add_lake(lake)
                    for elem_id in lake.elements:
                        lakes.add_lake_element(LakeElement(element_id=elem_id, lake_id=lake.id))

                model.lakes = lakes
            except _COMPONENT_LOAD_EXCEPTIONS as e:
                _record_component_failure(model, "lakes", config.lakes_file, e, strict=strict)

        model.metadata["source"] = "preprocessor"
        _resolve_stream_node_coordinates(model)
        return model

    @classmethod
    def from_preprocessor_binary(
        cls,
        binary_file: Path | str,
        name: str = "",
    ) -> IWFMModel:
        """Load a model from the native IWFM PreProcessor binary output.

        The preprocessor binary file (``ACCESS='STREAM'``) contains mesh,
        stratigraphy, stream/lake connectors, and component data compiled
        by the IWFM PreProcessor.

        Args:
            binary_file: Path to the preprocessor binary output file
            name: Model name (optional, defaults to file stem)

        Returns:
            IWFMModel with mesh, stratigraphy, streams, and lakes loaded

        Example:
            >>> model = IWFMModel.from_preprocessor_binary("PreprocessorOut.bin")
            >>> print(f"Loaded {model.n_nodes} nodes, {model.n_layers} layers")
        """
        from pyiwfm.io.preprocessor_binary import PreprocessorBinaryReader

        binary_file = Path(binary_file)
        reader = PreprocessorBinaryReader()
        data = reader.read(binary_file)
        model = _binary_data_to_model(data, name=name or binary_file.stem)
        model.metadata["binary_file"] = str(binary_file)
        _resolve_stream_node_coordinates(model)
        return model

    @classmethod
    def from_simulation(
        cls,
        simulation_file: Path | str,
    ) -> IWFMModel:
        """Load a complete IWFM model from a simulation main input file.

        Delegates to :class:`~pyiwfm.io.model_loader.CompleteModelLoader`
        which auto-detects the simulation file format and loads all
        components (mesh, stratigraphy, groundwater, streams, lakes,
        root zone, etc.).

        Args:
            simulation_file: Path to the simulation main input file

        Returns:
            IWFMModel instance with all components loaded

        Example:
            >>> model = IWFMModel.from_simulation("Simulation/Simulation.in")
            >>> print(f"Stream nodes: {len(model.streams.nodes)}")
        """
        from pyiwfm.io.model_loader import load_complete_model

        return load_complete_model(simulation_file)

    @classmethod
    def from_simulation_with_preprocessor(
        cls,
        simulation_file: Path | str,
        preprocessor_file: Path | str,
        load_timeseries: bool = False,
        *,
        strict: bool = False,
    ) -> IWFMModel:
        """
        Load a complete IWFM model using both simulation and preprocessor files.

        This method first loads the mesh and stratigraphy from the preprocessor
        input file (ASCII format), then loads all dynamic components from the
        simulation input file and its referenced component files.

        Use this method when:
        - You have both preprocessor input files and simulation input files
        - You want to load from ASCII preprocessor files rather than binary
        - The binary file path in the simulation file is incorrect or missing

        Args:
            simulation_file: Path to the simulation main input file
            preprocessor_file: Path to the preprocessor main input file
            load_timeseries: If True, also load time series data (slower)
            strict: If True, raise :class:`~pyiwfm.core.exceptions.ComponentLoadError`
                when an optional component (groundwater, streams, lakes, rootzone,
                small watersheds, unsaturated zone) cannot be parsed. The default
                (``False``) preserves the historical lenient behavior: log a
                structured warning, record the error in ``model.metadata``, and
                continue with whatever components loaded. Use ``strict=True`` from
                calibration / analysis pipelines that require a complete model.

        Returns:
            IWFMModel instance with all components loaded

        Example:
            >>> model = IWFMModel.from_simulation_with_preprocessor(
            ...     "Simulation/Simulation.in",
            ...     "Preprocessor/Preprocessor.in"
            ... )
        """
        from pyiwfm.io.groundwater import GroundwaterReader, GWMainFileReader
        from pyiwfm.io.gw_boundary import GWBoundaryReader
        from pyiwfm.io.gw_pumping import PumpingReader
        from pyiwfm.io.gw_subsidence import SubsidenceReader
        from pyiwfm.io.gw_tiledrain import TileDrainReader
        from pyiwfm.io.lakes import LakeMainFileReader, LakeReader
        from pyiwfm.io.preprocessor import _resolve_path
        from pyiwfm.io.rootzone import RootZoneMainFileReader, RootZoneReader
        from pyiwfm.io.simulation import SimulationReader
        from pyiwfm.io.stream_bypass import BypassSpecReader
        from pyiwfm.io.stream_diversion import DiversionSpecReader
        from pyiwfm.io.stream_inflow import InflowReader
        from pyiwfm.io.streams import StreamMainFileReader, StreamReader, StreamSpecReader

        # First load mesh and stratigraphy from preprocessor
        model = cls.from_preprocessor(preprocessor_file)

        # Now read simulation config and load dynamic components
        simulation_file = Path(simulation_file)
        base_dir = simulation_file.parent

        sim_reader = SimulationReader()
        sim_config = sim_reader.read(simulation_file)

        # Store source file paths for later write operations
        model.source_files["simulation_main"] = simulation_file
        model.source_files["preprocessor_main"] = Path(preprocessor_file)

        # Update metadata
        model.metadata["source"] = "simulation_with_preprocessor"
        model.metadata["simulation_file"] = str(simulation_file)
        model.metadata["preprocessor_file"] = str(preprocessor_file)
        model.metadata["start_date"] = sim_config.start_date.isoformat()
        model.metadata["end_date"] = sim_config.end_date.isoformat()
        model.metadata["time_step_length"] = sim_config.time_step_length
        model.metadata["time_step_unit"] = sim_config.time_step_unit.value

        # Store solver parameters
        model.metadata["matrix_solver"] = sim_config.matrix_solver
        model.metadata["relaxation"] = sim_config.relaxation
        model.metadata["max_iterations"] = sim_config.max_iterations
        model.metadata["max_supply_iterations"] = sim_config.max_supply_iterations
        model.metadata["convergence_tolerance"] = sim_config.convergence_tolerance
        model.metadata["convergence_volume"] = sim_config.convergence_volume
        model.metadata["convergence_supply"] = sim_config.convergence_supply
        model.metadata["supply_adjust_option"] = sim_config.supply_adjust_option
        model.metadata["debug_flag"] = sim_config.debug_flag
        model.metadata["cache_size"] = sim_config.cache_size

        # Store additional file paths (resolve to absolute using base_dir)
        if sim_config.binary_preprocessor_file:
            model.metadata["binary_preprocessor_file"] = str(sim_config.binary_preprocessor_file)
            model.source_files["binary_preprocessor"] = _resolve_path(
                base_dir, str(sim_config.binary_preprocessor_file)
            )
        if sim_config.irrigation_fractions_file:
            model.metadata["irrigation_fractions_file"] = str(sim_config.irrigation_fractions_file)
            model.source_files["irig_frac_ts"] = _resolve_path(
                base_dir, str(sim_config.irrigation_fractions_file)
            )
        if sim_config.supply_adjust_file:
            model.metadata["supply_adjust_file"] = str(sim_config.supply_adjust_file)
            sa_path = _resolve_path(base_dir, str(sim_config.supply_adjust_file))
            model.source_files["supply_adjust"] = sa_path
            if sa_path.exists():
                try:
                    from pyiwfm.io.supply_adjust import read_supply_adjustment

                    model.supply_adjustment = read_supply_adjustment(sa_path)
                except _COMPONENT_LOAD_EXCEPTIONS as e:
                    _record_component_failure(model, "supply_adjustment", sa_path, e, strict=strict)
        if sim_config.precipitation_file:
            model.metadata["precipitation_file"] = str(sim_config.precipitation_file)
            model.source_files["precipitation_ts"] = _resolve_path(
                base_dir, str(sim_config.precipitation_file)
            )
        if sim_config.et_file:
            model.metadata["et_file"] = str(sim_config.et_file)
            model.source_files["et_ts"] = _resolve_path(base_dir, str(sim_config.et_file))
        if sim_config.title_lines:
            model.metadata["title_lines"] = sim_config.title_lines

        # Load groundwater component using hierarchical reader
        if sim_config.groundwater_file:
            gw_file = _resolve_path(base_dir, str(sim_config.groundwater_file))
            model.source_files["gw_main"] = gw_file
            if gw_file.exists():
                try:
                    from pyiwfm.components.groundwater import AppGW

                    n_nodes = model.mesh.n_nodes if model.mesh else 0
                    n_elements = model.mesh.n_elements if model.mesh else 0
                    n_layers = model.n_layers

                    gw = AppGW(n_nodes=n_nodes, n_layers=n_layers, n_elements=n_elements)

                    # Try hierarchical reader first (for component main files)
                    try:
                        gw_main_reader = GWMainFileReader()
                        gw_config = gw_main_reader.read(gw_file, base_dir=base_dir)

                        # Add hydrograph locations from main file
                        for loc in gw_config.hydrograph_locations:
                            gw.add_hydrograph_location(loc)

                        # Store output file paths as metadata
                        model.metadata["gw_version"] = gw_config.version
                        if gw_config.budget_output_file:
                            model.metadata["gw_budget_file"] = str(gw_config.budget_output_file)
                        if gw_config.zbudget_output_file:
                            model.metadata["gw_zbudget_file"] = str(gw_config.zbudget_output_file)
                        if gw_config.head_all_output_file:
                            model.metadata["gw_head_all_file"] = str(gw_config.head_all_output_file)
                        if gw_config.hydrograph_output_file:
                            model.metadata["gw_hydrograph_file"] = str(
                                gw_config.hydrograph_output_file
                            )

                        # Store GW-specific output units for budget display
                        if gw_config.volume_output_unit:
                            model.metadata["gw_volume_output_unit"] = gw_config.volume_output_unit
                        if gw_config.head_output_unit:
                            model.metadata["gw_length_output_unit"] = gw_config.head_output_unit

                        # Load boundary conditions from sub-file
                        if gw_config.bc_file:
                            model.source_files["gw_bc_main"] = gw_config.bc_file
                        if gw_config.bc_file and gw_config.bc_file.exists():
                            try:
                                bc_reader = GWBoundaryReader()
                                bc_config = bc_reader.read(
                                    gw_config.bc_file,
                                    base_dir=base_dir,
                                )
                                model.metadata["gw_n_specified_flow_bc"] = (
                                    bc_config.n_specified_flow
                                )
                                model.metadata["gw_n_specified_head_bc"] = (
                                    bc_config.n_specified_head
                                )
                                model.metadata["gw_n_general_head_bc"] = bc_config.n_general_head
                                model.metadata["gw_n_constrained_gh_bc"] = (
                                    bc_config.n_constrained_gh
                                )
                                if bc_config.ts_data_file:
                                    model.source_files["gw_bc_ts"] = bc_config.ts_data_file

                                # Add BCs to the GW component
                                from pyiwfm.components.groundwater import BoundaryCondition

                                for sh_bc in bc_config.specified_head_bcs:
                                    gw.add_boundary_condition(
                                        BoundaryCondition(
                                            id=sh_bc.node_id,
                                            bc_type="specified_head",
                                            nodes=[sh_bc.node_id],
                                            values=[sh_bc.head_value],
                                            layer=sh_bc.layer,
                                            ts_column=sh_bc.ts_column,
                                        )
                                    )
                                for sf_bc in bc_config.specified_flow_bcs:
                                    gw.add_boundary_condition(
                                        BoundaryCondition(
                                            id=sf_bc.node_id,
                                            bc_type="specified_flow",
                                            nodes=[sf_bc.node_id],
                                            values=[sf_bc.base_flow],
                                            layer=sf_bc.layer,
                                            ts_column=sf_bc.ts_column,
                                        )
                                    )
                                for gh_bc in bc_config.general_head_bcs:
                                    gw.add_boundary_condition(
                                        BoundaryCondition(
                                            id=gh_bc.node_id,
                                            bc_type="general_head",
                                            nodes=[gh_bc.node_id],
                                            values=[gh_bc.external_head],
                                            layer=gh_bc.layer,
                                            conductance=[gh_bc.conductance],
                                        )
                                    )
                                for cgh_bc in bc_config.constrained_gh_bcs:
                                    gw.add_boundary_condition(
                                        BoundaryCondition(
                                            id=cgh_bc.node_id,
                                            bc_type="constrained_general_head",
                                            nodes=[cgh_bc.node_id],
                                            values=[cgh_bc.external_head],
                                            layer=cgh_bc.layer,
                                            conductance=[cgh_bc.conductance],
                                            constraining_head=cgh_bc.constraining_head,
                                            max_flow=cgh_bc.max_flow,
                                            ts_column=cgh_bc.ts_column,
                                            max_flow_ts_column=cgh_bc.max_flow_ts_column,
                                        )
                                    )
                                # Store BC config for roundtrip fidelity
                                gw.bc_config = bc_config
                                # Store BC time series file path
                                if bc_config.ts_data_file:
                                    gw.bc_ts_file = bc_config.ts_data_file
                                # Store NOUTB section for roundtrip fidelity
                                if bc_config.n_bc_output_nodes > 0:
                                    gw.n_bc_output_nodes = bc_config.n_bc_output_nodes
                                    gw.bc_output_specs = list(bc_config.bc_output_specs)
                                    gw.bc_output_file_raw = bc_config.bc_output_file_raw
                                    if bc_config.bc_output_file:
                                        gw.bc_output_file = str(bc_config.bc_output_file)
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                pass

                        # Load pumping from sub-file
                        if gw_config.pumping_file:
                            model.source_files["gw_pumping_main"] = gw_config.pumping_file
                        if gw_config.pumping_file and gw_config.pumping_file.exists():
                            try:
                                pump_reader = PumpingReader()
                                pump_config = pump_reader.read(
                                    gw_config.pumping_file,
                                    base_dir=base_dir,
                                    n_layers=n_layers,
                                )
                                model.metadata["gw_n_wells"] = pump_config.n_wells
                                model.metadata["gw_n_elem_pumping"] = pump_config.n_elem_pumping
                                if pump_config.ts_data_file:
                                    model.source_files["gw_pumping_ts"] = pump_config.ts_data_file

                                # Convert well specs to Well objects
                                from pyiwfm.components.groundwater import (
                                    ElementPumping,
                                    Well,
                                )

                                for ws in pump_config.well_specs:
                                    gw.add_well(
                                        Well(
                                            id=ws.id,
                                            x=ws.x,
                                            y=ws.y,
                                            element=0,
                                            name=ws.name,
                                            top_screen=ws.perf_top,
                                            bottom_screen=ws.perf_bottom,
                                            radius=ws.radius,
                                        )
                                    )

                                # Merge pumping spec data onto wells
                                for wps in pump_config.well_pumping_specs:
                                    well = gw.wells.get(wps.well_id)
                                    if well is not None:
                                        well.pump_column = wps.pump_column
                                        well.pump_fraction = wps.pump_fraction
                                        well.dist_method = wps.dist_method
                                        well.dest_type = wps.dest_type
                                        well.dest_id = wps.dest_id
                                        well.irig_frac_column = wps.irig_frac_column
                                        well.adjust_column = wps.adjust_column
                                        well.pump_max_column = wps.pump_max_column
                                        well.pump_max_fraction = wps.pump_max_fraction

                                # Convert element pumping specs
                                for eps in pump_config.elem_pumping_specs:
                                    gw.add_element_pumping(
                                        ElementPumping(
                                            element_id=eps.element_id,
                                            layer=0,
                                            pump_rate=0.0,
                                            layer_fraction=1.0,
                                            pump_column=eps.pump_column,
                                            pump_fraction=eps.pump_fraction,
                                            dist_method=eps.dist_method,
                                            layer_factors=eps.layer_factors,
                                            dest_type=eps.dest_type,
                                            dest_id=eps.dest_id,
                                            irig_frac_column=eps.irig_frac_column,
                                            adjust_column=eps.adjust_column,
                                            pump_max_column=eps.pump_max_column,
                                            pump_max_fraction=eps.pump_max_fraction,
                                        )
                                    )

                                # Store pumping TS file path
                                if pump_config.ts_data_file:
                                    gw.pumping_ts_file = pump_config.ts_data_file
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                # Fall back to simple well reader
                                try:
                                    gw_reader = GroundwaterReader()
                                    wells = gw_reader.read_wells(gw_config.pumping_file)
                                    for well in wells.values():
                                        gw.add_well(well)
                                except _COMPONENT_LOAD_EXCEPTIONS:
                                    pass

                        # Load tile drains from sub-file
                        if gw_config.tile_drain_file:
                            model.source_files["gw_tile_drain"] = gw_config.tile_drain_file
                        if gw_config.tile_drain_file and gw_config.tile_drain_file.exists():
                            try:
                                td_reader = TileDrainReader()
                                td_config = td_reader.read(gw_config.tile_drain_file)
                                model.metadata["gw_n_tile_drains"] = td_config.n_drains
                                model.metadata["gw_n_sub_irrigation"] = td_config.n_sub_irrigation

                                from pyiwfm.components.groundwater import (
                                    SubIrrigation,
                                    TileDrain,
                                )

                                for td in td_config.tile_drains:
                                    # IWFM TYPDST: 0=outside, 1=stream node
                                    is_stream = td.dest_type == 1
                                    dest_type = "stream" if is_stream else "outside"
                                    gw.add_tile_drain(
                                        TileDrain(
                                            id=td.id,
                                            element=td.gw_node,
                                            elevation=td.elevation,
                                            conductance=td.conductance,
                                            destination_type=dest_type,
                                            destination_id=td.dest_id if is_stream else None,
                                        )
                                    )
                                # Load sub-irrigation data
                                for si in td_config.sub_irrigations:
                                    gw.add_sub_irrigation(
                                        SubIrrigation(
                                            id=si.id,
                                            gw_node=si.gw_node,
                                            elevation=si.elevation,
                                            conductance=si.conductance,
                                        )
                                    )
                                # Preserve conversion factors for roundtrip writing
                                gw.td_elev_factor = td_config.drain_height_factor
                                gw.td_cond_factor = td_config.drain_conductance_factor
                                gw.td_time_unit = td_config.drain_time_unit
                                gw.si_elev_factor = td_config.subirig_height_factor
                                gw.si_cond_factor = td_config.subirig_conductance_factor
                                gw.si_time_unit = td_config.subirig_time_unit
                                # Preserve hydrograph output section
                                gw.td_n_hydro = td_config.n_td_hydro
                                gw.td_hydro_volume_factor = td_config.td_hydro_volume_factor
                                gw.td_hydro_volume_unit = td_config.td_hydro_volume_unit
                                gw.td_output_file_raw = td_config.td_output_file
                                if td_config.td_output_file:
                                    model.metadata["tile_drain_hydrograph_file"] = str(
                                        td_config.td_output_file
                                    )
                                model.metadata["tile_drain_n_hydrograph_outputs"] = (
                                    td_config.n_td_hydro
                                )
                                gw.td_hydro_specs = [
                                    {
                                        "id": s.id,
                                        "id_type": s.id_type,
                                        "name": s.name,
                                    }
                                    for s in td_config.td_hydro_specs
                                ]
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                pass

                        # Load subsidence from sub-file
                        if gw_config.subsidence_file:
                            model.source_files["gw_subsidence"] = gw_config.subsidence_file
                        if gw_config.subsidence_file and gw_config.subsidence_file.exists():
                            try:
                                subs_reader = SubsidenceReader()
                                subs_config = subs_reader.read(
                                    gw_config.subsidence_file,
                                    base_dir=base_dir,
                                    n_nodes=n_nodes,
                                    n_layers=n_layers,
                                )
                                model.metadata["gw_subsidence_version"] = subs_config.version
                                model.metadata["gw_subsidence_n_nodes"] = len(
                                    subs_config.node_params
                                )

                                # Store full config for roundtrip
                                gw.subsidence_config = subs_config

                                # Interpolate parametric grids if needed
                                if (
                                    not subs_config.node_params
                                    and subs_config.parametric_grids
                                    and model.mesh
                                ):
                                    subs_config.node_params = _apply_parametric_subsidence(
                                        subs_config,
                                        model.mesh,
                                        n_nodes,
                                        n_layers,
                                    )
                                    model.metadata["gw_subsidence_n_nodes"] = len(
                                        subs_config.node_params
                                    )

                                # Populate NodeSubsidence objects
                                from pyiwfm.components.groundwater import (
                                    NodeSubsidence,
                                )
                                from pyiwfm.components.groundwater import (
                                    Subsidence as SubsidenceComp,
                                )

                                for sub_p in subs_config.node_params:
                                    gw.add_node_subsidence(
                                        NodeSubsidence(
                                            node_id=sub_p.node_id,
                                            elastic_sc=sub_p.elastic_sc,
                                            inelastic_sc=sub_p.inelastic_sc,
                                            interbed_thick=sub_p.interbed_thick,
                                            interbed_thick_min=sub_p.interbed_thick_min,
                                            precompact_head=sub_p.precompact_head,
                                            kv_sub=sub_p.kv_sub,
                                            n_eq=sub_p.n_eq,
                                        )
                                    )
                                    # Also populate legacy Subsidence list
                                    for layer_idx in range(len(sub_p.elastic_sc)):
                                        gw.add_subsidence(
                                            SubsidenceComp(
                                                element=sub_p.node_id,
                                                layer=layer_idx + 1,
                                                elastic_storage=sub_p.elastic_sc[layer_idx],
                                                inelastic_storage=sub_p.inelastic_sc[layer_idx],
                                                preconsolidation_head=sub_p.precompact_head[
                                                    layer_idx
                                                ]
                                                if layer_idx < len(sub_p.precompact_head)
                                                else 0.0,
                                                interbed_thick=sub_p.interbed_thick[layer_idx]
                                                if layer_idx < len(sub_p.interbed_thick)
                                                else 0.0,
                                                interbed_thick_min=sub_p.interbed_thick_min[
                                                    layer_idx
                                                ]
                                                if layer_idx < len(sub_p.interbed_thick_min)
                                                else 0.0,
                                            )
                                        )

                                # Store subsidence hydrograph output metadata
                                if subs_config.hydrograph_output_file:
                                    model.metadata["subsidence_hydrograph_file"] = str(
                                        subs_config.hydrograph_output_file
                                    )
                                model.metadata["subsidence_n_hydrograph_outputs"] = (
                                    subs_config.n_hydrograph_outputs
                                )
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                pass

                        # Load aquifer parameters (inline in GW main file)
                        if gw_config.aquifer_params is not None:
                            try:
                                gw.set_aquifer_parameters(gw_config.aquifer_params)
                                model.metadata["gw_aquifer_params_loaded"] = True
                                model.metadata["gw_aquifer_n_nodes"] = (
                                    gw_config.aquifer_params.n_nodes
                                )
                                model.metadata["gw_aquifer_n_layers"] = (
                                    gw_config.aquifer_params.n_layers
                                )
                            except ValueError:
                                # n_nodes/n_layers mismatch — store anyway
                                gw.aquifer_params = gw_config.aquifer_params
                                model.metadata["gw_aquifer_params_loaded"] = True
                                model.metadata["gw_aquifer_params_mismatch"] = True
                        elif gw_config.parametric_grids and model.mesh:
                            # Parametric grid: interpolate onto model nodes
                            try:
                                ok = _apply_parametric_grids(
                                    gw,
                                    gw_config.parametric_grids,
                                    model.mesh,
                                )
                                if ok:
                                    model.metadata["gw_aquifer_params_loaded"] = True
                                    model.metadata["gw_parametric_grids"] = len(
                                        gw_config.parametric_grids
                                    )
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                pass

                        # Apply Kh anomaly overwrites
                        if gw_config.kh_anomalies and gw.aquifer_params is not None and model.mesh:
                            try:
                                applied = _apply_kh_anomalies(
                                    gw.aquifer_params,
                                    gw_config.kh_anomalies,
                                    model.mesh,
                                )
                                model.metadata["gw_kh_anomaly_count"] = len(gw_config.kh_anomalies)
                                model.metadata["gw_kh_anomaly_applied"] = applied
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                pass

                        # Load initial heads (inline in GW main file)
                        if gw_config.initial_heads is not None:
                            try:
                                gw.set_heads(gw_config.initial_heads)
                                model.metadata["gw_initial_heads_loaded"] = True
                            except ValueError:
                                # Shape mismatch — store as metadata
                                model.metadata["gw_initial_heads_shape"] = str(
                                    gw_config.initial_heads.shape
                                )

                        # Store full GW main config for roundtrip fidelity
                        gw.gw_main_config = gw_config

                    except _COMPONENT_LOAD_EXCEPTIONS:
                        # Fall back to treating file as wells file directly
                        try:
                            gw_reader = GroundwaterReader()
                            wells = gw_reader.read_wells(gw_file)
                            for well in wells.values():
                                gw.add_well(well)
                        except _COMPONENT_LOAD_EXCEPTIONS:
                            pass  # File format not recognized

                    model.groundwater = gw
                except _COMPONENT_LOAD_EXCEPTIONS as e:
                    _record_component_failure(model, "groundwater", gw_file, e, strict=strict)

        # Load streams component using hierarchical reader
        # Always enter when streams_file exists — simulation data (diversions,
        # bypasses, bed params, metadata) must augment the preprocessor-loaded
        # stream object when it already exists.
        if sim_config.streams_file:
            stream_file = _resolve_path(base_dir, str(sim_config.streams_file))
            model.source_files["stream_main"] = stream_file
            if stream_file.exists():
                try:
                    from pyiwfm.components.stream import (
                        AppStream,
                        Bypass,
                        Diversion,
                        StrmNode,
                    )

                    # Reuse preprocessor-loaded stream object if available
                    stream = model.streams if model.streams is not None else AppStream()

                    # Try hierarchical reader first (for component main files)
                    try:
                        stream_main_reader = StreamMainFileReader()
                        stream_config = stream_main_reader.read(stream_file, base_dir=base_dir)

                        model.metadata["stream_version"] = stream_config.version
                        model.metadata["stream_hydrograph_count"] = stream_config.hydrograph_count
                        model.metadata["stream_hydrograph_output_type"] = (
                            stream_config.hydrograph_output_type
                        )

                        # Store output file paths
                        if stream_config.budget_output_file:
                            model.metadata["stream_budget_file"] = str(
                                stream_config.budget_output_file
                            )
                        if stream_config.diversion_budget_file:
                            model.metadata["stream_diversion_budget_file"] = str(
                                stream_config.diversion_budget_file
                            )
                        if stream_config.hydrograph_output_file:
                            model.metadata["stream_hydrograph_file"] = str(
                                stream_config.hydrograph_output_file
                            )

                        # Store hydrograph specs as metadata
                        if stream_config.hydrograph_specs:
                            model.metadata["stream_hydrograph_specs"] = [
                                {"node_id": nid, "name": name}
                                for nid, name in stream_config.hydrograph_specs
                            ]

                        # Store stream sub-file source paths
                        if stream_config.inflow_file:
                            model.source_files["stream_inflow_ts"] = stream_config.inflow_file
                        if stream_config.diversion_spec_file:
                            model.source_files["stream_diversion_spec"] = (
                                stream_config.diversion_spec_file
                            )
                        if stream_config.bypass_spec_file:
                            model.source_files["stream_bypass_spec"] = (
                                stream_config.bypass_spec_file
                            )
                        if stream_config.diversion_file:
                            model.source_files["stream_diversion_ts"] = stream_config.diversion_file

                        # Load diversions from sub-file
                        if (
                            stream_config.diversion_spec_file
                            and stream_config.diversion_spec_file.exists()
                        ):
                            try:
                                div_reader = DiversionSpecReader()
                                div_config = div_reader.read(stream_config.diversion_spec_file)
                                model.metadata["stream_n_diversions"] = div_config.n_diversions
                                model.metadata["stream_n_element_groups"] = (
                                    div_config.n_element_groups
                                )

                                # Convert to Diversion objects
                                dest_map = {
                                    0: "outside",
                                    1: "element",
                                    2: "subregion",
                                    3: "outside",
                                    4: "element_set",
                                    6: "element_set",
                                }
                                for ds in div_config.diversions:
                                    stream.add_diversion(
                                        Diversion(
                                            id=ds.id,
                                            source_node=ds.stream_node,
                                            destination_type=dest_map.get(ds.dest_type, "outside"),
                                            destination_id=ds.dest_id,
                                            name=ds.name,
                                            max_div_column=ds.max_diver_col,
                                            max_div_fraction=ds.frac_max_diver,
                                            recoverable_loss_column=ds.recv_loss_col,
                                            recoverable_loss_fraction=ds.frac_recv_loss,
                                            non_recoverable_loss_column=ds.non_recv_loss_col,
                                            non_recoverable_loss_fraction=ds.frac_non_recv_loss,
                                            spill_column=ds.spill_col,
                                            spill_fraction=ds.frac_spill,
                                            delivery_dest_type=ds.dest_type,
                                            delivery_dest_id=ds.dest_id,
                                            delivery_column=ds.delivery_col,
                                            delivery_fraction=ds.frac_delivery,
                                            irrigation_fraction_column=ds.irrig_frac_col,
                                            adjustment_column=ds.adjustment_col,
                                        )
                                    )

                                # Store element groups and recharge zones
                                stream.diversion_element_groups = div_config.element_groups
                                stream.diversion_recharge_zones = div_config.recharge_zones
                                stream.diversion_spill_zones = div_config.spill_zones
                                stream.diversion_has_spills = div_config.has_spills
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                pass

                        # Load bypasses from sub-file
                        if (
                            stream_config.bypass_spec_file
                            and stream_config.bypass_spec_file.exists()
                        ):
                            try:
                                byp_reader = BypassSpecReader()
                                byp_config = byp_reader.read(stream_config.bypass_spec_file)
                                model.metadata["stream_n_bypasses"] = byp_config.n_bypasses

                                # Convert to Bypass objects
                                for bs in byp_config.bypasses:
                                    rt_flows: list[float] = []
                                    rt_spills: list[float] = []
                                    if bs.inline_rating is not None:
                                        # Undo reader's flow_factor to store raw file values
                                        ff = byp_config.flow_factor
                                        if ff and ff != 0 and ff != 1.0:
                                            rt_flows = (bs.inline_rating.flows / ff).tolist()
                                        else:
                                            rt_flows = bs.inline_rating.flows.tolist()
                                        rt_spills = bs.inline_rating.fractions.tolist()

                                    stream.add_bypass(
                                        Bypass(
                                            id=bs.id,
                                            source_node=bs.export_stream_node,
                                            destination_node=bs.dest_id,
                                            dest_type=bs.dest_type,
                                            name=bs.name,
                                            flow_factor=byp_config.flow_factor,
                                            flow_time_unit=byp_config.flow_time_unit,
                                            spill_factor=byp_config.bypass_factor,
                                            spill_time_unit=byp_config.bypass_time_unit,
                                            diversion_column=bs.rating_table_col,
                                            recoverable_loss_fraction=bs.frac_recoverable,
                                            non_recoverable_loss_fraction=bs.frac_non_recoverable,
                                            rating_table_flows=rt_flows,
                                            rating_table_spills=rt_spills,
                                        )
                                    )

                                # Map seepage zones to bypass objects
                                for sz in byp_config.seepage_zones:
                                    if sz.bypass_id in stream.bypasses:
                                        stream.bypasses[sz.bypass_id].seepage_locations.append(sz)
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                pass

                        # Load inflow info from sub-file
                        if stream_config.inflow_file and stream_config.inflow_file.exists():
                            try:
                                inflow_reader = InflowReader()
                                inflow_config = inflow_reader.read(stream_config.inflow_file)
                                model.metadata["stream_n_inflows"] = inflow_config.n_inflows
                                model.metadata["stream_inflow_nodes"] = inflow_config.inflow_nodes
                            except _COMPONENT_LOAD_EXCEPTIONS:
                                pass

                        # Populate stream bed parameters from main file
                        if stream_config.bed_params:
                            for bp in stream_config.bed_params:
                                if bp.node_id not in stream.nodes:
                                    # Create minimal node when nodes dict is empty
                                    stream.add_node(
                                        StrmNode(
                                            id=bp.node_id,
                                            x=0.0,
                                            y=0.0,
                                        )
                                    )
                                node = stream.nodes[bp.node_id]
                                node.conductivity = bp.conductivity
                                node.bed_thickness = bp.bed_thickness
                                if bp.wetted_perimeter is not None:
                                    node.wetted_perimeter = bp.wetted_perimeter
                                if bp.gw_node and bp.gw_node > 0:
                                    node.gw_node = bp.gw_node
                            stream.conductivity_factor = stream_config.conductivity_factor
                            stream.conductivity_time_unit = stream_config.conductivity_time_unit
                            stream.length_factor = stream_config.length_factor

                        # Interaction type
                        if stream_config.interaction_type is not None:
                            stream.interaction_type = stream_config.interaction_type

                        # Stream evaporation
                        if stream_config.evap_area_file:
                            stream.evap_area_file = str(stream_config.evap_area_file)
                        if stream_config.evap_node_specs:
                            from pyiwfm.components.stream import StrmEvapNodeSpec

                            stream.evap_node_specs = [
                                StrmEvapNodeSpec(
                                    node_id=ns[0],
                                    et_column=ns[1],
                                    area_column=ns[2],
                                )
                                for ns in stream_config.evap_node_specs
                            ]

                        # v5.0 cross-section data
                        if stream_config.cross_section_data:
                            from pyiwfm.components.stream import CrossSectionData

                            for cs in stream_config.cross_section_data:
                                if cs.node_id in stream.nodes:
                                    stream.nodes[cs.node_id].cross_section = CrossSectionData(
                                        bottom_elev=cs.bottom_elev,
                                        B0=cs.B0,
                                        s=cs.s,
                                        n=cs.n,
                                        max_flow_depth=cs.max_flow_depth,
                                    )
                            stream.roughness_factor = stream_config.roughness_factor
                            stream.cross_section_length_factor = (
                                stream_config.cross_section_length_factor
                            )

                        # v5.0 initial conditions
                        if stream_config.initial_conditions:
                            for ic_row in stream_config.initial_conditions:
                                if ic_row.node_id in stream.nodes:
                                    stream.nodes[ic_row.node_id].initial_condition = ic_row.value
                            stream.ic_type = stream_config.ic_type
                            stream.ic_factor = stream_config.ic_factor

                        # Budget node data
                        if stream_config.node_budget_count > 0:
                            stream.budget_node_count = stream_config.node_budget_count
                            stream.budget_node_ids = stream_config.node_budget_ids
                            if stream_config.node_budget_output_file:
                                stream.budget_output_file = str(
                                    stream_config.node_budget_output_file
                                )
                                model.metadata["stream_node_budget_file"] = str(
                                    stream_config.node_budget_output_file
                                )

                        # v5.0 final flow file
                        if stream_config.final_flow_file:
                            stream.final_flow_file = str(stream_config.final_flow_file)

                    except _COMPONENT_LOAD_EXCEPTIONS:
                        # Fall back to treating file as stream nodes file
                        try:
                            stream_reader = StreamReader()
                            nodes = stream_reader.read_stream_nodes(stream_file)
                            for node in nodes.values():
                                stream.add_node(node)
                        except _COMPONENT_LOAD_EXCEPTIONS:
                            pass

                    model.streams = stream

                    # ---- Enrich stream nodes (bottom_elev, rating) and reaches ----
                    if stream.nodes:
                        try:
                            from pyiwfm.components.stream import StrmReach
                            from pyiwfm.io.preprocessor import read_preprocessor_main

                            pp_config = read_preprocessor_main(preprocessor_file)
                            if pp_config.streams_file and pp_config.streams_file.exists():
                                # Store the path so the viewer can use it
                                # as a fallback for reach boundaries
                                model.source_files["streams_spec"] = pp_config.streams_file
                                spec_reader = StreamSpecReader()
                                _nr, _nrt, reach_specs = spec_reader.read(pp_config.streams_file)
                                for rs in reach_specs:
                                    # Enrich existing nodes with reach_id, gw_node,
                                    # bottom elevation, and rating tables
                                    for sn_id in rs.node_ids:
                                        if sn_id in stream.nodes:
                                            sn = stream.nodes[sn_id]
                                            if not getattr(sn, "reach_id", 0):
                                                sn.reach_id = rs.id
                                            gw_nid = rs.node_to_gw_node.get(sn_id)
                                            if gw_nid and gw_nid > 0 and sn.gw_node is None:
                                                sn.gw_node = gw_nid
                                            # Transfer bottom elevation
                                            if (
                                                sn_id in rs.node_bottom_elevations
                                                and sn.bottom_elev == 0.0
                                            ):
                                                sn.bottom_elev = rs.node_bottom_elevations[sn_id]
                                            # Transfer rating table
                                            if sn_id in rs.node_rating_tables and sn.rating is None:
                                                import numpy as np

                                                from pyiwfm.components.stream import StreamRating

                                                stages, flows = rs.node_rating_tables[sn_id]
                                                sn.rating = StreamRating(
                                                    stages=np.array(stages, dtype=np.float64),
                                                    flows=np.array(flows, dtype=np.float64),
                                                )
                                # Only add reaches if not already populated
                                if not stream.reaches:
                                    stream.add_reach(
                                        StrmReach(
                                            id=rs.id,
                                            upstream_node=(rs.node_ids[0] if rs.node_ids else 0),
                                            downstream_node=(rs.node_ids[-1] if rs.node_ids else 0),
                                            nodes=list(rs.node_ids),
                                            name=rs.name,
                                        )
                                    )
                        except _COMPONENT_LOAD_EXCEPTIONS as e:
                            logger.warning(
                                "Could not enrich stream reaches from preprocessor: %s",
                                e,
                            )
                            model.metadata["stream_reach_enrichment_error"] = str(e)

                    # Final safety net: build reaches from node reach_ids
                    _build_reaches_from_node_reach_ids(stream)

                    logger.debug(
                        "Stream loading complete: %d nodes, %d reaches",
                        len(stream.nodes),
                        len(stream.reaches) if stream.reaches else 0,
                    )

                except _COMPONENT_LOAD_EXCEPTIONS as e:
                    _record_component_failure(model, "streams", stream_file, e, strict=strict)

        # Load lakes component using hierarchical reader
        if sim_config.lakes_file and model.lakes is None:
            lake_file = _resolve_path(base_dir, str(sim_config.lakes_file))
            model.source_files["lake_main"] = lake_file
            if lake_file.exists():
                try:
                    from pyiwfm.components.lake import AppLake, LakeElement

                    lakes = AppLake()

                    # Try hierarchical reader first (for component main files)
                    try:
                        lake_main_reader = LakeMainFileReader()
                        lake_config = lake_main_reader.read(lake_file, base_dir=base_dir)

                        model.metadata["lake_version"] = lake_config.version
                        model.metadata["lake_n_lakes"] = len(lake_config.lake_params)
                        if lake_config.max_elev_file:
                            model.source_files["lake_max_elev_ts"] = lake_config.max_elev_file

                        # Store output file paths
                        if lake_config.budget_output_file:
                            model.metadata["lake_budget_file"] = str(lake_config.budget_output_file)

                        # Create Lake objects from param specs
                        from pyiwfm.components.lake import Lake

                        for lp in lake_config.lake_params:
                            lake = Lake(
                                id=lp.lake_id,
                                name=lp.name,
                                bed_conductivity=lp.conductance_coeff,
                                bed_thickness=lp.depth_denom,
                                max_elev_column=lp.max_elev_col,
                                et_column=lp.et_col,
                                precip_column=lp.precip_col,
                            )
                            lakes.add_lake(lake)
                            for elem_id in lake.elements:
                                lakes.add_lake_element(
                                    LakeElement(element_id=elem_id, lake_id=lake.id)
                                )

                        # Store conductance parameters
                        model.metadata["lake_conductance_factor"] = lake_config.conductance_factor
                        model.metadata["lake_depth_factor"] = lake_config.depth_factor

                        # v5.0 outflow ratings
                        if lake_config.outflow_ratings:
                            model.metadata["lake_n_outflow_ratings"] = len(
                                lake_config.outflow_ratings
                            )

                    except _COMPONENT_LOAD_EXCEPTIONS:
                        # Fall back to reading as lake definitions file
                        try:
                            lake_reader = LakeReader()
                            lakes_dict = lake_reader.read_lake_definitions(lake_file)
                            for lake in lakes_dict.values():
                                lakes.add_lake(lake)
                                for elem_id in lake.elements:
                                    lakes.add_lake_element(
                                        LakeElement(element_id=elem_id, lake_id=lake.id)
                                    )
                        except _COMPONENT_LOAD_EXCEPTIONS:
                            pass

                    model.lakes = lakes
                except _COMPONENT_LOAD_EXCEPTIONS as e:
                    _record_component_failure(model, "lakes", lake_file, e, strict=strict)

        # Load rootzone component using hierarchical reader
        if sim_config.rootzone_file:
            rz_file = _resolve_path(base_dir, str(sim_config.rootzone_file))
            model.source_files["rootzone_main"] = rz_file
            if rz_file.exists():
                try:
                    from pyiwfm.components.rootzone import (
                        CropType,
                        RootZone,
                        SoilParameters,
                    )
                    from pyiwfm.io.rootzone import version_ge

                    n_elements = model.mesh.n_elements if model.mesh else 0
                    rootzone = RootZone(n_elements=n_elements, n_layers=1)

                    # Try hierarchical reader first
                    try:
                        rz_main_reader = RootZoneMainFileReader()
                        rz_config = rz_main_reader.read(
                            rz_file,
                            base_dir=base_dir,
                            n_elements=n_elements,
                        )

                        model.metadata["rootzone_version"] = rz_config.version
                        model.metadata["rootzone_gw_uptake"] = rz_config.gw_uptake_enabled

                        # Store rootzone sub-file source paths
                        if rz_config.nonponded_crop_file:
                            model.source_files["rootzone_nonponded"] = rz_config.nonponded_crop_file
                        if rz_config.ponded_crop_file:
                            model.source_files["rootzone_ponded"] = rz_config.ponded_crop_file
                        if rz_config.urban_file:
                            model.source_files["rootzone_urban"] = rz_config.urban_file
                        if rz_config.native_veg_file:
                            model.source_files["rootzone_native"] = rz_config.native_veg_file
                        if rz_config.return_flow_file:
                            model.source_files["rootzone_return_flow_ts"] = (
                                rz_config.return_flow_file
                            )
                        if rz_config.reuse_file:
                            model.source_files["rootzone_reuse_ts"] = rz_config.reuse_file
                        if rz_config.irrigation_period_file:
                            model.source_files["rootzone_irig_period_ts"] = (
                                rz_config.irrigation_period_file
                            )
                        if rz_config.ag_water_demand_file:
                            model.source_files["rootzone_ag_demand_ts"] = (
                                rz_config.ag_water_demand_file
                            )
                        if rz_config.surface_flow_dest_file:
                            model.source_files["rootzone_surface_flow_dest"] = (
                                rz_config.surface_flow_dest_file
                            )

                        # Store sub-file paths as metadata
                        if rz_config.nonponded_crop_file:
                            model.metadata["rootzone_nonponded_file"] = str(
                                rz_config.nonponded_crop_file
                            )
                        if rz_config.ponded_crop_file:
                            model.metadata["rootzone_ponded_file"] = str(rz_config.ponded_crop_file)
                        if rz_config.urban_file:
                            model.metadata["rootzone_urban_file"] = str(rz_config.urban_file)
                        if rz_config.native_veg_file:
                            model.metadata["rootzone_native_veg_file"] = str(
                                rz_config.native_veg_file
                            )

                        # Store output file paths
                        if rz_config.lwu_budget_file:
                            model.metadata["rootzone_lwu_budget_file"] = str(
                                rz_config.lwu_budget_file
                            )
                        if rz_config.rz_budget_file:
                            model.metadata["rootzone_rz_budget_file"] = str(
                                rz_config.rz_budget_file
                            )
                        if rz_config.lwu_zone_budget_file:
                            model.metadata["rootzone_lwu_zbudget_file"] = str(
                                rz_config.lwu_zone_budget_file
                            )
                        if rz_config.rz_zone_budget_file:
                            model.metadata["rootzone_rz_zbudget_file"] = str(
                                rz_config.rz_zone_budget_file
                            )

                        # Store soil parameter conversion factors
                        model.metadata["rootzone_k_factor"] = rz_config.k_factor
                        model.metadata["rootzone_cprise_factor"] = rz_config.k_exdth_factor
                        model.metadata["rootzone_k_time_unit"] = rz_config.k_time_unit

                        # Populate soil parameters from main-file table
                        for row in rz_config.element_soil_params:
                            sp = SoilParameters(
                                porosity=row.total_porosity,
                                field_capacity=row.field_capacity,
                                wilting_point=row.wilting_point,
                                saturated_kv=(row.hydraulic_conductivity * rz_config.k_factor),
                                lambda_param=row.lambda_param,
                                kunsat_method=row.kunsat_method,
                                k_ponded=row.k_ponded,
                                capillary_rise=(row.capillary_rise * rz_config.k_exdth_factor),
                                precip_column=row.precip_column,
                                precip_factor=row.precip_factor,
                                generic_moisture_column=row.generic_moisture_column,
                            )
                            rootzone.set_soil_parameters(row.element_id, sp)

                            # Store surface flow destinations
                            # v4.12+: single signed element index per land-use
                            # (positive = stream node, negative = element,
                            #  0 = no destination). Store as (raw_value, 0)
                            # where the raw value preserves the Fortran sign
                            # convention for downstream interpretation.
                            if version_ge(rz_config.version, (4, 12)):
                                rootzone.surface_flow_dest_ag[row.element_id] = (
                                    row.dest_ag,
                                    abs(row.dest_ag),
                                )
                                rootzone.surface_flow_dest_urban_in[row.element_id] = (
                                    row.dest_urban_in,
                                    abs(row.dest_urban_in),
                                )
                                rootzone.surface_flow_dest_urban_out[row.element_id] = (
                                    row.dest_urban_out,
                                    abs(row.dest_urban_out),
                                )
                                rootzone.surface_flow_dest_nvrv[row.element_id] = (
                                    row.dest_nvrv,
                                    abs(row.dest_nvrv),
                                )
                            else:
                                rootzone.surface_flow_destinations[row.element_id] = (
                                    row.surface_flow_dest_type,
                                    row.surface_flow_dest_id,
                                )

                        n_soil = len(rootzone.soil_params)
                        if n_soil != n_elements:
                            logger.warning(
                                "Root zone soil params: read %d rows but mesh "
                                "has %d elements (missing %d)",
                                n_soil,
                                n_elements,
                                n_elements - n_soil,
                            )

                        # Read sub-files — dispatch on version.
                        # Each reader in its own try/except so one failure
                        # doesn't cascade to the others.
                        _use_v5 = version_ge(rz_config.version, (5, 0))

                        if _use_v5:
                            from pyiwfm.io.rootzone_native import (
                                NativeRiparianReader,
                            )
                            from pyiwfm.io.rootzone_nonponded import (
                                NonPondedCropReader,
                            )
                            from pyiwfm.io.rootzone_ponded import (
                                PondedCropReader,
                            )

                            if (
                                rz_config.nonponded_crop_file
                                and rz_config.nonponded_crop_file.exists()
                            ):
                                try:
                                    rootzone.nonponded_config = NonPondedCropReader().read(
                                        rz_config.nonponded_crop_file,
                                        base_dir,
                                    )
                                except _COMPONENT_LOAD_EXCEPTIONS as exc:
                                    logger.warning(
                                        "Failed to read nonponded sub-file (v5): %s",
                                        exc,
                                    )

                            if rz_config.ponded_crop_file and rz_config.ponded_crop_file.exists():
                                try:
                                    rootzone.ponded_config = PondedCropReader().read(
                                        rz_config.ponded_crop_file,
                                        base_dir,
                                    )
                                except _COMPONENT_LOAD_EXCEPTIONS as exc:
                                    logger.warning(
                                        "Failed to read ponded sub-file (v5): %s",
                                        exc,
                                    )

                            if rz_config.urban_file and rz_config.urban_file.exists():
                                try:
                                    from pyiwfm.io.rootzone_urban import (
                                        UrbanLandUseReader,
                                    )

                                    rootzone.urban_config = UrbanLandUseReader().read(
                                        rz_config.urban_file,
                                        base_dir,
                                    )
                                except _COMPONENT_LOAD_EXCEPTIONS as exc:
                                    logger.warning(
                                        "Failed to read urban sub-file (v5): %s",
                                        exc,
                                    )

                            if rz_config.native_veg_file and rz_config.native_veg_file.exists():
                                try:
                                    rootzone.native_riparian_config = NativeRiparianReader().read(
                                        rz_config.native_veg_file,
                                        base_dir,
                                    )
                                except _COMPONENT_LOAD_EXCEPTIONS as exc:
                                    logger.warning(
                                        "Failed to read native/riparian sub-file (v5): %s",
                                        exc,
                                    )

                        else:
                            from pyiwfm.io.rootzone_v4x import (
                                NativeRiparianReaderV4x,
                                NonPondedCropReaderV4x,
                                PondedCropReaderV4x,
                                UrbanReaderV4x,
                            )

                            if (
                                rz_config.nonponded_crop_file
                                and rz_config.nonponded_crop_file.exists()
                            ):
                                try:
                                    reader = NonPondedCropReaderV4x(n_elements=n_elements)
                                    rootzone.nonponded_config = reader.read(
                                        rz_config.nonponded_crop_file,
                                        base_dir,
                                    )
                                except _COMPONENT_LOAD_EXCEPTIONS as exc:
                                    logger.warning(
                                        "Failed to read nonponded sub-file: %s",
                                        exc,
                                    )

                            if rz_config.ponded_crop_file and rz_config.ponded_crop_file.exists():
                                try:
                                    ponded_reader = PondedCropReaderV4x(n_elements=n_elements)
                                    rootzone.ponded_config = ponded_reader.read(
                                        rz_config.ponded_crop_file, base_dir
                                    )
                                except _COMPONENT_LOAD_EXCEPTIONS as exc:
                                    logger.warning(
                                        "Failed to read ponded sub-file: %s",
                                        exc,
                                    )

                            if rz_config.urban_file and rz_config.urban_file.exists():
                                try:
                                    urban_reader = UrbanReaderV4x(n_elements=n_elements)
                                    rootzone.urban_config = urban_reader.read(
                                        rz_config.urban_file, base_dir
                                    )
                                except _COMPONENT_LOAD_EXCEPTIONS as exc:
                                    logger.warning(
                                        "Failed to read urban sub-file: %s",
                                        exc,
                                    )

                            if rz_config.native_veg_file and rz_config.native_veg_file.exists():
                                try:
                                    nr_reader = NativeRiparianReaderV4x(n_elements=n_elements)
                                    rootzone.native_riparian_config = nr_reader.read(
                                        rz_config.native_veg_file,
                                        base_dir,
                                    )
                                except _COMPONENT_LOAD_EXCEPTIONS as exc:
                                    logger.warning(
                                        "Failed to read native/riparian sub-file: %s",
                                        exc,
                                    )

                        # Extract crop types from v4.x sub-file configs
                        crop_id_offset = 0
                        if rootzone.nonponded_config is not None:
                            np_cfg = rootzone.nonponded_config
                            for i, code in enumerate(np_cfg.crop_codes):
                                crop_id = i + 1
                                rd = (
                                    np_cfg.root_depth_data[i].max_root_depth
                                    if i < len(np_cfg.root_depth_data)
                                    else 0.0
                                )
                                rootzone.add_crop_type(
                                    CropType(id=crop_id, name=code, root_depth=rd)
                                )
                            crop_id_offset = len(np_cfg.crop_codes)

                        if rootzone.ponded_config is not None:
                            _PONDED_NAMES = [
                                "RICE_FL",
                                "RICE_NFL",
                                "RICE_NDC",
                                "REFUGE_SL",
                                "REFUGE_PR",
                            ]
                            p_cfg = rootzone.ponded_config
                            for i, depth in enumerate(p_cfg.root_depths):
                                crop_id = crop_id_offset + i + 1
                                name = (
                                    _PONDED_NAMES[i]
                                    if i < len(_PONDED_NAMES)
                                    else f"PONDED_{i + 1}"
                                )
                                rootzone.add_crop_type(
                                    CropType(id=crop_id, name=name, root_depth=depth)
                                )

                    except _COMPONENT_LOAD_EXCEPTIONS:
                        # Fall back to treating file as crop types file
                        try:
                            rz_reader = RootZoneReader()
                            crops = rz_reader.read_crop_types(rz_file)
                            for crop in crops.values():
                                rootzone.add_crop_type(crop)
                        except _COMPONENT_LOAD_EXCEPTIONS:
                            pass

                    # Wire area data file paths for lazy loading.
                    # This runs OUTSIDE the big try/except above so that
                    # area files are wired even if crop type extraction
                    # or other secondary parsing fails.
                    # v5+ configs use "elemental_area_file" while v4x
                    # use "area_data_file"; try both with fallback.
                    try:
                        for cfg_attr, rz_attr in [
                            ("nonponded_config", "nonponded_area_file"),
                            ("ponded_config", "ponded_area_file"),
                            ("urban_config", "urban_area_file"),
                            ("native_riparian_config", "native_area_file"),
                        ]:
                            cfg = getattr(rootzone, cfg_attr, None)
                            if cfg is None:
                                continue
                            af = getattr(cfg, "area_data_file", None) or getattr(
                                cfg, "elemental_area_file", None
                            )
                            if af is not None:
                                resolved = af if af.is_absolute() else base_dir / af
                                setattr(rootzone, rz_attr, resolved)
                                logger.debug(
                                    "Wired %s -> %s (exists=%s)",
                                    rz_attr,
                                    resolved,
                                    resolved.exists(),
                                )
                    except _COMPONENT_LOAD_EXCEPTIONS as exc:
                        logger.warning("Failed to wire area data file paths: %s", exc)

                    model.rootzone = rootzone
                except _COMPONENT_LOAD_EXCEPTIONS as e:
                    _record_component_failure(model, "rootzone", rz_file, e, strict=strict)

        # Load small watershed component (optional)
        if sim_config.small_watershed_file:
            sw_file = _resolve_path(base_dir, str(sim_config.small_watershed_file))
            model.source_files["swshed_main"] = sw_file
            if sw_file.exists():
                try:
                    from pyiwfm.components.small_watershed import AppSmallWatershed
                    from pyiwfm.io.small_watershed import SmallWatershedMainReader

                    sw_reader = SmallWatershedMainReader()
                    sw_config = sw_reader.read(sw_file, base_dir=base_dir)
                    model.metadata["small_watershed_version"] = sw_config.version
                    model.metadata["small_watershed_count"] = sw_config.n_watersheds
                    if sw_config.budget_output_file:
                        model.metadata["small_watershed_budget_file"] = str(
                            sw_config.budget_output_file
                        )

                    # Build component from config
                    if sw_config.n_watersheds > 0:
                        model.small_watersheds = AppSmallWatershed.from_config(sw_config)
                except _COMPONENT_LOAD_EXCEPTIONS as e:
                    # Metadata key uses singular "small_watershed" for
                    # backward compatibility with callers that read this key.
                    _record_component_failure(model, "small_watershed", sw_file, e, strict=strict)

        # Load unsaturated zone component (optional)
        if sim_config.unsaturated_zone_file and sim_config.unsaturated_zone_file.is_file():
            uz_file = _resolve_path(base_dir, str(sim_config.unsaturated_zone_file))

            model.source_files["unsatzone_main"] = uz_file
            if uz_file.exists():
                try:
                    from pyiwfm.components.unsaturated_zone import AppUnsatZone
                    from pyiwfm.io.unsaturated_zone import UnsatZoneMainReader

                    uz_reader = UnsatZoneMainReader()
                    uz_config = uz_reader.read(uz_file, base_dir=base_dir)
                    model.metadata["unsat_zone_version"] = uz_config.version
                    model.metadata["unsat_zone_n_layers"] = uz_config.n_layers
                    if uz_config.budget_file:
                        model.metadata["unsat_zone_budget_file"] = str(uz_config.budget_file)
                    if uz_config.zbudget_file:
                        model.metadata["unsat_zone_zbudget_file"] = str(uz_config.zbudget_file)

                    # Build component from config
                    if uz_config.n_layers > 0:
                        model.unsaturated_zone = AppUnsatZone.from_config(uz_config)
                except _COMPONENT_LOAD_EXCEPTIONS as e:
                    # The metadata key is "unsat_zone_load_error" (not
                    # "unsaturated_zone_load_error") for backward
                    # compatibility with callers that read this key.
                    _record_component_failure(model, "unsat_zone", uz_file, e, strict=strict)

        _resolve_stream_node_coordinates(model)
        return model

    @classmethod
    def from_hdf5(cls, hdf5_file: Path | str) -> IWFMModel:
        """
        Load a model from HDF5 output file.

        This loads a complete model that was previously saved to HDF5 format
        using the to_hdf5() method or write_model_hdf5() function.

        Args:
            hdf5_file: Path to the HDF5 file

        Returns:
            Loaded IWFMModel instance

        Example:
            >>> model = IWFMModel.from_hdf5("model.h5")
        """
        from pyiwfm.io.hdf5 import read_model_hdf5

        return read_model_hdf5(hdf5_file)

    # ========================================================================
    # Instance Methods for Saving Models
    # ========================================================================

    def to_preprocessor(self, output_dir: Path | str) -> dict[str, Path]:
        """
        Write model to PreProcessor input files.

        Creates all preprocessor input files (nodes, elements, stratigraphy)
        in the specified output directory.

        Args:
            output_dir: Directory to write output files

        Returns:
            Dictionary mapping file type to output path
        """
        from pyiwfm.io.preprocessor import save_model_to_preprocessor

        config = save_model_to_preprocessor(self, output_dir, self.name)

        files: dict[str, Path] = {}
        if config.nodes_file:
            files["nodes"] = config.nodes_file
        if config.elements_file:
            files["elements"] = config.elements_file
        if config.stratigraphy_file:
            files["stratigraphy"] = config.stratigraphy_file
        if config.subregions_file:
            files["subregions"] = config.subregions_file

        return files

    def to_simulation(
        self,
        output_dir: Path | str,
        file_paths: dict[str, str] | None = None,
        ts_format: str = "text",
    ) -> dict[str, Path]:
        """
        Write complete model to simulation input files.

        Creates all input files required for an IWFM simulation, including
        preprocessor files, component files, and the simulation control file.

        Args:
            output_dir: Directory to write output files
            file_paths: Optional dict of {file_key: relative_path} overrides
                for custom directory layouts. If None, uses default nested layout.
            ts_format: Time series format - "text" or "dss"

        Returns:
            Dictionary mapping file type to output path
        """
        from pyiwfm.io.preprocessor import save_complete_model

        return save_complete_model(
            self,
            output_dir,
            timeseries_format=ts_format,
            file_paths=file_paths,
        )

    def to_hdf5(self, output_file: Path | str) -> None:
        """
        Write model to HDF5 file.

        Saves the complete model (mesh, stratigraphy, and all components)
        to a single HDF5 file for efficient storage and later loading.

        Args:
            output_file: Path to the output HDF5 file

        Example:
            >>> model.to_hdf5("model.h5")
        """
        from pyiwfm.io.hdf5 import write_model_hdf5

        write_model_hdf5(output_file, self)

    def to_binary(self, output_file: Path | str) -> None:
        """
        Write model mesh and stratigraphy to binary files.

        Args:
            output_file: Base path for output files (will create .bin and .strat.bin)
        """
        from pyiwfm.io.binary import write_binary_mesh, write_binary_stratigraphy

        output_file = Path(output_file)

        if self.mesh:
            write_binary_mesh(output_file, self.mesh)

        if self.stratigraphy:
            strat_file = output_file.with_suffix(".strat.bin")
            write_binary_stratigraphy(strat_file, self.stratigraphy)

    # ========================================================================
    # Validation Methods
    # ========================================================================

    def validate(self) -> list[str]:
        """
        Validate model structure and data.

        Returns:
            List of validation errors (empty if valid)

        Raises:
            ValidationError: If critical validation fails
        """
        errors: list[str] = []

        # Validate mesh
        if self.mesh is None:
            errors.append("Model has no mesh")
        else:
            try:
                self.mesh.validate()
            except Exception as e:
                errors.append(f"Mesh validation failed: {e}")

        # Validate stratigraphy
        if self.stratigraphy is None:
            errors.append("Model has no stratigraphy")
        else:
            try:
                warnings = self.stratigraphy.validate()
                errors.extend(warnings)
            except Exception as e:
                errors.append(f"Stratigraphy validation failed: {e}")

        # Check mesh/stratigraphy consistency
        if self.mesh is not None and self.stratigraphy is not None:
            if self.mesh.n_nodes != self.stratigraphy.n_nodes:
                errors.append(
                    f"Node count mismatch: mesh has {self.mesh.n_nodes}, "
                    f"stratigraphy has {self.stratigraphy.n_nodes}"
                )

        if errors:
            raise ValidationError(
                f"Model validation failed with {len(errors)} error(s)", errors=errors
            )

        return []

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def n_nodes(self) -> int:
        """Return number of nodes in the mesh."""
        if self.mesh is None:
            return 0
        return self.mesh.n_nodes

    @property
    def n_elements(self) -> int:
        """Return number of elements in the mesh."""
        if self.mesh is None:
            return 0
        return self.mesh.n_elements

    @property
    def n_layers(self) -> int:
        """Return number of layers in the stratigraphy."""
        if self.stratigraphy is None:
            return 0
        return self.stratigraphy.n_layers

    @property
    def grid(self) -> AppGrid | None:
        """Alias for mesh property for compatibility."""
        return self.mesh

    @grid.setter
    def grid(self, value: AppGrid | None) -> None:
        """Set the mesh/grid."""
        self.mesh = value

    # ========================================================================
    # Component Properties
    # ========================================================================

    @property
    def n_wells(self) -> int:
        """Return number of wells in the groundwater component."""
        if self.groundwater is None:
            return 0
        return self.groundwater.n_wells

    @property
    def n_stream_nodes(self) -> int:
        """Return number of stream nodes."""
        if self.streams is None:
            return 0
        return self.streams.n_nodes

    @property
    def n_stream_reaches(self) -> int:
        """Return number of stream reaches."""
        if self.streams is None:
            return 0
        return self.streams.n_reaches

    @property
    def n_diversions(self) -> int:
        """Return number of diversions."""
        if self.streams is None:
            return 0
        return self.streams.n_diversions

    @property
    def n_lakes(self) -> int:
        """Return number of lakes."""
        if self.lakes is None:
            return 0
        return self.lakes.n_lakes

    @property
    def n_crop_types(self) -> int:
        """Return number of crop types in the root zone."""
        if self.rootzone is None:
            return 0
        return self.rootzone.n_crop_types

    @property
    def has_groundwater(self) -> bool:
        """Return True if groundwater component is loaded."""
        return self.groundwater is not None

    @property
    def has_streams(self) -> bool:
        """Return True if stream component is loaded."""
        return self.streams is not None

    @property
    def has_lakes(self) -> bool:
        """Return True if lake component is loaded."""
        return self.lakes is not None

    @property
    def has_rootzone(self) -> bool:
        """Return True if root zone component is loaded."""
        return self.rootzone is not None

    @property
    def has_small_watersheds(self) -> bool:
        """Return True if small watershed component is loaded."""
        return self.small_watersheds is not None

    @property
    def has_unsaturated_zone(self) -> bool:
        """Return True if unsaturated zone component is loaded."""
        return self.unsaturated_zone is not None

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def summary(self) -> str:
        """
        Return a summary string of the model.

        Returns:
            Multi-line summary of model components
        """
        lines = [
            f"IWFM Model: {self.name}",
            "=" * (len(self.name) + 13),
            "",
            "Mesh & Stratigraphy:",
            f"  Nodes: {self.n_nodes}",
            f"  Elements: {self.n_elements}",
            f"  Layers: {self.n_layers}",
        ]

        if self.mesh is not None:
            lines.append(f"  Subregions: {self.mesh.n_subregions}")

        # Groundwater component
        lines.append("")
        lines.append("Groundwater Component:")
        if self.groundwater is not None:
            lines.append(f"  Wells: {self.groundwater.n_wells}")
            lines.append(f"  Hydrograph Locations: {self.groundwater.n_hydrograph_locations}")
            lines.append(f"  Boundary Conditions: {self.groundwater.n_boundary_conditions}")
            lines.append(f"  Tile Drains: {self.groundwater.n_tile_drains}")
            if self.groundwater.aquifer_params is not None:
                lines.append("  Aquifer Parameters: Loaded")
            else:
                lines.append("  Aquifer Parameters: Not loaded")
        else:
            lines.append("  Not loaded")

        # Stream component
        lines.append("")
        lines.append("Stream Component:")
        if self.streams is not None:
            lines.append(f"  Stream Nodes: {self.streams.n_nodes}")
            lines.append(f"  Reaches: {self.streams.n_reaches}")
            lines.append(f"  Diversions: {self.streams.n_diversions}")
            lines.append(f"  Bypasses: {self.streams.n_bypasses}")
        else:
            lines.append("  Not loaded")

        # Lake component
        lines.append("")
        lines.append("Lake Component:")
        if self.lakes is not None:
            lines.append(f"  Lakes: {self.lakes.n_lakes}")
            lines.append(f"  Lake Elements: {self.lakes.n_lake_elements}")
        else:
            lines.append("  Not loaded")

        # Root zone component
        lines.append("")
        lines.append("Root Zone Component:")
        if self.rootzone is not None:
            lines.append(f"  Crop Types: {self.rootzone.n_crop_types}")
            lines.append(f"  Land Use Assignments: {len(self.rootzone.element_landuse)}")
            lines.append(f"  Soil Parameter Sets: {len(self.rootzone.soil_params)}")
        else:
            lines.append("  Not loaded")

        # Small watershed component
        lines.append("")
        lines.append("Small Watershed Component:")
        if self.small_watersheds is not None:
            lines.append(f"  Watersheds: {self.small_watersheds.n_watersheds}")
        else:
            lines.append("  Not loaded")

        # Unsaturated zone component
        lines.append("")
        lines.append("Unsaturated Zone Component:")
        if self.unsaturated_zone is not None:
            lines.append(f"  Layers: {self.unsaturated_zone.n_layers}")
            lines.append(f"  Elements: {self.unsaturated_zone.n_elements}")
        else:
            lines.append("  Not loaded")

        # Metadata
        lines.append("")
        source = self.metadata.get("source", "unknown")
        lines.append(f"Source: {source}")

        return "\n".join(lines)

    # ========================================================================
    # Model mutation helpers
    # ========================================================================
    #
    # These convenience methods provide an ergonomic, validated path for
    # editing a loaded model in place. The underlying component attributes
    # (``model.groundwater.aquifer_params.kh``, etc.) are still accessible
    # for advanced use, but the helpers below are the documented, supported
    # API for callers building scenarios, calibration runs, or diff tools.
    #
    # Each helper:
    #   - validates inputs (raises ``ValueError`` with helpful messages)
    #   - records the modified component in ``self._dirty`` so save paths
    #     can warn if a dirty component fails to write
    #   - is non-breaking: existing direct-attribute mutation still works
    #
    # See ``docs/user_guide/mutating_models.rst`` for end-to-end examples.

    def mark_dirty(self, component_name: str) -> None:
        """Mark ``component_name`` as modified since the last load/save.

        Call this from custom mutation paths so :meth:`to_simulation` and
        related save methods can surface incomplete writes.
        """
        self._dirty.add(component_name)

    def set_aquifer_parameter(
        self,
        param: str,
        layer: int,
        values: NDArray[np.float64] | list[float],
    ) -> None:
        """Replace an aquifer parameter array for a single layer.

        Parameters
        ----------
        param
            One of ``"kh"``, ``"kv"``, ``"ss"``, ``"sy"``, ``"aquitard_kv"``
            (matches the :class:`AquiferParameters._PARAM_ATTRS` dispatcher).
        layer
            1-based aquifer layer (IWFM convention).
        values
            Array-like of length ``n_nodes`` with the new values for this
            layer.

        Raises
        ------
        ValueError
            If groundwater isn't loaded, the layer is out of range, the
            named parameter array isn't allocated, or the values length
            doesn't match ``n_nodes``.
        KeyError
            If ``param`` isn't a recognized parameter name.

        Examples
        --------
        >>> import numpy as np
        >>> new_kh = np.full(model.groundwater.n_nodes, 1e-4)
        >>> model.set_aquifer_parameter("kh", layer=1, values=new_kh)
        """
        from pyiwfm.core.ids import to_index

        if self.groundwater is None:
            raise ValueError("groundwater component is not loaded")
        params = self.groundwater.aquifer_params
        if params is None:
            raise ValueError("aquifer_params is not set on groundwater component")

        layer_idx = to_index(layer, params.n_layers, kind="layer")

        # Trigger KeyError for unknown param names, ValueError if unset.
        arr = params.get_array(param)

        values_arr = np.asarray(values, dtype=np.float64)
        if values_arr.shape != (params.n_nodes,):
            raise ValueError(
                f"values for {param!r} layer {layer} must have shape "
                f"({params.n_nodes},); got {values_arr.shape}"
            )
        arr[:, layer_idx] = values_arr
        self.mark_dirty("groundwater")

    def set_aquifer_parameter_at(
        self,
        param: str,
        node_id: int,
        layer: int,
        value: float,
    ) -> None:
        """Set a single (node, layer) cell of an aquifer parameter.

        Parameters
        ----------
        param
            See :meth:`set_aquifer_parameter`.
        node_id
            1-based node ID (IWFM convention).
        layer
            1-based aquifer layer.
        value
            New scalar value.

        Raises
        ------
        ValueError
            If groundwater isn't loaded, IDs are out of range, or the
            parameter array isn't allocated.
        """
        from pyiwfm.core.ids import to_index

        if self.groundwater is None:
            raise ValueError("groundwater component is not loaded")
        params = self.groundwater.aquifer_params
        if params is None:
            raise ValueError("aquifer_params is not set on groundwater component")

        node_idx = to_index(node_id, params.n_nodes, kind="node")
        layer_idx = to_index(layer, params.n_layers, kind="layer")
        params.get_array(param)[node_idx, layer_idx] = float(value)
        self.mark_dirty("groundwater")

    def set_stratigraphy_from_thicknesses(
        self,
        gs_elev: NDArray[np.float64] | list[float],
        aquitard_thicknesses: NDArray[np.float64] | list[list[float]],
        aquifer_thicknesses: NDArray[np.float64] | list[list[float]],
        active_node: NDArray[np.bool_] | None = None,
    ) -> None:
        """Replace ``self.stratigraphy`` with one built from thickness arrays.

        Wraps :meth:`Stratigraphy.from_thicknesses` and validates the result
        is consistent with the current mesh (if loaded).

        Parameters
        ----------
        gs_elev
            Ground surface elevations, shape ``(n_nodes,)``.
        aquitard_thicknesses
            Aquitard thicknesses, shape ``(n_nodes, n_layers)``. IWFM
            convention: aquitard ``k`` sits above aquifer layer ``k``.
        aquifer_thicknesses
            Aquifer thicknesses, shape ``(n_nodes, n_layers)``.
        active_node
            Optional active-node flags, shape ``(n_nodes, n_layers)``.

        Raises
        ------
        StratigraphyError
            If thickness shapes are inconsistent or any thickness is
            negative (validation comes from
            :meth:`Stratigraphy.from_thicknesses`).
        ValueError
            If a mesh is loaded and ``gs_elev`` length doesn't match
            ``mesh.n_nodes``.
        """
        from pyiwfm.core.stratigraphy import Stratigraphy

        gs_arr = np.asarray(gs_elev, dtype=np.float64)
        if self.mesh is not None and gs_arr.shape != (self.mesh.n_nodes,):
            raise ValueError(
                f"gs_elev shape {gs_arr.shape} does not match mesh ({self.mesh.n_nodes},)"
            )

        self.stratigraphy = Stratigraphy.from_thicknesses(
            gs_arr,
            np.asarray(aquitard_thicknesses, dtype=np.float64),
            np.asarray(aquifer_thicknesses, dtype=np.float64),
            active_node,
        )
        self.mark_dirty("stratigraphy")

    def add_observation_well(
        self,
        node_id: int,
        layer: int,
        x: float,
        y: float,
        name: str = "",
    ) -> None:
        """Append a groundwater hydrograph observation point.

        Mirrors the IWFM convention where each hydrograph location is
        identified by ``(node_id, layer)`` plus optional ``(x, y)`` for
        rendering and a free-form ``name``.

        Parameters
        ----------
        node_id
            1-based mesh node ID.
        layer
            1-based aquifer layer.
        x, y
            Coordinates (model CRS).
        name
            Optional descriptor (e.g. well name).

        Raises
        ------
        ValueError
            If groundwater isn't loaded, or IDs are out of range against
            the GW component's ``n_nodes`` / ``n_layers``.
        """
        from pyiwfm.components.groundwater import HydrographLocation
        from pyiwfm.core.ids import to_index

        if self.groundwater is None:
            raise ValueError("groundwater component is not loaded")

        # Validate IDs (raises ValueError if out of range)
        to_index(node_id, self.groundwater.n_nodes, kind="node")
        to_index(layer, self.groundwater.n_layers, kind="layer")

        self.groundwater.add_hydrograph_location(
            HydrographLocation(
                node_id=node_id,
                layer=layer,
                x=float(x),
                y=float(y),
                name=name,
            )
        )
        self.mark_dirty("groundwater")

    def remove_observation_well(self, name: str) -> int:
        """Remove all groundwater hydrograph locations whose ``name`` matches.

        Parameters
        ----------
        name
            Exact-match name (case-sensitive). To remove unnamed locations,
            pass an empty string.

        Returns
        -------
        int
            Number of locations removed.

        Raises
        ------
        ValueError
            If groundwater isn't loaded.
        """
        if self.groundwater is None:
            raise ValueError("groundwater component is not loaded")
        before = len(self.groundwater.hydrograph_locations)
        self.groundwater.hydrograph_locations = [
            loc for loc in self.groundwater.hydrograph_locations if loc.name != name
        ]
        removed = before - len(self.groundwater.hydrograph_locations)
        if removed:
            self.mark_dirty("groundwater")
        return removed

    def validate_components(self) -> list[str]:
        """
        Validate all model components.

        Returns:
            List of validation warnings/errors from components
        """
        warnings: list[str] = []

        if self.groundwater is not None:
            try:
                self.groundwater.validate()
            except Exception as e:
                warnings.append(f"Groundwater validation: {e}")

        if self.streams is not None:
            try:
                self.streams.validate()
            except Exception as e:
                warnings.append(f"Stream validation: {e}")

        if self.lakes is not None:
            try:
                self.lakes.validate()
            except Exception as e:
                warnings.append(f"Lake validation: {e}")

        if self.rootzone is not None:
            try:
                self.rootzone.validate()
            except Exception as e:
                warnings.append(f"Root zone validation: {e}")

        if self.small_watersheds is not None:
            try:
                self.small_watersheds.validate()
            except Exception as e:
                warnings.append(f"Small watershed validation: {e}")

        if self.unsaturated_zone is not None:
            try:
                self.unsaturated_zone.validate()
            except Exception as e:
                warnings.append(f"Unsaturated zone validation: {e}")

        return warnings

    def __repr__(self) -> str:
        return (
            f"IWFMModel(name='{self.name}', n_nodes={self.n_nodes}, "
            f"n_elements={self.n_elements}, n_layers={self.n_layers})"
        )
