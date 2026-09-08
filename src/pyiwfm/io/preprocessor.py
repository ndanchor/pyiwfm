"""
PreProcessor file I/O handlers for IWFM models.

This module provides functions for reading and writing IWFM PreProcessor
input files, which define the model structure and input file paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyiwfm.core.exceptions import FileFormatError
from pyiwfm.core.mesh import Subregion
from pyiwfm.core.model import IWFMModel
from pyiwfm.io.ascii import (
    write_elements,
    write_nodes,
    write_stratigraphy,
)
from pyiwfm.io.iwfm_reader import (
    is_comment_line as _is_comment_line,
)
from pyiwfm.io.iwfm_reader import (
    resolve_path as _resolve_path,
)
from pyiwfm.io.iwfm_reader import (
    strip_inline_comment as _strip_comment,
)


@dataclass
class PreProcessorConfig:
    """
    Configuration from an IWFM PreProcessor main input file.

    Contains paths to all component input files and model settings.
    """

    base_dir: Path
    model_name: str = ""

    # Core file paths
    nodes_file: Path | None = None
    elements_file: Path | None = None
    stratigraphy_file: Path | None = None
    subregions_file: Path | None = None

    # Optional component file paths
    streams_file: Path | None = None
    lakes_file: Path | None = None
    groundwater_file: Path | None = None
    rootzone_file: Path | None = None
    pumping_file: Path | None = None

    # Output settings
    output_dir: Path | None = None
    budget_output_file: Path | None = None
    heads_output_file: Path | None = None

    # Model settings
    n_layers: int = 1
    length_unit: str = "FT"
    """UNITLTOU -- the output/reporting length unit (not necessarily the
    model's internal simulation unit; see ``simulation_length_unit``)."""
    length_factor: float = 1.0
    """FACTLTOU -- factor converting the simulation length unit to
    ``length_unit`` for reporting."""
    simulation_length_unit: str | None = None
    """The model's native (simulation) coordinate length unit, ``"FEET"``
    or ``"METERS"``, derived from ``length_factor``/``length_unit`` via
    :func:`pyiwfm.core.units.resolve_model_length_unit`. ``None`` if it
    can't be determined."""
    area_unit: str = "ACRES"
    volume_unit: str = "AF"

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


def read_preprocessor_main(filepath: Path | str) -> PreProcessorConfig:
    """
    Read an IWFM PreProcessor main input file.

    The main input file contains paths to all component input files
    and basic model configuration.

    Expected format (simplified):
        Model Name                / Description
        nodes.dat                 / NODES_FILE
        elements.dat              / ELEMENTS_FILE
        stratigraphy.dat          / STRAT_FILE
        ...

    Args:
        filepath: Path to the main PreProcessor input file

    Returns:
        PreProcessorConfig with parsed values
    """
    filepath = Path(filepath)
    base_dir = filepath.parent

    config = PreProcessorConfig(base_dir=base_dir)

    with open(filepath) as f:
        line_num = 0
        data_lines: list[tuple[str, str]] = []  # (value, description)

        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value, desc = _strip_comment(line)
            if value:
                data_lines.append((value, desc.upper()))

    # Parse the data lines based on position and description hints
    # IWFM PreProcessor files have a fairly standard structure
    idx = 0

    # First non-comment line is typically the model name or a version
    if idx < len(data_lines):
        value, desc = data_lines[idx]
        if "NAME" in desc or "TITLE" in desc or idx == 0:
            config.model_name = value
            idx += 1

    # Look for file paths by description
    for value, desc in data_lines[idx:]:
        value_path = _resolve_path(base_dir, value) if not value.isdigit() else None

        if "NODE" in desc and "FILE" in desc:
            config.nodes_file = value_path
        elif "ELEM" in desc and "FILE" in desc:
            config.elements_file = value_path
        elif "STRAT" in desc and "FILE" in desc:
            config.stratigraphy_file = value_path
        elif "SUBREG" in desc and "FILE" in desc:
            config.subregions_file = value_path
        elif "STREAM" in desc and "FILE" in desc:
            config.streams_file = value_path
        elif "LAKE" in desc and "FILE" in desc:
            config.lakes_file = value_path
        elif "GROUND" in desc and "FILE" in desc:
            config.groundwater_file = value_path
        elif "ROOT" in desc and "FILE" in desc:
            config.rootzone_file = value_path
        elif "PUMP" in desc and "FILE" in desc:
            config.pumping_file = value_path
        elif "LAYER" in desc:
            try:
                config.n_layers = int(value)
            except ValueError:
                pass
        elif "FACTLTOU" in desc:
            try:
                config.length_factor = float(value)
            except ValueError:
                pass
        elif "UNITLTOU" in desc:
            config.length_unit = value.upper()
        elif "LENGTH" in desc and "UNIT" in desc:
            # Fallback for the simplified round-trip format written by
            # write_preprocessor_main() below (not real IWFM's FACTLTOU/
            # UNITLTOU tokens, which are matched explicitly above).
            config.length_unit = value.upper()
        elif "AREA" in desc and "UNIT" in desc:
            config.area_unit = value.upper()
        elif "VOLUME" in desc and "UNIT" in desc:
            config.volume_unit = value.upper()
        elif "OUTPUT" in desc and "DIR" in desc:
            config.output_dir = value_path

    from pyiwfm.core.units import resolve_model_length_unit

    config.simulation_length_unit = resolve_model_length_unit(
        config.length_unit, config.length_factor
    )

    return config


def read_subregions_file(filepath: Path | str) -> dict[int, Subregion]:
    """
    Read subregion definitions from a subregions file.

    Expected format:
        NSUBREGION                / Number of subregions
        ID  NAME                  (one line per subregion)

    Args:
        filepath: Path to the subregions file

    Returns:
        Dictionary mapping subregion ID to Subregion object
    """
    filepath = Path(filepath)
    subregions: dict[int, Subregion] = {}

    with open(filepath) as f:
        line_num = 0
        n_subregions = None

        # Find NSUBREGION
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            value, _ = _strip_comment(line)
            try:
                n_subregions = int(value)
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid NSUBREGION value: '{value}'", line_number=line_num
                ) from e
            break

        if n_subregions is None:
            raise FileFormatError("Could not find NSUBREGION in file")

        # Read subregion data
        for line in f:
            line_num += 1
            if _is_comment_line(line):
                continue

            parts = line.split(None, 1)  # Split on first whitespace
            if len(parts) < 1:
                continue

            try:
                sr_id = int(parts[0])
                sr_name = parts[1].strip() if len(parts) > 1 else ""
            except ValueError as e:
                raise FileFormatError(
                    f"Invalid subregion data: '{line.strip()}'", line_number=line_num
                ) from e

            subregions[sr_id] = Subregion(id=sr_id, name=sr_name)

    return subregions


def write_preprocessor_main(
    filepath: Path | str,
    config: PreProcessorConfig,
    header: str | None = None,
) -> None:
    """
    Write an IWFM PreProcessor main input file.

    Args:
        filepath: Path to the output file
        config: PreProcessorConfig with file paths and settings
        header: Optional header comment
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        # Write header
        if header:
            for line in header.strip().split("\n"):
                f.write(f"C  {line}\n")
        else:
            f.write("C  IWFM PreProcessor Main Input File\n")
            f.write("C  Generated by pyiwfm\n")
            f.write("C\n")

        # Write model name
        f.write(f"{config.model_name:<40} / MODEL_NAME\n")

        # Write core files
        if config.nodes_file:
            rel_path = _make_relative_path(filepath.parent, config.nodes_file)
            f.write(f"{rel_path:<40} / NODES_FILE\n")

        if config.elements_file:
            rel_path = _make_relative_path(filepath.parent, config.elements_file)
            f.write(f"{rel_path:<40} / ELEMENTS_FILE\n")

        if config.stratigraphy_file:
            rel_path = _make_relative_path(filepath.parent, config.stratigraphy_file)
            f.write(f"{rel_path:<40} / STRATIGRAPHY_FILE\n")

        if config.subregions_file:
            rel_path = _make_relative_path(filepath.parent, config.subregions_file)
            f.write(f"{rel_path:<40} / SUBREGIONS_FILE\n")

        # Write settings
        f.write(f"{config.n_layers:<40} / N_LAYERS\n")
        f.write(f"{config.length_unit:<40} / LENGTH_UNIT\n")
        f.write(f"{config.area_unit:<40} / AREA_UNIT\n")
        f.write(f"{config.volume_unit:<40} / VOLUME_UNIT\n")


def _make_relative_path(base_dir: Path, target_path: Path) -> str:
    """Make a path relative to a base directory if possible."""
    try:
        return str(target_path.relative_to(base_dir))
    except ValueError:
        return str(target_path)


def save_model_to_preprocessor(
    model: IWFMModel,
    output_dir: Path | str,
    model_name: str | None = None,
) -> PreProcessorConfig:
    """
    Save an IWFMModel to PreProcessor input files.

    Creates all necessary input files in the output directory.

    Args:
        model: IWFMModel to save
        output_dir: Directory for output files
        model_name: Model name (defaults to model.name)

    Returns:
        PreProcessorConfig with paths to created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = model_name or model.name or "iwfm_model"

    # Create config
    config = PreProcessorConfig(
        base_dir=output_dir,
        model_name=name,
        n_layers=model.n_layers,
    )

    # Write nodes
    if model.mesh:
        config.nodes_file = output_dir / "nodes.dat"
        write_nodes(config.nodes_file, model.mesh.nodes)

        # Write elements
        config.elements_file = output_dir / "elements.dat"
        n_sr = model.mesh.n_subregions or 1
        write_elements(config.elements_file, model.mesh.elements, n_subregions=n_sr)

        # Write subregions if present
        if model.mesh.subregions:
            config.subregions_file = output_dir / "subregions.dat"
            _write_subregions_file(config.subregions_file, model.mesh.subregions)

    # Write stratigraphy
    if model.stratigraphy:
        config.stratigraphy_file = output_dir / "stratigraphy.dat"
        write_stratigraphy(config.stratigraphy_file, model.stratigraphy)

    # Write main input file
    main_file = output_dir / f"{name}_pp.in"
    write_preprocessor_main(main_file, config)

    return config


def _write_subregions_file(
    filepath: Path | str,
    subregions: dict[int, Subregion],
) -> None:
    """Write subregion definitions to a file."""
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        f.write("C  Subregion definitions\n")
        f.write("C  ID  NAME\n")
        f.write(f"{len(subregions):<10}                    / NSUBREGION\n")

        for sr_id in sorted(subregions.keys()):
            sr = subregions[sr_id]
            f.write(f"{sr.id:<5} {sr.name}\n")


def save_complete_model(
    model: IWFMModel,
    output_dir: Path | str,
    timeseries_format: str = "ascii",
    dss_file: Path | str | None = None,
    file_paths: dict[str, str] | None = None,
) -> dict[str, Path]:
    """
    Save a complete IWFM model to all input files.

    This is the main function for exporting a complete model, including
    all component files (groundwater, streams, lakes, rootzone), time series,
    and the preprocessor/simulation control files.

    Delegates to :class:`~pyiwfm.io.model_writer.CompleteModelWriter` for
    the actual writing.

    Args:
        model: IWFMModel to save
        output_dir: Directory for output files
        timeseries_format: Format for time series data ("ascii" or "dss")
        dss_file: Path to DSS file for time series (if format is "dss")
        file_paths: Optional dict of {file_key: relative_path} overrides.
            If None, uses default nested layout.

    Returns:
        Dictionary mapping file type to output path

    Example:
        >>> from pyiwfm.io import save_complete_model
        >>> files = save_complete_model(model, Path("./model_output"))
        >>> print(f"Wrote {len(files)} files")
    """
    from pyiwfm.io.config import ModelWriteConfig, OutputFormat
    from pyiwfm.io.model_writer import CompleteModelWriter

    ts_fmt = OutputFormat.DSS if timeseries_format == "dss" else OutputFormat.TEXT

    # Propagate component versions from loaded model metadata so the
    # writer produces files matching the original model's version tags.
    version_kwargs: dict[str, str] = {}
    for key in ("gw_version", "stream_version", "lake_version", "rootzone_version"):
        val = model.metadata.get(key)
        if val:
            version_kwargs[key] = val

    config = ModelWriteConfig(
        output_dir=Path(output_dir),
        ts_format=ts_fmt,
        dss_file=str(dss_file) if dss_file else "model.dss",
        file_paths=file_paths or {},
        **version_kwargs,  # type: ignore[arg-type]
    )
    writer = CompleteModelWriter(model, config)
    result = writer.write_all()
    return result.files
