"""
Data export API routes: CSV, GeoJSON, GeoPackage, and plot downloads.

Includes timeseries export endpoints for head, hydrograph, and budget data.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import tempfile
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from starlette.responses import StreamingResponse

from pyiwfm.visualization.webapi.config import model_state, require_model

logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    """Sanitize a string for use in a filename.

    Replaces path separators and spaces with underscores to prevent path
    traversal while preserving the full location name.
    """
    import re

    # Replace path separators, colons, and other unsafe chars with underscores
    safe = re.sub(r'[/\\:*?"<>|]', "_", name)
    safe = safe.replace(" ", "_")
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_")


def _json_default(obj: object) -> object:
    """``default=`` callable for ``json.dumps`` covering NumPy / pandas types.

    GeoJSON payloads built from mesh data contain ``np.int64`` indices, and
    timeseries records may contain ``np.float64``/``pd.Timestamp`` values
    that ``json.dumps`` cannot serialize without help.
    """
    import numpy as np
    import pandas as pd

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp | np.datetime64):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/heads-csv")
def export_heads_csv(
    timestep: int = Query(default=0, ge=0, description="Timestep index"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> Response:
    """
    Export head values as a CSV file.

    Returns per-node head values for the specified timestep and layer.
    """
    loader = model_state.get_head_loader()
    if loader is None:
        raise HTTPException(status_code=404, detail="No head data available")

    if timestep >= loader.n_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Timestep {timestep} out of range [0, {loader.n_frames})",
        )

    frame = loader.get_frame(timestep)
    layer_idx = layer - 1
    if layer_idx >= frame.shape[1]:
        raise HTTPException(
            status_code=400,
            detail=f"Layer {layer} out of range [1, {frame.shape[1]}]",
        )

    values = frame[:, layer_idx]
    dt = loader.times[timestep] if timestep < len(loader.times) else None

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["node_id", "head_ft"])
    for i, val in enumerate(values):
        writer.writerow([i + 1, round(float(val), 3)])

    filename = f"heads_ts{timestep}_layer{layer}"
    if dt:
        filename += f"_{dt.strftime('%Y%m%d')}"
    filename += ".csv"

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/mesh-geojson")
def export_mesh_geojson(
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
) -> Response:
    """
    Export the mesh as a GeoJSON file.

    Returns element polygons in WGS84 as a downloadable GeoJSON file.
    """
    model_state.require_loaded()

    from pyiwfm.visualization.webapi.routes.mesh import get_mesh_geojson

    try:
        geojson = get_mesh_geojson(layer=layer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(
        content=json.dumps(geojson, default=_json_default),
        media_type="application/geo+json",
        headers={"Content-Disposition": f"attachment; filename=mesh_layer{layer}.geojson"},
    )


@router.get("/budget-csv")
def export_budget_csv(
    budget_type: str = Query(..., description="Budget type"),
    location: str = Query(default="", description="Location name or index"),
) -> Response:
    """
    Export budget time series data as a CSV file.
    """
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    loc = location if location else 0

    try:
        times_arr, values_arr = reader.get_values(loc)
        headers = reader.get_column_headers(loc)
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Build time strings
    ts = reader.header.timestep
    from datetime import timedelta

    use_months = "MON" in ts.unit.upper() if ts.unit else False
    if use_months:
        from dateutil.relativedelta import relativedelta

    time_strings = []
    if ts.start_datetime:
        for i in range(len(times_arr)):
            if use_months:
                dt = ts.start_datetime + relativedelta(months=i)
            else:
                dt = ts.start_datetime + timedelta(minutes=ts.delta_t_minutes * i)
            time_strings.append(dt.isoformat())
    else:
        time_strings = [str(t) for t in times_arr.tolist()]

    import numpy as np

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["datetime"] + headers)
    # Vectorize the per-cell round(float(...), 4) — np.round runs at C speed
    # and .tolist() yields native Python floats for csv.writer to format.
    rounded_rows = np.round(np.asarray(values_arr, dtype=float), 4).tolist()
    for ts_str, vals in zip(time_strings, rounded_rows, strict=True):
        writer.writerow([ts_str, *vals])

    loc_name = location or reader.locations[0]
    safe_name = _safe_filename(loc_name)
    filename = f"budget_{budget_type}_{safe_name}.csv"

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/budget-excel")
def export_budget_excel(
    budget_type: str = Query(..., description="Budget type"),
    location: str = Query(default="", description="Location name/index or empty for all"),
) -> Response:
    """Export budget data as an Excel workbook."""
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    from pyiwfm.io.budget_excel import budget_to_excel

    location_ids: list[int] | None = None
    if location:
        try:
            loc_idx = reader.get_location_index(location)
            location_ids = [loc_idx + 1]
        except (KeyError, IndexError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        from pathlib import Path

        budget_to_excel(
            reader=reader,
            output_path=tmp_path,
            location_ids=location_ids,
        )
        data = Path(tmp_path).read_bytes()
    except Exception as e:
        logger.exception("Budget Excel export failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        import os

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    safe_type = _safe_filename(budget_type)
    filename = f"budget_{safe_type}.xlsx"

    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/hydrograph-csv")
def export_hydrograph_csv(
    type: str = Query(..., description="Type: gw, stream, subsidence, tile_drain"),
    location_id: int = Query(..., description="Location/node ID"),
) -> Response:
    """
    Export hydrograph time series as a CSV file.
    """
    model_state.require_loaded()

    if type == "gw":
        reader = model_state.get_gw_hydrograph_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(status_code=404, detail="No GW hydrograph data available")

        phys_locs = model_state.get_gw_physical_locations()
        location_index = location_id - 1

        if phys_locs:
            if location_index < 0 or location_index >= len(phys_locs):
                raise HTTPException(
                    status_code=404,
                    detail=f"GW hydrograph {location_id} out of range",
                )
            col_idx = phys_locs[location_index]["columns"][0][0]
        else:
            # Fallback: raw column index
            col_idx = location_index

        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"GW hydrograph {location_id} out of range",
            )

        times, values = reader.get_time_series(col_idx)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["datetime", "head_ft"])
        for t, v in zip(times, values, strict=False):
            writer.writerow([t, round(v, 3)])

        filename = f"hydrograph_gw_{location_id}.csv"

    elif type == "stream":
        reader = model_state.get_stream_hydrograph_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No stream hydrograph data available",
            )

        col_idx = reader.find_column_by_node_id(location_id)
        if col_idx is None and location_id in reader.hydrograph_ids:
            col_idx = reader.hydrograph_ids.index(location_id)
        if col_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Stream node {location_id} not found",
            )

        times, values = reader.get_time_series(col_idx)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["datetime", "flow_cfs"])
        for t, v in zip(times, values, strict=False):
            writer.writerow([t, round(v, 3)])

        filename = f"hydrograph_stream_{location_id}.csv"

    elif type == "subsidence":
        reader = model_state.get_subsidence_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No subsidence hydrograph data available",
            )

        col_idx = location_id - 1
        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"Subsidence location {location_id} out of range",
            )

        times, values = reader.get_time_series(col_idx)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["datetime", "subsidence_ft"])
        for t, v in zip(times, values, strict=False):
            writer.writerow([t, round(v, 3)])

        filename = f"hydrograph_subsidence_{location_id}.csv"

    elif type == "tile_drain":
        reader = model_state.get_tile_drain_reader()
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No tile drain hydrograph data available",
            )

        col_idx = location_id - 1
        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"Tile drain location {location_id} out of range",
            )

        times, values = reader.get_time_series(col_idx)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["datetime", "flow_volume"])
        for t, v in zip(times, values, strict=False):
            writer.writerow([t, round(v, 3)])

        filename = f"hydrograph_tile_drain_{location_id}.csv"

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown type: {type}. Use: gw, stream, subsidence, tile_drain",
        )

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/geopackage")
def export_geopackage(
    include_streams: bool = Query(default=True, description="Include stream reaches"),
    include_subregions: bool = Query(default=True, description="Include subregion polygons"),
    include_boundary: bool = Query(default=True, description="Include model boundary"),
    model_length_unit: str | None = Query(
        default=None,
        description=(
            "Override the model's native coordinate length unit ('FEET' "
            "or 'METERS') when it can't be determined automatically from "
            "the PreProcessor main file or the Nodes file conversion "
            "factor. Defaults to the loaded model's own resolved unit."
        ),
    ),
) -> Response:
    """Export the model mesh as a GeoPackage file.

    Creates a multi-layer GeoPackage containing nodes, elements,
    and optionally streams, subregions, and boundary polygon.
    """
    model = model_state.require_model()
    if model.grid is None:
        raise HTTPException(status_code=404, detail="No mesh/grid loaded")

    from pyiwfm.visualization.gis_export import GISExporter

    exporter = GISExporter(
        grid=model.grid,
        stratigraphy=model.stratigraphy,
        streams=model.streams,
        crs=model_state._crs,
        model_length_unit=model_length_unit or model.metadata.get("simulation_length_unit"),
    )

    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        exporter.export_geopackage(
            tmp_path,
            include_streams=include_streams,
            include_subregions=include_subregions,
            include_boundary=include_boundary,
        )

        from pathlib import Path

        data = Path(tmp_path).read_bytes()

        model_name = model.name or "model"
        safe_name = _safe_filename(model_name)
        filename = f"{safe_name}.gpkg"

        headers = {"Content-Disposition": f"attachment; filename={filename}"}
        if exporter.resolved_model_length_unit:
            headers["X-Model-Length-Unit"] = exporter.resolved_model_length_unit
        else:
            headers["X-Model-Length-Unit-Ambiguous"] = "true"

        return Response(
            content=data,
            media_type="application/geopackage+sqlite3",
            headers=headers,
        )
    except Exception as e:
        logger.exception("GeoPackage export failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        import os

        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.get("/plot/{plot_type}")
def export_plot(
    plot_type: str,
    format: str = Query(default="png", description="Image format: png or svg"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
    timestep: int = Query(default=0, ge=0, description="Timestep index"),
    width: float = Query(default=10.0, gt=0, description="Figure width in inches"),
    height: float = Query(default=8.0, gt=0, description="Figure height in inches"),
    dpi: int = Query(default=150, ge=72, le=600, description="DPI for PNG output"),
) -> Response:
    """Generate publication-quality matplotlib figures.

    Supported plot types:
    - mesh: Model mesh with elements and nodes
    - heads: Head contour map for a timestep/layer
    - streams: Stream network colored by reach
    - elements: Elements colored by subregion
    """
    model = model_state.require_model()
    if model.grid is None:
        raise HTTPException(status_code=404, detail="No mesh/grid loaded")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyiwfm.visualization.plotting import (
        plot_elements,
        plot_mesh,
        plot_scalar_field,
        plot_streams,
    )

    try:
        fig = None
        if plot_type == "mesh":
            fig, _ax = plot_mesh(model.grid, figsize=(width, height))
        elif plot_type == "elements":
            fig, _ax = plot_elements(model.grid, figsize=(width, height))
        elif plot_type == "streams":
            if model.streams is None:
                raise HTTPException(status_code=404, detail="No stream network loaded")
            fig, _ax = plot_streams(model.streams, figsize=(width, height))
        elif plot_type == "heads":
            loader = model_state.get_head_loader()
            if loader is None:
                raise HTTPException(status_code=404, detail="No head data available")
            if timestep >= loader.n_frames:
                raise HTTPException(
                    status_code=400,
                    detail=f"Timestep {timestep} out of range [0, {loader.n_frames})",
                )
            import numpy as np

            frame = loader.get_frame(timestep)
            layer_idx = layer - 1
            if layer_idx >= frame.shape[1]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Layer {layer} out of range [1, {frame.shape[1]}]",
                )
            values = frame[:, layer_idx]
            # Mask dry cells
            values = np.where(values < -9000, np.nan, values)
            fig, _ax = plot_scalar_field(
                model.grid,
                values,
                figsize=(width, height),
            )
            _ax.set_title(f"Head - Layer {layer}, Timestep {timestep}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown plot type: {plot_type}. Supported: mesh, elements, streams, heads",
            )

        buf = io.BytesIO()
        if format == "svg":
            fig.savefig(buf, format="svg", bbox_inches="tight")
            media_type = "image/svg+xml"
            ext = "svg"
        else:
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            media_type = "image/png"
            ext = "png"
        plt.close(fig)
        buf.seek(0)

        filename = f"{plot_type}_layer{layer}_ts{timestep}.{ext}"
        return Response(
            content=buf.getvalue(),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Plot generation failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Helpers for timeseries export
# ---------------------------------------------------------------------------


def _sanitize_value(v: float) -> float | None:
    """Replace NaN/Inf with ``None`` for JSON or empty string for CSV."""
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, 4)


def _csv_streaming_response(
    rows: list[list[object]],
    headers: list[str],
    filename: str,
) -> StreamingResponse:
    """Build a ``StreamingResponse`` for CSV data."""

    def _generate() -> Iterator[str]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        buf.seek(0)
        yield buf.read()
        for row in rows:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(row)
            buf.seek(0)
            yield buf.read()

    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# GET /api/export/timeseries/head
# ---------------------------------------------------------------------------


@router.get("/report", response_model=None)
def export_model_report(
    format: str = Query(default="html", description="Report format: html or json"),
) -> Response:
    """Export a summary report of the loaded model."""
    model = require_model()

    if format == "json":
        report_data = _build_report_data(model)
        content = json.dumps(report_data, indent=2, default=_json_default)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=model_report.json"},
        )

    # HTML report
    html = _build_html_report(model)
    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": "attachment; filename=model_report.html"},
    )


def _build_report_data(model: object) -> dict[str, object]:
    """Build report data dictionary from model."""
    data: dict[str, object] = {"title": "IWFM Model Summary Report"}

    grid = getattr(model, "grid", None)
    if grid:
        data["mesh"] = {
            "n_nodes": grid.n_nodes,
            "n_elements": grid.n_elements,
        }

    strat = getattr(model, "stratigraphy", None)
    if strat:
        data["stratigraphy"] = {
            "n_layers": strat.n_layers,
        }

    # Add component summaries
    for comp_name in [
        "groundwater",
        "streams",
        "lakes",
        "rootzone",
        "small_watersheds",
        "unsaturated_zone",
    ]:
        comp = getattr(model, comp_name, None)
        if comp:
            data[comp_name] = {"loaded": True, "n_items": getattr(comp, "n_items", None)}
        else:
            data[comp_name] = {"loaded": False}

    return data


def _build_html_report(model: object) -> str:
    """Build an HTML summary report."""
    data = _build_report_data(model)

    lines = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>IWFM Model Report</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,"
        " sans-serif; margin: 40px; color: #333; }",
        "h1 { color: #1976d2; border-bottom: 2px solid #1976d2; padding-bottom: 8px; }",
        "h2 { color: #555; margin-top: 24px; }",
        "table { border-collapse: collapse; margin: 12px 0; width: 100%; max-width: 600px; }",
        "td, th { padding: 8px 16px; border: 1px solid #ddd; text-align: left; }",
        "th { background: #f5f5f5; font-weight: 600; }",
        ".loaded { color: #2e7d32; } .not-loaded { color: #999; }",
        ".footer { margin-top: 40px; font-size: 12px; color: #999;"
        " border-top: 1px solid #eee; padding-top: 8px; }",
        "</style>",
        "</head><body>",
        f"<h1>{data['title']}</h1>",
    ]

    # Mesh section
    mesh_data = data.get("mesh")
    if isinstance(mesh_data, dict):
        lines.append("<h2>Mesh</h2>")
        lines.append("<table>")
        n_nodes = mesh_data.get("n_nodes", "N/A")
        n_elements = mesh_data.get("n_elements", "N/A")
        lines.append(f"<tr><th>Nodes</th><td>{n_nodes:,}</td></tr>")
        lines.append(f"<tr><th>Elements</th><td>{n_elements:,}</td></tr>")
        lines.append("</table>")

    strat_data = data.get("stratigraphy")
    if isinstance(strat_data, dict):
        n_layers = strat_data.get("n_layers", "N/A")
        lines.append(f"<p>Layers: <strong>{n_layers}</strong></p>")

    # Components
    lines.append("<h2>Components</h2>")
    lines.append("<table>")
    lines.append("<tr><th>Component</th><th>Status</th><th>Items</th></tr>")
    for comp_name in [
        "groundwater",
        "streams",
        "lakes",
        "rootzone",
        "small_watersheds",
        "unsaturated_zone",
    ]:
        comp = data.get(comp_name)
        if isinstance(comp, dict):
            loaded = comp.get("loaded", False)
            if loaded:
                status = "<span class='loaded'>Loaded</span>"
                n_items = comp.get("n_items", "-")
            else:
                status = "<span class='not-loaded'>Not loaded</span>"
                n_items = "-"
            display_name = comp_name.replace("_", " ").title()
            lines.append(f"<tr><td>{display_name}</td><td>{status}</td><td>{n_items}</td></tr>")
    lines.append("</table>")

    # Footer
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"<div class='footer'>Generated by pyiwfm on {now}</div>")
    lines.append("</body></html>")

    return "\n".join(lines)


@router.get("/timeseries/head", response_model=None)
def export_timeseries_head(
    node_id: int = Query(..., ge=1, description="Node ID (1-based)"),
    layer: int = Query(default=1, ge=1, description="Layer number (1-based)"),
    format: Literal["csv", "json"] = Query(default="csv", description="Output format"),
) -> Response | StreamingResponse:
    """Export head timeseries at a specific node as CSV or JSON.

    Iterates over all available timesteps and extracts the head value
    for the requested node and layer.
    """
    loader = model_state.get_head_loader()
    if loader is None:
        raise HTTPException(status_code=404, detail="No head data available")

    layer_idx = layer - 1
    if layer_idx >= loader.n_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Layer {layer} out of range [1, {loader.n_layers}]",
        )

    node_idx = node_id - 1
    if node_idx < 0 or node_idx >= loader.n_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Node {node_id} out of range [1, {loader.n_nodes}]",
        )

    # Extract timeseries across all frames
    times = loader.times
    rows: list[list[object]] = []
    for i in range(loader.n_frames):
        frame = loader.get_frame(i)
        raw_val = float(frame[node_idx, layer_idx])
        clean = _sanitize_value(raw_val)
        dt_str = times[i].isoformat() if i < len(times) else str(i)
        rows.append([dt_str, clean if clean is not None else ""])

    if format == "json":
        records = []
        for row in rows:
            records.append({"datetime": row[0], "head_value": row[1] if row[1] != "" else None})
        return Response(
            content=json.dumps(records, default=_json_default),
            media_type="application/json",
        )

    filename = f"head_node{node_id}_layer{layer}.csv"
    return _csv_streaming_response(rows, ["datetime", "head_value"], filename)


# ---------------------------------------------------------------------------
# GET /api/export/timeseries/hydrograph
# ---------------------------------------------------------------------------


@router.get("/timeseries/hydrograph", response_model=None)
def export_timeseries_hydrograph(
    location_id: int = Query(..., ge=1, description="Location/node ID (1-based)"),
    type: Literal["gw", "stream", "subsidence", "tile_drain"] = Query(
        ..., description="Hydrograph type"
    ),
    format: Literal["csv", "json"] = Query(default="csv", description="Output format"),
) -> Response | StreamingResponse:
    """Export hydrograph timeseries data as CSV or JSON.

    Delegates to the appropriate hydrograph reader based on *type*.
    """
    model_state.require_loaded()

    value_label: str
    if type == "gw":
        reader = model_state.get_gw_hydrograph_reader()
        value_label = "head_ft"
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(status_code=404, detail="No GW hydrograph data available")

        phys_locs = model_state.get_gw_physical_locations()
        location_index = location_id - 1

        if phys_locs:
            if location_index < 0 or location_index >= len(phys_locs):
                raise HTTPException(
                    status_code=404,
                    detail=f"GW hydrograph {location_id} out of range",
                )
            col_idx = phys_locs[location_index]["columns"][0][0]
        else:
            col_idx = location_index

        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"GW hydrograph {location_id} out of range",
            )

    elif type == "stream":
        reader = model_state.get_stream_hydrograph_reader()
        value_label = "flow_cfs"
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No stream hydrograph data available",
            )

        col_idx_maybe = reader.find_column_by_node_id(location_id)
        if col_idx_maybe is None and location_id in reader.hydrograph_ids:
            col_idx_maybe = reader.hydrograph_ids.index(location_id)
        if col_idx_maybe is None:
            raise HTTPException(
                status_code=404,
                detail=f"Stream node {location_id} not found",
            )
        col_idx = col_idx_maybe

    elif type == "subsidence":
        reader = model_state.get_subsidence_reader()
        value_label = "subsidence_ft"
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No subsidence hydrograph data available",
            )

        col_idx = location_id - 1
        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"Subsidence location {location_id} out of range",
            )

    else:  # tile_drain
        reader = model_state.get_tile_drain_reader()
        value_label = "flow_volume"
        if reader is None or reader.n_timesteps == 0:
            raise HTTPException(
                status_code=404,
                detail="No tile drain hydrograph data available",
            )

        col_idx = location_id - 1
        if col_idx < 0 or col_idx >= reader.n_columns:
            raise HTTPException(
                status_code=404,
                detail=f"Tile drain location {location_id} out of range",
            )

    times_list, values_list = reader.get_time_series(col_idx)

    rows: list[list[object]] = []
    for t, v in zip(times_list, values_list, strict=False):
        clean = _sanitize_value(float(v))
        rows.append([t, clean if clean is not None else ""])

    if format == "json":
        records = []
        for row in rows:
            records.append({"datetime": row[0], value_label: row[1] if row[1] != "" else None})
        return Response(
            content=json.dumps(records, default=_json_default),
            media_type="application/json",
        )

    filename = f"hydrograph_{type}_{location_id}.csv"
    return _csv_streaming_response(rows, ["datetime", value_label], filename)


# ---------------------------------------------------------------------------
# GET /api/export/timeseries/budget
# ---------------------------------------------------------------------------


@router.get("/timeseries/budget", response_model=None)
def export_timeseries_budget(
    budget_type: str = Query(..., description="Budget type"),
    location: str = Query(default="", description="Location name or index"),
    format: Literal["csv", "json"] = Query(default="csv", description="Output format"),
) -> Response | StreamingResponse:
    """Export budget timeseries data as CSV or JSON.

    Returns all budget columns for the specified budget type and location.
    """
    reader = model_state.get_budget_reader(budget_type)
    if reader is None:
        raise HTTPException(
            status_code=404,
            detail=f"Budget type '{budget_type}' not available",
        )

    loc: str | int = location if location else 0

    try:
        times_arr, values_arr = reader.get_values(loc)
        col_headers = reader.get_column_headers(loc)
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Build time strings from budget header
    ts = reader.header.timestep
    from datetime import timedelta

    use_months = "MON" in ts.unit.upper() if ts.unit else False
    if use_months:
        from dateutil.relativedelta import relativedelta

    time_strings: list[str] = []
    if ts.start_datetime:
        for i in range(len(times_arr)):
            if use_months:
                dt = ts.start_datetime + relativedelta(months=i)
            else:
                dt = ts.start_datetime + timedelta(minutes=ts.delta_t_minutes * i)
            time_strings.append(dt.isoformat())
    else:
        time_strings = [str(t) for t in times_arr.tolist()]

    # Build rows with sanitized values. Vectorize the round + NaN/Inf check
    # over the whole matrix; the per-cell `_sanitize_value(float(...))` loop
    # was the hot spot when exporting wide budget tables.
    import numpy as np

    arr = np.asarray(values_arr, dtype=float)
    finite_mask = np.isfinite(arr)
    rounded_rows = np.round(arr, 4).tolist()
    finite_rows = finite_mask.tolist()
    rows: list[list[object]] = [
        [
            ts_str,
            *(v if ok else "" for v, ok in zip(row_vals, row_ok, strict=True)),
        ]
        for ts_str, row_vals, row_ok in zip(time_strings, rounded_rows, finite_rows, strict=True)
    ]

    all_headers = ["datetime"] + col_headers

    if format == "json":
        records = []
        for row in rows:
            record: dict[str, object] = {"datetime": row[0]}
            for k, header in enumerate(col_headers):
                val = row[k + 1]
                record[header] = val if val != "" else None
            records.append(record)
        return Response(
            content=json.dumps(records, default=_json_default),
            media_type="application/json",
        )

    loc_name = location or reader.locations[0]
    safe_name = _safe_filename(str(loc_name))
    filename = f"budget_ts_{budget_type}_{safe_name}.csv"
    return _csv_streaming_response(rows, all_headers, filename)
