"""
Length unit helpers shared across the preprocessor reader and GIS export.

IWFM models carry length information in two independent, easily-confused
places:

- The **PreProcessor Nodes file** has an optional conversion factor
  (``FACT``) applied to the raw ``X``/``Y`` values it contains so that the
  resulting coordinates are expressed in the model's internal/simulation
  length unit. :func:`pyiwfm.io.ascii.read_nodes` already applies this
  factor, so ``Node.x``/``Node.y`` are always in that internal unit -- but
  the factor alone doesn't say *which* unit that is (a factor of ``1.0``
  is genuinely ambiguous: it just means the raw file was already in the
  model's native unit).
- The **PreProcessor main input file** declares ``FACTLTOU``/``UNITLTOU``:
  the factor and unit name used to convert the simulation length unit to
  an output/reporting unit (e.g. for printed geometry summaries). When
  ``FACTLTOU`` is known, it can be inverted against ``UNITLTOU`` to
  recover the *actual* simulation length unit -- this is authoritative
  and should be preferred over guessing from the Nodes file factor alone.

Both pieces of information reduce to the same two-valued question this
module answers: is a given length quantity in feet or meters?
"""

from __future__ import annotations

import math

METERS_PER_FOOT = 0.3048
FEET_PER_METER = 1.0 / METERS_PER_FOOT


def normalize_length_unit_name(name: str | None) -> str | None:
    """
    Normalize a free-form length unit string to ``"FEET"``, ``"METERS"``,
    or ``None`` if it isn't recognized.

    Handles both IWFM-style tokens (``"FEET"``, ``"FT"``, ``"METERS"``,
    ``"M"``) and pyproj/PROJ axis unit names (``"US survey foot"``,
    ``"metre"``, ``"foot"``, ...).
    """
    if not name:
        return None
    n = name.strip().lower()
    if not n:
        return None
    if "foot" in n or "feet" in n or n in ("ft", "ft."):
        return "FEET"
    if "met" in n or n in ("m", "m."):
        return "METERS"
    return None


def resolve_model_length_unit(
    unitltou: str | None,
    factltou: float | None,
    rel_tol: float = 1e-3,
) -> str | None:
    """
    Derive a model's native (simulation) length unit from the PreProcessor
    main file's ``FACTLTOU``/``UNITLTOU`` pair.

    IWFM defines ``FACTLTOU`` as the factor that converts a
    simulation-unit length value to the reported ``UNITLTOU`` value::

        output_value = simulation_value * FACTLTOU

    so one simulation unit is worth ``FACTLTOU`` output units, and its
    physical size (in meters) is ``FACTLTOU * size_of_one(UNITLTOU)``.
    Matching that size against 1 foot / 1 meter tells us whether the
    model's node coordinates (already expressed in simulation units by
    :func:`pyiwfm.io.ascii.read_nodes`) are in feet or meters.

    Parameters
    ----------
    unitltou : str, optional
        The ``UNITLTOU`` output length unit string (e.g. ``"FEET"``).
    factltou : float, optional
        The ``FACTLTOU`` conversion factor.

    Returns
    -------
    str or None
        ``"FEET"``, ``"METERS"``, or ``None`` if the simulation unit
        can't be determined (missing/unrecognized ``UNITLTOU``, missing
        or non-positive ``FACTLTOU``, or a factor that matches neither
        unit within tolerance).
    """
    output_unit = normalize_length_unit_name(unitltou)
    if output_unit is None or factltou is None or factltou <= 0:
        return None

    output_unit_size_m = METERS_PER_FOOT if output_unit == "FEET" else 1.0
    sim_unit_size_m = factltou * output_unit_size_m

    if math.isclose(sim_unit_size_m, METERS_PER_FOOT, rel_tol=rel_tol):
        return "FEET"
    if math.isclose(sim_unit_size_m, 1.0, rel_tol=rel_tol):
        return "METERS"
    return None
