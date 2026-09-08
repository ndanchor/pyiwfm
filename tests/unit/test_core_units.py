"""Unit tests for pyiwfm.core.units."""

from __future__ import annotations

import pytest

from pyiwfm.core.units import (
    FEET_PER_METER,
    METERS_PER_FOOT,
    normalize_length_unit_name,
    resolve_model_length_unit,
)


class TestNormalizeLengthUnitName:
    """Tests for normalize_length_unit_name()."""

    @pytest.mark.parametrize(
        "raw",
        ["FEET", "feet", "FT", "ft", "ft.", "Foot", "US survey foot", "foot"],
    )
    def test_recognizes_feet_variants(self, raw: str) -> None:
        assert normalize_length_unit_name(raw) == "FEET"

    @pytest.mark.parametrize(
        "raw",
        ["METERS", "meters", "M", "m", "m.", "metre", "Metre", "METRES"],
    )
    def test_recognizes_meters_variants(self, raw: str) -> None:
        assert normalize_length_unit_name(raw) == "METERS"

    @pytest.mark.parametrize("raw", [None, "", "   ", "furlongs", "degree"])
    def test_unrecognized_or_empty_returns_none(self, raw: str | None) -> None:
        assert normalize_length_unit_name(raw) is None


class TestResolveModelLengthUnit:
    """Tests for resolve_model_length_unit()."""

    def test_factltou_1_unitltou_feet_is_feet(self) -> None:
        """The common case (no output conversion): simulation unit ==
        output unit == feet."""
        assert resolve_model_length_unit("FEET", 1.0) == "FEET"

    def test_factltou_1_unitltou_meters_is_meters(self) -> None:
        assert resolve_model_length_unit("METERS", 1.0) == "METERS"

    def test_feet_output_from_meter_simulation(self) -> None:
        """UNITLTOU=FEET with FACTLTOU=3.28084 means one simulation unit
        (a meter) reports as 3.28084 feet."""
        assert resolve_model_length_unit("FEET", FEET_PER_METER) == "METERS"

    def test_meters_output_from_foot_simulation(self) -> None:
        """UNITLTOU=METERS with FACTLTOU=0.3048 means one simulation unit
        (a foot) reports as 0.3048 meters."""
        assert resolve_model_length_unit("METERS", METERS_PER_FOOT) == "FEET"

    def test_missing_unitltou_returns_none(self) -> None:
        assert resolve_model_length_unit(None, 1.0) is None

    def test_missing_factltou_returns_none(self) -> None:
        assert resolve_model_length_unit("FEET", None) is None

    @pytest.mark.parametrize("bad_factor", [0.0, -1.0])
    def test_nonpositive_factltou_returns_none(self, bad_factor: float) -> None:
        assert resolve_model_length_unit("FEET", bad_factor) is None

    def test_unrecognized_factor_returns_none(self) -> None:
        """A factor matching neither 1 foot nor 1 meter in physical size
        can't be resolved to either unit."""
        assert resolve_model_length_unit("FEET", 2.5) is None
