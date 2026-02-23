"""
Unit tests for registration helper functions.

Tests are pure (no I/O) and cover:
- _registration_utils_v2: add_translation_to_roi
- find_registration_consensus: _get_roi_translation, _find_roi_consensus,
  _apply_consensus_to_roi
- apply_registration_to_image: _get_ref_path_heuristic
"""

import pytest
from ngio import PixelSize, Roi

from fractal_tasks_core._registration_utils_v2 import add_translation_to_roi
from fractal_tasks_core.apply_registration_to_image import _get_ref_path_heuristic
from fractal_tasks_core.find_registration_consensus import (
    _apply_consensus_to_roi,
    _find_roi_consensus,
    _get_roi_translation,
)

# ---------------------------------------------------------------------------
# _registration_utils_v2.add_translation_to_roi
# ---------------------------------------------------------------------------


def _make_roi(**extra) -> Roi:
    return Roi(y=0.0, x=0.0, z=0.0, y_length=20.0, x_length=20.0, z_length=1.0, **extra)


def _pixel_size(y: float = 1.3, x: float = 1.3, z: float = 1.0) -> PixelSize:
    return PixelSize(y=y, x=x, z=z)


def test_add_translation_to_roi_2d():
    roi = _make_roi()
    pixel_size = _pixel_size()
    updated = add_translation_to_roi(roi, [2.0, 3.0], pixel_size)
    assert updated.model_extra is not None
    assert updated.model_extra["translation_z"] == pytest.approx(0.0)
    assert updated.model_extra["translation_y"] == pytest.approx(2.0 * 1.3)
    assert updated.model_extra["translation_x"] == pytest.approx(3.0 * 1.3)


def test_add_translation_to_roi_3d():
    roi = _make_roi()
    pixel_size = _pixel_size(y=1.3, x=1.3, z=2.0)
    updated = add_translation_to_roi(roi, [1.0, 2.0, 3.0], pixel_size)
    assert updated.model_extra is not None
    assert updated.model_extra["translation_z"] == pytest.approx(1.0 * 2.0)
    assert updated.model_extra["translation_y"] == pytest.approx(2.0 * 1.3)
    assert updated.model_extra["translation_x"] == pytest.approx(3.0 * 1.3)


def test_add_translation_to_roi_invalid():
    roi = _make_roi()
    pixel_size = _pixel_size()
    with pytest.raises(ValueError, match="Wrong input"):
        add_translation_to_roi(roi, [1.0], pixel_size)


# ---------------------------------------------------------------------------
# find_registration_consensus._get_roi_translation
# ---------------------------------------------------------------------------


def test_get_roi_translation_present():
    roi = _make_roi(translation_y=-1.3, translation_x=-2.6, translation_z=0.0)
    assert _get_roi_translation(roi, "y") == pytest.approx(-1.3)
    assert _get_roi_translation(roi, "x") == pytest.approx(-2.6)
    assert _get_roi_translation(roi, "z") == pytest.approx(0.0)


def test_get_roi_translation_absent():
    roi = _make_roi()
    assert _get_roi_translation(roi, "y") == pytest.approx(0.0)
    assert _get_roi_translation(roi, "x") == pytest.approx(0.0)
    assert _get_roi_translation(roi, "z") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# find_registration_consensus._find_roi_consensus
# ---------------------------------------------------------------------------


def test_find_roi_consensus_zero_shift():
    """With all translations at zero the consensus equals the base ROI."""
    base = Roi(
        name="FOV_1", y=5.0, x=10.0, z=0.0, y_length=20.0, x_length=20.0, z_length=1.0
    )
    rois = [base, base.model_copy()]
    consensus = _find_roi_consensus(rois)
    assert consensus.y == pytest.approx(5.0)
    assert consensus.x == pytest.approx(10.0)
    assert consensus.y_length == pytest.approx(20.0)
    assert consensus.x_length == pytest.approx(20.0)


def test_find_roi_consensus_known_shift():
    """
    Acquisition 0 (reference): no translations.
    Acquisition 1: translation_y=-1.3, translation_x=-2.6.

    Expected consensus:
      pos   = base.pos + max(0, -1.3)   = base.pos + 0        = base.pos
      len_y = 20.8 - max(0,-1.3) + min(0,-1.3) = 20.8 - 0 + (-1.3) = 19.5
      len_x = 20.8 - max(0,-2.6) + min(0,-2.6) = 20.8 - 0 + (-2.6) = 18.2
    """
    roi_ref = Roi(
        name="image", y=0.0, x=0.0, z=0.0, y_length=20.8, x_length=20.8, z_length=1.0
    )
    roi_acq = roi_ref.model_copy(
        update={"translation_y": -1.3, "translation_x": -2.6, "translation_z": 0.0}
    )
    consensus = _find_roi_consensus([roi_ref, roi_acq])
    assert consensus.y == pytest.approx(0.0)
    assert consensus.x == pytest.approx(0.0)
    assert consensus.y_length == pytest.approx(19.5, abs=0.01)
    assert consensus.x_length == pytest.approx(18.2, abs=0.01)


# ---------------------------------------------------------------------------
# find_registration_consensus._apply_consensus_to_roi
# ---------------------------------------------------------------------------


def test_apply_consensus_to_roi_reference():
    """Reference acquisition (zero shift) → position equals the consensus position."""
    roi_ref = Roi(
        name="image", y=0.0, x=0.0, z=0.0, y_length=20.8, x_length=20.8, z_length=1.0
    )
    roi_acq = roi_ref.model_copy(
        update={"translation_y": -1.3, "translation_x": -2.6, "translation_z": 0.0}
    )
    consensus = _find_roi_consensus([roi_ref, roi_acq])

    shifted_ref = _apply_consensus_to_roi(roi_ref, consensus)
    # Reference has own_translation=0 → position = consensus.pos - 0 = 0,0
    assert shifted_ref.y == pytest.approx(0.0)
    assert shifted_ref.x == pytest.approx(0.0)
    assert shifted_ref.y_length == pytest.approx(consensus.y_length)
    assert shifted_ref.x_length == pytest.approx(consensus.x_length)


def test_apply_consensus_to_roi_shifted():
    """Shifted acquisition → position = consensus.pos - own_translation."""
    roi_ref = Roi(
        name="image", y=0.0, x=0.0, z=0.0, y_length=20.8, x_length=20.8, z_length=1.0
    )
    roi_acq = roi_ref.model_copy(
        update={"translation_y": -1.3, "translation_x": -2.6, "translation_z": 0.0}
    )
    consensus = _find_roi_consensus([roi_ref, roi_acq])

    shifted_acq = _apply_consensus_to_roi(roi_acq, consensus)
    # own_y=-1.3, own_x=-2.6 → pos = 0 - (-1.3) = 1.3 ; 0 - (-2.6) = 2.6
    assert shifted_acq.y == pytest.approx(1.3, abs=0.01)
    assert shifted_acq.x == pytest.approx(2.6, abs=0.01)
    assert shifted_acq.y_length == pytest.approx(consensus.y_length)
    assert shifted_acq.x_length == pytest.approx(consensus.x_length)


# ---------------------------------------------------------------------------
# apply_registration_to_image._get_ref_path_heuristic
# ---------------------------------------------------------------------------


def test_get_ref_path_heuristic_suffix_match():
    """Path '1_illum_corr' → pick the entry in the list with the same suffix."""
    result = _get_ref_path_heuristic(["0", "0_illum_corr"], "1_illum_corr")
    assert result == "0_illum_corr"


def test_get_ref_path_heuristic_no_suffix():
    """Path '1' (no suffix) → pick the entry with no suffix."""
    result = _get_ref_path_heuristic(["0", "0_illum_corr"], "1")
    assert result == "0"


def test_get_ref_path_heuristic_fallback(caplog):
    """No suffix match → return first sorted entry and emit a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        result = _get_ref_path_heuristic(["0_foo", "0_bar"], "1_baz")
    assert result == "0_bar"  # first sorted
    assert "heuristic" in caplog.text.lower()
