import numpy as np
import pytest

pytest.importorskip("PySide6")

from blab.ui.balloon import (
    HORIZONTAL_ANGLE_SCALAR_NAME,
    SPL_SCALAR_NAME,
    VERTICAL_ANGLE_SCALAR_NAME,
    _balloon_angle_arrays,
    _balloon_hover_text,
    _contour_levels,
)


def test_balloon_angle_arrays_match_horizontal_and_vertical_planes() -> None:
    theta = np.array(
        [
            [0.0, 0.0],
            [np.pi / 2.0, np.pi / 2.0],
        ],
        dtype=float,
    )
    phi = np.array(
        [
            [0.0, np.pi / 2.0],
            [0.0, np.pi / 2.0],
        ],
        dtype=float,
    )

    horizontal, vertical = _balloon_angle_arrays(theta, phi)

    np.testing.assert_allclose(horizontal, [0.0, 90.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(vertical, [0.0, 0.0, 0.0, 90.0], atol=1e-5)


def test_balloon_hover_text_formats_angles_and_spl() -> None:
    mesh = {
        HORIZONTAL_ANGLE_SCALAR_NAME: np.array([-15.25], dtype=np.float32),
        VERTICAL_ANGLE_SCALAR_NAME: np.array([7.75], dtype=np.float32),
        SPL_SCALAR_NAME: np.array([-12.125], dtype=np.float32),
    }

    text = _balloon_hover_text(mesh, 0)

    assert text == " | ".join(
        (
            "Horizontal Angle: -15.2 deg",
            "Vertical Angle: +7.8 deg",
            "Normalized SPL: -12.1 dB",
        )
    )


def test_balloon_contour_levels_skip_configured_maximum() -> None:
    assert _contour_levels(-30.0, 0.0, 6.0) == [-24.0, -18.0, -12.0, -6.0]
