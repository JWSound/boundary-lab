import pytest

from blab.ath_config import native_check_open_edges_for_ath_config, parse_ath_solve_settings


def test_parse_ath_solve_settings_reads_abec_frequency_and_polar_metadata() -> None:
    settings = parse_ath_solve_settings(
        """
        ; Ath version V2025-06
        ABEC.Abscissa = 1
        ABEC.MeshFrequency = 1000
        ABEC.NumFrequencies = 40
        ABEC.Polars:SPL_H = {
        Distance = 2
        MapAngleRange = 0,180,37
        NormAngle = 0
        }
        ABEC.Polars:SPL_V = {
        Distance = 2
        Inclination = 90
        MapAngleRange = 0,180,37
        NormAngle = 5
        }
        ABEC.SimType = 2
        ABEC.f1 = 100 ; [Hz]
        ABEC.f2 = 20000
        """
    )

    assert settings.freq_min_hz == 100.0
    assert settings.freq_max_hz == 20000.0
    assert settings.freq_count == 40
    assert settings.polar_min_angle_deg == 0.0
    assert settings.polar_max_angle_deg == 180.0
    assert settings.polar_step_deg == pytest.approx(5.0)
    assert settings.polar_distance_m == 2.0
    assert settings.horizontal_norm_angle_deg == 0.0
    assert settings.vertical_norm_angle_deg == 5.0


def test_parse_ath_solve_settings_accepts_named_horizontal_polar_aliases() -> None:
    settings = parse_ath_solve_settings(
        """
        ABEC.NumFrequencies = 96
        ABEC.Polars:hor = {
        MapAngleRange = -180,180,96
        NormAngle = 20
        }
        ABEC.f1 = 500
        ABEC.f2 = 16000
        """
    )

    assert settings.freq_count == 96
    assert settings.polar_min_angle_deg == -180.0
    assert settings.polar_max_angle_deg == 180.0
    assert settings.polar_step_deg == pytest.approx(360.0 / 95.0)
    assert settings.horizontal_norm_angle_deg == 20.0
    assert settings.vertical_norm_angle_deg == 20.0


@pytest.mark.parametrize(
    ("config_text", "expected"),
    (
        ("Mesh.Mode = bare\n", False),
        ("mode = open\n", False),
        ("mode = freestanding\n", True),
        ("ABEC.SimType = 1\n", True),
        ("Enclosure.Depth = 120\n", True),
        ("", True),
    ),
)
def test_native_open_edge_check_only_opts_out_for_bare_modes(config_text: str, expected: bool) -> None:
    assert native_check_open_edges_for_ath_config(config_text) is expected
