import pytest

from blab.driver_text import format_driver_text, parse_driver_text

EXAMPLE = """Example Driver

Sd=522.0
Bl=19.9
Cms=136.0E-06
Rms=4.55
Mmd=62.4
Le=1.3
Re=5.0
Pmax=1400
Xmax=8.0
Re'=0.00
Leb=0.00
Le=0.00
Ke=0.00
Rss=0.00
Rms=0.00
Ams=0.00
LossyLe=0
SemiInductance=0
Damping=0
"""


def test_example_units_and_disabled_placeholders():
    driver = parse_driver_text("\ufeff" + EXAMPLE.replace("\n", "\r\n"))
    assert driver.sd_m2 == pytest.approx(0.0522)
    assert driver.parameters == pytest.approx(
        {
            "re_ohm": 5,
            "le_h": 0.0013,
            "bl_n_per_a": 19.9,
            "mmd_kg": 0.0624,
            "cms_m_per_n": 0.000136,
            "rms_n_s_per_m": 4.55,
        }
    )
    text = format_driver_text(driver)
    assert "Pmax" not in text and "Xmax" not in text
    restored = parse_driver_text(text)
    assert restored.parameters == pytest.approx(driver.parameters)
    assert restored.sd_m2 == pytest.approx(driver.sd_m2)


def test_separate_semi_inductance_values_round_trip():
    text = EXAMPLE.replace("Re'=0.00", "Re'=6.2").replace("Leb=0.00", "Leb=0.1")
    text = text.replace("Le=0.00", "Le=2.5").replace("Ke=0.00", "Ke=0.04")
    text = text.replace("Rss=0.00", "Rss=1000").replace("SemiInductance=0", "SemiInductance=1")
    driver = parse_driver_text(text)
    assert driver.parameters["le_h"] == pytest.approx(0.0013)
    assert driver.parameters["rms_n_s_per_m"] == pytest.approx(4.55)
    assert driver.parameters["semi_inductance"] == {
        "enabled": True,
        "re_prime_ohm": 6.2,
        "leb_h": 0.0001,
        "le_h": 0.0025,
        "ke_semi_h": 0.04,
        "rss_ohm": 1000,
    }
    assert parse_driver_text(format_driver_text(driver)) == driver


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("Mmd=62.4", "Mms=62.4", "Mms is not Mmd"),
        ("Mmd=62.4", "", "Missing"),
        ("Le=1.3", "Le=1.3\nLe=2", "Duplicate"),
        ("Re=5.0", "Re=nan", "finite"),
        ("Re=5.0", "Re=-1", "nonnegative"),
        ("Re=5.0", "Re=0", "greater than zero"),
        ("Cms=136.0E-06", "Cms=0,000136", "decimal point"),
        ("Damping=0", "Damping=1", "not supported"),
        ("LossyLe=0", "LossyLe=1", "not supported"),
        ("SemiInductance=0", "SemiInductance=1", "Semi-inductance"),
        ("SemiInductance=0", "", "requires"),
        ("Damping=0", "Damping=2", "0 or 1"),
    ],
)
def test_invalid_file_is_rejected(old, new, error):
    with pytest.raises(ValueError, match=error):
        parse_driver_text(EXAMPLE.replace(old, new))


def test_basic_file_without_optional_sections():
    text = EXAMPLE.split("Pmax")[0].replace("Sd=522.0\n", "")
    driver = parse_driver_text(text)
    assert driver.sd_m2 is None
    assert parse_driver_text(format_driver_text(driver)) == driver
