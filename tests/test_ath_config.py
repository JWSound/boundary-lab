import pytest

from blab.ath_config import native_check_open_edges_for_ath_config


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


def test_mesh_mode_inside_polar_block_is_ignored() -> None:
    config_text = "\n".join(
        (
            "ABEC.Polars:SPL_H = {",
            "  Mode = bare",
            "}",
            "",
        )
    )
    assert native_check_open_edges_for_ath_config(config_text) is True


def test_comments_do_not_hide_the_mesh_mode() -> None:
    assert native_check_open_edges_for_ath_config("Mesh.Mode = bare ; open-mouth horn\n") is False
