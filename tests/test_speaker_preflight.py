from __future__ import annotations

from pathlib import Path

from blab.headless import load_headless_project
from blab.speaker_preflight import estimate_level_three_package


def test_s218bp_level_three_preflight_tracks_full_domain_storage() -> None:
    project_path = Path(__file__).parents[1] / "examples" / "S218BP" / "S218.blab.json"
    project = load_headless_project(project_path)

    estimate = estimate_level_three_package(
        project.physical_system,
        symmetry=project.symmetry,
        frequency_count=100,
        complex_bytes=8,
        rom_rank=256,
        sphere_point_count=6600,
    )

    assert estimate.source_symmetry == "xy"
    assert estimate.symmetry_image_count == 4
    assert estimate.bem_node_count == 4156
    assert estimate.bem_face_count == 8308
    assert estimate.retained_fem_node_count == 3127
    assert estimate.interface_flux_count == 1536
    assert estimate.transducer_count == 2
    assert estimate.excitation_count == 2
    assert estimate.state_count == 4667
    assert estimate.exact_package_bytes_estimate < estimate.dense_sampled_package_bytes_estimate // 1000
    assert estimate.eight_cabinet_shared_schur_bytes == estimate.current_frequency_schur_bytes
    assert estimate.eight_cabinet_independent_schur_bytes == 8 * estimate.current_frequency_schur_bytes

