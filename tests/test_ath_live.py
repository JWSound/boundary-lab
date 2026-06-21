from pathlib import Path
import json

import meshio
import numpy as np

from blab.ath import (
    AthProcessRunner,
    ath_mirror_axes_for_result,
    ath_mirror_axes_from_solving_file,
    clean_ath_mesh_output,
    clean_ath_reduced_mesh_output,
    detect_ath_radiators,
    discover_ath_output,
    find_physical_tag_by_name,
    read_ath_output_root,
    read_surface_physical_names,
    write_ath_gmsh_path,
    write_ath_output_root,
)
from blab.live import (
    FrequencyResult,
    LiveSolveDataset,
    build_log_frequencies,
    order_frequencies_for_live_plotting,
    split_frequency_order_for_workers,
)
from blab.balloon import BalloonPrepConfig, _grid_spl_surface, prepare_balloon_data
from blab.config import ChannelConfig
from blab.exporting import export_balloon_data, export_polar_text_files
from blab.mesh_clean import triangle_quality_warning
from blab.postprocess import PrepConfig


def _write_minimal_msh(path: Path) -> None:
    path.write_text(
        """
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
2 1 "Rigid"
2 2 "SD1D1001"
$EndPhysicalNames
""".strip(),
        encoding="utf-8",
    )


class _FakeAthProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.pid = 12345
        self.terminated = False
        self.killed = False

    def communicate(self, timeout=None):
        return self.stdout, self.stderr

    def poll(self):
        return None if not self.terminated and not self.killed else self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_ath_process_runner_discovers_output_after_process_exit(tmp_path: Path, monkeypatch) -> None:
    ath_dir = tmp_path / "ath"
    ath_dir.mkdir()
    ath_exe = ath_dir / "ath.exe"
    ath_exe.write_text("", encoding="utf-8")
    output_root = tmp_path / "output"
    (ath_dir / "ath.cfg").write_text(f'OutputRootDir = "{output_root}"\n', encoding="utf-8")
    mesh_dir = output_root / "case" / "ABEC_FreeStanding"
    mesh_dir.mkdir(parents=True)
    _write_minimal_msh(mesh_dir / "case.msh")

    popen_calls = []

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        return _FakeAthProcess()

    monkeypatch.setattr("blab.ath.subprocess.Popen", fake_popen)

    result = AthProcessRunner().run(
        ath_exe=ath_exe,
        config_text="Length = 10",
        run_root=tmp_path / "runs",
        case_name="case",
    )

    assert result.msh_path == mesh_dir / "case.msh"
    assert (tmp_path / "runs" / "case.cfg").read_text(encoding="utf-8") == "Length = 10"
    assert popen_calls[0][0][0] == [str(ath_exe.resolve()), str(tmp_path / "runs" / "case.cfg")]


def test_ath_process_runner_stop_terminates_active_process(monkeypatch) -> None:
    runner = AthProcessRunner()
    process = _FakeAthProcess()
    runner._process = process
    monkeypatch.setattr("blab.ath.os.name", "posix")

    runner.stop()

    assert runner.cancel_requested
    assert process.terminated


def test_find_physical_tag_by_name_reads_ath_driven_group(tmp_path: Path) -> None:
    msh_path = tmp_path / "waveguide.msh"
    _write_minimal_msh(msh_path)

    assert find_physical_tag_by_name(msh_path, "SD1D1001") == 2


def test_read_surface_physical_names_ignores_non_surface_groups(tmp_path: Path) -> None:
    msh_path = tmp_path / "waveguide.msh"
    msh_path.write_text(
        """
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
3
1 10 "Curve"
2 2 "SD1D1001"
3 30 "Volume"
$EndPhysicalNames
""".strip(),
        encoding="utf-8",
    )

    assert read_surface_physical_names(msh_path) == {"SD1D1001": 2}


def test_discover_ath_output_finds_msh_and_driven_tag(tmp_path: Path) -> None:
    output_dir = tmp_path / "case"
    mesh_dir = output_dir / "ABEC_FreeStanding"
    mesh_dir.mkdir(parents=True)
    _write_minimal_msh(mesh_dir / "case.msh")

    result = discover_ath_output(run_root=tmp_path, case_name="case", config_path=tmp_path / "case.cfg")

    assert result.msh_path == mesh_dir / "case.msh"
    assert result.driven_tag == 2
    assert [(r.name, r.tag, r.level_db) for r in result.radiators] == [("throat", 2, 0.0)]


def test_detect_ath_radiators_uses_weighted_complex_dome_groups(tmp_path: Path) -> None:
    msh_path = tmp_path / "complex.msh"
    msh_path.write_text(
        """
$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
4
2 1 "SD1G0"
2 2 "SD1D1001"
2 3 "SD1D1002"
2 4 "SD1D1003"
$EndPhysicalNames
""".strip(),
        encoding="utf-8",
    )

    radiators = detect_ath_radiators(msh_path)

    assert [(r.name, r.tag, r.level_db) for r in radiators] == [
        ("dome", 4, 0.0),
        ("surround_inner", 3, -2.5),
        ("surround_outer", 2, -12.0),
    ]


def test_read_ath_output_root_reads_companion_config(tmp_path: Path) -> None:
    ath_cfg = tmp_path / "ath.cfg"
    ath_cfg.write_text(
        'OutputRootDir = "E:\\AthGUI"\nMeshCmd = "C:\\gmsh\\gmsh.exe %f -"\n',
        encoding="utf-8",
    )

    assert read_ath_output_root(ath_cfg) == Path("E:\\AthGUI")


def test_write_ath_output_root_updates_companion_config(tmp_path: Path) -> None:
    ath_cfg = tmp_path / "ath.cfg"
    ath_cfg.write_text(
        'OutputRootDir = "E:\\old"\nMeshCmd = "C:\\gmsh\\gmsh.exe %f -"\n',
        encoding="utf-8",
    )
    output_root = tmp_path / "runs" / "ath_output"

    written_root = write_ath_output_root(ath_cfg, output_root)

    assert written_root == output_root.resolve()
    assert read_ath_output_root(ath_cfg) == output_root.resolve()
    assert 'MeshCmd = "C:\\gmsh\\gmsh.exe %f -"' in ath_cfg.read_text(encoding="utf-8")


def test_write_ath_gmsh_path_updates_mesh_command(tmp_path: Path) -> None:
    ath_cfg = tmp_path / "ath.cfg"
    ath_cfg.write_text(
        'OutputRootDir = "E:\\old"\nMeshCmd = "C:\\gmsh\\gmsh.exe %f -"\nGnuplotPath = "C:\\gnuplot"\n',
        encoding="utf-8",
    )
    gmsh_exe = tmp_path / "gmsh" / "gmsh.exe"
    gmsh_exe.parent.mkdir()
    gmsh_exe.write_text("", encoding="utf-8")

    written_gmsh = write_ath_gmsh_path(ath_cfg, gmsh_exe)
    cfg_text = ath_cfg.read_text(encoding="utf-8")

    assert written_gmsh == gmsh_exe.resolve()
    assert f'MeshCmd = "{gmsh_exe.resolve()} %f -"' in cfg_text
    assert 'OutputRootDir = "E:\\old"' in cfg_text
    assert 'GnuplotPath = "C:\\gnuplot"' in cfg_text


def test_write_ath_gmsh_path_inserts_mesh_command_when_missing(tmp_path: Path) -> None:
    ath_cfg = tmp_path / "ath.cfg"
    ath_cfg.write_text('OutputRootDir = "E:\\old"', encoding="utf-8")
    gmsh_exe = tmp_path / "gmsh.exe"

    write_ath_gmsh_path(ath_cfg, gmsh_exe)

    assert ath_cfg.read_text(encoding="utf-8").splitlines() == [
        f'MeshCmd = "{gmsh_exe.resolve()} %f -"',
        'OutputRootDir = "E:\\old"',
    ]


def test_ath_mirror_axes_from_solving_symmetry_line(tmp_path: Path) -> None:
    solving_path = tmp_path / "solving.txt"
    solving_path.write_text(
        "Control_Solver\n  Abscissa=log; Dim=3D; MeshFrequency=1000; Sym=xy\n",
        encoding="utf-8",
    )

    assert ath_mirror_axes_from_solving_file(solving_path) == ("x", "y")


def test_ath_mirror_axes_are_empty_without_symmetry_line(tmp_path: Path) -> None:
    solving_path = tmp_path / "solving.txt"
    solving_path.write_text("Control_Solver\n  Abscissa=log; Dim=3D\n", encoding="utf-8")

    assert ath_mirror_axes_from_solving_file(solving_path) == ()


def test_clean_ath_mesh_output_writes_cleaned_solver_mesh(tmp_path: Path) -> None:
    output_dir = tmp_path / "case"
    mesh_dir = output_dir / "ABEC_FreeStanding"
    mesh_dir.mkdir(parents=True)

    raw_msh = mesh_dir / "case.msh"
    mesh = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([2], dtype=np.int32)]},
        field_data={"SD1D1001": np.array([2, 2], dtype=np.int32)},
    )
    meshio.write(raw_msh, mesh, file_format="gmsh22", binary=False)

    result = discover_ath_output(run_root=tmp_path, case_name="case", config_path=tmp_path / "case.cfg")
    cleaned = clean_ath_mesh_output(result)

    assert cleaned.msh_path == raw_msh
    assert cleaned.cleaned_msh_path == mesh_dir / "case_clean.msh"
    assert cleaned.solver_msh_path == cleaned.cleaned_msh_path
    assert cleaned.solver_msh_path.exists()
    assert find_physical_tag_by_name(cleaned.solver_msh_path, "SD1D1001") == 2
    assert [(r.name, r.tag, r.level_db) for r in cleaned.radiators] == [("throat", 2, 0.0)]


def test_clean_ath_mesh_output_uses_solving_symmetry_axes(tmp_path: Path) -> None:
    output_dir = tmp_path / "case"
    mesh_dir = output_dir / "ABEC_InfiniteBaffle"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "solving.txt").write_text(
        "Control_Solver\n  Abscissa=log; Dim=3D; MeshFrequency=1000; Sym=xy\n",
        encoding="utf-8",
    )

    raw_msh = mesh_dir / "case.msh"
    mesh = meshio.Mesh(
        points=np.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([2], dtype=np.int32)]},
        field_data={"SD1D1001": np.array([2, 2], dtype=np.int32)},
    )
    meshio.write(raw_msh, mesh, file_format="gmsh22", binary=False)

    result = discover_ath_output(run_root=tmp_path, case_name="case", config_path=tmp_path / "case.cfg")
    cleaned = clean_ath_mesh_output(result)
    cleaned_mesh = meshio.read(cleaned.solver_msh_path)

    assert ath_mirror_axes_for_result(result) == ("x", "y")
    assert cleaned_mesh.cells_dict["triangle"].shape[0] == 4


def test_clean_ath_reduced_mesh_output_keeps_fundamental_domain(tmp_path: Path) -> None:
    output_dir = tmp_path / "case"
    mesh_dir = output_dir / "ABEC_InfiniteBaffle"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "solving.txt").write_text(
        "Control_Solver\n  Abscissa=log; Dim=3D; MeshFrequency=1000; Sym=xy\n",
        encoding="utf-8",
    )

    raw_msh = mesh_dir / "case.msh"
    mesh = meshio.Mesh(
        points=np.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([2], dtype=np.int32)]},
        field_data={"SD1D1001": np.array([2, 2], dtype=np.int32)},
    )
    meshio.write(raw_msh, mesh, file_format="gmsh22", binary=False)

    result = discover_ath_output(run_root=tmp_path, case_name="case", config_path=tmp_path / "case.cfg")
    expanded = clean_ath_mesh_output(result)
    reduced = clean_ath_reduced_mesh_output(expanded)

    expanded_mesh = meshio.read(expanded.solver_msh_path_for_symmetry("off"))
    reduced_mesh = meshio.read(reduced.solver_msh_path_for_symmetry("xy"))

    assert expanded_mesh.cells_dict["triangle"].shape[0] == 4
    assert reduced_mesh.cells_dict["triangle"].shape[0] == 1
    assert reduced.cleaned_msh_path == expanded.cleaned_msh_path
    assert reduced.reduced_cleaned_msh_path == mesh_dir / "case_clean_reduced.msh"


def test_triangle_quality_warning_detects_float32_singular_sliver() -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [
                [0.13800001, 0.095886745, 0.0368],
                [0.13800001, 0.12680551, 0.0368],
                [0.137996, 0.11434301, 0.0368],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        cells=[("triangle", np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64))],
    )

    warning = triangle_quality_warning(mesh)

    assert warning.has_warnings
    assert warning.sliver_triangles == 1
    assert warning.float32_singular_triangles == 1
    assert warning.worst_triangle_index == 1
    assert warning.worst_altitude_edge_ratio < 1e-3


def test_clean_ath_mesh_output_reports_sliver_warning(tmp_path: Path) -> None:
    output_dir = tmp_path / "case"
    mesh_dir = output_dir / "ABEC_FreeStanding"
    mesh_dir.mkdir(parents=True)

    raw_msh = mesh_dir / "case.msh"
    mesh = meshio.Mesh(
        points=np.array(
            [
                [0.13800001, 0.095886745, 0.0368],
                [0.13800001, 0.12680551, 0.0368],
                [0.137996, 0.11434301, 0.0368],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        cells=[("triangle", np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([1, 2], dtype=np.int32)]},
        field_data={"Rigid": np.array([1, 2], dtype=np.int32), "SD1D1001": np.array([2, 2], dtype=np.int32)},
    )
    meshio.write(raw_msh, mesh, file_format="gmsh22", binary=False)

    result = discover_ath_output(run_root=tmp_path, case_name="case", config_path=tmp_path / "case.cfg")
    cleaned = clean_ath_mesh_output(result)

    assert cleaned.quality_warning is not None
    assert cleaned.quality_warning.has_warnings
    assert cleaned.quality_warning.float32_singular_triangles == 1


def test_live_frequency_order_starts_with_limits_and_preserves_all_points() -> None:
    freqs = build_log_frequencies(200.0, 20000.0, 24)
    ordered = order_frequencies_for_live_plotting(freqs)

    assert ordered[0] == freqs[0]
    assert ordered[1] == freqs[-1]
    assert len(np.unique(ordered)) == len(freqs)
    assert set(np.round(ordered, 5)) == set(np.round(freqs, 5))


def test_live_frequency_order_uses_van_der_corput_interior_indices() -> None:
    freqs = np.arange(9, dtype=np.float32)
    ordered = order_frequencies_for_live_plotting(freqs)

    assert ordered.tolist() == [0.0, 8.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0]


def test_split_frequency_order_for_workers_round_robins_ordered_frequencies() -> None:
    freqs = np.arange(10, dtype=np.float32)
    chunks = split_frequency_order_for_workers(freqs, worker_count=3)

    assert [chunk.tolist() for chunk in chunks] == [
        [0.0, 3.0, 6.0, 9.0],
        [1.0, 4.0, 7.0],
        [2.0, 5.0, 8.0],
    ]
    assert np.concatenate(chunks).size == freqs.size


def test_live_dataset_builds_visualization_dataset_from_results() -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    dataset = LiveSolveDataset(angles, radiator_names=np.array(["throat"]))
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.array([-6.0, 0.0, -6.0]),
            vertical_spl_norm_db=np.array([-8.0, 0.0, -8.0]),
            impedance=np.array([[1.0, 0.2]], dtype=np.float32),
        )
    )
    dataset.add(
        FrequencyResult(
            freq_hz=200.0,
            horizontal_spl_norm_db=np.array([-3.0, 0.0, -3.0]),
            vertical_spl_norm_db=np.array([-4.0, 0.0, -4.0]),
            impedance=np.array([[0.5, 0.1]], dtype=np.float32),
        )
    )

    prepared = dataset.as_visualization_dataset(
        PrepConfig(angle_samples=None, freq_samples=None, octave_smoothing=None)
    )

    assert prepared is not None
    assert prepared["freq_hz"].tolist() == [200.0, 1000.0]
    assert prepared["horizontal_spl_norm_db"].shape == (2, 3)
    assert prepared["impedance_real"].tolist() == [[0.5, 1.0]]


def test_live_dataset_resynthesizes_channel_basis_after_gain_change() -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    dataset = LiveSolveDataset(
        angles,
        radiator_names=np.array(["lf", "hf"]),
        channel_configs=(ChannelConfig(name="LF"), ChannelConfig(name="HF")),
        flat_target_normalization_enabled=False,
    )
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.zeros(3, dtype=np.float32),
            vertical_spl_norm_db=np.zeros(3, dtype=np.float32),
            impedance=np.array([[1.0, 0.2], [2.0, 0.4]], dtype=np.float32),
            channel_names=np.array(["LF", "HF"]),
            horizontal_pressure=np.array(
                [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]],
                dtype=np.complex64,
            ),
            vertical_pressure=np.array(
                [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]],
                dtype=np.complex64,
            ),
        )
    )

    _, _, raw_before, _ = dataset.as_raw_polar_arrays()
    dataset.set_channel_synthesis((ChannelConfig(name="LF"), ChannelConfig(name="HF", level_db=-6.0)))
    _, _, raw_after, _ = dataset.as_raw_polar_arrays()

    assert dataset.supports_channel_resynthesis
    assert raw_after[0, 1] < raw_before[0, 1]
    assert dataset.solved_count == 1


def test_live_dataset_exposes_channel_on_axis_curves() -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    dataset = LiveSolveDataset(
        angles,
        radiator_names=np.array(["lf", "hf"]),
        channel_configs=(ChannelConfig(name="LF"), ChannelConfig(name="HF", level_db=-6.0)),
        flat_target_normalization_enabled=False,
    )
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.zeros(3, dtype=np.float32),
            vertical_spl_norm_db=np.zeros(3, dtype=np.float32),
            impedance=np.array([[1.0, 0.2], [2.0, 0.4]], dtype=np.float32),
            channel_names=np.array(["LF", "HF"]),
            horizontal_pressure=np.array(
                [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]],
                dtype=np.complex64,
            ),
            vertical_pressure=np.array(
                [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]],
                dtype=np.complex64,
            ),
        )
    )

    prepared = dataset.as_visualization_dataset(
        PrepConfig(angle_samples=None, freq_samples=None, octave_smoothing=None)
    )

    assert prepared["channel_on_axis_names"].tolist() == ["LF", "HF"]
    assert prepared["channel_on_axis_spl_db"].shape == (2, 1)
    assert prepared["channel_on_axis_spl_db"][1, 0] < prepared["channel_on_axis_spl_db"][0, 0]


def test_live_dataset_builds_balloon_bundle_from_sphere_results() -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    theta = np.linspace(0.1, np.pi - 0.1, 8, dtype=np.float32)
    phi = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
    dataset = LiveSolveDataset(
        angles,
        radiator_names=np.array(["throat"]),
        sphere_r_distance_m=np.full(8, 2.0, dtype=np.float32),
        sphere_theta_polar_rad=theta,
        sphere_phi_azimuth_rad=phi,
    )
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.array([-6.0, 0.0, -6.0]),
            vertical_spl_norm_db=np.array([-8.0, 0.0, -8.0]),
            impedance=np.array([[1.0, 0.2]], dtype=np.float32),
            sphere_spl_norm_db=np.linspace(-12.0, 0.0, 8, dtype=np.float32),
        )
    )

    bundle = dataset.as_balloon_raw_bundle()

    assert bundle is not None
    assert bundle["freq_hz"].tolist() == [1000.0]
    assert bundle["spl_norm"].shape == (1, 8)


def test_live_dataset_balloon_bundle_does_not_reindex_float32_frequencies() -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    theta = np.linspace(0.1, np.pi - 0.1, 4, dtype=np.float32)
    phi = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False, dtype=np.float32)
    dataset = LiveSolveDataset(
        angles,
        radiator_names=np.array(["throat"]),
        sphere_r_distance_m=np.full(4, 2.0, dtype=np.float32),
        sphere_theta_polar_rad=theta,
        sphere_phi_azimuth_rad=phi,
    )
    dataset.add(
        FrequencyResult(
            freq_hz=241.35852050781247,
            horizontal_spl_norm_db=np.array([-6.0, 0.0, -6.0]),
            vertical_spl_norm_db=np.array([-8.0, 0.0, -8.0]),
            impedance=np.array([[1.0, 0.2]], dtype=np.float32),
            sphere_spl_norm_db=np.linspace(-12.0, 0.0, 4, dtype=np.float32),
        )
    )

    bundle = dataset.as_balloon_raw_bundle()

    assert bundle is not None
    assert bundle["spl_norm"].shape == (1, 4)


def test_prepare_balloon_data_builds_surface_arrays() -> None:
    theta_values = np.linspace(0.05, np.pi - 0.05, 8, dtype=np.float32)
    phi_values = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
    theta, phi = np.meshgrid(theta_values, phi_values, indexing="ij")
    spl = -12.0 + 12.0 * np.cos(theta.ravel()) ** 2
    raw = {
        "freq_hz": np.array([500.0], dtype=np.float32),
        "r_distance_m": np.full(theta.size, 2.0, dtype=np.float32),
        "theta_polar_rad": theta.ravel().astype(np.float32),
        "phi_azimuth_rad": phi.ravel().astype(np.float32),
        "spl_norm": spl[np.newaxis, :].astype(np.float32),
    }

    prepared = prepare_balloon_data(raw, BalloonPrepConfig(theta_samples=10, phi_samples=12))

    assert prepared["balloon_x"].shape == (1, 10, 12)
    assert prepared["balloon_surface_spl"].shape == (1, 10, 12)
    assert float(prepared["balloon_surface_spl"].max()) <= 0.0


def test_prepare_balloon_data_matches_griddata_surface_interpolation() -> None:
    theta, phi = np.meshgrid(
        np.linspace(0.0, np.pi, 6, dtype=np.float32),
        np.linspace(0.0, 2.0 * np.pi, 8, dtype=np.float32),
        indexing="ij",
    )
    spl = (
        -12.0
        + 3.0 * np.cos(theta.ravel())
        + 2.0 * np.sin(phi.ravel())
    ).astype(np.float32)
    raw = {
        "freq_hz": np.array([1000.0], dtype=np.float32),
        "theta_polar_rad": theta.ravel().astype(np.float32),
        "phi_azimuth_rad": phi.ravel().astype(np.float32),
        "spl_norm": spl[np.newaxis, :],
    }

    prepared = prepare_balloon_data(raw, BalloonPrepConfig(theta_samples=9, phi_samples=11, min_db=-30.0))
    expected = _grid_spl_surface(
        raw["theta_polar_rad"],
        raw["phi_azimuth_rad"],
        spl,
        prepared["theta_grid_rad"],
        prepared["phi_grid_rad"],
        -30.0,
    )

    np.testing.assert_allclose(prepared["balloon_surface_spl"][0], expected, atol=1e-5)


def test_export_balloon_data_writes_fixed_topology_artifact(tmp_path: Path) -> None:
    theta, phi = np.meshgrid(
        np.linspace(0.0, np.pi, 5, dtype=np.float32),
        np.linspace(0.0, 2.0 * np.pi, 7, dtype=np.float32),
        indexing="ij",
    )
    spl = np.stack(
        [
            -30.0 + 30.0 * np.cos(theta.ravel()) ** 2,
            -24.0 + 24.0 * np.sin(theta.ravel()) ** 2,
        ],
        axis=0,
    ).astype(np.float32)
    raw = {
        "freq_hz": np.array([500.0, 1000.0], dtype=np.float32),
        "theta_polar_rad": theta.ravel().astype(np.float32),
        "phi_azimuth_rad": phi.ravel().astype(np.float32),
        "spl_norm": spl,
    }
    prepared = prepare_balloon_data(
        raw,
        BalloonPrepConfig(theta_samples=5, phi_samples=7, min_db=-30.0, max_db=0.0),
    )

    result = export_balloon_data(prepared, tmp_path)

    assert result.frequency_count == 2
    assert result.point_count == 35
    assert result.quad_count == 24
    assert {path.name for path in result.files} == {
        "metadata.json",
        "topology.npz",
        "spl_db.npy",
        "radius_norm.npy",
        "positions_xyz.npy",
    }

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["point_order"].startswith("row-major")
    assert metadata["array_shapes"]["positions_xyz"] == ["frequency", "theta", "phi", "xyz"]

    with np.load(tmp_path / "topology.npz") as topology:
        assert topology["directions_xyz"].shape == (35, 3)
        assert topology["quad_indices"].shape == (24, 4)
        np.testing.assert_array_equal(topology["quad_indices"][0], [0, 1, 8, 7])
        np.testing.assert_allclose(topology["freq_hz"], [500.0, 1000.0])

    spl_db = np.load(tmp_path / "spl_db.npy")
    radius_norm = np.load(tmp_path / "radius_norm.npy")
    positions = np.load(tmp_path / "positions_xyz.npy")
    assert spl_db.shape == (2, 5, 7)
    assert radius_norm.shape == (2, 5, 7)
    assert positions.shape == (2, 5, 7, 3)
    np.testing.assert_allclose(radius_norm, np.clip((spl_db + 30.0) / 30.0, 0.0, 1.0), atol=1e-6)


def test_export_polar_text_files_writes_one_file_per_plane_angle(tmp_path: Path) -> None:
    angles = np.array([-10.0, 0.0, 10.5], dtype=np.float32)
    dataset = LiveSolveDataset(angles, radiator_names=np.array(["throat"]))
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.array([-6.0, 0.0, -3.25]),
            vertical_spl_norm_db=np.array([-8.0, -1.0, -4.5]),
            impedance=np.array([[1.0, 0.2]], dtype=np.float32),
        )
    )
    dataset.add(
        FrequencyResult(
            freq_hz=200.0,
            horizontal_spl_norm_db=np.array([-3.0, 0.0, -2.25]),
            vertical_spl_norm_db=np.array([-4.0, -0.5, -3.5]),
            impedance=np.array([[0.5, 0.1]], dtype=np.float32),
        )
    )

    written = export_polar_text_files(dataset, tmp_path, include_phase=False)

    assert len(written) == 6
    assert (tmp_path / "H 0.txt").read_text(encoding="utf-8").splitlines() == [
        "200.000000\t0.000",
        "1000.000000\t0.000",
    ]
    assert (tmp_path / "V 10.5.txt").read_text(encoding="utf-8").splitlines() == [
        "200.000000\t-3.500",
        "1000.000000\t-4.500",
    ]


def test_export_polar_text_files_writes_relative_phase_for_channel_basis(tmp_path: Path) -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    dataset = LiveSolveDataset(
        angles,
        radiator_names=np.array(["throat"]),
        channel_configs=(ChannelConfig(name="main"),),
        flat_target_normalization_enabled=False,
    )
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.zeros(3, dtype=np.float32),
            vertical_spl_norm_db=np.zeros(3, dtype=np.float32),
            impedance=np.array([[1.0, 0.2]], dtype=np.float32),
            channel_names=np.array(["main"]),
            horizontal_pressure=np.array([[1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 1.0j]], dtype=np.complex64),
            vertical_pressure=np.array([[1.0 + 0.0j, 1.0 + 0.0j, 0.0 - 1.0j]], dtype=np.complex64),
        )
    )

    written = export_polar_text_files(dataset, tmp_path)

    assert len(written) == 6
    assert (tmp_path / "H 0.txt").read_text(encoding="utf-8").splitlines() == [
        "1000.000000\t0.000\t0.000",
    ]
    assert (tmp_path / "H 90.txt").read_text(encoding="utf-8").splitlines() == [
        "1000.000000\t0.000\t90.000",
    ]
    assert (tmp_path / "V 90.txt").read_text(encoding="utf-8").splitlines() == [
        "1000.000000\t0.000\t-90.000",
    ]
