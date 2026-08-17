import json
from pathlib import Path

import meshio
import numpy as np
import pytest

import blab.live as live_module
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
from blab.balloon import BalloonPrepConfig, BalloonSurfaceSampler, prepare_balloon_data
from blab.config import ChannelConfig
from blab.exporting import export_balloon_data, export_on_axis_text_files, export_polar_text_files
from blab.live import (
    FrequencyResult,
    LiveSolveDataset,
    build_log_frequencies,
    order_frequencies_for_live_plotting,
    split_frequency_order_for_workers,
)
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


@pytest.mark.parametrize("platform_name", ("linux", "darwin"))
def test_ath_process_runner_uses_wine_for_ath_exe_on_linux_or_macos(
    tmp_path: Path, monkeypatch, platform_name: str
) -> None:
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

    monkeypatch.setattr("blab.ath.sys.platform", platform_name)
    monkeypatch.setattr("blab.ath.shutil.which", lambda name: "/usr/bin/wine" if name == "wine" else None)
    monkeypatch.setattr("blab.ath.subprocess.Popen", fake_popen)

    AthProcessRunner().run(
        ath_exe=ath_exe,
        config_text="Length = 10",
        run_root=tmp_path / "runs",
        case_name="case",
    )

    assert popen_calls[0][0][0] == [
        "/usr/bin/wine",
        str(ath_exe.resolve()),
        str(tmp_path / "runs" / "case.cfg"),
    ]


def test_ath_process_runner_reports_missing_wine_on_linux(tmp_path: Path, monkeypatch) -> None:
    ath_dir = tmp_path / "ath"
    ath_dir.mkdir()
    ath_exe = ath_dir / "ath.exe"
    ath_exe.write_text("", encoding="utf-8")
    (ath_dir / "ath.cfg").write_text('OutputRootDir = "output"\n', encoding="utf-8")

    monkeypatch.setattr("blab.ath.sys.platform", "linux")
    monkeypatch.setattr("blab.ath.shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="requires Wine"):
        AthProcessRunner().run(
            ath_exe=ath_exe,
            config_text="Length = 10",
            run_root=tmp_path / "runs",
            case_name="case",
        )


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
    assert warning.worst_altitude_edge_ratio < 2e-3


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
                [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0j, 1.0j, 1.0j]],
                dtype=np.complex64,
            ),
            vertical_pressure=np.array(
                [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0j, 1.0j, 1.0j]],
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


def test_live_dataset_preserves_complex_pressure_directivity_for_isobar_and_balloon() -> None:
    angles = np.array([-180.0, -90.0, 0.0, 90.0, 180.0], dtype=np.float32)
    sphere_points = 5
    pressure = np.array([[0.01j, 0.1j, 1.0j, 0.1j, 0.01j]], dtype=np.complex64)
    dataset = LiveSolveDataset(
        angles,
        radiator_names=np.array(["driver"]),
        channel_configs=(ChannelConfig(name="main"),),
        flat_target_normalization_enabled=False,
        sphere_r_distance_m=np.full(sphere_points, 2.0, dtype=np.float32),
        sphere_theta_polar_rad=np.linspace(0.0, np.pi, sphere_points, dtype=np.float32),
        sphere_phi_azimuth_rad=np.zeros(sphere_points, dtype=np.float32),
    )
    dataset.add(
        FrequencyResult(
            freq_hz=8000.0,
            # Deliberately inconsistent legacy arrays: complex pressure must
            # remain the source of truth when channel-basis data is present.
            horizontal_spl_norm_db=np.zeros(angles.size, dtype=np.float32),
            vertical_spl_norm_db=np.zeros(angles.size, dtype=np.float32),
            impedance=np.array([[1.0, 0.2]], dtype=np.float32),
            channel_names=np.array(["main"]),
            horizontal_pressure=pressure,
            vertical_pressure=pressure,
            sphere_pressure=pressure,
        )
    )

    prepared = dataset.as_visualization_dataset(
        PrepConfig(
            angle_samples=None,
            freq_samples=None,
            octave_smoothing=None,
            hor_ref_angle=0.0,
            vert_ref_angle=0.0,
        )
    )
    balloon = dataset.as_balloon_raw_bundle()

    np.testing.assert_allclose(
        prepared["horizontal_spl_norm_db"][0],
        np.array([-30.0, -20.0, 0.0, -20.0, -30.0], dtype=np.float32),
        atol=1e-5,
    )
    assert balloon is not None
    np.testing.assert_allclose(
        balloon["spl_norm"][0],
        np.array([-40.0, -20.0, 0.0, -20.0, -40.0], dtype=np.float32),
        atol=1e-5,
    )


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
                [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0j, 1.0j, 1.0j]],
                dtype=np.complex64,
            ),
            vertical_pressure=np.array(
                [[1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j], [1.0j, 1.0j, 1.0j]],
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
    np.testing.assert_allclose(prepared["channel_on_axis_phase_deg"][:, 0], [0.0, 90.0])
    np.testing.assert_allclose(prepared["on_axis_phase_deg"], [26.5], atol=0.2)


def test_on_axis_phase_removes_propagation_delay_and_tracks_post_solve_channel_delay() -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    dataset = LiveSolveDataset(
        angles,
        channel_configs=(ChannelConfig(name="main"),),
        flat_target_normalization_enabled=False,
        polar_observation_distance_m=0.343,
        exterior_sound_speed_m_per_s=343.0,
    )
    pressure = np.full((1, 3), 1.0j, dtype=np.complex64)
    dataset.add(
        FrequencyResult(
            freq_hz=250.0,
            horizontal_spl_norm_db=np.zeros(3, dtype=np.float32),
            vertical_spl_norm_db=np.zeros(3, dtype=np.float32),
            impedance=np.array([[1.0, 0.0]], dtype=np.float32),
            channel_names=np.array(["main"]),
            horizontal_pressure=pressure,
            vertical_pressure=pressure.copy(),
        )
    )

    prepared = dataset.as_visualization_dataset(
        PrepConfig(angle_samples=None, freq_samples=None, octave_smoothing=None)
    )
    np.testing.assert_allclose(prepared["channel_on_axis_phase_deg"], [[0.0]], atol=1e-5)
    np.testing.assert_allclose(prepared["on_axis_phase_deg"], [0.0], atol=1e-5)

    dataset.set_channel_synthesis((ChannelConfig(name="main", delay_ms=0.5),))
    prepared = dataset.as_visualization_dataset(
        PrepConfig(angle_samples=None, freq_samples=None, octave_smoothing=None)
    )
    np.testing.assert_allclose(prepared["channel_on_axis_phase_deg"], [[-45.0]], atol=1e-4)
    np.testing.assert_allclose(prepared["on_axis_phase_deg"], [-45.0], atol=1e-4)


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


def test_visualization_skips_sphere_synthesis_and_balloon_bundle_synthesizes_once(monkeypatch) -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    sphere_points = 4
    dataset = LiveSolveDataset(
        angles,
        radiator_names=np.array(["driver"]),
        channel_configs=(ChannelConfig(name="main"),),
        flat_target_normalization_enabled=False,
        sphere_r_distance_m=np.full(sphere_points, 2.0, dtype=np.float32),
        sphere_theta_polar_rad=np.linspace(0.1, np.pi - 0.1, sphere_points, dtype=np.float32),
        sphere_phi_azimuth_rad=np.linspace(0.0, 2.0 * np.pi, sphere_points, endpoint=False, dtype=np.float32),
    )
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.zeros(angles.size, dtype=np.float32),
            vertical_spl_norm_db=np.zeros(angles.size, dtype=np.float32),
            impedance=np.array([[1.0, 0.2]], dtype=np.float32),
            channel_names=np.array(["main"]),
            horizontal_pressure=np.ones((1, angles.size), dtype=np.complex64),
            vertical_pressure=np.ones((1, angles.size), dtype=np.complex64),
            sphere_pressure=np.ones((1, sphere_points), dtype=np.complex64),
        )
    )
    original_synthesize = live_module.synthesize_channel_basis_spl
    sphere_arguments = []

    def record_synthesis(**kwargs):
        sphere_arguments.append(kwargs.get("sphere_pressure"))
        return original_synthesize(**kwargs)

    monkeypatch.setattr(live_module, "synthesize_channel_basis_spl", record_synthesis)

    assert dataset.has_balloon_data
    assert (
        dataset.as_visualization_dataset(PrepConfig(angle_samples=None, freq_samples=None, octave_smoothing=None))
        is not None
    )
    assert sphere_arguments == [None]

    bundle = dataset.as_balloon_raw_bundle()

    assert bundle is not None
    assert len(sphere_arguments) == 2
    assert sphere_arguments[1] is dataset.results[1000.0].sphere_pressure


def test_prepare_balloon_data_builds_surface_arrays() -> None:
    theta, phi = _fibonacci_angles(32)
    spl = -12.0 + 12.0 * np.cos(theta) ** 2
    raw = {
        "freq_hz": np.array([500.0], dtype=np.float32),
        "r_distance_m": np.full(theta.size, 2.0, dtype=np.float32),
        "theta_polar_rad": theta,
        "phi_azimuth_rad": phi,
        "spl_norm": spl[np.newaxis, :].astype(np.float32),
    }

    prepared = prepare_balloon_data(raw)

    assert prepared["directions_xyz"].shape == (32, 3)
    assert prepared["triangle_indices"].shape == (60, 3)
    assert prepared["balloon_surface_spl"].shape == (1, 32)
    assert float(prepared["balloon_surface_spl"].max()) <= 0.0
    edges = np.sort(
        prepared["triangle_indices"][:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(edge_counts == 2)


def test_balloon_surface_sampler_is_exact_at_solved_vertices() -> None:
    theta, phi = _fibonacci_angles(64)
    values = (-20.0 + np.arange(64, dtype=np.float32) / 8.0)[np.newaxis, :]
    prepared = prepare_balloon_data(
        {
            "freq_hz": np.array([1000.0], dtype=np.float32),
            "theta_polar_rad": theta,
            "phi_azimuth_rad": phi,
            "spl_norm": values,
        }
    )
    sampler = BalloonSurfaceSampler(prepared["directions_xyz"], prepared["triangle_indices"])

    sampled = sampler.interpolate(prepared["balloon_surface_spl"], prepared["directions_xyz"])

    np.testing.assert_allclose(sampled, prepared["balloon_surface_spl"], atol=1e-5)


def test_prepare_balloon_data_preserves_solved_vertex_values() -> None:
    theta, phi = _fibonacci_angles(48)
    spl = (-12.0 + 3.0 * np.cos(theta) + 2.0 * np.sin(phi)).astype(np.float32)
    raw = {
        "freq_hz": np.array([1000.0], dtype=np.float32),
        "theta_polar_rad": theta,
        "phi_azimuth_rad": phi,
        "spl_norm": spl[np.newaxis, :],
    }

    prepared = prepare_balloon_data(raw, BalloonPrepConfig(min_db=-30.0))

    np.testing.assert_array_equal(prepared["balloon_surface_spl"][0], np.clip(spl, -30.0, 0.0))


def test_export_balloon_data_writes_fixed_topology_artifact(tmp_path: Path) -> None:
    theta, phi = _fibonacci_angles(35)
    spl = np.stack(
        [
            -30.0 + 30.0 * np.cos(theta) ** 2,
            -24.0 + 24.0 * np.sin(theta) ** 2,
        ],
        axis=0,
    ).astype(np.float32)
    raw = {
        "freq_hz": np.array([500.0, 1000.0], dtype=np.float32),
        "theta_polar_rad": theta,
        "phi_azimuth_rad": phi,
        "spl_norm": spl,
    }
    prepared = prepare_balloon_data(
        raw,
        BalloonPrepConfig(min_db=-30.0, max_db=0.0),
    )

    result = export_balloon_data(prepared, tmp_path)

    assert result.frequency_count == 2
    assert result.point_count == 35
    assert result.triangle_count == 66
    assert {path.name for path in result.files} == {
        "metadata.json",
        "topology.npz",
        "spl_db.npy",
        "radius_norm.npy",
    }

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["point_order"] == "original solver Fibonacci sample order"
    assert metadata["array_shapes"]["directions_xyz"] == ["point", "xyz"]

    with np.load(tmp_path / "topology.npz") as topology:
        assert topology["directions_xyz"].shape == (35, 3)
        assert topology["triangle_indices"].shape == (66, 3)
        np.testing.assert_allclose(topology["freq_hz"], [500.0, 1000.0])

    spl_db = np.load(tmp_path / "spl_db.npy")
    radius_norm = np.load(tmp_path / "radius_norm.npy")
    assert spl_db.shape == (2, 35)
    assert radius_norm.shape == (2, 35)
    np.testing.assert_allclose(radius_norm, np.clip((spl_db + 30.0) / 30.0, 0.0, 1.0), atol=1e-6)


def _fibonacci_angles(count: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    phi = indices * np.pi * (3.0 - np.sqrt(5.0))
    return np.arccos(z).astype(np.float32), phi.astype(np.float32)


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


def test_export_on_axis_text_files_writes_single_channel_to_selected_file(tmp_path: Path) -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    dataset = LiveSolveDataset(
        angles,
        channel_configs=(ChannelConfig(name="main"),),
        flat_target_normalization_enabled=False,
        polar_observation_distance_m=0.343,
        exterior_sound_speed_m_per_s=343.0,
    )
    for freq_hz, pressure in ((1000.0, 1.0 + 0.0j), (200.0, 0.0 + 1.0j)):
        channel_pressure = np.full((1, 3), pressure, dtype=np.complex64)
        dataset.add(
            FrequencyResult(
                freq_hz=freq_hz,
                horizontal_spl_norm_db=np.zeros(3, dtype=np.float32),
                vertical_spl_norm_db=np.zeros(3, dtype=np.float32),
                impedance=np.array([[1.0, 0.0]], dtype=np.float32),
                channel_names=np.array(["main"]),
                horizontal_pressure=channel_pressure,
                vertical_pressure=channel_pressure.copy(),
            )
        )

    written = export_on_axis_text_files(dataset, tmp_path / "selected-response")

    assert written == [tmp_path / "selected-response.txt"]
    assert written[0].read_text(encoding="utf-8").splitlines() == [
        "200.000000\t93.979\t18.000",
        "1000.000000\t93.979\t0.000",
    ]


def test_export_on_axis_text_files_writes_only_individual_channels_with_safe_names(tmp_path: Path) -> None:
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    channel_names = np.array(["LF/woofer", "LF:woofer"])
    dataset = LiveSolveDataset(
        angles,
        channel_configs=(ChannelConfig(name="LF/woofer"), ChannelConfig(name="LF:woofer")),
        flat_target_normalization_enabled=False,
    )
    horizontal_pressure = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
            [0.0 - 1.0j, 0.0 - 1.0j, 0.0 - 1.0j],
        ],
        dtype=np.complex64,
    )
    dataset.add(
        FrequencyResult(
            freq_hz=1000.0,
            horizontal_spl_norm_db=np.zeros(3, dtype=np.float32),
            vertical_spl_norm_db=np.zeros(3, dtype=np.float32),
            impedance=np.ones((2, 2), dtype=np.float32),
            channel_names=channel_names,
            horizontal_pressure=horizontal_pressure,
            vertical_pressure=horizontal_pressure.copy(),
        )
    )

    written = export_on_axis_text_files(dataset, tmp_path / "channels")

    assert [path.name for path in written] == ["on_axis_LF_woofer.txt", "on_axis_LF_woofer_2.txt"]
    assert written[0].read_text(encoding="utf-8").splitlines() == [
        "1000.000000\t93.979\t0.000",
    ]
    assert written[1].read_text(encoding="utf-8").splitlines() == [
        "1000.000000\t93.979\t-90.000",
    ]
    assert len(list((tmp_path / "channels").glob("*.txt"))) == 2
