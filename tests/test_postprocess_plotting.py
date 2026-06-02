from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from blab.plotting import VisualizerConfig, generate_plots, load_data
from blab.postprocess import PrepConfig, prepare_visualization_data, prepare_visualization_data_from_arrays


def _write_raw_solver_npz(path: Path) -> None:
    freq_hz = np.array([200.0, 1000.0, 5000.0], dtype=np.float32)
    angles = np.array([-180.0, 0.0, 180.0], dtype=np.float32)
    horizontal = np.array(
        [
            [-12.0, 0.0, -12.0],
            [-10.0, 0.0, -10.0],
            [-8.0, 0.0, -8.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        path,
        freq_hz=freq_hz,
        polar_angle_deg=angles,
        horizontal_spl_db=horizontal + 80.0,
        vertical_spl_db=horizontal + 78.0,
        horizontal_spl_norm_db=horizontal,
        vertical_spl_norm_db=horizontal,
        impedance_freq_hz=freq_hz,
        impedance_radiator_names=np.array(["HF", "LF"]),
        impedance_real=np.ones((2, 3), dtype=np.float32),
        impedance_imag=np.zeros((2, 3), dtype=np.float32),
    )


def test_prepare_visualization_data_preserves_multiradiator_impedance(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.npz"
    formatted_path = tmp_path / "formatted.npz"
    _write_raw_solver_npz(raw_path)

    prepare_visualization_data(
        PrepConfig(
            input_polar_npz=raw_path,
            output_npz=formatted_path,
            angle_samples=5,
            freq_samples=5,
            octave_smoothing=None,
        )
    )

    with np.load(formatted_path) as data:
        assert data["impedance_real"].shape == (2, 3)
        assert data["impedance_imag"].shape == (2, 3)
        assert data["impedance_radiator_names"].tolist() == ["HF", "LF"]
        assert data["horizontal_isobar_db"].shape == (5, 5)
        assert np.allclose(data["horizontal_spl_db"], horizontal := np.array(
            [[68.0, 80.0, 68.0], [70.0, 80.0, 70.0], [72.0, 80.0, 72.0]],
            dtype=np.float32,
        ))


def test_generate_plots_writes_expected_pngs(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.npz"
    formatted_path = tmp_path / "formatted.npz"
    _write_raw_solver_npz(raw_path)
    prepare_visualization_data(
        PrepConfig(
            input_polar_npz=raw_path,
            output_npz=formatted_path,
            angle_samples=5,
            freq_samples=5,
            octave_smoothing=None,
        )
    )

    outputs = generate_plots(
        load_data(formatted_path),
        VisualizerConfig(
            input_npz=formatted_path,
            output_dir=tmp_path,
            figure_dpi=72,
        ),
    )

    assert set(outputs) == {
        "horizontal_isobar_png",
        "vertical_isobar_png",
        "acoustic_impedance_png",
        "on_axis_frequency_response_png",
        "spinorama_png",
    }
    assert (tmp_path / "horizontal_isobar.png").exists()
    assert (tmp_path / "vertical_isobar.png").exists()
    assert (tmp_path / "acoustic_impedance.png").exists()
    assert (tmp_path / "on_axis_frequency_response.png").exists()
    assert (tmp_path / "spinorama.png").exists()


def test_prepare_visualization_data_can_skip_normalization_and_auto_span() -> None:
    freqs = np.array([200.0, 1000.0], dtype=np.float32)
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    horizontal = np.array([[61.2, 68.4, 62.5], [64.0, 70.1, 65.0]], dtype=np.float32)
    vertical = horizontal - 2.0

    dataset = prepare_visualization_data_from_arrays(
        freq_hz=freqs,
        polar_angle_deg=angles,
        horizontal_spl_norm_db=horizontal,
        vertical_spl_norm_db=vertical,
        impedance_freq_hz=freqs,
        impedance_radiator_names=np.array(["throat"]),
        impedance_real=np.ones((1, 2), dtype=np.float32),
        impedance_imag=np.zeros((1, 2), dtype=np.float32),
        cfg=PrepConfig(
            angle_samples=None,
            freq_samples=None,
            octave_smoothing=None,
            normalize_polar=False,
            auto_db_span=True,
        ),
    )

    assert dataset["clip_min_db"] == 59.0
    assert dataset["clip_max_db"] == 71.0
    assert np.allclose(dataset["horizontal_spl_norm_db"], horizontal)


def test_prepare_visualization_data_preserves_raw_spl_for_on_axis_response() -> None:
    freqs = np.array([200.0, 1000.0], dtype=np.float32)
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    normalized = np.array([[-6.0, 0.0, -6.0], [-5.0, 0.0, -5.0]], dtype=np.float32)
    raw = normalized + 86.0

    dataset = prepare_visualization_data_from_arrays(
        freq_hz=freqs,
        polar_angle_deg=angles,
        horizontal_spl_norm_db=normalized,
        vertical_spl_norm_db=normalized,
        horizontal_spl_db=raw,
        vertical_spl_db=raw - 1.0,
        impedance_freq_hz=freqs,
        impedance_radiator_names=np.array(["throat"]),
        impedance_real=np.ones((1, 2), dtype=np.float32),
        impedance_imag=np.zeros((1, 2), dtype=np.float32),
        cfg=PrepConfig(angle_samples=None, freq_samples=None, octave_smoothing=None),
    )

    assert np.allclose(dataset["horizontal_spl_db"], raw)
    assert np.allclose(dataset["horizontal_spl_norm_db"][:, 1], 0.0)


def test_prepare_visualization_data_keeps_normalization_and_spin_reference_angles_separate() -> None:
    freqs = np.array([200.0, 1000.0], dtype=np.float32)
    angles = np.array([-90.0, 0.0, 90.0], dtype=np.float32)
    horizontal = np.array([[-6.0, 0.0, -4.0], [-5.0, 0.0, -3.0]], dtype=np.float32)
    vertical = horizontal - 1.0

    dataset = prepare_visualization_data_from_arrays(
        freq_hz=freqs,
        polar_angle_deg=angles,
        horizontal_spl_norm_db=horizontal,
        vertical_spl_norm_db=vertical,
        impedance_freq_hz=freqs,
        impedance_radiator_names=np.array(["throat"]),
        impedance_real=np.ones((1, 2), dtype=np.float32),
        impedance_imag=np.zeros((1, 2), dtype=np.float32),
        cfg=PrepConfig(
            angle_samples=None,
            freq_samples=None,
            octave_smoothing=None,
            hor_ref_angle=30.0,
            vert_ref_angle=-20.0,
            spin_hor_ref_angle=5.0,
            spin_vert_ref_angle=-10.0,
        ),
    )

    assert dataset["horizontal_normalization_angle_deg"] == np.float32(30.0)
    assert dataset["vertical_normalization_angle_deg"] == np.float32(-20.0)
    assert dataset["spin_horizontal_reference_angle_deg"] == np.float32(5.0)
    assert dataset["spin_vertical_reference_angle_deg"] == np.float32(-10.0)
