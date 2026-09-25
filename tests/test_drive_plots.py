"""Physical synthesis checks for voltage-dependent plot data."""

from dataclasses import replace

import numpy as np
import pytest

from blab.config import ChannelConfig
from blab.live import ElectricalImpedanceDataset, InterfaceVelocityDataset, LiveSolveDataset
from blab.solve_results.model import INTERFACE_VELOCITY_ID, VOICE_COIL_CURRENT_ID
from blab.solvers.base import FrequencyResult
from blab.system_contract import QuantityResult, SystemFrequencyResult


def acoustic(channels):
    result = LiveSolveDataset(
        polar_angle_deg=np.asarray([0.0]),
        channel_configs=tuple(channels),
        flat_target_normalization_enabled=False,
        voltage_channel_names=frozenset(channel.name for channel in channels),
    )
    result.add(
        FrequencyResult(
            freq_hz=100.0,
            horizontal_spl_norm_db=np.zeros(1),
            vertical_spl_norm_db=np.zeros(1),
            impedance=np.zeros((len(channels), 2)),
            channel_names=np.asarray([c.name for c in channels]),
            horizontal_pressure=np.ones((len(channels), 1), dtype=complex),
            vertical_pressure=np.ones((len(channels), 1), dtype=complex),
        )
    )
    return result


def electrical(values, reference=2.0, orbits=(1, 1)):
    data = ElectricalImpedanceDataset(
        excitation_port_ids=("a", "b"),
        excitation_channel_names=np.asarray(["A", "B"]),
        excitation_component_ids=np.asarray(["a", "b"]),
        transducer_component_ids=np.asarray(["a", "b"]),
        physical_driver_orbit_counts=np.asarray(orbits),
        channel_names=np.asarray(["A", "B"]),
    )
    # Reverse both axes to exercise ID alignment as well as cross-channel terms.
    data.add(
        SystemFrequencyResult(
            freq_hz=100.0,
            excitation_port_ids=("b", "a"),
            diagnostics={"transducer_reference_voltage_v": reference},
            quantities=(
                QuantityResult(
                    id=VOICE_COIL_CURRENT_ID,
                    quantity="voice_coil_current",
                    unit="A",
                    axes=("excitation", "transducer"),
                    metadata={"component_ids": ["b", "a"]},
                    values=np.asarray(values, dtype=complex)[::-1, ::-1],
                ),
            ),
        )
    )
    return data


def test_real_power_cross_channel_symmetry_and_voltage_scaling():
    data = electrical([[1, 0.2], [0.3, 0.5]], orbits=(2, 1))
    live = acoustic([ChannelConfig(name="A", voltage_v=2), ChannelConfig(name="B", voltage_v=4)])
    _, names, powers = data.as_power_arrays(live)
    assert names.tolist() == ["A", "B", "Total input power"]
    # I_A = 1 + 2*.3; I_B = .2 + 2*.5, with two physical A drivers.
    np.testing.assert_allclose(powers[:, 0], [6.4, 4.8, 11.2], rtol=1e-6)
    live.set_channel_synthesis(tuple(replace(c, voltage_v=c.voltage_v * 2) for c in live.channel_configs))
    np.testing.assert_allclose(data.as_power_arrays(live)[2], powers * 4, rtol=1e-6)


def test_real_power_reactive_load_zero_drive_and_negative_channel_power():
    live = acoustic([ChannelConfig(name="A", voltage_v=2), ChannelConfig(name="B", voltage_v=2)])
    np.testing.assert_allclose(electrical([[1j, 0], [0, -0.5j]]).as_power_arrays(live)[2], 0, atol=1e-10)
    values = electrical([[1, -2], [-2, 4]])  # passive positive-semidefinite admittance
    np.testing.assert_allclose(values.as_power_arrays(live)[2][:, 0], [-2, 4, 2], rtol=1e-6)
    live.set_channel_synthesis((ChannelConfig(name="A", voltage_v=0), ChannelConfig(name="B", voltage_v=2)))
    np.testing.assert_allclose(values.as_power_arrays(live)[2][:, 0], [0, 8, 8], rtol=1e-6)


def test_interface_velocity_complex_cancellation_and_current_drive():
    data = InterfaceVelocityDataset(
        excitation_port_ids=("a", "b"),
        excitation_channel_names=np.asarray(["A", "B"]),
        voltage_excitation_mask=np.asarray([True, True]),
        interface_ids=("port",),
        interface_names=np.asarray(["Port"]),
    )
    result = SystemFrequencyResult(
        freq_hz=100,
        excitation_port_ids=("b", "a"),
        diagnostics={"transducer_reference_voltage_v": 2.0},
        quantities=(
            QuantityResult(
                id=INTERFACE_VELOCITY_ID,
                quantity="interface_average_normal_velocity",
                unit="m/s",
                axes=("excitation", "interface"),
                metadata={"interface_ids": ["port"]},
                values=np.asarray([[-1j], [1j]]),
            ),
        ),
    )
    data.add(result)
    live = acoustic([ChannelConfig(name="A", voltage_v=2), ChannelConfig(name="B", voltage_v=2)])
    np.testing.assert_allclose(data.as_velocity_arrays(live)[2], 0, atol=1e-10)
    live.set_channel_synthesis((ChannelConfig(name="A", voltage_v=4), ChannelConfig(name="B", voltage_v=2)))
    np.testing.assert_allclose(data.as_velocity_arrays(live)[2], 1, rtol=1e-6)
    live.set_channel_synthesis((ChannelConfig(name="A", voltage_v=8), ChannelConfig(name="B", voltage_v=4)))
    np.testing.assert_allclose(data.as_velocity_arrays(live)[2], 2, rtol=1e-6)
    bad = replace(result, quantities=(replace(result.quantities[0], metadata={"interface_ids": ["wrong"]}),))
    with pytest.raises(ValueError, match="IDs"):
        data.add(bad)


def test_power_missing_reference_is_unavailable_not_guessed():
    live = acoustic([ChannelConfig(name="A"), ChannelConfig(name="B")])
    assert np.isnan(electrical([[1, 0], [0, 1]], reference=float("nan")).as_power_arrays(live)[2]).all()


def test_real_power_respects_channel_phase_and_shared_channel_drivers():
    live = acoustic([ChannelConfig(name="A", voltage_v=2), ChannelConfig(name="B", voltage_v=2, delay_ms=2.5)])
    # At 100 Hz, B is -90 degrees; imaginary cross-coupling becomes real current.
    data = electrical([[1, 1j], [1j, 1]])
    np.testing.assert_allclose(data.as_power_arrays(live)[2][:, 0], [4, 0, 4], atol=1e-6)
    data.excitation_channel_names = np.asarray(["A", "A"])
    data.channel_names = np.asarray(["A"])
    live = acoustic([ChannelConfig(name="A", voltage_v=2)])
    np.testing.assert_allclose(data.as_power_arrays(live)[2][:, 0], [4, 4], atol=1e-6)


def test_drive_plot_canvases_stream_capture_and_export(main_window, monkeypatch, tmp_path):
    from blab.ui.result_projection import FrequencyTraceProjection, VisualizationProjection

    window = main_window
    frequencies = np.asarray([30.0, 60.0, 120.0])
    power = FrequencyTraceProjection(
        frequencies, np.asarray(["A", "Total"]), np.asarray([[-1.0, 2.0, 3.0], [1.0, 3.0, 5.0]])
    )
    velocity = FrequencyTraceProjection(frequencies, np.asarray(["Port"]), np.asarray([[0.1, 0.3, 0.2]]))
    projection = VisualizationProjection(None, None, None, real_input_power=power, interface_velocity=velocity)
    assert not window.interface_velocity_plot.isEnabled()
    window._update_real_input_power_plot(projection)
    window._update_interface_velocity_plot(projection)
    assert window.interface_velocity_plot.isEnabled()
    assert window.real_input_power_plot.axes.get_ylim()[0] < -1
    assert window.interface_velocity_plot.axes.get_ylim()[0] == 0
    line = window.interface_velocity_plot._lines["Port"]
    window._update_interface_velocity_plot(projection)
    assert window.interface_velocity_plot._lines["Port"] is line
    snapshot = projection.snapshot()
    velocity.values[:] = 9
    assert snapshot.interface_velocity.values[0, 0] == 0.1
    window.interface_velocity_plot.set_comparison_plot(
        snapshot.interface_velocity.freq_hz,
        snapshot.interface_velocity.trace_names,
        snapshot.interface_velocity.values,
    )
    window.interface_velocity_plot.set_series_visible("Port", False)
    assert not line.get_visible()
    window.interface_velocity_plot.set_series_visible("Port", True)
    window.live_dataset = acoustic([ChannelConfig(name="A")])
    monkeypatch.setattr(window, "prepared_live_dataset", lambda **_kwargs: snapshot)
    for plot_id, expected in (
        ("real_input_power", "Real power"),
        ("interface_velocity", "Average normal velocity RMS"),
    ):
        target = tmp_path / f"{plot_id}.txt"
        assert window._write_plot_data(plot_id, target) == [target]
        assert expected in target.read_text()
    window._update_interface_velocity_plot(VisualizationProjection(None, None, None))
    assert not window.interface_velocity_plot.isEnabled()
    assert window.interface_velocity_plot._plot_state is None


def test_interface_velocity_prescribed_basis_and_duplicate_names():
    data = InterfaceVelocityDataset(
        excitation_port_ids=("source",),
        excitation_channel_names=np.asarray(["A"]),
        voltage_excitation_mask=np.asarray([False]),
        interface_ids=("left", "right"),
        interface_names=np.asarray(["Port", "Port"]),
    )
    result = SystemFrequencyResult(
        freq_hz=100,
        excitation_port_ids=("source",),
        quantities=(
            QuantityResult(
                id=INTERFACE_VELOCITY_ID,
                quantity="interface_average_normal_velocity",
                unit="m/s",
                axes=("excitation", "interface"),
                metadata={"interface_ids": ["right", "left"]},
                values=np.asarray([[3j, 4j]]),
            ),
        ),
    )
    data.add(result)
    live = acoustic([ChannelConfig(name="A")])
    live.voltage_channel_names = frozenset()
    _, names, values = data.as_velocity_arrays(live)
    assert names.tolist() == ["Port [left]", "Port [right]"]
    np.testing.assert_allclose(values[:, 0], [4, 3])
    live.set_channel_synthesis((ChannelConfig(name="A", polarity=-1),))
    np.testing.assert_allclose(data.as_velocity_arrays(live)[2], values)


def radiation_data():
    from blab.interface_radiation import InterfaceRadiationDataset
    from blab.solve_results.model import INTERFACE_RADIATION_ID

    data = InterfaceRadiationDataset(
        excitation_port_ids=("a", "b"),
        excitation_channel_names=np.asarray(["A", "B"]),
        voltage_excitation_mask=np.asarray([True, True]),
        source_ids=("front", "rear", "radiation:total"),
        source_names=np.asarray(["Port", "Port", "Combined"]),
        points_m=np.asarray([[0.0, 0.0, 3.0]]),
    )
    # Each independent excitation's source sum must reproduce its total.
    values = np.asarray([[1j, -0.9j, 0.1j], [2.0, -1.0, 1.0]], dtype=complex)[:, :, None]
    result = SystemFrequencyResult(
        freq_hz=100,
        excitation_port_ids=("b", "a"),
        diagnostics={"transducer_reference_voltage_v": 2.0},
        quantities=(
            QuantityResult(
                id=INTERFACE_RADIATION_ID,
                quantity="interface_radiated_pressure",
                unit="Pa",
                axes=("excitation", "radiation_source", "observation"),
                values=values[::-1, ::-1],
                metadata={"radiation_source_ids": list(reversed(data.source_ids)), "points_m": [[0, 0, 3]]},
            ),
        ),
    )
    data.add(result)
    return data, result


def test_interface_radiation_coherent_dsp_and_id_alignment():
    data, _ = radiation_data()
    live = acoustic([ChannelConfig(name="A", voltage_v=2), ChannelConfig(name="B", voltage_v=2)])
    _, names, p = data.as_pressure_arrays(live)
    assert names.tolist() == ["Port [front]", "Port [rear]", "Combined"]
    np.testing.assert_allclose(p[:, 0], [2 + 1j, -1 - 0.9j, 1 + 0.1j])
    np.testing.assert_allclose(p[:-1].sum(axis=0), p[-1])
    # Voltage scaling, gain, polarity and time delay are applied before magnitude.
    live.set_channel_synthesis(
        (ChannelConfig(name="A", voltage_v=4, polarity=-1), ChannelConfig(name="B", voltage_v=2, delay_ms=2.5))
    )
    processed = data.as_pressure_arrays(live)[2]
    np.testing.assert_allclose(processed[:, 0], [-4j, 2.8j, -1.2j], atol=1e-6)
    np.testing.assert_allclose(processed[:-1].sum(axis=0), processed[-1])
    np.testing.assert_allclose(data.as_spl_arrays(live)[2], 20 * np.log10(abs(processed) / 2e-5))


def test_interface_radiation_rejects_invalid_contract_and_handles_old_results():
    data, result = radiation_data()
    for quantity in (
        replace(result.quantities[0], values=result.quantities[0].values.real),
        replace(result.quantities[0], values=np.full((2, 3, 1), complex(np.nan, 0))),
        replace(
            result.quantities[0],
            metadata={"radiation_source_ids": list(reversed(data.source_ids)), "points_m": [[0, 0, 1]]},
        ),
    ):
        with pytest.raises(ValueError):
            data.add(replace(result, quantities=(quantity,)))
    data.results.clear()
    data.add(replace(result, quantities=()))
    live = acoustic([ChannelConfig(name="A"), ChannelConfig(name="B")])
    assert data.as_spl_arrays(live) is None
    data.add(replace(result, diagnostics={}))
    assert np.isnan(data.as_pressure_arrays(live)[2]).all()


def test_interface_radiation_canvas_export_and_snapshot(main_window, tmp_path, monkeypatch):
    window = main_window
    from blab.ui.result_projection import FrequencyTraceProjection, VisualizationProjection

    data, _ = radiation_data()
    live = acoustic([ChannelConfig(name="A", voltage_v=2), ChannelConfig(name="B", voltage_v=2)])
    window.live_dataset = live
    window._solve_session().interface_radiation = data
    monkeypatch.setattr(window, "channel_configs", lambda: live.channel_configs)
    plot_data = FrequencyTraceProjection(*data.as_spl_arrays(live))
    projection = VisualizationProjection(None, None, None, interface_radiation=plot_data)
    assert not window.interface_radiation_plot.isEnabled()
    window._update_interface_radiation_plot(projection)
    assert window.interface_radiation_plot.isEnabled()
    assert len(window.interface_radiation_plot._lines) == 3
    snapshot = projection.snapshot()
    plot_data.values[:] = -40
    assert np.all(snapshot.interface_radiation.values > 0)
    line = window.interface_radiation_plot._lines["Port [front]"]
    window.interface_radiation_plot.set_series_visible("Port [front]", False)
    assert not line.get_visible()
    target = tmp_path / "radiation.txt"
    assert window._write_plot_data("interface_radiation", target) == [target]
    text = target.read_text()
    assert "Real pressure" in text and "Imaginary pressure" in text and "Phase" in text
    window._update_interface_radiation_plot(VisualizationProjection(None, None, None))
    assert not window.interface_radiation_plot.isEnabled()
