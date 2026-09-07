import type { ElectricalData, ElectricalTrace } from "../components/ElectricalPlot";
import type { MicrophoneSweepResult } from "./types";

export type AcousticLoadingSample = {
  [Key in keyof NonNullable<MicrophoneSweepResult["acoustic_loading"]>]: (number | null)[];
};

export function differentialMagnitude(real: number | null | undefined, imag: number | null | undefined): number {
  return real == null || imag == null ? NaN : Math.hypot(real, imag);
}

export function pressureDisplayScale(peak: boolean, kilopascals: boolean): number {
  return (peak ? Math.SQRT2 : 1) / (kilopascals ? 1000 : 1);
}

/** Insert one solved frequency without mutating previously rendered data. */
export function updateAcousticLoading(
  current: ElectricalData, progress: {
    frequency_hz: number; transducer_ids: string[]; transducer_names: string[];
    acoustic_loading?: AcousticLoadingSample | null;
  },
): ElectricalData {
  const loading = progress.acoustic_loading;
  if (!loading) return current;
  const frequencyIndex = current.frequenciesHz.findIndex(
    (frequency) => Math.abs(frequency - progress.frequency_hz) <= Math.max(1e-4, frequency * 1e-6),
  );
  if (frequencyIndex < 0) return current;
  const traces = new Map(current.traces);
  progress.transducer_ids.forEach((id, index) => {
    const existing = traces.get(id);
    const insert = (previous: Float32Array | undefined, value: number | null | undefined) => {
      const values = previous?.slice() ?? new Float32Array(current.frequenciesHz.length).fill(NaN);
      values[frequencyIndex] = value ?? NaN;
      return values;
    };
    const pressure = differentialMagnitude(loading.pressure_real_pa?.[index], loading.pressure_imag_pa?.[index]);
    if (!existing && !Number.isFinite(pressure) && !Number.isFinite(loading.resistance[index] ?? NaN) && !Number.isFinite(loading.reactance[index] ?? NaN)) return;
    traces.set(id, {
      name: progress.transducer_names[index] ?? id,
      differentialPressurePa: insert(existing?.differentialPressurePa, pressure),
      acousticResistance: insert(existing?.acousticResistance, loading.resistance[index]),
      acousticReactance: insert(existing?.acousticReactance, loading.reactance[index]),
      impedanceMagnitudeOhm: new Float32Array(), impedancePhaseDeg: new Float32Array(),
      rmsCurrentA: new Float32Array(), realPowerW: new Float32Array(),
    });
  });
  return { ...current, traces };
}

/** Preserve missing samples as gaps, and each capture's original frequency grid. */
export function acousticLoadingTraces(result: MicrophoneSweepResult): Map<string, ElectricalTrace> {
  const traces = new Map<string, ElectricalTrace>();
  const loading = result.acoustic_loading;
  if (!loading) return traces;
  const samples = (values?: (number | null)[]) => Float32Array.from(
    result.frequencies_hz.map((_, index) => values?.[index] ?? NaN),
  );
  result.transducer_ids.forEach((id, index) => {
    const resistance = samples(loading.resistance[index]);
    const reactance = samples(loading.reactance[index]);
    const pressure = Float32Array.from(result.frequencies_hz.map((_, f) =>
      differentialMagnitude(loading.pressure_real_pa?.[index]?.[f], loading.pressure_imag_pa?.[index]?.[f])));
    if (!pressure.some(Number.isFinite) && !resistance.some(Number.isFinite) && !reactance.some(Number.isFinite)) return;
    traces.set(id, {
      name: result.transducer_names[index] ?? id,
      differentialPressurePa: pressure,
      frequenciesHz: Float64Array.from(result.frequencies_hz),
      acousticResistance: resistance, acousticReactance: reactance,
      impedanceMagnitudeOhm: new Float32Array(), impedancePhaseDeg: new Float32Array(),
      rmsCurrentA: new Float32Array(), realPowerW: new Float32Array(),
    });
  });
  return traces;
}
