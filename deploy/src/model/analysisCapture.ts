import type { BemResponseData, MicrophoneOverlay } from "../components/MicrophoneResponsePlot";
import type { DriverExcursionData } from "../components/DriverExcursionPlot";
import type { ElectricalData } from "../components/ElectricalPlot";
import type { MicrophoneResponseSet, MicrophoneSweepResult } from "./types";

export interface AnalysisCapture {
  id: string;
  name: string;
  createdAt: string;
  project: string;
  pattern: MicrophoneResponseSet;
  boundary: BemResponseData | null;
  coupled: BemResponseData | null;
  excursion: DriverExcursionData | null;
  electrical: ElectricalData | null;
  raw: Partial<Record<"boundary" | "coupled", MicrophoneSweepResult>>;
}

/** Copy arrays and maps so subsequent streaming updates cannot mutate a capture. */
export function captureAnalysis(value: AnalysisCapture): AnalysisCapture {
  return structuredClone(value);
}

export function microphoneOverlays(captures: AnalysisCapture[]): MicrophoneOverlay[] {
  return captures.flatMap((capture) => {
    const names = new Map(capture.pattern.traces.map((trace) => [trace.microphoneId, trace.microphoneName]));
    const traces: MicrophoneOverlay[] = capture.pattern.traces.map((trace) => ({
      id: `${capture.id}:pattern:${trace.microphoneId}`,
      name: `${capture.name} / ${trace.microphoneName} / Pattern`,
      method: "pattern",
      frequenciesHz: capture.pattern.frequenciesHz,
      values: trace.splDb,
    }));
    for (const method of ["boundary", "coupled"] as const) {
      const response = capture[method];
      if (response) for (const [id, values] of response.traces) traces.push({
        id: `${capture.id}:${method}:${id}`,
        name: `${capture.name} / ${names.get(id) ?? id} / ${method}`,
        method,
        frequenciesHz: response.frequenciesHz,
        values,
      });
    }
    return traces;
  });
}

export function serializeCapture(capture: AnalysisCapture): string {
  return JSON.stringify({ schema: "boundary-lab-deploy-analysis", schema_version: 1, ...capture }, (_key, value) => {
    if (value instanceof Map) return Object.fromEntries(value);
    if (ArrayBuffer.isView(value)) return Array.from(value as unknown as ArrayLike<number>);
    return value;
  }, 2);
}
