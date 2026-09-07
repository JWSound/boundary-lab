import assert from "node:assert/strict";
import { acousticLoadingTraces, updateAcousticLoading } from "../src/model/acousticLoading";
import type { MicrophoneSweepResult } from "../src/model/types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ElectricalPlot } from "../src/components/ElectricalPlot";
import { captureAnalysis, microphoneOverlays, serializeCapture, type AnalysisCapture } from "../src/model/analysisCapture";

const original: AnalysisCapture = {
  id: "capture-1", name: "Wide array", createdAt: "2026-09-06T00:00:00Z", project: "{}",
  pattern: { frequenciesHz: new Float64Array([100, 200]), traces: [{ microphoneId: "mic", microphoneName: "Front", splDb: new Float32Array([80, 90]), clippedNearFieldSamples: 0 }] },
  boundary: { key: "scene", frequenciesHz: new Float64Array([100, 200]), traces: new Map([["mic", new Float32Array([81, 91])]]) },
  coupled: null, excursion: null, electrical: null, raw: {},
};
const saved = captureAnalysis(original);
original.pattern.traces[0].splDb[0] = 12;
original.boundary!.traces.get("mic")![0] = 13;
assert.equal(saved.pattern.traces[0].splDb[0], 80);
assert.equal(saved.boundary!.traces.get("mic")![0], 81);
const other = captureAnalysis({ ...saved, id: "capture-2", name: "Narrow array" });
other.pattern.frequenciesHz = new Float64Array([150, 300]);
const overlays = microphoneOverlays([saved, other]);
assert.equal(new Set(overlays.map((trace) => trace.id)).size, 4);
assert.deepEqual(Array.from(overlays[2].frequenciesHz), [150, 300]);
assert.deepEqual(Array.from(overlays[0].frequenciesHz), [100, 200]);
assert.equal(overlays[1].method, "boundary");
const portable = JSON.parse(serializeCapture(saved));
assert.deepEqual(portable.boundary.traces.mic, [81, 91]);
assert.deepEqual(portable.pattern.frequenciesHz, [100, 200]);
assert.equal(portable.schema, "boundary-lab-deploy-analysis");
console.log("Analysis capture isolation, frequency grids and serialization passed.");
const acoustic = {
  frequencies_hz: [40, 80, 160], transducer_ids: ["cab:driver"], transducer_names: ["Cab / Driver"],
  acoustic_loading: {
    resistance: [[1, null, -2]], reactance: [[-3, null, 4]],
    isolated_resistance: [[0.5, null, 1]], isolated_reactance: [[-1, null, 2]],
  },
} as MicrophoneSweepResult;
const acousticCapture = captureAnalysis({ ...original, raw: { coupled: acoustic } });
acoustic.acoustic_loading!.resistance[0][0] = 99;
const acousticTrace = acousticLoadingTraces(acousticCapture.raw.coupled!).get("cab:driver")!;
assert.equal(acousticTrace.acousticResistance![0], 1);
assert.ok(Number.isNaN(acousticTrace.acousticResistance![1]));
assert.equal(acousticTrace.acousticResistance![2], -2);
assert.deepEqual(Array.from(acousticTrace.frequenciesHz!), [40, 80, 160]);
assert.equal(JSON.parse(serializeCapture(acousticCapture)).raw.coupled.acoustic_loading.resistance[0][1], null);
assert.equal(acousticLoadingTraces({ ...acoustic, acoustic_loading: undefined }).size, 0);
console.log("Acoustic loading captures, missing samples and legacy results passed.");
const markup = renderToStaticMarkup(createElement(ElectricalPlot, {
  data: { key: "test", frequenciesHz: new Float64Array(), traces: new Map([["cab:driver", acousticTrace]]) },
  view: "acoustic", coupledSelected: true, currentFrequencyHz: 80, frequencyPosition: 1,
  frequencyCount: 3, onFrequencyPositionChange: () => {}, canCalculate: true,
  calculating: false, completedCount: 3, totalCount: 3, onCalculateOrStop: () => {},
}));
assert.ok(markup.includes("R / (ρcSd)"));
assert.ok(markup.includes('class="bem-trace"'));
assert.ok(markup.includes('class="electrical-phase-trace"'));
assert.ok(markup.includes("matched driver motion"));
assert.ok(!markup.includes("NaN"));
console.log("Acoustic chart rendering and isolated overlay passed.");
const emptyLoading = { key: "live", frequenciesHz: Float64Array.from([40, 80, 160]), traces: new Map() };
const sample = {
  frequency_hz: 160, transducer_ids: ["cab:driver"], transducer_names: ["Cab / Driver"],
  acoustic_loading: { resistance: [-2], reactance: [4], isolated_resistance: [1], isolated_reactance: [2] },
};
const first = updateAcousticLoading(emptyLoading, sample);
assert.equal(emptyLoading.traces.size, 0);
assert.deepEqual(Array.from(first.traces.get("cab:driver")!.acousticResistance!), [NaN, NaN, -2]);
const second = updateAcousticLoading(first, { ...sample, frequency_hz: 40.000001 });
assert.deepEqual(Array.from(second.traces.get("cab:driver")!.acousticResistance!), [-2, NaN, -2]);
assert.deepEqual(Array.from(first.traces.get("cab:driver")!.acousticResistance!), [NaN, NaN, -2]);
assert.deepEqual(Array.from(second.traces.get("cab:driver")!.isolatedReactance!), [2, NaN, 2]);
assert.equal(updateAcousticLoading(second, { ...sample, frequency_hz: 999 }), second);
assert.equal(updateAcousticLoading(second, { ...sample, acoustic_loading: null }), second);
const gap = updateAcousticLoading(second, {
  ...sample, frequency_hz: 40,
  acoustic_loading: { resistance: [null], reactance: [null], isolated_resistance: [null], isolated_reactance: [null] },
});
assert.ok(Number.isNaN(gap.traces.get("cab:driver")!.acousticResistance![0]));
console.log("Streaming acoustic loading, out-of-order frequencies, immutable updates and gaps passed.");
