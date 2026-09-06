import assert from "node:assert/strict";
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
