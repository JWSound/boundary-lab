import assert from "node:assert/strict";
import { Euler, Quaternion, Vector3 } from "three";
import { createDemoPackage } from "../src/model/demoPackage";
import { buildSourceInstance, buildPatternLookup, computeFieldFrame, computeMixedFieldFrame, computeMicrophonePatternResponses, computeMixedMicrophonePatternResponses } from "../src/model/field";
import type { SourceConfiguration, ObservationPlane } from "../src/model/types";

const pkg = createDemoPackage();
pkg.frequenciesHz = Float64Array.from([100]);
pkg.pressureShape = [1, 1, 6];
pkg.directionsPackage = Float32Array.from([1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1]);
pkg.radiiM = new Float32Array(6).fill(1);
pkg.pressure = { real: new Float32Array(6).fill(1), imag: new Float32Array(6).fill(0.25) };
const source: SourceConfiguration = {
  id: "s", name: "Speaker", packageId: pkg.id, channelId: "main", equalizer: { filters: [] },
  positionX: 0, positionHeightM: 1.4, positionZ: 0, pitchDeg: 0, yawDeg: 0, rollDeg: 0,
  levelDb: 0, delayMs: 0, polarity: 1,
};
const plane: ObservationPlane = {
  widthM: 0, depthM: 0, centerXM: 2, nearM: 7, heightM: 0, pitchDeg: 0, yawDeg: 0, rollDeg: 0,
  columns: 1, rows: 1, heatmapMinimumDb: 30, heatmapMaximumDb: 120, heatmapBandingDb: 3,
  displayMode: "spl", pressureScalePa: 1, phaseAnimationSpeedHz: 1,
};
const close = (actual: number, expected: number, tolerance = 1e-5) => assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);
function expected(config: SourceConfiguration, observation: ObservationPlane, image: boolean) {
  const inverse = new Quaternion().setFromEuler(new Euler(config.pitchDeg * Math.PI / 180, config.yawDeg * Math.PI / 180, config.rollDeg * Math.PI / 180, "YXZ")).invert();
  const ray = new Vector3(observation.centerXM - config.positionX, (image ? -1 : 1) * observation.heightM - config.positionHeightM, observation.nearM - config.positionZ);
  const r = Math.max(0.02, ray.length());
  ray.normalize().applyQuaternion(inverse);
  const local = [ray.x, ray.z, -ray.y];
  let best = 0;
  let dot = -Infinity;
  for (let i = 0; i < 6; i++) {
    const value = local.reduce((sum, axis, j) => sum + axis * pkg.directionsPackage[3 * i + j], 0);
    if (value > dot) { best = i; dot = value; }
  }
  const phase = 2 * Math.PI * 100 * ((r - 1) / pkg.manifest.medium.sound_speed_m_per_s + config.delayMs / 1000);
  const gain = (config.muted ? 0 : 10 ** (config.levelDb / 20)) * config.polarity / r;
  return [gain * (pkg.pressure.real[best] * Math.cos(phase) - pkg.pressure.imag[best] * Math.sin(phase)),
    gain * (pkg.pressure.real[best] * Math.sin(phase) + pkg.pressure.imag[best] * Math.cos(phase))];
}
function check(configs: SourceConfiguration[], observation: ObservationPlane) {
  const sources = configs.map(buildSourceInstance);
  const lookup = buildPatternLookup(pkg, 0);
  const map = new Map([[pkg.id, pkg]]);
  const frame = computeMixedFieldFrame(map, new Map([[pkg.id, lookup]]), sources, configs, observation, 100);
  const expectedSum = configs.flatMap((config) => [expected(config, observation, false), expected(config, observation, true)])
    .reduce((sum, value) => [sum[0] + value[0], sum[1] + value[1]], [0, 0]);
  close(frame.pressureReal[0], expectedSum[0]); close(frame.pressureImag[0], expectedSum[1]);
  const single = computeFieldFrame(pkg, sources, configs, observation, 0, lookup);
  close(single.pressureReal[0], frame.pressureReal[0]);
  const microphones = [{ id: "m", name: "Mic", positionX: observation.centerXM, positionHeightM: observation.heightM, positionZ: observation.nearM }];
  const response = computeMixedMicrophonePatternResponses(map, sources, configs, microphones);
  assert.equal(response.environment, "rigid_y0_half_space");
  close(response.traces[0].splDb[0], frame.splDb[0]);
  close(computeMicrophonePatternResponses(pkg, sources, configs, microphones).traces[0].splDb[0], frame.splDb[0]);
  return frame;
}
const ground = check([source], plane);
const direct = expected(source, plane, false);
close(ground.splDb[0] - 20 * Math.log10(Math.hypot(...direct) / 20e-6), 20 * Math.log10(2));
check([source], { ...plane, heightM: 2.3 });
check([{ ...source, levelDb: -4, delayMs: 1.7, polarity: -1 }], { ...plane, heightM: 1.8 });
check([source, { ...source, id: "muted", muted: true, positionX: 3 }], plane);
check([source, { ...source, id: "other", positionX: -3, positionHeightM: 2, delayMs: 2.1 }], { ...plane, heightM: 1.8 });
// A strongly asymmetric pattern catches incorrect reflection of source position alone.
pkg.pressure.real = Float32Array.from([1, 2, 3, 4, 5, 6]);
pkg.pressure.imag = Float32Array.from([0.2, -0.4, 0.6, -0.8, 1, -1.2]);
check([{ ...source, pitchDeg: 32, yawDeg: 57, rollDeg: -23 }], { ...plane, heightM: 4.5 });
const below = computeFieldFrame(pkg, [buildSourceInstance(source)], [source], { ...plane, heightM: -1 }, 0, buildPatternLookup(pkg, 0));
assert.equal(below.validMask[0], 0);
console.log("Rigid-ground pattern: complex image sum, +6 dB, rotated directivity, drive controls and microphone/map parity passed.");
