import { strict as assert } from "node:assert";
import { readFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { build } from "esbuild";

const packagePath = resolve(process.argv[2]);
const outputPath = resolve(".tmp/package-reader.mjs");
await mkdir(dirname(outputPath), { recursive: true });
await build({
  entryPoints: [resolve("scripts/package-smoke-entry.ts")],
  outfile: outputPath,
  bundle: true,
  platform: "node",
  format: "esm",
  logLevel: "silent",
});
const {
  loadSpeakerPackage,
  buildSourceInstance,
  buildPatternLookup,
  computeFieldFrame,
  nearestFrequencyIndex,
  heatmapColorBoundaries,
  heatmapLegendGradient,
  writeHeatmapColor,
  createDeployProject,
  parseDeployProject,
  serializeDeployProject,
} = await import(`${pathToFileURL(outputPath).href}?${Date.now()}`);
const bytes = await readFile(packagePath);
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const speaker = loadSpeakerPackage(buffer, packagePath.split(/[\\/]/).at(-1));

assert.equal(speaker.manifest.schema, "boundary-lab-speaker-package");
assert.equal(speaker.manifest.schema_version, 1);
assert.ok(speaker.frequenciesHz.length > 0);
assert.equal(speaker.pressureShape[0], speaker.frequenciesHz.length);
assert.equal(speaker.pressureShape[2] * 3, speaker.directionsPackage.length);
assert.equal(speaker.pressure.real.length, speaker.pressure.imag.length);
assert.ok(speaker.mesh?.indices.length > 0);
const frequencyIndex = nearestFrequencyIndex(speaker, 80);
const sourceConfig = {
  id: "subwoofer-1",
  positionX: 0,
  positionHeightM: 0.4,
  positionZ: 0,
  pitchDeg: 0,
  yawDeg: 0,
  rollDeg: 0,
  levelDb: -3,
  delayMs: 0,
  polarity: 1,
};
const source = buildSourceInstance(sourceConfig);
const lookup = buildPatternLookup(speaker, frequencyIndex);
const field = computeFieldFrame(
  speaker,
  [source],
  [sourceConfig],
  { widthM: 12, depthM: 12, centerXM: 0, nearM: 2, heightM: 1.2, pitchDeg: 0, yawDeg: 0, rollDeg: 0, columns: 24, rows: 20, heatmapMinimumDb: 50, heatmapMaximumDb: 145, heatmapBandingDb: 0 },
  frequencyIndex,
  lookup,
);
assert.equal(field.splDb.length, 480);
assert.ok(field.splDb.every(Number.isFinite));
assert.ok(Number.isFinite(field.averageDb));
const clippedField = computeFieldFrame(
  speaker,
  [source],
  [sourceConfig],
  { widthM: 12, depthM: 12, centerXM: 0, nearM: 2, heightM: 0, pitchDeg: 30, yawDeg: 0, rollDeg: 0, columns: 12, rows: 12, heatmapMinimumDb: 50, heatmapMaximumDb: 145, heatmapBandingDb: 0 },
  frequencyIndex,
  lookup,
);
const clippedValidCount = clippedField.validMask.reduce((sum, value) => sum + value, 0);
assert.ok(clippedValidCount > 0 && clippedValidCount < clippedField.validMask.length);

assert.deepEqual(heatmapColorBoundaries(50, 145, 0), []);
const fiveDbBoundaries = heatmapColorBoundaries(50, 145, 5);
assert.deepEqual(fiveDbBoundaries.slice(0, 3), [50, 55, 60]);
assert.deepEqual(fiveDbBoundaries.slice(-2), [140, 145]);
const continuousPixels = new Uint8Array(8);
writeHeatmapColor(50, 50, 145, [], continuousPixels, 0);
writeHeatmapColor(145, 50, 145, [], continuousPixels, 4);
assert.deepEqual(Array.from(continuousPixels.slice(0, 3)), [0, 0, 143]);
assert.deepEqual(Array.from(continuousPixels.slice(4, 7)), [143, 0, 0]);
const bandedPixels = new Uint8Array(12);
writeHeatmapColor(60, 50, 145, fiveDbBoundaries, bandedPixels, 0);
writeHeatmapColor(64.9, 50, 145, fiveDbBoundaries, bandedPixels, 4);
writeHeatmapColor(65, 50, 145, fiveDbBoundaries, bandedPixels, 8);
assert.deepEqual(Array.from(bandedPixels.slice(0, 3)), Array.from(bandedPixels.slice(4, 7)));
assert.notDeepEqual(Array.from(bandedPixels.slice(4, 7)), Array.from(bandedPixels.slice(8, 11)));
assert.match(heatmapLegendGradient(50, 145, 5), /5\.263%/);

const projectText = serializeDeployProject(createDeployProject(
  "Smoke Project",
  speaker,
  [sourceConfig],
  [{ id: "microphone-1", name: "Microphone 1", positionX: 0, positionHeightM: 1.2, positionZ: 6 }],
  { widthM: 12, depthM: 10, centerXM: 1, nearM: 2, heightM: 1.2, pitchDeg: 0, yawDeg: 5, rollDeg: 0, columns: 24, rows: 20, heatmapMinimumDb: 50, heatmapMaximumDb: 145, heatmapBandingDb: 0 },
  speaker.frequenciesHz[frequencyIndex],
  "boundary",
));
const parsedProject = parseDeployProject(projectText);
assert.equal(parsedProject.name, "Smoke Project");
assert.equal(parsedProject.sources[0].id, sourceConfig.id);
assert.equal(parsedProject.microphones[0].name, "Microphone 1");
assert.equal(parsedProject.observation_plane.columns, 24);
assert.equal(parsedProject.requested_fidelity, "boundary");
const unsupportedProject = JSON.parse(projectText);
unsupportedProject.schema_version = 1;
assert.throws(() => parseDeployProject(JSON.stringify(unsupportedProject)), /Unsupported.*version 1/);
const incompleteProject = JSON.parse(projectText);
delete incompleteProject.observation_plane.heatmapMinimumDb;
assert.throws(() => parseDeployProject(JSON.stringify(incompleteProject)), /heatmapMinimumDb/);
assert.throws(
  () => parseDeployProject(JSON.stringify({ ...JSON.parse(projectText), sources: [sourceConfig, sourceConfig] })),
  /unique id/,
);

console.log(JSON.stringify({
  name: speaker.manifest.name,
  fidelity: speaker.manifest.fidelity,
  frequencies: speaker.frequenciesHz.length,
  directions: speaker.pressureShape[2],
  meshVertices: speaker.mesh.positions.length / 3,
  meshTriangles: speaker.mesh.indices.length / 3,
  boundsM: speaker.boundsM.map((value) => Number(value.toFixed(3))),
  previewFrequencyHz: speaker.frequenciesHz[frequencyIndex],
  previewAverageDb: Number(field.averageDb.toFixed(2)),
}));
