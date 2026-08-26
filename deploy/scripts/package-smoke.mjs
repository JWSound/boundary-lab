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
  yawDeg: 0,
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
  { widthM: 12, depthM: 12, nearM: 2, heightM: 1.2, columns: 24, rows: 20 },
  frequencyIndex,
  lookup,
);
assert.equal(field.splDb.length, 480);
assert.ok(field.splDb.every(Number.isFinite));
assert.ok(Number.isFinite(field.averageDb));

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
