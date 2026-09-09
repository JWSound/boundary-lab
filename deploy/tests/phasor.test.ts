import assert from "node:assert/strict";
import { strToU8, zipSync } from "fflate";
import { loadSpeakerPackage } from "../src/io/speakerPackage";

function npy(values: Float32Array | Float64Array, shape: number[], descr: string): Uint8Array {
  const header = strToU8(`{'descr': '${descr}', 'fortran_order': False, 'shape': (${shape.join(", ")},), }`);
  const headerSize = Math.ceil((10 + header.length + 1) / 64) * 64 - 10;
  const output = new Uint8Array(10 + headerSize + values.byteLength);
  output.set([147, 78, 85, 77, 80, 89, 1, 0]);
  new DataView(output.buffer).setUint16(8, headerSize, true);
  output.fill(32, 10, 10 + headerSize);
  output.set(header, 10);
  output[9 + headerSize] = 10;
  output.set(new Uint8Array(values.buffer), 10 + headerSize);
  return output;
}
function archive(convention?: string): ArrayBuffer {
  return zipSync({
    "manifest.json": strToU8(JSON.stringify({
      schema: "boundary-lab-speaker-package", schema_version: 1,
      phasor_convention: convention, name: "test", frequencies_hz: [100],
      excitation_port_ids: ["drive"], files: { patterns: { path: "patterns.npz" } },
    })),
    "patterns.npz": zipSync({
      "frequencies_hz.npy": npy(new Float64Array([100]), [1], "<f8"),
      "directions_xyz.npy": npy(new Float32Array([0,1,0]), [1,3], "<f4"),
      "radius_m.npy": npy(new Float32Array([1]), [1], "<f4"),
      "pressure_pa.npy": npy(new Float32Array([1,2]), [1,1,1], "<c8"),
    }),
  }).buffer as ArrayBuffer;
}
for (const convention of [undefined, "exp(-i omega t)", "exp(+i omega t)"]) {
  const bytes = archive(convention);
  const original = new Uint8Array(bytes).slice();
  const result = loadSpeakerPackage(bytes, "test.blabsp");
  assert.equal(result.manifest.phasor_convention, "exp(+i omega t)");
  assert.equal(result.pressure.real[0], 1);
  assert.equal(result.pressure.imag[0], convention === "exp(+i omega t)" ? 2 : -2);
  assert.deepEqual(new Uint8Array(bytes), original);
}
assert.throws(() => loadSpeakerPackage(archive("unknown"), "test.blabsp"), /Unsupported phasor/);
console.log("Legacy and positive-time package ingestion passed.");
