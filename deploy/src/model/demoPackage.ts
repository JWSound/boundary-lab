import type { LoadedSpeakerPackage, SpeakerMesh } from "./types";

function fibonacciDirections(count: number): Float32Array {
  const values = new Float32Array(count * 3);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let index = 0; index < count; index += 1) {
    const vertical = 1 - (2 * (index + 0.5)) / count;
    const radial = Math.sqrt(Math.max(0, 1 - vertical * vertical));
    const angle = goldenAngle * index;
    // Package frame is X right, Y forward, Z down/up depending on source convention.
    values[index * 3] = radial * Math.cos(angle);
    values[index * 3 + 1] = vertical;
    values[index * 3 + 2] = radial * Math.sin(angle);
  }
  return values;
}

function cabinetMesh(width: number, depth: number, height: number): SpeakerMesh {
  const x = width / 2;
  const y0 = -depth * 0.7;
  const y1 = depth * 0.3;
  const z = height / 2;
  const positions = new Float32Array([
    -x, y0, -z, x, y0, -z, x, y1, -z, -x, y1, -z,
    -x, y0, z, x, y0, z, x, y1, z, -x, y1, z,
  ]);
  const indices = new Uint32Array([
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
  ]);
  return { positions, indices };
}

export function createDemoPackage(): LoadedSpeakerPackage {
  const directionCount = 320;
  const frequencies = new Float64Array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
  ]);
  const directions = fibonacciDirections(directionCount);
  const real = new Float32Array(frequencies.length * directionCount);
  const imag = new Float32Array(real.length);
  const radii = new Float32Array(directionCount).fill(1);

  for (let frequencyIndex = 0; frequencyIndex < frequencies.length; frequencyIndex += 1) {
    const frequency = frequencies[frequencyIndex];
    const directionality = 0.08 + 0.25 * Math.pow(frequency / 500, 1.3);
    const response = 14 * (1 - 0.08 * Math.sin(Math.log2(frequency / 63) * Math.PI));
    const phase = -0.2 * Math.log(frequency / 20);
    for (let directionIndex = 0; directionIndex < directionCount; directionIndex += 1) {
      const forward = Math.max(-1, Math.min(1, directions[directionIndex * 3 + 1]));
      const frontWeight = Math.pow(Math.max(0, (forward + 1) / 2), 1.4);
      const magnitude = response * (1 - directionality * (1 - frontWeight));
      const offset = frequencyIndex * directionCount + directionIndex;
      real[offset] = magnitude * Math.cos(phase);
      imag[offset] = magnitude * Math.sin(phase);
    }
  }

  return {
    id: "s218bp-fallback",
    fileName: "Bundled S218BP preview",
    sourcePath: null,
    manifest: {
      schema: "boundary-lab-speaker-package",
      schema_version: 1,
      name: "S218BP",
      fidelity: "fixed",
      fidelity_level: 2,
      capabilities: [
        "complex_spherical_pattern",
        "fixed_distributed_sources",
      ],
      frequencies_hz: Array.from(frequencies),
      excitation_port_ids: ["input:full-range"],
      phasor_convention: "exp(-i omega t)",
      coordinate_system: { forward_axis: "+Y", unit: "m" },
      files: { patterns: { path: "data/patterns.npz" } },
      medium: { sound_speed_m_per_s: 343, density_kg_per_m3: 1.21 },
    },
    frequenciesHz: frequencies,
    directionsPackage: directions,
    radiiM: radii,
    pressure: { real, imag },
    pressureShape: [frequencies.length, 1, directionCount],
    mesh: cabinetMesh(1.18, 0.85, 0.72),
    boundsM: [1.18, 0.85, 0.72],
    isDemo: true,
  };
}
