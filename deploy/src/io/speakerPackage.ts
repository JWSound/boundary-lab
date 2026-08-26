import { strFromU8, unzipSync } from "fflate";
import { asComplexFloat32, asFloat32, asFloat64, parseNpy, type NpyArray } from "./npy";
import type { LoadedSpeakerPackage, SpeakerMesh, SpeakerPackageManifest } from "../model/types";

function npzEntries(bytes: Uint8Array): Record<string, NpyArray> {
  const files = unzipSync(bytes);
  return Object.fromEntries(
    Object.entries(files)
      .filter(([name]) => name.endsWith(".npy"))
      .map(([name, payload]) => [name.replace(/\.npy$/, ""), parseNpy(payload)]),
  );
}

function parseGmshSurface(text: string): SpeakerMesh | null {
  const lines = text.split(/\r?\n/);
  const nodeStart = lines.indexOf("$Nodes");
  const elementStart = lines.indexOf("$Elements");
  if (nodeStart < 0 || elementStart < 0) return null;
  const nodeCount = Number.parseInt(lines[nodeStart + 1], 10);
  const positions = new Float32Array(nodeCount * 3);
  const nodeMap = new Map<number, number>();
  for (let index = 0; index < nodeCount; index += 1) {
    const values = lines[nodeStart + 2 + index].trim().split(/\s+/).map(Number);
    nodeMap.set(values[0], index);
    positions[index * 3] = values[1];
    positions[index * 3 + 1] = values[2];
    positions[index * 3 + 2] = values[3];
  }
  const elementCount = Number.parseInt(lines[elementStart + 1], 10);
  const triangles: number[] = [];
  for (let index = 0; index < elementCount; index += 1) {
    const values = lines[elementStart + 2 + index].trim().split(/\s+/).map(Number);
    if (values[1] !== 2) continue;
    const tagCount = values[2];
    const start = 3 + tagCount;
    const a = nodeMap.get(values[start]);
    const b = nodeMap.get(values[start + 1]);
    const c = nodeMap.get(values[start + 2]);
    if (a !== undefined && b !== undefined && c !== undefined) triangles.push(a, b, c);
  }
  return triangles.length ? { positions, indices: Uint32Array.from(triangles) } : null;
}

function boundsFromMesh(mesh: SpeakerMesh | null): [number, number, number] {
  if (!mesh || mesh.positions.length < 3) return [0.5, 0.5, 0.35];
  const low = [Infinity, Infinity, Infinity];
  const high = [-Infinity, -Infinity, -Infinity];
  for (let index = 0; index < mesh.positions.length; index += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      low[axis] = Math.min(low[axis], mesh.positions[index + axis]);
      high[axis] = Math.max(high[axis], mesh.positions[index + axis]);
    }
  }
  return [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
}

function stableId(name: string, manifest: SpeakerPackageManifest): string {
  const source = `${name}:${manifest.name}:${manifest.frequencies_hz.length}:${manifest.fidelity_level}`;
  let hash = 2166136261;
  for (let index = 0; index < source.length; index += 1) {
    hash ^= source.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `package-${(hash >>> 0).toString(16)}`;
}

export function loadSpeakerPackage(bytes: ArrayBuffer, fileName: string): LoadedSpeakerPackage {
  const files = unzipSync(new Uint8Array(bytes));
  const manifestPayload = files["manifest.json"];
  if (!manifestPayload) throw new Error("Speaker package is missing manifest.json.");
  const manifest = JSON.parse(strFromU8(manifestPayload)) as SpeakerPackageManifest;
  if (manifest.schema !== "boundary-lab-speaker-package" || manifest.schema_version !== 1) {
    throw new Error("This prototype supports Boundary Lab speaker package schema version 1.");
  }
  if (manifest.phasor_convention !== "exp(-i omega t)") {
    throw new Error(`Unsupported phasor convention ${manifest.phasor_convention}.`);
  }
  const patternPath = manifest.files.patterns?.path;
  if (!patternPath || !files[patternPath]) throw new Error("Speaker package does not contain its pattern payload.");
  const pattern = npzEntries(files[patternPath]);
  const frequencies = asFloat64(pattern.frequencies_hz);
  const directions = asFloat32(pattern.directions_xyz);
  const radii = asFloat32(pattern.radius_m);
  const pressureArray = pattern.pressure_pa;
  const pressure = asComplexFloat32(pressureArray);
  if (pressureArray.shape.length !== 3) throw new Error("Pattern pressure must have frequency, excitation, and direction axes.");

  const geometryPath = manifest.files.fixed_sources?.geometry_mesh;
  const mesh = typeof geometryPath === "string" && files[geometryPath]
    ? parseGmshSurface(strFromU8(files[geometryPath]))
    : null;

  return {
    id: stableId(fileName, manifest),
    fileName,
    manifest,
    frequenciesHz: frequencies,
    directionsPackage: directions,
    radiiM: radii,
    pressure,
    pressureShape: pressureArray.shape as [number, number, number],
    mesh,
    boundsM: boundsFromMesh(mesh),
    isDemo: false,
  };
}
