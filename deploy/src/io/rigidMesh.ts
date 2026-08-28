import type { RigidMeshAsset, SpeakerMesh } from "../model/types";

function stableId(bytes: Uint8Array): string {
  let hash = 2166136261;
  for (const value of bytes) {
    hash ^= value;
    hash = Math.imul(hash, 16777619);
  }
  return `rigid-mesh-${(hash >>> 0).toString(16)}`;
}

function boundsFromMesh(mesh: SpeakerMesh): [number, number, number] {
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

export function loadRigidMesh(
  bytes: ArrayBuffer,
  fileName: string,
  sourcePath: string | null = null,
  scaleToMeters = 0.001,
): RigidMeshAsset {
  if (!Number.isFinite(scaleToMeters) || scaleToMeters <= 0) throw new Error("Rigid mesh scale must be positive.");
  const payload = new Uint8Array(bytes);
  const lines = new TextDecoder().decode(payload).split(/\r?\n/);
  const formatStart = lines.indexOf("$MeshFormat");
  if (formatStart < 0 || !lines[formatStart + 1]?.startsWith("2.2 0")) {
    throw new Error("Rigid mesh import currently requires a Gmsh 2.2 ASCII surface mesh.");
  }
  const nodeStart = lines.indexOf("$Nodes");
  const elementStart = lines.indexOf("$Elements");
  if (nodeStart < 0 || elementStart < 0) throw new Error("Rigid mesh is missing Gmsh nodes or elements.");
  const nodeCount = Number.parseInt(lines[nodeStart + 1], 10);
  const positions = new Float32Array(nodeCount * 3);
  const nodeMap = new Map<number, number>();
  for (let index = 0; index < nodeCount; index += 1) {
    const values = lines[nodeStart + 2 + index].trim().split(/\s+/).map(Number);
    if (values.length < 4 || values.some((value) => !Number.isFinite(value))) throw new Error("Rigid mesh contains an invalid node.");
    nodeMap.set(values[0], index);
    positions[index * 3] = values[1] * scaleToMeters;
    // Store in Deploy's package-like local frame so the shared cabinet
    // renderer maps conventional Gmsh Z-up coordinates to scene Y-up.
    positions[index * 3 + 1] = values[2] * scaleToMeters;
    positions[index * 3 + 2] = -values[3] * scaleToMeters;
  }
  const elementCount = Number.parseInt(lines[elementStart + 1], 10);
  const triangles: number[] = [];
  const edgeCounts = new Map<string, number>();
  for (let index = 0; index < elementCount; index += 1) {
    const values = lines[elementStart + 2 + index].trim().split(/\s+/).map(Number);
    if (values[1] !== 2) continue;
    const start = 3 + values[2];
    const face = [nodeMap.get(values[start]), nodeMap.get(values[start + 1]), nodeMap.get(values[start + 2])];
    if (face.some((value) => value === undefined)) throw new Error("Rigid mesh triangle references an unknown node.");
    const [a, b, c] = face as number[];
    if (a === b || b === c || c === a) throw new Error("Rigid mesh contains a collapsed triangle.");
    triangles.push(a, b, c);
    for (const [left, right] of [[a, b], [b, c], [c, a]]) {
      const key = left < right ? `${left}:${right}` : `${right}:${left}`;
      edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
    }
  }
  if (triangles.length === 0) throw new Error("Rigid mesh does not contain linear triangular surface elements.");
  if ([...edgeCounts.values()].some((count) => count !== 2)) {
    throw new Error("Rigid mesh must be a closed two-manifold surface; open edges were found.");
  }
  const mesh = { positions, indices: Uint32Array.from(triangles) };
  return {
    id: stableId(payload),
    name: fileName.replace(/\.msh$/i, ""),
    fileName,
    sourcePath,
    scaleToMeters,
    mesh,
    boundsM: boundsFromMesh(mesh),
    vertexCount: nodeCount,
    triangleCount: triangles.length / 3,
  };
}
