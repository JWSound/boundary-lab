import { BufferAttribute, Group, Mesh } from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { MTLLoader } from "three/examples/jsm/loaders/MTLLoader.js";
import type { SpeakerMesh } from "../model/types";

export interface ViewportModel {
  object: Group;
  mesh: SpeakerMesh;
}

export function parseViewportModel(obj: string, mtl?: string): ViewportModel {
  const loader = new OBJLoader();
  if (mtl) {
    // Packages support color materials only. Never let an asset fetch external textures.
    const colors = mtl.split(/\r?\n/).filter((line) =>
      !/^(map_\S+|bump|disp|decal|refl|norm)\s/i.test(line.trim()),
    ).join("\n");
    loader.setMaterials(new MTLLoader().parse(colors, ""));
  }
  const object = loader.parse(obj);
  const positions: number[] = [];
  object.traverse((child) => {
    if (!(child instanceof Mesh)) return;
    const geometry = child.geometry;
    const points = geometry.getAttribute("position") as BufferAttribute;
    for (let i = 0; i < points.count; i += 1) {
      const x = points.getX(i), y = points.getY(i), z = points.getZ(i);
      if (![x, y, z].every(Number.isFinite)) throw new Error("Viewport model contains invalid coordinates.");
      positions.push(x, y, z);
      // Package +Y forward to scene +Z forward, matching the acoustic mesh.
      points.setXYZ(i, x, -z, y);
    }
    const normals = geometry.getAttribute("normal");
    if (normals) {
      for (let i = 0; i < normals.count; i += 1) {
        const y = normals.getY(i), z = normals.getZ(i);
        normals.setXYZ(i, normals.getX(i), -z, y);
      }
    } else geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    child.castShadow = child.receiveShadow = true;
  });
  if (!positions.length) throw new Error("Viewport model contains no surface geometry.");
  return { object, mesh: { positions: Float32Array.from(positions), indices: new Uint32Array() } };
}
