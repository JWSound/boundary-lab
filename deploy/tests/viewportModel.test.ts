import assert from "node:assert/strict";
import { Mesh, MeshPhongMaterial } from "three";
import { parseViewportModel } from "../src/io/viewportModel";
import { cabinetLocalBounds } from "../src/model/cabinetPlacement";
import { minimumSourceHeightM } from "../src/model/field";
import type { LoadedSpeakerPackage } from "../src/model/types";

const model = parseViewportModel(
  "v -1 -2 -3\nv 1 2 3\nv 0 0 0\nvn 0 1 0\nusemtl wood\nf 1//1 2//1 3//1\n",
  "newmtl wood\nKd 0.5 0.2 0.1\nmap_Kd https://invalid.example/texture.png\n",
);
const child = model.object.children[0] as Mesh;
assert.deepEqual(Array.from(child.geometry.getAttribute("position").array).slice(0, 3), [-1, 3, -2]);
assert.equal(child.geometry.getAttribute("normal").getZ(0), 1);
assert.equal((child.material as MeshPhongMaterial).map, null);
assert.equal((child.material as MeshPhongMaterial).name, "wood");
const pkg = { viewportModel: model, boundsM: [2, 4, 6], mesh: {
  positions: new Float32Array([0, 0, 4]), indices: new Uint32Array(),
} } as LoadedSpeakerPackage;
assert.deepEqual(cabinetLocalBounds(pkg).minimum, [-1, -3, -2]);
assert.equal(minimumSourceHeightM(pkg), 4); // Acoustic mesh extends below the visual model.
pkg.mesh = null;
assert.equal(minimumSourceHeightM(pkg), 3);
const offset = parseViewportModel("v 0 0 -2\nv 1 0 -3\nv 0 1 -2\nf 1 2 3\n");
const offsetPackage = { ...pkg, viewportModel: offset };
assert.equal(minimumSourceHeightM(offsetPackage), -2); // Preserve an off-center origin without a fictitious acoustic box.
const second = model.object.clone(true);
second.position.x = 10;
assert.equal(model.object.position.x, 0);
console.log("Viewport model transforms, materials, bounds, and ground placement passed.");
