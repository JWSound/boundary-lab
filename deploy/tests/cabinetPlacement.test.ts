import assert from "node:assert/strict";
import { Vector3 } from "three";
import { cabinetClearanceViolations, constrainCabinetPoses, type BoundaryMeshAsset } from "../src/model/cabinetPlacement";
import { matchingCornerPaddingDirections, paddedCornerSnapDelta } from "../src/model/transformControls";
import type { SpeakerInstance } from "../src/model/types";

const width = 1.2000000476837158;
const height = 0.5389999747276306;
const depth = 0.7999999709427357;
const asset = {
  id: "s218bp-test",
  boundsM: [width, depth, height],
  mesh: null,
} as BoundaryMeshAsset;
const packages = new Map([[asset.id, asset]]);
const axes: [Vector3, Vector3, Vector3] = [
  new Vector3(1, 0, 0),
  new Vector3(0, 1, 0),
  new Vector3(0, 0, 1),
];

const directions = matchingCornerPaddingDirections([-1, -1, 1], axes, [-1, 1, 1], axes);
assert.equal(directions.length, 1);
assert.deepEqual(directions[0].toArray(), [0, 1, 0]);

const startCorner = new Vector3(2, 0, 0);
const target = {
  key: "lower:top-left-front",
  position: new Vector3(-width / 2, height / 2, depth / 2),
  objectCenter: new Vector3(),
  paddingDirections: directions,
};
const snappedDelta = paddedCornerSnapDelta(
  startCorner,
  new Vector3(2 + width / 2, height / 2, 0),
  new Vector3(),
  // Deliberately make Z dominant: geometric face matching must still choose Y.
  new Vector3(-2, height, 5),
  target,
);
assert.ok(Math.abs(startCorner.y + snappedDelta.y - (height / 2 + 0.01)) < 1e-12);

const pose = (id: string, x: number, y: number): SpeakerInstance => ({
  id,
  packageId: asset.id,
  position: [x, y, 0],
  pitchDeg: 0,
  yawDeg: 0,
  rollDeg: 0,
});
const groundY = height / 2;
const topY = groundY + height + 0.01;
const leftX = -1.600000023841858;
const rightX = leftX + width + 0.01;
const grid = [
  pose("lower-left", leftX, groundY),
  pose("lower-right", rightX, groundY),
  pose("upper-left", leftX, topY),
  pose("upper-right", rightX, topY),
];
assert.deepEqual(cabinetClearanceViolations(packages, grid), []);
const released = constrainCabinetPoses(packages, grid, [grid[3]]);
assert.deepEqual(released[0].position, grid[3].position);

console.log("cabinet placement regression tests passed");
