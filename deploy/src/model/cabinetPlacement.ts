import { Euler, MathUtils, Quaternion, Vector3 } from "three";
import type { LoadedSpeakerPackage, RigidMeshAsset, SpeakerInstance } from "./types";

export type BoundaryMeshAsset = LoadedSpeakerPackage | RigidMeshAsset;

export const SOURCE_SURFACE_PADDING_M = 0.01;

export interface CabinetLocalBounds {
  minimum: [number, number, number];
  maximum: [number, number, number];
  corners: Array<[number, number, number]>;
}

interface CabinetObb {
  center: Vector3;
  axes: [Vector3, Vector3, Vector3];
  halfSize: [number, number, number];
}

const localBoundsCache = new WeakMap<BoundaryMeshAsset, CabinetLocalBounds>();
const AXIS_EPSILON = 1e-10;
const POSE_EPSILON = 1e-8;

export function cabinetLocalBounds(pkg: BoundaryMeshAsset): CabinetLocalBounds {
  const cached = localBoundsCache.get(pkg);
  if (cached) return cached;
  let minimum: [number, number, number] = [-pkg.boundsM[0] / 2, -pkg.boundsM[2] / 2, -pkg.boundsM[1] / 2];
  let maximum: [number, number, number] = [pkg.boundsM[0] / 2, pkg.boundsM[2] / 2, pkg.boundsM[1] / 2];
  if (pkg.mesh) {
    minimum = [Infinity, Infinity, Infinity];
    maximum = [-Infinity, -Infinity, -Infinity];
    for (let index = 0; index < pkg.mesh.positions.length; index += 3) {
      const point = [pkg.mesh.positions[index], -pkg.mesh.positions[index + 2], pkg.mesh.positions[index + 1]];
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], point[axis]);
        maximum[axis] = Math.max(maximum[axis], point[axis]);
      }
    }
  }
  const corners: Array<[number, number, number]> = [];
  for (const x of [minimum[0], maximum[0]]) {
    for (const y of [minimum[1], maximum[1]]) {
      for (const z of [minimum[2], maximum[2]]) corners.push([x, y, z]);
    }
  }
  const result = { minimum, maximum, corners };
  localBoundsCache.set(pkg, result);
  return result;
}

function poseQuaternion(instance: SpeakerInstance): Quaternion {
  return new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(instance.pitchDeg),
    MathUtils.degToRad(instance.yawDeg),
    MathUtils.degToRad(instance.rollDeg),
    "YXZ",
  ));
}

function cabinetObb(pkg: BoundaryMeshAsset, instance: SpeakerInstance): CabinetObb {
  const bounds = cabinetLocalBounds(pkg);
  const localCenter = new Vector3(
    (bounds.minimum[0] + bounds.maximum[0]) / 2,
    (bounds.minimum[1] + bounds.maximum[1]) / 2,
    (bounds.minimum[2] + bounds.maximum[2]) / 2,
  );
  const rotation = poseQuaternion(instance);
  const position = new Vector3(...instance.position);
  return {
    center: localCenter.applyQuaternion(rotation).add(position),
    axes: [
      new Vector3(1, 0, 0).applyQuaternion(rotation),
      new Vector3(0, 1, 0).applyQuaternion(rotation),
      new Vector3(0, 0, 1).applyQuaternion(rotation),
    ],
    halfSize: [
      (bounds.maximum[0] - bounds.minimum[0]) / 2,
      (bounds.maximum[1] - bounds.minimum[1]) / 2,
      (bounds.maximum[2] - bounds.minimum[2]) / 2,
    ],
  };
}

function separatingAxes(left: CabinetObb, right: CabinetObb): Vector3[] {
  const axes = [...left.axes.map((axis) => axis.clone()), ...right.axes.map((axis) => axis.clone())];
  for (const leftAxis of left.axes) {
    for (const rightAxis of right.axes) {
      const cross = new Vector3().crossVectors(leftAxis, rightAxis);
      if (cross.lengthSq() > AXIS_EPSILON) axes.push(cross.normalize());
    }
  }
  return axes;
}

function projectionRadius(obb: CabinetObb, axis: Vector3): number {
  return obb.halfSize[0] * Math.abs(axis.dot(obb.axes[0]))
    + obb.halfSize[1] * Math.abs(axis.dot(obb.axes[1]))
    + obb.halfSize[2] * Math.abs(axis.dot(obb.axes[2]));
}

function obbsViolateClearance(left: CabinetObb, right: CabinetObb, clearanceM: number): boolean {
  const centerDelta = left.center.clone().sub(right.center);
  for (const axis of separatingAxes(left, right)) {
    const required = projectionRadius(left, axis) + projectionRadius(right, axis) + clearanceM;
    if (Math.abs(centerDelta.dot(axis)) >= required - POSE_EPSILON) return false;
  }
  return true;
}

function obbSeparation(left: CabinetObb, right: CabinetObb): { margin: number; normal: Vector3 } {
  const centerDelta = left.center.clone().sub(right.center);
  let margin = -Infinity;
  let normal = new Vector3(1, 0, 0);
  for (const axis of separatingAxes(left, right)) {
    const signedDistance = centerDelta.dot(axis);
    const axisMargin = Math.abs(signedDistance) - projectionRadius(left, axis) - projectionRadius(right, axis);
    if (axisMargin > margin) {
      margin = axisMargin;
      normal = axis.clone().multiplyScalar(signedDistance < 0 ? -1 : 1);
    }
  }
  return { margin, normal };
}

function packageFor(instance: SpeakerInstance, packages: ReadonlyMap<string, BoundaryMeshAsset>): BoundaryMeshAsset {
  const pkg = packages.get(instance.packageId);
  if (!pkg) throw new Error(`Speaker ${instance.id} references a package that is not loaded.`);
  return pkg;
}

export function cabinetClearanceViolations(
  packages: ReadonlyMap<string, BoundaryMeshAsset>,
  instances: readonly SpeakerInstance[],
  clearanceM = SOURCE_SURFACE_PADDING_M,
): Array<[string, string]> {
  const violations: Array<[string, string]> = [];
  const obbs = instances.map((instance) => cabinetObb(packageFor(instance, packages), instance));
  for (let left = 0; left < instances.length; left += 1) {
    for (let right = left + 1; right < instances.length; right += 1) {
      if (obbsViolateClearance(obbs[left], obbs[right], clearanceM)) {
        violations.push([instances[left].id, instances[right].id]);
      }
    }
  }
  return violations;
}

function firstSweptCollisionFraction(
  moving: CabinetObb,
  stationary: CabinetObb,
  delta: Vector3,
  clearanceM: number,
): { fraction: number; normal: Vector3 } | null {
  const centerDelta = moving.center.clone().sub(stationary.center);
  let entry = -Infinity;
  let exit = Infinity;
  let entryAxis = new Vector3(1, 0, 0);
  for (const axis of separatingAxes(moving, stationary)) {
    const position = centerDelta.dot(axis);
    const velocity = delta.dot(axis);
    const limit = projectionRadius(moving, axis) + projectionRadius(stationary, axis) + clearanceM;
    if (Math.abs(velocity) <= AXIS_EPSILON) {
      if (Math.abs(position) >= limit - POSE_EPSILON) return null;
      continue;
    }
    const first = (-limit - position) / velocity;
    const second = (limit - position) / velocity;
    const axisEntry = Math.min(first, second);
    if (axisEntry > entry) {
      entry = axisEntry;
      entryAxis = axis.clone();
    }
    exit = Math.min(exit, Math.max(first, second));
    if (entry > exit) return null;
  }
  if (exit <= POSE_EPSILON || entry > 1 || exit < 0) return null;
  const fraction = Math.max(0, entry);
  const contactDelta = centerDelta.addScaledVector(delta, fraction);
  if (contactDelta.dot(entryAxis) < 0) entryAxis.multiplyScalar(-1);
  return { fraction, normal: entryAxis };
}

function sameRotation(left: SpeakerInstance, right: SpeakerInstance): boolean {
  return Math.abs(left.pitchDeg - right.pitchDeg) <= POSE_EPSILON
    && Math.abs(left.yawDeg - right.yawDeg) <= POSE_EPSILON
    && Math.abs(left.rollDeg - right.rollDeg) <= POSE_EPSILON;
}

function rigidTranslation(
  current: readonly SpeakerInstance[],
  proposed: readonly SpeakerInstance[],
): Vector3 | null {
  let delta: Vector3 | null = null;
  for (const proposedInstance of proposed) {
    const currentInstance = current.find((candidate) => candidate.id === proposedInstance.id);
    if (!currentInstance || !sameRotation(currentInstance, proposedInstance)) return null;
    const currentDelta = new Vector3(...proposedInstance.position).sub(new Vector3(...currentInstance.position));
    if (delta && currentDelta.distanceToSquared(delta) > POSE_EPSILON * POSE_EPSILON) return null;
    delta = currentDelta;
  }
  return delta;
}

function normalizedAngleDegrees(value: number): number {
  return ((value + 180) % 360 + 360) % 360 - 180;
}

function interpolatePose(current: SpeakerInstance, proposed: SpeakerInstance, fraction: number): SpeakerInstance {
  const rotation = poseQuaternion(current).slerp(poseQuaternion(proposed), fraction);
  const euler = new Euler().setFromQuaternion(rotation, "YXZ");
  return {
    ...proposed,
    position: [
      current.position[0] + (proposed.position[0] - current.position[0]) * fraction,
      current.position[1] + (proposed.position[1] - current.position[1]) * fraction,
      current.position[2] + (proposed.position[2] - current.position[2]) * fraction,
    ],
    pitchDeg: normalizedAngleDegrees(MathUtils.radToDeg(euler.x)),
    yawDeg: normalizedAngleDegrees(MathUtils.radToDeg(euler.y)),
    rollDeg: normalizedAngleDegrees(MathUtils.radToDeg(euler.z)),
  };
}

function movingPosesViolate(
  packages: ReadonlyMap<string, BoundaryMeshAsset>,
  moving: readonly SpeakerInstance[],
  stationary: readonly SpeakerInstance[],
  clearanceM: number,
): boolean {
  const movingObbs = moving.map((instance) => cabinetObb(packageFor(instance, packages), instance));
  const stationaryObbs = stationary.map((instance) => cabinetObb(packageFor(instance, packages), instance));
  for (const movingObb of movingObbs) {
    if (stationaryObbs.some((stationaryObb) => obbsViolateClearance(movingObb, stationaryObb, clearanceM))) return true;
  }
  return false;
}

export function constrainCabinetPoses(
  packages: ReadonlyMap<string, BoundaryMeshAsset>,
  current: readonly SpeakerInstance[],
  proposedMoving: readonly SpeakerInstance[],
  clearanceM = SOURCE_SURFACE_PADDING_M,
): SpeakerInstance[] {
  if (proposedMoving.length === 0) return [];
  const movingIds = new Set(proposedMoving.map((instance) => instance.id));
  const currentMoving = current.filter((instance) => movingIds.has(instance.id));
  const stationary = current.filter((instance) => !movingIds.has(instance.id));
  if (currentMoving.length !== proposedMoving.length) throw new Error("Every proposed speaker pose must have a current scene instance.");

  const translation = rigidTranslation(currentMoving, proposedMoving);
  if (translation) {
    const resolvedDelta = new Vector3();
    let remainingDelta = translation.clone();
    const stationaryObbs = stationary.map((instance) => cabinetObb(packageFor(instance, packages), instance));
    const movingObbs = currentMoving.map((instance) => cabinetObb(packageFor(instance, packages), instance));
    for (let iteration = 0; iteration < 4 && remainingDelta.lengthSq() > POSE_EPSILON * POSE_EPSILON; iteration += 1) {
      let collisionFraction = 1;
      let collisionNormal: Vector3 | null = null;
      for (const movingObb of movingObbs) {
        for (const stationaryObb of stationaryObbs) {
          if (obbsViolateClearance(movingObb, stationaryObb, clearanceM)) {
            const startSeparation = obbSeparation(movingObb, stationaryObb);
            const endObb = { ...movingObb, center: movingObb.center.clone().add(remainingDelta) };
            if (obbSeparation(endObb, stationaryObb).margin > startSeparation.margin + POSE_EPSILON) continue;
            collisionFraction = 0;
            collisionNormal = startSeparation.normal;
            continue;
          }
          const collision = firstSweptCollisionFraction(movingObb, stationaryObb, remainingDelta, clearanceM);
          if (collision && collision.fraction < collisionFraction) {
            collisionFraction = collision.fraction;
            collisionNormal = collision.normal;
          }
        }
      }
      const travelFraction = collisionNormal ? Math.max(0, collisionFraction - 1e-9) : 1;
      const travel = remainingDelta.clone().multiplyScalar(travelFraction);
      resolvedDelta.add(travel);
      movingObbs.forEach((obb) => obb.center.add(travel));
      if (!collisionNormal) break;
      remainingDelta.multiplyScalar(1 - collisionFraction);
      const inwardDistance = remainingDelta.dot(collisionNormal);
      if (inwardDistance < 0) remainingDelta.addScaledVector(collisionNormal, -inwardDistance);
    }
    return proposedMoving.map((proposed) => {
      const start = currentMoving.find((instance) => instance.id === proposed.id)!;
      return { ...proposed, position: new Vector3(...start.position).add(resolvedDelta).toArray() as [number, number, number] };
    });
  }

  let maximumAngleDeg = 0;
  for (const proposed of proposedMoving) {
    const start = currentMoving.find((instance) => instance.id === proposed.id)!;
    maximumAngleDeg = Math.max(maximumAngleDeg, MathUtils.radToDeg(poseQuaternion(start).angleTo(poseQuaternion(proposed))));
  }
  const steps = Math.max(4, Math.min(90, Math.ceil(maximumAngleDeg / 2)));
  let validFraction = 0;
  let invalidFraction: number | null = null;
  for (let step = 1; step <= steps; step += 1) {
    const fraction = step / steps;
    const candidates = proposedMoving.map((proposed) => interpolatePose(
      currentMoving.find((instance) => instance.id === proposed.id)!,
      proposed,
      fraction,
    ));
    if (movingPosesViolate(packages, candidates, stationary, clearanceM)) {
      invalidFraction = fraction;
      break;
    }
    validFraction = fraction;
  }
  if (invalidFraction === null) return proposedMoving.map((instance) => ({ ...instance, position: [...instance.position] }));
  let upperFraction: number = invalidFraction;
  for (let iteration = 0; iteration < 24; iteration += 1) {
    const fraction: number = (validFraction + upperFraction) / 2;
    const candidates = proposedMoving.map((proposed) => interpolatePose(
      currentMoving.find((instance) => instance.id === proposed.id)!,
      proposed,
      fraction,
    ));
    if (movingPosesViolate(packages, candidates, stationary, clearanceM)) upperFraction = fraction;
    else validFraction = fraction;
  }
  return proposedMoving.map((proposed) => interpolatePose(
    currentMoving.find((instance) => instance.id === proposed.id)!,
    proposed,
    validFraction,
  ));
}

export function findClearSourcePlacement(
  packages: ReadonlyMap<string, BoundaryMeshAsset>,
  existing: readonly SpeakerInstance[],
  proposed: SpeakerInstance,
  clearanceM = SOURCE_SURFACE_PADDING_M,
): SpeakerInstance {
  if (!movingPosesViolate(packages, [proposed], existing, clearanceM)) return proposed;
  const bounds = cabinetLocalBounds(packageFor(proposed, packages));
  const step = Math.max(bounds.maximum[0] - bounds.minimum[0], bounds.maximum[2] - bounds.minimum[2]) + clearanceM;
  for (let ring = 1; ring <= 100; ring += 1) {
    for (const [x, z] of [[ring, 0], [ring, ring], [0, ring], [-ring, ring], [-ring, 0], [-ring, -ring], [0, -ring], [ring, -ring]]) {
      const candidate = { ...proposed, position: [proposed.position[0] + x * step, proposed.position[1], proposed.position[2] + z * step] as [number, number, number] };
      if (!movingPosesViolate(packages, [candidate], existing, clearanceM)) return candidate;
    }
  }
  throw new Error(`Could not find a collision-free placement for speaker ${proposed.id}.`);
}
