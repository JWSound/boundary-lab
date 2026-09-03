import { type Camera, Euler, MathUtils, type Group, type Material, Quaternion, Vector3 } from "three";

export interface TransformControlsInternal {
  axis: string | null;
  gizmo?: TransformControlsGizmoInternal;
  _gizmo?: TransformControlsGizmoInternal;
}

interface TransformControlsGizmoInternal {
  gizmo?: Record<string, Group>;
  picker?: Record<string, Group>;
}

function normalizedDegrees(value: number): number {
  return ((value + 180) % 360 + 360) % 360 - 180;
}

export function configureAxisOnlyRotation(control: unknown): TransformControlsInternal | null {
  const transform = control as TransformControlsInternal | null;
  const gizmo = transform?._gizmo ?? transform?.gizmo;
  if (!gizmo) return transform;
  for (const group of [gizmo.gizmo?.rotate, gizmo.picker?.rotate]) {
    group?.traverse((object) => {
      if (object.name !== "E" && object.name !== "XYZE") return;
      object.raycast = () => undefined;
      const material = (object as unknown as { material?: Material | Material[] }).material;
      for (const item of Array.isArray(material) ? material : material ? [material] : []) item.visible = false;
    });
  }
  return transform;
}

export function groundParallelPosition(
  start: readonly [number, number, number],
  pointerDelta: Vector3,
): [number, number, number] {
  return [start[0] + pointerDelta.x, start[1], start[2] + pointerDelta.z];
}

export function groundParallelDelta(pointerDelta: Vector3): Vector3 {
  return new Vector3(pointerDelta.x, 0, pointerDelta.z);
}

export function snapGroundParallelDelta(
  pointerDelta: Vector3,
  startCorner: Vector3,
  targetCorners: readonly Vector3[],
  snapDistanceM = 0.18,
): Vector3 {
  const groundDelta = groundParallelDelta(pointerDelta);
  const movingCorner = startCorner.clone().add(groundDelta);
  const probeCorner = startCorner.clone().add(pointerDelta);
  let result = groundDelta;
  let bestDistance = snapDistanceM;
  for (const target of targetCorners) {
    const distance = probeCorner.distanceTo(target);
    if (distance >= bestDistance) continue;
    result = groundDelta.clone().add(target.clone().sub(movingCorner));
    bestDistance = distance;
  }
  return result;
}

export interface CornerSnapTarget {
  key: string;
  position: Vector3;
  objectCenter: Vector3;
  paddingDirections?: readonly Vector3[];
}

/** Return target-face outward normals that oppose faces at the dragged corner. */
export function matchingCornerPaddingDirections(
  movingSigns: readonly [number, number, number],
  movingAxes: readonly [Vector3, Vector3, Vector3],
  targetSigns: readonly [number, number, number],
  targetAxes: readonly [Vector3, Vector3, Vector3],
): Vector3[] {
  const directions: Vector3[] = [];
  for (let movingAxis = 0; movingAxis < 3; movingAxis += 1) {
    const movingNormal = movingAxes[movingAxis].clone().multiplyScalar(movingSigns[movingAxis]);
    for (let targetAxis = 0; targetAxis < 3; targetAxis += 1) {
      const targetNormal = targetAxes[targetAxis].clone().multiplyScalar(targetSigns[targetAxis]);
      if (movingNormal.dot(targetNormal) <= -1 + 1e-8) directions.push(targetNormal);
    }
  }
  return directions;
}

export interface ScreenViewport {
  left: number;
  top: number;
  width: number;
  height: number;
}

function targetScreenDistance(
  target: CornerSnapTarget,
  pointerClientX: number,
  pointerClientY: number,
  camera: Camera,
  viewport: ScreenViewport,
): number {
  const projected = target.position.clone().project(camera);
  if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y) || projected.z < -1 || projected.z > 1) {
    return Number.POSITIVE_INFINITY;
  }
  const screenX = viewport.left + (projected.x + 1) * viewport.width / 2;
  const screenY = viewport.top + (1 - projected.y) * viewport.height / 2;
  return Math.hypot(pointerClientX - screenX, pointerClientY - screenY);
}

/**
 * Acquire a corner in screen pixels and retain it through a wider release
 * radius. The hysteresis prevents a snap from chattering as the pointer moves
 * across the acquisition boundary.
 */
export function stickyCornerSnapTarget(
  pointerClientX: number,
  pointerClientY: number,
  targets: readonly CornerSnapTarget[],
  camera: Camera,
  viewport: ScreenViewport,
  lockedKey: string | null,
  acquirePx = 14,
  releasePx = 28,
): CornerSnapTarget | null {
  if (viewport.width <= 0 || viewport.height <= 0) return null;
  if (lockedKey) {
    const locked = targets.find((target) => target.key === lockedKey);
    if (locked && targetScreenDistance(locked, pointerClientX, pointerClientY, camera, viewport) <= releasePx) {
      return locked;
    }
  }
  let closest: CornerSnapTarget | null = null;
  let closestDistance = acquirePx;
  for (const target of targets) {
    const distance = targetScreenDistance(target, pointerClientX, pointerClientY, camera, viewport);
    if (distance >= closestDistance) continue;
    closest = target;
    closestDistance = distance;
  }
  return closest;
}

/** Resolve an acquired corner to a padded face-to-face position. */
export function paddedCornerSnapDelta(
  startCorner: Vector3,
  startObjectCenter: Vector3,
  pointerDelta: Vector3,
  probeDelta: Vector3,
  target: CornerSnapTarget | null,
  paddingM = 0.01,
): Vector3 {
  const groundDelta = groundParallelDelta(pointerDelta);
  if (!target) return groundDelta;

  // The relative object-center direction identifies the face from which the
  // dragged object approaches. This is more stable than using the two corners,
  // which become coincident at the snap point.
  const sourceCenterProbe = startObjectCenter.clone().add(probeDelta);
  const separation = sourceCenterProbe.sub(target.objectCenter);
  if (target.paddingDirections?.length) {
    let direction = target.paddingDirections[0];
    let bestProjection = separation.dot(direction);
    for (const candidate of target.paddingDirections.slice(1)) {
      const projection = separation.dot(candidate);
      if (projection > bestProjection) {
        direction = candidate;
        bestProjection = projection;
      }
    }
    return target.position.clone().addScaledVector(direction, paddingM).sub(startCorner);
  }
  const components = [Math.abs(separation.x), Math.abs(separation.y), Math.abs(separation.z)];
  let axis = 0;
  if (components[1] > components[axis]) axis = 1;
  if (components[2] > components[axis]) axis = 2;
  const direction = new Vector3();
  const component = axis === 0 ? separation.x : axis === 1 ? separation.y : separation.z;
  direction.setComponent(axis, component < 0 ? -1 : 1);

  return target.position.clone().addScaledVector(direction, paddingM).sub(startCorner);
}

export function translationReadout(start: Vector3, current: Vector3, axis: string | null | undefined): string {
  const delta = current.clone().sub(start);
  const component = axis === "X" ? delta.x : axis === "Y" ? delta.y : axis === "Z" ? delta.z : null;
  if (component !== null) return `${axis} ${component >= 0 ? "+" : ""}${component.toFixed(3)} m`;
  return `Δ ${delta.length().toFixed(3)} m`;
}

export function rotationReadout(quaternion: Quaternion, axis: string | null | undefined): string {
  const rotation = new Euler().setFromQuaternion(quaternion, "YXZ");
  const degrees = axis === "X"
    ? normalizedDegrees(MathUtils.radToDeg(rotation.x))
    : axis === "Y"
      ? normalizedDegrees(MathUtils.radToDeg(rotation.y))
      : axis === "Z"
        ? normalizedDegrees(MathUtils.radToDeg(rotation.z))
        : 0;
  return `${axis === "X" || axis === "Y" || axis === "Z" ? `${axis} ` : ""}${degrees.toFixed(0)}°`;
}
