import { Grid, Html, OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type MutableRefObject } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  ClampToEdgeWrapping,
  DataTexture,
  DoubleSide,
  Euler,
  FloatType,
  Group,
  NearestFilter,
  MathUtils,
  Plane,
  Quaternion,
  Ray,
  RedFormat,
  ShaderMaterial,
  UnsignedByteType,
  Vector2,
  Vector3,
} from "three";
import type { FieldFrame, LoadedSpeakerPackage, ObservationPlane, SpeakerInstance } from "../model/types";
import { SOURCE_GROUND_CLEARANCE_M, SOURCE_SURFACE_PADDING_M } from "../model/field";

export type SceneTransformMode = "select" | "translate" | "rotate" | "scale";

export interface SourcePoseUpdate {
  positionX: number;
  positionHeightM: number;
  positionZ: number;
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
}

export interface SourceGroupPoseUpdate extends SourcePoseUpdate {
  id: string;
}

export type ObservationPoseUpdate = Pick<ObservationPlane, "centerXM" | "nearM" | "heightM" | "pitchDeg" | "yawDeg" | "rollDeg">;

export interface ObservationResizeUpdate {
  widthM: number;
  depthM: number;
  centerXM: number;
  centerZM: number;
  heightM: number;
}

interface SceneBounds {
  minimum: [number, number, number];
  maximum: [number, number, number];
  corners: Array<[number, number, number]>;
}

interface SceneViewProps {
  pkg: LoadedSpeakerPackage;
  sources: SpeakerInstance[];
  observation: ObservationPlane;
  field: FieldFrame;
  selectedInstances: readonly string[];
  activeInstance: string | null;
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onSelectInstance: (id: string | null, additive?: boolean) => void;
  onTransformSource: (id: string, pose: SourcePoseUpdate) => void;
  onTransformSources: (poses: SourceGroupPoseUpdate[]) => void;
  onTransformObservation: (pose: ObservationPoseUpdate) => void;
  onResizeObservation: (resize: ObservationResizeUpdate) => void;
  onManipulationEnd: () => void;
  onFieldTextureReady?: (profile: FieldTextureProfile) => void;
}

export interface FieldTextureProfile {
  pointCount: number;
  textureBytes: number;
  rasterMs: number;
  commitToFrameMs: number;
}

function packageSceneBounds(pkg: LoadedSpeakerPackage): SceneBounds {
  let minimum: [number, number, number] = [-pkg.boundsM[0] / 2, -pkg.boundsM[2] / 2, -pkg.boundsM[1] / 2];
  let maximum: [number, number, number] = [pkg.boundsM[0] / 2, pkg.boundsM[2] / 2, pkg.boundsM[1] / 2];
  if (pkg.mesh) {
    minimum = [Infinity, Infinity, Infinity];
    maximum = [-Infinity, -Infinity, -Infinity];
    for (let index = 0; index < pkg.mesh.positions.length; index += 3) {
      const point: [number, number, number] = [
        pkg.mesh.positions[index],
        -pkg.mesh.positions[index + 2],
        pkg.mesh.positions[index + 1],
      ];
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
  return { minimum, maximum, corners };
}

function cornerInWorld(
  corner: [number, number, number],
  instance: SpeakerInstance,
  position: [number, number, number] = instance.position,
): Vector3 {
  const rotation = new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(instance.pitchDeg),
    MathUtils.degToRad(instance.yawDeg),
    MathUtils.degToRad(instance.rollDeg),
    "YXZ",
  ));
  return new Vector3(...corner).applyQuaternion(rotation).add(new Vector3(...position));
}

function normalizedYaw(yawDeg: number): number {
  return ((yawDeg + 180) % 360 + 360) % 360 - 180;
}

function finiteVector(vector: Vector3): boolean {
  return Number.isFinite(vector.x) && Number.isFinite(vector.y) && Number.isFinite(vector.z);
}

const FIELD_PLANE_VERTEX_SHADER = /* glsl */ `
  varying vec2 vUv;

  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FIELD_PLANE_FRAGMENT_SHADER = /* glsl */ `
  uniform sampler2D uSplMap;
  uniform sampler2D uValidityMap;
  uniform vec2 uTextureSize;
  uniform float uMinimumDb;
  uniform float uMaximumDb;
  uniform float uBandingDb;
  varying vec2 vUv;

  float validityAt(vec2 uv) {
    // The CPU mask stores 0 or 1 in an unsigned normalized texture.
    return step(0.001, texture2D(uValidityMap, uv).r);
  }

  float nearestValidity(vec2 uv) {
    vec2 nearestIndex = floor(uv * (uTextureSize - 1.0) + 0.5);
    return validityAt((nearestIndex + 0.5) / uTextureSize);
  }

  float filteredSpl(vec2 uv) {
    // Perform mask-aware bilinear interpolation. Invalid ground-clipped samples
    // contribute no weight, preventing a dark fringe along the clipping edge.
    // Grid samples include both physical plane edges, hence size - 1 here.
    vec2 samplePosition = uv * (uTextureSize - 1.0);
    vec2 base = floor(samplePosition);
    vec2 fraction = fract(samplePosition);
    vec2 maximumIndex = uTextureSize - 1.0;
    vec2 uv00 = (clamp(base, vec2(0.0), maximumIndex) + 0.5) / uTextureSize;
    vec2 uv10 = (clamp(base + vec2(1.0, 0.0), vec2(0.0), maximumIndex) + 0.5) / uTextureSize;
    vec2 uv01 = (clamp(base + vec2(0.0, 1.0), vec2(0.0), maximumIndex) + 0.5) / uTextureSize;
    vec2 uv11 = (clamp(base + vec2(1.0), vec2(0.0), maximumIndex) + 0.5) / uTextureSize;
    vec4 weights = vec4(
      (1.0 - fraction.x) * (1.0 - fraction.y),
      fraction.x * (1.0 - fraction.y),
      (1.0 - fraction.x) * fraction.y,
      fraction.x * fraction.y
    );
    vec4 validity = vec4(
      validityAt(uv00), validityAt(uv10), validityAt(uv01), validityAt(uv11)
    );
    vec4 validWeights = weights * validity;
    float weightSum = dot(validWeights, vec4(1.0));
    vec4 samples = vec4(
      texture2D(uSplMap, uv00).r,
      texture2D(uSplMap, uv10).r,
      texture2D(uSplMap, uv01).r,
      texture2D(uSplMap, uv11).r
    );
    return weightSum > 0.0001 ? dot(samples, validWeights) / weightSum : uMinimumDb;
  }

  float colorPosition(float valueDb) {
    float rangeDb = max(0.0001, uMaximumDb - uMinimumDb);
    if (uBandingDb < 0.5) return clamp((valueDb - uMinimumDb) / rangeDb, 0.0, 1.0);

    float firstBoundary = ceil(uMinimumDb / uBandingDb) * uBandingDb;
    if (firstBoundary <= uMinimumDb + 0.0001) firstBoundary += uBandingDb;
    float internalBoundaryCount = max(0.0, ceil((uMaximumDb - firstBoundary) / uBandingDb));
    float regionCount = internalBoundaryCount + 1.0;
    if (regionCount < 1.5) return 0.5;
    float region = valueDb < firstBoundary
      ? 0.0
      : floor((valueDb - firstBoundary) / uBandingDb) + 1.0;
    return clamp(region / (regionCount - 1.0), 0.0, 1.0);
  }

  vec3 palette(float position) {
    float scaled = clamp(position, 0.0, 1.0) * 9.0;
    if (scaled < 1.0) return mix(vec3(0.0, 0.0, 0.5608), vec3(0.0, 0.0, 1.0), scaled);
    if (scaled < 2.0) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 0.4353, 1.0), scaled - 1.0);
    if (scaled < 3.0) return mix(vec3(0.0, 0.4353, 1.0), vec3(0.0, 0.8745, 1.0), scaled - 2.0);
    if (scaled < 4.0) return mix(vec3(0.0, 0.8745, 1.0), vec3(0.3098, 1.0, 0.7490), scaled - 3.0);
    if (scaled < 5.0) return mix(vec3(0.3098, 1.0, 0.7490), vec3(0.7490, 1.0, 0.3098), scaled - 4.0);
    if (scaled < 6.0) return mix(vec3(0.7490, 1.0, 0.3098), vec3(1.0, 0.8745, 0.0), scaled - 5.0);
    if (scaled < 7.0) return mix(vec3(1.0, 0.8745, 0.0), vec3(1.0, 0.4353, 0.0), scaled - 6.0);
    if (scaled < 8.0) return mix(vec3(1.0, 0.4353, 0.0), vec3(1.0, 0.0, 0.0), scaled - 7.0);
    return mix(vec3(1.0, 0.0, 0.0), vec3(0.5608, 0.0, 0.0), scaled - 8.0);
  }

  vec3 srgbToLinear(vec3 value) {
    vec3 low = value / 12.92;
    vec3 high = pow((value + 0.055) / 1.055, vec3(2.4));
    return mix(low, high, step(vec3(0.04045), value));
  }

  void main() {
    // Keep the clipping boundary discrete even though valid SPL values are smooth.
    if (nearestValidity(vUv) < 0.5) discard;
    vec3 color = srgbToLinear(palette(colorPosition(filteredSpl(vUv))));
    gl_FragColor = vec4(color, 0.9412);
    #include <colorspace_fragment>
  }
`;

function FieldPlane({
  observation,
  field,
  selected,
  active,
  transformMode,
  angleSnapDisabled,
  onTransform,
  onResize,
  onManipulationEnd,
  onTextureReady,
}: {
  observation: ObservationPlane;
  field: FieldFrame;
  selected: boolean;
  active: boolean;
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onTransform: (pose: ObservationPoseUpdate) => void;
  onResize: (resize: ObservationResizeUpdate) => void;
  onManipulationEnd: () => void;
  onTextureReady?: (profile: FieldTextureProfile) => void;
}) {
  const planeRef = useRef<Group>(null);
  const orbitControls = useThree((state) => state.controls) as { enabled: boolean } | null;
  const resizeState = useRef<{
    pointerId: number;
    plane: Plane;
    origin: Vector3;
    quaternion: Quaternion;
    inverseQuaternion: Quaternion;
    oppositeLocal: Vector3;
    signX: number;
    signZ: number;
  } | null>(null);
  const textureProfile = useRef<Omit<FieldTextureProfile, "commitToFrameMs"> | null>(null);

  const cancelResize = (flushSolve = false) => {
    const wasResizing = resizeState.current !== null;
    resizeState.current = null;
    orbitControls && (orbitControls.enabled = true);
    if (flushSolve && wasResizing) onManipulationEnd();
  };

  useEffect(() => {
    const finish = () => cancelResize(true);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    window.addEventListener("blur", finish);
    return () => {
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      window.removeEventListener("blur", finish);
    };
  }, [onManipulationEnd, orbitControls]);

  useEffect(() => {
    if (transformMode !== "scale") cancelResize();
  }, [transformMode]);
  const textures = useMemo(() => {
    const rasterStarted = performance.now();
    const spl = new DataTexture(field.splDb, field.columns, field.rows, RedFormat, FloatType);
    spl.minFilter = NearestFilter;
    spl.magFilter = NearestFilter;
    spl.wrapS = ClampToEdgeWrapping;
    spl.wrapT = ClampToEdgeWrapping;
    spl.generateMipmaps = false;
    spl.needsUpdate = true;
    const validity = new DataTexture(field.validMask, field.columns, field.rows, RedFormat, UnsignedByteType);
    validity.minFilter = NearestFilter;
    validity.magFilter = NearestFilter;
    validity.wrapS = ClampToEdgeWrapping;
    validity.wrapT = ClampToEdgeWrapping;
    validity.generateMipmaps = false;
    validity.needsUpdate = true;
    textureProfile.current = {
      pointCount: field.columns * field.rows,
      textureBytes: field.splDb.byteLength + field.validMask.byteLength,
      rasterMs: performance.now() - rasterStarted,
    };
    return { spl, validity };
  }, [field]);
  const heatmapMaterial = useMemo(() => new ShaderMaterial({
    uniforms: {
      uSplMap: { value: textures.spl },
      uValidityMap: { value: textures.validity },
      uTextureSize: { value: new Vector2(field.columns, field.rows) },
      uMinimumDb: { value: observation.heatmapMinimumDb },
      uMaximumDb: { value: observation.heatmapMaximumDb },
      uBandingDb: { value: observation.heatmapBandingDb },
    },
    vertexShader: FIELD_PLANE_VERTEX_SHADER,
    fragmentShader: FIELD_PLANE_FRAGMENT_SHADER,
    transparent: true,
    side: DoubleSide,
    toneMapped: false,
  }), [field.columns, field.rows, textures]);
  heatmapMaterial.uniforms.uMinimumDb.value = observation.heatmapMinimumDb;
  heatmapMaterial.uniforms.uMaximumDb.value = observation.heatmapMaximumDb;
  heatmapMaterial.uniforms.uBandingDb.value = observation.heatmapBandingDb;

  useEffect(() => {
    const profile = textureProfile.current;
    const committedAt = performance.now();
    const frame = requestAnimationFrame(() => {
      if (profile) onTextureReady?.({
        ...profile,
        commitToFrameMs: performance.now() - committedAt,
      });
    });
    return () => {
      cancelAnimationFrame(frame);
    };
  }, [onTextureReady, textures]);

  useEffect(() => () => {
    textures.spl.dispose();
    textures.validity.dispose();
  }, [textures]);

  useEffect(() => () => heatmapMaterial.dispose(), [heatmapMaterial]);

  const applyGizmoTransform = () => {
    const object = planeRef.current;
    if (!object || !finiteVector(object.position)) return;
    const rotation = new Euler().setFromQuaternion(object.quaternion, "YXZ");
    const pitchDeg = normalizedYaw(MathUtils.radToDeg(rotation.x));
    onTransform({
      centerXM: object.position.x,
      nearM: object.position.z - observation.depthM / 2,
      heightM: object.position.y,
      pitchDeg,
      yawDeg: normalizedYaw(MathUtils.radToDeg(rotation.y)),
      rollDeg: normalizedYaw(MathUtils.radToDeg(rotation.z)),
    });
  };

  const startResize = (event: ThreeEvent<PointerEvent>, signX: number, signZ: number) => {
    if (event.nativeEvent.button !== 0 || !planeRef.current) return;
    event.stopPropagation();
    const normal = new Vector3(0, 1, 0).applyQuaternion(planeRef.current.quaternion);
    const plane = new Plane().setFromNormalAndCoplanarPoint(normal, planeRef.current.position);
    if (!event.ray.intersectPlane(plane, new Vector3())) return;
    const quaternion = planeRef.current.quaternion.clone();
    resizeState.current = {
      pointerId: event.pointerId,
      plane,
      origin: planeRef.current.position.clone(),
      quaternion,
      inverseQuaternion: quaternion.clone().invert(),
      oppositeLocal: new Vector3(-signX * observation.widthM / 2, 0, -signZ * observation.depthM / 2),
      signX,
      signZ,
    };
    orbitControls && (orbitControls.enabled = false);
    (event.target as unknown as { setPointerCapture?: (pointerId: number) => void }).setPointerCapture?.(event.pointerId);
  };

  const moveResize = (event: ThreeEvent<PointerEvent>) => {
    const resize = resizeState.current;
    const object = planeRef.current;
    if (!resize || !object || resize.pointerId !== event.pointerId) return;
    if ((event.nativeEvent.buttons & 1) === 0) {
      cancelResize(true);
      return;
    }
    event.stopPropagation();
    const point = event.ray.intersectPlane(resize.plane, new Vector3());
    if (!point) return;
    const local = point.sub(resize.origin).applyQuaternion(resize.inverseQuaternion);
    const widthM = Math.max(0.1, resize.signX * (local.x - resize.oppositeLocal.x));
    const depthM = Math.max(0.1, resize.signZ * (local.z - resize.oppositeLocal.z));
    const draggedLocal = new Vector3(
      resize.oppositeLocal.x + resize.signX * widthM,
      0,
      resize.oppositeLocal.z + resize.signZ * depthM,
    );
    const centerWorld = draggedLocal.add(resize.oppositeLocal).multiplyScalar(0.5)
      .applyQuaternion(resize.quaternion)
      .add(resize.origin);
    onResize({ widthM, depthM, centerXM: centerWorld.x, centerZM: centerWorld.z, heightM: centerWorld.y });
  };

  const finishResize = (event: ThreeEvent<PointerEvent>) => {
    if (!resizeState.current || resizeState.current.pointerId !== event.pointerId) return;
    event.stopPropagation();
    cancelResize(true);
    (event.target as unknown as { releasePointerCapture?: (pointerId: number) => void }).releasePointerCapture?.(event.pointerId);
  };

  const planeQuaternion = useMemo(() => new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(observation.pitchDeg),
    MathUtils.degToRad(observation.yawDeg),
    MathUtils.degToRad(observation.rollDeg),
    "YXZ",
  )), [observation.pitchDeg, observation.rollDeg, observation.yawDeg]);

  return (
    <>
      <group
        ref={planeRef}
        position={[observation.centerXM, observation.heightM, observation.nearM + observation.depthM / 2]}
        quaternion={planeQuaternion}
      >
        <mesh
          // DataTexture row zero is the near edge of the computed field. A +90°
          // rotation maps the plane's lower V edge toward the source (-scene Z).
          rotation={[Math.PI / 2, 0, 0]}
          raycast={() => undefined}
        >
          <planeGeometry args={[observation.widthM, observation.depthM]} />
          <primitive object={heatmapMaterial} attach="material" />
        </mesh>
        <Grid
          position={[0, 0.008, 0]}
          args={[observation.widthM, observation.depthM]}
          cellSize={1}
          cellThickness={0.35}
          cellColor={selected ? "#dce9a7" : "#d9e0d0"}
          sectionSize={5}
          sectionThickness={0.8}
          sectionColor={selected ? "#c6d68d" : "#f1f6ea"}
          fadeDistance={40}
          fadeStrength={1}
          infiniteGrid={false}
          raycast={() => undefined}
        />
        {active && transformMode === "scale" && [
          { signX: -1, signZ: -1 },
          { signX: 1, signZ: -1 },
          { signX: -1, signZ: 1 },
          { signX: 1, signZ: 1 },
        ].map(({ signX, signZ }, index) => (
          <mesh
            key={index}
            position={[signX * observation.widthM / 2, 0.025, signZ * observation.depthM / 2]}
            renderOrder={20}
            onPointerDown={(event) => startResize(event, signX, signZ)}
            onPointerMove={moveResize}
            onPointerUp={finishResize}
            onPointerCancel={finishResize}
          >
            <sphereGeometry args={[0.14, 14, 10]} />
            <meshBasicMaterial color="#dce9a7" depthTest={false} toneMapped={false} />
          </mesh>
        ))}
      </group>
      {active && (transformMode === "translate" || transformMode === "rotate") && (
        <TransformControls
          object={planeRef as MutableRefObject<Group>}
          mode={transformMode}
          space="world"
          size={0.82}
          showX
          showY
          showZ
          rotationSnap={transformMode === "rotate" && !angleSnapDisabled ? MathUtils.degToRad(5) : null}
          onObjectChange={applyGizmoTransform}
          onMouseUp={onManipulationEnd}
        />
      )}
    </>
  );
}

function SpeakerGeometry({
  pkg,
  instance,
  allInstances,
  selected,
  active,
  individualControls,
  movingInstanceIds,
  transformMode,
  angleSnapDisabled,
  onSelect,
  onTransform,
  onManipulationEnd,
}: {
  pkg: LoadedSpeakerPackage;
  instance: SpeakerInstance;
  allInstances: SpeakerInstance[];
  selected: boolean;
  active: boolean;
  individualControls: boolean;
  movingInstanceIds: readonly string[];
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onSelect: (additive: boolean) => void;
  onTransform: (pose: SourcePoseUpdate) => void;
  onManipulationEnd: () => void;
}) {
  const speakerRef = useRef<Group>(null);
  const orbitControls = useThree((state) => state.controls) as { enabled: boolean } | null;
  const camera = useThree((state) => state.camera);
  const dragState = useRef<{
    pointerId: number;
    plane: Plane;
    startPoint: Vector3;
    startPosition: [number, number, number];
    corner: [number, number, number];
  } | null>(null);
  const cancelCornerDrag = (flushSolve = false) => {
    const wasDragging = dragState.current !== null;
    dragState.current = null;
    orbitControls && (orbitControls.enabled = true);
    if (flushSolve && wasDragging) onManipulationEnd();
  };

  useEffect(() => {
    const finish = () => cancelCornerDrag(true);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    window.addEventListener("blur", finish);
    return () => {
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      window.removeEventListener("blur", finish);
    };
  }, [onManipulationEnd, orbitControls]);
  const geometry = useMemo(() => {
    if (!pkg.mesh) return null;
    const converted = new Float32Array(pkg.mesh.positions.length);
    for (let index = 0; index < pkg.mesh.positions.length; index += 3) {
      converted[index] = pkg.mesh.positions[index];
      converted[index + 1] = -pkg.mesh.positions[index + 2];
      converted[index + 2] = pkg.mesh.positions[index + 1];
    }
    const result = new BufferGeometry();
    result.setAttribute("position", new BufferAttribute(converted, 3));
    result.setIndex(new BufferAttribute(pkg.mesh.indices, 1));
    result.computeVertexNormals();
    return result;
  }, [pkg]);
  const quaternion = useMemo(
    () => new Quaternion().setFromEuler(
      new Euler(
        MathUtils.degToRad(instance.pitchDeg),
        MathUtils.degToRad(instance.yawDeg),
        MathUtils.degToRad(instance.rollDeg),
        "YXZ",
      ),
    ),
    [instance.pitchDeg, instance.rollDeg, instance.yawDeg],
  );
  const bounds = useMemo(() => packageSceneBounds(pkg), [pkg]);
  const sceneBounds: [number, number, number] = [
    bounds.maximum[0] - bounds.minimum[0],
    bounds.maximum[1] - bounds.minimum[1],
    bounds.maximum[2] - bounds.minimum[2],
  ];
  const boundsCenter: [number, number, number] = [
    (bounds.minimum[0] + bounds.maximum[0]) / 2,
    (bounds.minimum[1] + bounds.maximum[1]) / 2,
    (bounds.minimum[2] + bounds.maximum[2]) / 2,
  ];

  const snapDraggedCorner = (
    rawPosition: [number, number, number],
    draggedCorner: [number, number, number],
    pointerRay: Ray,
  ): [number, number, number] => {
    const movingCorner = cornerInWorld(draggedCorner, instance, rawPosition);
    let bestDistance = 0.18;
    let result = rawPosition;
    for (const other of allInstances) {
      if (movingInstanceIds.includes(other.id)) continue;
      for (const targetCorner of bounds.corners) {
        const target = cornerInWorld(targetCorner, other);
        const distance = Math.sqrt(pointerRay.distanceSqToPoint(target));
        if (distance >= bestDistance) continue;
        const alignedPosition = new Vector3(
          rawPosition[0] + target.x - movingCorner.x,
          rawPosition[1] + target.y - movingCorner.y,
          rawPosition[2] + target.z - movingCorner.z,
        );
        const away = new Vector3(
          alignedPosition.x - other.position[0],
          alignedPosition.y - other.position[1],
          alignedPosition.z - other.position[2],
        );
        if (away.lengthSq() > 1e-10) away.normalize().multiplyScalar(SOURCE_SURFACE_PADDING_M);
        result = [
          alignedPosition.x + away.x,
          alignedPosition.y + away.y,
          alignedPosition.z + away.z,
        ];
        bestDistance = distance;
      }
    }
    return result;
  };

  const startCornerDrag = (event: ThreeEvent<PointerEvent>, corner: [number, number, number]) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    if (!selected) onSelect(false);
    const handleWorld = cornerInWorld(corner, instance);
    // A horizontal drag plane becomes numerically unstable when viewed near
    // edge-on. A view-facing plane keeps screen-space movement bounded; the
    // resulting displacement is still constrained to scene X/Z below.
    const plane = new Plane().setFromNormalAndCoplanarPoint(
      camera.getWorldDirection(new Vector3()).normalize(),
      handleWorld,
    );
    const startPoint = event.ray.intersectPlane(plane, new Vector3());
    if (!startPoint) return;
    dragState.current = {
      pointerId: event.pointerId,
      plane,
      startPoint,
      startPosition: [...instance.position],
      corner,
    };
    orbitControls && (orbitControls.enabled = false);
    (event.target as unknown as { setPointerCapture?: (pointerId: number) => void }).setPointerCapture?.(event.pointerId);
  };

  const moveCornerDrag = (event: ThreeEvent<PointerEvent>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if ((event.nativeEvent.buttons & 1) === 0) {
      cancelCornerDrag(true);
      return;
    }
    event.stopPropagation();
    const point = event.ray.intersectPlane(drag.plane, new Vector3());
    if (!point || !finiteVector(point)) return;
    const rawPosition: [number, number, number] = [
      drag.startPosition[0] + point.x - drag.startPoint.x,
      drag.startPosition[1],
      drag.startPosition[2] + point.z - drag.startPoint.z,
    ];
    const snapped = snapDraggedCorner(rawPosition, drag.corner, event.ray);
    onTransform({
      positionX: snapped[0],
      positionHeightM: snapped[1],
      positionZ: snapped[2],
      pitchDeg: instance.pitchDeg,
      yawDeg: instance.yawDeg,
      rollDeg: instance.rollDeg,
    });
  };

  const finishCornerDrag = (event: ThreeEvent<PointerEvent>) => {
    if (!dragState.current || dragState.current.pointerId !== event.pointerId) return;
    event.stopPropagation();
    cancelCornerDrag(true);
    (event.target as unknown as { releasePointerCapture?: (pointerId: number) => void }).releasePointerCapture?.(event.pointerId);
  };

  const applyGizmoTransform = () => {
    const object = speakerRef.current;
    if (!object || !finiteVector(object.position)) return;
    const minimumRotatedY = Math.min(...bounds.corners.map((corner) => (
      new Vector3(...corner).applyQuaternion(object.quaternion).y
    )));
    const minimumHeight = SOURCE_GROUND_CLEARANCE_M - minimumRotatedY;
    object.position.y = Math.max(minimumHeight, object.position.y);
    const rotation = new Euler().setFromQuaternion(object.quaternion, "YXZ");
    onTransform({
      positionX: object.position.x,
      positionHeightM: object.position.y,
      positionZ: object.position.z,
      pitchDeg: normalizedYaw(MathUtils.radToDeg(rotation.x)),
      yawDeg: normalizedYaw(MathUtils.radToDeg(rotation.y)),
      rollDeg: normalizedYaw(MathUtils.radToDeg(rotation.z)),
    });
  };

  return (
    <>
      <group
        ref={speakerRef}
        position={instance.position}
        quaternion={quaternion}
        onClick={(event) => {
          event.stopPropagation();
          onSelect(event.nativeEvent.ctrlKey || event.nativeEvent.metaKey);
        }}
      >
        {geometry ? (
          <mesh geometry={geometry} castShadow receiveShadow>
            <meshStandardMaterial
              color={selected ? "#d4d0c8" : "#667176"}
              roughness={0.72}
              metalness={0.08}
              emissive={selected ? "#2a3218" : "#000000"}
            />
          </mesh>
        ) : (
          <mesh position={boundsCenter} castShadow receiveShadow>
            <boxGeometry args={sceneBounds} />
            <meshStandardMaterial color={selected ? "#d4d0c8" : "#667176"} roughness={0.72} />
          </mesh>
        )}
        {pkg.isDemo && (
          <>
            <mesh position={[0, 0.055, bounds.maximum[2] + 0.003]}>
              <circleGeometry args={[sceneBounds[0] * 0.27, 40]} />
              <meshStandardMaterial color="#1e2224" roughness={0.5} />
            </mesh>
            <mesh position={[0, -0.075, bounds.maximum[2] + 0.005]}>
              <circleGeometry args={[sceneBounds[0] * 0.12, 36]} />
              <meshStandardMaterial color="#3f484c" roughness={0.4} />
            </mesh>
          </>
        )}
        {active && individualControls && bounds.corners.map((corner, index) => (
          <mesh
            key={index}
            position={corner}
            renderOrder={20}
            onPointerDown={(event) => startCornerDrag(event, corner)}
            onPointerMove={moveCornerDrag}
            onPointerUp={finishCornerDrag}
            onPointerCancel={finishCornerDrag}
          >
            <sphereGeometry args={[0.055, 14, 10]} />
            <meshBasicMaterial color="#dce9a7" depthTest={false} toneMapped={false} />
          </mesh>
        ))}
        {active && (
          <Html position={[bounds.maximum[0] + 0.08, bounds.maximum[1], boundsCenter[2]]} center distanceFactor={10}>
            <div className="scene-label">{instance.id.toUpperCase()}</div>
          </Html>
        )}
      </group>
      {active && individualControls && (transformMode === "translate" || transformMode === "rotate") && (
        <TransformControls
          object={speakerRef as MutableRefObject<Group>}
          mode={transformMode}
          space="world"
          size={0.82}
          showX
          showY
          showZ
          rotationSnap={transformMode === "rotate" && !angleSnapDisabled ? MathUtils.degToRad(5) : null}
          onObjectChange={applyGizmoTransform}
          onMouseUp={onManipulationEnd}
        />
      )}
    </>
  );
}

function selectionWorldBounds(pkg: LoadedSpeakerPackage, instances: readonly SpeakerInstance[]): SceneBounds {
  const cabinetBounds = packageSceneBounds(pkg);
  const minimum: [number, number, number] = [Infinity, Infinity, Infinity];
  const maximum: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const instance of instances) {
    for (const corner of cabinetBounds.corners) {
      const world = cornerInWorld(corner, instance);
      minimum[0] = Math.min(minimum[0], world.x);
      minimum[1] = Math.min(minimum[1], world.y);
      minimum[2] = Math.min(minimum[2], world.z);
      maximum[0] = Math.max(maximum[0], world.x);
      maximum[1] = Math.max(maximum[1], world.y);
      maximum[2] = Math.max(maximum[2], world.z);
    }
  }
  const corners: Array<[number, number, number]> = [];
  for (const x of [minimum[0], maximum[0]]) {
    for (const y of [minimum[1], maximum[1]]) {
      for (const z of [minimum[2], maximum[2]]) corners.push([x, y, z]);
    }
  }
  return { minimum, maximum, corners };
}

function sourceGroupPose(instance: SpeakerInstance, position: Vector3, quaternion: Quaternion): SourceGroupPoseUpdate {
  const rotation = new Euler().setFromQuaternion(quaternion, "YXZ");
  return {
    id: instance.id,
    positionX: position.x,
    positionHeightM: position.y,
    positionZ: position.z,
    pitchDeg: normalizedYaw(MathUtils.radToDeg(rotation.x)),
    yawDeg: normalizedYaw(MathUtils.radToDeg(rotation.y)),
    rollDeg: normalizedYaw(MathUtils.radToDeg(rotation.z)),
  };
}

function SpeakerSelectionControls({
  pkg,
  instances,
  allInstances,
  transformMode,
  angleSnapDisabled,
  onTransform,
  onManipulationEnd,
}: {
  pkg: LoadedSpeakerPackage;
  instances: SpeakerInstance[];
  allInstances: SpeakerInstance[];
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onTransform: (poses: SourceGroupPoseUpdate[]) => void;
  onManipulationEnd: () => void;
}) {
  const pivotRef = useRef<Group>(null);
  const camera = useThree((state) => state.camera);
  const orbitControls = useThree((state) => state.controls) as { enabled: boolean } | null;
  const bounds = useMemo(() => selectionWorldBounds(pkg, instances), [instances, pkg]);
  const center = useMemo(() => new Vector3(
    (bounds.minimum[0] + bounds.maximum[0]) / 2,
    (bounds.minimum[1] + bounds.maximum[1]) / 2,
    (bounds.minimum[2] + bounds.maximum[2]) / 2,
  ), [bounds]);
  const cabinetBounds = useMemo(() => packageSceneBounds(pkg), [pkg]);
  const dragState = useRef<{
    pointerId: number;
    plane: Plane;
    startPoint: Vector3;
    startCorner: Vector3;
    startCenter: Vector3;
    instances: SpeakerInstance[];
  } | null>(null);
  const gizmoState = useRef<{
    pivot: Vector3;
    instances: SpeakerInstance[];
    mode: "translate" | "rotate";
  } | null>(null);

  useEffect(() => {
    if (!gizmoState.current && pivotRef.current) {
      pivotRef.current.position.copy(center);
      pivotRef.current.quaternion.identity();
    }
  }, [center]);

  const emitTranslation = (startInstances: SpeakerInstance[], delta: Vector3) => {
    onTransform(startInstances.map((instance) => sourceGroupPose(
      instance,
      new Vector3(...instance.position).add(delta),
      new Quaternion().setFromEuler(new Euler(
        MathUtils.degToRad(instance.pitchDeg),
        MathUtils.degToRad(instance.yawDeg),
        MathUtils.degToRad(instance.rollDeg),
        "YXZ",
      )),
    )));
  };

  const finishHandleDrag = (flushSolve = false) => {
    const wasDragging = dragState.current !== null;
    dragState.current = null;
    orbitControls && (orbitControls.enabled = true);
    if (flushSolve && wasDragging) onManipulationEnd();
  };

  useEffect(() => {
    const finish = () => finishHandleDrag(true);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    window.addEventListener("blur", finish);
    return () => {
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      window.removeEventListener("blur", finish);
    };
  }, [onManipulationEnd, orbitControls]);

  const startHandleDrag = (event: ThreeEvent<PointerEvent>, corner: [number, number, number]) => {
    if (event.nativeEvent.button !== 0) return;
    event.stopPropagation();
    const startCorner = new Vector3(...corner);
    const plane = new Plane().setFromNormalAndCoplanarPoint(
      camera.getWorldDirection(new Vector3()).normalize(),
      startCorner,
    );
    const startPoint = event.ray.intersectPlane(plane, new Vector3());
    if (!startPoint) return;
    dragState.current = {
      pointerId: event.pointerId,
      plane,
      startPoint,
      startCorner,
      startCenter: center.clone(),
      instances: instances.map((instance) => ({ ...instance, position: [...instance.position] })),
    };
    orbitControls && (orbitControls.enabled = false);
    (event.target as unknown as { setPointerCapture?: (pointerId: number) => void }).setPointerCapture?.(event.pointerId);
  };

  const moveHandleDrag = (event: ThreeEvent<PointerEvent>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if ((event.nativeEvent.buttons & 1) === 0) {
      finishHandleDrag(true);
      return;
    }
    event.stopPropagation();
    const point = event.ray.intersectPlane(drag.plane, new Vector3());
    if (!point || !finiteVector(point)) return;
    const rawDelta = new Vector3(point.x - drag.startPoint.x, 0, point.z - drag.startPoint.z);
    const snappedDelta = rawDelta.clone();
    const movingCorner = drag.startCorner.clone().add(rawDelta);
    let bestDistance = 0.18;
    for (const other of allInstances) {
      if (instances.some((instance) => instance.id === other.id)) continue;
      for (const targetCorner of cabinetBounds.corners) {
        const target = cornerInWorld(targetCorner, other);
        const distance = Math.sqrt(event.ray.distanceSqToPoint(target));
        if (distance >= bestDistance) continue;
        const candidateDelta = rawDelta.clone().add(target.clone().sub(movingCorner));
        const away = drag.startCenter.clone().add(candidateDelta).sub(new Vector3(...other.position));
        if (away.lengthSq() > 1e-10) candidateDelta.add(away.normalize().multiplyScalar(SOURCE_SURFACE_PADDING_M));
        snappedDelta.copy(candidateDelta);
        bestDistance = distance;
      }
    }
    emitTranslation(drag.instances, snappedDelta);
  };

  const finishHandlePointer = (event: ThreeEvent<PointerEvent>) => {
    if (!dragState.current || dragState.current.pointerId !== event.pointerId) return;
    event.stopPropagation();
    finishHandleDrag(true);
    (event.target as unknown as { releasePointerCapture?: (pointerId: number) => void }).releasePointerCapture?.(event.pointerId);
  };

  const startGizmo = () => {
    if (!pivotRef.current || (transformMode !== "translate" && transformMode !== "rotate")) return;
    gizmoState.current = {
      pivot: pivotRef.current.position.clone(),
      instances: instances.map((instance) => ({ ...instance, position: [...instance.position] })),
      mode: transformMode,
    };
    pivotRef.current.quaternion.identity();
  };

  const applyGizmoTransform = () => {
    const object = pivotRef.current;
    const state = gizmoState.current;
    if (!object || !state) return;
    if (state.mode === "translate") {
      emitTranslation(state.instances, object.position.clone().sub(state.pivot));
      return;
    }
    const deltaRotation = object.quaternion.clone().normalize();
    onTransform(state.instances.map((instance) => {
      const startRotation = new Quaternion().setFromEuler(new Euler(
        MathUtils.degToRad(instance.pitchDeg),
        MathUtils.degToRad(instance.yawDeg),
        MathUtils.degToRad(instance.rollDeg),
        "YXZ",
      ));
      const position = new Vector3(...instance.position)
        .sub(state.pivot)
        .applyQuaternion(deltaRotation)
        .add(state.pivot);
      return sourceGroupPose(instance, position, deltaRotation.clone().multiply(startRotation));
    }));
  };

  const finishGizmo = () => {
    if (!gizmoState.current) return;
    gizmoState.current = null;
    if (pivotRef.current) {
      pivotRef.current.position.copy(center);
      pivotRef.current.quaternion.identity();
    }
    onManipulationEnd();
  };

  return (
    <>
      {bounds.corners.map((corner, index) => (
        <mesh
          key={index}
          position={corner}
          renderOrder={20}
          onPointerDown={(event) => startHandleDrag(event, corner)}
          onPointerMove={moveHandleDrag}
          onPointerUp={finishHandlePointer}
          onPointerCancel={finishHandlePointer}
        >
          <sphereGeometry args={[0.075, 14, 10]} />
          <meshBasicMaterial color="#dce9a7" depthTest={false} toneMapped={false} />
        </mesh>
      ))}
      <group ref={pivotRef} />
      {(transformMode === "translate" || transformMode === "rotate") && (
        <TransformControls
          object={pivotRef as MutableRefObject<Group>}
          mode={transformMode}
          space="world"
          size={0.92}
          showX
          showY
          showZ
          rotationSnap={transformMode === "rotate" && !angleSnapDisabled ? MathUtils.degToRad(5) : null}
          onMouseDown={startGizmo}
          onObjectChange={applyGizmoTransform}
          onMouseUp={finishGizmo}
        />
      )}
    </>
  );
}

function AcousticScene(props: SceneViewProps) {
  const selectedInstances = new Set(props.selectedInstances);
  const selectedSources = props.sources.filter((source) => selectedInstances.has(source.id));
  const groupedSelection = selectedSources.length > 1 && selectedSources.some((source) => source.id === props.activeInstance);
  return (
    <>
      <color attach="background" args={["#293134"]} />
      <fog attach="fog" args={["#293134", 70, 220]} />
      <ambientLight intensity={0.65} />
      <directionalLight
        position={[-7, 12, -5]}
        intensity={2.2}
        color="#f5f0de"
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <directionalLight position={[9, 5, 10]} intensity={0.85} color="#acc7d2" />
      <group>
        {props.sources.map((source) => (
          <SpeakerGeometry
            key={source.id}
            pkg={props.pkg}
            instance={source}
            allInstances={props.sources}
            selected={selectedInstances.has(source.id)}
            active={props.activeInstance === source.id}
            individualControls={!groupedSelection}
            movingInstanceIds={props.selectedInstances}
            transformMode={props.transformMode}
            angleSnapDisabled={props.angleSnapDisabled}
            onSelect={(additive) => props.onSelectInstance(source.id, additive)}
            onTransform={(pose) => props.onTransformSource(source.id, pose)}
            onManipulationEnd={props.onManipulationEnd}
          />
        ))}
      </group>
      {groupedSelection && (
        <SpeakerSelectionControls
          pkg={props.pkg}
          instances={selectedSources}
          allInstances={props.sources}
          transformMode={props.transformMode}
          angleSnapDisabled={props.angleSnapDisabled}
          onTransform={props.onTransformSources}
          onManipulationEnd={props.onManipulationEnd}
        />
      )}
      <FieldPlane
        observation={props.observation}
        field={props.field}
        selected={selectedInstances.has("audience-plane")}
        active={props.activeInstance === "audience-plane"}
        transformMode={props.transformMode}
        angleSnapDisabled={props.angleSnapDisabled}
        onTransform={props.onTransformObservation}
        onResize={props.onResizeObservation}
        onManipulationEnd={props.onManipulationEnd}
        onTextureReady={props.onFieldTextureReady}
      />
      <gridHelper args={[50, 50, "#303831", "#242a25"]} position={[0, 0, 12]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.012, 12]} receiveShadow>
        <planeGeometry args={[50, 50]} />
        <shadowMaterial transparent opacity={0.26} />
      </mesh>
      <OrbitControls
        makeDefault
        target={[0, 1.6, 7]}
        minDistance={3}
        maxDistance={120}
        maxPolarAngle={Math.PI / 2 - 0.02}
        enableDamping
        dampingFactor={0.08}
      />
    </>
  );
}

export function SceneView(props: SceneViewProps) {
  return (
    <Canvas
      shadows
      dpr={[1, 1.7]}
      camera={{ position: [9.5, 7.5, 13.5], fov: 44, near: 0.05, far: 300 }}
      gl={{ antialias: true, alpha: false }}
      onPointerMissed={() => props.onSelectInstance(null)}
    >
      <AcousticScene {...props} />
    </Canvas>
  );
}
