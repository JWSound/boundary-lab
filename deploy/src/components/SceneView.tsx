import { Grid, Html, OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
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
  RedFormat,
  RGFormat,
  ShaderMaterial,
  UnsignedByteType,
  Vector2,
  Vector3,
} from "three";
import type { FieldFrame, LoadedSpeakerPackage, MicrophoneConfiguration, ObservationPlane, RigidMeshAsset, SpeakerInstance } from "../model/types";
import { SOURCE_GROUND_CLEARANCE_M } from "../model/field";
import { cabinetClearanceViolations, cabinetLocalBounds, type BoundaryMeshAsset } from "../model/cabinetPlacement";
import {
  configureAxisOnlyRotation,
  groundParallelPosition,
  matchingCornerPaddingDirections,
  paddedCornerSnapDelta,
  rotationReadout,
  stickyCornerSnapTarget,
  translationReadout,
  type CornerSnapTarget,
  type TransformControlsInternal,
} from "../model/transformControls";

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

export type MicrophonePoseUpdate = Pick<MicrophoneConfiguration, "positionX" | "positionHeightM" | "positionZ">;

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
  packages: LoadedSpeakerPackage[];
  rigidMeshes: RigidMeshAsset[];
  sources: SpeakerInstance[];
  rigidObjects: SpeakerInstance[];
  microphones: MicrophoneConfiguration[];
  observation: ObservationPlane;
  field: FieldFrame;
  phaseAnimationEnabled: boolean;
  selectedInstances: readonly string[];
  activeInstance: string | null;
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onSelectInstance: (id: string | null, additive?: boolean) => void;
  onTransformSource: (id: string, pose: SourcePoseUpdate) => void;
  onTransformRigid: (id: string, pose: SourcePoseUpdate) => void;
  onTransformSources: (poses: SourceGroupPoseUpdate[]) => void;
  onTransformMicrophone: (id: string, pose: MicrophonePoseUpdate) => void;
  onTransformObservation: (pose: ObservationPoseUpdate) => void;
  onResizeObservation: (resize: ObservationResizeUpdate) => void;
  onSourceManipulationStart: (ids: readonly string[]) => void;
  onSourceManipulationEnd: () => void;
  onManipulationEnd: () => void;
  onFieldTextureReady?: (profile: FieldTextureProfile) => void;
}

const MICROPHONE_COLORS = ["#ffdf00", "#00dfff", "#ff6f00", "#7fe35b", "#e08cff", "#ff748c"];

export interface FieldTextureProfile {
  pointCount: number;
  textureBytes: number;
  rasterMs: number;
  commitToFrameMs: number;
}

const packageSceneBounds = cabinetLocalBounds;

function instanceAxes(instance: SpeakerInstance): [Vector3, Vector3, Vector3] {
  const rotation = new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(instance.pitchDeg),
    MathUtils.degToRad(instance.yawDeg),
    MathUtils.degToRad(instance.rollDeg),
    "YXZ",
  ));
  return [
    new Vector3(1, 0, 0).applyQuaternion(rotation),
    new Vector3(0, 1, 0).applyQuaternion(rotation),
    new Vector3(0, 0, 1).applyQuaternion(rotation),
  ];
}

function cornerSigns(corner: readonly [number, number, number], bounds: SceneBounds): [number, number, number] {
  return [
    corner[0] === bounds.minimum[0] ? -1 : 1,
    corner[1] === bounds.minimum[1] ? -1 : 1,
    corner[2] === bounds.minimum[2] ? -1 : 1,
  ];
}

function snappedPosesHaveClearance(
  packages: readonly BoundaryMeshAsset[],
  moving: readonly SpeakerInstance[],
  stationary: readonly SpeakerInstance[],
): boolean {
  const movingIds = new Set(moving.map((instance) => instance.id));
  const packageMap = new Map(packages.map((item) => [item.id, item]));
  return !cabinetClearanceViolations(packageMap, [...moving, ...stationary]).some(
    ([left, right]) => movingIds.has(left) || movingIds.has(right),
  );
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

function TransformReadout({ text, position }: { text: string | null; position: [number, number, number] }) {
  if (!text) return null;
  return (
    <Html position={position} center distanceFactor={9} zIndexRange={[80, 0]}>
      <div className="transform-readout">{text}</div>
    </Html>
  );
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
  uniform sampler2D uPressureMap;
  uniform sampler2D uValidityMap;
  uniform vec2 uTextureSize;
  uniform float uMinimumDb;
  uniform float uMaximumDb;
  uniform float uBandingDb;
  uniform float uDisplayMode;
  uniform float uPressureScalePa;
  uniform float uPhaseRad;
  uniform float uPhaseAnimationEnabled;
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

  vec2 filteredPressure(vec2 uv) {
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
    vec4 validity = vec4(validityAt(uv00), validityAt(uv10), validityAt(uv01), validityAt(uv11));
    vec4 validWeights = weights * validity;
    float weightSum = dot(validWeights, vec4(1.0));
    vec2 value =
      texture2D(uPressureMap, uv00).rg * validWeights.x +
      texture2D(uPressureMap, uv10).rg * validWeights.y +
      texture2D(uPressureMap, uv01).rg * validWeights.z +
      texture2D(uPressureMap, uv11).rg * validWeights.w;
    return weightSum > 0.0001 ? value / weightSum : vec2(0.0);
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

  vec3 pressurePalette(float position) {
    vec3 blue = vec3(0.1059, 0.3020, 0.6902);
    vec3 neutral = vec3(0.9569, 0.9569, 0.9569);
    vec3 red = vec3(0.7608, 0.1059, 0.1373);
    return position < 0.5
      ? mix(blue, neutral, position * 2.0)
      : mix(neutral, red, (position - 0.5) * 2.0);
  }

  vec3 srgbToLinear(vec3 value) {
    vec3 low = value / 12.92;
    vec3 high = pow((value + 0.055) / 1.055, vec3(2.4));
    return mix(low, high, step(vec3(0.04045), value));
  }

  void main() {
    // Keep the clipping boundary discrete even though valid SPL values are smooth.
    if (nearestValidity(vUv) < 0.5) discard;
    vec3 color;
    if (uDisplayMode < 0.5) {
      color = palette(colorPosition(filteredSpl(vUv)));
    } else {
      vec2 pressure = filteredPressure(vUv);
      float valuePa = uDisplayMode < 1.5 ? pressure.x : pressure.y;
      if (uPhaseAnimationEnabled > 0.5) {
        valuePa = pressure.x * cos(uPhaseRad) + pressure.y * sin(uPhaseRad);
      }
      float position = clamp(0.5 + valuePa / (2.0 * max(0.0001, uPressureScalePa)), 0.0, 1.0);
      color = pressurePalette(position);
    }
    color = srgbToLinear(color);
    gl_FragColor = vec4(color, 0.9412);
    #include <colorspace_fragment>
  }
`;

function FieldPlane({
  observation,
  field,
  phaseAnimationEnabled,
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
  phaseAnimationEnabled: boolean;
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
  const gizmoRef = useRef<TransformControlsInternal | null>(null);
  const gizmoStartPosition = useRef<Vector3 | null>(null);
  const [gizmoReadout, setGizmoReadout] = useState<string | null>(null);
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
  const phaseRad = useRef(0);

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
    const complexPressure = new Float32Array(field.pressureReal.length * 2);
    for (let index = 0; index < field.pressureReal.length; index += 1) {
      complexPressure[index * 2] = field.pressureReal[index];
      complexPressure[index * 2 + 1] = field.pressureImag[index];
    }
    const pressure = new DataTexture(complexPressure, field.columns, field.rows, RGFormat, FloatType);
    pressure.minFilter = NearestFilter;
    pressure.magFilter = NearestFilter;
    pressure.wrapS = ClampToEdgeWrapping;
    pressure.wrapT = ClampToEdgeWrapping;
    pressure.generateMipmaps = false;
    pressure.needsUpdate = true;
    const validity = new DataTexture(field.validMask, field.columns, field.rows, RedFormat, UnsignedByteType);
    validity.minFilter = NearestFilter;
    validity.magFilter = NearestFilter;
    validity.wrapS = ClampToEdgeWrapping;
    validity.wrapT = ClampToEdgeWrapping;
    validity.generateMipmaps = false;
    validity.needsUpdate = true;
    textureProfile.current = {
      pointCount: field.columns * field.rows,
      textureBytes: field.splDb.byteLength + complexPressure.byteLength + field.validMask.byteLength,
      rasterMs: performance.now() - rasterStarted,
    };
    return { spl, pressure, validity };
  }, [field]);
  const heatmapMaterial = useMemo(() => new ShaderMaterial({
    uniforms: {
      uSplMap: { value: textures.spl },
      uPressureMap: { value: textures.pressure },
      uValidityMap: { value: textures.validity },
      uTextureSize: { value: new Vector2(field.columns, field.rows) },
      uMinimumDb: { value: observation.heatmapMinimumDb },
      uMaximumDb: { value: observation.heatmapMaximumDb },
      uBandingDb: { value: observation.heatmapBandingDb },
      uDisplayMode: { value: 0 },
      uPressureScalePa: { value: observation.pressureScalePa },
      uPhaseRad: { value: 0 },
      uPhaseAnimationEnabled: { value: 0 },
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
  heatmapMaterial.uniforms.uDisplayMode.value = observation.displayMode === "spl" ? 0 : observation.displayMode === "real_pressure" ? 1 : 2;
  heatmapMaterial.uniforms.uPressureScalePa.value = observation.pressureScalePa;
  heatmapMaterial.uniforms.uPhaseAnimationEnabled.value = phaseAnimationEnabled && observation.displayMode !== "spl" ? 1 : 0;

  useEffect(() => {
    phaseRad.current = 0;
    heatmapMaterial.uniforms.uPhaseRad.value = 0;
  }, [heatmapMaterial, observation.displayMode, phaseAnimationEnabled]);

  useFrame((_state, delta) => {
    if (!phaseAnimationEnabled || observation.displayMode === "spl") return;
    phaseRad.current = (phaseRad.current + Math.PI * 2 * observation.phaseAnimationSpeedHz * delta) % (Math.PI * 2);
    heatmapMaterial.uniforms.uPhaseRad.value = phaseRad.current;
  });

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
    textures.pressure.dispose();
    textures.validity.dispose();
  }, [textures]);

  useEffect(() => () => heatmapMaterial.dispose(), [heatmapMaterial]);

  const applyGizmoTransform = () => {
    const object = planeRef.current;
    if (!object || !finiteVector(object.position)) return;
    if (gizmoStartPosition.current) {
      setGizmoReadout(transformMode === "rotate"
        ? rotationReadout(object.quaternion, gizmoRef.current?.axis)
        : translationReadout(gizmoStartPosition.current, object.position, gizmoRef.current?.axis));
    }
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

  const startGizmoTransform = () => {
    const object = planeRef.current;
    if (!object) return;
    gizmoStartPosition.current = object.position.clone();
    setGizmoReadout(transformMode === "rotate"
      ? rotationReadout(object.quaternion, gizmoRef.current?.axis)
      : translationReadout(object.position, object.position, gizmoRef.current?.axis));
  };

  const finishGizmoTransform = () => {
    gizmoStartPosition.current = null;
    setGizmoReadout(null);
    onManipulationEnd();
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
        <TransformReadout text={gizmoReadout} position={[0, 0.35, 0]} />
      </group>
      {active && (transformMode === "translate" || transformMode === "rotate") && (
        <TransformControls
          ref={(control) => { gizmoRef.current = configureAxisOnlyRotation(control); }}
          object={planeRef as MutableRefObject<Group>}
          mode={transformMode}
          space="world"
          size={0.82}
          showX
          showY
          showZ
          rotationSnap={transformMode === "rotate" && !angleSnapDisabled ? MathUtils.degToRad(5) : null}
          onMouseDown={startGizmoTransform}
          onObjectChange={applyGizmoTransform}
          onMouseUp={finishGizmoTransform}
        />
      )}
    </>
  );
}

function SpeakerGeometry({
  pkg,
  packages,
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
  onManipulationStart,
  onManipulationEnd,
}: {
  pkg: BoundaryMeshAsset;
  packages: BoundaryMeshAsset[];
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
  onManipulationStart: () => void;
  onManipulationEnd: () => void;
}) {
  const speakerRef = useRef<Group>(null);
  const gizmoRef = useRef<TransformControlsInternal | null>(null);
  const gizmoStartPosition = useRef<Vector3 | null>(null);
  const [gizmoReadout, setGizmoReadout] = useState<string | null>(null);
  const orbitControls = useThree((state) => state.controls) as { enabled: boolean } | null;
  const camera = useThree((state) => state.camera);
  const gl = useThree((state) => state.gl);
  const [snapHighlight, setSnapHighlight] = useState<Vector3 | null>(null);
  const dragState = useRef<{
    pointerId: number;
    trackingPlane: Plane;
    probePlane: Plane;
    startTrackingPoint: Vector3;
    startProbePoint: Vector3;
    startPosition: [number, number, number];
    startObjectCenter: Vector3;
    corner: [number, number, number];
    maximumDeltaM: number;
    snapKey: string | null;
  } | null>(null);
  const cancelCornerDrag = (flushSolve = false) => {
    const wasDragging = dragState.current !== null;
    dragState.current = null;
    setSnapHighlight(null);
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
  const rigidBoundary = !("manifest" in pkg);

  const startCornerDrag = (event: ThreeEvent<PointerEvent>, corner: [number, number, number]) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    if (!selected) onSelect(false);
    const handleWorld = cornerInWorld(corner, instance);
    const probePlane = new Plane().setFromNormalAndCoplanarPoint(
      camera.getWorldDirection(new Vector3()).normalize(),
      handleWorld,
    );
    // Track directly on the cabinet's horizontal plane so the handle remains
    // under the cursor. Near the horizon, use the bounded view-facing fallback.
    const groundPlane = new Plane().setFromNormalAndCoplanarPoint(new Vector3(0, 1, 0), handleWorld);
    const useGroundPlane = Math.abs(event.ray.direction.y) >= 0.075;
    const trackingPlane = useGroundPlane ? groundPlane : probePlane;
    const startTrackingPoint = event.ray.intersectPlane(trackingPlane, new Vector3());
    const startProbePoint = event.ray.intersectPlane(probePlane, new Vector3());
    if (!startTrackingPoint || !startProbePoint) return;
    dragState.current = {
      pointerId: event.pointerId,
      trackingPlane,
      probePlane,
      startTrackingPoint,
      startProbePoint,
      startPosition: [...instance.position],
      startObjectCenter: new Vector3(...instance.position),
      corner,
      maximumDeltaM: Math.max(10, camera.position.distanceTo(handleWorld) * 4),
      snapKey: null,
    };
    onManipulationStart();
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
    const point = event.ray.intersectPlane(drag.trackingPlane, new Vector3());
    if (!point || !finiteVector(point)) return;
    const pointerDelta = point.clone().sub(drag.startTrackingPoint);
    if (pointerDelta.length() > drag.maximumDeltaM) return;
    const probePoint = event.ray.intersectPlane(drag.probePlane, new Vector3());
    const probeDelta = probePoint && finiteVector(probePoint)
      ? probePoint.sub(drag.startProbePoint)
      : pointerDelta;
    const targetCorners: CornerSnapTarget[] = [];
    const movingBounds = packageSceneBounds(pkg);
    const movingSigns = cornerSigns(drag.corner, movingBounds);
    const movingAxes = instanceAxes(instance);
    for (const other of allInstances) {
      if (movingInstanceIds.includes(other.id)) continue;
      const targetPackage = packages.find((candidate) => candidate.id === other.packageId) ?? pkg;
      const targetBounds = packageSceneBounds(targetPackage);
      const targetAxes = instanceAxes(other);
      for (const [cornerIndex, targetCorner] of targetBounds.corners.entries()) {
        targetCorners.push({
          key: `${other.id}:${cornerIndex}`,
          position: cornerInWorld(targetCorner, other),
          objectCenter: new Vector3(...other.position),
          paddingDirections: matchingCornerPaddingDirections(
            movingSigns,
            movingAxes,
            cornerSigns(targetCorner, targetBounds),
            targetAxes,
          ),
        });
      }
    }
    const viewport = gl.domElement.getBoundingClientRect();
    const snapTarget = stickyCornerSnapTarget(
      event.nativeEvent.clientX,
      event.nativeEvent.clientY,
      targetCorners,
      camera,
      viewport,
      drag.snapKey,
    );
    const snappedDelta = paddedCornerSnapDelta(
      cornerInWorld(drag.corner, instance, drag.startPosition),
      drag.startObjectCenter,
      pointerDelta,
      probeDelta,
      snapTarget,
    );
    const snapped = groundParallelPosition(drag.startPosition, snappedDelta);
    snapped[1] += snappedDelta.y;
    const snappedInstance = { ...instance, position: snapped };
    const stationary = allInstances.filter((other) => !movingInstanceIds.includes(other.id));
    const validSnap = snapTarget && snappedPosesHaveClearance(packages, [snappedInstance], stationary);
    drag.snapKey = validSnap ? snapTarget.key : null;
    setSnapHighlight(validSnap ? snapTarget.position.clone() : null);
    if (snapTarget && !validSnap) {
      const unsnapped = groundParallelPosition(drag.startPosition, pointerDelta);
      snapped[0] = unsnapped[0];
      snapped[1] = unsnapped[1];
      snapped[2] = unsnapped[2];
    }
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
    if (gizmoStartPosition.current) {
      setGizmoReadout(transformMode === "rotate"
        ? rotationReadout(object.quaternion, gizmoRef.current?.axis)
        : translationReadout(gizmoStartPosition.current, object.position, gizmoRef.current?.axis));
    }
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

  const startGizmoTransform = () => {
    const object = speakerRef.current;
    if (!object) return;
    gizmoStartPosition.current = object.position.clone();
    setGizmoReadout(transformMode === "rotate"
      ? rotationReadout(object.quaternion, gizmoRef.current?.axis)
      : translationReadout(object.position, object.position, gizmoRef.current?.axis));
    onManipulationStart();
  };

  const finishGizmoTransform = () => {
    gizmoStartPosition.current = null;
    setGizmoReadout(null);
    onManipulationEnd();
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
              color={selected ? "#e0d5bd" : rigidBoundary ? "#7b7162" : "#667176"}
              roughness={0.72}
              metalness={0.08}
              emissive={selected ? "#2a3218" : "#000000"}
            />
          </mesh>
        ) : (
          <mesh position={boundsCenter} castShadow receiveShadow>
            <boxGeometry args={sceneBounds} />
            <meshStandardMaterial color={selected ? "#e0d5bd" : rigidBoundary ? "#7b7162" : "#667176"} roughness={0.72} />
          </mesh>
        )}
        {"isDemo" in pkg && pkg.isDemo && (
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
        <TransformReadout text={gizmoReadout} position={[0, bounds.maximum[1] + 0.22, boundsCenter[2]]} />
      </group>
      {active && individualControls && (transformMode === "translate" || transformMode === "rotate") && (
        <TransformControls
          ref={(control) => { gizmoRef.current = configureAxisOnlyRotation(control); }}
          object={speakerRef as MutableRefObject<Group>}
          mode={transformMode}
          space="world"
          size={0.82}
          showX
          showY
          showZ
          rotationSnap={transformMode === "rotate" && !angleSnapDisabled ? MathUtils.degToRad(5) : null}
          onMouseDown={startGizmoTransform}
          onObjectChange={applyGizmoTransform}
          onMouseUp={finishGizmoTransform}
        />
      )}
      {snapHighlight && (
        <mesh position={snapHighlight} renderOrder={30}>
          <sphereGeometry args={[0.095, 16, 12]} />
          <meshBasicMaterial color="#fff36b" wireframe depthTest={false} toneMapped={false} />
        </mesh>
      )}
    </>
  );
}

function MicrophoneGeometry({
  microphone,
  color,
  selected,
  active,
  transformMode,
  onSelect,
  onTransform,
  onManipulationEnd,
}: {
  microphone: MicrophoneConfiguration;
  color: string;
  selected: boolean;
  active: boolean;
  transformMode: SceneTransformMode;
  onSelect: (additive: boolean) => void;
  onTransform: (pose: MicrophonePoseUpdate) => void;
  onManipulationEnd: () => void;
}) {
  const microphoneRef = useRef<Group>(null);
  const gizmoRef = useRef<TransformControlsInternal | null>(null);
  const gizmoStartPosition = useRef<Vector3 | null>(null);
  const [gizmoReadout, setGizmoReadout] = useState<string | null>(null);
  const orbitControls = useThree((state) => state.controls) as { enabled: boolean } | null;
  const camera = useThree((state) => state.camera);
  const dragState = useRef<{ pointerId: number; plane: Plane; startPoint: Vector3; startPosition: Vector3 } | null>(null);
  const finishDrag = (flush = false) => {
    const wasDragging = dragState.current !== null;
    dragState.current = null;
    orbitControls && (orbitControls.enabled = true);
    if (flush && wasDragging) onManipulationEnd();
  };
  useEffect(() => {
    const finish = () => finishDrag(true);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    window.addEventListener("blur", finish);
    return () => {
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      window.removeEventListener("blur", finish);
    };
  }, [onManipulationEnd, orbitControls]);
  const startDrag = (event: ThreeEvent<PointerEvent>) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    if (!selected) onSelect(false);
    const origin = new Vector3(microphone.positionX, microphone.positionHeightM, microphone.positionZ);
    const plane = new Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new Vector3()).normalize(), origin);
    const startPoint = event.ray.intersectPlane(plane, new Vector3());
    if (!startPoint) return;
    dragState.current = { pointerId: event.pointerId, plane, startPoint, startPosition: origin };
    orbitControls && (orbitControls.enabled = false);
    (event.target as unknown as { setPointerCapture?: (pointerId: number) => void }).setPointerCapture?.(event.pointerId);
  };
  const moveDrag = (event: ThreeEvent<PointerEvent>) => {
    const drag = dragState.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if ((event.nativeEvent.buttons & 1) === 0) return finishDrag(true);
    event.stopPropagation();
    const point = event.ray.intersectPlane(drag.plane, new Vector3());
    if (!point || !finiteVector(point)) return;
    onTransform({
      positionX: drag.startPosition.x + point.x - drag.startPoint.x,
      positionHeightM: drag.startPosition.y,
      positionZ: drag.startPosition.z + point.z - drag.startPoint.z,
    });
  };
  const endDrag = (event: ThreeEvent<PointerEvent>) => {
    if (!dragState.current || dragState.current.pointerId !== event.pointerId) return;
    event.stopPropagation();
    finishDrag(true);
    (event.target as unknown as { releasePointerCapture?: (pointerId: number) => void }).releasePointerCapture?.(event.pointerId);
  };
  const applyGizmo = () => {
    const object = microphoneRef.current;
    if (!object || !finiteVector(object.position)) return;
    object.position.y = Math.max(0, object.position.y);
    if (gizmoStartPosition.current) {
      setGizmoReadout(translationReadout(gizmoStartPosition.current, object.position, gizmoRef.current?.axis));
    }
    onTransform({ positionX: object.position.x, positionHeightM: object.position.y, positionZ: object.position.z });
  };
  const startGizmo = () => {
    const object = microphoneRef.current;
    if (!object) return;
    gizmoStartPosition.current = object.position.clone();
    setGizmoReadout(translationReadout(object.position, object.position, gizmoRef.current?.axis));
  };
  const finishGizmo = () => {
    gizmoStartPosition.current = null;
    setGizmoReadout(null);
    onManipulationEnd();
  };
  return (
    <>
      <group
        ref={microphoneRef}
        position={[microphone.positionX, microphone.positionHeightM, microphone.positionZ]}
        onClick={(event) => {
          event.stopPropagation();
          onSelect(event.nativeEvent.ctrlKey || event.nativeEvent.metaKey);
        }}
      >
        <mesh
          renderOrder={22}
          onPointerDown={startDrag}
          onPointerMove={moveDrag}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          <sphereGeometry args={[active ? 0.105 : 0.08, 18, 12]} />
          <meshBasicMaterial color={color} depthTest={false} toneMapped={false} />
        </mesh>
        <mesh rotation={[0, 0, Math.PI / 2]} renderOrder={21}>
          <cylinderGeometry args={[0.008, 0.008, 0.42, 8]} />
          <meshBasicMaterial color={selected ? "#ffffff" : color} depthTest={false} toneMapped={false} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]} renderOrder={21}>
          <cylinderGeometry args={[0.008, 0.008, 0.42, 8]} />
          <meshBasicMaterial color={selected ? "#ffffff" : color} depthTest={false} toneMapped={false} />
        </mesh>
        {active && (
          <Html position={[0.18, 0.18, 0]} center distanceFactor={10}>
            <div className="scene-label">{microphone.name.toUpperCase()}</div>
          </Html>
        )}
        <TransformReadout text={gizmoReadout} position={[0, 0.35, 0]} />
      </group>
      {active && transformMode === "translate" && (
        <TransformControls
          ref={(control) => { gizmoRef.current = configureAxisOnlyRotation(control); }}
          object={microphoneRef as MutableRefObject<Group>}
          mode="translate"
          space="world"
          size={0.82}
          showX
          showY
          showZ
          onMouseDown={startGizmo}
          onObjectChange={applyGizmo}
          onMouseUp={finishGizmo}
        />
      )}
    </>
  );
}

function selectionWorldBounds(packages: BoundaryMeshAsset[], instances: readonly SpeakerInstance[]): SceneBounds {
  const minimum: [number, number, number] = [Infinity, Infinity, Infinity];
  const maximum: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const instance of instances) {
    const cabinetBounds = packageSceneBounds(packages.find((pkg) => pkg.id === instance.packageId) ?? packages[0]);
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
  packages,
  instances,
  allInstances,
  transformMode,
  angleSnapDisabled,
  onTransform,
  onManipulationStart,
  onManipulationEnd,
}: {
  packages: BoundaryMeshAsset[];
  instances: SpeakerInstance[];
  allInstances: SpeakerInstance[];
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onTransform: (poses: SourceGroupPoseUpdate[]) => void;
  onManipulationStart: () => void;
  onManipulationEnd: () => void;
}) {
  const pivotRef = useRef<Group>(null);
  const transformControlsRef = useRef<TransformControlsInternal | null>(null);
  const [gizmoReadout, setGizmoReadout] = useState<string | null>(null);
  const camera = useThree((state) => state.camera);
  const gl = useThree((state) => state.gl);
  const orbitControls = useThree((state) => state.controls) as { enabled: boolean } | null;
  const [snapHighlight, setSnapHighlight] = useState<Vector3 | null>(null);
  const bounds = useMemo(() => selectionWorldBounds(packages, instances), [instances, packages]);
  const center = useMemo(() => new Vector3(
    (bounds.minimum[0] + bounds.maximum[0]) / 2,
    (bounds.minimum[1] + bounds.maximum[1]) / 2,
    (bounds.minimum[2] + bounds.maximum[2]) / 2,
  ), [bounds]);
  const dragState = useRef<{
    pointerId: number;
    trackingPlane: Plane;
    probePlane: Plane;
    startTrackingPoint: Vector3;
    startProbePoint: Vector3;
    startCorner: Vector3;
    startCenter: Vector3;
    instances: SpeakerInstance[];
    maximumDeltaM: number;
    snapKey: string | null;
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
    setSnapHighlight(null);
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
    const probePlane = new Plane().setFromNormalAndCoplanarPoint(
      camera.getWorldDirection(new Vector3()).normalize(),
      startCorner,
    );
    // Track directly on the selection's horizontal plane so the handle remains
    // under the cursor. Near the horizon, use the bounded view-facing fallback.
    const groundPlane = new Plane().setFromNormalAndCoplanarPoint(new Vector3(0, 1, 0), startCorner);
    const useGroundPlane = Math.abs(event.ray.direction.y) >= 0.075;
    const trackingPlane = useGroundPlane ? groundPlane : probePlane;
    const startTrackingPoint = event.ray.intersectPlane(trackingPlane, new Vector3());
    const startProbePoint = event.ray.intersectPlane(probePlane, new Vector3());
    if (!startTrackingPoint || !startProbePoint) return;
    dragState.current = {
      pointerId: event.pointerId,
      trackingPlane,
      probePlane,
      startTrackingPoint,
      startProbePoint,
      startCorner,
      startCenter: center.clone(),
      instances: instances.map((instance) => ({ ...instance, position: [...instance.position] })),
      maximumDeltaM: Math.max(10, camera.position.distanceTo(startCorner) * 4),
      snapKey: null,
    };
    onManipulationStart();
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
    const point = event.ray.intersectPlane(drag.trackingPlane, new Vector3());
    if (!point || !finiteVector(point)) return;
    const rawDelta = point.clone().sub(drag.startTrackingPoint);
    if (rawDelta.length() > drag.maximumDeltaM) return;
    const probePoint = event.ray.intersectPlane(drag.probePlane, new Vector3());
    const probeDelta = probePoint && finiteVector(probePoint)
      ? probePoint.sub(drag.startProbePoint)
      : rawDelta;
    const targetCorners: CornerSnapTarget[] = [];
    const movingSigns = cornerSigns(
      [drag.startCorner.x, drag.startCorner.y, drag.startCorner.z],
      bounds,
    );
    const movingAxes: [Vector3, Vector3, Vector3] = [
      new Vector3(1, 0, 0),
      new Vector3(0, 1, 0),
      new Vector3(0, 0, 1),
    ];
    for (const other of allInstances) {
      if (instances.some((instance) => instance.id === other.id)) continue;
      const targetPackage = packages.find((pkg) => pkg.id === other.packageId) ?? packages[0];
      const targetBounds = packageSceneBounds(targetPackage);
      const targetAxes = instanceAxes(other);
      for (const [cornerIndex, targetCorner] of targetBounds.corners.entries()) {
        targetCorners.push({
          key: `${other.id}:${cornerIndex}`,
          position: cornerInWorld(targetCorner, other),
          objectCenter: new Vector3(...other.position),
          paddingDirections: matchingCornerPaddingDirections(
            movingSigns,
            movingAxes,
            cornerSigns(targetCorner, targetBounds),
            targetAxes,
          ),
        });
      }
    }
    const viewport = gl.domElement.getBoundingClientRect();
    const snapTarget = stickyCornerSnapTarget(
      event.nativeEvent.clientX,
      event.nativeEvent.clientY,
      targetCorners,
      camera,
      viewport,
      drag.snapKey,
    );
    const snappedDelta = paddedCornerSnapDelta(
      drag.startCorner,
      drag.startCenter,
      rawDelta,
      probeDelta,
      snapTarget,
    );
    const snappedInstances = drag.instances.map((start) => ({
      ...start,
      position: new Vector3(...start.position).add(snappedDelta).toArray() as [number, number, number],
    }));
    const stationary = allInstances.filter((other) => !instances.some((moving) => moving.id === other.id));
    const validSnap = snapTarget && snappedPosesHaveClearance(packages, snappedInstances, stationary);
    drag.snapKey = validSnap ? snapTarget.key : null;
    setSnapHighlight(validSnap ? snapTarget.position.clone() : null);
    emitTranslation(drag.instances, snapTarget && !validSnap ? rawDelta : snappedDelta);
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
    setGizmoReadout(transformMode === "rotate"
      ? rotationReadout(pivotRef.current.quaternion, transformControlsRef.current?.axis)
      : translationReadout(gizmoState.current.pivot, pivotRef.current.position, transformControlsRef.current?.axis));
    onManipulationStart();
    pivotRef.current.quaternion.identity();
  };

  const applyGizmoTransform = () => {
    const object = pivotRef.current;
    const state = gizmoState.current;
    if (!object || !state) return;
    if (state.mode === "translate") {
      setGizmoReadout(translationReadout(state.pivot, object.position, transformControlsRef.current?.axis));
      emitTranslation(state.instances, object.position.clone().sub(state.pivot));
      return;
    }
    const deltaRotation = object.quaternion.clone().normalize();
    setGizmoReadout(rotationReadout(deltaRotation, transformControlsRef.current?.axis));
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
    setGizmoReadout(null);
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
      <group ref={pivotRef}>
        <TransformReadout text={gizmoReadout} position={[0, 0.4, 0]} />
      </group>
      {(transformMode === "translate" || transformMode === "rotate") && (
        <TransformControls
          ref={(control) => { transformControlsRef.current = configureAxisOnlyRotation(control); }}
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
      {snapHighlight && (
        <mesh position={snapHighlight} renderOrder={30}>
          <sphereGeometry args={[0.11, 16, 12]} />
          <meshBasicMaterial color="#fff36b" wireframe depthTest={false} toneMapped={false} />
        </mesh>
      )}
    </>
  );
}

function AcousticScene(props: SceneViewProps) {
  const selectedInstances = new Set(props.selectedInstances);
  const boundaryAssets: BoundaryMeshAsset[] = [...props.packages, ...props.rigidMeshes];
  const allBoundaryObjects = [...props.sources, ...props.rigidObjects];
  const selectedSources = allBoundaryObjects.filter((source) => selectedInstances.has(source.id));
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
            pkg={props.packages.find((pkg) => pkg.id === source.packageId) ?? props.packages[0]}
            packages={boundaryAssets}
            instance={source}
            allInstances={allBoundaryObjects}
            selected={selectedInstances.has(source.id)}
            active={props.activeInstance === source.id}
            individualControls={!groupedSelection}
            movingInstanceIds={props.selectedInstances}
            transformMode={props.transformMode}
            angleSnapDisabled={props.angleSnapDisabled}
            onSelect={(additive) => props.onSelectInstance(source.id, additive)}
            onTransform={(pose) => props.onTransformSource(source.id, pose)}
            onManipulationStart={() => props.onSourceManipulationStart(
              selectedSources.some((selected) => selected.id === source.id)
                ? selectedSources.map((selected) => selected.id)
                : [source.id],
            )}
            onManipulationEnd={props.onSourceManipulationEnd}
          />
        ))}
      </group>
      <group>
        {props.rigidObjects.map((object) => {
          const asset = props.rigidMeshes.find((candidate) => candidate.id === object.packageId);
          if (!asset) return null;
          return (
            <SpeakerGeometry
              key={object.id}
              pkg={asset}
              packages={boundaryAssets}
              instance={object}
              allInstances={allBoundaryObjects}
              selected={selectedInstances.has(object.id)}
              active={props.activeInstance === object.id}
              individualControls={!groupedSelection}
              movingInstanceIds={props.selectedInstances}
              transformMode={props.transformMode}
              angleSnapDisabled={props.angleSnapDisabled}
              onSelect={(additive) => props.onSelectInstance(object.id, additive)}
              onTransform={(pose) => props.onTransformRigid(object.id, pose)}
              onManipulationStart={() => props.onSourceManipulationStart(
                selectedSources.some((selected) => selected.id === object.id)
                  ? selectedSources.map((selected) => selected.id)
                  : [object.id],
              )}
              onManipulationEnd={props.onSourceManipulationEnd}
            />
          );
        })}
      </group>
      {groupedSelection && (
        <SpeakerSelectionControls
          packages={boundaryAssets}
          instances={selectedSources}
          allInstances={allBoundaryObjects}
          transformMode={props.transformMode}
          angleSnapDisabled={props.angleSnapDisabled}
          onTransform={props.onTransformSources}
          onManipulationStart={() => props.onSourceManipulationStart(selectedSources.map((source) => source.id))}
          onManipulationEnd={props.onSourceManipulationEnd}
        />
      )}
      {props.microphones.map((microphone, index) => (
        <MicrophoneGeometry
          key={microphone.id}
          microphone={microphone}
          color={MICROPHONE_COLORS[index % MICROPHONE_COLORS.length]}
          selected={selectedInstances.has(microphone.id)}
          active={props.activeInstance === microphone.id}
          transformMode={props.transformMode}
          onSelect={(additive) => props.onSelectInstance(microphone.id, additive)}
          onTransform={(pose) => props.onTransformMicrophone(microphone.id, pose)}
          onManipulationEnd={props.onManipulationEnd}
        />
      ))}
      <FieldPlane
        observation={props.observation}
        field={props.field}
        phaseAnimationEnabled={props.phaseAnimationEnabled}
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
