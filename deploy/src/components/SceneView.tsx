import { Grid, Html, OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type MutableRefObject } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DataTexture,
  DoubleSide,
  Euler,
  Group,
  MathUtils,
  Plane,
  Quaternion,
  Ray,
  RGBAFormat,
  SRGBColorSpace,
  UnsignedByteType,
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
  selectedInstance: string | null;
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onSelectInstance: (id: string | null) => void;
  onTransformSource: (id: string, pose: SourcePoseUpdate) => void;
  onTransformObservation: (pose: ObservationPoseUpdate) => void;
  onResizeObservation: (resize: ObservationResizeUpdate) => void;
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

const PRESSURE_COLOR_STOPS: Array<[number, number, number, number]> = ([
  [0, "#182a47"],
  [0.2, "#205d77"],
  [0.4, "#2b9b8f"],
  [0.6, "#9ac548"],
  [0.8, "#f2b53f"],
  [1, "#ef5b3f"],
] satisfies Array<[number, string]>).map(([position, value]) => {
  const color = new Color(value);
  return [position, color.r * 255, color.g * 255, color.b * 255] as [number, number, number, number];
});

function writePressureColor(normalized: number, pixels: Uint8Array, offset: number): void {
  const clamped = Math.max(0, Math.min(1, normalized));
  const stopIndex = Math.min(
    PRESSURE_COLOR_STOPS.length - 2,
    Math.floor(clamped * (PRESSURE_COLOR_STOPS.length - 1)),
  );
  const left = PRESSURE_COLOR_STOPS[stopIndex];
  const right = PRESSURE_COLOR_STOPS[stopIndex + 1];
  const local = (clamped - left[0]) / (right[0] - left[0]);
  pixels[offset] = Math.round(left[1] + (right[1] - left[1]) * local);
  pixels[offset + 1] = Math.round(left[2] + (right[2] - left[2]) * local);
  pixels[offset + 2] = Math.round(left[3] + (right[3] - left[3]) * local);
}

function FieldPlane({
  observation,
  field,
  selected,
  transformMode,
  angleSnapDisabled,
  onTransform,
  onResize,
  onTextureReady,
}: {
  observation: ObservationPlane;
  field: FieldFrame;
  selected: boolean;
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onTransform: (pose: ObservationPoseUpdate) => void;
  onResize: (resize: ObservationResizeUpdate) => void;
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

  const cancelResize = () => {
    resizeState.current = null;
    orbitControls && (orbitControls.enabled = true);
  };

  useEffect(() => {
    const finish = () => cancelResize();
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    window.addEventListener("blur", finish);
    return () => {
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      window.removeEventListener("blur", finish);
    };
  }, [orbitControls]);

  useEffect(() => {
    if (transformMode !== "scale") cancelResize();
  }, [transformMode]);
  const texture = useMemo(() => {
    const rasterStarted = performance.now();
    const pixels = new Uint8Array(field.columns * field.rows * 4);
    const low = field.maximumDb - 24;
    const high = field.maximumDb;
    const range = Math.max(1, high - low);
    for (let index = 0; index < field.splDb.length; index += 1) {
      const offset = index * 4;
      writePressureColor((field.splDb[index] - low) / range, pixels, offset);
      pixels[offset + 3] = field.validMask[index] ? 222 : 0;
    }
    const result = new DataTexture(pixels, field.columns, field.rows, RGBAFormat, UnsignedByteType);
    result.needsUpdate = true;
    result.colorSpace = SRGBColorSpace;
    textureProfile.current = {
      pointCount: field.columns * field.rows,
      textureBytes: pixels.byteLength,
      rasterMs: performance.now() - rasterStarted,
    };
    return result;
  }, [field]);

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
      texture.dispose();
    };
  }, [onTextureReady, texture]);

  const applyGizmoTransform = () => {
    const object = planeRef.current;
    if (!object) return;
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
      cancelResize();
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
    cancelResize();
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
          <meshBasicMaterial map={texture} transparent opacity={0.88} side={DoubleSide} toneMapped={false} />
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
        {selected && transformMode === "scale" && [
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
      {selected && (transformMode === "translate" || transformMode === "rotate") && (
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
  transformMode,
  angleSnapDisabled,
  onSelect,
  onTransform,
}: {
  pkg: LoadedSpeakerPackage;
  instance: SpeakerInstance;
  allInstances: SpeakerInstance[];
  selected: boolean;
  transformMode: SceneTransformMode;
  angleSnapDisabled: boolean;
  onSelect: () => void;
  onTransform: (pose: SourcePoseUpdate) => void;
}) {
  const speakerRef = useRef<Group>(null);
  const orbitControls = useThree((state) => state.controls) as { enabled: boolean } | null;
  const dragState = useRef<{
    pointerId: number;
    plane: Plane;
    startPoint: Vector3;
    startPosition: [number, number, number];
    corner: [number, number, number];
  } | null>(null);
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
      if (other.id === instance.id) continue;
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
    onSelect();
    const handleWorld = cornerInWorld(corner, instance);
    const plane = new Plane(new Vector3(0, 1, 0), -handleWorld.y);
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
    event.stopPropagation();
    const point = event.ray.intersectPlane(drag.plane, new Vector3());
    if (!point) return;
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
    dragState.current = null;
    orbitControls && (orbitControls.enabled = true);
    (event.target as unknown as { releasePointerCapture?: (pointerId: number) => void }).releasePointerCapture?.(event.pointerId);
  };

  const applyGizmoTransform = () => {
    const object = speakerRef.current;
    if (!object) return;
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
      <group ref={speakerRef} position={instance.position} quaternion={quaternion} onClick={(event) => { event.stopPropagation(); onSelect(); }}>
        {geometry ? (
          <mesh geometry={geometry} castShadow receiveShadow>
            <meshStandardMaterial
              color={selected ? "#b8c986" : "#3a403b"}
              roughness={0.72}
              metalness={0.08}
              emissive={selected ? "#2a3218" : "#000000"}
            />
          </mesh>
        ) : (
          <mesh position={boundsCenter} castShadow receiveShadow>
            <boxGeometry args={sceneBounds} />
            <meshStandardMaterial color={selected ? "#b8c986" : "#3a403b"} roughness={0.72} />
          </mesh>
        )}
        {pkg.isDemo && (
          <>
            <mesh position={[0, 0.055, bounds.maximum[2] + 0.003]}>
              <circleGeometry args={[sceneBounds[0] * 0.27, 40]} />
              <meshStandardMaterial color="#171b18" roughness={0.5} />
            </mesh>
            <mesh position={[0, -0.075, bounds.maximum[2] + 0.005]}>
              <circleGeometry args={[sceneBounds[0] * 0.12, 36]} />
              <meshStandardMaterial color="#252b26" roughness={0.4} />
            </mesh>
          </>
        )}
        {selected && bounds.corners.map((corner, index) => (
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
        {selected && (
          <Html position={[bounds.maximum[0] + 0.08, bounds.maximum[1], boundsCenter[2]]} center distanceFactor={10}>
            <div className="scene-label">{instance.id.toUpperCase()}</div>
          </Html>
        )}
      </group>
      {selected && (transformMode === "translate" || transformMode === "rotate") && (
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
        />
      )}
    </>
  );
}

function AcousticScene(props: SceneViewProps) {
  return (
    <>
      <color attach="background" args={["#171b18"]} />
      <fog attach="fog" args={["#171b18", 16, 48]} />
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
            selected={props.selectedInstance === source.id}
            transformMode={props.transformMode}
            angleSnapDisabled={props.angleSnapDisabled}
            onSelect={() => props.onSelectInstance(source.id)}
            onTransform={(pose) => props.onTransformSource(source.id, pose)}
          />
        ))}
      </group>
      <FieldPlane
        observation={props.observation}
        field={props.field}
        selected={props.selectedInstance === "audience-plane"}
        transformMode={props.transformMode}
        angleSnapDisabled={props.angleSnapDisabled}
        onTransform={props.onTransformObservation}
        onResize={props.onResizeObservation}
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
        maxDistance={45}
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
      camera={{ position: [9.5, 7.5, 13.5], fov: 44, near: 0.05, far: 100 }}
      gl={{ antialias: true, alpha: false }}
    >
      <AcousticScene {...props} />
    </Canvas>
  );
}
