import { Grid, Html, OrbitControls, TransformControls } from "@react-three/drei";
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber";
import { useMemo, useRef, type MutableRefObject } from "react";
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
  RGBAFormat,
  SRGBColorSpace,
  UnsignedByteType,
  Vector3,
} from "three";
import type { FieldFrame, LoadedSpeakerPackage, ObservationPlane, SpeakerInstance } from "../model/types";
import { SOURCE_SURFACE_PADDING_M } from "../model/field";

export type SceneTransformMode = "select" | "translate" | "rotate";

export interface SourcePoseUpdate {
  positionX: number;
  positionHeightM: number;
  positionZ: number;
  yawDeg: number;
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
  const yaw = MathUtils.degToRad(instance.yawDeg);
  const cosine = Math.cos(yaw);
  const sine = Math.sin(yaw);
  return new Vector3(
    cosine * corner[0] + sine * corner[2] + position[0],
    corner[1] + position[1],
    -sine * corner[0] + cosine * corner[2] + position[2],
  );
}

function normalizedYaw(yawDeg: number): number {
  return ((yawDeg + 180) % 360 + 360) % 360 - 180;
}

function pressureColor(normalized: number): [number, number, number] {
  const stops: Array<[number, string]> = [
    [0, "#182a47"],
    [0.2, "#205d77"],
    [0.4, "#2b9b8f"],
    [0.6, "#9ac548"],
    [0.8, "#f2b53f"],
    [1, "#ef5b3f"],
  ];
  const clamped = Math.max(0, Math.min(1, normalized));
  const stopIndex = Math.min(stops.length - 2, Math.floor(clamped * (stops.length - 1)));
  const left = stops[stopIndex];
  const right = stops[stopIndex + 1];
  const local = (clamped - left[0]) / (right[0] - left[0]);
  const a = new Color(left[1]);
  const b = new Color(right[1]);
  a.lerp(b, local);
  return [Math.round(a.r * 255), Math.round(a.g * 255), Math.round(a.b * 255)];
}

function FieldPlane({
  observation,
  field,
  selected,
}: {
  observation: ObservationPlane;
  field: FieldFrame;
  selected: boolean;
}) {
  const texture = useMemo(() => {
    const pixels = new Uint8Array(field.columns * field.rows * 4);
    const low = field.maximumDb - 24;
    const high = field.maximumDb;
    for (let index = 0; index < field.splDb.length; index += 1) {
      const [red, green, blue] = pressureColor((field.splDb[index] - low) / Math.max(1, high - low));
      pixels[index * 4] = red;
      pixels[index * 4 + 1] = green;
      pixels[index * 4 + 2] = blue;
      pixels[index * 4 + 3] = 222;
    }
    const result = new DataTexture(pixels, field.columns, field.rows, RGBAFormat, UnsignedByteType);
    result.needsUpdate = true;
    result.colorSpace = SRGBColorSpace;
    return result;
  }, [field]);

  return (
    <group>
      <mesh
        position={[0, observation.heightM, observation.nearM + observation.depthM / 2]}
        // DataTexture row zero is the near edge of the computed field. A +90°
        // rotation maps the plane's lower V edge toward the source (-scene Z).
        rotation={[Math.PI / 2, 0, 0]}
        raycast={() => undefined}
      >
        <planeGeometry args={[observation.widthM, observation.depthM]} />
        <meshBasicMaterial map={texture} transparent opacity={0.88} side={DoubleSide} toneMapped={false} />
      </mesh>
      <Grid
        position={[0, observation.heightM + 0.008, observation.nearM + observation.depthM / 2]}
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
    </group>
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
  const camera = useThree((state) => state.camera);
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
      new Euler(MathUtils.degToRad(instance.pitchDeg), MathUtils.degToRad(instance.yawDeg), 0, "YXZ"),
    ),
    [instance.pitchDeg, instance.yawDeg],
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
  ): [number, number, number] => {
    const movingCorner = cornerInWorld(draggedCorner, instance, rawPosition);
    let bestDistance = 0.2;
    let result = rawPosition;
    for (const other of allInstances) {
      if (other.id === instance.id) continue;
      for (const targetCorner of bounds.corners) {
        const target = cornerInWorld(targetCorner, other);
        const distance = target.distanceTo(movingCorner);
        if (distance >= bestDistance) continue;
        const away = new Vector3(
          rawPosition[0] - other.position[0],
          rawPosition[1] - other.position[1],
          rawPosition[2] - other.position[2],
        );
        if (away.lengthSq() > 1e-10) away.normalize().multiplyScalar(SOURCE_SURFACE_PADDING_M);
        result = [
          rawPosition[0] + target.x - movingCorner.x + away.x,
          rawPosition[1] + target.y - movingCorner.y + away.y,
          rawPosition[2] + target.z - movingCorner.z + away.z,
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
    const plane = new Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new Vector3()), handleWorld);
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
      Math.max(SOURCE_SURFACE_PADDING_M - bounds.minimum[1], drag.startPosition[1] + point.y - drag.startPoint.y),
      drag.startPosition[2] + point.z - drag.startPoint.z,
    ];
    const snapped = snapDraggedCorner(rawPosition, drag.corner);
    onTransform({
      positionX: snapped[0],
      positionHeightM: snapped[1],
      positionZ: snapped[2],
      yawDeg: instance.yawDeg,
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
    const minimumHeight = SOURCE_SURFACE_PADDING_M - bounds.minimum[1];
    object.position.y = Math.max(minimumHeight, object.position.y);
    const rotation = new Euler().setFromQuaternion(object.quaternion, "YXZ");
    onTransform({
      positionX: object.position.x,
      positionHeightM: object.position.y,
      positionZ: object.position.z,
      yawDeg: normalizedYaw(MathUtils.radToDeg(rotation.y)),
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
      {selected && transformMode !== "select" && (
        <TransformControls
          object={speakerRef as MutableRefObject<Group>}
          mode={transformMode}
          space="world"
          size={0.82}
          showX={transformMode === "translate"}
          showY
          showZ={transformMode === "translate"}
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
      <group onPointerMissed={() => props.onSelectInstance(null)}>
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
