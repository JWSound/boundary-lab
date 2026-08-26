import { Grid, Html, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DataTexture,
  DoubleSide,
  Euler,
  MathUtils,
  Quaternion,
  RGBAFormat,
  SRGBColorSpace,
  UnsignedByteType,
} from "three";
import type { FieldFrame, LoadedSpeakerPackage, ObservationPlane, SpeakerInstance } from "../model/types";

interface SceneViewProps {
  pkg: LoadedSpeakerPackage;
  sources: SpeakerInstance[];
  observation: ObservationPlane;
  field: FieldFrame;
  selectedInstance: string | null;
  onSelectInstance: (id: string | null) => void;
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

function FieldPlane({ observation, field }: { observation: ObservationPlane; field: FieldFrame }) {
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
      >
        <planeGeometry args={[observation.widthM, observation.depthM]} />
        <meshBasicMaterial map={texture} transparent opacity={0.88} side={DoubleSide} toneMapped={false} />
      </mesh>
      <Grid
        position={[0, observation.heightM + 0.008, observation.nearM + observation.depthM / 2]}
        args={[observation.widthM, observation.depthM]}
        cellSize={1}
        cellThickness={0.35}
        cellColor="#d9e0d0"
        sectionSize={5}
        sectionThickness={0.8}
        sectionColor="#f1f6ea"
        fadeDistance={40}
        fadeStrength={1}
        infiniteGrid={false}
      />
    </group>
  );
}

function SpeakerGeometry({
  pkg,
  instance,
  selected,
  onSelect,
}: {
  pkg: LoadedSpeakerPackage;
  instance: SpeakerInstance;
  selected: boolean;
  onSelect: () => void;
}) {
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
  const sceneBounds: [number, number, number] = [pkg.boundsM[0], pkg.boundsM[2], pkg.boundsM[1]];

  return (
    <group position={instance.position} quaternion={quaternion} onClick={(event) => { event.stopPropagation(); onSelect(); }}>
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
        <mesh castShadow receiveShadow>
          <boxGeometry args={sceneBounds} />
          <meshStandardMaterial color={selected ? "#b8c986" : "#3a403b"} roughness={0.72} />
        </mesh>
      )}
      {pkg.isDemo && (
        <>
          <mesh position={[0, 0.055, sceneBounds[2] / 2 + 0.003]}>
            <circleGeometry args={[sceneBounds[0] * 0.27, 40]} />
            <meshStandardMaterial color="#171b18" roughness={0.5} />
          </mesh>
          <mesh position={[0, -0.075, sceneBounds[2] / 2 + 0.005]}>
            <circleGeometry args={[sceneBounds[0] * 0.12, 36]} />
            <meshStandardMaterial color="#252b26" roughness={0.4} />
          </mesh>
        </>
      )}
      {selected && (
        <Html position={[sceneBounds[0] / 2 + 0.08, sceneBounds[1] / 2, 0]} center distanceFactor={10}>
          <div className="scene-label">{instance.id.toUpperCase()}</div>
        </Html>
      )}
    </group>
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
            selected={props.selectedInstance === source.id}
            onSelect={() => props.onSelectInstance(source.id)}
          />
        ))}
      </group>
      <FieldPlane observation={props.observation} field={props.field} />
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
