import {
  ChevronRight,
  CircleHelp,
  Box,
  FolderOpen,
  Grid3X3,
  Import,
  Maximize2,
  Menu,
  Mic2,
  MousePointer2,
  Move3D,
  Pause,
  Play,
  Plus,
  Rotate3D,
  Save,
  Copy,
  Settings2,
  SlidersHorizontal,
  Speaker,
  Trash2,
  Waves,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import { SceneView, type FieldTextureProfile, type ObservationResizeUpdate, type SceneTransformMode, type SourceGroupPoseUpdate, type SourcePoseUpdate } from "./components/SceneView";
import { type BemResponseData, MicrophoneResponsePlot } from "./components/MicrophoneResponsePlot";
import { DriverExcursionPlot, type DriverExcursionData } from "./components/DriverExcursionPlot";
import { ElectricalPlot, type ElectricalData, type ElectricalTrace } from "./components/ElectricalPlot";
import {
  browserFileHandler,
  MicrophoneInspector,
  PackageCard,
  planeGridShape,
  PlaneResolutionInspector,
  SceneTree,
  SectionHeader,
  SourceInspector,
  RigidMeshCard,
  RigidMeshInspector,
} from "./components/Controls";
import { loadSpeakerPackage } from "./io/speakerPackage";
import { loadRigidMesh } from "./io/rigidMesh";
import { createDeployProject, parseDeployProject, serializeDeployProject, type DeployProject } from "./io/deployProject";
import { createDemoPackage } from "./model/demoPackage";
import {
  buildPackagePatternLookups,
  buildSourceInstance,
  computeMixedFieldFrame,
  computeMixedMicrophonePatternResponses,
  fieldFrameFromSpl,
  minimumSourceHeightM,
  nearestFrequencyIndex,
} from "./model/field";
import type { Fidelity, FieldFrame, LoadedSpeakerPackage, MicrophoneConfiguration, ObservationPlane, RigidMeshAsset, RigidMeshConfiguration, SourceConfiguration } from "./model/types";
import { heatmapLegendGradient } from "./model/heatmap";
import { cabinetClearanceViolations, constrainCabinetPoses, findClearSourcePlacement, type BoundaryMeshAsset } from "./model/cabinetPlacement";

function peakExcursionMillimeters(real: number, imag: number, frequencyHz: number): number {
  return Math.SQRT2 * Math.hypot(real, imag) * 1000 / (2 * Math.PI * frequencyHz);
}

function electricalSample(voltageReal: number, voltageImag: number, currentReal: number, currentImag: number) {
  const currentMagnitude = Math.hypot(currentReal, currentImag);
  if (currentMagnitude <= Number.EPSILON) return {
    impedanceMagnitudeOhm: Number.NaN,
    impedancePhaseDeg: Number.NaN,
    rmsCurrentA: 0,
    realPowerW: 0,
  };
  const impedanceReal = (voltageReal * currentReal + voltageImag * currentImag) / (currentMagnitude * currentMagnitude);
  const impedanceImag = (voltageImag * currentReal - voltageReal * currentImag) / (currentMagnitude * currentMagnitude);
  return {
    impedanceMagnitudeOhm: Math.hypot(impedanceReal, impedanceImag),
    impedancePhaseDeg: Math.atan2(impedanceImag, impedanceReal) * 180 / Math.PI,
    rmsCurrentA: currentMagnitude,
    realPowerW: voltageReal * currentReal + voltageImag * currentImag,
  };
}

type SolvedFieldEntry = { key: string; field: FieldFrame };
type SolvedFieldCache = Record<"boundary" | "coupled", SolvedFieldEntry | null>;

const emptySolvedFieldCache = (): SolvedFieldCache => ({ boundary: null, coupled: null });

function defaultSources(pkg: LoadedSpeakerPackage): SourceConfiguration[] {
  const centerSpacingM = pkg.boundsM[0] + 2;
  const positionHeightM = minimumSourceHeightM(pkg);
  return [
    {
      id: "subwoofer-1",
      name: `${pkg.manifest.name} 1`,
      packageId: pkg.id,
      positionX: -centerSpacingM / 2,
      positionHeightM,
      positionZ: 0,
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
      levelDb: -3,
      delayMs: 0,
      polarity: 1,
    },
    {
      id: "subwoofer-2",
      name: `${pkg.manifest.name} 2`,
      packageId: pkg.id,
      positionX: centerSpacingM / 2,
      positionHeightM,
      positionZ: 0,
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
      levelDb: -3,
      delayMs: 0,
      polarity: 1,
    },
  ];
}

function buildRigidInstance(config: RigidMeshConfiguration) {
  return {
    id: config.id,
    packageId: config.assetId,
    position: [config.positionX, config.positionHeightM, config.positionZ] as [number, number, number],
    pitchDeg: config.pitchDeg,
    yawDeg: config.yawDeg,
    rollDeg: config.rollDeg,
  };
}

const defaultObservation: ObservationPlane = {
  widthM: 24,
  depthM: 24,
  centerXM: 0,
  nearM: 1.5,
  heightM: 1.2,
  pitchDeg: 0,
  yawDeg: 0,
  rollDeg: 0,
  columns: 54,
  rows: 54,
  heatmapMinimumDb: 50,
  heatmapMaximumDb: 145,
  heatmapBandingDb: 0,
  displayMode: "spl",
  pressureScalePa: 10,
  phaseAnimationSpeedHz: 1,
};

function observationAcousticState(value: ObservationPlane) {
  return {
    widthM: value.widthM,
    depthM: value.depthM,
    centerXM: value.centerXM,
    nearM: value.nearM,
    heightM: value.heightM,
    pitchDeg: value.pitchDeg,
    yawDeg: value.yawDeg,
    rollDeg: value.rollDeg,
    columns: value.columns,
    rows: value.rows,
  };
}

function formatFrequency(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)} kHz` : `${Math.round(value)} Hz`;
}

function FidelitySwitcher({
  value,
  onChange,
  packageLevel,
  boundaryAvailable,
  boundaryUnavailableReason,
  coupledAvailable,
  coupledUnavailableReason,
}: {
  value: Fidelity;
  onChange: (value: Fidelity) => void;
  packageLevel: number;
  boundaryAvailable: boolean;
  boundaryUnavailableReason?: string;
  coupledAvailable: boolean;
  coupledUnavailableReason?: string;
}) {
  const levels: Array<{ id: Fidelity; label: string; level: number }> = [
    { id: "pattern", label: "Pattern", level: 1 },
    { id: "boundary", label: "Boundary", level: 2 },
    { id: "coupled", label: "Coupled", level: 3 },
  ];
  return (
    <div className="fidelity-switcher">
      {levels.map((item) => {
        const available = item.level <= packageLevel;
        const interactive = item.id === "pattern" ||
          (item.id === "boundary" && available && boundaryAvailable) ||
          (item.id === "coupled" && available && coupledAvailable);
        return (
          <button
            key={item.id}
            className={`${value === item.id ? "active" : ""} ${!interactive ? "engine-required" : ""}`}
            onClick={() => interactive && onChange(item.id)}
            title={interactive
              ? (item.id === "boundary"
                  ? "Exterior BEM with fixed distributed sources"
                  : item.id === "coupled"
                    ? "Exact coupled FEM–BEM interiors and transducers"
                    : "Live complex pattern field")
              : item.id === "boundary" && boundaryUnavailableReason
                ? boundaryUnavailableReason
                : item.id === "coupled" && coupledUnavailableReason
                  ? coupledUnavailableReason
                : available ? "This fidelity is not connected yet" : "Package does not contain this fidelity"}
          >
            <span>{item.label}</span>
            {item.id !== "pattern" && interactive && <small>CUDA</small>}
            {!interactive && item.id !== "pattern" && <small>{available ? "ENGINE" : "N/A"}</small>}
          </button>
        );
      })}
    </div>
  );
}

export function App() {
  const [packages, setPackages] = useState<LoadedSpeakerPackage[]>(() => [createDemoPackage()]);
  const [activePackageId, setActivePackageId] = useState(() => packages[0].id);
  const pkg = packages.find((candidate) => candidate.id === activePackageId) ?? packages[0];
  const [sourceConfigs, setSourceConfigs] = useState<SourceConfiguration[]>(() => defaultSources(pkg));
  const [rigidMeshes, setRigidMeshes] = useState<RigidMeshAsset[]>([]);
  const [activeRigidMeshId, setActiveRigidMeshId] = useState<string | null>(null);
  const [rigidObjects, setRigidObjects] = useState<RigidMeshConfiguration[]>([]);
  const [microphones, setMicrophones] = useState<MicrophoneConfiguration[]>([]);
  const [observation, setObservation] = useState(defaultObservation);
  const [phaseAnimationEnabled, setPhaseAnimationEnabled] = useState(false);
  const [frequencyIndex, setFrequencyIndex] = useState(() => nearestFrequencyIndex(pkg, 80));
  const [fidelity, setFidelity] = useState<Fidelity>("pattern");
  const [selectedInstances, setSelectedInstances] = useState<string[]>(["subwoofer-1"]);
  const [error, setError] = useState<string | null>(null);
  const [leftTab, setLeftTab] = useState<"library" | "scene">("library");
  const [solvedFields, setSolvedFields] = useState<SolvedFieldCache>(emptySolvedFieldCache);
  const [boundaryGeometryKey, setBoundaryGeometryKey] = useState<string | null>(null);
  const [solveRevision, setSolveRevision] = useState(0);
  const [solveState, setSolveState] = useState<"idle" | "solving" | "complete" | "error">("idle");
  const [solveMessage, setSolveMessage] = useState("Ready to solve");
  const [liveSolveEnabled, setLiveSolveEnabled] = useState(false);
  const [transformMode, setTransformMode] = useState<SceneTransformMode>("select");
  const [angleSnapDisabled, setAngleSnapDisabled] = useState(false);
  const [projectName, setProjectName] = useState("S218BP Subwoofer Study");
  const [projectFileName, setProjectFileName] = useState("s218bp-subwoofer-study.blabdeploy.json");
  const [savedProjectSnapshot, setSavedProjectSnapshot] = useState<string | null>(null);
  const [solveReleaseRevision, setSolveReleaseRevision] = useState(0);
  const [speakerManipulationActive, setSpeakerManipulationActive] = useState(false);
  const [microphoneSweepState, setMicrophoneSweepState] = useState<"idle" | "solving" | "complete" | "error">("idle");
  const [microphoneSweepProgress, setMicrophoneSweepProgress] = useState({ completed: 0, total: 0 });
  const [bemMicrophoneResponses, setBemMicrophoneResponses] = useState<BemResponseData | null>(null);
  const [driverExcursion, setDriverExcursion] = useState<DriverExcursionData | null>(null);
  const [electricalResponse, setElectricalResponse] = useState<ElectricalData | null>(null);
  const [analysisTab, setAnalysisTab] = useState<"microphones" | "excursion" | "electrical">("microphones");
  const [analysisDrawerHeight, setAnalysisDrawerHeight] = useState(220);
  const [analysisDrawerResizing, setAnalysisDrawerResizing] = useState(false);
  const packageFileInput = useRef<HTMLInputElement>(null);
  const rigidMeshFileInput = useRef<HTMLInputElement>(null);
  const projectFileInput = useRef<HTMLInputElement>(null);
  const solveGeneration = useRef(0);
  const pendingRenderProfile = useRef<Record<string, unknown> | null>(null);
  const sourceConfigsRef = useRef(sourceConfigs);
  const rigidObjectsRef = useRef(rigidObjects);
  const observationRef = useRef(observation);
  const microphonesRef = useRef(microphones);
  const microphoneSweepKeyRef = useRef<string | null>(null);
  const stoppingMicrophoneSweep = useRef(false);
  const flushLiveSolveRef = useRef(false);
  const sourceManipulationRef = useRef<{ start: SourceConfiguration[]; ids: Set<string> } | null>(null);
  const analysisResizeRef = useRef<{ startY: number; startHeight: number } | null>(null);

  const clampAnalysisDrawerHeight = useCallback((height: number) => {
    const maximum = Math.max(150, window.innerHeight - 58 - 180);
    return Math.round(Math.max(150, Math.min(maximum, height)));
  }, []);

  const beginAnalysisResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    analysisResizeRef.current = { startY: event.clientY, startHeight: analysisDrawerHeight };
    setAnalysisDrawerResizing(true);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }, [analysisDrawerHeight]);

  const moveAnalysisResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const resize = analysisResizeRef.current;
    if (!resize) return;
    setAnalysisDrawerHeight(clampAnalysisDrawerHeight(resize.startHeight + resize.startY - event.clientY));
  }, [clampAnalysisDrawerHeight]);

  const finishAnalysisResize = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    analysisResizeRef.current = null;
    setAnalysisDrawerResizing(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }, []);

  useEffect(() => {
    const clampOnResize = () => setAnalysisDrawerHeight((height) => clampAnalysisDrawerHeight(height));
    window.addEventListener("resize", clampOnResize);
    return () => window.removeEventListener("resize", clampOnResize);
  }, [clampAnalysisDrawerHeight]);

  useEffect(() => {
    sourceConfigsRef.current = sourceConfigs;
  }, [sourceConfigs]);

  useEffect(() => {
    rigidObjectsRef.current = rigidObjects;
  }, [rigidObjects]);

  useEffect(() => {
    observationRef.current = observation;
  }, [observation]);

  useEffect(() => {
    microphonesRef.current = microphones;
  }, [microphones]);

  const recordFieldTexture = useCallback((texture: FieldTextureProfile) => {
    const pending = pendingRenderProfile.current;
    if (!pending) return;
    window.boundaryLabDeployProfile = {
      ...pending,
      renderer: {
        ...((pending.renderer as Record<string, unknown> | undefined) ?? {}),
        heatmap_point_count: texture.pointCount,
        heatmap_texture_bytes: texture.textureBytes,
        heatmap_raster_s: texture.rasterMs / 1000,
        heatmap_commit_to_frame_s: texture.commitToFrameMs / 1000,
      },
      texture_ready: true,
    };
    pendingRenderProfile.current = null;
  }, []);

  const packageById = useMemo(() => new Map(packages.map((item) => [item.id, item])), [packages]);
  const rigidMeshById = useMemo(() => new Map(rigidMeshes.map((item) => [item.id, item])), [rigidMeshes]);
  const boundaryAssetById = useMemo(() => new Map<string, BoundaryMeshAsset>([
    ...packages.map((item) => [item.id, item] as const),
    ...rigidMeshes.map((item) => [item.id, item] as const),
  ]), [packages, rigidMeshes]);
  const rigidInstances = useMemo(() => rigidObjects.map(buildRigidInstance), [rigidObjects]);
  const constrainSourceConfigs = useCallback((
    current: SourceConfiguration[],
    proposed: SourceConfiguration[],
    movingIds: ReadonlySet<string>,
  ): SourceConfiguration[] => {
    const resolved = constrainCabinetPoses(
      boundaryAssetById,
      [...current.map(buildSourceInstance), ...rigidObjectsRef.current.map(buildRigidInstance)],
      proposed.filter((source) => movingIds.has(source.id)).map(buildSourceInstance),
    );
    const poseById = new Map(resolved.map((source) => [source.id, source]));
    return proposed.map((source) => {
      const pose = poseById.get(source.id);
      return pose ? {
        ...source,
        positionX: pose.position[0],
        positionHeightM: pose.position[1],
        positionZ: pose.position[2],
        pitchDeg: pose.pitchDeg,
        yawDeg: pose.yawDeg,
        rollDeg: pose.rollDeg,
      } : source;
    });
  }, [boundaryAssetById]);
  const activeSourcePackageIds = [...new Set(sourceConfigs.map((source) => source.packageId))];
  const activeSourcePackageIdsKey = JSON.stringify(activeSourcePackageIds.slice().sort());
  const acousticPackages = useMemo(
    () => (JSON.parse(activeSourcePackageIdsKey) as string[]).map((id) => packageById.get(id)).filter(Boolean) as LoadedSpeakerPackage[],
    [activeSourcePackageIdsKey, packageById],
  );
  const commonMinimumFrequencyHz = Math.max(...acousticPackages.map((item) => Math.min(...item.frequenciesHz)));
  const commonMaximumFrequencyHz = Math.min(...acousticPackages.map((item) => Math.max(...item.frequenciesHz)));
  const sortedFrequencyIndices = useMemo(
    () => Array.from(pkg.frequenciesHz.keys())
      .filter((index) => pkg.frequenciesHz[index] >= commonMinimumFrequencyHz && pkg.frequenciesHz[index] <= commonMaximumFrequencyHz)
      .sort((a, b) => pkg.frequenciesHz[a] - pkg.frequenciesHz[b]),
    [commonMaximumFrequencyHz, commonMinimumFrequencyHz, pkg],
  );
  const usableFrequencyIndices = sortedFrequencyIndices.length > 0
    ? sortedFrequencyIndices
    : Array.from(pkg.frequenciesHz.keys()).sort((a, b) => pkg.frequenciesHz[a] - pkg.frequenciesHz[b]);
  const sortedPosition = Math.max(0, usableFrequencyIndices.indexOf(frequencyIndex));
  useEffect(() => {
    if (!usableFrequencyIndices.includes(frequencyIndex)) setFrequencyIndex(usableFrequencyIndices[0]);
  }, [frequencyIndex, usableFrequencyIndices]);
  const packageForSource = useCallback(
    (source: SourceConfiguration) => packageById.get(source.packageId) ?? pkg,
    [packageById, pkg],
  );
  const sources = useMemo(() => sourceConfigs.map(buildSourceInstance), [sourceConfigs]);
  const sceneClearanceValid = useMemo(
    () => cabinetClearanceViolations(boundaryAssetById, [...sources, ...rigidInstances]).length === 0,
    [boundaryAssetById, rigidInstances, sources],
  );
  const selectedFrequencyHz = pkg.frequenciesHz[frequencyIndex];
  const patternLookups = useMemo(
    () => buildPackagePatternLookups(acousticPackages, selectedFrequencyHz),
    [acousticPackages, selectedFrequencyHz],
  );
  const selectedInstance = selectedInstances.at(-1) ?? null;
  const selectedSourceIndex = sourceConfigs.findIndex((source) => source.id === selectedInstance);
  const selectedSource = selectedSourceIndex >= 0 ? sourceConfigs[selectedSourceIndex] : null;
  const selectedRigidIndex = rigidObjects.findIndex((object) => object.id === selectedInstance);
  const selectedRigid = selectedRigidIndex >= 0 ? rigidObjects[selectedRigidIndex] : null;
  const selectedMicrophoneIndex = microphones.findIndex((microphone) => microphone.id === selectedInstance);
  const selectedMicrophone = selectedMicrophoneIndex >= 0 ? microphones[selectedMicrophoneIndex] : null;
  const selectedSourceIds = useMemo(
    () => selectedInstances.filter((id) => sourceConfigs.some((source) => source.id === id)),
    [selectedInstances, sourceConfigs],
  );
  const selectedRigidIds = useMemo(
    () => selectedInstances.filter((id) => rigidObjects.some((object) => object.id === id)),
    [rigidObjects, selectedInstances],
  );
  const selectedMicrophoneIds = useMemo(
    () => selectedInstances.filter((id) => microphones.some((microphone) => microphone.id === id)),
    [microphones, selectedInstances],
  );
  const selectedSourcePackage = selectedSource ? packageById.get(selectedSource.packageId) ?? pkg : pkg;
  const sourceMinimumHeightM = minimumSourceHeightM(selectedSourcePackage);
  const observationAcousticKey = JSON.stringify(observationAcousticState(observation));
  const patternField = useMemo(
    () => computeMixedFieldFrame(packageById, patternLookups, sources, sourceConfigs, observation, selectedFrequencyHz),
    [packageById, patternLookups, sources, sourceConfigs, observationAcousticKey, selectedFrequencyHz],
  );
  const microphonePatternResponses = useMemo(
    () => computeMixedMicrophonePatternResponses(packageById, sources, sourceConfigs, microphones),
    [packageById, sources, sourceConfigs, microphones],
  );
  const microphoneSweepKey = useMemo(() => JSON.stringify({
    fidelity,
    packages: packages.map((item) => ({ id: item.id, sourcePath: item.sourcePath })),
    sources: sourceConfigs,
    rigidObjects,
    microphones,
    frequencies: Array.from(microphonePatternResponses.frequenciesHz),
  }), [fidelity, microphonePatternResponses.frequenciesHz, microphones, packages, rigidObjects, sourceConfigs]);
  const currentSolveKey = useMemo(() => JSON.stringify({
    fidelity,
    packages: sourceConfigs.map((source) => source.packageId),
    frequency: pkg.frequenciesHz[frequencyIndex],
    sources: sourceConfigs,
    rigidObjects,
    observation: observationAcousticKey,
  }), [fidelity, pkg.id, pkg.frequenciesHz, frequencyIndex, rigidObjects, sourceConfigs, observationAcousticKey]);
  const currentGeometryKey = useMemo(() => JSON.stringify({
    fidelity,
    packages: sourceConfigs.map((source) => source.packageId),
    frequency: pkg.frequenciesHz[frequencyIndex],
    sources: sourceConfigs,
    rigidObjects,
  }), [fidelity, pkg.id, pkg.frequenciesHz, frequencyIndex, rigidObjects, sourceConfigs]);
  const selectedSolvedField = fidelity === "pattern" ? null : solvedFields[fidelity];
  const field = selectedSolvedField?.key === currentSolveKey
    ? selectedSolvedField.field
    : patternField;
  const boundaryCurrent = selectedSolvedField?.key === currentSolveKey;
  const level2Package = activeSourcePackageIds.length === 1 ? packageById.get(activeSourcePackageIds[0]) ?? null : null;
  const level2FrequencyAvailable = Boolean(level2Package && Array.from(level2Package.frequenciesHz).some(
    (frequency) => Math.abs(frequency - selectedFrequencyHz) <= Math.max(1e-4, selectedFrequencyHz * 1e-6),
  ));
  const rigidMeshesAvailable = rigidObjects.every((object) => Boolean(rigidMeshById.get(object.assetId)?.sourcePath));
  const boundaryAvailable = Boolean(
    window.boundaryLabDesktop && level2Package?.sourcePath && level2Package.manifest.fidelity_level >= 2 && level2FrequencyAvailable && rigidMeshesAvailable,
  );
  const coupledRepresentation = level2Package?.manifest.files.coupled_model?.representation;
  const coupledRepresentationSupported = coupledRepresentation === "parity_petrov_galerkin_rom";
  const coupledAvailable = Boolean(
    boundaryAvailable && level2Package && level2Package.manifest.fidelity_level >= 3 &&
    coupledRepresentationSupported,
  );
  const scenePackageLevel = Math.min(...sourceConfigs.map(
    (source) => packageById.get(source.packageId)?.manifest.fidelity_level ?? 1,
  ));
  const boundaryUnavailableReason = activeSourcePackageIds.length > 1
    ? "Level 2 currently requires all speakers to use the same package; mixed-package Level 1 remains available."
    : !level2Package?.sourcePath
      ? "Level 2 requires a disk-backed speaker package in the desktop app."
      : !rigidMeshesAvailable
        ? "Level 2 requires every rigid mesh to be loaded from disk in the desktop app."
      : !level2FrequencyAvailable
        ? "The selected frequency was not exported by the active Level 2 package."
        : undefined;
  const coupledUnavailableReason = boundaryUnavailableReason ?? (
    (level2Package?.manifest.fidelity_level ?? 0) < 3
      ? "The active speaker package does not contain Level 3 data."
      : !coupledRepresentationSupported
        ? "Level 3 Deploy requires a parity Petrov–Galerkin ROM package."
        : undefined
  );
  const selectedSolverAvailable = fidelity === "boundary"
    ? boundaryAvailable
    : fidelity === "coupled"
      ? coupledAvailable
      : false;
  const currentBemMicrophoneResponses = bemMicrophoneResponses?.key === microphoneSweepKey ? bemMicrophoneResponses : null;
  const currentDriverExcursion = driverExcursion?.key === microphoneSweepKey ? driverExcursion : null;
  const currentElectricalResponse = electricalResponse?.key === microphoneSweepKey ? electricalResponse : null;
  const currentProjectContents = serializeDeployProject(createDeployProject(
    projectName,
    packages,
    rigidMeshes,
    sourceConfigs,
    rigidObjects,
    microphones,
    observation,
    pkg.frequenciesHz[frequencyIndex],
    fidelity,
  ));
  const projectEdited = savedProjectSnapshot === null || savedProjectSnapshot !== currentProjectContents;

  const initializePackage = (next: LoadedSpeakerPackage) => {
    solveGeneration.current += 1;
    setPackages([next]);
    setActivePackageId(next.id);
    setSourceConfigs(defaultSources(next));
    setRigidMeshes([]);
    setActiveRigidMeshId(null);
    setRigidObjects([]);
    rigidObjectsRef.current = [];
    setMicrophones([]);
    setSelectedInstances(["subwoofer-1"]);
    setFrequencyIndex(nearestFrequencyIndex(next, 80));
    setFidelity("pattern");
    setSolvedFields(emptySolvedFieldCache());
    setBoundaryGeometryKey(null);
    setSolveRevision(0);
    setSolveState("idle");
    setLiveSolveEnabled(false);
    setMicrophoneSweepState("idle");
    setBemMicrophoneResponses(null);
    setDriverExcursion(null);
    setElectricalResponse(null);
    setTransformMode("select");
    setProjectName(`${next.manifest.name} Subwoofer Study`);
    setProjectFileName(`${next.manifest.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "deploy"}-study.blabdeploy.json`);
    setSavedProjectSnapshot(null);
    setError(null);
  };

  const importPackage = (next: LoadedSpeakerPackage) => {
    setPackages((current) => {
      const existingIndex = current.findIndex((candidate) => candidate.id === next.id);
      if (existingIndex < 0) return [...current, next];
      const updated = current.slice();
      updated[existingIndex] = next;
      return updated;
    });
    setActivePackageId(next.id);
    setFrequencyIndex(nearestFrequencyIndex(next, pkg.frequenciesHz[frequencyIndex]));
    setSavedProjectSnapshot(null);
    setError(null);
  };

  const applyProject = (
    project: DeployProject,
    nextPackages: LoadedSpeakerPackage[],
    nextRigidMeshes: RigidMeshAsset[],
    fileName: string,
  ) => {
    const nextPackageById = new Map(nextPackages.map((item) => [item.id, item]));
    for (const reference of project.packages) {
      const loaded = nextPackageById.get(reference.id);
      if (!loaded || loaded.manifest.name !== reference.name) {
        throw new Error(`Project package ${reference.name} was not loaded correctly.`);
      }
    }
    const nextRigidMeshById = new Map(nextRigidMeshes.map((item) => [item.id, item]));
    for (const reference of project.rigid_meshes) {
      if (!nextRigidMeshById.has(reference.id)) throw new Error(`Project rigid mesh ${reference.name} was not loaded correctly.`);
    }
    solveGeneration.current += 1;
    const nextSources = project.sources.map((source) => ({
      ...source,
      positionHeightM: Math.max(
        minimumSourceHeightM(nextPackageById.get(source.packageId)!, source.pitchDeg, source.rollDeg),
        source.positionHeightM,
      ),
    }));
    const nextRigidObjects = project.rigid_objects.map((object) => ({
      ...object,
      positionHeightM: Math.max(
        minimumSourceHeightM(nextRigidMeshById.get(object.assetId)!, object.pitchDeg, object.rollDeg),
        object.positionHeightM,
      ),
    }));
    const nextPackage = nextPackageById.get(nextSources[0].packageId) ?? nextPackages[0];
    const nextBoundaryAssets = new Map<string, BoundaryMeshAsset>([...nextPackageById, ...nextRigidMeshById]);
    const clearanceViolations = cabinetClearanceViolations(
      nextBoundaryAssets,
      [...nextSources.map(buildSourceInstance), ...nextRigidObjects.map(buildRigidInstance)],
    );
    const nextFrequencyIndex = nearestFrequencyIndex(nextPackage, project.selected_frequency_hz);
    const homogeneousProject = new Set(nextSources.map((source) => source.packageId)).size === 1;
    const requestedSolverFidelity = project.requested_fidelity === "boundary" ||
      project.requested_fidelity === "coupled";
    const nextFidelity: Fidelity = requestedSolverFidelity &&
      homogeneousProject &&
      Boolean(
        window.boundaryLabDesktop && nextPackage.sourcePath &&
        nextPackage.manifest.fidelity_level >= (project.requested_fidelity === "coupled" ? 3 : 2) &&
        (project.requested_fidelity !== "coupled" ||
          nextPackage.manifest.files.coupled_model?.representation === "parity_petrov_galerkin_rom"),
      )
      ? project.requested_fidelity
      : "pattern";
    const normalizedContents = serializeDeployProject(createDeployProject(
      project.name,
      nextPackages,
      nextRigidMeshes,
      nextSources,
      nextRigidObjects,
      project.microphones,
      project.observation_plane,
      nextPackage.frequenciesHz[nextFrequencyIndex],
      nextFidelity,
    ));
    setPackages(nextPackages);
    setRigidMeshes(nextRigidMeshes);
    setActiveRigidMeshId(nextRigidMeshes[0]?.id ?? null);
    setRigidObjects(nextRigidObjects);
    rigidObjectsRef.current = nextRigidObjects;
    setActivePackageId(nextPackage.id);
    setSourceConfigs(nextSources);
    sourceConfigsRef.current = nextSources;
    setMicrophones(project.microphones);
    microphonesRef.current = project.microphones;
    setObservation(project.observation_plane);
    observationRef.current = project.observation_plane;
    setFrequencyIndex(nextFrequencyIndex);
    setFidelity(nextFidelity);
    setSelectedInstances([nextSources[0].id]);
    setSolvedFields(emptySolvedFieldCache());
    setBoundaryGeometryKey(null);
    setSolveRevision(0);
    setSolveState("idle");
    setSolveMessage("Ready to solve");
    setLiveSolveEnabled(false);
    setMicrophoneSweepState("idle");
    setBemMicrophoneResponses(null);
    setDriverExcursion(null);
    setElectricalResponse(null);
    setTransformMode("select");
    setProjectName(project.name);
    setProjectFileName(fileName);
    setSavedProjectSnapshot(normalizedContents);
    setError(clearanceViolations.length > 0
      ? `Loaded project contains ${clearanceViolations.length} speaker clearance violation${clearanceViolations.length === 1 ? "" : "s"}. Move the affected cabinets apart before placing them closer together.`
      : null);
  };

  useEffect(() => window.boundaryLabDesktop?.onSolveStatus((status) => {
    if (status.type === "status" && status.message) setSolveMessage(status.message);
    if (status.type === "initialized") setSolveMessage("BEAT CUDA initialized");
  }), []);

  useEffect(() => window.boundaryLabDesktop?.onMicrophoneSweepProgress((progress) => {
    setMicrophoneSweepProgress({ completed: progress.completed_count, total: progress.total_count });
    setBemMicrophoneResponses((current) => {
      if (!current || current.key !== microphoneSweepKeyRef.current) return current;
      const frequencyIndex = Array.from(current.frequenciesHz).findIndex(
        (frequency) => Math.abs(frequency - progress.frequency_hz) <= Math.max(1e-4, frequency * 1e-6),
      );
      if (frequencyIndex < 0) return current;
      const traces = new Map(current.traces);
      progress.microphone_ids.forEach((id, index) => {
        const values = traces.get(id);
        if (!values) return;
        const next = values.slice();
        next[frequencyIndex] = progress.spl_db[index];
        traces.set(id, next);
      });
      return { ...current, traces };
    });
    setDriverExcursion((current) => {
      if (!current || current.key !== microphoneSweepKeyRef.current) return current;
      const frequencyIndex = Array.from(current.frequenciesHz).findIndex(
        (frequency) => Math.abs(frequency - progress.frequency_hz) <= Math.max(1e-4, frequency * 1e-6),
      );
      if (frequencyIndex < 0) return current;
      const traces = new Map(current.traces);
      progress.transducer_ids.forEach((id, index) => {
        const existing = traces.get(id);
        const values = existing?.excursionMm.slice() ?? new Float32Array(current.frequenciesHz.length).fill(Number.NaN);
        values[frequencyIndex] = peakExcursionMillimeters(
          progress.transducer_velocity.real[index],
          progress.transducer_velocity.imag[index],
          progress.frequency_hz,
        );
        traces.set(id, { name: progress.transducer_names[index] ?? id, excursionMm: values });
      });
      return { ...current, traces };
    });
    setElectricalResponse((current) => {
      if (!current || current.key !== microphoneSweepKeyRef.current) return current;
      const frequencyIndex = Array.from(current.frequenciesHz).findIndex(
        (frequency) => Math.abs(frequency - progress.frequency_hz) <= Math.max(1e-4, frequency * 1e-6),
      );
      if (frequencyIndex < 0) return current;
      const traces = new Map(current.traces);
      progress.speaker_ids.forEach((id, index) => {
        const existing = traces.get(id);
        const blank = () => new Float32Array(current.frequenciesHz.length).fill(Number.NaN);
        const next: ElectricalTrace = existing ? {
          name: existing.name,
          impedanceMagnitudeOhm: existing.impedanceMagnitudeOhm.slice(),
          impedancePhaseDeg: existing.impedancePhaseDeg.slice(),
          rmsCurrentA: existing.rmsCurrentA.slice(),
          realPowerW: existing.realPowerW.slice(),
        } : {
          name: progress.speaker_names[index] ?? id,
          impedanceMagnitudeOhm: blank(),
          impedancePhaseDeg: blank(),
          rmsCurrentA: blank(),
          realPowerW: blank(),
        };
        const sample = electricalSample(
          progress.speaker_voltage.real[index], progress.speaker_voltage.imag[index],
          progress.speaker_current.real[index], progress.speaker_current.imag[index],
        );
        next.impedanceMagnitudeOhm[frequencyIndex] = sample.impedanceMagnitudeOhm;
        next.impedancePhaseDeg[frequencyIndex] = sample.impedancePhaseDeg;
        next.rmsCurrentA[frequencyIndex] = sample.rmsCurrentA;
        next.realPowerW[frequencyIndex] = sample.realPowerW;
        traces.set(id, next);
      });
      return { ...current, traces };
    });
  }), []);

  useEffect(() => {
    let active = true;
    if (!window.boundaryLabDesktop) return () => { active = false; };
    void window.boundaryLabDesktop.loadBundledExample()
      .then((selection) => {
        if (active && selection) initializePackage(loadSpeakerPackage(selection.bytes, selection.name, selection.path));
      })
      .catch((caught) => {
        if (active) setError(caught instanceof Error ? caught.message : String(caught));
      });
    return () => { active = false; };
  }, []);

  const openPackage = async () => {
    try {
      if (window.boundaryLabDesktop) {
        const selection = await window.boundaryLabDesktop.openSpeakerPackage();
        if (selection) importPackage(loadSpeakerPackage(selection.bytes, selection.name, selection.path));
      } else {
        packageFileInput.current?.click();
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const loadBrowserFile = async (file: File) => {
    try {
      importPackage(loadSpeakerPackage(await file.arrayBuffer(), file.name));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const importRigidAsset = (asset: RigidMeshAsset) => {
    setRigidMeshes((current) => {
      const existing = current.findIndex((item) => item.id === asset.id);
      if (existing < 0) return [...current, asset];
      const next = current.slice();
      next[existing] = asset;
      return next;
    });
    setActiveRigidMeshId(asset.id);
    setSavedProjectSnapshot(null);
    setError(null);
  };

  const openRigidMesh = async () => {
    try {
      if (window.boundaryLabDesktop) {
        const selection = await window.boundaryLabDesktop.openRigidMesh();
        if (selection) importRigidAsset(loadRigidMesh(selection.bytes, selection.name, selection.path));
      } else {
        rigidMeshFileInput.current?.click();
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const loadBrowserRigidMesh = async (file: File) => {
    try {
      importRigidAsset(loadRigidMesh(await file.arrayBuffer(), file.name));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const openProject = async () => {
    try {
      if (projectEdited && !window.confirm("Open another project and discard unsaved changes?")) return;
      if (window.boundaryLabDesktop) {
        const selection = await window.boundaryLabDesktop.openProject();
        if (!selection) return;
        const project = parseDeployProject(selection.contents);
        if (selection.packages.length !== project.packages.length) {
          throw new Error("One or more speaker packages referenced by this project were not located.");
        }
        if (selection.rigidMeshes.length !== project.rigid_meshes.length) {
          throw new Error("One or more rigid meshes referenced by this project were not located.");
        }
        const nextPackages = selection.packages.map((item, index) => ({
          ...loadSpeakerPackage(item.bytes, item.name, item.path),
          id: project.packages[index].id,
        }));
        const nextRigidMeshes = selection.rigidMeshes.map((item, index) => {
          const reference = project.rigid_meshes[index];
          const loaded = loadRigidMesh(item.bytes, item.name, item.path, reference.scale_to_meters);
          if (loaded.id !== reference.id) throw new Error(`Located mesh does not match ${reference.name}.`);
          return { ...loaded, name: reference.name };
        });
        applyProject(project, nextPackages, nextRigidMeshes, selection.name);
      } else {
        projectFileInput.current?.click();
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const loadBrowserProject = async (file: File) => {
    try {
      const project = parseDeployProject(await file.text());
      const loadedById = new Map(packages.map((item) => [item.id, item]));
      const missing = project.packages.filter((reference) => !loadedById.has(reference.id));
      if (missing.length > 0) {
        throw new Error(`Import ${missing.map((item) => item.name).join(", ")} before loading this project in a browser.`);
      }
      const loadedRigidById = new Map(rigidMeshes.map((item) => [item.id, item]));
      const missingRigid = project.rigid_meshes.filter((reference) => !loadedRigidById.has(reference.id));
      if (missingRigid.length > 0) throw new Error(`Import ${missingRigid.map((item) => item.name).join(", ")} before loading this project.`);
      applyProject(
        project,
        project.packages.map((reference) => loadedById.get(reference.id)!),
        project.rigid_meshes.map((reference) => loadedRigidById.get(reference.id)!),
        file.name,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const saveProject = async () => {
    try {
      if (window.boundaryLabDesktop) {
        const savedPath = await window.boundaryLabDesktop.saveProject(currentProjectContents, projectFileName);
        if (!savedPath) return;
        setProjectFileName(savedPath.split(/[\\/]/).at(-1) ?? projectFileName);
        setSavedProjectSnapshot(currentProjectContents);
        return;
      }
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([currentProjectContents], { type: "application/json" }));
      link.download = projectFileName;
      link.click();
      URL.revokeObjectURL(link.href);
      setSavedProjectSnapshot(currentProjectContents);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const solveLevel2 = useCallback(async () => {
    if (!window.boundaryLabDesktop || !level2Package?.sourcePath) {
      setError("Boundary and coupled solving require every speaker to use the same disk-backed package.");
      return;
    }
    const coupled = fidelity === "coupled";
    const fidelityLabel = coupled ? "Level 3" : "Level 2";
    const generation = ++solveGeneration.current;
    const requestedKey = currentSolveKey;
    const requestedGeometryKey = currentGeometryKey;
    setSolveState("solving");
    setSolveMessage("Starting BEAT CUDA worker");
    setError(null);
    if (!patternField.validMask.some((value) => value !== 0)) {
      setSolvedFields((current) => ({
        ...current,
        [coupled ? "coupled" : "boundary"]: { key: requestedKey, field: patternField },
      }));
      setSolveRevision((revision) => revision + 1);
      setSolveState("complete");
      setSolveMessage("No audience-plane samples above ground");
      return;
    }
    try {
      const rendererRequestStarted = performance.now();
      const reuseBoundary = !coupled && boundaryGeometryKey === requestedGeometryKey;
      const request: DesktopLevel2SolveRequest = {
        packagePath: level2Package.sourcePath,
        frequencyHz: pkg.frequenciesHz[frequencyIndex],
        backend: "cuda",
        fidelity: coupled ? "coupled" : "boundary",
        sources: sourceConfigs,
        rigidObjects: rigidObjects.map((object) => ({
          ...object,
          meshPath: rigidMeshById.get(object.assetId)?.sourcePath ?? "",
          scaleToMeters: rigidMeshById.get(object.assetId)?.scaleToMeters ?? 0.001,
        })),
        observation,
        solutionKey: requestedGeometryKey,
        reuseBoundary,
        includeComplexPressure: true,
      };
      let result;
      try {
        result = await window.boundaryLabDesktop.solveLevel2(request);
      } catch (reuseError) {
        if (!reuseBoundary) throw reuseError;
        setBoundaryGeometryKey(null);
        result = await window.boundaryLabDesktop.solveLevel2({ ...request, reuseBoundary: false });
      }
      if (generation !== solveGeneration.current) return;
      const rendererResultReceived = performance.now();
      const fieldParseStarted = performance.now();
      if (!result.field_pressure) throw new Error("The solver did not return complex field pressure.");
      const nextField = fieldFrameFromSpl(result.spl_db, result.columns, result.rows, result.sample_indices, result.field_pressure);
      const fieldParseSeconds = (performance.now() - fieldParseStarted) / 1000;
      pendingRenderProfile.current = {
        generation,
        columns: result.columns,
        rows: result.rows,
        sample_count: result.spl_db.length,
        julia: result.timings,
        pipeline: result.pipeline ?? {},
        renderer: {
          ipc_roundtrip_s: (rendererResultReceived - rendererRequestStarted) / 1000,
          field_frame_parse_s: fieldParseSeconds,
          received_numeric_values:
            result.spl_db.length + result.sample_indices.length +
            (result.field_pressure?.real.length ?? 0) + (result.field_pressure?.imag.length ?? 0),
        },
      };
      setSolvedFields((current) => ({
        ...current,
        [coupled ? "coupled" : "boundary"]: { key: requestedKey, field: nextField },
      }));
      if (!coupled) setBoundaryGeometryKey(requestedGeometryKey);
      setSolveRevision((revision) => revision + 1);
      setSolveState("complete");
      setSolveMessage(`Live ${fidelityLabel} field current`);
    } catch (caught) {
      if (generation !== solveGeneration.current) return;
      setSolveState("error");
      setLiveSolveEnabled(false);
      setSolveMessage(`${fidelityLabel} solve failed`);
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [boundaryGeometryKey, currentGeometryKey, currentSolveKey, fidelity, frequencyIndex, level2Package, observation, patternField, pkg.frequenciesHz, rigidMeshById, rigidObjects, sourceConfigs]);

  const stopMicrophoneSweep = useCallback(async () => {
    if (!window.boundaryLabDesktop || microphoneSweepState !== "solving") return;
    stoppingMicrophoneSweep.current = true;
    try {
      await window.boundaryLabDesktop.cancelMicrophoneSweep();
    } catch (caught) {
      stoppingMicrophoneSweep.current = false;
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [microphoneSweepState]);

  const calculateMicrophoneSweep = useCallback(async () => {
    if (!window.boundaryLabDesktop || !level2Package?.sourcePath || (microphones.length === 0 && fidelity !== "coupled")) return;
    const requestedKey = microphoneSweepKey;
    microphoneSweepKeyRef.current = requestedKey;
    stoppingMicrophoneSweep.current = false;
    setLiveSolveEnabled(false);
    setMicrophoneSweepState("solving");
    setMicrophoneSweepProgress({ completed: 0, total: microphonePatternResponses.frequenciesHz.length });
    setBemMicrophoneResponses({
      key: requestedKey,
      frequenciesHz: microphonePatternResponses.frequenciesHz.slice(),
      traces: new Map(microphones.map((microphone) => [
        microphone.id,
        new Float32Array(microphonePatternResponses.frequenciesHz.length).fill(Number.NaN),
      ])),
    });
    setDriverExcursion({
      key: requestedKey,
      frequenciesHz: microphonePatternResponses.frequenciesHz.slice(),
      traces: new Map(),
    });
    setElectricalResponse({
      key: requestedKey,
      frequenciesHz: microphonePatternResponses.frequenciesHz.slice(),
      traces: new Map(),
    });
    setError(null);
    try {
      const result = await window.boundaryLabDesktop.calculateMicrophoneSweep({
        packagePath: level2Package.sourcePath,
        backend: "cuda",
        fidelity: fidelity === "coupled" ? "coupled" : "boundary",
        sources: sourceConfigs,
        rigidObjects: rigidObjects.map((object) => ({
          ...object,
          meshPath: rigidMeshById.get(object.assetId)?.sourcePath ?? "",
          scaleToMeters: rigidMeshById.get(object.assetId)?.scaleToMeters ?? 0.001,
        })),
        microphones,
      });
      if (result.cancelled) {
        setMicrophoneSweepState("idle");
        return;
      }
      if (microphoneSweepKeyRef.current !== requestedKey) return;
      if (result.pipeline) {
        window.boundaryLabDeployProfile = {
          kind: "microphone-sweep",
          frequency_count: result.completed_count,
          pipeline: result.pipeline,
        };
      }
      const traces = new Map<string, Float32Array>();
      result.microphone_ids.forEach((id, index) => traces.set(id, Float32Array.from(result.spl_db[index])));
      setBemMicrophoneResponses({ key: requestedKey, frequenciesHz: Float64Array.from(result.frequencies_hz), traces });
      const excursionTraces = new Map<string, { name: string; excursionMm: Float32Array }>();
      result.transducer_ids.forEach((id, transducerIndex) => {
        excursionTraces.set(id, {
          name: result.transducer_names[transducerIndex] ?? id,
          excursionMm: Float32Array.from(result.frequencies_hz.map((frequencyHz, frequencyIndex) => peakExcursionMillimeters(
            result.transducer_velocity.real[transducerIndex][frequencyIndex],
            result.transducer_velocity.imag[transducerIndex][frequencyIndex],
            frequencyHz,
          ))),
        });
      });
      setDriverExcursion({ key: requestedKey, frequenciesHz: Float64Array.from(result.frequencies_hz), traces: excursionTraces });
      const electricalTraces = new Map<string, ElectricalTrace>();
      result.speaker_ids.forEach((id, speakerIndex) => {
        const samples = result.frequencies_hz.map((_frequencyHz, frequencyIndex) => electricalSample(
          result.speaker_voltage.real[speakerIndex][frequencyIndex],
          result.speaker_voltage.imag[speakerIndex][frequencyIndex],
          result.speaker_current.real[speakerIndex][frequencyIndex],
          result.speaker_current.imag[speakerIndex][frequencyIndex],
        ));
        electricalTraces.set(id, {
          name: result.speaker_names[speakerIndex] ?? id,
          impedanceMagnitudeOhm: Float32Array.from(samples.map((sample) => sample.impedanceMagnitudeOhm)),
          impedancePhaseDeg: Float32Array.from(samples.map((sample) => sample.impedancePhaseDeg)),
          rmsCurrentA: Float32Array.from(samples.map((sample) => sample.rmsCurrentA)),
          realPowerW: Float32Array.from(samples.map((sample) => sample.realPowerW)),
        });
      });
      setElectricalResponse({ key: requestedKey, frequenciesHz: Float64Array.from(result.frequencies_hz), traces: electricalTraces });
      setMicrophoneSweepProgress({ completed: result.completed_count, total: result.total_count });
      setMicrophoneSweepState("complete");
    } catch (caught) {
      if (stoppingMicrophoneSweep.current) {
        setMicrophoneSweepState("idle");
      } else {
        setMicrophoneSweepState("error");
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    } finally {
      stoppingMicrophoneSweep.current = false;
    }
  }, [fidelity, level2Package, microphonePatternResponses.frequenciesHz, microphoneSweepKey, microphones, rigidMeshById, rigidObjects, sourceConfigs]);

  const calculateOrStopMicrophoneSweep = () => {
    if (microphoneSweepState === "solving") void stopMicrophoneSweep();
    else void calculateMicrophoneSweep();
  };

  useEffect(() => {
    if (microphoneSweepState !== "solving" || microphoneSweepKeyRef.current === microphoneSweepKey) return;
    void stopMicrophoneSweep();
  }, [microphoneSweepKey, microphoneSweepState, stopMicrophoneSweep]);

  const updateSelectedSource = (next: SourceConfiguration) => {
    const sourcePackage = packageForSource(next);
    const grounded = {
      ...next,
      positionHeightM: Math.max(minimumSourceHeightM(sourcePackage, next.pitchDeg, next.rollDeg), next.positionHeightM),
    };
    const current = sourceConfigsRef.current;
    const proposed = current.map((source) => source.id === grounded.id ? grounded : source);
    const resolved = constrainSourceConfigs(current, proposed, new Set([grounded.id]));
    sourceConfigsRef.current = resolved;
    setSourceConfigs(resolved);
  };

  const updateSelectedRigid = (next: RigidMeshConfiguration) => {
    updateRigidPose(next.id, {
      positionX: next.positionX,
      positionHeightM: next.positionHeightM,
      positionZ: next.positionZ,
      pitchDeg: next.pitchDeg,
      yawDeg: next.yawDeg,
      rollDeg: next.rollDeg,
    });
  };

  const beginSourceManipulation = useCallback((ids: readonly string[]) => {
    if (!sourceManipulationRef.current) {
      sourceManipulationRef.current = {
        start: sourceConfigsRef.current.map((source) => ({ ...source })),
        ids: new Set(ids),
      };
    }
    setSpeakerManipulationActive(true);
  }, []);

  const updateSelectedMicrophone = (next: MicrophoneConfiguration) => {
    const grounded = { ...next, positionHeightM: Math.max(0, next.positionHeightM) };
    setMicrophones((current) => current.map((microphone) => microphone.id === grounded.id ? grounded : microphone));
  };

  const updateMicrophonePose = (id: string, pose: Pick<MicrophoneConfiguration, "positionX" | "positionHeightM" | "positionZ">) => {
    const current = microphonesRef.current;
    const active = current.find((microphone) => microphone.id === id);
    if (!active) return;
    const movingIds = selectedMicrophoneIds.includes(id) ? new Set(selectedMicrophoneIds) : new Set([id]);
    const delta = {
      x: pose.positionX - active.positionX,
      y: pose.positionHeightM - active.positionHeightM,
      z: pose.positionZ - active.positionZ,
    };
    let minimumDeltaY = -Infinity;
    for (const microphone of current) if (movingIds.has(microphone.id)) {
      minimumDeltaY = Math.max(minimumDeltaY, -microphone.positionHeightM);
    }
    for (const source of sourceConfigsRef.current) if (selectedSourceIds.includes(source.id)) {
      minimumDeltaY = Math.max(
        minimumDeltaY,
        minimumSourceHeightM(packageForSource(source), source.pitchDeg, source.rollDeg) - source.positionHeightM,
      );
    }
    delta.y = Math.max(delta.y, minimumDeltaY);
    if (selectedSourceIds.length > 0) {
      const currentSources = sourceConfigsRef.current;
      const movingSources = new Set(selectedSourceIds);
      const anchorSource = currentSources.find((source) => movingSources.has(source.id))!;
      const proposedSources = currentSources.map((source) => movingSources.has(source.id) ? {
        ...source,
        positionX: source.positionX + delta.x,
        positionHeightM: source.positionHeightM + delta.y,
        positionZ: source.positionZ + delta.z,
      } : source);
      const nextSources = constrainSourceConfigs(currentSources, proposedSources, movingSources);
      const resolvedAnchor = nextSources.find((source) => source.id === anchorSource.id)!;
      delta.x = resolvedAnchor.positionX - anchorSource.positionX;
      delta.y = resolvedAnchor.positionHeightM - anchorSource.positionHeightM;
      delta.z = resolvedAnchor.positionZ - anchorSource.positionZ;
      sourceConfigsRef.current = nextSources;
      setSourceConfigs(nextSources);
    }
    const next = current.map((microphone) => movingIds.has(microphone.id) ? {
      ...microphone,
      positionX: microphone.positionX + delta.x,
      positionHeightM: microphone.positionHeightM + delta.y,
      positionZ: microphone.positionZ + delta.z,
    } : microphone);
    microphonesRef.current = next;
    setMicrophones(next);
    if (selectedInstances.includes("audience-plane")) {
      const currentObservation = observationRef.current;
      const nextObservation = {
        ...currentObservation,
        centerXM: currentObservation.centerXM + delta.x,
        heightM: currentObservation.heightM + delta.y,
        nearM: currentObservation.nearM + delta.z,
      };
      observationRef.current = nextObservation;
      setObservation(nextObservation);
    }
  };

  const updateSourcePose = (id: string, pose: SourcePoseUpdate) => {
    const currentSources = sourceConfigsRef.current;
    const active = currentSources.find((source) => source.id === id);
    if (!active) return;
    const movingIds = selectedSourceIds.includes(id) ? new Set(selectedSourceIds) : new Set([id]);
    const positionDelta = {
      x: pose.positionX - active.positionX,
      y: pose.positionHeightM - active.positionHeightM,
      z: pose.positionZ - active.positionZ,
    };
    let minimumDeltaY = -Infinity;
    for (const source of currentSources) {
      if (!movingIds.has(source.id)) continue;
      const pitchDeg = source.id === id ? pose.pitchDeg : source.pitchDeg;
      const rollDeg = source.id === id ? pose.rollDeg : source.rollDeg;
      minimumDeltaY = Math.max(
        minimumDeltaY,
        minimumSourceHeightM(packageForSource(source), pitchDeg, rollDeg) - source.positionHeightM,
      );
    }
    for (const microphone of microphonesRef.current) if (selectedMicrophoneIds.includes(microphone.id)) {
      minimumDeltaY = Math.max(minimumDeltaY, -microphone.positionHeightM);
    }
    positionDelta.y = Math.max(positionDelta.y, minimumDeltaY);
    const proposedSources = currentSources.map((source) => {
      if (!movingIds.has(source.id)) return source;
      return {
        ...source,
        ...(source.id === id ? {
          pitchDeg: pose.pitchDeg,
          yawDeg: pose.yawDeg,
          rollDeg: pose.rollDeg,
        } : {}),
        positionX: source.positionX + positionDelta.x,
        positionHeightM: source.positionHeightM + positionDelta.y,
        positionZ: source.positionZ + positionDelta.z,
      };
    });
    const nextSources = sourceManipulationRef.current
      ? proposedSources
      : constrainSourceConfigs(currentSources, proposedSources, movingIds);
    const resolvedActive = nextSources.find((source) => source.id === id)!;
    const appliedDelta = {
      x: resolvedActive.positionX - active.positionX,
      y: resolvedActive.positionHeightM - active.positionHeightM,
      z: resolvedActive.positionZ - active.positionZ,
    };
    sourceConfigsRef.current = nextSources;
    setSourceConfigs(nextSources);
    if (selectedMicrophoneIds.length > 0) {
      const movingMicrophones = new Set(selectedMicrophoneIds);
      const nextMicrophones = microphonesRef.current.map((microphone) => movingMicrophones.has(microphone.id) ? {
        ...microphone,
        positionX: microphone.positionX + appliedDelta.x,
        positionHeightM: microphone.positionHeightM + appliedDelta.y,
        positionZ: microphone.positionZ + appliedDelta.z,
      } : microphone);
      microphonesRef.current = nextMicrophones;
      setMicrophones(nextMicrophones);
    }
    if (selectedInstances.includes("audience-plane")) {
      const currentObservation = observationRef.current;
      const nextObservation = {
        ...currentObservation,
        centerXM: currentObservation.centerXM + appliedDelta.x,
        nearM: currentObservation.nearM + appliedDelta.z,
        heightM: currentObservation.heightM + appliedDelta.y,
      };
      observationRef.current = nextObservation;
      setObservation(nextObservation);
    }
  };

  const updateRigidPose = (id: string, pose: SourcePoseUpdate) => {
    const currentRigid = rigidObjectsRef.current;
    const active = currentRigid.find((object) => object.id === id);
    if (!active) return;
    const asset = rigidMeshById.get(active.assetId);
    if (!asset) return;
    const groundedPose = {
      ...pose,
      positionHeightM: Math.max(minimumSourceHeightM(asset, pose.pitchDeg, pose.rollDeg), pose.positionHeightM),
    };
    const proposed = currentRigid.map((object) => object.id === id ? {
      ...object,
      positionX: groundedPose.positionX,
      positionHeightM: groundedPose.positionHeightM,
      positionZ: groundedPose.positionZ,
      pitchDeg: groundedPose.pitchDeg,
      yawDeg: groundedPose.yawDeg,
      rollDeg: groundedPose.rollDeg,
    } : object);
    const resolved = constrainCabinetPoses(
      boundaryAssetById,
      [...sourceConfigsRef.current.map(buildSourceInstance), ...currentRigid.map(buildRigidInstance)],
      proposed.filter((object) => object.id === id).map(buildRigidInstance),
    )[0];
    const next = proposed.map((object) => object.id === id ? {
      ...object,
      positionX: resolved.position[0],
      positionHeightM: resolved.position[1],
      positionZ: resolved.position[2],
      pitchDeg: resolved.pitchDeg,
      yawDeg: resolved.yawDeg,
      rollDeg: resolved.rollDeg,
    } : object);
    rigidObjectsRef.current = next;
    setRigidObjects(next);
  };

  const updateSourceGroupPoses = (poses: SourceGroupPoseUpdate[]) => {
    if (poses.length === 0) return;
    const includesRigid = poses.some((pose) => rigidObjectsRef.current.some((object) => object.id === pose.id));
    if (includesRigid) {
      const currentInstances = [
        ...sourceConfigsRef.current.map(buildSourceInstance),
        ...rigidObjectsRef.current.map(buildRigidInstance),
      ];
      const proposedInstances = poses.map((pose) => {
        const current = currentInstances.find((instance) => instance.id === pose.id)!;
        const asset = boundaryAssetById.get(current.packageId)!;
        return {
          ...current,
          position: [
            pose.positionX,
            Math.max(minimumSourceHeightM(asset, pose.pitchDeg, pose.rollDeg), pose.positionHeightM),
            pose.positionZ,
          ] as [number, number, number],
          pitchDeg: pose.pitchDeg,
          yawDeg: pose.yawDeg,
          rollDeg: pose.rollDeg,
        };
      });
      const resolved = constrainCabinetPoses(boundaryAssetById, currentInstances, proposedInstances);
      const poseById = new Map(resolved.map((pose) => [pose.id, pose]));
      const nextSources = sourceConfigsRef.current.map((source) => {
        const pose = poseById.get(source.id);
        return pose ? {
          ...source,
          positionX: pose.position[0], positionHeightM: pose.position[1], positionZ: pose.position[2],
          pitchDeg: pose.pitchDeg, yawDeg: pose.yawDeg, rollDeg: pose.rollDeg,
        } : source;
      });
      const nextRigid = rigidObjectsRef.current.map((object) => {
        const pose = poseById.get(object.id);
        return pose ? {
          ...object,
          positionX: pose.position[0], positionHeightM: pose.position[1], positionZ: pose.position[2],
          pitchDeg: pose.pitchDeg, yawDeg: pose.yawDeg, rollDeg: pose.rollDeg,
        } : object;
      });
      sourceConfigsRef.current = nextSources;
      rigidObjectsRef.current = nextRigid;
      setSourceConfigs(nextSources);
      setRigidObjects(nextRigid);
      return;
    }
    const poseById = new Map(poses.map((pose) => [pose.id, pose]));
    const anchorSource = sourceConfigsRef.current.find((source) => source.id === poses[0].id);
    const translationOnly = poses.every((pose) => {
      const source = sourceConfigsRef.current.find((candidate) => candidate.id === pose.id);
      return source && source.pitchDeg === pose.pitchDeg && source.yawDeg === pose.yawDeg && source.rollDeg === pose.rollDeg;
    });
    let groupLiftM = 0;
    for (const pose of poses) {
      const source = sourceConfigsRef.current.find((candidate) => candidate.id === pose.id);
      if (!source) continue;
      groupLiftM = Math.max(
        groupLiftM,
        minimumSourceHeightM(packageForSource(source), pose.pitchDeg, pose.rollDeg) - pose.positionHeightM,
      );
    }
    if (translationOnly && anchorSource) {
      const requestedDeltaY = poses[0].positionHeightM - anchorSource.positionHeightM;
      for (const microphone of microphonesRef.current) if (selectedMicrophoneIds.includes(microphone.id)) {
        groupLiftM = Math.max(groupLiftM, -microphone.positionHeightM - requestedDeltaY);
      }
    }
    const currentSources = sourceConfigsRef.current;
    const movingIds = new Set(poses.map((pose) => pose.id));
    const proposedSources = currentSources.map((source) => {
      const pose = poseById.get(source.id);
      if (!pose) return source;
      return {
        ...source,
        positionX: pose.positionX,
        positionHeightM: pose.positionHeightM + groupLiftM,
        positionZ: pose.positionZ,
        pitchDeg: pose.pitchDeg,
        yawDeg: pose.yawDeg,
        rollDeg: pose.rollDeg,
      };
    });
    const nextSources = sourceManipulationRef.current
      ? proposedSources
      : constrainSourceConfigs(currentSources, proposedSources, movingIds);
    sourceConfigsRef.current = nextSources;
    setSourceConfigs(nextSources);
    if (translationOnly && anchorSource) {
      const anchorPose = nextSources.find((source) => source.id === anchorSource.id)!;
      const delta = {
        x: anchorPose.positionX - anchorSource.positionX,
        y: anchorPose.positionHeightM - anchorSource.positionHeightM,
        z: anchorPose.positionZ - anchorSource.positionZ,
      };
      if (selectedMicrophoneIds.length > 0) {
        const movingMicrophones = new Set(selectedMicrophoneIds);
        const nextMicrophones = microphonesRef.current.map((microphone) => movingMicrophones.has(microphone.id) ? {
          ...microphone,
          positionX: microphone.positionX + delta.x,
          positionHeightM: microphone.positionHeightM + delta.y,
          positionZ: microphone.positionZ + delta.z,
        } : microphone);
        microphonesRef.current = nextMicrophones;
        setMicrophones(nextMicrophones);
      }
      if (selectedInstances.includes("audience-plane")) {
        const currentObservation = observationRef.current;
        const nextObservation = {
          ...currentObservation,
          centerXM: currentObservation.centerXM + delta.x,
          heightM: currentObservation.heightM + delta.y,
          nearM: currentObservation.nearM + delta.z,
        };
        observationRef.current = nextObservation;
        setObservation(nextObservation);
      }
    }
  };

  const updateObservationPose = (pose: Pick<ObservationPlane, "centerXM" | "nearM" | "heightM" | "pitchDeg" | "yawDeg" | "rollDeg">) => {
    const currentObservation = observationRef.current;
    const currentSources = sourceConfigsRef.current;
    const positionDelta = {
      x: pose.centerXM - currentObservation.centerXM,
      y: pose.heightM - currentObservation.heightM,
      z: pose.nearM - currentObservation.nearM,
    };
    let minimumDeltaY = -Infinity;
    for (const source of currentSources) {
      if (!selectedSourceIds.includes(source.id)) continue;
      minimumDeltaY = Math.max(
        minimumDeltaY,
        minimumSourceHeightM(packageForSource(source), source.pitchDeg, source.rollDeg) - source.positionHeightM,
      );
    }
    for (const microphone of microphonesRef.current) if (selectedMicrophoneIds.includes(microphone.id)) {
      minimumDeltaY = Math.max(minimumDeltaY, -microphone.positionHeightM);
    }
    positionDelta.y = Math.max(positionDelta.y, minimumDeltaY);
    if (selectedSourceIds.length > 0) {
      const movingIds = new Set(selectedSourceIds);
      const anchorSource = currentSources.find((source) => movingIds.has(source.id))!;
      const proposedSources = currentSources.map((source) => movingIds.has(source.id) ? {
        ...source,
        positionX: source.positionX + positionDelta.x,
        positionHeightM: source.positionHeightM + positionDelta.y,
        positionZ: source.positionZ + positionDelta.z,
      } : source);
      const nextSources = constrainSourceConfigs(currentSources, proposedSources, movingIds);
      const resolvedAnchor = nextSources.find((source) => source.id === anchorSource.id)!;
      positionDelta.x = resolvedAnchor.positionX - anchorSource.positionX;
      positionDelta.y = resolvedAnchor.positionHeightM - anchorSource.positionHeightM;
      positionDelta.z = resolvedAnchor.positionZ - anchorSource.positionZ;
      sourceConfigsRef.current = nextSources;
      setSourceConfigs(nextSources);
    }
    const nextObservation = {
      ...currentObservation,
      ...pose,
      centerXM: currentObservation.centerXM + positionDelta.x,
      nearM: currentObservation.nearM + positionDelta.z,
      heightM: currentObservation.heightM + positionDelta.y,
    };
    observationRef.current = nextObservation;
    setObservation(nextObservation);
    if (selectedMicrophoneIds.length > 0) {
      const movingMicrophones = new Set(selectedMicrophoneIds);
      const nextMicrophones = microphonesRef.current.map((microphone) => movingMicrophones.has(microphone.id) ? {
        ...microphone,
        positionX: microphone.positionX + positionDelta.x,
        positionHeightM: microphone.positionHeightM + positionDelta.y,
        positionZ: microphone.positionZ + positionDelta.z,
      } : microphone);
      microphonesRef.current = nextMicrophones;
      setMicrophones(nextMicrophones);
    }
  };

  const resizeObservation = (resize: ObservationResizeUpdate) => {
    setObservation((current) => {
      const resolution = Math.max(current.columns, current.rows);
      const [columns, rows] = planeGridShape(resize.widthM, resize.depthM, resolution);
      return {
        ...current,
        widthM: resize.widthM,
        depthM: resize.depthM,
        centerXM: resize.centerXM,
        nearM: resize.centerZM - resize.depthM / 2,
        heightM: resize.heightM,
        columns,
        rows,
      };
    });
  };

  const selectSceneObject = (id: string | null, additive = false) => {
    if (id === null) {
      setSelectedInstances([]);
      setTransformMode("select");
      return;
    }
    setSelectedInstances((current) => {
      if (!additive) return [id];
      if (current.includes(id)) return current.filter((selectedId) => selectedId !== id);
      return [...current, id];
    });
    if (!additive && !sourceConfigs.some((source) => source.id === id)) setTransformMode("select");
  };

  const addSource = (packageId = activePackageId) => {
    const sourcePackage = packageById.get(packageId) ?? pkg;
    const existingIds = new Set([...sourceConfigs.map((source) => source.id), ...rigidObjects.map((object) => object.id)]);
    let suffix = sourceConfigs.length + 1;
    while (existingIds.has(`subwoofer-${suffix}`)) suffix += 1;
    const rightmostX = sourceConfigs.length > 0 ? Math.max(...sourceConfigs.map((source) => source.positionX)) : 0;
    const packageInstanceCount = sourceConfigs.filter((source) => source.packageId === sourcePackage.id).length;
    const requested: SourceConfiguration = {
      id: `subwoofer-${suffix}`,
      name: `${sourcePackage.manifest.name} ${packageInstanceCount + 1}`,
      packageId: sourcePackage.id,
      positionX: rightmostX + sourcePackage.boundsM[0] + 2,
      positionHeightM: minimumSourceHeightM(sourcePackage),
      positionZ: 0,
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
      levelDb: -3,
      delayMs: 0,
      polarity: 1,
    };
    const placed = findClearSourcePlacement(boundaryAssetById, [...sources, ...rigidInstances], buildSourceInstance(requested));
    const next = {
      ...requested,
      positionX: placed.position[0],
      positionHeightM: placed.position[1],
      positionZ: placed.position[2],
    };
    const nextSources = [...sourceConfigs, next];
    sourceConfigsRef.current = nextSources;
    setSourceConfigs(nextSources);
    setSelectedInstances([next.id]);
    setTransformMode("select");
  };

  const addRigidObject = (assetId = activeRigidMeshId) => {
    if (!assetId) return;
    const asset = rigidMeshById.get(assetId);
    if (!asset) return;
    const existingIds = new Set([
      ...sourceConfigs.map((source) => source.id),
      ...rigidObjects.map((object) => object.id),
      ...microphones.map((microphone) => microphone.id),
    ]);
    let suffix = rigidObjects.length + 1;
    while (existingIds.has(`rigid-${suffix}`)) suffix += 1;
    const proposed: RigidMeshConfiguration = {
      id: `rigid-${suffix}`,
      name: `${asset.name} ${rigidObjects.filter((object) => object.assetId === assetId).length + 1}`,
      assetId,
      positionX: 0,
      positionHeightM: minimumSourceHeightM(asset),
      positionZ: 0,
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
    };
    const placed = findClearSourcePlacement(
      boundaryAssetById,
      [...sources, ...rigidInstances],
      buildRigidInstance(proposed),
    );
    const next = {
      ...proposed,
      positionX: placed.position[0],
      positionHeightM: placed.position[1],
      positionZ: placed.position[2],
    };
    setRigidObjects((current) => [...current, next]);
    setSelectedInstances([next.id]);
    setTransformMode("select");
  };

  const duplicateSelectedSources = () => {
    if (selectedSourceIds.length === 0 && selectedRigidIds.length === 0) return;
    const existingIds = new Set([...sourceConfigs.map((source) => source.id), ...rigidObjects.map((object) => object.id)]);
    const copies: SourceConfiguration[] = [];
    for (const source of sourceConfigs.filter((candidate) => selectedSourceIds.includes(candidate.id))) {
      let suffix = sourceConfigs.length + copies.length + 1;
      while (existingIds.has(`subwoofer-${suffix}`)) suffix += 1;
      const id = `subwoofer-${suffix}`;
      existingIds.add(id);
      copies.push({
        ...source,
        id,
        name: `${source.name} copy`,
        positionX: source.positionX + 0.5,
        positionZ: source.positionZ + 0.5,
      });
    }
    const occupied = [...sourceConfigs.map(buildSourceInstance), ...rigidObjects.map(buildRigidInstance)];
    const placedCopies = copies.map((copy) => {
      const placed = findClearSourcePlacement(boundaryAssetById, occupied, buildSourceInstance(copy));
      occupied.push(placed);
      return {
        ...copy,
        positionX: placed.position[0],
        positionHeightM: placed.position[1],
        positionZ: placed.position[2],
      };
    });
    const nextSources = [...sourceConfigs, ...placedCopies];
    sourceConfigsRef.current = nextSources;
    setSourceConfigs(nextSources);
    const rigidCopies: RigidMeshConfiguration[] = [];
    for (const object of rigidObjects.filter((candidate) => selectedRigidIds.includes(candidate.id))) {
      let suffix = rigidObjects.length + rigidCopies.length + 1;
      while (existingIds.has(`rigid-${suffix}`)) suffix += 1;
      const requested = {
        ...object,
        id: `rigid-${suffix}`,
        name: `${object.name} copy`,
        positionX: object.positionX + 0.5,
        positionZ: object.positionZ + 0.5,
      };
      existingIds.add(requested.id);
      const placed = findClearSourcePlacement(boundaryAssetById, occupied, buildRigidInstance(requested));
      occupied.push(placed);
      rigidCopies.push({
        ...requested,
        positionX: placed.position[0],
        positionHeightM: placed.position[1],
        positionZ: placed.position[2],
      });
    }
    const nextRigidObjects = [...rigidObjects, ...rigidCopies];
    rigidObjectsRef.current = nextRigidObjects;
    setRigidObjects(nextRigidObjects);
    setSelectedInstances([...placedCopies.map((source) => source.id), ...rigidCopies.map((object) => object.id)]);
    setTransformMode("select");
  };

  const addMicrophone = () => {
    const existingIds = new Set([...sourceConfigs.map((source) => source.id), ...microphones.map((microphone) => microphone.id)]);
    let suffix = microphones.length + 1;
    while (existingIds.has(`microphone-${suffix}`)) suffix += 1;
    const next: MicrophoneConfiguration = {
      id: `microphone-${suffix}`,
      name: `Microphone ${suffix}`,
      positionX: 0,
      positionHeightM: 1.2,
      positionZ: 6 + microphones.length * 0.75,
    };
    setMicrophones((current) => [...current, next]);
    setSelectedInstances([next.id]);
    setTransformMode("select");
  };

  const canRemoveSelectedSources = selectedSourceIds.length > 0 && selectedSourceIds.length < sourceConfigs.length;
  const canRemoveSelectedObjects = canRemoveSelectedSources || selectedRigidIds.length > 0 || selectedMicrophoneIds.length > 0;
  const removeSelectedObjects = useCallback(() => {
    if (!canRemoveSelectedObjects) return;
    const removedSources = canRemoveSelectedSources ? new Set(selectedSourceIds) : new Set<string>();
    const removedMicrophones = new Set(selectedMicrophoneIds);
    const removedRigid = new Set(selectedRigidIds);
    const removed = new Set([...removedSources, ...removedRigid, ...removedMicrophones]);
    setSourceConfigs((current) => current.filter((source) => !removedSources.has(source.id)));
    setMicrophones((current) => current.filter((microphone) => !removedMicrophones.has(microphone.id)));
    setRigidObjects((current) => current.filter((object) => !removedRigid.has(object.id)));
    setSelectedInstances((current) => current.filter((id) => !removed.has(id)));
    setTransformMode("select");
  }, [canRemoveSelectedObjects, canRemoveSelectedSources, selectedMicrophoneIds, selectedRigidIds, selectedSourceIds]);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (event.key === "Alt") setAngleSnapDisabled(true);
      const target = event.target;
      const transformableSelected = Boolean(selectedSource) || Boolean(selectedRigid) || Boolean(selectedMicrophone) || selectedInstance === "audience-plane";
      if (target instanceof Element && target.matches("input, textarea, [contenteditable='true']")) return;
      if ((event.key === "Delete" || event.key === "Backspace") && canRemoveSelectedObjects) {
        event.preventDefault();
        removeSelectedObjects();
        return;
      }
      if (event.ctrlKey && event.key.toLowerCase() === "d" && (selectedSourceIds.length > 0 || selectedRigidIds.length > 0)) {
        event.preventDefault();
        duplicateSelectedSources();
        return;
      }
      if (!transformableSelected) return;
      if (event.key.toLowerCase() === "w") {
        event.preventDefault();
        setTransformMode("translate");
      } else if (event.key.toLowerCase() === "e" && !selectedMicrophone) {
        event.preventDefault();
        setTransformMode("rotate");
      } else if (event.key.toLowerCase() === "r" && selectedInstance === "audience-plane") {
        event.preventDefault();
        setTransformMode("scale");
      }
    };
    const keyUp = (event: KeyboardEvent) => {
      if (event.key === "Alt") setAngleSnapDisabled(false);
    };
    const windowBlur = () => setAngleSnapDisabled(false);
    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);
    window.addEventListener("blur", windowBlur);
    return () => {
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      window.removeEventListener("blur", windowBlur);
    };
  }, [canRemoveSelectedObjects, removeSelectedObjects, selectedInstance, selectedMicrophone, selectedRigid, selectedRigidIds, selectedSource, selectedSourceIds]);

  useEffect(() => {
    if (!liveSolveEnabled || fidelity === "pattern" || !selectedSolverAvailable) {
      flushLiveSolveRef.current = false;
      return;
    }
    if (speakerManipulationActive && !sceneClearanceValid) {
      flushLiveSolveRef.current = false;
      return;
    }
    if (selectedSolvedField?.key === currentSolveKey) {
      flushLiveSolveRef.current = false;
      return;
    }
    if (solveState === "solving" || microphoneSweepState === "solving") return;
    const delayMs = flushLiveSolveRef.current ? 0 : 300;
    const timeout = window.setTimeout(() => {
      flushLiveSolveRef.current = false;
      void solveLevel2();
    }, delayMs);
    return () => window.clearTimeout(timeout);
  }, [currentSolveKey, fidelity, liveSolveEnabled, microphoneSweepState, sceneClearanceValid, selectedSolvedField?.key, selectedSolverAvailable, solveLevel2, solveReleaseRevision, solveState, speakerManipulationActive]);

  const flushLiveSolve = useCallback(() => {
    if (!liveSolveEnabled || fidelity === "pattern" || !selectedSolverAvailable) return;
    flushLiveSolveRef.current = true;
    setSolveReleaseRevision((revision) => revision + 1);
  }, [fidelity, liveSolveEnabled, selectedSolverAvailable]);

  const endSourceManipulation = useCallback(() => {
    const manipulation = sourceManipulationRef.current;
    if (!manipulation) return;
    const current = sourceConfigsRef.current;
    let resolved = current;
    if (cabinetClearanceViolations(
      boundaryAssetById,
      [...current.map(buildSourceInstance), ...rigidObjectsRef.current.map(buildRigidInstance)],
    ).length > 0) {
      resolved = constrainSourceConfigs(manipulation.start, current, manipulation.ids);
      sourceConfigsRef.current = resolved;
      setSourceConfigs(resolved);

      const anchorId = manipulation.ids.values().next().value as string | undefined;
      const before = current.find((source) => source.id === anchorId);
      const after = resolved.find((source) => source.id === anchorId);
      if (before && after) {
        const correction = {
          x: after.positionX - before.positionX,
          y: after.positionHeightM - before.positionHeightM,
          z: after.positionZ - before.positionZ,
        };
        if (selectedMicrophoneIds.length > 0) {
          const movingMicrophones = new Set(selectedMicrophoneIds);
          const nextMicrophones = microphonesRef.current.map((microphone) => movingMicrophones.has(microphone.id) ? {
            ...microphone,
            positionX: microphone.positionX + correction.x,
            positionHeightM: microphone.positionHeightM + correction.y,
            positionZ: microphone.positionZ + correction.z,
          } : microphone);
          microphonesRef.current = nextMicrophones;
          setMicrophones(nextMicrophones);
        }
        if (selectedInstances.includes("audience-plane")) {
          const currentObservation = observationRef.current;
          const nextObservation = {
            ...currentObservation,
            centerXM: currentObservation.centerXM + correction.x,
            heightM: currentObservation.heightM + correction.y,
            nearM: currentObservation.nearM + correction.z,
          };
          observationRef.current = nextObservation;
          setObservation(nextObservation);
        }
      }
    }
    sourceManipulationRef.current = null;
    setSpeakerManipulationActive(false);
    flushLiveSolve();
  }, [boundaryAssetById, constrainSourceConfigs, flushLiveSolve, selectedInstances, selectedMicrophoneIds]);

  useEffect(() => {
    if (fidelity === "pattern" || !selectedSolverAvailable) setLiveSolveEnabled(false);
    if (fidelity === "boundary" && !boundaryAvailable) setFidelity("pattern");
    if (fidelity === "coupled" && !coupledAvailable) setFidelity("pattern");
  }, [boundaryAvailable, coupledAvailable, fidelity, selectedSolverAvailable]);

  return (
    <main
      className={`app-shell ${analysisDrawerResizing ? "resizing-analysis" : ""}`}
      style={{ "--analysis-drawer-height": `${analysisDrawerHeight}px` } as CSSProperties}
    >
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Waves size={20} /></div>
          <div><strong>Boundary Lab</strong><span>DEPLOY</span></div>
        </div>
        <div className="project-breadcrumb"><span>Projects</span><ChevronRight size={13} /><strong>{projectName}</strong>{projectEdited && <i>Edited</i>}</div>
        <FidelitySwitcher
          value={fidelity}
          onChange={setFidelity}
          packageLevel={scenePackageLevel}
          boundaryAvailable={boundaryAvailable}
          boundaryUnavailableReason={boundaryUnavailableReason}
          coupledAvailable={coupledAvailable}
          coupledUnavailableReason={coupledUnavailableReason}
        />
        <div className="topbar-actions">
          <button className="icon-button" title="Open project" aria-label="Open project" onClick={openProject}><FolderOpen size={17} /></button>
          <button className="icon-button" title="Save project" onClick={saveProject}><Save size={17} /></button>
          <button className="icon-button" title="Settings"><Settings2 size={17} /></button>
          <button
            className={`primary-button ${liveSolveEnabled ? "live" : ""}`}
            disabled={fidelity === "pattern" || !selectedSolverAvailable}
            title={fidelity !== "pattern" ? (liveSolveEnabled ? "Pause automatic solves" : "Start automatic solves as the scene changes") : "Select Boundary or Coupled fidelity to solve"}
            aria-pressed={liveSolveEnabled}
            onClick={() => setLiveSolveEnabled((enabled) => !enabled)}
          >{liveSolveEnabled ? <Pause size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />} {liveSolveEnabled ? "Pause solve" : "Solve field"}</button>
        </div>
      </header>

      <aside className="left-panel panel">
        <div className="panel-tabs">
          <button className={leftTab === "library" ? "active" : ""} onClick={() => setLeftTab("library")}>Library</button>
          <button className={leftTab === "scene" ? "active" : ""} onClick={() => setLeftTab("scene")}>Scene</button>
        </div>
        {leftTab === "library" ? (
          <>
            <SectionHeader icon={Import} title="Speaker library" action={<button className="text-button" onClick={openPackage}>Import</button>} />
            <div className="panel-content package-library">
              {packages.map((item) => (
                <PackageCard
                  key={item.id}
                  pkg={item}
                  active={item.id === activePackageId}
                  onSelect={() => {
                    setActivePackageId(item.id);
                    setFrequencyIndex(nearestFrequencyIndex(item, pkg.frequenciesHz[frequencyIndex]));
                  }}
                  onAdd={() => addSource(item.id)}
                />
              ))}
            </div>
            <SectionHeader icon={Box} title="Rigid mesh library" action={<button className="text-button" onClick={openRigidMesh}>Import mesh</button>} />
            <div className="panel-content package-library rigid-mesh-library">
              {rigidMeshes.length === 0 ? <div className="library-empty">No rigid meshes imported</div> : rigidMeshes.map((asset) => (
                <RigidMeshCard
                  key={asset.id}
                  asset={asset}
                  active={asset.id === activeRigidMeshId}
                  onSelect={() => setActiveRigidMeshId(asset.id)}
                  onAdd={() => addRigidObject(asset.id)}
                />
              ))}
            </div>
            <SectionHeader
              icon={Speaker}
              title="Scene objects"
              action={(
                <div className="section-actions">
                  <button className="section-action" title="Add active speaker" aria-label="Add speaker" onClick={() => addSource()}><Plus size={14} /></button>
                  <button className="section-action" title="Add active rigid mesh" aria-label="Add rigid object" disabled={!activeRigidMeshId} onClick={() => addRigidObject()}><Box size={13} /></button>
                  <button className="section-action" title="Duplicate selected boundary objects (Ctrl+D)" aria-label="Duplicate selected boundary objects" disabled={selectedSourceIds.length === 0 && selectedRigidIds.length === 0} onClick={duplicateSelectedSources}><Copy size={13} /></button>
                  <button className="section-action" title="Add microphone" aria-label="Add microphone" onClick={addMicrophone}><Mic2 size={14} /></button>
                  <button
                    className="section-action"
                    title={canRemoveSelectedObjects ? "Remove selected objects (Delete)" : "Select a removable scene object"}
                    aria-label="Remove selected objects"
                    disabled={!canRemoveSelectedObjects}
                    onClick={removeSelectedObjects}
                  ><Trash2 size={13} /></button>
                </div>
              )}
            />
            <SceneTree packages={packages} rigidMeshes={rigidMeshes} sources={sourceConfigs} rigidObjects={rigidObjects} microphones={microphones} selectedIds={selectedInstances} activeId={selectedInstance} onSelect={selectSceneObject} />
          </>
        ) : (
          <>
            <SectionHeader
              icon={Speaker}
              title="Scene hierarchy"
              action={(
                <div className="section-actions">
                  <button className="section-action" title="Add active speaker" aria-label="Add speaker" onClick={() => addSource()}><Plus size={14} /></button>
                  <button className="section-action" title="Add active rigid mesh" aria-label="Add rigid object" disabled={!activeRigidMeshId} onClick={() => addRigidObject()}><Box size={13} /></button>
                  <button className="section-action" title="Duplicate selected boundary objects (Ctrl+D)" aria-label="Duplicate selected boundary objects" disabled={selectedSourceIds.length === 0 && selectedRigidIds.length === 0} onClick={duplicateSelectedSources}><Copy size={13} /></button>
                  <button className="section-action" title="Add microphone" aria-label="Add microphone" onClick={addMicrophone}><Mic2 size={14} /></button>
                  <button
                    className="section-action"
                    title={canRemoveSelectedObjects ? "Remove selected objects (Delete)" : "Select a removable scene object"}
                    aria-label="Remove selected objects"
                    disabled={!canRemoveSelectedObjects}
                    onClick={removeSelectedObjects}
                  ><Trash2 size={13} /></button>
                </div>
              )}
            />
            <SceneTree packages={packages} rigidMeshes={rigidMeshes} sources={sourceConfigs} rigidObjects={rigidObjects} microphones={microphones} selectedIds={selectedInstances} activeId={selectedInstance} onSelect={selectSceneObject} />
            <div className="scene-summary">
              <span>Subwoofer sources</span><strong>{sourceConfigs.length}</strong>
              <span>Rigid objects</span><strong>{rigidObjects.length}</strong>
              <span>Observation points</span><strong>{observation.columns * observation.rows}</strong>
              <span>Microphones</span><strong>{microphones.length}</strong>
              <span>Excitation ports</span><strong>{pkg.manifest.excitation_port_ids.length}</strong>
            </div>
          </>
        )}
      </aside>

      <section
        className="viewport"
        data-transform-mode={transformMode}
        data-angle-snap-disabled={angleSnapDisabled}
        data-selected-object-count={selectedInstances.length}
        data-grab-point-count={selectedSource || selectedRigid ? 8 : selectedMicrophone ? 1 : selectedInstance === "audience-plane" && transformMode === "scale" ? 4 : 0}
      >
        <SceneView
          packages={packages}
          rigidMeshes={rigidMeshes}
          sources={sources}
          rigidObjects={rigidInstances}
          microphones={microphones}
          observation={observation}
          field={field}
          phaseAnimationEnabled={phaseAnimationEnabled}
          selectedInstances={selectedInstances}
          activeInstance={selectedInstance}
          transformMode={transformMode}
          angleSnapDisabled={angleSnapDisabled}
          onSelectInstance={selectSceneObject}
          onTransformSource={updateSourcePose}
          onTransformRigid={updateRigidPose}
          onTransformSources={updateSourceGroupPoses}
          onTransformMicrophone={updateMicrophonePose}
          onTransformObservation={updateObservationPose}
          onResizeObservation={resizeObservation}
          onSourceManipulationStart={beginSourceManipulation}
          onSourceManipulationEnd={endSourceManipulation}
          onManipulationEnd={flushLiveSolve}
          onFieldTextureReady={recordFieldTexture}
        />
        <div className="viewport-toolbar">
          <button className={transformMode === "select" ? "active" : ""} title="Select (corner drag)" onClick={() => setTransformMode("select")}><MousePointer2 size={15} /></button>
          <button className={transformMode === "translate" ? "active" : ""} disabled={!selectedInstance} title="Translate (W)" onClick={() => setTransformMode("translate")}><Move3D size={15} /></button>
          <button className={transformMode === "rotate" ? "active" : ""} disabled={!selectedInstance || Boolean(selectedMicrophone)} title="Rotate (E)" onClick={() => setTransformMode("rotate")}><Rotate3D size={15} /></button>
          <button className={transformMode === "scale" ? "active" : ""} disabled={selectedInstance !== "audience-plane"} title="Resize plane (R)" onClick={() => setTransformMode("scale")}><Maximize2 size={15} /></button>
          <span />
          <button><Menu size={15} /></button>
        </div>
        <div className="solve-status" data-solve-revision={solveRevision}>
          <span className={solveState === "solving" ? "live-dot solving" : "live-dot"} />
          <div>
            <strong>{fidelity !== "pattern" ? (boundaryCurrent ? `${fidelity === "coupled" ? "Coupled" : "Boundary"} solution` : `${fidelity === "coupled" ? "Coupled" : "Boundary"} preview`) : "Pattern preview"}</strong>
            <small>{solveState === "solving" ? solveMessage : `${liveSolveEnabled ? "Live" : boundaryCurrent ? "BEAT CUDA" : "Current"} · ${formatFrequency(pkg.frequenciesHz[frequencyIndex])}`}</small>
          </div>
          <em>{field.columns} × {field.rows}</em>
        </div>
        <div className="viewport-hint">Ctrl+click: multi-select · Orbit: left drag · Pan: right drag · Zoom: wheel</div>
      </section>

      <aside className="right-panel panel">
        {selectedInstance === "audience-plane" ? (
          <>
            <div className="inspector-heading">
              <div className="object-icon"><Grid3X3 size={19} /></div>
              <div><small>{selectedInstances.length > 1 ? `${selectedInstances.length} OBJECTS SELECTED` : "SELECTED OBJECT"}</small><strong>Audience plane</strong><span>Observation surface</span></div>
              <button className="icon-button quiet"><SlidersHorizontal size={15} /></button>
            </div>
            <PlaneResolutionInspector
              value={observation}
              onChange={setObservation}
              phaseAnimationEnabled={phaseAnimationEnabled}
              onPhaseAnimationEnabledChange={setPhaseAnimationEnabled}
            />
          </>
        ) : selectedSource ? (
          <>
            <div className="inspector-heading">
              <div className="object-icon"><Speaker size={19} /></div>
              <div><small>{selectedInstances.length > 1 ? `${selectedInstances.length} OBJECTS SELECTED` : "SELECTED OBJECT"}</small><strong>{selectedSource.name}</strong><span>{selectedSourcePackage.manifest.name} source</span></div>
              <button className="icon-button quiet"><SlidersHorizontal size={15} /></button>
            </div>
            <SourceInspector config={selectedSource} minimumHeightM={sourceMinimumHeightM} onChange={updateSelectedSource} />
          </>
        ) : selectedRigid ? (
          <>
            <div className="inspector-heading">
              <div className="object-icon"><Box size={19} /></div>
              <div><small>{selectedInstances.length > 1 ? `${selectedInstances.length} OBJECTS SELECTED` : "SELECTED OBJECT"}</small><strong>{selectedRigid.name}</strong><span>Rigid Neumann boundary</span></div>
              <button className="icon-button quiet"><SlidersHorizontal size={15} /></button>
            </div>
            <RigidMeshInspector config={selectedRigid} onChange={updateSelectedRigid} />
          </>
        ) : selectedMicrophone ? (
          <>
            <div className="inspector-heading">
              <div className="object-icon"><Mic2 size={19} /></div>
              <div><small>{selectedInstances.length > 1 ? `${selectedInstances.length} OBJECTS SELECTED` : "SELECTED OBJECT"}</small><strong>{selectedMicrophone.name}</strong><span>Point pressure probe</span></div>
              <button className="icon-button quiet"><SlidersHorizontal size={15} /></button>
            </div>
            <MicrophoneInspector config={selectedMicrophone} onChange={updateSelectedMicrophone} />
          </>
        ) : null}
      </aside>

      <section className="analysis-drawer">
        <div
          className="analysis-resize-handle"
          role="separator"
          aria-label="Resize frequency response pane"
          aria-orientation="horizontal"
          aria-valuemin={150}
          aria-valuemax={Math.max(150, window.innerHeight - 58 - 180)}
          aria-valuenow={analysisDrawerHeight}
          tabIndex={0}
          onPointerDown={beginAnalysisResize}
          onPointerMove={moveAnalysisResize}
          onPointerUp={finishAnalysisResize}
          onPointerCancel={finishAnalysisResize}
          onKeyDown={(event) => {
            if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
            event.preventDefault();
            setAnalysisDrawerHeight((height) => clampAnalysisDrawerHeight(height + (event.key === "ArrowUp" ? 16 : -16)));
          }}
        />
        <div className="analysis-body">
          <div className="analysis-plot-stack">
            <div className="analysis-tabs" role="tablist" aria-label="Frequency analysis plots">
              <button role="tab" aria-selected={analysisTab === "microphones"} className={analysisTab === "microphones" ? "active" : ""} onClick={() => setAnalysisTab("microphones")}><Mic2 size={11} /> Microphones</button>
              <button role="tab" aria-selected={analysisTab === "excursion"} className={analysisTab === "excursion" ? "active" : ""} onClick={() => setAnalysisTab("excursion")}><Waves size={11} /> Driver excursion</button>
              <button role="tab" aria-selected={analysisTab === "electrical"} className={analysisTab === "electrical" ? "active" : ""} onClick={() => setAnalysisTab("electrical")}><SlidersHorizontal size={11} /> Electrical</button>
            </div>
            {analysisTab === "microphones" ? <MicrophoneResponsePlot
              pattern={microphonePatternResponses}
              bem={currentBemMicrophoneResponses}
              currentFrequencyHz={pkg.frequenciesHz[frequencyIndex]}
              frequencyPosition={sortedPosition}
              frequencyCount={usableFrequencyIndices.length}
              onFrequencyPositionChange={(position) => setFrequencyIndex(usableFrequencyIndices[position])}
              canCalculatePressure={selectedSolverAvailable && microphones.length > 0 && solveState !== "solving"}
              calculationLabel={fidelity === "coupled" ? "Calculate Coupled Pressure" : "Calculate BEM Pressure"}
              calculating={microphoneSweepState === "solving"}
              completedCount={microphoneSweepProgress.completed}
              totalCount={microphoneSweepProgress.total}
              onCalculateOrStop={calculateOrStopMicrophoneSweep}
            /> : analysisTab === "excursion" ? <DriverExcursionPlot
              data={currentDriverExcursion}
              coupledSelected={fidelity === "coupled"}
              currentFrequencyHz={pkg.frequenciesHz[frequencyIndex]}
              frequencyPosition={sortedPosition}
              frequencyCount={usableFrequencyIndices.length}
              onFrequencyPositionChange={(position) => setFrequencyIndex(usableFrequencyIndices[position])}
              canCalculate={coupledAvailable && solveState !== "solving"}
              calculating={microphoneSweepState === "solving"}
              completedCount={microphoneSweepProgress.completed}
              totalCount={microphoneSweepProgress.total}
              onCalculateOrStop={calculateOrStopMicrophoneSweep}
            /> : <ElectricalPlot
              data={currentElectricalResponse}
              coupledSelected={fidelity === "coupled"}
              currentFrequencyHz={pkg.frequenciesHz[frequencyIndex]}
              frequencyPosition={sortedPosition}
              frequencyCount={usableFrequencyIndices.length}
              onFrequencyPositionChange={(position) => setFrequencyIndex(usableFrequencyIndices[position])}
              canCalculate={coupledAvailable && solveState !== "solving"}
              calculating={microphoneSweepState === "solving"}
              completedCount={microphoneSweepProgress.completed}
              totalCount={microphoneSweepProgress.total}
              onCalculateOrStop={calculateOrStopMicrophoneSweep}
            />}
          </div>
          <div className="legend-block">
            {analysisTab === "microphones" ? observation.displayMode === "spl" ? <>
              <div className="legend-title"><span>SPL</span></div>
              <div className="color-legend" style={{ background: heatmapLegendGradient(observation.heatmapMinimumDb, observation.heatmapMaximumDb, observation.heatmapBandingDb) }} />
              <div><span>{observation.heatmapMinimumDb.toFixed(0)}</span><span>{observation.heatmapMaximumDb.toFixed(0)} dB</span></div>
            </> : <>
              <div className="legend-title"><span>{phaseAnimationEnabled ? "Phase animation" : observation.displayMode === "real_pressure" ? "Real pressure" : "Imaginary pressure"}</span></div>
              <div className="color-legend pressure-color-legend" />
              <div><span>-{observation.pressureScalePa.toFixed(0)}</span><span>0</span><span>+{observation.pressureScalePa.toFixed(0)} Pa</span></div>
            </> : analysisTab === "excursion"
              ? <div className="excursion-note"><strong>PEAK EXCURSION</strong><span>Derived from RMS diaphragm velocity</span><code>sqrt(2) |v| / 2 pi f</code></div>
              : <div className="excursion-note"><strong>CABINET INPUT</strong><span>Complex RMS voltage and summed coil current</span><code>Z = V / I</code></div>}
          </div>
        </div>
      </section>

      {field.clippedNearFieldPoints > 0 && (
        <div className="near-field-warning"><CircleHelp size={15} /><span>Some plane samples are inside the package reference sphere; use Boundary fidelity for authoritative near-field results.</span></div>
      )}
      {error && <div className="error-toast" onClick={() => setError(null)}><strong>Boundary Lab Deploy</strong><span>{error}</span></div>}
      <input
        ref={packageFileInput}
        className="hidden-file-input"
        type="file"
        accept=".blabsp"
        onChange={browserFileHandler(loadBrowserFile)}
      />
      <input
        ref={projectFileInput}
        className="hidden-file-input"
        type="file"
        accept=".blabdeploy.json,application/json"
        onChange={browserFileHandler(loadBrowserProject)}
      />
      <input
        ref={rigidMeshFileInput}
        className="hidden-file-input"
        type="file"
        accept=".msh"
        onChange={browserFileHandler(loadBrowserRigidMesh)}
      />
    </main>
  );
}
