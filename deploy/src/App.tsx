import {
  ChevronRight,
  CircleHelp,
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
  Settings2,
  SlidersHorizontal,
  Speaker,
  Trash2,
  Waves,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SceneView, type FieldTextureProfile, type ObservationResizeUpdate, type SceneTransformMode, type SourceGroupPoseUpdate, type SourcePoseUpdate } from "./components/SceneView";
import { type BemResponseData, MicrophoneResponsePlot } from "./components/MicrophoneResponsePlot";
import {
  browserFileHandler,
  MicrophoneInspector,
  PackageCard,
  planeGridShape,
  PlaneResolutionInspector,
  SceneTree,
  SectionHeader,
  SourceInspector,
} from "./components/Controls";
import { loadSpeakerPackage } from "./io/speakerPackage";
import { createDeployProject, parseDeployProject, serializeDeployProject, type DeployProject } from "./io/deployProject";
import { createDemoPackage } from "./model/demoPackage";
import {
  buildSourceInstance,
  buildPatternLookup,
  computeFieldFrame,
  computeMicrophonePatternResponses,
  fieldFrameFromSpl,
  minimumSourceHeightM,
  nearestFrequencyIndex,
} from "./model/field";
import type { Fidelity, FieldFrame, LoadedSpeakerPackage, MicrophoneConfiguration, ObservationPlane, SourceConfiguration } from "./model/types";
import { heatmapLegendGradient } from "./model/heatmap";

function defaultSources(pkg: LoadedSpeakerPackage): SourceConfiguration[] {
  const centerSpacingM = pkg.boundsM[0] + 2;
  const positionHeightM = minimumSourceHeightM(pkg);
  return [
    {
      id: "subwoofer-1",
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
}: {
  value: Fidelity;
  onChange: (value: Fidelity) => void;
  packageLevel: number;
  boundaryAvailable: boolean;
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
        const interactive = item.id === "pattern" || (item.id === "boundary" && available && boundaryAvailable);
        return (
          <button
            key={item.id}
            className={`${value === item.id ? "active" : ""} ${!interactive ? "engine-required" : ""}`}
            onClick={() => interactive && onChange(item.id)}
            title={interactive ? (item.id === "boundary" ? "Exterior BEM with fixed distributed sources" : "Live complex pattern field") : available ? "This fidelity is not connected yet" : "Package does not contain this fidelity"}
          >
            <span>{item.label}</span>
            {item.id === "boundary" && interactive && <small>CUDA</small>}
            {!interactive && item.id !== "pattern" && <small>{available ? "ENGINE" : "N/A"}</small>}
          </button>
        );
      })}
    </div>
  );
}

export function App() {
  const [pkg, setPackage] = useState<LoadedSpeakerPackage>(() => createDemoPackage());
  const [sourceConfigs, setSourceConfigs] = useState<SourceConfiguration[]>(() => defaultSources(pkg));
  const [microphones, setMicrophones] = useState<MicrophoneConfiguration[]>([]);
  const [observation, setObservation] = useState(defaultObservation);
  const [frequencyIndex, setFrequencyIndex] = useState(() => nearestFrequencyIndex(pkg, 80));
  const [fidelity, setFidelity] = useState<Fidelity>("pattern");
  const [selectedInstances, setSelectedInstances] = useState<string[]>(["subwoofer-1"]);
  const [error, setError] = useState<string | null>(null);
  const [leftTab, setLeftTab] = useState<"library" | "scene">("library");
  const [boundaryField, setBoundaryField] = useState<FieldFrame | null>(null);
  const [boundarySolveKey, setBoundarySolveKey] = useState<string | null>(null);
  const [boundaryGeometryKey, setBoundaryGeometryKey] = useState<string | null>(null);
  const [solveRevision, setSolveRevision] = useState(0);
  const [solveState, setSolveState] = useState<"idle" | "solving" | "complete" | "error">("idle");
  const [solveMessage, setSolveMessage] = useState("Ready for a Level 2 solve");
  const [liveSolveEnabled, setLiveSolveEnabled] = useState(false);
  const [transformMode, setTransformMode] = useState<SceneTransformMode>("select");
  const [angleSnapDisabled, setAngleSnapDisabled] = useState(false);
  const [projectName, setProjectName] = useState("S218BP Subwoofer Study");
  const [projectFileName, setProjectFileName] = useState("s218bp-subwoofer-study.blabdeploy.json");
  const [savedProjectSnapshot, setSavedProjectSnapshot] = useState<string | null>(null);
  const [solveReleaseRevision, setSolveReleaseRevision] = useState(0);
  const [microphoneSweepState, setMicrophoneSweepState] = useState<"idle" | "solving" | "complete" | "error">("idle");
  const [microphoneSweepProgress, setMicrophoneSweepProgress] = useState({ completed: 0, total: 0 });
  const [bemMicrophoneResponses, setBemMicrophoneResponses] = useState<BemResponseData | null>(null);
  const packageFileInput = useRef<HTMLInputElement>(null);
  const projectFileInput = useRef<HTMLInputElement>(null);
  const solveGeneration = useRef(0);
  const pendingRenderProfile = useRef<Record<string, unknown> | null>(null);
  const sourceConfigsRef = useRef(sourceConfigs);
  const observationRef = useRef(observation);
  const microphonesRef = useRef(microphones);
  const microphoneSweepKeyRef = useRef<string | null>(null);
  const stoppingMicrophoneSweep = useRef(false);
  const flushLiveSolveRef = useRef(false);

  useEffect(() => {
    sourceConfigsRef.current = sourceConfigs;
  }, [sourceConfigs]);

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

  const sortedFrequencyIndices = useMemo(
    () => Array.from(pkg.frequenciesHz.keys()).sort((a, b) => pkg.frequenciesHz[a] - pkg.frequenciesHz[b]),
    [pkg],
  );
  const sortedPosition = Math.max(0, sortedFrequencyIndices.indexOf(frequencyIndex));
  const sources = useMemo(() => sourceConfigs.map(buildSourceInstance), [sourceConfigs]);
  const selectedInstance = selectedInstances.at(-1) ?? null;
  const selectedSourceIndex = sourceConfigs.findIndex((source) => source.id === selectedInstance);
  const selectedSource = selectedSourceIndex >= 0 ? sourceConfigs[selectedSourceIndex] : null;
  const selectedMicrophoneIndex = microphones.findIndex((microphone) => microphone.id === selectedInstance);
  const selectedMicrophone = selectedMicrophoneIndex >= 0 ? microphones[selectedMicrophoneIndex] : null;
  const selectedSourceIds = useMemo(
    () => selectedInstances.filter((id) => sourceConfigs.some((source) => source.id === id)),
    [selectedInstances, sourceConfigs],
  );
  const selectedMicrophoneIds = useMemo(
    () => selectedInstances.filter((id) => microphones.some((microphone) => microphone.id === id)),
    [microphones, selectedInstances],
  );
  const sourceMinimumHeightM = minimumSourceHeightM(pkg);
  const lookup = useMemo(() => buildPatternLookup(pkg, frequencyIndex), [pkg, frequencyIndex]);
  const observationAcousticKey = JSON.stringify(observationAcousticState(observation));
  const patternField = useMemo(
    () => computeFieldFrame(pkg, sources, sourceConfigs, observation, frequencyIndex, lookup),
    [pkg, sources, sourceConfigs, observationAcousticKey, frequencyIndex, lookup],
  );
  const microphonePatternResponses = useMemo(
    () => computeMicrophonePatternResponses(pkg, sources, sourceConfigs, microphones),
    [pkg, sources, sourceConfigs, microphones],
  );
  const microphoneSweepKey = useMemo(() => JSON.stringify({
    package: pkg.id,
    sourcePath: pkg.sourcePath,
    sources: sourceConfigs,
    microphones,
    frequencies: Array.from(microphonePatternResponses.frequenciesHz),
  }), [microphonePatternResponses.frequenciesHz, microphones, pkg.id, pkg.sourcePath, sourceConfigs]);
  const currentSolveKey = useMemo(() => JSON.stringify({
    package: pkg.id,
    frequency: pkg.frequenciesHz[frequencyIndex],
    sources: sourceConfigs,
    observation: observationAcousticKey,
  }), [pkg.id, pkg.frequenciesHz, frequencyIndex, sourceConfigs, observationAcousticKey]);
  const currentGeometryKey = useMemo(() => JSON.stringify({
    package: pkg.id,
    frequency: pkg.frequenciesHz[frequencyIndex],
    sources: sourceConfigs,
  }), [pkg.id, pkg.frequenciesHz, frequencyIndex, sourceConfigs]);
  const field = fidelity === "boundary" && boundaryField && boundarySolveKey === currentSolveKey
    ? boundaryField
    : patternField;
  const boundaryCurrent = fidelity === "boundary" && boundaryField !== null && boundarySolveKey === currentSolveKey;
  const boundaryAvailable = Boolean(window.boundaryLabDesktop && pkg.sourcePath && pkg.manifest.fidelity_level >= 2);
  const currentBemMicrophoneResponses = bemMicrophoneResponses?.key === microphoneSweepKey ? bemMicrophoneResponses : null;
  const currentProjectContents = serializeDeployProject(createDeployProject(
    projectName,
    pkg,
    sourceConfigs,
    microphones,
    observation,
    pkg.frequenciesHz[frequencyIndex],
    fidelity,
  ));
  const projectEdited = savedProjectSnapshot === null || savedProjectSnapshot !== currentProjectContents;

  const applyPackage = (next: LoadedSpeakerPackage) => {
    solveGeneration.current += 1;
    setPackage(next);
    setSourceConfigs(defaultSources(next));
    setMicrophones([]);
    setSelectedInstances(["subwoofer-1"]);
    setFrequencyIndex(nearestFrequencyIndex(next, 80));
    setFidelity("pattern");
    setBoundaryField(null);
    setBoundarySolveKey(null);
    setBoundaryGeometryKey(null);
    setSolveRevision(0);
    setSolveState("idle");
    setLiveSolveEnabled(false);
    setMicrophoneSweepState("idle");
    setBemMicrophoneResponses(null);
    setTransformMode("select");
    setProjectName(`${next.manifest.name} Subwoofer Study`);
    setProjectFileName(`${next.manifest.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "deploy"}-study.blabdeploy.json`);
    setSavedProjectSnapshot(null);
    setError(null);
  };

  const applyProject = (project: DeployProject, nextPackage: LoadedSpeakerPackage, fileName: string) => {
    if (project.package.name !== nextPackage.manifest.name) {
      throw new Error(`Project expects ${project.package.name}, but ${nextPackage.manifest.name} was loaded.`);
    }
    solveGeneration.current += 1;
    const nextSources = project.sources.map((source) => ({
      ...source,
      positionHeightM: Math.max(
        minimumSourceHeightM(nextPackage, source.pitchDeg, source.rollDeg),
        source.positionHeightM,
      ),
    }));
    const nextFrequencyIndex = nearestFrequencyIndex(nextPackage, project.selected_frequency_hz);
    const nextFidelity: Fidelity = project.requested_fidelity === "boundary" &&
      Boolean(window.boundaryLabDesktop && nextPackage.sourcePath && nextPackage.manifest.fidelity_level >= 2)
      ? "boundary"
      : "pattern";
    const normalizedContents = serializeDeployProject(createDeployProject(
      project.name,
      nextPackage,
      nextSources,
      project.microphones,
      project.observation_plane,
      nextPackage.frequenciesHz[nextFrequencyIndex],
      nextFidelity,
    ));
    setPackage(nextPackage);
    setSourceConfigs(nextSources);
    sourceConfigsRef.current = nextSources;
    setMicrophones(project.microphones);
    microphonesRef.current = project.microphones;
    setObservation(project.observation_plane);
    observationRef.current = project.observation_plane;
    setFrequencyIndex(nextFrequencyIndex);
    setFidelity(nextFidelity);
    setSelectedInstances([nextSources[0].id]);
    setBoundaryField(null);
    setBoundarySolveKey(null);
    setBoundaryGeometryKey(null);
    setSolveRevision(0);
    setSolveState("idle");
    setSolveMessage("Ready for a Level 2 solve");
    setLiveSolveEnabled(false);
    setMicrophoneSweepState("idle");
    setBemMicrophoneResponses(null);
    setTransformMode("select");
    setProjectName(project.name);
    setProjectFileName(fileName);
    setSavedProjectSnapshot(normalizedContents);
    setError(null);
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
  }), []);

  useEffect(() => {
    let active = true;
    if (!window.boundaryLabDesktop) return () => { active = false; };
    void window.boundaryLabDesktop.loadBundledExample()
      .then((selection) => {
        if (active && selection) applyPackage(loadSpeakerPackage(selection.bytes, selection.name, selection.path));
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
        if (selection) applyPackage(loadSpeakerPackage(selection.bytes, selection.name, selection.path));
      } else {
        packageFileInput.current?.click();
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  };

  const loadBrowserFile = async (file: File) => {
    try {
      applyPackage(loadSpeakerPackage(await file.arrayBuffer(), file.name));
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
        if (!selection.package) throw new Error("The speaker package referenced by this project was not located.");
        const nextPackage = loadSpeakerPackage(
          selection.package.bytes,
          selection.package.name,
          selection.package.path,
        );
        applyProject(project, nextPackage, selection.name);
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
      const referencedName = project.package.source_file?.split(/[\\/]/).at(-1);
      if (project.package.name !== pkg.manifest.name && referencedName !== pkg.fileName) {
        throw new Error(`Open the ${project.package.name} speaker package before loading this project in a browser.`);
      }
      applyProject(project, pkg, file.name);
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
    if (!window.boundaryLabDesktop || !pkg.sourcePath) {
      setError("Level 2 solving requires the desktop app and a package loaded from disk.");
      return;
    }
    const generation = ++solveGeneration.current;
    const requestedKey = currentSolveKey;
    const requestedGeometryKey = currentGeometryKey;
    setSolveState("solving");
    setSolveMessage("Starting BEAT CUDA worker");
    setError(null);
    if (!patternField.validMask.some((value) => value !== 0)) {
      setBoundaryField(patternField);
      setBoundarySolveKey(requestedKey);
      setSolveRevision((revision) => revision + 1);
      setSolveState("complete");
      setSolveMessage("No audience-plane samples above ground");
      return;
    }
    try {
      const rendererRequestStarted = performance.now();
      const reuseBoundary = boundaryGeometryKey === requestedGeometryKey;
      const request: DesktopLevel2SolveRequest = {
        packagePath: pkg.sourcePath,
        frequencyHz: pkg.frequenciesHz[frequencyIndex],
        backend: "cuda",
        sources: sourceConfigs,
        observation,
        solutionKey: requestedGeometryKey,
        reuseBoundary,
        includeComplexPressure: false,
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
      const nextField = fieldFrameFromSpl(result.spl_db, result.columns, result.rows, result.sample_indices);
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
      setBoundaryField(nextField);
      setBoundarySolveKey(requestedKey);
      setBoundaryGeometryKey(requestedGeometryKey);
      setSolveRevision((revision) => revision + 1);
      setSolveState("complete");
      setSolveMessage("Live Level 2 field current");
    } catch (caught) {
      if (generation !== solveGeneration.current) return;
      setSolveState("error");
      setLiveSolveEnabled(false);
      setSolveMessage("Level 2 solve failed");
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, [boundaryGeometryKey, currentGeometryKey, currentSolveKey, frequencyIndex, observation, patternField, pkg.frequenciesHz, pkg.sourcePath, sourceConfigs]);

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
    if (!window.boundaryLabDesktop || !pkg.sourcePath || microphones.length === 0) return;
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
    setError(null);
    try {
      const result = await window.boundaryLabDesktop.calculateMicrophoneSweep({
        packagePath: pkg.sourcePath,
        backend: "cuda",
        sources: sourceConfigs,
        microphones,
      });
      if (result.cancelled) {
        setMicrophoneSweepState("idle");
        return;
      }
      if (microphoneSweepKeyRef.current !== requestedKey) return;
      const traces = new Map<string, Float32Array>();
      result.microphone_ids.forEach((id, index) => traces.set(id, Float32Array.from(result.spl_db[index])));
      setBemMicrophoneResponses({ key: requestedKey, frequenciesHz: Float64Array.from(result.frequencies_hz), traces });
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
  }, [microphonePatternResponses.frequenciesHz, microphoneSweepKey, microphones, pkg.sourcePath, sourceConfigs]);

  const calculateOrStopMicrophoneSweep = () => {
    if (microphoneSweepState === "solving") void stopMicrophoneSweep();
    else void calculateMicrophoneSweep();
  };

  useEffect(() => {
    if (microphoneSweepState !== "solving" || microphoneSweepKeyRef.current === microphoneSweepKey) return;
    void stopMicrophoneSweep();
  }, [microphoneSweepKey, microphoneSweepState, stopMicrophoneSweep]);

  const updateSelectedSource = (next: SourceConfiguration) => {
    const grounded = {
      ...next,
      positionHeightM: Math.max(minimumSourceHeightM(pkg, next.pitchDeg, next.rollDeg), next.positionHeightM),
    };
    setSourceConfigs((current) => current.map((source) => source.id === grounded.id ? grounded : source));
  };

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
        minimumSourceHeightM(pkg, source.pitchDeg, source.rollDeg) - source.positionHeightM,
      );
    }
    delta.y = Math.max(delta.y, minimumDeltaY);
    const next = current.map((microphone) => movingIds.has(microphone.id) ? {
      ...microphone,
      positionX: microphone.positionX + delta.x,
      positionHeightM: microphone.positionHeightM + delta.y,
      positionZ: microphone.positionZ + delta.z,
    } : microphone);
    microphonesRef.current = next;
    setMicrophones(next);
    if (selectedSourceIds.length > 0) {
      const movingSources = new Set(selectedSourceIds);
      const nextSources = sourceConfigsRef.current.map((source) => movingSources.has(source.id) ? {
        ...source,
        positionX: source.positionX + delta.x,
        positionHeightM: source.positionHeightM + delta.y,
        positionZ: source.positionZ + delta.z,
      } : source);
      sourceConfigsRef.current = nextSources;
      setSourceConfigs(nextSources);
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
        minimumSourceHeightM(pkg, pitchDeg, rollDeg) - source.positionHeightM,
      );
    }
    for (const microphone of microphonesRef.current) if (selectedMicrophoneIds.includes(microphone.id)) {
      minimumDeltaY = Math.max(minimumDeltaY, -microphone.positionHeightM);
    }
    positionDelta.y = Math.max(positionDelta.y, minimumDeltaY);
    const nextSources = currentSources.map((source) => {
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
    sourceConfigsRef.current = nextSources;
    setSourceConfigs(nextSources);
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
    if (selectedInstances.includes("audience-plane")) {
      const currentObservation = observationRef.current;
      const nextObservation = {
        ...currentObservation,
        centerXM: currentObservation.centerXM + positionDelta.x,
        nearM: currentObservation.nearM + positionDelta.z,
        heightM: currentObservation.heightM + positionDelta.y,
      };
      observationRef.current = nextObservation;
      setObservation(nextObservation);
    }
  };

  const updateSourceGroupPoses = (poses: SourceGroupPoseUpdate[]) => {
    if (poses.length === 0) return;
    const poseById = new Map(poses.map((pose) => [pose.id, pose]));
    const anchorSource = sourceConfigsRef.current.find((source) => source.id === poses[0].id);
    const translationOnly = poses.every((pose) => {
      const source = sourceConfigsRef.current.find((candidate) => candidate.id === pose.id);
      return source && source.pitchDeg === pose.pitchDeg && source.yawDeg === pose.yawDeg && source.rollDeg === pose.rollDeg;
    });
    let groupLiftM = 0;
    for (const pose of poses) {
      groupLiftM = Math.max(
        groupLiftM,
        minimumSourceHeightM(pkg, pose.pitchDeg, pose.rollDeg) - pose.positionHeightM,
      );
    }
    if (translationOnly && anchorSource) {
      const requestedDeltaY = poses[0].positionHeightM - anchorSource.positionHeightM;
      for (const microphone of microphonesRef.current) if (selectedMicrophoneIds.includes(microphone.id)) {
        groupLiftM = Math.max(groupLiftM, -microphone.positionHeightM - requestedDeltaY);
      }
    }
    const nextSources = sourceConfigsRef.current.map((source) => {
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
    sourceConfigsRef.current = nextSources;
    setSourceConfigs(nextSources);
    if (translationOnly && anchorSource) {
      const anchorPose = poses[0];
      const delta = {
        x: anchorPose.positionX - anchorSource.positionX,
        y: anchorPose.positionHeightM + groupLiftM - anchorSource.positionHeightM,
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
        minimumSourceHeightM(pkg, source.pitchDeg, source.rollDeg) - source.positionHeightM,
      );
    }
    for (const microphone of microphonesRef.current) if (selectedMicrophoneIds.includes(microphone.id)) {
      minimumDeltaY = Math.max(minimumDeltaY, -microphone.positionHeightM);
    }
    positionDelta.y = Math.max(positionDelta.y, minimumDeltaY);
    const nextObservation = {
      ...currentObservation,
      ...pose,
      centerXM: currentObservation.centerXM + positionDelta.x,
      nearM: currentObservation.nearM + positionDelta.z,
      heightM: currentObservation.heightM + positionDelta.y,
    };
    observationRef.current = nextObservation;
    setObservation(nextObservation);
    if (selectedSourceIds.length > 0) {
      const movingIds = new Set(selectedSourceIds);
      const nextSources = currentSources.map((source) => movingIds.has(source.id) ? {
        ...source,
        positionX: source.positionX + positionDelta.x,
        positionHeightM: source.positionHeightM + positionDelta.y,
        positionZ: source.positionZ + positionDelta.z,
      } : source);
      sourceConfigsRef.current = nextSources;
      setSourceConfigs(nextSources);
    }
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

  const addSource = () => {
    const existingIds = new Set(sourceConfigs.map((source) => source.id));
    let suffix = sourceConfigs.length + 1;
    while (existingIds.has(`subwoofer-${suffix}`)) suffix += 1;
    const rightmostX = Math.max(...sourceConfigs.map((source) => source.positionX));
    const next: SourceConfiguration = {
      id: `subwoofer-${suffix}`,
      positionX: rightmostX + pkg.boundsM[0] + 2,
      positionHeightM: minimumSourceHeightM(pkg),
      positionZ: 0,
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
      levelDb: -3,
      delayMs: 0,
      polarity: 1,
    };
    setSourceConfigs((current) => [...current, next]);
    setSelectedInstances([next.id]);
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
  const canRemoveSelectedObjects = canRemoveSelectedSources || selectedMicrophoneIds.length > 0;
  const removeSelectedObjects = useCallback(() => {
    if (!canRemoveSelectedObjects) return;
    const removedSources = canRemoveSelectedSources ? new Set(selectedSourceIds) : new Set<string>();
    const removedMicrophones = new Set(selectedMicrophoneIds);
    const removed = new Set([...removedSources, ...removedMicrophones]);
    setSourceConfigs((current) => current.filter((source) => !removedSources.has(source.id)));
    setMicrophones((current) => current.filter((microphone) => !removedMicrophones.has(microphone.id)));
    setSelectedInstances((current) => current.filter((id) => !removed.has(id)));
    setTransformMode("select");
  }, [canRemoveSelectedObjects, canRemoveSelectedSources, selectedMicrophoneIds, selectedSourceIds]);

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (event.key === "Alt") setAngleSnapDisabled(true);
      const target = event.target;
      const transformableSelected = Boolean(selectedSource) || Boolean(selectedMicrophone) || selectedInstance === "audience-plane";
      if (target instanceof Element && target.matches("input, textarea, [contenteditable='true']")) return;
      if ((event.key === "Delete" || event.key === "Backspace") && canRemoveSelectedObjects) {
        event.preventDefault();
        removeSelectedObjects();
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
  }, [canRemoveSelectedObjects, removeSelectedObjects, selectedInstance, selectedMicrophone, selectedSource]);

  useEffect(() => {
    if (!liveSolveEnabled || fidelity !== "boundary" || !boundaryAvailable) {
      flushLiveSolveRef.current = false;
      return;
    }
    if (boundarySolveKey === currentSolveKey) {
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
  }, [boundaryAvailable, boundarySolveKey, currentSolveKey, fidelity, liveSolveEnabled, microphoneSweepState, solveLevel2, solveReleaseRevision, solveState]);

  const flushLiveSolve = useCallback(() => {
    if (!liveSolveEnabled || fidelity !== "boundary" || !boundaryAvailable) return;
    flushLiveSolveRef.current = true;
    setSolveReleaseRevision((revision) => revision + 1);
  }, [boundaryAvailable, fidelity, liveSolveEnabled]);

  useEffect(() => {
    if (fidelity !== "boundary") setLiveSolveEnabled(false);
  }, [fidelity]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Waves size={20} /></div>
          <div><strong>Boundary Lab</strong><span>DEPLOY</span></div>
        </div>
        <div className="project-breadcrumb"><span>Projects</span><ChevronRight size={13} /><strong>{projectName}</strong>{projectEdited && <i>Edited</i>}</div>
        <FidelitySwitcher value={fidelity} onChange={setFidelity} packageLevel={pkg.manifest.fidelity_level} boundaryAvailable={boundaryAvailable} />
        <div className="topbar-actions">
          <button className="icon-button" title="Open project" aria-label="Open project" onClick={openProject}><FolderOpen size={17} /></button>
          <button className="icon-button" title="Save project" onClick={saveProject}><Save size={17} /></button>
          <button className="icon-button" title="Settings"><Settings2 size={17} /></button>
          <button
            className={`primary-button ${liveSolveEnabled ? "live" : ""}`}
            disabled={fidelity !== "boundary" || !boundaryAvailable}
            title={fidelity === "boundary" ? (liveSolveEnabled ? "Pause automatic Level 2 solves" : "Start automatic Level 2 solves as the scene changes") : "Select Boundary fidelity to run Level 2"}
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
            <SectionHeader icon={Import} title="Speaker package" action={<button className="text-button" onClick={openPackage}>Open</button>} />
            <div className="panel-content"><PackageCard pkg={pkg} onOpen={openPackage} /></div>
            <SectionHeader
              icon={Speaker}
              title="Scene objects"
              action={(
                <div className="section-actions">
                  <button className="section-action" title="Add speaker" aria-label="Add speaker" onClick={addSource}><Plus size={14} /></button>
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
            <SceneTree pkg={pkg} sources={sourceConfigs} microphones={microphones} selectedIds={selectedInstances} activeId={selectedInstance} onSelect={selectSceneObject} />
          </>
        ) : (
          <>
            <SectionHeader
              icon={Speaker}
              title="Scene hierarchy"
              action={(
                <div className="section-actions">
                  <button className="section-action" title="Add speaker" aria-label="Add speaker" onClick={addSource}><Plus size={14} /></button>
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
            <SceneTree pkg={pkg} sources={sourceConfigs} microphones={microphones} selectedIds={selectedInstances} activeId={selectedInstance} onSelect={selectSceneObject} />
            <div className="scene-summary">
              <span>Subwoofer sources</span><strong>{sourceConfigs.length}</strong>
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
        data-grab-point-count={selectedSource ? 8 : selectedMicrophone ? 1 : selectedInstance === "audience-plane" && transformMode === "scale" ? 4 : 0}
      >
        <SceneView
          pkg={pkg}
          sources={sources}
          microphones={microphones}
          observation={observation}
          field={field}
          selectedInstances={selectedInstances}
          activeInstance={selectedInstance}
          transformMode={transformMode}
          angleSnapDisabled={angleSnapDisabled}
          onSelectInstance={selectSceneObject}
          onTransformSource={updateSourcePose}
          onTransformSources={updateSourceGroupPoses}
          onTransformMicrophone={updateMicrophonePose}
          onTransformObservation={updateObservationPose}
          onResizeObservation={resizeObservation}
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
            <strong>{fidelity === "boundary" ? (boundaryCurrent ? "Boundary solution" : "Boundary preview") : "Pattern preview"}</strong>
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
            <PlaneResolutionInspector value={observation} onChange={setObservation} />
          </>
        ) : selectedSource ? (
          <>
            <div className="inspector-heading">
              <div className="object-icon"><Speaker size={19} /></div>
              <div><small>{selectedInstances.length > 1 ? `${selectedInstances.length} OBJECTS SELECTED` : "SELECTED OBJECT"}</small><strong>{pkg.manifest.name} {selectedSourceIndex + 1}</strong><span>Subwoofer source</span></div>
              <button className="icon-button quiet"><SlidersHorizontal size={15} /></button>
            </div>
            <SourceInspector config={selectedSource} minimumHeightM={sourceMinimumHeightM} onChange={updateSelectedSource} />
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
        <div className="analysis-body">
          <MicrophoneResponsePlot
            pattern={microphonePatternResponses}
            bem={currentBemMicrophoneResponses}
            currentFrequencyHz={pkg.frequenciesHz[frequencyIndex]}
            frequencyPosition={sortedPosition}
            frequencyCount={sortedFrequencyIndices.length}
            onFrequencyPositionChange={(position) => setFrequencyIndex(sortedFrequencyIndices[position])}
            canCalculateBem={fidelity === "boundary" && boundaryAvailable && microphones.length > 0 && solveState !== "solving"}
            calculating={microphoneSweepState === "solving"}
            completedCount={microphoneSweepProgress.completed}
            totalCount={microphoneSweepProgress.total}
            onCalculateOrStop={calculateOrStopMicrophoneSweep}
          />
          <div className="legend-block">
            <div className="legend-title"><span>SPL</span></div>
            <div
              className="color-legend"
              style={{ background: heatmapLegendGradient(
                observation.heatmapMinimumDb,
                observation.heatmapMaximumDb,
                observation.heatmapBandingDb,
              ) }}
            />
            <div><span>{observation.heatmapMinimumDb.toFixed(0)}</span><span>{observation.heatmapMaximumDb.toFixed(0)} dB</span></div>
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
    </main>
  );
}
