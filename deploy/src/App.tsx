import {
  ChevronRight,
  CircleHelp,
  Grid3X3,
  Import,
  Menu,
  MousePointer2,
  Move3D,
  Pause,
  Play,
  Rotate3D,
  Save,
  Settings2,
  SlidersHorizontal,
  Speaker,
  Waves,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SceneView, type SceneTransformMode, type SourcePoseUpdate } from "./components/SceneView";
import {
  browserFileHandler,
  PackageCard,
  PlaneResolutionInspector,
  SceneTree,
  SectionHeader,
  SourceInspector,
} from "./components/Controls";
import { loadSpeakerPackage } from "./io/speakerPackage";
import { createDemoPackage } from "./model/demoPackage";
import {
  buildSourceInstance,
  buildPatternLookup,
  computeFieldFrame,
  fieldFrameFromSpl,
  minimumSourceHeightM,
  nearestFrequencyIndex,
} from "./model/field";
import type { Fidelity, FieldFrame, LoadedSpeakerPackage, ObservationPlane, SourceConfiguration } from "./model/types";

function defaultSources(pkg: LoadedSpeakerPackage): SourceConfiguration[] {
  const centerSpacingM = pkg.boundsM[0] + 2;
  const positionHeightM = minimumSourceHeightM(pkg);
  return [
    {
      id: "subwoofer-1",
      positionX: -centerSpacingM / 2,
      positionHeightM,
      positionZ: 0,
      yawDeg: 0,
      levelDb: -3,
      delayMs: 0,
      polarity: 1,
    },
    {
      id: "subwoofer-2",
      positionX: centerSpacingM / 2,
      positionHeightM,
      positionZ: 0,
      yawDeg: 0,
      levelDb: -3,
      delayMs: 0,
      polarity: 1,
    },
  ];
}

const defaultObservation: ObservationPlane = {
  widthM: 24,
  depthM: 24,
  nearM: 1.5,
  heightM: 1.2,
  columns: 54,
  rows: 54,
};

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
  const [observation, setObservation] = useState(defaultObservation);
  const [frequencyIndex, setFrequencyIndex] = useState(() => nearestFrequencyIndex(pkg, 80));
  const [fidelity, setFidelity] = useState<Fidelity>("pattern");
  const [selectedInstance, setSelectedInstance] = useState<string | null>("subwoofer-1");
  const [error, setError] = useState<string | null>(null);
  const [leftTab, setLeftTab] = useState<"library" | "scene">("library");
  const [boundaryField, setBoundaryField] = useState<FieldFrame | null>(null);
  const [boundarySolveKey, setBoundarySolveKey] = useState<string | null>(null);
  const [solveRevision, setSolveRevision] = useState(0);
  const [solveState, setSolveState] = useState<"idle" | "solving" | "complete" | "error">("idle");
  const [solveMessage, setSolveMessage] = useState("Ready for a Level 2 solve");
  const [liveSolveEnabled, setLiveSolveEnabled] = useState(false);
  const [transformMode, setTransformMode] = useState<SceneTransformMode>("select");
  const [angleSnapDisabled, setAngleSnapDisabled] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const solveGeneration = useRef(0);

  const sortedFrequencyIndices = useMemo(
    () => Array.from(pkg.frequenciesHz.keys()).sort((a, b) => pkg.frequenciesHz[a] - pkg.frequenciesHz[b]),
    [pkg],
  );
  const sortedPosition = Math.max(0, sortedFrequencyIndices.indexOf(frequencyIndex));
  const sources = useMemo(() => sourceConfigs.map(buildSourceInstance), [sourceConfigs]);
  const selectedSourceIndex = sourceConfigs.findIndex((source) => source.id === selectedInstance);
  const selectedSource = selectedSourceIndex >= 0 ? sourceConfigs[selectedSourceIndex] : null;
  const sourceMinimumHeightM = minimumSourceHeightM(pkg);
  const lookup = useMemo(() => buildPatternLookup(pkg, frequencyIndex), [pkg, frequencyIndex]);
  const patternField = useMemo(
    () => computeFieldFrame(pkg, sources, sourceConfigs, observation, frequencyIndex, lookup),
    [pkg, sources, sourceConfigs, observation, frequencyIndex, lookup],
  );
  const currentSolveKey = useMemo(() => JSON.stringify({
    package: pkg.id,
    frequency: pkg.frequenciesHz[frequencyIndex],
    sources: sourceConfigs,
    observation,
  }), [pkg.id, pkg.frequenciesHz, frequencyIndex, sourceConfigs, observation]);
  const field = fidelity === "boundary" && boundaryField && boundarySolveKey === currentSolveKey
    ? boundaryField
    : patternField;
  const boundaryCurrent = fidelity === "boundary" && boundaryField !== null && boundarySolveKey === currentSolveKey;
  const boundaryAvailable = Boolean(window.boundaryLabDesktop && pkg.sourcePath && pkg.manifest.fidelity_level >= 2);

  const applyPackage = (next: LoadedSpeakerPackage) => {
    solveGeneration.current += 1;
    setPackage(next);
    setSourceConfigs(defaultSources(next));
    setSelectedInstance("subwoofer-1");
    setFrequencyIndex(nearestFrequencyIndex(next, 80));
    setFidelity("pattern");
    setBoundaryField(null);
    setBoundarySolveKey(null);
    setSolveRevision(0);
    setSolveState("idle");
    setLiveSolveEnabled(false);
    setTransformMode("select");
    setError(null);
  };

  useEffect(() => window.boundaryLabDesktop?.onSolveStatus((status) => {
    if (status.type === "status" && status.message) setSolveMessage(status.message);
    if (status.type === "initialized") setSolveMessage("BEAT CUDA initialized");
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
        fileInput.current?.click();
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

  const saveProject = async () => {
    const project = JSON.stringify({
      schema: "boundary-lab-deploy-project",
      schema_version: 2,
      name: "S218BP Subwoofer Study",
      package: {
        id: pkg.id,
        name: pkg.manifest.name,
        source_file: pkg.sourcePath,
      },
      sources: sourceConfigs,
      observation_plane: observation,
      selected_frequency_hz: pkg.frequenciesHz[frequencyIndex],
      requested_fidelity: fidelity,
    }, null, 2) + "\n";
    if (window.boundaryLabDesktop) {
      await window.boundaryLabDesktop.saveProject(project, "s218bp-subwoofer-study.blabdeploy.json");
      return;
    }
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([project], { type: "application/json" }));
    link.download = "s218bp-subwoofer-study.blabdeploy.json";
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const solveLevel2 = useCallback(async () => {
    if (!window.boundaryLabDesktop || !pkg.sourcePath) {
      setError("Level 2 solving requires the desktop app and a package loaded from disk.");
      return;
    }
    const generation = ++solveGeneration.current;
    const requestedKey = currentSolveKey;
    setSolveState("solving");
    setSolveMessage("Starting BEAT CUDA worker");
    setError(null);
    try {
      const result = await window.boundaryLabDesktop.solveLevel2({
        packagePath: pkg.sourcePath,
        frequencyHz: pkg.frequenciesHz[frequencyIndex],
        backend: "cuda",
        sources: sourceConfigs,
        observation,
      });
      if (generation !== solveGeneration.current) return;
      setBoundaryField(fieldFrameFromSpl(result.spl_db, result.columns, result.rows));
      setBoundarySolveKey(requestedKey);
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
  }, [currentSolveKey, frequencyIndex, observation, pkg.frequenciesHz, pkg.sourcePath, sourceConfigs]);

  const updateSelectedSource = (next: SourceConfiguration) => {
    const grounded = {
      ...next,
      positionHeightM: Math.max(sourceMinimumHeightM, next.positionHeightM),
    };
    setSourceConfigs((current) => current.map((source) => source.id === grounded.id ? grounded : source));
  };

  const updateSourcePose = (id: string, pose: SourcePoseUpdate) => {
    setSourceConfigs((current) => current.map((source) => source.id === id ? {
      ...source,
      ...pose,
      positionHeightM: Math.max(sourceMinimumHeightM, pose.positionHeightM),
    } : source));
  };

  const selectSceneObject = (id: string | null) => {
    setSelectedInstance(id);
    if (!sourceConfigs.some((source) => source.id === id)) setTransformMode("select");
  };

  useEffect(() => {
    const keyDown = (event: KeyboardEvent) => {
      if (event.key === "Alt") setAngleSnapDisabled(true);
      const target = event.target;
      if ((target instanceof Element && target.matches("input, textarea, [contenteditable='true']")) || !selectedSource) return;
      if (event.key.toLowerCase() === "w") {
        event.preventDefault();
        setTransformMode("translate");
      } else if (event.key.toLowerCase() === "e") {
        event.preventDefault();
        setTransformMode("rotate");
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
  }, [selectedSource]);

  useEffect(() => {
    if (!liveSolveEnabled || fidelity !== "boundary" || !boundaryAvailable) return;
    if (solveState === "solving" || boundarySolveKey === currentSolveKey) return;
    const timeout = window.setTimeout(() => void solveLevel2(), 300);
    return () => window.clearTimeout(timeout);
  }, [boundaryAvailable, boundarySolveKey, currentSolveKey, fidelity, liveSolveEnabled, solveLevel2, solveState]);

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
        <div className="project-breadcrumb"><span>Projects</span><ChevronRight size={13} /><strong>S218BP Subwoofer Study</strong><i>Edited</i></div>
        <FidelitySwitcher value={fidelity} onChange={setFidelity} packageLevel={pkg.manifest.fidelity_level} boundaryAvailable={boundaryAvailable} />
        <div className="topbar-actions">
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
            <SectionHeader icon={Speaker} title="Scene objects" />
            <SceneTree pkg={pkg} sources={sourceConfigs} selectedId={selectedInstance} onSelect={selectSceneObject} />
          </>
        ) : (
          <>
            <SectionHeader icon={Speaker} title="Scene hierarchy" />
            <SceneTree pkg={pkg} sources={sourceConfigs} selectedId={selectedInstance} onSelect={selectSceneObject} />
            <div className="scene-summary">
              <span>Subwoofer sources</span><strong>{sourceConfigs.length}</strong>
              <span>Observation points</span><strong>{observation.columns * observation.rows}</strong>
              <span>Excitation ports</span><strong>{pkg.manifest.excitation_port_ids.length}</strong>
            </div>
          </>
        )}
      </aside>

      <section
        className="viewport"
        data-transform-mode={transformMode}
        data-angle-snap-disabled={angleSnapDisabled}
        data-grab-point-count={selectedSource ? 8 : 0}
      >
        <SceneView
          pkg={pkg}
          sources={sources}
          observation={observation}
          field={field}
          selectedInstance={selectedInstance}
          transformMode={transformMode}
          angleSnapDisabled={angleSnapDisabled}
          onSelectInstance={selectSceneObject}
          onTransformSource={updateSourcePose}
        />
        <div className="viewport-toolbar">
          <button className={transformMode === "select" ? "active" : ""} title="Select (corner drag)" onClick={() => setTransformMode("select")}><MousePointer2 size={15} /></button>
          <button className={transformMode === "translate" ? "active" : ""} title="Translate (W)" onClick={() => setTransformMode("translate")}><Move3D size={15} /></button>
          <button className={transformMode === "rotate" ? "active" : ""} title="Rotate (E)" onClick={() => setTransformMode("rotate")}><Rotate3D size={15} /></button>
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
        <div className="viewport-hint">Orbit: left drag · Pan: right drag · Zoom: wheel</div>
      </section>

      <aside className="right-panel panel">
        {selectedInstance === "audience-plane" ? (
          <>
            <div className="inspector-heading">
              <div className="object-icon"><Grid3X3 size={19} /></div>
              <div><small>SELECTED OBJECT</small><strong>Audience plane</strong><span>Observation surface</span></div>
              <button className="icon-button quiet"><SlidersHorizontal size={15} /></button>
            </div>
            <PlaneResolutionInspector value={observation} onChange={setObservation} />
          </>
        ) : selectedSource ? (
          <>
            <div className="inspector-heading">
              <div className="object-icon"><Speaker size={19} /></div>
              <div><small>SELECTED OBJECT</small><strong>{pkg.manifest.name} {selectedSourceIndex + 1}</strong><span>Subwoofer source</span></div>
              <button className="icon-button quiet"><SlidersHorizontal size={15} /></button>
            </div>
            <SourceInspector config={selectedSource} minimumHeightM={sourceMinimumHeightM} onChange={updateSelectedSource} />
          </>
        ) : null}
      </aside>

      <section className="analysis-drawer">
        <div className="analysis-body">
          <div className="frequency-control">
            <div className="frequency-label"><span>Frequency</span><strong>{formatFrequency(pkg.frequenciesHz[frequencyIndex])}</strong></div>
            <input
              aria-label="Frequency"
              type="range"
              min={0}
              max={Math.max(0, sortedFrequencyIndices.length - 1)}
              step={1}
              value={sortedPosition}
              onChange={(event) => setFrequencyIndex(sortedFrequencyIndices[Number(event.target.value)])}
            />
            <div className="frequency-extents"><span>{formatFrequency(pkg.frequenciesHz[sortedFrequencyIndices[0]])}</span><span>{formatFrequency(pkg.frequenciesHz[sortedFrequencyIndices.at(-1)!])}</span></div>
          </div>
          <div className="metrics-grid">
            <div><span>Average</span><strong>{field.averageDb.toFixed(1)}<small> dB</small></strong></div>
            <div><span>Peak</span><strong>{field.maximumDb.toFixed(1)}<small> dB</small></strong></div>
            <div><span>P10–P90</span><strong>{field.spreadDb.toFixed(1)}<small> dB</small></strong></div>
            <div><span>Sources</span><strong>{sourceConfigs.length}<small>{boundaryCurrent ? " BEM" : " live"}</small></strong></div>
          </div>
          <div className="legend-block">
            <div className="legend-title"><span>SPL</span></div>
            <div className="color-legend" />
            <div><span>{(field.maximumDb - 24).toFixed(0)}</span><span>{field.maximumDb.toFixed(0)} dB</span></div>
          </div>
        </div>
      </section>

      {field.clippedNearFieldPoints > 0 && (
        <div className="near-field-warning"><CircleHelp size={15} /><span>Some plane samples are inside the package reference sphere; use Boundary fidelity for authoritative near-field results.</span></div>
      )}
      {error && <div className="error-toast" onClick={() => setError(null)}><strong>Boundary Lab Deploy</strong><span>{error}</span></div>}
      <input
        ref={fileInput}
        className="hidden-file-input"
        type="file"
        accept=".blabsp"
        onChange={browserFileHandler(loadBrowserFile)}
      />
    </main>
  );
}
