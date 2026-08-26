import {
  Activity,
  ChevronRight,
  CircleHelp,
  Gauge,
  Import,
  Info,
  Menu,
  MousePointer2,
  Play,
  Rotate3D,
  Save,
  Settings2,
  SlidersHorizontal,
  Speaker,
  Sparkles,
  Waves,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { SceneView } from "./components/SceneView";
import {
  browserFileHandler,
  ObservationInspector,
  PackageCard,
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
  nearestFrequencyIndex,
} from "./model/field";
import type { Fidelity, LoadedSpeakerPackage, ObservationPlane, SourceConfiguration } from "./model/types";

const defaultSource: SourceConfiguration = {
  positionX: 0,
  positionHeightM: 0.4,
  positionZ: 0,
  yawDeg: 0,
  levelDb: -3,
  delayMs: 0,
  polarity: 1,
};

const defaultObservation: ObservationPlane = {
  widthM: 24,
  depthM: 24,
  nearM: 1.5,
  heightM: 1.2,
  columns: 54,
  rows: 46,
};

function formatFrequency(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)} kHz` : `${Math.round(value)} Hz`;
}

function FidelitySwitcher({
  value,
  onChange,
  packageLevel,
}: {
  value: Fidelity;
  onChange: (value: Fidelity) => void;
  packageLevel: number;
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
        const interactive = item.id === "pattern";
        return (
          <button
            key={item.id}
            className={`${value === item.id ? "active" : ""} ${!interactive ? "engine-required" : ""}`}
            onClick={() => interactive && onChange(item.id)}
            title={interactive ? "Live complex pattern field" : available ? "Boundary solve engine connection is the next prototype stage" : "Package does not contain this fidelity"}
          >
            <span>{item.label}</span>
            {!interactive && <small>{available ? "ENGINE" : "N/A"}</small>}
          </button>
        );
      })}
    </div>
  );
}

function ResponseSparkline({ pkg, frequencyIndex }: { pkg: LoadedSpeakerPackage; frequencyIndex: number }) {
  const points = useMemo(() => {
    const directionCount = pkg.pressureShape[2];
    const excitationCount = pkg.pressureShape[1];
    const values = Array.from(pkg.frequenciesHz, (_, index) => {
      let forwardIndex = 0;
      let forward = -Infinity;
      for (let direction = 0; direction < directionCount; direction += 1) {
        const value = pkg.directionsPackage[direction * 3 + 1];
        if (value > forward) { forward = value; forwardIndex = direction; }
      }
      const offset = (index * excitationCount) * directionCount + forwardIndex;
      return 20 * Math.log10(Math.max(1e-12, Math.hypot(pkg.pressure.real[offset], pkg.pressure.imag[offset])) / 20e-6);
    });
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    return values.map((value, index) => {
      const x = (index / Math.max(1, values.length - 1)) * 480;
      const y = 52 - ((value - minimum) / Math.max(1, maximum - minimum)) * 38;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  }, [pkg]);
  const selectedX = (frequencyIndex / Math.max(1, pkg.frequenciesHz.length - 1)) * 480;
  return (
    <svg className="response-sparkline" viewBox="0 0 480 60" preserveAspectRatio="none">
      <path d="M0 52 H480" className="spark-grid" />
      <polyline points={points} className="spark-line" />
      <line x1={selectedX} x2={selectedX} y1="4" y2="56" className="spark-cursor" />
    </svg>
  );
}

export function App() {
  const [pkg, setPackage] = useState<LoadedSpeakerPackage>(() => createDemoPackage());
  const [sourceConfig, setSourceConfig] = useState(defaultSource);
  const [observation, setObservation] = useState(defaultObservation);
  const [frequencyIndex, setFrequencyIndex] = useState(() => nearestFrequencyIndex(pkg, 80));
  const [fidelity, setFidelity] = useState<Fidelity>("pattern");
  const [selectedInstance, setSelectedInstance] = useState<string | null>("subwoofer-1");
  const [error, setError] = useState<string | null>(null);
  const [leftTab, setLeftTab] = useState<"library" | "scene">("library");
  const fileInput = useRef<HTMLInputElement>(null);

  const sortedFrequencyIndices = useMemo(
    () => Array.from(pkg.frequenciesHz.keys()).sort((a, b) => pkg.frequenciesHz[a] - pkg.frequenciesHz[b]),
    [pkg],
  );
  const sortedPosition = Math.max(0, sortedFrequencyIndices.indexOf(frequencyIndex));
  const source = useMemo(() => buildSourceInstance(sourceConfig), [sourceConfig]);
  const lookup = useMemo(() => buildPatternLookup(pkg, frequencyIndex), [pkg, frequencyIndex]);
  const field = useMemo(
    () => computeFieldFrame(pkg, source, sourceConfig, observation, frequencyIndex, lookup),
    [pkg, source, sourceConfig, observation, frequencyIndex, lookup],
  );

  const applyPackage = (next: LoadedSpeakerPackage) => {
    setPackage(next);
    setFrequencyIndex(nearestFrequencyIndex(next, 80));
    setFidelity("pattern");
    setError(null);
  };

  useEffect(() => {
    let active = true;
    if (!window.boundaryLabDesktop) return () => { active = false; };
    void window.boundaryLabDesktop.loadBundledExample()
      .then((selection) => {
        if (active && selection) applyPackage(loadSpeakerPackage(selection.bytes, selection.name));
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
        if (selection) applyPackage(loadSpeakerPackage(selection.bytes, selection.name));
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
      schema_version: 1,
      name: "S218BP Subwoofer Study",
      package: {
        id: pkg.id,
        name: pkg.manifest.name,
        source_file: pkg.isDemo ? null : pkg.fileName,
      },
      source: sourceConfig,
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

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Waves size={20} /></div>
          <div><strong>Boundary Lab</strong><span>DEPLOY</span></div>
        </div>
        <div className="project-breadcrumb"><span>Projects</span><ChevronRight size={13} /><strong>S218BP Subwoofer Study</strong><i>Edited</i></div>
        <FidelitySwitcher value={fidelity} onChange={setFidelity} packageLevel={pkg.manifest.fidelity_level} />
        <div className="topbar-actions">
          <button className="icon-button" title="Save project" onClick={saveProject}><Save size={17} /></button>
          <button className="icon-button" title="Settings"><Settings2 size={17} /></button>
          <button className="primary-button" disabled title="Boundary solve engine is not connected in this prototype"><Play size={14} fill="currentColor" /> Solve field</button>
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
            <SceneTree pkg={pkg} />
          </>
        ) : (
          <>
            <SectionHeader icon={Speaker} title="Scene hierarchy" />
            <SceneTree pkg={pkg} />
            <div className="scene-summary">
              <span>Subwoofer sources</span><strong>1</strong>
              <span>Observation points</span><strong>{observation.columns * observation.rows}</strong>
              <span>Excitation ports</span><strong>{pkg.manifest.excitation_port_ids.length}</strong>
            </div>
          </>
        )}
        <div className="left-footer">
          <Info size={14} /><span>Free-field scene · {pkg.manifest.medium.sound_speed_m_per_s} m/s</span>
        </div>
      </aside>

      <section className="viewport">
        <SceneView
          pkg={pkg}
          source={source}
          observation={observation}
          field={field}
          selectedInstance={selectedInstance}
          onSelectInstance={setSelectedInstance}
        />
        <div className="viewport-toolbar">
          <button className="active"><MousePointer2 size={15} /></button>
          <button><Rotate3D size={15} /></button>
          <span />
          <button><Menu size={15} /></button>
        </div>
        <div className="solve-status">
          <span className="live-dot" />
          <div><strong>Pattern preview</strong><small>Current · {formatFrequency(pkg.frequenciesHz[frequencyIndex])}</small></div>
          <em>{field.columns} × {field.rows}</em>
        </div>
        <div className="viewport-hint">Orbit: left drag · Pan: right drag · Zoom: wheel</div>
        <ObservationInspector value={observation} onChange={setObservation} />
      </section>

      <aside className="right-panel panel">
        <div className="inspector-heading">
          <div className="object-icon"><Speaker size={19} /></div>
          <div><small>SELECTED OBJECT</small><strong>{pkg.manifest.name}</strong><span>Subwoofer source</span></div>
          <button className="icon-button quiet"><SlidersHorizontal size={15} /></button>
        </div>
        <SourceInspector config={sourceConfig} onChange={setSourceConfig} />
        <div className="engine-callout">
          <Sparkles size={15} />
          <div><strong>Progressive physics</strong><span>A Boundary solve will refine this live pattern result when the exterior BEM engine is connected.</span></div>
        </div>
      </aside>

      <section className="analysis-drawer">
        <div className="analysis-heading">
          <div><Activity size={15} /><strong>Coverage analysis</strong><span>Complex pressure · audience plane</span></div>
          <button className="icon-button quiet"><Gauge size={15} /></button>
        </div>
        <div className="analysis-body">
          <div className="frequency-control">
            <div className="frequency-label"><span>Frequency</span><strong>{formatFrequency(pkg.frequenciesHz[frequencyIndex])}</strong></div>
            <ResponseSparkline pkg={pkg} frequencyIndex={frequencyIndex} />
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
            <div><span>Sources</span><strong>1<small> live</small></strong></div>
          </div>
          <div className="legend-block">
            <div className="legend-title"><span>SPL</span><small>relative 24 dB window</small></div>
            <div className="color-legend" />
            <div><span>{(field.maximumDb - 24).toFixed(0)}</span><span>{field.maximumDb.toFixed(0)} dB</span></div>
          </div>
        </div>
      </section>

      {field.clippedNearFieldPoints > 0 && (
        <div className="near-field-warning"><CircleHelp size={15} /><span>Some plane samples are inside the package reference sphere; use Boundary fidelity for authoritative near-field results.</span></div>
      )}
      {error && <div className="error-toast" onClick={() => setError(null)}><strong>Package could not be opened</strong><span>{error}</span></div>}
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
