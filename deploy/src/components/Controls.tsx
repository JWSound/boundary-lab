import type { LucideIcon } from "lucide-react";
import { Box, CircleDot, Grid3X3, Mic2, Palette, Plus, Radio, Speaker } from "lucide-react";
import type { ChangeEvent, MouseEvent, ReactNode } from "react";
import type {
  LoadedSpeakerPackage,
  MicrophoneConfiguration,
  ObservationPlane,
  RigidMeshAsset,
  RigidMeshConfiguration,
  SourceConfiguration,
} from "../model/types";

export function SectionHeader({ icon: Icon, title, action }: { icon: LucideIcon; title: string; action?: ReactNode }) {
  return (
    <div className="section-header">
      <div className="section-title"><Icon size={15} strokeWidth={1.8} /><span>{title}</span></div>
      {action}
    </div>
  );
}

export function RigidMeshCard({
  asset,
  active,
  onSelect,
  onAdd,
}: {
  asset: RigidMeshAsset;
  active: boolean;
  onSelect: () => void;
  onAdd: () => void;
}) {
  return (
    <div className={`package-card ${active ? "active" : ""}`} data-rigid-mesh-id={asset.id} onClick={onSelect} role="button" tabIndex={0}>
      <div className="package-visual"><Box size={29} strokeWidth={1.2} /></div>
      <div className="package-details">
        <div className="package-name">{asset.name}</div>
        <div className="package-subtitle">{asset.fileName}</div>
        <div className="mesh-metrics">{asset.vertexCount.toLocaleString()} vertices / {asset.triangleCount.toLocaleString()} faces</div>
      </div>
      <button
        className="icon-button quiet"
        title={`Add ${asset.name} to scene`}
        aria-label={`Add ${asset.name} to scene`}
        onClick={(event) => { event.stopPropagation(); onAdd(); }}
      ><Plus size={15} /></button>
    </div>
  );
}

export function PackageCard({
  pkg,
  active,
  onSelect,
  onAdd,
}: {
  pkg: LoadedSpeakerPackage;
  active: boolean;
  onSelect: () => void;
  onAdd: () => void;
}) {
  const level = pkg.manifest.fidelity_level;
  return (
    <div className={`package-card ${active ? "active" : ""}`} data-package-id={pkg.id} onClick={onSelect} role="button" tabIndex={0}>
      <div className="package-visual"><Speaker size={30} strokeWidth={1.2} /></div>
      <div className="package-details">
        <div className="package-name">{pkg.manifest.name}</div>
        <div className="package-subtitle">{pkg.isDemo ? "Built-in prototype model" : pkg.fileName}</div>
        <div className="fidelity-ticks" aria-label={`Fidelity level ${level}`}>
          {[1, 2, 3].map((item) => <span key={item} className={item <= level ? "active" : ""} />)}
          <small>L{level}</small>
        </div>
      </div>
      <button
        className="icon-button quiet"
        title={`Add ${pkg.manifest.name} to scene`}
        aria-label={`Add ${pkg.manifest.name} to scene`}
        onClick={(event) => { event.stopPropagation(); onAdd(); }}
      ><Plus size={15} /></button>
    </div>
  );
}

export function SceneTree({
  packages,
  rigidMeshes,
  sources,
  rigidObjects,
  microphones,
  selectedIds,
  activeId,
  onSelect,
}: {
  packages: LoadedSpeakerPackage[];
  rigidMeshes: RigidMeshAsset[];
  sources: SourceConfiguration[];
  rigidObjects: RigidMeshConfiguration[];
  microphones: MicrophoneConfiguration[];
  selectedIds: readonly string[];
  activeId: string | null;
  onSelect: (id: string, additive: boolean) => void;
}) {
  const packageById = new Map(packages.map((pkg) => [pkg.id, pkg]));
  const rigidMeshById = new Map(rigidMeshes.map((asset) => [asset.id, asset]));
  const select = (id: string, event: MouseEvent<HTMLButtonElement>) => {
    onSelect(id, event.ctrlKey || event.metaKey);
  };
  return (
    <div className="scene-tree">
      {sources.map((source) => (
        <button
          key={source.id}
          data-object-id={source.id}
          aria-selected={selectedIds.includes(source.id)}
          className={`tree-row tree-button ${selectedIds.includes(source.id) ? "selected" : ""} ${activeId === source.id ? "active-selection" : ""}`}
          onClick={(event) => select(source.id, event)}
        ><Speaker size={15} /><span>{source.name}</span><em>{packageById.get(source.packageId)?.manifest.name ?? "SUB"}</em></button>
      ))}
      {rigidObjects.map((object) => (
        <button
          key={object.id}
          data-object-id={object.id}
          aria-selected={selectedIds.includes(object.id)}
          className={`tree-row tree-button ${selectedIds.includes(object.id) ? "selected" : ""} ${activeId === object.id ? "active-selection" : ""}`}
          onClick={(event) => select(object.id, event)}
        ><Box size={15} /><span>{object.name}</span><em>{rigidMeshById.get(object.assetId)?.name ?? "RIGID"}</em></button>
      ))}
      {microphones.map((microphone) => (
        <button
          key={microphone.id}
          data-object-id={microphone.id}
          aria-selected={selectedIds.includes(microphone.id)}
          className={`tree-row tree-button ${selectedIds.includes(microphone.id) ? "selected" : ""} ${activeId === microphone.id ? "active-selection" : ""}`}
          onClick={(event) => select(microphone.id, event)}
        ><Mic2 size={15} /><span>{microphone.name}</span><em>MIC</em></button>
      ))}
      <button
        data-object-id="audience-plane"
        aria-selected={selectedIds.includes("audience-plane")}
        className={`tree-row tree-button ${selectedIds.includes("audience-plane") ? "selected" : ""} ${activeId === "audience-plane" ? "active-selection" : ""}`}
        onClick={(event) => select("audience-plane", event)}
      ><Grid3X3 size={15} /><span>Audience plane</span><em>PLANE</em></button>
    </div>
  );
}

export function RigidMeshInspector({
  config,
  onChange,
}: {
  config: RigidMeshConfiguration;
  onChange: (next: RigidMeshConfiguration) => void;
}) {
  const set = <K extends keyof RigidMeshConfiguration>(key: K, value: RigidMeshConfiguration[K]) => onChange({ ...config, [key]: value });
  return (
    <>
      <SectionHeader icon={Box} title="Placement" />
      <div className="inspector-section two-column-fields">
        <NumberField label="X" value={config.positionX} unit="m" step={0.1} onChange={(value) => set("positionX", value)} />
        <NumberField label="Height" value={config.positionHeightM} unit="m" step={0.1} minimum={0} onChange={(value) => set("positionHeightM", value)} />
        <NumberField label="Depth" value={config.positionZ} unit="m" step={0.1} onChange={(value) => set("positionZ", value)} />
        <NumberField label="Pitch" value={config.pitchDeg} unit="deg" step={0.5} onChange={(value) => set("pitchDeg", value)} />
        <NumberField label="Yaw" value={config.yawDeg} unit="deg" step={0.5} onChange={(value) => set("yawDeg", value)} />
        <NumberField label="Roll" value={config.rollDeg} unit="deg" step={0.5} onChange={(value) => set("rollDeg", value)} />
      </div>
    </>
  );
}

export function MicrophoneInspector({
  config,
  onChange,
}: {
  config: MicrophoneConfiguration;
  onChange: (next: MicrophoneConfiguration) => void;
}) {
  const set = <K extends keyof MicrophoneConfiguration>(key: K, value: MicrophoneConfiguration[K]) => (
    onChange({ ...config, [key]: value })
  );
  return (
    <>
      <SectionHeader icon={CircleDot} title="Placement" />
      <div className="inspector-section two-column-fields">
        <NumberField label="X" value={config.positionX} unit="m" step={0.1} onChange={(value) => set("positionX", value)} />
        <NumberField label="Height" value={config.positionHeightM} unit="m" step={0.1} minimum={0} onChange={(value) => set("positionHeightM", value)} />
        <NumberField label="Depth" value={config.positionZ} unit="m" step={0.1} onChange={(value) => set("positionZ", value)} />
      </div>
    </>
  );
}

interface SliderProps {
  label: string;
  value: number;
  minimum: number;
  maximum: number;
  step: number;
  unit?: string;
  onChange: (value: number) => void;
}

export function Slider({ label, value, minimum, maximum, step, unit = "", onChange }: SliderProps) {
  return (
    <label className="control-row slider-row">
      <span>{label}</span>
      <input
        aria-label={label}
        type="range"
        min={minimum}
        max={maximum}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <output>{value.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0)}{unit}</output>
    </label>
  );
}

export function NumberField({
  label,
  value,
  unit,
  step,
  minimum,
  maximum,
  onChange,
}: {
  label: string;
  value: number;
  unit: string;
  step: number;
  minimum?: number;
  maximum?: number;
  onChange: (value: number) => void;
}) {
  const constrained = (next: number) => Math.min(maximum ?? Infinity, Math.max(minimum ?? -Infinity, next));
  return (
    <label className="control-row number-row">
      <span>{label}</span>
      <div><input aria-label={label} type="number" value={value} min={minimum} max={maximum} step={step} onChange={(event) => onChange(constrained(Number(event.target.value)))} /><em>{unit}</em></div>
    </label>
  );
}

export function SourceInspector({
  config,
  minimumHeightM,
  onChange,
}: {
  config: SourceConfiguration;
  minimumHeightM: number;
  onChange: (next: SourceConfiguration) => void;
}) {
  const set = <K extends keyof SourceConfiguration>(key: K, value: SourceConfiguration[K]) => onChange({ ...config, [key]: value });
  return (
    <>
      <SectionHeader icon={CircleDot} title="Placement" />
      <div className="inspector-section two-column-fields">
        <NumberField label="X" value={config.positionX} unit="m" step={0.1} onChange={(value) => set("positionX", value)} />
        <NumberField label="Height" value={config.positionHeightM} unit="m" step={0.1} minimum={minimumHeightM} onChange={(value) => set("positionHeightM", value)} />
        <NumberField label="Depth" value={config.positionZ} unit="m" step={0.1} onChange={(value) => set("positionZ", value)} />
        <NumberField label="Pitch" value={config.pitchDeg} unit="deg" step={0.5} onChange={(value) => set("pitchDeg", value)} />
        <NumberField label="Yaw" value={config.yawDeg} unit="°" step={0.5} onChange={(value) => set("yawDeg", value)} />
        <NumberField label="Roll" value={config.rollDeg} unit="deg" step={0.5} onChange={(value) => set("rollDeg", value)} />
      </div>
      <SectionHeader icon={Radio} title="Drive" />
      <div className="inspector-section">
        <Slider label="Level" value={config.levelDb} minimum={-24} maximum={12} step={0.5} unit=" dB" onChange={(value) => set("levelDb", value)} />
        <Slider label="Delay" value={config.delayMs} minimum={0} maximum={20} step={0.05} unit=" ms" onChange={(value) => set("delayMs", value)} />
        <label className="control-row toggle-row">
          <span>Polarity</span>
          <button
            className={config.polarity === -1 ? "toggle active" : "toggle"}
            onClick={() => set("polarity", config.polarity === 1 ? -1 : 1)}
          >{config.polarity === 1 ? "Normal" : "Inverted"}</button>
        </label>
      </div>
    </>
  );
}

export function planeGridShape(widthM: number, depthM: number, majorSamples: number): [number, number] {
  if (widthM >= depthM) {
    return [majorSamples, Math.max(2, Math.round(((majorSamples - 1) * depthM) / widthM) + 1)];
  }
  return [Math.max(2, Math.round(((majorSamples - 1) * widthM) / depthM) + 1), majorSamples];
}

export function PlaneResolutionInspector({
  value,
  onChange,
}: {
  value: ObservationPlane;
  onChange: (next: ObservationPlane) => void;
}) {
  const resolution = Math.max(value.columns, value.rows);
  const set = <K extends keyof ObservationPlane>(key: K, next: ObservationPlane[K]) => {
    onChange({ ...value, [key]: next });
  };
  const setResolution = (next: number) => {
    const [columns, rows] = planeGridShape(value.widthM, value.depthM, next);
    onChange({ ...value, columns, rows });
  };
  return (
    <>
      <SectionHeader icon={CircleDot} title="Placement" />
      <div className="inspector-section two-column-fields">
        <NumberField label="X" value={value.centerXM} unit="m" step={0.1} onChange={(next) => set("centerXM", next)} />
        <NumberField label="Near" value={value.nearM} unit="m" step={0.1} onChange={(next) => set("nearM", next)} />
        <NumberField label="Height" value={value.heightM} unit="m" step={0.1} onChange={(next) => set("heightM", next)} />
        <NumberField label="Pitch" value={value.pitchDeg} unit="deg" step={0.5} onChange={(next) => set("pitchDeg", next)} />
        <NumberField label="Yaw" value={value.yawDeg} unit="°" step={0.5} onChange={(next) => set("yawDeg", next)} />
        <NumberField label="Roll" value={value.rollDeg} unit="deg" step={0.5} onChange={(next) => set("rollDeg", next)} />
      </div>
      <SectionHeader icon={Grid3X3} title="Size" />
      <div className="inspector-section">
        <div className="plane-size-readout">
          <span>Length × width</span>
          <output>{value.depthM.toFixed(2)} × {value.widthM.toFixed(2)} m</output>
        </div>
      </div>
      <SectionHeader icon={Grid3X3} title="Sampling" />
      <div className="inspector-section">
        <label className="control-row plane-resolution-row">
          <span>Resolution</span>
          <input
            aria-label="Plane resolution"
            type="range"
            min={12}
            max={200}
            step={2}
            value={resolution}
            onChange={(event) => setResolution(Number(event.target.value))}
          />
          <output>{value.columns} × {value.rows}</output>
        </label>
      </div>
      <SectionHeader icon={Palette} title="Heatmap" />
      <div className="inspector-section">
        <div className="two-column-fields">
          <NumberField
            label="Scale minimum"
            value={value.heatmapMinimumDb}
            unit="dB"
            step={1}
            maximum={value.heatmapMaximumDb - 1}
            onChange={(next) => set("heatmapMinimumDb", next)}
          />
          <NumberField
            label="Scale maximum"
            value={value.heatmapMaximumDb}
            unit="dB"
            step={1}
            minimum={value.heatmapMinimumDb + 1}
            onChange={(next) => set("heatmapMaximumDb", next)}
          />
        </div>
        <Slider
          label="Banding"
          value={value.heatmapBandingDb}
          minimum={0}
          maximum={12}
          step={1}
          unit=" dB"
          onChange={(next) => set("heatmapBandingDb", next)}
        />
      </div>
    </>
  );
}

export function browserFileHandler(onFile: (file: File) => void): (event: ChangeEvent<HTMLInputElement>) => void {
  return (event) => {
    const file = event.target.files?.[0];
    if (file) onFile(file);
    event.target.value = "";
  };
}
