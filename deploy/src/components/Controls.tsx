import type { LucideIcon } from "lucide-react";
import { CircleDot, Grid3X3, Radio, Route, Speaker, Upload } from "lucide-react";
import type { ChangeEvent, ReactNode } from "react";
import type {
  LoadedSpeakerPackage,
  ObservationPlane,
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

export function PackageCard({ pkg, onOpen }: { pkg: LoadedSpeakerPackage; onOpen: () => void }) {
  const level = pkg.manifest.fidelity_level;
  return (
    <div className="package-card">
      <div className="package-visual"><Speaker size={30} strokeWidth={1.2} /></div>
      <div className="package-details">
        <div className="package-name">{pkg.manifest.name}</div>
        <div className="package-subtitle">{pkg.isDemo ? "Built-in prototype model" : pkg.fileName}</div>
        <div className="fidelity-ticks" aria-label={`Fidelity level ${level}`}>
          {[1, 2, 3].map((item) => <span key={item} className={item <= level ? "active" : ""} />)}
          <small>L{level}</small>
        </div>
      </div>
      <button className="icon-button quiet" title="Replace package" onClick={onOpen}><Upload size={15} /></button>
    </div>
  );
}

export function SceneTree({ pkg }: { pkg: LoadedSpeakerPackage }) {
  return (
    <div className="scene-tree">
      <div className="tree-row selected"><Speaker size={15} /><span>{pkg.manifest.name}</span><em>SUB</em></div>
      <div className="tree-row"><Grid3X3 size={15} /><span>Audience plane</span></div>
      <div className="tree-row"><Route size={15} /><span>Main DSP bus</span></div>
    </div>
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
  onChange,
}: {
  label: string;
  value: number;
  unit: string;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="control-row number-row">
      <span>{label}</span>
      <div><input type="number" value={value} step={step} onChange={(event) => onChange(Number(event.target.value))} /><em>{unit}</em></div>
    </label>
  );
}

export function SourceInspector({
  config,
  onChange,
}: {
  config: SourceConfiguration;
  onChange: (next: SourceConfiguration) => void;
}) {
  const set = <K extends keyof SourceConfiguration>(key: K, value: SourceConfiguration[K]) => onChange({ ...config, [key]: value });
  return (
    <>
      <SectionHeader icon={CircleDot} title="Placement" />
      <div className="inspector-section two-column-fields">
        <NumberField label="X" value={config.positionX} unit="m" step={0.1} onChange={(value) => set("positionX", value)} />
        <NumberField label="Height" value={config.positionHeightM} unit="m" step={0.1} onChange={(value) => set("positionHeightM", value)} />
        <NumberField label="Depth" value={config.positionZ} unit="m" step={0.1} onChange={(value) => set("positionZ", value)} />
        <NumberField label="Yaw" value={config.yawDeg} unit="°" step={0.5} onChange={(value) => set("yawDeg", value)} />
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

export function ObservationInspector({
  value,
  onChange,
}: {
  value: ObservationPlane;
  onChange: (next: ObservationPlane) => void;
}) {
  const set = <K extends keyof ObservationPlane>(key: K, next: ObservationPlane[K]) => onChange({ ...value, [key]: next });
  return (
    <div className="observation-quickbar">
      <span><Grid3X3 size={14} /> Audience plane</span>
      <NumberField label="Width" value={value.widthM} unit="m" step={1} onChange={(next) => set("widthM", next)} />
      <NumberField label="Depth" value={value.depthM} unit="m" step={1} onChange={(next) => set("depthM", next)} />
      <NumberField label="Height" value={value.heightM} unit="m" step={0.1} onChange={(next) => set("heightM", next)} />
    </div>
  );
}

export function browserFileHandler(onFile: (file: File) => void): (event: ChangeEvent<HTMLInputElement>) => void {
  return (event) => {
    const file = event.target.files?.[0];
    if (file) onFile(file);
    event.target.value = "";
  };
}
