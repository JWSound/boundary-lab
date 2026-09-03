import { Gauge, Square, Waves } from "lucide-react";
import { useMemo, useState } from "react";
import { driverTraceColor } from "./DriverExcursionPlot";
import { TraceVisibilityFilter } from "./TraceVisibilityFilter";
import { usePlotDimensions } from "./usePlotDimensions";

const FREQUENCY_MINIMUM_HZ = 20;
const MAJOR_FREQUENCIES_HZ = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

export interface ElectricalTrace {
  name: string;
  impedanceMagnitudeOhm: Float32Array;
  impedancePhaseDeg: Float32Array;
  rmsCurrentA: Float32Array;
  realPowerW: Float32Array;
}

export interface ElectricalData {
  key: string;
  frequenciesHz: Float64Array;
  traces: Map<string, ElectricalTrace>;
}

type ElectricalView = "impedance" | "current" | "power";

function formatFrequency(value: number): string {
  if (value < 1000) return `${Math.round(value)}`;
  const kilohertz = value / 1000;
  return `${Number.isInteger(kilohertz) ? kilohertz.toFixed(0) : kilohertz.toFixed(1)}k`;
}

function niceMaximum(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  return (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10) * magnitude;
}

function minorFrequencyTicks(maximumHz: number): number[] {
  const major = new Set(MAJOR_FREQUENCIES_HZ);
  const result: number[] = [];
  for (let decade = 10; decade <= maximumHz; decade *= 10) for (let multiple = 2; multiple < 10; multiple += 1) {
    const frequency = decade * multiple;
    if (frequency >= FREQUENCY_MINIMUM_HZ && frequency <= maximumHz && !major.has(frequency)) result.push(frequency);
  }
  return result;
}

export function ElectricalPlot({
  data,
  coupledSelected,
  currentFrequencyHz,
  frequencyPosition,
  frequencyCount,
  onFrequencyPositionChange,
  canCalculate,
  calculating,
  completedCount,
  totalCount,
  onCalculateOrStop,
}: {
  data: ElectricalData | null;
  coupledSelected: boolean;
  currentFrequencyHz: number;
  frequencyPosition: number;
  frequencyCount: number;
  onFrequencyPositionChange: (position: number) => void;
  canCalculate: boolean;
  calculating: boolean;
  completedCount: number;
  totalCount: number;
  onCalculateOrStop: () => void;
}) {
  const [view, setView] = useState<ElectricalView>("impedance");
  const [frequencyMaximum, setFrequencyMaximum] = useState<2000 | 20000>(2000);
  const [hiddenTraceIds, setHiddenTraceIds] = useState<Set<string>>(() => new Set());
  const traces = data ? Array.from(data.traces.entries()) : [];
  const { ref: chartRef, width, height } = usePlotDimensions();
  const padding = { left: 55, right: view === "impedance" ? 55 : 24, top: 13, bottom: 34 };
  const plotRight = Math.max(padding.left + 1, width - padding.right);
  const plotBottom = Math.max(padding.top + 1, height - padding.bottom);
  const logMinimum = Math.log10(FREQUENCY_MINIMUM_HZ);
  const logRange = Math.log10(frequencyMaximum) - logMinimum;
  const x = (frequency: number) => padding.left + ((Math.log10(frequency) - logMinimum) / logRange) * (plotRight - padding.left);
  const limits = useMemo(() => {
    let maximum = 0;
    if (data) for (const trace of data.traces.values()) {
      const values = view === "impedance" ? trace.impedanceMagnitudeOhm : view === "current" ? trace.rmsCurrentA : trace.realPowerW;
      for (const value of values) if (Number.isFinite(value)) maximum = Math.max(maximum, view === "power" ? Math.abs(value) : value);
    }
    const upper = niceMaximum(maximum * 1.08);
    return view === "power" ? [-upper, upper] as const : [0, upper] as const;
  }, [data, view]);
  const y = (value: number) => padding.top + ((limits[1] - value) / (limits[1] - limits[0])) * (plotBottom - padding.top);
  const phaseY = (value: number) => padding.top + ((180 - value) / 360) * (plotBottom - padding.top);
  const path = (frequencies: Float64Array, values: Float32Array, ordinate: (value: number) => number) => {
    let result = "";
    for (let index = 0; index < Math.min(frequencies.length, values.length); index += 1) {
      if (!Number.isFinite(values[index]) || frequencies[index] > frequencyMaximum) continue;
      result += `${result ? " L" : "M"}${x(frequencies[index]).toFixed(2)},${ordinate(values[index]).toFixed(2)}`;
    }
    return result;
  };
  const yTicks = Array.from({ length: 5 }, (_, index) => limits[0] + (limits[1] - limits[0]) * index / 4);
  const phaseTicks = [-180, -90, 0, 90, 180];
  const majorFrequencies = MAJOR_FREQUENCIES_HZ.filter((frequency) => frequency <= frequencyMaximum);
  const cursorX = x(Math.max(FREQUENCY_MINIMUM_HZ, Math.min(frequencyMaximum, currentFrequencyHz)));
  const unit = view === "impedance" ? "|Z| (ohm)" : view === "current" ? "RMS current (A)" : "Real input power (W)";
  const toggleTrace = (id: string) => setHiddenTraceIds((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });

  return <div className="microphone-response electrical-response">
    <div className="response-toolbar">
      <div className="response-title"><Gauge size={14} /><strong>Electrical</strong></div>
      <div className="electrical-view-switcher" role="tablist" aria-label="Electrical quantity">
        <button className={view === "impedance" ? "active" : ""} onClick={() => setView("impedance")}>Impedance</button>
        <button className={view === "current" ? "active" : ""} onClick={() => setView("current")}>RMS current</button>
        <button className={view === "power" ? "active" : ""} onClick={() => setView("power")}>Real power</button>
      </div>
      <label className="response-frequency">
        <span>{formatFrequency(currentFrequencyHz)} Hz</span>
        <input aria-label="Frequency" type="range" min={0} max={Math.max(0, frequencyCount - 1)} step={1} value={frequencyPosition} onChange={(event) => onFrequencyPositionChange(Number(event.target.value))} />
      </label>
      <button className={`bem-pressure-button ${calculating ? "stop" : ""}`} disabled={!calculating && !canCalculate} onClick={onCalculateOrStop}>
        {calculating ? <Square size={11} fill="currentColor" /> : <Waves size={12} />} {calculating ? `Stop ${completedCount}/${totalCount}` : "Calculate Coupled Sweep"}
      </button>
    </div>
    <div className="response-content">
      {!coupledSelected ? <div className="response-empty">Select Level 3 Coupled fidelity to calculate electrical response.</div>
        : traces.length === 0 ? <div className="response-empty">Run the coupled frequency sweep to display each speaker object.</div>
          : <div className="response-plot-layout">
            <div className="response-plot-area">
            <svg ref={chartRef} className="response-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${unit} per speaker over frequency`}>
              <defs><clipPath id="electrical-plot-clip"><rect x={padding.left} y={padding.top} width={plotRight - padding.left} height={plotBottom - padding.top} /></clipPath></defs>
              <rect x={padding.left} y={padding.top} width={plotRight - padding.left} height={plotBottom - padding.top} className="plot-well" />
              {minorFrequencyTicks(frequencyMaximum).map((tick) => <line key={`xm-${tick}`} x1={x(tick)} x2={x(tick)} y1={padding.top} y2={plotBottom} className="plot-grid minor" />)}
              {yTicks.map((tick, index) => <g key={index}><line x1={padding.left} x2={plotRight} y1={y(tick)} y2={y(tick)} className="plot-grid major" /><text x={padding.left - 7} y={y(tick) + 3} textAnchor="end">{Math.abs(tick) < 0.1 ? tick.toFixed(3) : tick.toFixed(1)}</text></g>)}
              {majorFrequencies.map((tick) => <g key={tick}><line x1={x(tick)} x2={x(tick)} y1={padding.top} y2={plotBottom} className="plot-grid major" /><text x={x(tick)} y={height - 18} textAnchor="middle">{formatFrequency(tick)}</text></g>)}
              {view === "impedance" && phaseTicks.map((tick) => <text key={tick} x={plotRight + 7} y={phaseY(tick) + 3}>{tick}°</text>)}
              <text x={(padding.left + plotRight) / 2} y={height - 5} textAnchor="middle" className="axis-title">Frequency (Hz)</text>
              <text x={13} y={(padding.top + plotBottom) / 2} transform={`rotate(-90 13 ${(padding.top + plotBottom) / 2})`} textAnchor="middle" className="axis-title">{unit}</text>
              {view === "impedance" && <text x={width - 12} y={(padding.top + plotBottom) / 2} transform={`rotate(90 ${width - 12} ${(padding.top + plotBottom) / 2})`} textAnchor="middle" className="axis-title">Phase (deg)</text>}
              <g clipPath="url(#electrical-plot-clip)">
                <line x1={cursorX} x2={cursorX} y1={padding.top} y2={plotBottom} className="frequency-cursor" />
                {traces.map(([id, trace], index) => hiddenTraceIds.has(id) ? null : <path key={`${id}-${view}`} d={path(data!.frequenciesHz, view === "impedance" ? trace.impedanceMagnitudeOhm : view === "current" ? trace.rmsCurrentA : trace.realPowerW, y)} stroke={driverTraceColor(index)} className="bem-trace" />)}
                {view === "impedance" && traces.map(([id, trace], index) => hiddenTraceIds.has(id) ? null : <path key={`${id}-phase`} d={path(data!.frequenciesHz, trace.impedancePhaseDeg, phaseY)} stroke={driverTraceColor(index)} className="electrical-phase-trace" />)}
              </g>
            </svg>
            <div className="response-legend electrical-legend">
              {view === "impedance" && <em><b className="line-sample" />Magnitude <b className="line-sample phase" />Phase</em>}
            </div>
            <button className="response-range-toggle" type="button" onClick={() => setFrequencyMaximum((current) => current === 2000 ? 20000 : 2000)}>20 Hz–{frequencyMaximum === 2000 ? "2 kHz" : "20 kHz"}</button>
            </div>
            <TraceVisibilityFilter items={traces.map(([id, trace], index) => ({ id, name: trace.name, color: driverTraceColor(index) }))} hiddenIds={hiddenTraceIds} onToggle={toggleTrace} />
          </div>}
    </div>
  </div>;
}
