import { Gauge, Square, Waves } from "lucide-react";
import { useMemo, useState } from "react";
import { driverTraceColor } from "./DriverExcursionPlot";
import { TraceVisibilityFilter } from "./TraceVisibilityFilter";
import { usePlotDimensions } from "./usePlotDimensions";
import { pressureDisplayScale } from "../model/acousticLoading";

const FREQUENCY_MINIMUM_HZ = 20;
const MAJOR_FREQUENCIES_HZ = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

export interface ElectricalTrace {
  frequenciesHz?: Float64Array;
  name: string;
  impedanceMagnitudeOhm: Float32Array;
  impedancePhaseDeg: Float32Array;
  rmsCurrentA: Float32Array;
  realPowerW: Float32Array;
  acousticResistance?: Float32Array;
  differentialPressurePa?: Float32Array;
  acousticReactance?: Float32Array;
}

export interface ElectricalData {
  key: string;
  frequenciesHz: Float64Array;
  traces: Map<string, ElectricalTrace>;
}

type ElectricalView = "impedance" | "current" | "power" | "acoustic" | "differential";

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
  view = "impedance",
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
  view?: ElectricalView;
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
  const [frequencyMaximum, setFrequencyMaximum] = useState<2000 | 20000>(2000);
  const [acousticPart, setAcousticPart] = useState<"resistance" | "reactance">("resistance");
  const [pressurePeak, setPressurePeak] = useState(false);
  const [pressureKpa, setPressureKpa] = useState(false);
  const loadingView = view === "acoustic" || view === "differential";
  const pressureValues = (values?: Float32Array) => values?.map((value) => value * pressureDisplayScale(pressurePeak, pressureKpa)) ?? new Float32Array();
  const valuesFor = (trace: ElectricalTrace) => view === "differential" ? pressureValues(trace.differentialPressurePa) : view === "acoustic"
    ? (acousticPart === "resistance" ? trace.acousticResistance : trace.acousticReactance) ?? new Float32Array()
    : view === "impedance" ? trace.impedanceMagnitudeOhm : view === "current" ? trace.rmsCurrentA : trace.realPowerW;
  const [hiddenTraceIds, setHiddenTraceIds] = useState<Set<string>>(() => new Set());
  const traces = data ? Array.from(data.traces.entries()).filter(([, trace]) => view !== "differential" || trace.differentialPressurePa?.some(Number.isFinite)) : [];
  const { ref: chartRef, width, height } = usePlotDimensions();
  const padding = { left: 55, right: view === "impedance" ? 55 : 24, top: loadingView ? 55 : 13, bottom: 34 };
  const plotRight = Math.max(padding.left + 1, width - padding.right);
  const plotBottom = Math.max(padding.top + 1, height - padding.bottom);
  const logMinimum = Math.log10(FREQUENCY_MINIMUM_HZ);
  const logRange = Math.log10(frequencyMaximum) - logMinimum;
  const x = (frequency: number) => padding.left + ((Math.log10(frequency) - logMinimum) / logRange) * (plotRight - padding.left);
  const limits = useMemo(() => {
    let maximum = 0;
    if (data) for (const trace of data.traces.values()) {
      const values = valuesFor(trace);
      for (const value of values) if (Number.isFinite(value)) maximum = Math.max(maximum, view === "power" || view === "acoustic" ? Math.abs(value) : value);
    }
    const upper = niceMaximum(maximum * 1.08);
    return view === "power" || view === "acoustic" ? [-upper, upper] as const : [0, upper] as const;
  }, [data, view, acousticPart, pressurePeak, pressureKpa]);
  const y = (value: number) => padding.top + ((limits[1] - value) / (limits[1] - limits[0])) * (plotBottom - padding.top);
  const phaseY = (value: number) => padding.top + ((180 - value) / 360) * (plotBottom - padding.top);
  const path = (frequencies: Float64Array, values: Float32Array, ordinate: (value: number) => number) => {
    let result = "";
    let connected = false;
    for (let index = 0; index < Math.min(frequencies.length, values.length); index += 1) {
      if (!Number.isFinite(values[index]) || !Number.isFinite(frequencies[index]) || frequencies[index] < FREQUENCY_MINIMUM_HZ || frequencies[index] > frequencyMaximum) { connected = false; continue; }
      result += `${connected ? " L" : " M"}${x(frequencies[index]).toFixed(2)},${ordinate(values[index]).toFixed(2)}`;
      connected = true;
    }
    return result;
  };
  const yTicks = Array.from({ length: 5 }, (_, index) => limits[0] + (limits[1] - limits[0]) * index / 4);
  const phaseTicks = [-180, -90, 0, 90, 180];
  const majorFrequencies = MAJOR_FREQUENCIES_HZ.filter((frequency) => frequency <= frequencyMaximum);
  const cursorX = x(Math.max(FREQUENCY_MINIMUM_HZ, Math.min(frequencyMaximum, currentFrequencyHz)));
  const unit = view === "differential" ? `|Δp| ${pressurePeak ? "peak" : "RMS"} (${pressureKpa ? "kPa" : "Pa"})` : view === "acoustic" ? `${acousticPart === "resistance" ? "R" : "X"} / (ρcSd)` : view === "impedance" ? "|Z| (ohm)" : view === "current" ? "RMS current (A)" : "Real input power (W)";
  const toggleTrace = (id: string) => setHiddenTraceIds((current) => {
    const next = new Set(current);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });

  return <div className="microphone-response electrical-response">
    <div className="response-toolbar">
      <div className="response-title"><Gauge size={14} /><strong>{view === "differential" ? "Diaphragm Δp" : view === "acoustic" ? "Acoustic loading" : "Electrical"}</strong></div>
      <div className="electrical-view-switcher">
        {view === "differential" ? <>
          <button aria-label="RMS pressure differential" className={!pressurePeak ? "active" : ""} onClick={() => setPressurePeak(false)}>RMS</button>
          <button aria-label="Peak pressure differential" className={pressurePeak ? "active" : ""} onClick={() => setPressurePeak(true)}>Peak</button>
          <button aria-label="Pressure units Pa" className={!pressureKpa ? "active" : ""} onClick={() => setPressureKpa(false)}>Pa</button>
          <button aria-label="Pressure units kPa" className={pressureKpa ? "active" : ""} onClick={() => setPressureKpa(true)}>kPa</button>
        </> : view === "acoustic" ? <>
          <button className={acousticPart === "resistance" ? "active" : ""} onClick={() => setAcousticPart("resistance")}>Resistance</button>
          <button className={acousticPart === "reactance" ? "active" : ""} onClick={() => setAcousticPart("reactance")}>Reactance</button>
        </> : <span>{view === "impedance" ? "Impedance" : view === "current" ? "RMS current" : "Real input power"}</span>}
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
      {!coupledSelected ? <div className="response-empty">Select Level 3 Coupled fidelity to calculate {view === "differential" ? "diaphragm pressure differential" : view === "acoustic" ? "acoustic loading" : "electrical response"}.</div>
        : traces.length === 0 ? <div className="response-empty">{view === "differential" ? "Run a new coupled sweep to display force-equivalent diaphragm pressure differential. Valid transducer parameters and effective area are required. Select a speaker if the selection is empty." : view === "acoustic" ? "Run a coupled sweep to display normalized driver loading as frequencies are solved. A valid effective diaphragm area is required; select a speaker if the current selection is empty." : "Run the coupled frequency sweep to display each speaker object."}</div>
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
                {traces.map(([id, trace], index) => hiddenTraceIds.has(id) ? null : <path key={`${id}-${view}`} d={path(trace.frequenciesHz ?? data!.frequenciesHz, valuesFor(trace), y)} stroke={driverTraceColor(index)} className="bem-trace" />)}
                {view === "impedance" && traces.map(([id, trace], index) => hiddenTraceIds.has(id) ? null : <path key={`${id}-phase`} d={path(trace.frequenciesHz ?? data!.frequenciesHz, trace.impedancePhaseDeg, phaseY)} stroke={driverTraceColor(index)} className="electrical-phase-trace" />)}
              </g>
            </svg>
            <div className="response-legend electrical-legend">
              {view === "impedance" && <em><b className="line-sample" />Magnitude <b className="line-sample phase" />Phase</em>}
              {loadingView && <>
                <span style={{ pointerEvents: "auto" }} title={view === "differential" ? "Net opposing force divided by effective projected area. Peak is sinusoidal amplitude, not local cone stress or a damage limit. Zero-velocity pressure remains valid." : "Net opposing acoustic load divided by ρcSd. Reactance uses exp(+iωt); near-zero velocity impedance is omitted."}>{view === "differential" ? "Loading diagnostic, not a damage limit. " : ""}For comparison, sweep a single-cabinet scene and Capture results.</span></>}
            </div>
            <button className="response-range-toggle" type="button" onClick={() => setFrequencyMaximum((current) => current === 2000 ? 20000 : 2000)}>20 Hz–{frequencyMaximum === 2000 ? "2 kHz" : "20 kHz"}</button>
            </div>
            <TraceVisibilityFilter items={traces.map(([id, trace], index) => ({ id, name: trace.name, color: driverTraceColor(index) }))} hiddenIds={hiddenTraceIds} onToggle={toggleTrace} />
          </div>}
    </div>
  </div>;
}
