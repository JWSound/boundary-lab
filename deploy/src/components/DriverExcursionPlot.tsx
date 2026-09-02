import { Activity, Square } from "lucide-react";
import { useMemo, useState } from "react";
import { usePlotDimensions } from "./usePlotDimensions";

const AUDIO_FREQUENCY_MINIMUM_HZ = 20;
const AUDIO_FREQUENCY_MAJOR_TICKS_HZ = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

export interface DriverExcursionTrace {
  name: string;
  excursionMm: Float32Array;
}

export interface DriverExcursionData {
  key: string;
  frequenciesHz: Float64Array;
  traces: Map<string, DriverExcursionTrace>;
}

export function driverTraceColor(index: number): string {
  return `hsl(${(52 + index * 137.508) % 360} 82% 61%)`;
}

function formatFrequency(value: number): string {
  if (value < 1000) return `${Math.round(value)}`;
  const kilohertz = value / 1000;
  return `${Number.isInteger(kilohertz) ? kilohertz.toFixed(0) : kilohertz.toFixed(1)}k`;
}

function logarithmicMinorTicks(maximumHz: number): number[] {
  const major = new Set(AUDIO_FREQUENCY_MAJOR_TICKS_HZ);
  const ticks: number[] = [];
  for (let decade = 10; decade <= maximumHz; decade *= 10) for (let multiple = 2; multiple < 10; multiple += 1) {
    const frequency = decade * multiple;
    if (frequency >= AUDIO_FREQUENCY_MINIMUM_HZ && frequency <= maximumHz && !major.has(frequency)) ticks.push(frequency);
  }
  return ticks;
}

function niceMaximum(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  const rounded = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return rounded * magnitude;
}

export function DriverExcursionPlot({
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
  data: DriverExcursionData | null;
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
  const { ref: chartRef, width, height } = usePlotDimensions();
  const padding = { left: 54, right: 24, top: 13, bottom: 34 };
  const [frequencyMaximum, setFrequencyMaximum] = useState<2000 | 20000>(2000);
  const traces = data ? Array.from(data.traces.entries()) : [];
  const maximumExcursion = useMemo(() => {
    let maximum = 0;
    if (data) for (const trace of data.traces.values()) for (const value of trace.excursionMm) {
      if (Number.isFinite(value)) maximum = Math.max(maximum, value);
    }
    return niceMaximum(maximum * 1.08);
  }, [data]);
  const logMinimum = Math.log10(AUDIO_FREQUENCY_MINIMUM_HZ);
  const logRange = Math.log10(frequencyMaximum) - logMinimum;
  const plotRight = Math.max(padding.left + 1, width - padding.right);
  const plotBottom = Math.max(padding.top + 1, height - padding.bottom);
  const x = (frequency: number) => padding.left + ((Math.log10(frequency) - logMinimum) / logRange) * (plotRight - padding.left);
  const y = (value: number) => padding.top + (1 - value / maximumExcursion) * (plotBottom - padding.top);
  const paths = (frequencies: Float64Array, values: Float32Array) => {
    const result: string[] = [];
    let current = "";
    for (let index = 0; index < Math.min(frequencies.length, values.length); index += 1) {
      const value = values[index];
      if (!Number.isFinite(value) || frequencies[index] > frequencyMaximum) {
        if (current) result.push(current);
        current = "";
      } else current += `${current ? " L" : "M"}${x(frequencies[index]).toFixed(2)},${y(value).toFixed(2)}`;
    }
    if (current) result.push(current);
    return result;
  };
  const yTicks = Array.from({ length: 6 }, (_, index) => maximumExcursion * index / 5);
  const xMajorTicks = AUDIO_FREQUENCY_MAJOR_TICKS_HZ.filter((frequency) => frequency <= frequencyMaximum);
  const xMinorTicks = logarithmicMinorTicks(frequencyMaximum);
  const cursorX = x(Math.max(AUDIO_FREQUENCY_MINIMUM_HZ, Math.min(frequencyMaximum, currentFrequencyHz)));

  return <div className="microphone-response driver-excursion-response">
    <div className="response-toolbar">
      <div className="response-title"><Activity size={14} /><strong>Driver excursion</strong></div>
      <span>Peak displacement magnitude</span>
      <label className="response-frequency">
        <span>{formatFrequency(currentFrequencyHz)} Hz</span>
        <input aria-label="Frequency" type="range" min={0} max={Math.max(0, frequencyCount - 1)} step={1} value={frequencyPosition} onChange={(event) => onFrequencyPositionChange(Number(event.target.value))} />
      </label>
      <button className={`bem-pressure-button ${calculating ? "stop" : ""}`} disabled={!calculating && !canCalculate} onClick={onCalculateOrStop}>
        {calculating ? <Square size={11} fill="currentColor" /> : <Activity size={12} />} {calculating ? `Stop ${completedCount}/${totalCount}` : "Calculate Coupled Sweep"}
      </button>
    </div>
    <div className="response-content">
      {!coupledSelected ? <div className="response-empty">Select Level 3 Coupled fidelity to calculate driver excursion.</div>
        : traces.length === 0 ? <div className="response-empty">Run the coupled frequency sweep to display every transducer in the scene.</div>
          : <>
            <svg ref={chartRef} className="response-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Per-transducer peak driver excursion over frequency">
              <defs><clipPath id="driver-excursion-clip"><rect x={padding.left} y={padding.top} width={plotRight - padding.left} height={plotBottom - padding.top} /></clipPath></defs>
              <rect x={padding.left} y={padding.top} width={plotRight - padding.left} height={plotBottom - padding.top} className="plot-well" />
              {xMinorTicks.map((tick) => <line key={`xm-${tick}`} x1={x(tick)} x2={x(tick)} y1={padding.top} y2={plotBottom} className="plot-grid minor" />)}
              {yTicks.map((tick, index) => <g key={index}><line x1={padding.left} x2={plotRight} y1={y(tick)} y2={y(tick)} className="plot-grid major" /><text x={padding.left - 7} y={y(tick) + 3} textAnchor="end">{tick < 0.1 ? tick.toFixed(3) : tick.toFixed(1)}</text></g>)}
              {xMajorTicks.map((tick) => <g key={tick}><line x1={x(tick)} x2={x(tick)} y1={padding.top} y2={plotBottom} className="plot-grid major" /><text x={x(tick)} y={height - 18} textAnchor="middle">{formatFrequency(tick)}</text></g>)}
              <text x={(padding.left + plotRight) / 2} y={height - 5} textAnchor="middle" className="axis-title">Frequency (Hz)</text>
              <text x={13} y={(padding.top + plotBottom) / 2} transform={`rotate(-90 13 ${(padding.top + plotBottom) / 2})`} textAnchor="middle" className="axis-title">Peak excursion (mm)</text>
              <g clipPath="url(#driver-excursion-clip)">
                <line x1={cursorX} x2={cursorX} y1={padding.top} y2={plotBottom} className="frequency-cursor" />
                {traces.flatMap(([id, trace], index) => paths(data!.frequenciesHz, trace.excursionMm).map((path, pathIndex) => <path key={`${id}-${pathIndex}`} d={path} stroke={driverTraceColor(index)} className="bem-trace" />))}
              </g>
            </svg>
            <div className="response-legend driver-response-legend">{traces.map(([id, trace], index) => <span key={id}><i style={{ background: driverTraceColor(index) }} />{trace.name}</span>)}</div>
            <button className="response-range-toggle" type="button" onClick={() => setFrequencyMaximum((current) => current === 2000 ? 20000 : 2000)}>20 Hz–{frequencyMaximum === 2000 ? "2 kHz" : "20 kHz"}</button>
          </>}
    </div>
  </div>;
}
