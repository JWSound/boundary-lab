import { Square, Waves } from "lucide-react";
import { useMemo } from "react";
import type { MicrophoneResponseSet } from "../model/types";

const TRACE_COLORS = ["#ffdf00", "#00dfff", "#ff6f00", "#7fe35b", "#e08cff", "#ff748c"];

export interface BemResponseData {
  key: string;
  frequenciesHz: Float64Array;
  traces: Map<string, Float32Array>;
}

function formatFrequency(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : `${Math.round(value)}`;
}

export function MicrophoneResponsePlot({
  pattern,
  bem,
  currentFrequencyHz,
  frequencyPosition,
  frequencyCount,
  onFrequencyPositionChange,
  canCalculateBem,
  calculating,
  completedCount,
  totalCount,
  onCalculateOrStop,
}: {
  pattern: MicrophoneResponseSet;
  bem: BemResponseData | null;
  currentFrequencyHz: number;
  frequencyPosition: number;
  frequencyCount: number;
  onFrequencyPositionChange: (position: number) => void;
  canCalculateBem: boolean;
  calculating: boolean;
  completedCount: number;
  totalCount: number;
  onCalculateOrStop: () => void;
}) {
  const width = 1000;
  const height = 154;
  const padding = { left: 46, right: 16, top: 12, bottom: 25 };
  const frequencies = pattern.frequenciesHz;
  const limits = useMemo(() => {
    const values: number[] = [];
    for (const trace of pattern.traces) for (const value of trace.splDb) if (Number.isFinite(value)) values.push(value);
    if (bem) for (const trace of bem.traces.values()) for (const value of trace) if (Number.isFinite(value)) values.push(value);
    if (!values.length) return [50, 145] as const;
    const minimum = Math.floor((Math.min(...values) - 3) / 10) * 10;
    const maximum = Math.ceil((Math.max(...values) + 3) / 10) * 10;
    return [minimum, Math.max(minimum + 40, maximum)] as const;
  }, [bem, pattern]);
  const frequencyMinimum = frequencies[0] ?? 20;
  const frequencyMaximum = frequencies.at(-1) ?? 250;
  const logMinimum = Math.log10(frequencyMinimum);
  const logRange = Math.max(1e-9, Math.log10(frequencyMaximum) - logMinimum);
  const x = (frequency: number) => padding.left + ((Math.log10(frequency) - logMinimum) / logRange) * (width - padding.left - padding.right);
  const y = (spl: number) => padding.top + ((limits[1] - spl) / (limits[1] - limits[0])) * (height - padding.top - padding.bottom);
  const paths = (responseFrequencies: Float64Array, values: Float32Array) => {
    const result: string[] = [];
    let current = "";
    for (let index = 0; index < Math.min(responseFrequencies.length, values.length); index += 1) {
      const value = values[index];
      if (!Number.isFinite(value)) {
        if (current) result.push(current);
        current = "";
        continue;
      }
      current += `${current ? " L" : "M"}${x(responseFrequencies[index]).toFixed(2)},${y(value).toFixed(2)}`;
    }
    if (current) result.push(current);
    return result;
  };
  const yTicks = Array.from({ length: 5 }, (_, index) => limits[0] + ((limits[1] - limits[0]) * index) / 4);
  const xTicks = frequencies.length ? [frequencyMinimum, frequencies[Math.floor((frequencies.length - 1) / 2)], frequencyMaximum] : [];
  const cursorX = x(Math.max(frequencyMinimum, Math.min(frequencyMaximum, currentFrequencyHz)));

  return (
    <div className="microphone-response">
      <div className="response-toolbar">
        <div className="response-title"><Waves size={14} /><strong>Microphone response</strong></div>
        <span>{frequencyMinimum.toFixed(0)}–{frequencyMaximum.toFixed(0)} Hz · {frequencies.length} package frequencies</span>
        <label className="response-frequency">
          <span>{formatFrequency(currentFrequencyHz)} Hz</span>
          <input
            aria-label="Frequency"
            type="range"
            min={0}
            max={Math.max(0, frequencyCount - 1)}
            step={1}
            value={frequencyPosition}
            onChange={(event) => onFrequencyPositionChange(Number(event.target.value))}
          />
        </label>
        <button
          className={`bem-pressure-button ${calculating ? "stop" : ""}`}
          disabled={!calculating && !canCalculateBem}
          onClick={onCalculateOrStop}
        >{calculating ? <Square size={11} fill="currentColor" /> : <Waves size={12} />} {calculating ? `Stop ${completedCount}/${totalCount}` : "Calculate BEM Pressure"}</button>
      </div>
      <div className="response-content">
        {pattern.traces.length === 0 ? (
          <div className="response-empty">Add a microphone to display its package-derived frequency response.</div>
        ) : (
          <>
            <svg className="response-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="Microphone frequency response plot">
              <rect x={padding.left} y={padding.top} width={width - padding.left - padding.right} height={height - padding.top - padding.bottom} className="plot-well" />
              {yTicks.map((tick) => (
                <g key={tick}>
                  <line x1={padding.left} x2={width - padding.right} y1={y(tick)} y2={y(tick)} className="plot-grid" />
                  <text x={padding.left - 7} y={y(tick) + 3} textAnchor="end">{tick.toFixed(0)}</text>
                </g>
              ))}
              {xTicks.map((tick) => (
                <g key={tick}>
                  <line x1={x(tick)} x2={x(tick)} y1={padding.top} y2={height - padding.bottom} className="plot-grid" />
                  <text x={x(tick)} y={height - 7} textAnchor="middle">{formatFrequency(tick)} Hz</text>
                </g>
              ))}
              <line x1={cursorX} x2={cursorX} y1={padding.top} y2={height - padding.bottom} className="frequency-cursor" />
              {pattern.traces.flatMap((trace, index) => paths(pattern.frequenciesHz, trace.splDb).map((path, pathIndex) => (
                <path key={`pattern-${trace.microphoneId}-${pathIndex}`} d={path} stroke={TRACE_COLORS[index % TRACE_COLORS.length]} className="pattern-trace" />
              )))}
              {bem && pattern.traces.flatMap((trace, index) => {
                const values = bem.traces.get(trace.microphoneId);
                return values ? paths(bem.frequenciesHz, values).map((path, pathIndex) => (
                  <path key={`bem-${trace.microphoneId}-${pathIndex}`} d={path} stroke={TRACE_COLORS[index % TRACE_COLORS.length]} className="bem-trace" />
                )) : [];
              })}
              <text x={13} y={padding.top + 8} transform={`rotate(-90 13 ${padding.top + 8})`} className="axis-title">SPL dB</text>
            </svg>
            <div className="response-legend">
              {pattern.traces.map((trace, index) => (
                <span key={trace.microphoneId}><i style={{ background: TRACE_COLORS[index % TRACE_COLORS.length] }} />{trace.microphoneName}</span>
              ))}
              <em><b className="line-sample pattern" />Pattern</em>
              {bem && <em><b className="line-sample bem" />BEM</em>}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
