import { Square, Waves } from "lucide-react";
import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { MicrophoneResponseSet } from "../model/types";

const TRACE_COLORS = ["#ffdf00", "#00dfff", "#ff6f00", "#7fe35b", "#e08cff", "#ff748c"];
const AUDIO_FREQUENCY_MINIMUM_HZ = 20;
const AUDIO_FREQUENCY_MAJOR_TICKS_HZ = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
const RESPONSE_DB_SPAN = 50;

export interface BemResponseData {
  key: string;
  frequenciesHz: Float64Array;
  traces: Map<string, Float32Array>;
}

function formatFrequency(value: number): string {
  if (value < 1000) return `${Math.round(value)}`;
  const kilohertz = value / 1000;
  return `${Number.isInteger(kilohertz) ? kilohertz.toFixed(0) : kilohertz.toFixed(1)}k`;
}

function logarithmicMinorTicks(maximumHz: number): number[] {
  const major = new Set(AUDIO_FREQUENCY_MAJOR_TICKS_HZ);
  const ticks: number[] = [];
  for (let decade = 10; decade <= maximumHz; decade *= 10) {
    for (let multiple = 2; multiple < 10; multiple += 1) {
      const frequency = decade * multiple;
      if (frequency >= AUDIO_FREQUENCY_MINIMUM_HZ && frequency <= maximumHz && !major.has(frequency)) ticks.push(frequency);
    }
  }
  return ticks;
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
  const height = 240;
  const padding = { left: 48, right: 86, top: 13, bottom: 30 };
  const [frequencyMaximum, setFrequencyMaximum] = useState<2000 | 20000>(2000);
  const [crosshair, setCrosshair] = useState<{ frequencyHz: number; splDb: number } | null>(null);
  const [crosshairDragging, setCrosshairDragging] = useState(false);
  const crosshairDraggingRef = useRef(false);
  const frequencies = pattern.frequenciesHz;
  const limits = useMemo(() => {
    const values: number[] = [];
    for (const trace of pattern.traces) for (const value of trace.splDb) if (Number.isFinite(value)) values.push(value);
    if (bem) for (const trace of bem.traces.values()) for (const value of trace) if (Number.isFinite(value)) values.push(value);
    const maximum = values.length ? Math.ceil(Math.max(...values) / 5) * 5 : 145;
    return [maximum - RESPONSE_DB_SPAN, maximum] as const;
  }, [bem, pattern]);
  const logMinimum = Math.log10(AUDIO_FREQUENCY_MINIMUM_HZ);
  const logRange = Math.max(1e-9, Math.log10(frequencyMaximum) - logMinimum);
  const plotRight = width - padding.right;
  const plotBottom = height - padding.bottom;
  const x = (frequency: number) => padding.left + ((Math.log10(frequency) - logMinimum) / logRange) * (plotRight - padding.left);
  const y = (spl: number) => padding.top + ((limits[1] - spl) / RESPONSE_DB_SPAN) * (plotBottom - padding.top);
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
  const firstMajorY = Math.ceil(limits[0] / 10) * 10;
  const firstMinorY = Math.ceil(limits[0] / 5) * 5;
  const yMajorTicks = Array.from(
    { length: Math.max(0, Math.floor((limits[1] - firstMajorY) / 10) + 1) },
    (_, index) => firstMajorY + index * 10,
  );
  const yMinorTicks = Array.from(
    { length: Math.max(0, Math.floor((limits[1] - firstMinorY) / 5) + 1) },
    (_, index) => firstMinorY + index * 5,
  ).filter((tick) => tick % 10 !== 0);
  const xMajorTicks = AUDIO_FREQUENCY_MAJOR_TICKS_HZ.filter((frequency) => frequency <= frequencyMaximum);
  const xMinorTicks = logarithmicMinorTicks(frequencyMaximum);
  const cursorX = x(Math.max(AUDIO_FREQUENCY_MINIMUM_HZ, Math.min(frequencyMaximum, currentFrequencyHz)));

  const updateCrosshair = (event: ReactPointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    const svgX = ((event.clientX - bounds.left) / bounds.width) * width;
    const svgY = ((event.clientY - bounds.top) / bounds.height) * height;
    const clampedX = Math.max(padding.left, Math.min(plotRight, svgX));
    const clampedY = Math.max(padding.top, Math.min(plotBottom, svgY));
    const frequencyHz = 10 ** (logMinimum + ((clampedX - padding.left) / (plotRight - padding.left)) * logRange);
    const splDb = limits[1] - ((clampedY - padding.top) / (plotBottom - padding.top)) * RESPONSE_DB_SPAN;
    setCrosshair({ frequencyHz, splDb });
  };

  const beginCrosshair = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    const svgX = ((event.clientX - bounds.left) / bounds.width) * width;
    const svgY = ((event.clientY - bounds.top) / bounds.height) * height;
    if (svgX < padding.left || svgX > plotRight || svgY < padding.top || svgY > plotBottom) return;
    crosshairDraggingRef.current = true;
    setCrosshairDragging(true);
    event.currentTarget.setPointerCapture(event.pointerId);
    updateCrosshair(event);
  };

  const finishCrosshair = (event: ReactPointerEvent<SVGSVGElement>) => {
    crosshairDraggingRef.current = false;
    setCrosshairDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const crosshairX = crosshair ? x(crosshair.frequencyHz) : 0;
  const crosshairY = crosshair ? y(crosshair.splDb) : 0;
  const crosshairXLabelWidth = 65;
  const crosshairXLabelX = Math.max(padding.left, Math.min(plotRight - crosshairXLabelWidth, crosshairX - crosshairXLabelWidth / 2));
  const crosshairYLabelWidth = 42;
  const crosshairYLabelY = Math.max(padding.top, Math.min(plotBottom - 16, crosshairY - 8));

  return (
    <div className="microphone-response">
      <div className="response-toolbar">
        <div className="response-title"><Waves size={14} /><strong>Microphone response</strong></div>
        <span>{frequencies.length} package frequencies</span>
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
            <svg
              className={`response-chart ${crosshairDragging ? "dragging" : ""}`}
              viewBox={`0 0 ${width} ${height}`}
              preserveAspectRatio="none"
              role="img"
              aria-label="Microphone frequency response plot. Drag inside the plot for frequency and SPL coordinates; double-click to clear the crosshair."
              data-frequency-maximum-hz={frequencyMaximum}
              onPointerDown={beginCrosshair}
              onPointerMove={(event) => { if (crosshairDraggingRef.current) updateCrosshair(event); }}
              onPointerUp={finishCrosshair}
              onPointerCancel={finishCrosshair}
              onDoubleClick={() => setCrosshair(null)}
            >
              <defs><clipPath id="microphone-response-clip"><rect x={padding.left} y={padding.top} width={plotRight - padding.left} height={plotBottom - padding.top} /></clipPath></defs>
              <rect x={padding.left} y={padding.top} width={plotRight - padding.left} height={plotBottom - padding.top} className="plot-well" />
              {yMinorTicks.map((tick) => <line key={`y-minor-${tick}`} x1={padding.left} x2={plotRight} y1={y(tick)} y2={y(tick)} className="plot-grid minor" />)}
              {xMinorTicks.map((tick) => <line key={`x-minor-${tick}`} x1={x(tick)} x2={x(tick)} y1={padding.top} y2={plotBottom} className="plot-grid minor" />)}
              {yMajorTicks.map((tick) => (
                <g key={tick}>
                  <line x1={padding.left} x2={plotRight} y1={y(tick)} y2={y(tick)} className="plot-grid major" />
                  <text x={padding.left - 7} y={y(tick) + 3} textAnchor="end">{tick.toFixed(0)}</text>
                </g>
              ))}
              {xMajorTicks.map((tick) => (
                <g key={tick}>
                  <line x1={x(tick)} x2={x(tick)} y1={padding.top} y2={plotBottom} className="plot-grid major" />
                  <text x={x(tick)} y={height - 14} textAnchor="middle">{formatFrequency(tick)}</text>
                </g>
              ))}
              <text x={(padding.left + plotRight) / 2} y={height - 3} textAnchor="middle" className="axis-title">Frequency (Hz)</text>
              <g clipPath="url(#microphone-response-clip)">
                <line x1={cursorX} x2={cursorX} y1={padding.top} y2={plotBottom} className="frequency-cursor" />
                {pattern.traces.flatMap((trace, index) => paths(pattern.frequenciesHz, trace.splDb).map((path, pathIndex) => (
                  <path key={`pattern-${trace.microphoneId}-${pathIndex}`} d={path} stroke={TRACE_COLORS[index % TRACE_COLORS.length]} className="pattern-trace" />
                )))}
                {bem && pattern.traces.flatMap((trace, index) => {
                  const values = bem.traces.get(trace.microphoneId);
                  return values ? paths(bem.frequenciesHz, values).map((path, pathIndex) => (
                    <path key={`bem-${trace.microphoneId}-${pathIndex}`} d={path} stroke={TRACE_COLORS[index % TRACE_COLORS.length]} className="bem-trace" />
                  )) : [];
                })}
                {crosshair && <>
                  <line x1={crosshairX} x2={crosshairX} y1={padding.top} y2={plotBottom} className="plot-crosshair" />
                  <line x1={padding.left} x2={plotRight} y1={crosshairY} y2={crosshairY} className="plot-crosshair" />
                </>}
              </g>
              {crosshair && <g className="crosshair-labels">
                <rect x={crosshairXLabelX} y={plotBottom + 2} width={crosshairXLabelWidth} height={16} />
                <text x={crosshairXLabelX + crosshairXLabelWidth / 2} y={plotBottom + 13} textAnchor="middle">{formatFrequency(crosshair.frequencyHz)} Hz</text>
                <rect x={padding.left - crosshairYLabelWidth - 3} y={crosshairYLabelY} width={crosshairYLabelWidth} height={16} />
                <text x={padding.left - crosshairYLabelWidth / 2 - 3} y={crosshairYLabelY + 11} textAnchor="middle">{crosshair.splDb.toFixed(1)}</text>
              </g>}
              <text x={13} y={padding.top + 8} transform={`rotate(-90 13 ${padding.top + 8})`} className="axis-title">SPL (dB)</text>
            </svg>
            <div className="response-legend">
              {pattern.traces.map((trace, index) => (
                <span key={trace.microphoneId}><i style={{ background: TRACE_COLORS[index % TRACE_COLORS.length] }} />{trace.microphoneName}</span>
              ))}
              <em><b className="line-sample pattern" />Pattern</em>
              {bem && <em><b className="line-sample bem" />BEM</em>}
            </div>
            <button
              className="response-range-toggle"
              type="button"
              title={`Switch frequency axis to 20 Hz–${frequencyMaximum === 2000 ? "20 kHz" : "2 kHz"}`}
              onClick={() => setFrequencyMaximum((current) => current === 2000 ? 20000 : 2000)}
            >20 Hz–{frequencyMaximum === 2000 ? "2 kHz" : "20 kHz"}</button>
          </>
        )}
      </div>
    </div>
  );
}
