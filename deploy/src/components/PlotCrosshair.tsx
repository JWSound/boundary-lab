import { useEffect, useRef, useState, type PointerEvent } from "react";

interface Axes {
  width: number; height: number; left: number; right: number; top: number; bottom: number;
  minimum: number; maximum: number; frequencyMaximum: number; identity: string;
}

export function crosshairCoordinates(axes: Axes, x: number, y: number) {
  const horizontal = Math.max(0, Math.min(1, (x - axes.left) / (axes.right - axes.left)));
  const vertical = Math.max(0, Math.min(1, (y - axes.top) / (axes.bottom - axes.top)));
  return {
    frequency: 20 * (axes.frequencyMaximum / 20) ** horizontal,
    value: axes.maximum - vertical * (axes.maximum - axes.minimum),
  };
}

/** Raw axis coordinates, matching the microphone cursor (not trace snapping). */
export function usePlotCrosshair(axes: Axes) {
  const pointer = useRef<number | null>(null);
  const [point, setPoint] = useState<{ frequency: number; value: number; identity: string } | null>(null);
  useEffect(() => { setPoint(null); pointer.current = null; }, [axes.identity]);
  const position = (event: PointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return null;
    return { x: (event.clientX - bounds.left) / bounds.width * axes.width, y: (event.clientY - bounds.top) / bounds.height * axes.height };
  };
  const update = (event: PointerEvent<SVGSVGElement>) => {
    const p = position(event);
    if (p) setPoint({ ...crosshairCoordinates(axes, p.x, p.y), identity: axes.identity });
  };
  const finish = (event: PointerEvent<SVGSVGElement>) => {
    if (pointer.current !== event.pointerId) return;
    pointer.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };
  const visible = point?.identity === axes.identity && point.frequency <= axes.frequencyMaximum
    && point.value >= axes.minimum && point.value <= axes.maximum ? point : null;
  return {
    point: visible,
    handlers: {
      onPointerDown: (event: PointerEvent<SVGSVGElement>) => {
        const p = position(event);
        if (event.button !== 0 || !p || p.x < axes.left || p.x > axes.right || p.y < axes.top || p.y > axes.bottom) return;
        pointer.current = event.pointerId;
        event.currentTarget.setPointerCapture(event.pointerId);
        update(event);
      },
      onPointerMove: (event: PointerEvent<SVGSVGElement>) => { if (pointer.current === event.pointerId) update(event); },
      onPointerUp: finish, onPointerCancel: finish,
      onLostPointerCapture: () => { pointer.current = null; },
      onDoubleClick: () => { pointer.current = null; setPoint(null); },
    },
  };
}

export function PlotCrosshair({ point, axes, secondary }: {
  point: { frequency: number; value: number } | null; axes: Axes; secondary?: boolean;
}) {
  if (!point) return null;
  const x = axes.left + Math.log(point.frequency / 20) / Math.log(axes.frequencyMaximum / 20) * (axes.right - axes.left);
  const fraction = (axes.maximum - point.value) / (axes.maximum - axes.minimum);
  const y = axes.top + fraction * (axes.bottom - axes.top);
  const labelY = Math.max(axes.top, Math.min(axes.bottom - 16, y - 8));
  const labelX = Math.max(axes.left, Math.min(axes.right - 72, x - 36));
  return <g style={{ pointerEvents: "none" }}>
    <line x1={x} x2={x} y1={axes.top} y2={axes.bottom} className="plot-crosshair" />
    <line x1={axes.left} x2={axes.right} y1={y} y2={y} className="plot-crosshair" />
    <g className="crosshair-labels">
      <rect x={labelX} y={axes.bottom + 2} width={72} height={16} />
      <text x={labelX + 36} y={axes.bottom + 13} textAnchor="middle">{point.frequency.toFixed(1)} Hz</text>
      <rect x={axes.left - 51} y={labelY} width={48} height={16} />
      <text x={axes.left - 27} y={labelY + 11} textAnchor="middle">{Math.abs(point.value) >= 1e5 || (Math.abs(point.value) > 0 && Math.abs(point.value) < 0.001) ? point.value.toExponential(1) : Number(point.value.toPrecision(4))}</text>
      {secondary && <><rect x={axes.right + 3} y={labelY} width={48} height={16} />
        <text x={axes.right + 27} y={labelY + 11} textAnchor="middle">{(180 - fraction * 360).toFixed(1)}°</text></>}
    </g>
  </g>;
}
