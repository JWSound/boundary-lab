import { useCallback, useLayoutEffect, useState } from "react";

export interface PlotDimensions {
  width: number;
  height: number;
}

export interface ResponsivePlotDimensions extends PlotDimensions {
  ref: (element: SVGSVGElement | null) => void;
}

export function usePlotDimensions(
  fallback: PlotDimensions = { width: 1000, height: 240 },
): ResponsivePlotDimensions {
  const [element, setElement] = useState<SVGSVGElement | null>(null);
  const [dimensions, setDimensions] = useState(fallback);
  const ref = useCallback((next: SVGSVGElement | null) => setElement(next), []);

  useLayoutEffect(() => {
    if (!element) return;

    const update = () => {
      const bounds = element.getBoundingClientRect();
      const width = Math.max(1, Math.round(bounds.width));
      const height = Math.max(1, Math.round(bounds.height));
      setDimensions((current) => current.width === width && current.height === height ? current : { width, height });
    };

    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [element]);

  return { ...dimensions, ref };
}
