// Matches VisualizerConfig.custom_colors in the base Boundary Lab application.
export const ISOBAR_COLORS = [
  "#00008F",
  "#0000FF",
  "#006FFF",
  "#00DFFF",
  "#4FFFBF",
  "#BFFF4F",
  "#FFDF00",
  "#FF6F00",
  "#FF0000",
  "#8F0000",
] as const;

const ISOBAR_RGB = ISOBAR_COLORS.map((value) => {
  const packed = Number.parseInt(value.slice(1), 16);
  return [(packed >> 16) & 0xff, (packed >> 8) & 0xff, packed & 0xff] as const;
});

export function heatmapColorBoundaries(minimumDb: number, maximumDb: number, bandingDb: number): number[] {
  if (bandingDb <= 0 || maximumDb <= minimumDb) return [];
  const result = [minimumDb];
  const first = Math.ceil(minimumDb / bandingDb) * bandingDb;
  for (let boundary = first; boundary < maximumDb; boundary += bandingDb) {
    if (boundary > minimumDb) result.push(boundary);
  }
  result.push(maximumDb);
  return result;
}

function colorPosition(
  valueDb: number,
  minimumDb: number,
  maximumDb: number,
  boundaries: readonly number[],
): number {
  const range = Math.max(Number.EPSILON, maximumDb - minimumDb);
  if (boundaries.length < 2) return Math.max(0, Math.min(1, (valueDb - minimumDb) / range));
  const regionCount = boundaries.length - 1;
  if (regionCount === 1) return 0.5;
  if (valueDb <= minimumDb) return 0;
  if (valueDb >= maximumDb) return 1;
  let low = 1;
  let high = boundaries.length - 1;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (valueDb < boundaries[middle]) high = middle;
    else low = middle + 1;
  }
  return (low - 1) / (regionCount - 1);
}

function interpolatedColor(normalized: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, normalized));
  const scaled = clamped * (ISOBAR_RGB.length - 1);
  const leftIndex = Math.min(ISOBAR_RGB.length - 2, Math.floor(scaled));
  const local = scaled - leftIndex;
  const left = ISOBAR_RGB[leftIndex];
  const right = ISOBAR_RGB[leftIndex + 1];
  return [
    Math.round(left[0] + (right[0] - left[0]) * local),
    Math.round(left[1] + (right[1] - left[1]) * local),
    Math.round(left[2] + (right[2] - left[2]) * local),
  ];
}

export function writeHeatmapColor(
  valueDb: number,
  minimumDb: number,
  maximumDb: number,
  boundaries: readonly number[],
  pixels: Uint8Array,
  offset: number,
): void {
  const [red, green, blue] = interpolatedColor(colorPosition(valueDb, minimumDb, maximumDb, boundaries));
  pixels[offset] = red;
  pixels[offset + 1] = green;
  pixels[offset + 2] = blue;
}

export function heatmapLegendGradient(
  minimumDb: number,
  maximumDb: number,
  bandingDb: number,
): string {
  const boundaries = heatmapColorBoundaries(minimumDb, maximumDb, bandingDb);
  if (boundaries.length < 2) return `linear-gradient(90deg, ${ISOBAR_COLORS.join(", ")})`;
  const range = Math.max(Number.EPSILON, maximumDb - minimumDb);
  const regionCount = boundaries.length - 1;
  const stops: string[] = [];
  for (let region = 0; region < regionCount; region += 1) {
    const normalized = regionCount === 1 ? 0.5 : region / (regionCount - 1);
    const [red, green, blue] = interpolatedColor(normalized);
    const color = `rgb(${red} ${green} ${blue})`;
    const start = ((boundaries[region] - minimumDb) / range) * 100;
    const end = ((boundaries[region + 1] - minimumDb) / range) * 100;
    stops.push(`${color} ${start.toFixed(3)}%`, `${color} ${end.toFixed(3)}%`);
  }
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}
