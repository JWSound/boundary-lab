import type { FieldFrame, LoadedSpeakerPackage, ObservationPlane, RigidMeshConfiguration, SourceConfiguration } from "./types";
import { minimumSourceHeightM } from "./field";
import { DEFAULT_CHANNEL_ID } from "./channels";

export function peakExcursionMillimeters(real: number, imag: number, frequencyHz: number): number {
  return Math.SQRT2 * Math.hypot(real, imag) * 1000 / (2 * Math.PI * frequencyHz);
}

export function electricalSample(voltageReal: number, voltageImag: number, currentReal: number, currentImag: number) {
  const currentMagnitude = Math.hypot(currentReal, currentImag);
  if (currentMagnitude <= Number.EPSILON) return {
    impedanceMagnitudeOhm: Number.NaN,
    impedancePhaseDeg: Number.NaN,
    rmsCurrentA: 0,
    realPowerW: 0,
  };
  const impedanceReal = (voltageReal * currentReal + voltageImag * currentImag) / (currentMagnitude * currentMagnitude);
  const impedanceImag = (voltageImag * currentReal - voltageReal * currentImag) / (currentMagnitude * currentMagnitude);
  return {
    impedanceMagnitudeOhm: Math.hypot(impedanceReal, impedanceImag),
    impedancePhaseDeg: Math.atan2(impedanceImag, impedanceReal) * 180 / Math.PI,
    rmsCurrentA: currentMagnitude,
    realPowerW: voltageReal * currentReal + voltageImag * currentImag,
  };
}

export function emptyFieldFrame(observation: ObservationPlane): FieldFrame {
  const pointCount = observation.columns * observation.rows;
  return {
    splDb: new Float32Array(pointCount),
    pressureReal: new Float32Array(pointCount),
    pressureImag: new Float32Array(pointCount),
    validMask: new Uint8Array(pointCount),
    columns: observation.columns,
    rows: observation.rows,
    minimumDb: 0,
    maximumDb: 0,
    averageDb: 0,
    spreadDb: 0,
    clippedNearFieldPoints: 0,
  };
}

export type SolvedFieldEntry = { key: string; field: FieldFrame };
export type SolvedFieldCache = Record<"boundary" | "coupled", SolvedFieldEntry | null>;

export const emptySolvedFieldCache = (): SolvedFieldCache => ({ boundary: null, coupled: null });

export function defaultSources(pkg: LoadedSpeakerPackage): SourceConfiguration[] {
  const centerSpacingM = pkg.boundsM[0] + 2;
  const positionHeightM = minimumSourceHeightM(pkg);
  return [
    {
      id: "subwoofer-1",
      name: `${pkg.manifest.name} 1`,
      packageId: pkg.id,
      positionX: -centerSpacingM / 2,
      positionHeightM,
      positionZ: 0,
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
      channelId: DEFAULT_CHANNEL_ID,
      levelDb: -3,
      delayMs: 0,
      polarity: 1,
      equalizer: { filters: [] },
    },
    {
      id: "subwoofer-2",
      name: `${pkg.manifest.name} 2`,
      packageId: pkg.id,
      positionX: centerSpacingM / 2,
      positionHeightM,
      positionZ: 0,
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
      channelId: DEFAULT_CHANNEL_ID,
      levelDb: -3,
      delayMs: 0,
      polarity: 1,
      equalizer: { filters: [] },
    },
  ];
}

export function buildRigidInstance(config: RigidMeshConfiguration) {
  return {
    id: config.id,
    packageId: config.assetId,
    position: [config.positionX, config.positionHeightM, config.positionZ] as [number, number, number],
    pitchDeg: config.pitchDeg,
    yawDeg: config.yawDeg,
    rollDeg: config.rollDeg,
  };
}

export const defaultObservation: ObservationPlane = {
  widthM: 24,
  depthM: 24,
  centerXM: 0,
  nearM: 1.5,
  heightM: 1.2,
  pitchDeg: 0,
  yawDeg: 0,
  rollDeg: 0,
  columns: 54,
  rows: 54,
  heatmapMinimumDb: 50,
  heatmapMaximumDb: 145,
  heatmapBandingDb: 0,
  displayMode: "spl",
  pressureScalePa: 10,
  phaseAnimationSpeedHz: 1,
};

export function observationAcousticState(value: ObservationPlane) {
  return {
    widthM: value.widthM,
    depthM: value.depthM,
    centerXM: value.centerXM,
    nearM: value.nearM,
    heightM: value.heightM,
    pitchDeg: value.pitchDeg,
    yawDeg: value.yawDeg,
    rollDeg: value.rollDeg,
    columns: value.columns,
    rows: value.rows,
  };
}

export function formatFrequency(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)} kHz` : `${Math.round(value)} Hz`;
}
