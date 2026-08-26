export type Fidelity = "pattern" | "boundary" | "coupled";

export interface ComplexSamples {
  real: Float32Array;
  imag: Float32Array;
}

export interface SpeakerMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

export interface SpeakerPackageManifest {
  schema: string;
  schema_version: number;
  name: string;
  fidelity: Fidelity | "fixed";
  fidelity_level: number;
  capabilities: string[];
  frequencies_hz: number[];
  excitation_port_ids: string[];
  phasor_convention: string;
  coordinate_system?: {
    forward_axis?: string;
    unit?: string;
  };
  files: Record<string, { path: string; [key: string]: unknown }>;
  medium: {
    sound_speed_m_per_s: number;
    density_kg_per_m3: number;
  };
}

export interface LoadedSpeakerPackage {
  id: string;
  fileName: string;
  manifest: SpeakerPackageManifest;
  frequenciesHz: Float64Array;
  directionsPackage: Float32Array;
  radiiM: Float32Array;
  pressure: ComplexSamples;
  pressureShape: [number, number, number];
  mesh: SpeakerMesh | null;
  boundsM: [number, number, number];
  isDemo: boolean;
}

export interface SourceConfiguration {
  positionX: number;
  positionHeightM: number;
  positionZ: number;
  yawDeg: number;
  levelDb: number;
  delayMs: number;
  polarity: 1 | -1;
}

export interface SpeakerInstance {
  id: string;
  position: [number, number, number];
  pitchDeg: number;
  yawDeg: number;
}

export interface ObservationPlane {
  widthM: number;
  depthM: number;
  nearM: number;
  heightM: number;
  columns: number;
  rows: number;
}

export interface PatternLookup {
  azimuthBins: number;
  elevationBins: number;
  real: Float32Array;
  imag: Float32Array;
  radius: Float32Array;
}

export interface FieldFrame {
  splDb: Float32Array;
  columns: number;
  rows: number;
  minimumDb: number;
  maximumDb: number;
  averageDb: number;
  spreadDb: number;
  clippedNearFieldPoints: number;
}
