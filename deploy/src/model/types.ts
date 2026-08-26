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
  sourcePath: string | null;
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
  id: string;
  positionX: number;
  positionHeightM: number;
  positionZ: number;
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
  levelDb: number;
  delayMs: number;
  polarity: 1 | -1;
}

export interface SpeakerInstance {
  id: string;
  position: [number, number, number];
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
}

export interface ObservationPlane {
  widthM: number;
  depthM: number;
  centerXM: number;
  nearM: number;
  heightM: number;
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
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
  validMask: Uint8Array;
  columns: number;
  rows: number;
  minimumDb: number;
  maximumDb: number;
  averageDb: number;
  spreadDb: number;
  clippedNearFieldPoints: number;
}

export interface Level2SolveResult {
  frequency_hz: number;
  rows: number;
  columns: number;
  spl_db: number[];
  sample_indices: number[];
  field_pressure: { real: number[]; imag: number[] };
  timings: { assembly_s: number; solve_s: number; field_s: number };
  diagnostics: {
    backend: string;
    source_count: number;
    node_count: number;
    face_count: number;
    singular_pair_count: number;
    near_face_pair_count: number;
    quadrature_order: number;
    singular_order: number;
    close_pair_quadrature_order: number;
    close_pair_distance_m: number;
    isolated_trace_relative_difference: number;
    minimum_surface_distance_m: number | null;
    close_pair_count: number;
    surface_padding_m: number;
  };
}
