import type { Fidelity, LoadedSpeakerPackage, ObservationPlane, SourceConfiguration } from "../model/types";

export const DEPLOY_PROJECT_SCHEMA = "boundary-lab-deploy-project";
export const DEPLOY_PROJECT_SCHEMA_VERSION = 2;

export interface DeployProject {
  schema: typeof DEPLOY_PROJECT_SCHEMA;
  schema_version: typeof DEPLOY_PROJECT_SCHEMA_VERSION;
  name: string;
  package: {
    id: string;
    name: string;
    source_file: string | null;
  };
  sources: SourceConfiguration[];
  observation_plane: ObservationPlane;
  selected_frequency_hz: number;
  requested_fidelity: Fidelity;
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} must be a finite number.`);
  return value;
}

function positive(value: unknown, label: string): number {
  const result = finite(value, label);
  if (result <= 0) throw new Error(`${label} must be greater than zero.`);
  return result;
}

function gridSize(value: unknown, label: string): number {
  const result = finite(value, label);
  if (!Number.isInteger(result) || result < 2 || result > 200) {
    throw new Error(`${label} must be an integer between 2 and 200.`);
  }
  return result;
}

function sourceConfiguration(value: unknown, index: number): SourceConfiguration {
  const source = record(value, `sources[${index}]`);
  if (typeof source.id !== "string" || source.id.trim().length === 0) {
    throw new Error(`sources[${index}].id must be a non-empty string.`);
  }
  const polarity = finite(source.polarity, `sources[${index}].polarity`);
  if (polarity !== 1 && polarity !== -1) throw new Error(`sources[${index}].polarity must be 1 or -1.`);
  return {
    id: source.id,
    positionX: finite(source.positionX, `sources[${index}].positionX`),
    positionHeightM: finite(source.positionHeightM, `sources[${index}].positionHeightM`),
    positionZ: finite(source.positionZ, `sources[${index}].positionZ`),
    pitchDeg: finite(source.pitchDeg, `sources[${index}].pitchDeg`),
    yawDeg: finite(source.yawDeg, `sources[${index}].yawDeg`),
    rollDeg: finite(source.rollDeg, `sources[${index}].rollDeg`),
    levelDb: finite(source.levelDb, `sources[${index}].levelDb`),
    delayMs: finite(source.delayMs, `sources[${index}].delayMs`),
    polarity: polarity as 1 | -1,
  };
}

function observationPlane(value: unknown): ObservationPlane {
  const plane = record(value, "observation_plane");
  const minimumDb = finite(plane.heatmapMinimumDb, "observation_plane.heatmapMinimumDb");
  const maximumDb = finite(plane.heatmapMaximumDb, "observation_plane.heatmapMaximumDb");
  if (maximumDb <= minimumDb) throw new Error("The heatmap maximum must be greater than its minimum.");
  const bandingDb = finite(plane.heatmapBandingDb, "observation_plane.heatmapBandingDb");
  if (bandingDb < 0) throw new Error("observation_plane.heatmapBandingDb cannot be negative.");
  return {
    widthM: positive(plane.widthM, "observation_plane.widthM"),
    depthM: positive(plane.depthM, "observation_plane.depthM"),
    centerXM: finite(plane.centerXM, "observation_plane.centerXM"),
    nearM: finite(plane.nearM, "observation_plane.nearM"),
    heightM: finite(plane.heightM, "observation_plane.heightM"),
    pitchDeg: finite(plane.pitchDeg, "observation_plane.pitchDeg"),
    yawDeg: finite(plane.yawDeg, "observation_plane.yawDeg"),
    rollDeg: finite(plane.rollDeg, "observation_plane.rollDeg"),
    columns: gridSize(plane.columns, "observation_plane.columns"),
    rows: gridSize(plane.rows, "observation_plane.rows"),
    heatmapMinimumDb: minimumDb,
    heatmapMaximumDb: maximumDb,
    heatmapBandingDb: bandingDb,
  };
}

export function parseDeployProject(contents: string): DeployProject {
  let raw: unknown;
  try {
    raw = JSON.parse(contents);
  } catch {
    throw new Error("The selected project is not valid JSON.");
  }
  const project = record(raw, "Project");
  if (project.schema !== DEPLOY_PROJECT_SCHEMA) throw new Error("This is not a Boundary Lab Deploy project.");
  const version = finite(project.schema_version, "schema_version");
  if (version !== DEPLOY_PROJECT_SCHEMA_VERSION) {
    throw new Error(`Unsupported Boundary Lab Deploy project schema version ${version}.`);
  }
  if (typeof project.name !== "string" || project.name.trim().length === 0) {
    throw new Error("name must be a non-empty string.");
  }
  const packageReference = record(project.package, "package");
  if (typeof packageReference.id !== "string" || packageReference.id.trim().length === 0) {
    throw new Error("package.id must be a non-empty string.");
  }
  if (typeof packageReference.name !== "string" || packageReference.name.trim().length === 0) {
    throw new Error("package.name must be a non-empty string.");
  }
  if (packageReference.source_file !== null && typeof packageReference.source_file !== "string") {
    throw new Error("package.source_file must be a string or null.");
  }
  const sourcesValue = project.sources;
  if (!Array.isArray(sourcesValue) || sourcesValue.length === 0) throw new Error("A project must contain at least one source.");
  const sources = sourcesValue.map(sourceConfiguration);
  if (new Set(sources.map((source) => source.id)).size !== sources.length) {
    throw new Error("Every project source must have a unique id.");
  }
  const frequencyHz = positive(project.selected_frequency_hz, "selected_frequency_hz");
  const requestedFidelity = project.requested_fidelity;
  if (requestedFidelity !== "pattern" && requestedFidelity !== "boundary" && requestedFidelity !== "coupled") {
    throw new Error("requested_fidelity must be pattern, boundary, or coupled.");
  }
  return {
    schema: DEPLOY_PROJECT_SCHEMA,
    schema_version: DEPLOY_PROJECT_SCHEMA_VERSION,
    name: project.name.trim(),
    package: {
      id: packageReference.id,
      name: packageReference.name,
      source_file: packageReference.source_file,
    },
    sources,
    observation_plane: observationPlane(project.observation_plane),
    selected_frequency_hz: frequencyHz,
    requested_fidelity: requestedFidelity,
  };
}

export function createDeployProject(
  name: string,
  pkg: LoadedSpeakerPackage,
  sources: SourceConfiguration[],
  observation: ObservationPlane,
  selectedFrequencyHz: number,
  requestedFidelity: Fidelity,
): DeployProject {
  return {
    schema: DEPLOY_PROJECT_SCHEMA,
    schema_version: DEPLOY_PROJECT_SCHEMA_VERSION,
    name,
    package: { id: pkg.id, name: pkg.manifest.name, source_file: pkg.sourcePath },
    sources,
    observation_plane: observation,
    selected_frequency_hz: selectedFrequencyHz,
    requested_fidelity: requestedFidelity,
  };
}

export function serializeDeployProject(project: DeployProject): string {
  return `${JSON.stringify(project, null, 2)}\n`;
}
