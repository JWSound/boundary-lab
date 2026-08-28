import type { Fidelity, LoadedSpeakerPackage, MicrophoneConfiguration, ObservationPlane, RigidMeshAsset, RigidMeshConfiguration, SourceConfiguration } from "../model/types";

export const DEPLOY_PROJECT_SCHEMA = "boundary-lab-deploy-project";
export const DEPLOY_PROJECT_SCHEMA_VERSION = 5;

export interface DeployPackageReference {
  id: string;
  name: string;
  source_file: string | null;
}

export interface DeployRigidMeshReference {
  id: string;
  name: string;
  source_file: string | null;
  scale_to_meters: number;
}

export interface DeployProject {
  schema: typeof DEPLOY_PROJECT_SCHEMA;
  schema_version: typeof DEPLOY_PROJECT_SCHEMA_VERSION;
  name: string;
  packages: DeployPackageReference[];
  rigid_meshes: DeployRigidMeshReference[];
  sources: SourceConfiguration[];
  rigid_objects: RigidMeshConfiguration[];
  microphones: MicrophoneConfiguration[];
  observation_plane: ObservationPlane;
  selected_frequency_hz: number;
  requested_fidelity: Fidelity;
}

function microphoneConfiguration(value: unknown, index: number): MicrophoneConfiguration {
  const microphone = record(value, `microphones[${index}]`);
  if (typeof microphone.id !== "string" || microphone.id.trim().length === 0) {
    throw new Error(`microphones[${index}].id must be a non-empty string.`);
  }
  if (typeof microphone.name !== "string" || microphone.name.trim().length === 0) {
    throw new Error(`microphones[${index}].name must be a non-empty string.`);
  }
  const positionHeightM = finite(microphone.positionHeightM, `microphones[${index}].positionHeightM`);
  if (positionHeightM < 0) throw new Error(`microphones[${index}].positionHeightM cannot be below ground.`);
  return {
    id: microphone.id,
    name: microphone.name.trim(),
    positionX: finite(microphone.positionX, `microphones[${index}].positionX`),
    positionHeightM,
    positionZ: finite(microphone.positionZ, `microphones[${index}].positionZ`),
  };
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
  if (typeof source.name !== "string" || source.name.trim().length === 0) {
    throw new Error(`sources[${index}].name must be a non-empty string.`);
  }
  if (typeof source.packageId !== "string" || source.packageId.trim().length === 0) {
    throw new Error(`sources[${index}].packageId must be a non-empty string.`);
  }
  const polarity = finite(source.polarity, `sources[${index}].polarity`);
  if (polarity !== 1 && polarity !== -1) throw new Error(`sources[${index}].polarity must be 1 or -1.`);
  return {
    id: source.id,
    name: source.name.trim(),
    packageId: source.packageId,
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

function rigidConfiguration(value: unknown, index: number): RigidMeshConfiguration {
  const object = record(value, `rigid_objects[${index}]`);
  for (const field of ["id", "name", "assetId"] as const) {
    if (typeof object[field] !== "string" || object[field].trim().length === 0) {
      throw new Error(`rigid_objects[${index}].${field} must be a non-empty string.`);
    }
  }
  return {
    id: String(object.id),
    name: String(object.name).trim(),
    assetId: String(object.assetId),
    positionX: finite(object.positionX, `rigid_objects[${index}].positionX`),
    positionHeightM: finite(object.positionHeightM, `rigid_objects[${index}].positionHeightM`),
    positionZ: finite(object.positionZ, `rigid_objects[${index}].positionZ`),
    pitchDeg: finite(object.pitchDeg, `rigid_objects[${index}].pitchDeg`),
    yawDeg: finite(object.yawDeg, `rigid_objects[${index}].yawDeg`),
    rollDeg: finite(object.rollDeg, `rigid_objects[${index}].rollDeg`),
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
  if (!Array.isArray(project.packages) || project.packages.length === 0) {
    throw new Error("A project must contain at least one speaker package.");
  }
  const packages = project.packages.map((value, index): DeployPackageReference => {
    const packageReference = record(value, `packages[${index}]`);
    if (typeof packageReference.id !== "string" || packageReference.id.trim().length === 0) {
      throw new Error(`packages[${index}].id must be a non-empty string.`);
    }
    if (typeof packageReference.name !== "string" || packageReference.name.trim().length === 0) {
      throw new Error(`packages[${index}].name must be a non-empty string.`);
    }
    if (packageReference.source_file !== null && typeof packageReference.source_file !== "string") {
      throw new Error(`packages[${index}].source_file must be a string or null.`);
    }
    return {
      id: packageReference.id,
      name: packageReference.name.trim(),
      source_file: packageReference.source_file,
    };
  });
  if (new Set(packages.map((item) => item.id)).size !== packages.length) {
    throw new Error("Every project package must have a unique id.");
  }
  if (!Array.isArray(project.rigid_meshes)) throw new Error("rigid_meshes must be an array.");
  const rigidMeshes = project.rigid_meshes.map((value, index): DeployRigidMeshReference => {
    const reference = record(value, `rigid_meshes[${index}]`);
    if (typeof reference.id !== "string" || reference.id.trim().length === 0) throw new Error(`rigid_meshes[${index}].id must be a non-empty string.`);
    if (typeof reference.name !== "string" || reference.name.trim().length === 0) throw new Error(`rigid_meshes[${index}].name must be a non-empty string.`);
    if (reference.source_file !== null && typeof reference.source_file !== "string") throw new Error(`rigid_meshes[${index}].source_file must be a string or null.`);
    return {
      id: reference.id,
      name: reference.name.trim(),
      source_file: reference.source_file,
      scale_to_meters: positive(reference.scale_to_meters, `rigid_meshes[${index}].scale_to_meters`),
    };
  });
  if (new Set(rigidMeshes.map((item) => item.id)).size !== rigidMeshes.length) throw new Error("Every rigid mesh asset must have a unique id.");
  const sourcesValue = project.sources;
  if (!Array.isArray(sourcesValue) || sourcesValue.length === 0) throw new Error("A project must contain at least one source.");
  const sources = sourcesValue.map(sourceConfiguration);
  if (new Set(sources.map((source) => source.id)).size !== sources.length) {
    throw new Error("Every project source must have a unique id.");
  }
  const packageIds = new Set(packages.map((item) => item.id));
  if (sources.some((source) => !packageIds.has(source.packageId))) {
    throw new Error("Every project source must reference an imported package.");
  }
  if (!Array.isArray(project.rigid_objects)) throw new Error("rigid_objects must be an array.");
  const rigidObjects = project.rigid_objects.map(rigidConfiguration);
  const rigidAssetIds = new Set(rigidMeshes.map((item) => item.id));
  if (rigidObjects.some((object) => !rigidAssetIds.has(object.assetId))) throw new Error("Every rigid object must reference an imported rigid mesh.");
  if (new Set(rigidObjects.map((object) => object.id)).size !== rigidObjects.length) throw new Error("Every rigid object must have a unique id.");
  const microphonesValue = project.microphones;
  if (!Array.isArray(microphonesValue)) throw new Error("microphones must be an array.");
  const microphones = microphonesValue.map(microphoneConfiguration);
  if (new Set(microphones.map((microphone) => microphone.id)).size !== microphones.length) {
    throw new Error("Every project microphone must have a unique id.");
  }
  const objectIds = new Set(sources.map((source) => source.id));
  for (const object of rigidObjects) {
    if (objectIds.has(object.id) || object.id === "audience-plane") throw new Error("Project scene-object ids must be unique.");
    objectIds.add(object.id);
  }
  if (microphones.some((microphone) => objectIds.has(microphone.id) || microphone.id === "audience-plane")) {
    throw new Error("Project scene-object ids must be unique.");
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
    packages,
    rigid_meshes: rigidMeshes,
    sources,
    rigid_objects: rigidObjects,
    microphones,
    observation_plane: observationPlane(project.observation_plane),
    selected_frequency_hz: frequencyHz,
    requested_fidelity: requestedFidelity,
  };
}

export function createDeployProject(
  name: string,
  packages: LoadedSpeakerPackage[],
  rigidMeshes: RigidMeshAsset[],
  sources: SourceConfiguration[],
  rigidObjects: RigidMeshConfiguration[],
  microphones: MicrophoneConfiguration[],
  observation: ObservationPlane,
  selectedFrequencyHz: number,
  requestedFidelity: Fidelity,
): DeployProject {
  return {
    schema: DEPLOY_PROJECT_SCHEMA,
    schema_version: DEPLOY_PROJECT_SCHEMA_VERSION,
    name,
    packages: packages.map((pkg) => ({ id: pkg.id, name: pkg.manifest.name, source_file: pkg.sourcePath })),
    rigid_meshes: rigidMeshes.map((asset) => ({
      id: asset.id,
      name: asset.name,
      source_file: asset.sourcePath,
      scale_to_meters: asset.scaleToMeters,
    })),
    sources,
    rigid_objects: rigidObjects,
    microphones,
    observation_plane: observation,
    selected_frequency_hz: selectedFrequencyHz,
    requested_fidelity: requestedFidelity,
  };
}

export function serializeDeployProject(project: DeployProject): string {
  return `${JSON.stringify(project, null, 2)}\n`;
}
