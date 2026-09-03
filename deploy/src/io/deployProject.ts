import { createDefaultChannel, DEFAULT_CHANNEL_ID } from "../model/channels";
import type { DeployChannel, EqualizerConfiguration, Fidelity, LoadedSpeakerPackage, MicrophoneConfiguration, ObservationPlane, RigidMeshAsset, RigidMeshConfiguration, SourceConfiguration } from "../model/types";

export const DEPLOY_PROJECT_SCHEMA = "boundary-lab-deploy-project";
export const DEPLOY_PROJECT_SCHEMA_VERSION = 7;

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
  channels: DeployChannel[];
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

function equalizerConfiguration(value: unknown, label: string, legacy: boolean): EqualizerConfiguration {
  if (legacy || value === undefined) return { filters: [] };
  const equalizer = record(value, label);
  if (!Array.isArray(equalizer.filters)) throw new Error(`${label}.filters must be an array.`);
  // The editor is intentionally a placeholder. Reject non-empty banks until the
  // filter evaluator and its schema are implemented together.
  if (equalizer.filters.length > 0) throw new Error(`${label}.filters are not supported by this version.`);
  return { filters: [] };
}

function channelConfiguration(value: unknown, index: number): DeployChannel {
  const channel = record(value, `channels[${index}]`);
  for (const field of ["id", "name", "color"] as const) {
    if (typeof channel[field] !== "string" || channel[field].trim().length === 0) {
      throw new Error(`channels[${index}].${field} must be a non-empty string.`);
    }
  }
  const polarity = finite(channel.polarity, `channels[${index}].polarity`);
  if (polarity !== 1 && polarity !== -1) throw new Error(`channels[${index}].polarity must be 1 or -1.`);
  if (typeof channel.muted !== "boolean") throw new Error(`channels[${index}].muted must be a boolean.`);
  return {
    id: String(channel.id),
    name: String(channel.name).trim(),
    color: String(channel.color),
    levelDb: finite(channel.levelDb, `channels[${index}].levelDb`),
    delayMs: finite(channel.delayMs, `channels[${index}].delayMs`),
    polarity: polarity as 1 | -1,
    muted: channel.muted,
    equalizer: equalizerConfiguration(channel.equalizer, `channels[${index}].equalizer`, false),
  };
}

function sourceConfiguration(value: unknown, index: number, legacy: boolean): SourceConfiguration {
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
    channelId: legacy ? DEFAULT_CHANNEL_ID : String(source.channelId ?? ""),
    levelDb: finite(source.levelDb, `sources[${index}].levelDb`),
    delayMs: finite(source.delayMs, `sources[${index}].delayMs`),
    polarity: polarity as 1 | -1,
    equalizer: equalizerConfiguration(source.equalizer, `sources[${index}].equalizer`, legacy),
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

function observationPlane(value: unknown, legacyVersion = false): ObservationPlane {
  const plane = record(value, "observation_plane");
  const minimumDb = finite(plane.heatmapMinimumDb, "observation_plane.heatmapMinimumDb");
  const maximumDb = finite(plane.heatmapMaximumDb, "observation_plane.heatmapMaximumDb");
  if (maximumDb <= minimumDb) throw new Error("The heatmap maximum must be greater than its minimum.");
  const bandingDb = finite(plane.heatmapBandingDb, "observation_plane.heatmapBandingDb");
  if (bandingDb < 0) throw new Error("observation_plane.heatmapBandingDb cannot be negative.");
  const displayMode = legacyVersion && plane.displayMode === undefined ? "spl" : plane.displayMode;
  if (displayMode !== "spl" && displayMode !== "real_pressure" && displayMode !== "imaginary_pressure") {
    throw new Error("observation_plane.displayMode must be spl, real_pressure, or imaginary_pressure.");
  }
  const pressureScalePa = legacyVersion && plane.pressureScalePa === undefined ? 10 : finite(plane.pressureScalePa, "observation_plane.pressureScalePa");
  if (pressureScalePa < 1 || pressureScalePa > 100) {
    throw new Error("observation_plane.pressureScalePa must be between 1 and 100.");
  }
  const phaseAnimationSpeedHz = legacyVersion && plane.phaseAnimationSpeedHz === undefined ? 1 : finite(plane.phaseAnimationSpeedHz, "observation_plane.phaseAnimationSpeedHz");
  if (phaseAnimationSpeedHz < 0.1 || phaseAnimationSpeedHz > 4) {
    throw new Error("observation_plane.phaseAnimationSpeedHz must be between 0.1 and 4 Hz.");
  }
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
    displayMode,
    pressureScalePa,
    phaseAnimationSpeedHz,
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
  if (version !== 5 && version !== 6 && version !== DEPLOY_PROJECT_SCHEMA_VERSION) {
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
  if (!Array.isArray(sourcesValue)) throw new Error("sources must be an array.");
  const legacyChannels = version < 7;
  if (!legacyChannels && !Array.isArray(project.channels)) throw new Error("channels must be an array.");
  const channels = legacyChannels ? [createDefaultChannel()] : (project.channels as unknown[]).map(channelConfiguration);
  if (channels.length === 0) throw new Error("A project must contain at least one channel.");
  if (new Set(channels.map((channel) => channel.id)).size !== channels.length) throw new Error("Every project channel must have a unique id.");
  const sources = sourcesValue.map((value, index) => sourceConfiguration(value, index, legacyChannels));
  if (new Set(sources.map((source) => source.id)).size !== sources.length) {
    throw new Error("Every project source must have a unique id.");
  }
  const packageIds = new Set(packages.map((item) => item.id));
  if (sources.some((source) => !packageIds.has(source.packageId))) {
    throw new Error("Every project source must reference an imported package.");
  }
  const channelIds = new Set(channels.map((channel) => channel.id));
  if (sources.some((source) => !channelIds.has(source.channelId))) throw new Error("Every project source must reference a channel.");
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
    channels,
    sources,
    rigid_objects: rigidObjects,
    microphones,
    observation_plane: observationPlane(project.observation_plane, version === 5),
    selected_frequency_hz: frequencyHz,
    requested_fidelity: requestedFidelity,
  };
}

export function createDeployProject(
  name: string,
  packages: LoadedSpeakerPackage[],
  rigidMeshes: RigidMeshAsset[],
  channels: DeployChannel[],
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
    channels,
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
