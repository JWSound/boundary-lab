import assert from "node:assert/strict";
import { applyChannelProcessing, createDefaultChannel, DEFAULT_CHANNEL_ID } from "../src/model/channels";
import { parseDeployProject } from "../src/io/deployProject";
import type { SourceConfiguration } from "../src/model/types";

const source: SourceConfiguration = {
  id: "speaker-1",
  name: "Speaker 1",
  packageId: "package-1",
  positionX: 0,
  positionHeightM: 0,
  positionZ: 0,
  pitchDeg: 0,
  yawDeg: 0,
  rollDeg: 0,
  channelId: DEFAULT_CHANNEL_ID,
  levelDb: -3,
  delayMs: 1.25,
  polarity: -1,
  equalizer: { filters: [] },
};
const channel = {
  ...createDefaultChannel(),
  levelDb: 6,
  delayMs: 2.5,
  polarity: -1 as const,
};
const [effective] = applyChannelProcessing([source], [channel]);
assert.equal(effective.levelDb, 3);
assert.equal(effective.delayMs, 3.75);
assert.equal(effective.polarity, 1);
assert.equal(effective.muted, false);

const [muted] = applyChannelProcessing([source], [{ ...channel, muted: true }]);
assert.equal(muted.muted, true);
assert.equal(source.muted, undefined);

const { channelId: _legacyChannelId, equalizer: _legacyEqualizer, ...legacySource } = source;
const migrated = parseDeployProject(JSON.stringify({
  schema: "boundary-lab-deploy-project",
  schema_version: 6,
  name: "Legacy channel migration",
  packages: [{ id: "package-1", name: "Package 1", source_file: null }],
  rigid_meshes: [],
  sources: [legacySource],
  rigid_objects: [],
  microphones: [],
  observation_plane: {
    widthM: 10, depthM: 10, centerXM: 0, nearM: 0, heightM: 1,
    pitchDeg: 0, yawDeg: 0, rollDeg: 0, columns: 10, rows: 10,
    heatmapMinimumDb: 50, heatmapMaximumDb: 120, heatmapBandingDb: 3,
    displayMode: "spl", pressureScalePa: 10, phaseAnimationSpeedHz: 1,
  },
  selected_frequency_hz: 80,
  requested_fidelity: "pattern",
}));
assert.equal(migrated.schema_version, 7);
assert.equal(migrated.channels[0].id, DEFAULT_CHANNEL_ID);
assert.equal(migrated.sources[0].channelId, DEFAULT_CHANNEL_ID);
assert.deepEqual(migrated.sources[0].equalizer, { filters: [] });

console.log("channel processing regression tests passed");
