import { strict as assert } from "node:assert";
import { readFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { build } from "esbuild";
import { Euler, Group, Mesh, MeshBasicMaterial, PerspectiveCamera, Quaternion, SphereGeometry, Vector3 } from "three";

const packagePath = resolve(process.argv[2]);
const outputPath = resolve(".tmp/package-reader.mjs");
await mkdir(dirname(outputPath), { recursive: true });
await build({
  entryPoints: [resolve("scripts/package-smoke-entry.ts")],
  outfile: outputPath,
  bundle: true,
  platform: "node",
  format: "esm",
  logLevel: "silent",
});
const {
  loadSpeakerPackage,
  buildSourceInstance,
  buildPatternLookup,
  buildPackagePatternLookups,
  computeFieldFrame,
  computeMixedFieldFrame,
  computeMixedMicrophonePatternResponses,
  nearestFrequencyIndex,
  heatmapColorBoundaries,
  heatmapLegendGradient,
  writeHeatmapColor,
  createDeployProject,
  parseDeployProject,
  serializeDeployProject,
  loadRigidMesh,
  configureAxisOnlyRotation,
  groundParallelDelta,
  groundParallelPosition,
  paddedCornerSnapDelta,
  snapGroundParallelDelta,
  stickyCornerSnapTarget,
  rotationReadout,
  translationReadout,
  SOURCE_SURFACE_PADDING_M,
  cabinetClearanceViolations,
  cabinetLocalBounds,
  constrainCabinetPoses,
  findClearSourcePlacement,
} = await import(`${pathToFileURL(outputPath).href}?${Date.now()}`);
const bytes = await readFile(packagePath);
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const rigidBytes = await readFile(resolve("library/RigidStage_LOD.msh"));
const rigidBuffer = rigidBytes.buffer.slice(rigidBytes.byteOffset, rigidBytes.byteOffset + rigidBytes.byteLength);
const rigidMesh = loadRigidMesh(rigidBuffer, "RigidStage_LOD.msh", resolve("library/RigidStage_LOD.msh"));
assert.deepEqual(rigidMesh.boundsM, [4, 2, 0.5]);
assert.equal(rigidMesh.vertexCount, 8);
assert.equal(rigidMesh.triangleCount, 12);
assert.deepEqual(groundParallelPosition([1, 2, 3], new Vector3(4, 50, 6)), [5, 2, 9]);
assert.deepEqual(groundParallelDelta(new Vector3(4, 50, 6)).toArray(), [4, 0, 6]);
assert.deepEqual(
  snapGroundParallelDelta(new Vector3(4, 50, 6), new Vector3(), []).toArray(),
  [4, 0, 6],
);
assert.deepEqual(
  snapGroundParallelDelta(new Vector3(4, 0.5, 6), new Vector3(), [new Vector3(4, 0.5, 6)]).toArray(),
  [4, 0.5, 6],
);
const snapCamera = new PerspectiveCamera(60, 1, 0.1, 100);
snapCamera.position.set(0, 0, 10);
snapCamera.lookAt(0, 0, 0);
snapCamera.updateMatrixWorld();
const screenSnapTarget = { key: "cabinet:0", position: new Vector3(), objectCenter: new Vector3() };
const snapViewport = { left: 0, top: 0, width: 100, height: 100 };
assert.equal(stickyCornerSnapTarget(62, 50, [screenSnapTarget], snapCamera, snapViewport, null)?.key, "cabinet:0");
assert.equal(stickyCornerSnapTarget(75, 50, [screenSnapTarget], snapCamera, snapViewport, "cabinet:0")?.key, "cabinet:0");
assert.equal(stickyCornerSnapTarget(90, 50, [screenSnapTarget], snapCamera, snapViewport, "cabinet:0"), null);
assert.deepEqual(
  paddedCornerSnapDelta(
    new Vector3(),
    new Vector3(2, 0, 0),
    new Vector3(0.8, 4, 0.8),
    new Vector3(0.8, 0, 0.8),
    { key: "cabinet:1", position: new Vector3(1, 1, 1), objectCenter: new Vector3() },
  ).toArray(),
  [1.01, 1, 1],
);
assert.equal(translationReadout(new Vector3(), new Vector3(0.1234, 0, 0), "X"), "X +0.123 m");
assert.equal(
  rotationReadout(new Quaternion().setFromEuler(new Euler(0, Math.PI / 4, 0, "YXZ")), "Y"),
  "Y 45°",
);
const visibleRotation = new Group();
const freeRotationHandle = new Mesh(new SphereGeometry(1), new MeshBasicMaterial());
freeRotationHandle.name = "E";
visibleRotation.add(freeRotationHandle);
const rotationPicker = new Group();
const freeRotationPicker = new Mesh(new SphereGeometry(1), new MeshBasicMaterial());
freeRotationPicker.name = "XYZE";
rotationPicker.add(freeRotationPicker);
configureAxisOnlyRotation({
  axis: null,
  gizmo: { gizmo: { rotate: visibleRotation }, picker: { rotate: rotationPicker } },
});
assert.equal(freeRotationHandle.material.visible, false);
assert.equal(freeRotationPicker.material.visible, false);
assert.equal(freeRotationPicker.raycast(), undefined);
const speaker = loadSpeakerPackage(buffer, packagePath.split(/[\\/]/).at(-1));

assert.equal(speaker.manifest.schema, "boundary-lab-speaker-package");
assert.equal(speaker.manifest.schema_version, 1);
assert.ok(speaker.frequenciesHz.length > 0);
assert.equal(speaker.pressureShape[0], speaker.frequenciesHz.length);
assert.equal(speaker.pressureShape[2] * 3, speaker.directionsPackage.length);
assert.equal(speaker.pressure.real.length, speaker.pressure.imag.length);
assert.ok(speaker.mesh?.indices.length > 0);
const frequencyIndex = nearestFrequencyIndex(speaker, 80);
const sourceConfig = {
  id: "subwoofer-1",
  name: "S218BP 1",
  packageId: speaker.id,
  positionX: 0,
  positionHeightM: 0.4,
  positionZ: 0,
  pitchDeg: 0,
  yawDeg: 0,
  rollDeg: 0,
  levelDb: -3,
  delayMs: 0,
  polarity: 1,
};
const source = buildSourceInstance(sourceConfig);
const lookup = buildPatternLookup(speaker, frequencyIndex);
const field = computeFieldFrame(
  speaker,
  [source],
  [sourceConfig],
  { widthM: 12, depthM: 12, centerXM: 0, nearM: 2, heightM: 1.2, pitchDeg: 0, yawDeg: 0, rollDeg: 0, columns: 24, rows: 20, heatmapMinimumDb: 50, heatmapMaximumDb: 145, heatmapBandingDb: 0 },
  frequencyIndex,
  lookup,
);
assert.equal(field.splDb.length, 480);
assert.ok(field.splDb.every(Number.isFinite));
assert.ok(Number.isFinite(field.averageDb));
const clippedField = computeFieldFrame(
  speaker,
  [source],
  [sourceConfig],
  { widthM: 12, depthM: 12, centerXM: 0, nearM: 2, heightM: 0, pitchDeg: 30, yawDeg: 0, rollDeg: 0, columns: 12, rows: 12, heatmapMinimumDb: 50, heatmapMaximumDb: 145, heatmapBandingDb: 0 },
  frequencyIndex,
  lookup,
);
const clippedValidCount = clippedField.validMask.reduce((sum, value) => sum + value, 0);
assert.ok(clippedValidCount > 0 && clippedValidCount < clippedField.validMask.length);

const secondSpeaker = loadSpeakerPackage(buffer, `alternate-${packagePath.split(/[\\/]/).at(-1)}`);
const secondSourceConfig = {
  ...sourceConfig,
  id: "subwoofer-2",
  name: "Alternate S218BP 1",
  packageId: secondSpeaker.id,
  positionX: 3,
};
const packageRegistry = new Map([[speaker.id, speaker], [secondSpeaker.id, secondSpeaker]]);
const packageLookups = buildPackagePatternLookups(packageRegistry.values(), speaker.frequenciesHz[frequencyIndex]);
const mixedField = computeMixedFieldFrame(
  packageRegistry,
  packageLookups,
  [source, buildSourceInstance(secondSourceConfig)],
  [sourceConfig, secondSourceConfig],
  { widthM: 12, depthM: 12, centerXM: 0, nearM: 2, heightM: 1.2, pitchDeg: 0, yawDeg: 0, rollDeg: 0, columns: 18, rows: 16, heatmapMinimumDb: 50, heatmapMaximumDb: 145, heatmapBandingDb: 0 },
  speaker.frequenciesHz[frequencyIndex],
);
assert.equal(mixedField.splDb.length, 288);
assert.ok(mixedField.splDb.every(Number.isFinite));
const mixedMicrophones = computeMixedMicrophonePatternResponses(
  packageRegistry,
  [source, buildSourceInstance(secondSourceConfig)],
  [sourceConfig, secondSourceConfig],
  [{ id: "microphone-1", name: "Microphone 1", positionX: 0, positionHeightM: 1.2, positionZ: 6 }],
);
assert.equal(mixedMicrophones.traces.length, 1);
assert.ok(mixedMicrophones.traces[0].splDb.every(Number.isFinite));

const cabinetBounds = cabinetLocalBounds(speaker);
const cabinetWidth = cabinetBounds.maximum[0] - cabinetBounds.minimum[0];
const clearanceSource = { ...source, id: "clearance-source", position: [0, 1, 0] };
const clearanceObstacle = { ...source, id: "clearance-obstacle", position: [3, 1, 0] };
const clearanceRegistry = new Map([[speaker.id, speaker]]);
const sweptPose = constrainCabinetPoses(
  clearanceRegistry,
  [clearanceSource, clearanceObstacle],
  [{ ...clearanceSource, position: [6, 1, 0] }],
);
assert.ok(sweptPose[0].position[0] < clearanceObstacle.position[0]);
assert.deepEqual(cabinetClearanceViolations(clearanceRegistry, [sweptPose[0], clearanceObstacle]), []);
const slidingPose = constrainCabinetPoses(
  clearanceRegistry,
  [clearanceSource, { ...clearanceObstacle, position: [2, 1, 0] }],
  [{ ...clearanceSource, position: [4, 1, 1.5] }],
);
assert.ok(slidingPose[0].position[2] > 1);
assert.deepEqual(cabinetClearanceViolations(
  clearanceRegistry,
  [slidingPose[0], { ...clearanceObstacle, position: [2, 1, 0] }],
), []);

const touchingObstacle = {
  ...clearanceObstacle,
  position: [cabinetWidth + SOURCE_SURFACE_PADDING_M, 1, 0],
};
const rotatedPose = constrainCabinetPoses(
  clearanceRegistry,
  [clearanceSource, touchingObstacle],
  [{ ...clearanceSource, yawDeg: 45 }],
);
assert.ok(Math.abs(rotatedPose[0].yawDeg) < 45);
assert.deepEqual(cabinetClearanceViolations(clearanceRegistry, [rotatedPose[0], touchingObstacle]), []);
const groupSpacing = cabinetWidth + SOURCE_SURFACE_PADDING_M;
const groupLeft = { ...clearanceSource, id: "group-left", position: [-groupSpacing / 2, 1, 0] };
const groupRight = { ...clearanceSource, id: "group-right", position: [groupSpacing / 2, 1, 0] };
const rotatedGroup = constrainCabinetPoses(
  clearanceRegistry,
  [groupLeft, groupRight],
  [
    { ...groupLeft, position: [0, 1, groupSpacing / 2], yawDeg: 90 },
    { ...groupRight, position: [0, 1, -groupSpacing / 2], yawDeg: 90 },
  ],
);
assert.ok(rotatedGroup.every((instance) => Math.abs(instance.yawDeg - 90) < 1e-6));
assert.deepEqual(cabinetClearanceViolations(clearanceRegistry, rotatedGroup), []);
assert.equal(cabinetClearanceViolations(
  clearanceRegistry,
  [clearanceSource, { ...clearanceObstacle, position: [cabinetWidth, 1, 0] }],
).length, 1);
const overlappingObstacle = { ...clearanceObstacle, position: [cabinetWidth * 0.5, 1, 0] };
const recoveringPose = constrainCabinetPoses(
  clearanceRegistry,
  [clearanceSource, overlappingObstacle],
  [{ ...clearanceSource, position: [-cabinetWidth, 1, 0] }],
);
assert.ok(recoveringPose[0].position[0] < clearanceSource.position[0]);

const placedSource = findClearSourcePlacement(
  clearanceRegistry,
  [clearanceSource],
  { ...clearanceObstacle, position: [...clearanceSource.position] },
);
assert.deepEqual(cabinetClearanceViolations(clearanceRegistry, [clearanceSource, placedSource]), []);

assert.deepEqual(heatmapColorBoundaries(50, 145, 0), []);
const fiveDbBoundaries = heatmapColorBoundaries(50, 145, 5);
assert.deepEqual(fiveDbBoundaries.slice(0, 3), [50, 55, 60]);
assert.deepEqual(fiveDbBoundaries.slice(-2), [140, 145]);
const continuousPixels = new Uint8Array(8);
writeHeatmapColor(50, 50, 145, [], continuousPixels, 0);
writeHeatmapColor(145, 50, 145, [], continuousPixels, 4);
assert.deepEqual(Array.from(continuousPixels.slice(0, 3)), [0, 0, 143]);
assert.deepEqual(Array.from(continuousPixels.slice(4, 7)), [143, 0, 0]);
const bandedPixels = new Uint8Array(12);
writeHeatmapColor(60, 50, 145, fiveDbBoundaries, bandedPixels, 0);
writeHeatmapColor(64.9, 50, 145, fiveDbBoundaries, bandedPixels, 4);
writeHeatmapColor(65, 50, 145, fiveDbBoundaries, bandedPixels, 8);
assert.deepEqual(Array.from(bandedPixels.slice(0, 3)), Array.from(bandedPixels.slice(4, 7)));
assert.notDeepEqual(Array.from(bandedPixels.slice(4, 7)), Array.from(bandedPixels.slice(8, 11)));
assert.match(heatmapLegendGradient(50, 145, 5), /5\.263%/);

const projectText = serializeDeployProject(createDeployProject(
  "Smoke Project",
  [speaker],
  [rigidMesh],
  [sourceConfig],
  [{ id: "rigid-1", name: "Stage 1", assetId: rigidMesh.id, positionX: 5, positionHeightM: 0.25, positionZ: 0, pitchDeg: 0, yawDeg: 0, rollDeg: 0 }],
  [{ id: "microphone-1", name: "Microphone 1", positionX: 0, positionHeightM: 1.2, positionZ: 6 }],
  { widthM: 12, depthM: 10, centerXM: 1, nearM: 2, heightM: 1.2, pitchDeg: 0, yawDeg: 5, rollDeg: 0, columns: 24, rows: 20, heatmapMinimumDb: 50, heatmapMaximumDb: 145, heatmapBandingDb: 0 },
  speaker.frequenciesHz[frequencyIndex],
  "boundary",
));
const parsedProject = parseDeployProject(projectText);
assert.equal(parsedProject.name, "Smoke Project");
assert.equal(parsedProject.sources[0].id, sourceConfig.id);
assert.equal(parsedProject.sources[0].packageId, speaker.id);
assert.equal(parsedProject.packages[0].id, speaker.id);
assert.equal(parsedProject.rigid_meshes[0].id, rigidMesh.id);
assert.equal(parsedProject.rigid_objects[0].assetId, rigidMesh.id);
assert.equal(parsedProject.microphones[0].name, "Microphone 1");
assert.equal(parsedProject.observation_plane.columns, 24);
assert.equal(parsedProject.requested_fidelity, "boundary");
const unsupportedProject = JSON.parse(projectText);
unsupportedProject.schema_version = 1;
assert.throws(() => parseDeployProject(JSON.stringify(unsupportedProject)), /Unsupported.*version 1/);
const incompleteProject = JSON.parse(projectText);
delete incompleteProject.observation_plane.heatmapMinimumDb;
assert.throws(() => parseDeployProject(JSON.stringify(incompleteProject)), /heatmapMinimumDb/);
assert.throws(
  () => parseDeployProject(JSON.stringify({ ...JSON.parse(projectText), sources: [sourceConfig, sourceConfig] })),
  /unique id/,
);

console.log(JSON.stringify({
  name: speaker.manifest.name,
  fidelity: speaker.manifest.fidelity,
  frequencies: speaker.frequenciesHz.length,
  directions: speaker.pressureShape[2],
  meshVertices: speaker.mesh.positions.length / 3,
  meshTriangles: speaker.mesh.indices.length / 3,
  boundsM: speaker.boundsM.map((value) => Number(value.toFixed(3))),
  previewFrequencyHz: speaker.frequenciesHz[frequencyIndex],
  previewAverageDb: Number(field.averageDb.toFixed(2)),
}));
