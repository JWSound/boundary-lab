import { Euler, MathUtils, Quaternion, Vector3 } from "three";
import type {
  FieldFrame,
  LoadedSpeakerPackage,
  ObservationPlane,
  PatternLookup,
  SourceConfiguration,
  SpeakerInstance,
} from "./types";

const PRESSURE_REFERENCE_PA = 20e-6;

function selectKth(values: Float32Array, target: number): number {
  let left = 0;
  let right = values.length - 1;
  while (left < right) {
    const pivot = values[(left + right) >> 1];
    let lower = left;
    let upper = right;
    while (lower <= upper) {
      while (values[lower] < pivot) lower += 1;
      while (values[upper] > pivot) upper -= 1;
      if (lower <= upper) {
        const temporary = values[lower];
        values[lower] = values[upper];
        values[upper] = temporary;
        lower += 1;
        upper -= 1;
      }
    }
    if (target <= upper) right = upper;
    else if (target >= lower) left = lower;
    else return values[target];
  }
  return values[target];
}

function percentileSpread(values: Float32Array): number {
  if (values.length === 0) return 0;
  const percentile10 = selectKth(values, Math.floor((values.length - 1) * 0.1));
  const percentile90 = selectKth(values, Math.floor((values.length - 1) * 0.9));
  return percentile90 - percentile10;
}

export function fieldFrameFromSpl(
  samples: ArrayLike<number>,
  columns: number,
  rows: number,
  sampleIndices?: ArrayLike<number>,
): FieldFrame {
  const pointCount = columns * rows;
  const indices = sampleIndices ?? Array.from({ length: pointCount }, (_, index) => index);
  if (samples.length !== indices.length) {
    throw new Error("Level 2 field dimensions do not match the audience-plane samples.");
  }
  const values = new Float32Array(pointCount);
  const validMask = new Uint8Array(pointCount);
  const validValues = new Float32Array(samples.length);
  let sum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let sampleIndex = 0; sampleIndex < samples.length; sampleIndex += 1) {
    const value = Number(samples[sampleIndex]);
    const gridIndex = Number(indices[sampleIndex]);
    if (!Number.isFinite(value)) throw new Error("Level 2 field contains a non-finite SPL value.");
    if (!Number.isInteger(gridIndex) || gridIndex < 0 || gridIndex >= pointCount || validMask[gridIndex]) {
      throw new Error("Level 2 field contains an invalid audience-plane sample index.");
    }
    values[gridIndex] = value;
    validMask[gridIndex] = 1;
    validValues[sampleIndex] = value;
    sum += value;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  if (validValues.length === 0) minimum = maximum = 0;
  return {
    splDb: values,
    validMask,
    columns,
    rows,
    minimumDb: minimum,
    maximumDb: maximum,
    averageDb: validValues.length ? sum / validValues.length : 0,
    spreadDb: percentileSpread(validValues),
    clippedNearFieldPoints: 0,
  };
}

export function buildSourceInstance(config: SourceConfiguration): SpeakerInstance {
  return {
    id: config.id,
    position: [config.positionX, config.positionHeightM, config.positionZ],
    pitchDeg: config.pitchDeg,
    yawDeg: config.yawDeg,
    rollDeg: config.rollDeg,
  };
}

export const SOURCE_SURFACE_PADDING_M = 0.01;
export const SOURCE_GROUND_CLEARANCE_M = 0;

export function minimumSourceHeightM(pkg: LoadedSpeakerPackage, pitchDeg = 0, rollDeg = 0): number {
  const rotation = new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(pitchDeg),
    0,
    MathUtils.degToRad(rollDeg),
    "YXZ",
  ));
  if (!pkg.mesh) {
    let minimumY = Infinity;
    for (const x of [-pkg.boundsM[0] / 2, pkg.boundsM[0] / 2]) {
      for (const y of [-pkg.boundsM[2] / 2, pkg.boundsM[2] / 2]) {
        for (const z of [-pkg.boundsM[1] / 2, pkg.boundsM[1] / 2]) {
          minimumY = Math.min(minimumY, new Vector3(x, y, z).applyQuaternion(rotation).y);
        }
      }
    }
    return SOURCE_GROUND_CLEARANCE_M - minimumY;
  }
  let minimumSceneY = Infinity;
  for (let index = 0; index < pkg.mesh.positions.length; index += 3) {
    const point = new Vector3(
      pkg.mesh.positions[index],
      -pkg.mesh.positions[index + 2],
      pkg.mesh.positions[index + 1],
    ).applyQuaternion(rotation);
    minimumSceneY = Math.min(minimumSceneY, point.y);
  }
  return SOURCE_GROUND_CLEARANCE_M - minimumSceneY;
}

export function nearestFrequencyIndex(pkg: LoadedSpeakerPackage, targetHz: number): number {
  let nearest = 0;
  let distance = Infinity;
  for (let index = 0; index < pkg.frequenciesHz.length; index += 1) {
    const current = Math.abs(pkg.frequenciesHz[index] - targetHz);
    if (current < distance) {
      nearest = index;
      distance = current;
    }
  }
  return nearest;
}

export function buildPatternLookup(
  pkg: LoadedSpeakerPackage,
  frequencyIndex: number,
  excitationIndex = 0,
): PatternLookup {
  const azimuthBins = 72;
  const elevationBins = 37;
  const real = new Float32Array(azimuthBins * elevationBins);
  const imag = new Float32Array(real.length);
  const radius = new Float32Array(real.length);
  const directionCount = pkg.pressureShape[2];
  const excitationCount = pkg.pressureShape[1];
  const pressureOffset = (frequencyIndex * excitationCount + excitationIndex) * directionCount;

  for (let elevationIndex = 0; elevationIndex < elevationBins; elevationIndex += 1) {
    const elevation = -Math.PI / 2 + (Math.PI * elevationIndex) / (elevationBins - 1);
    const elevationCos = Math.cos(elevation);
    const targetZ = Math.sin(elevation);
    for (let azimuthIndex = 0; azimuthIndex < azimuthBins; azimuthIndex += 1) {
      const azimuth = -Math.PI + (2 * Math.PI * azimuthIndex) / azimuthBins;
      const targetX = elevationCos * Math.sin(azimuth);
      const targetY = elevationCos * Math.cos(azimuth);
      let bestDirection = 0;
      let bestDot = -Infinity;
      for (let directionIndex = 0; directionIndex < directionCount; directionIndex += 1) {
        const offset = directionIndex * 3;
        const dot =
          targetX * pkg.directionsPackage[offset] +
          targetY * pkg.directionsPackage[offset + 1] +
          targetZ * pkg.directionsPackage[offset + 2];
        if (dot > bestDot) {
          bestDot = dot;
          bestDirection = directionIndex;
        }
      }
      const lookupOffset = elevationIndex * azimuthBins + azimuthIndex;
      real[lookupOffset] = pkg.pressure.real[pressureOffset + bestDirection];
      imag[lookupOffset] = pkg.pressure.imag[pressureOffset + bestDirection];
      radius[lookupOffset] = pkg.radiiM[bestDirection];
    }
  }
  return { azimuthBins, elevationBins, real, imag, radius };
}

function lookupPattern(lookup: PatternLookup, x: number, y: number, z: number): [number, number, number] {
  const elevation = Math.asin(Math.max(-1, Math.min(1, z)));
  const azimuth = Math.atan2(x, y);
  const azimuthIndex = Math.round(((azimuth + Math.PI) / (2 * Math.PI)) * lookup.azimuthBins) % lookup.azimuthBins;
  const elevationIndex = Math.max(
    0,
    Math.min(lookup.elevationBins - 1, Math.round(((elevation + Math.PI / 2) / Math.PI) * (lookup.elevationBins - 1))),
  );
  const index = elevationIndex * lookup.azimuthBins + azimuthIndex;
  return [lookup.real[index], lookup.imag[index], lookup.radius[index]];
}

export function computeFieldFrame(
  pkg: LoadedSpeakerPackage,
  sources: SpeakerInstance[],
  configs: SourceConfiguration[],
  observation: ObservationPlane,
  frequencyIndex: number,
  lookup: PatternLookup,
): FieldFrame {
  const pointCount = observation.columns * observation.rows;
  const values = new Float32Array(pointCount);
  const frequency = pkg.frequenciesHz[frequencyIndex];
  const wavenumber = (2 * Math.PI * frequency) / pkg.manifest.medium.sound_speed_m_per_s;
  if (sources.length !== configs.length || sources.length === 0) {
    throw new Error("Pattern field requires matching non-empty source instances and configurations.");
  }
  const sourceData = sources.map((source, index) => {
    const config = configs[index];
    const level = Math.pow(10, config.levelDb / 20) * config.polarity;
    const drivePhase = 2 * Math.PI * frequency * config.delayMs / 1000;
    return {
      source,
      driveReal: level * Math.cos(drivePhase),
      driveImag: level * Math.sin(drivePhase),
      inverseRotation: new Quaternion()
        .setFromEuler(new Euler(
          MathUtils.degToRad(source.pitchDeg),
          MathUtils.degToRad(source.yawDeg),
          MathUtils.degToRad(source.rollDeg),
          "YXZ",
        ))
        .invert(),
    };
  });
  const direction = new Vector3();
  const planePoint = new Vector3();
  const validMask = new Uint8Array(pointCount);
  const validValues = new Float32Array(pointCount);
  let validCount = 0;
  let sum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let clippedNearFieldPoints = 0;
  const planeRotation = new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(observation.pitchDeg),
    MathUtils.degToRad(observation.yawDeg),
    MathUtils.degToRad(observation.rollDeg),
    "YXZ",
  ));
  const planeCenterZ = observation.nearM + observation.depthM / 2;

  for (let row = 0; row < observation.rows; row += 1) {
    const localZ = -observation.depthM / 2 + (row / Math.max(1, observation.rows - 1)) * observation.depthM;
    for (let column = 0; column < observation.columns; column += 1) {
      const localX = -observation.widthM / 2 + (column / Math.max(1, observation.columns - 1)) * observation.widthM;
      planePoint.set(localX, 0, localZ).applyQuaternion(planeRotation);
      const worldX = observation.centerXM + planePoint.x;
      const worldY = observation.heightM + planePoint.y;
      const worldZ = planeCenterZ + planePoint.z;
      const index = row * observation.columns + column;
      if (worldY < 0) {
        values[index] = 0;
        continue;
      }
      validMask[index] = 1;
      let totalReal = 0;
      let totalImag = 0;
      for (const sourceDatum of sourceData) {
        direction.set(
          worldX - sourceDatum.source.position[0],
          worldY - sourceDatum.source.position[1],
          worldZ - sourceDatum.source.position[2],
        );
        const distance = Math.max(0.02, direction.length());
        direction.multiplyScalar(1 / distance).applyQuaternion(sourceDatum.inverseRotation);
        // Scene X/Y/Z -> package X/Y/Z: X right, package Y forward, package Z opposite scene up.
        const [sampleReal, sampleImag, referenceRadius] = lookupPattern(
          lookup,
          direction.x,
          direction.z,
          -direction.y,
        );
        if (distance < referenceRadius) clippedNearFieldPoints += 1;
        const scale = referenceRadius / distance;
        const propagationPhase = wavenumber * (distance - referenceRadius);
        const propagationReal = Math.cos(propagationPhase) * scale;
        const propagationImag = Math.sin(propagationPhase) * scale;
        const fieldReal = sampleReal * propagationReal - sampleImag * propagationImag;
        const fieldImag = sampleReal * propagationImag + sampleImag * propagationReal;
        totalReal += fieldReal * sourceDatum.driveReal - fieldImag * sourceDatum.driveImag;
        totalImag += fieldReal * sourceDatum.driveImag + fieldImag * sourceDatum.driveReal;
      }
      const magnitude = Math.max(Number.MIN_VALUE, Math.hypot(totalReal, totalImag));
      const spl = 20 * Math.log10(magnitude / PRESSURE_REFERENCE_PA);
      values[index] = spl;
      sum += spl;
      minimum = Math.min(minimum, spl);
      maximum = Math.max(maximum, spl);
      validValues[validCount] = spl;
      validCount += 1;
    }
  }
  if (validCount === 0) minimum = maximum = 0;
  const populatedValues = validCount === validValues.length ? validValues : validValues.slice(0, validCount);
  return {
    splDb: values,
    validMask,
    columns: observation.columns,
    rows: observation.rows,
    minimumDb: minimum,
    maximumDb: maximum,
    averageDb: validCount ? sum / validCount : 0,
    spreadDb: percentileSpread(populatedValues),
    clippedNearFieldPoints,
  };
}
