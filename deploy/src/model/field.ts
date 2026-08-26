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

export function buildSourceInstance(config: SourceConfiguration): SpeakerInstance {
  return {
    id: "subwoofer-1",
    position: [config.positionX, config.positionHeightM, config.positionZ],
    pitchDeg: 0,
    yawDeg: config.yawDeg,
  };
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
  source: SpeakerInstance,
  config: SourceConfiguration,
  observation: ObservationPlane,
  frequencyIndex: number,
  lookup: PatternLookup,
): FieldFrame {
  const pointCount = observation.columns * observation.rows;
  const values = new Float32Array(pointCount);
  const frequency = pkg.frequenciesHz[frequencyIndex];
  const wavenumber = (2 * Math.PI * frequency) / pkg.manifest.medium.sound_speed_m_per_s;
  const level = Math.pow(10, config.levelDb / 20) * config.polarity;
  const drivePhase = 2 * Math.PI * frequency * config.delayMs / 1000;
  const driveReal = level * Math.cos(drivePhase);
  const driveImag = level * Math.sin(drivePhase);
  const inverseRotation = new Quaternion()
    .setFromEuler(new Euler(MathUtils.degToRad(source.pitchDeg), MathUtils.degToRad(source.yawDeg), 0, "YXZ"))
    .invert();
  const direction = new Vector3();
  const sorted: number[] = [];
  let sum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let clippedNearFieldPoints = 0;

  for (let row = 0; row < observation.rows; row += 1) {
    const worldZ = observation.nearM + (row / Math.max(1, observation.rows - 1)) * observation.depthM;
    for (let column = 0; column < observation.columns; column += 1) {
      const worldX = -observation.widthM / 2 + (column / Math.max(1, observation.columns - 1)) * observation.widthM;
      const worldY = observation.heightM;
      let totalReal = 0;
      let totalImag = 0;
      direction.set(
        worldX - source.position[0],
        worldY - source.position[1],
        worldZ - source.position[2],
      );
      const distance = Math.max(0.02, direction.length());
      direction.multiplyScalar(1 / distance).applyQuaternion(inverseRotation);
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
      totalReal = fieldReal * driveReal - fieldImag * driveImag;
      totalImag = fieldReal * driveImag + fieldImag * driveReal;
      const magnitude = Math.max(Number.MIN_VALUE, Math.hypot(totalReal, totalImag));
      const spl = 20 * Math.log10(magnitude / PRESSURE_REFERENCE_PA);
      const index = row * observation.columns + column;
      values[index] = spl;
      sum += spl;
      minimum = Math.min(minimum, spl);
      maximum = Math.max(maximum, spl);
      sorted.push(spl);
    }
  }
  sorted.sort((a, b) => a - b);
  const percentile10 = sorted[Math.floor((sorted.length - 1) * 0.1)];
  const percentile90 = sorted[Math.floor((sorted.length - 1) * 0.9)];
  return {
    splDb: values,
    columns: observation.columns,
    rows: observation.rows,
    minimumDb: minimum,
    maximumDb: maximum,
    averageDb: sum / pointCount,
    spreadDb: percentile90 - percentile10,
    clippedNearFieldPoints,
  };
}
