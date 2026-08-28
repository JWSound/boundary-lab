import { Euler, MathUtils, Quaternion, Vector3 } from "three";
import type {
  FieldFrame,
  LoadedSpeakerPackage,
  MicrophoneConfiguration,
  MicrophoneResponseSet,
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

export function computeMicrophonePatternResponses(
  pkg: LoadedSpeakerPackage,
  sources: SpeakerInstance[],
  configs: SourceConfiguration[],
  microphones: MicrophoneConfiguration[],
): MicrophoneResponseSet {
  const frequencyIndices = Array.from(pkg.frequenciesHz.keys()).sort(
    (left, right) => pkg.frequenciesHz[left] - pkg.frequenciesHz[right],
  );
  const frequenciesHz = Float64Array.from(frequencyIndices, (index) => pkg.frequenciesHz[index]);
  if (sources.length !== configs.length) throw new Error("Microphone responses require matching source instances and configurations.");
  const sourceDirectionData = microphones.map((microphone) => sources.map((source) => {
    const direction = new Vector3(
      microphone.positionX - source.position[0],
      microphone.positionHeightM - source.position[1],
      microphone.positionZ - source.position[2],
    );
    const distance = Math.max(0.02, direction.length());
    direction.multiplyScalar(1 / distance).applyQuaternion(new Quaternion().setFromEuler(new Euler(
      MathUtils.degToRad(source.pitchDeg),
      MathUtils.degToRad(source.yawDeg),
      MathUtils.degToRad(source.rollDeg),
      "YXZ",
    )).invert());
    const packageDirection = new Vector3(direction.x, direction.z, -direction.y);
    let directionIndex = 0;
    let bestDot = -Infinity;
    for (let index = 0; index < pkg.pressureShape[2]; index += 1) {
      const offset = index * 3;
      const dot = packageDirection.x * pkg.directionsPackage[offset]
        + packageDirection.y * pkg.directionsPackage[offset + 1]
        + packageDirection.z * pkg.directionsPackage[offset + 2];
      if (dot > bestDot) {
        bestDot = dot;
        directionIndex = index;
      }
    }
    return { distance, directionIndex, referenceRadius: pkg.radiiM[directionIndex] };
  }));
  const excitationCount = pkg.pressureShape[1];
  const directionCount = pkg.pressureShape[2];
  return {
    frequenciesHz,
    traces: microphones.map((microphone, microphoneIndex) => {
      const splDb = new Float32Array(frequencyIndices.length);
      let clippedNearFieldSamples = 0;
      frequencyIndices.forEach((frequencyIndex, sortedIndex) => {
        const frequency = pkg.frequenciesHz[frequencyIndex];
        const wavenumber = (2 * Math.PI * frequency) / pkg.manifest.medium.sound_speed_m_per_s;
        let totalReal = 0;
        let totalImag = 0;
        sources.forEach((_source, sourceIndex) => {
          const sample = sourceDirectionData[microphoneIndex][sourceIndex];
          const config = configs[sourceIndex];
          const pressureIndex = (frequencyIndex * excitationCount) * directionCount + sample.directionIndex;
          const sampleReal = pkg.pressure.real[pressureIndex];
          const sampleImag = pkg.pressure.imag[pressureIndex];
          const scale = sample.referenceRadius / sample.distance;
          const propagationPhase = wavenumber * (sample.distance - sample.referenceRadius);
          const propagationReal = Math.cos(propagationPhase) * scale;
          const propagationImag = Math.sin(propagationPhase) * scale;
          const fieldReal = sampleReal * propagationReal - sampleImag * propagationImag;
          const fieldImag = sampleReal * propagationImag + sampleImag * propagationReal;
          const driveMagnitude = Math.pow(10, config.levelDb / 20) * config.polarity;
          const drivePhase = 2 * Math.PI * frequency * config.delayMs / 1000;
          const driveReal = driveMagnitude * Math.cos(drivePhase);
          const driveImag = driveMagnitude * Math.sin(drivePhase);
          totalReal += fieldReal * driveReal - fieldImag * driveImag;
          totalImag += fieldReal * driveImag + fieldImag * driveReal;
          if (sample.distance < sample.referenceRadius) clippedNearFieldSamples += 1;
        });
        splDb[sortedIndex] = 20 * Math.log10(Math.max(Number.MIN_VALUE, Math.hypot(totalReal, totalImag)) / PRESSURE_REFERENCE_PA);
      });
      return { microphoneId: microphone.id, microphoneName: microphone.name, splDb, clippedNearFieldSamples };
    }),
  };
}

export function computeMixedMicrophonePatternResponses(
  packages: ReadonlyMap<string, LoadedSpeakerPackage>,
  sources: SpeakerInstance[],
  configs: SourceConfiguration[],
  microphones: MicrophoneConfiguration[],
): MicrophoneResponseSet {
  if (sources.length !== configs.length) throw new Error("Microphone responses require matching source instances and configurations.");
  const sourcePackages = configs.map((config) => packages.get(config.packageId)).filter(Boolean) as LoadedSpeakerPackage[];
  const commonMinimum = Math.max(...sourcePackages.map((pkg) => Math.min(...pkg.frequenciesHz)));
  const commonMaximum = Math.min(...sourcePackages.map((pkg) => Math.max(...pkg.frequenciesHz)));
  const commonFrequencies = [...new Set(
    sourcePackages.flatMap((pkg) => Array.from(pkg.frequenciesHz)),
  )].filter((frequency) => frequency >= commonMinimum && frequency <= commonMaximum).sort((left, right) => left - right);
  if (commonFrequencies.length === 0) throw new Error("Active speaker packages do not share an overlapping frequency range.");
  const frequenciesHz = Float64Array.from(commonFrequencies);
  const sourceData = microphones.map((microphone) => sources.map((source, sourceIndex) => {
    const config = configs[sourceIndex];
    const pkg = packages.get(config.packageId);
    if (!pkg) throw new Error(`Source ${config.name} references a package that is not loaded.`);
    const direction = new Vector3(
      microphone.positionX - source.position[0],
      microphone.positionHeightM - source.position[1],
      microphone.positionZ - source.position[2],
    );
    const distance = Math.max(0.02, direction.length());
    direction.multiplyScalar(1 / distance).applyQuaternion(new Quaternion().setFromEuler(new Euler(
      MathUtils.degToRad(source.pitchDeg),
      MathUtils.degToRad(source.yawDeg),
      MathUtils.degToRad(source.rollDeg),
      "YXZ",
    )).invert());
    const packageDirection = new Vector3(direction.x, direction.z, -direction.y);
    let directionIndex = 0;
    let bestDot = -Infinity;
    for (let index = 0; index < pkg.pressureShape[2]; index += 1) {
      const offset = index * 3;
      const dot = packageDirection.x * pkg.directionsPackage[offset]
        + packageDirection.y * pkg.directionsPackage[offset + 1]
        + packageDirection.z * pkg.directionsPackage[offset + 2];
      if (dot > bestDot) { bestDot = dot; directionIndex = index; }
    }
    return { pkg, config, distance, directionIndex, referenceRadius: pkg.radiiM[directionIndex] };
  }));
  return {
    frequenciesHz,
    traces: microphones.map((microphone, microphoneIndex) => {
      const splDb = new Float32Array(frequenciesHz.length);
      let clippedNearFieldSamples = 0;
      frequenciesHz.forEach((frequency, frequencyOutputIndex) => {
        let totalReal = 0;
        let totalImag = 0;
        sourceData[microphoneIndex].forEach((sample) => {
          const [lower, upper, mix] = frequencyBracket(sample.pkg, frequency);
          const excitationCount = sample.pkg.pressureShape[1];
          const directionCount = sample.pkg.pressureShape[2];
          const lowerOffset = (lower * excitationCount) * directionCount + sample.directionIndex;
          const upperOffset = (upper * excitationCount) * directionCount + sample.directionIndex;
          const sampleReal = sample.pkg.pressure.real[lowerOffset]
            + (sample.pkg.pressure.real[upperOffset] - sample.pkg.pressure.real[lowerOffset]) * mix;
          const sampleImag = sample.pkg.pressure.imag[lowerOffset]
            + (sample.pkg.pressure.imag[upperOffset] - sample.pkg.pressure.imag[lowerOffset]) * mix;
          const scale = sample.referenceRadius / sample.distance;
          const wavenumber = (2 * Math.PI * frequency) / sample.pkg.manifest.medium.sound_speed_m_per_s;
          const propagationPhase = wavenumber * (sample.distance - sample.referenceRadius);
          const propagationReal = Math.cos(propagationPhase) * scale;
          const propagationImag = Math.sin(propagationPhase) * scale;
          const fieldReal = sampleReal * propagationReal - sampleImag * propagationImag;
          const fieldImag = sampleReal * propagationImag + sampleImag * propagationReal;
          const driveMagnitude = Math.pow(10, sample.config.levelDb / 20) * sample.config.polarity;
          const drivePhase = 2 * Math.PI * frequency * sample.config.delayMs / 1000;
          const driveReal = driveMagnitude * Math.cos(drivePhase);
          const driveImag = driveMagnitude * Math.sin(drivePhase);
          totalReal += fieldReal * driveReal - fieldImag * driveImag;
          totalImag += fieldReal * driveImag + fieldImag * driveReal;
          if (sample.distance < sample.referenceRadius) clippedNearFieldSamples += 1;
        });
        splDb[frequencyOutputIndex] = 20 * Math.log10(Math.max(Number.MIN_VALUE, Math.hypot(totalReal, totalImag)) / PRESSURE_REFERENCE_PA);
      });
      return { microphoneId: microphone.id, microphoneName: microphone.name, splDb, clippedNearFieldSamples };
    }),
  };
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
    packageId: config.packageId,
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

function frequencyBracket(pkg: LoadedSpeakerPackage, frequencyHz: number): [number, number, number] {
  const ordered = Array.from(pkg.frequenciesHz.keys()).sort((a, b) => pkg.frequenciesHz[a] - pkg.frequenciesHz[b]);
  if (frequencyHz <= pkg.frequenciesHz[ordered[0]]) return [ordered[0], ordered[0], 0];
  const last = ordered.at(-1)!;
  if (frequencyHz >= pkg.frequenciesHz[last]) return [last, last, 0];
  for (let index = 1; index < ordered.length; index += 1) {
    const upper = ordered[index];
    if (pkg.frequenciesHz[upper] < frequencyHz) continue;
    const lower = ordered[index - 1];
    const span = pkg.frequenciesHz[upper] - pkg.frequenciesHz[lower];
    return [lower, upper, span > 0 ? (frequencyHz - pkg.frequenciesHz[lower]) / span : 0];
  }
  return [last, last, 0];
}

export function buildInterpolatedPatternLookup(pkg: LoadedSpeakerPackage, frequencyHz: number): PatternLookup {
  const [lowerIndex, upperIndex, mix] = frequencyBracket(pkg, frequencyHz);
  const lower = buildPatternLookup(pkg, lowerIndex);
  if (lowerIndex === upperIndex) return lower;
  const upper = buildPatternLookup(pkg, upperIndex);
  for (let index = 0; index < lower.real.length; index += 1) {
    lower.real[index] += (upper.real[index] - lower.real[index]) * mix;
    lower.imag[index] += (upper.imag[index] - lower.imag[index]) * mix;
    lower.radius[index] += (upper.radius[index] - lower.radius[index]) * mix;
  }
  return lower;
}

export function buildPackagePatternLookups(
  packages: Iterable<LoadedSpeakerPackage>,
  frequencyHz: number,
): ReadonlyMap<string, PatternLookup> {
  const lookups = new Map<string, PatternLookup>();
  for (const pkg of packages) {
    if (!lookups.has(pkg.id)) lookups.set(pkg.id, buildInterpolatedPatternLookup(pkg, frequencyHz));
  }
  return lookups;
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

export function computeMixedFieldFrame(
  packages: ReadonlyMap<string, LoadedSpeakerPackage>,
  lookups: ReadonlyMap<string, PatternLookup>,
  sources: SpeakerInstance[],
  configs: SourceConfiguration[],
  observation: ObservationPlane,
  frequencyHz: number,
): FieldFrame {
  if (sources.length !== configs.length || sources.length === 0) {
    throw new Error("Pattern field requires matching non-empty source instances and configurations.");
  }
  const pointCount = observation.columns * observation.rows;
  const values = new Float32Array(pointCount);
  const validMask = new Uint8Array(pointCount);
  const validValues = new Float32Array(pointCount);
  const sourceData = sources.map((source, index) => {
    const config = configs[index];
    const pkg = packages.get(config.packageId);
    if (!pkg) throw new Error(`Source ${config.name} references a package that is not loaded.`);
    const lookup = lookups.get(config.packageId);
    if (!lookup) throw new Error(`Source ${config.name} has no pattern lookup for its package.`);
    const level = Math.pow(10, config.levelDb / 20) * config.polarity;
    const drivePhase = 2 * Math.PI * frequencyHz * config.delayMs / 1000;
    return {
      source,
      lookup,
      wavenumber: (2 * Math.PI * frequencyHz) / pkg.manifest.medium.sound_speed_m_per_s,
      driveReal: level * Math.cos(drivePhase),
      driveImag: level * Math.sin(drivePhase),
      inverseRotation: new Quaternion().setFromEuler(new Euler(
        MathUtils.degToRad(source.pitchDeg),
        MathUtils.degToRad(source.yawDeg),
        MathUtils.degToRad(source.rollDeg),
        "YXZ",
      )).invert(),
    };
  });
  const direction = new Vector3();
  const planePoint = new Vector3();
  const planeRotation = new Quaternion().setFromEuler(new Euler(
    MathUtils.degToRad(observation.pitchDeg),
    MathUtils.degToRad(observation.yawDeg),
    MathUtils.degToRad(observation.rollDeg),
    "YXZ",
  ));
  const planeCenterZ = observation.nearM + observation.depthM / 2;
  let validCount = 0;
  let sum = 0;
  let minimum = Infinity;
  let maximum = -Infinity;
  let clippedNearFieldPoints = 0;
  for (let row = 0; row < observation.rows; row += 1) {
    const localZ = -observation.depthM / 2 + (row / Math.max(1, observation.rows - 1)) * observation.depthM;
    for (let column = 0; column < observation.columns; column += 1) {
      const localX = -observation.widthM / 2 + (column / Math.max(1, observation.columns - 1)) * observation.widthM;
      planePoint.set(localX, 0, localZ).applyQuaternion(planeRotation);
      const worldX = observation.centerXM + planePoint.x;
      const worldY = observation.heightM + planePoint.y;
      const worldZ = planeCenterZ + planePoint.z;
      const index = row * observation.columns + column;
      if (worldY < 0) continue;
      validMask[index] = 1;
      let totalReal = 0;
      let totalImag = 0;
      for (const datum of sourceData) {
        direction.set(worldX - datum.source.position[0], worldY - datum.source.position[1], worldZ - datum.source.position[2]);
        const distance = Math.max(0.02, direction.length());
        direction.multiplyScalar(1 / distance).applyQuaternion(datum.inverseRotation);
        const [sampleReal, sampleImag, referenceRadius] = lookupPattern(datum.lookup, direction.x, direction.z, -direction.y);
        if (distance < referenceRadius) clippedNearFieldPoints += 1;
        const scale = referenceRadius / distance;
        const propagationPhase = datum.wavenumber * (distance - referenceRadius);
        const propagationReal = Math.cos(propagationPhase) * scale;
        const propagationImag = Math.sin(propagationPhase) * scale;
        const fieldReal = sampleReal * propagationReal - sampleImag * propagationImag;
        const fieldImag = sampleReal * propagationImag + sampleImag * propagationReal;
        totalReal += fieldReal * datum.driveReal - fieldImag * datum.driveImag;
        totalImag += fieldReal * datum.driveImag + fieldImag * datum.driveReal;
      }
      const spl = 20 * Math.log10(Math.max(Number.MIN_VALUE, Math.hypot(totalReal, totalImag)) / PRESSURE_REFERENCE_PA);
      values[index] = spl;
      validValues[validCount++] = spl;
      sum += spl;
      minimum = Math.min(minimum, spl);
      maximum = Math.max(maximum, spl);
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
