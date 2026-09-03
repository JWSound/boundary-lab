export interface NpyArray {
  shape: number[];
  descr: string;
  fortranOrder: boolean;
  data: ArrayBufferView;
}

function parseHeader(header: string): { descr: string; fortranOrder: boolean; shape: number[] } {
  const descr = /['"]descr['"]\s*:\s*['"]([^'"]+)['"]/.exec(header)?.[1];
  const fortran = /['"]fortran_order['"]\s*:\s*(True|False)/.exec(header)?.[1];
  const shapeText = /['"]shape['"]\s*:\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || !fortran || shapeText === undefined) {
    throw new Error("Unsupported NPY header.");
  }
  const shape = shapeText
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number.parseInt(item, 10));
  return { descr, fortranOrder: fortran === "True", shape };
}

export function parseNpy(bytes: Uint8Array): NpyArray {
  if (bytes.length < 12 || bytes[0] !== 0x93 || new TextDecoder().decode(bytes.subarray(1, 6)) !== "NUMPY") {
    throw new Error("Invalid NPY payload.");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const major = bytes[6];
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(bytes.subarray(headerOffset, headerOffset + headerLength));
  const metadata = parseHeader(header);
  if (metadata.fortranOrder) throw new Error("Fortran-ordered NPY arrays are not supported.");
  const offset = bytes.byteOffset + headerOffset + headerLength;
  const count = metadata.shape.reduce((product, value) => product * value, 1);
  const buffer = bytes.buffer;
  let data: ArrayBufferView;
  switch (metadata.descr) {
    case "<f4": data = new Float32Array(buffer, offset, count); break;
    case "<f8": data = new Float64Array(buffer, offset, count); break;
    case "<i4": data = new Int32Array(buffer, offset, count); break;
    case "<i8": data = new BigInt64Array(buffer, offset, count); break;
    case "<u4": data = new Uint32Array(buffer, offset, count); break;
    case "|i1": data = new Int8Array(buffer, offset, count); break;
    case "|u1": data = new Uint8Array(buffer, offset, count); break;
    case "<c8": data = new Float32Array(buffer, offset, count * 2); break;
    case "<c16": data = new Float64Array(buffer, offset, count * 2); break;
    default: throw new Error(`Unsupported NPY dtype ${metadata.descr}.`);
  }
  return { ...metadata, data };
}

export function asFloat64(array: NpyArray): Float64Array {
  if (array.data instanceof Float64Array) return new Float64Array(array.data);
  if (array.data instanceof Float32Array) return Float64Array.from(array.data);
  throw new Error(`Expected a floating-point NPY array, received ${array.descr}.`);
}

export function asFloat32(array: NpyArray): Float32Array {
  if (array.data instanceof Float32Array) return new Float32Array(array.data);
  if (array.data instanceof Float64Array) return Float32Array.from(array.data);
  throw new Error(`Expected a floating-point NPY array, received ${array.descr}.`);
}

export function asComplexFloat32(array: NpyArray): { real: Float32Array; imag: Float32Array } {
  if (array.descr !== "<c8" && array.descr !== "<c16") {
    throw new Error(`Expected a complex NPY array, received ${array.descr}.`);
  }
  const source = array.data as Float32Array | Float64Array;
  const count = source.length / 2;
  const real = new Float32Array(count);
  const imag = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    real[index] = Number(source[index * 2]);
    imag[index] = Number(source[index * 2 + 1]);
  }
  return { real, imag };
}
