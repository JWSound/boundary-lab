/// <reference types="vite/client" />

interface DesktopPackageSelection {
  name: string;
  path: string;
  bytes: ArrayBuffer;
}

interface DesktopLevel2SolveRequest {
  packagePath: string;
  frequencyHz: number;
  backend: "cuda";
  source: import("./model/types").SourceConfiguration;
  observation: import("./model/types").ObservationPlane;
}

interface DesktopSolveStatus {
  id: number;
  type: "status" | "initialized";
  message?: string;
  metadata?: Record<string, unknown>;
}

interface Window {
  boundaryLabDesktop?: {
    loadBundledExample: () => Promise<DesktopPackageSelection | null>;
    openSpeakerPackage: () => Promise<DesktopPackageSelection | null>;
    saveProject: (contents: string, suggestedName: string) => Promise<string | null>;
    solveLevel2: (request: DesktopLevel2SolveRequest) => Promise<import("./model/types").Level2SolveResult>;
    onSolveStatus: (listener: (status: DesktopSolveStatus) => void) => () => void;
  };
}
