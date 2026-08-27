/// <reference types="vite/client" />

interface DesktopPackageSelection {
  name: string;
  path: string;
  bytes: ArrayBuffer;
}

interface DesktopProjectSelection {
  name: string;
  path: string;
  contents: string;
  package: DesktopPackageSelection | null;
}

interface DesktopLevel2SolveRequest {
  packagePath: string;
  frequencyHz: number;
  backend: "cuda";
  sources: import("./model/types").SourceConfiguration[];
  observation: import("./model/types").ObservationPlane;
  solutionKey?: string;
  reuseBoundary?: boolean;
  includeComplexPressure?: boolean;
}

interface DesktopSolveStatus {
  id: number;
  type: "status" | "initialized";
  message?: string;
  metadata?: Record<string, unknown>;
}

interface Window {
  boundaryLabDeployProfile?: Record<string, unknown>;
  boundaryLabDesktop?: {
    loadBundledExample: () => Promise<DesktopPackageSelection | null>;
    openProject: () => Promise<DesktopProjectSelection | null>;
    openSpeakerPackage: () => Promise<DesktopPackageSelection | null>;
    saveProject: (contents: string, suggestedName: string) => Promise<string | null>;
    solveLevel2: (request: DesktopLevel2SolveRequest) => Promise<import("./model/types").Level2SolveResult>;
    onSolveStatus: (listener: (status: DesktopSolveStatus) => void) => () => void;
  };
}
