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
  packages: DesktopPackageSelection[];
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

interface DesktopMicrophoneSweepRequest {
  packagePath: string;
  backend: "cuda";
  sources: import("./model/types").SourceConfiguration[];
  microphones: import("./model/types").MicrophoneConfiguration[];
}

interface DesktopMicrophoneSweepProgress {
  type: "microphone-progress";
  frequency_hz: number;
  completed_count: number;
  total_count: number;
  microphone_ids: string[];
  spl_db: number[];
}

interface Window {
  boundaryLabDeployProfile?: Record<string, unknown>;
  boundaryLabDesktop?: {
    loadBundledExample: () => Promise<DesktopPackageSelection | null>;
    openProject: () => Promise<DesktopProjectSelection | null>;
    openSpeakerPackage: () => Promise<DesktopPackageSelection | null>;
    saveProject: (contents: string, suggestedName: string) => Promise<string | null>;
    solveLevel2: (request: DesktopLevel2SolveRequest) => Promise<import("./model/types").Level2SolveResult>;
    calculateMicrophoneSweep: (request: DesktopMicrophoneSweepRequest) => Promise<import("./model/types").MicrophoneSweepResult>;
    cancelMicrophoneSweep: () => Promise<boolean>;
    onSolveStatus: (listener: (status: DesktopSolveStatus) => void) => () => void;
    onMicrophoneSweepProgress: (listener: (progress: DesktopMicrophoneSweepProgress) => void) => () => void;
  };
}
