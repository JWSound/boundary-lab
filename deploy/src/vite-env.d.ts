/// <reference types="vite/client" />

interface DesktopPackageSelection {
  name: string;
  path: string;
  bytes: ArrayBuffer;
}

interface Window {
  boundaryLabDesktop?: {
    loadBundledExample: () => Promise<DesktopPackageSelection | null>;
    openSpeakerPackage: () => Promise<DesktopPackageSelection | null>;
    saveProject: (contents: string, suggestedName: string) => Promise<string | null>;
  };
}
