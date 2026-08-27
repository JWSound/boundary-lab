const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("boundaryLabDesktop", {
  loadBundledExample: () => ipcRenderer.invoke("deploy:load-bundled-example"),
  openProject: () => ipcRenderer.invoke("deploy:open-project"),
  openSpeakerPackage: () => ipcRenderer.invoke("deploy:open-speaker-package"),
  saveProject: (contents, suggestedName) => ipcRenderer.invoke("deploy:save-project", contents, suggestedName),
  solveLevel2: (payload) => ipcRenderer.invoke("deploy:solve-level2", payload),
  onSolveStatus: (listener) => {
    const handler = (_event, payload) => listener(payload);
    ipcRenderer.on("deploy:solve-status", handler);
    return () => ipcRenderer.removeListener("deploy:solve-status", handler);
  },
});
