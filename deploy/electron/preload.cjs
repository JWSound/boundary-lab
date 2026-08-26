const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("boundaryLabDesktop", {
  loadBundledExample: () => ipcRenderer.invoke("deploy:load-bundled-example"),
  openSpeakerPackage: () => ipcRenderer.invoke("deploy:open-speaker-package"),
  saveProject: (contents, suggestedName) => ipcRenderer.invoke("deploy:save-project", contents, suggestedName),
});
