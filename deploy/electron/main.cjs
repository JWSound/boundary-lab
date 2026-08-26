const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { readFile, writeFile } = require("node:fs/promises");
const { join } = require("node:path");

const here = __dirname;

function createWindow() {
  const smokeTest = process.argv.includes("--smoke-test");
  const window = new BrowserWindow({
    width: 1540,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: "#101311",
    show: !smokeTest,
    titleBarStyle: "hiddenInset",
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(here, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  if (smokeTest) {
    const consoleErrors = [];
    window.webContents.on("console-message", (_event, level, message) => {
      if (level >= 2) consoleErrors.push(message);
    });
    window.webContents.once("did-finish-load", async () => {
      await window.webContents.executeJavaScript(`new Promise((resolve) => {
        const deadline = Date.now() + 10000;
        const check = () => {
          const source = document.querySelector('.package-subtitle')?.textContent || '';
          if (source.includes('S218BP.blabsp') || Date.now() >= deadline) resolve(source);
          else setTimeout(check, 50);
        };
        check();
      })`);
      const snapshot = await window.webContents.executeJavaScript(`({
        title: document.title,
        shell: Boolean(document.querySelector('.app-shell')),
        canvasCount: document.querySelectorAll('canvas').length,
        packageName: document.querySelector('.package-name')?.textContent,
        packageSource: document.querySelector('.package-subtitle')?.textContent,
        metricCount: document.querySelectorAll('.metrics-grid > div').length
      })`);
      console.log(JSON.stringify({ ...snapshot, consoleErrors }));
      app.quit();
    });
    setTimeout(() => {
      console.error("Deploy desktop smoke test timed out.");
      app.exit(1);
    }, 15000).unref();
  }

  if (app.isPackaged || process.argv.includes("--built")) {
    void window.loadFile(join(here, "../dist/index.html"));
  } else {
    void window.loadURL("http://127.0.0.1:5173");
  }
}

async function readPackageSelection(path) {
  const bytes = await readFile(path);
  return {
    name: path.split(/[\\/]/).pop() ?? "speaker.blabsp",
    path,
    bytes: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  };
}

ipcMain.handle("deploy:load-bundled-example", async () => {
  const path = join(here, "../../examples/S218BP/S218BP.blabsp");
  try {
    return await readPackageSelection(path);
  } catch {
    return null;
  }
});

ipcMain.handle("deploy:open-speaker-package", async () => {
  const selection = await dialog.showOpenDialog({
    title: "Open Boundary Lab speaker package",
    filters: [
      { name: "Boundary Lab speaker packages", extensions: ["blabsp"] },
      { name: "All files", extensions: ["*"] },
    ],
    properties: ["openFile"],
  });
  if (selection.canceled || selection.filePaths.length === 0) return null;
  return readPackageSelection(selection.filePaths[0]);
});

ipcMain.handle("deploy:save-project", async (_event, contents, suggestedName) => {
  const selection = await dialog.showSaveDialog({
    title: "Save Boundary Lab Deploy project",
    defaultPath: suggestedName,
    filters: [
      { name: "Boundary Lab Deploy projects", extensions: ["blabdeploy.json"] },
      { name: "JSON files", extensions: ["json"] },
    ],
  });
  if (selection.canceled || !selection.filePath) return null;
  await writeFile(selection.filePath, contents, "utf8");
  return selection.filePath;
});

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
