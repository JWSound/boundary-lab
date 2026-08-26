const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { spawn } = require("node:child_process");
const { readFile, writeFile } = require("node:fs/promises");
const { join } = require("node:path");

const here = __dirname;
const repositoryRoot = join(here, "../..");

class DeployWorkerClient {
  constructor() {
    this.process = null;
    this.stdoutBuffer = "";
    this.pending = new Map();
    this.nextId = 1;
    this.readyPromise = null;
    this.resolveReady = null;
    this.rejectReady = null;
  }

  ensureStarted() {
    if (this.process && this.readyPromise) return this.readyPromise;
    const python = process.env.BLAB_PYTHON_EXE || "python";
    const pythonPath = [join(repositoryRoot, "src"), process.env.PYTHONPATH].filter(Boolean).join(require("node:path").delimiter);
    this.process = spawn(python, ["-m", "blab.deploy_worker"], {
      cwd: repositoryRoot,
      env: { ...process.env, PYTHONPATH: pythonPath },
      windowsHide: true,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.readyPromise = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.process.stdout.setEncoding("utf8");
    this.process.stdout.on("data", (chunk) => this.consumeStdout(chunk));
    this.process.stderr.setEncoding("utf8");
    this.process.stderr.on("data", (chunk) => {
      const message = chunk.trim();
      if (message) console.error(`Deploy solve worker: ${message}`);
    });
    this.process.once("error", (error) => this.handleExit(error));
    this.process.once("exit", (code) => this.handleExit(new Error(`Deploy solve worker exited with code ${code}.`)));
    return this.readyPromise;
  }

  consumeStdout(chunk) {
    this.stdoutBuffer += chunk;
    let newline = this.stdoutBuffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.stdoutBuffer.slice(0, newline).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
      if (line) {
        try {
          this.handleMessage(JSON.parse(line));
        } catch (error) {
          console.error("Invalid Deploy solve worker response", error, line);
        }
      }
      newline = this.stdoutBuffer.indexOf("\n");
    }
  }

  handleMessage(message) {
    if (message.type === "ready") {
      this.resolveReady?.();
      this.resolveReady = null;
      this.rejectReady = null;
      return;
    }
    const job = this.pending.get(message.id);
    if (!job) return;
    if (message.type === "status" || message.type === "initialized") {
      if (!job.sender.isDestroyed()) job.sender.send("deploy:solve-status", message);
    } else if (message.type === "result") {
      job.result = message.result;
    } else if (message.type === "completed") {
      this.pending.delete(message.id);
      if (job.result) job.resolve(job.result);
      else job.reject(new Error("Level 2 solve completed without returning a field."));
    } else if (message.type === "cancelled") {
      this.pending.delete(message.id);
      job.reject(new Error("Level 2 solve was cancelled."));
    } else if (message.type === "failed") {
      this.pending.delete(message.id);
      job.reject(new Error(message.error || "Level 2 solve failed."));
    }
  }

  handleExit(error) {
    this.rejectReady?.(error);
    for (const job of this.pending.values()) job.reject(error);
    this.pending.clear();
    this.process = null;
    this.readyPromise = null;
    this.resolveReady = null;
    this.rejectReady = null;
  }

  async solve(payload, sender) {
    await this.ensureStarted();
    if (!this.process?.stdin.writable) throw new Error("Deploy solve worker is unavailable.");
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, sender, result: null });
      this.process.stdin.write(`${JSON.stringify({ id, operation: "solve", payload })}\n`);
    });
  }

  close() {
    if (this.process && !this.process.killed) this.process.kill();
    this.process = null;
  }
}

const deployWorker = new DeployWorkerClient();

function createWindow() {
  const level2Smoke = process.argv.includes("--smoke-level2");
  const smokeTest = process.argv.includes("--smoke-test") || level2Smoke;
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
          if (source.includes('S218BP_LOD.blabsp') || Date.now() >= deadline) resolve(source);
          else setTimeout(check, 50);
        };
        check();
      })`);
      await window.webContents.executeJavaScript(`(() => {
        document.querySelectorAll('.fidelity-switcher button')[1]?.click();
        return new Promise((resolve) => setTimeout(resolve, 0));
      })()`);
      const transformInteraction = await window.webContents.executeJavaScript(`new Promise((resolve) => {
        const viewport = document.querySelector('.viewport');
        window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w', bubbles: true }));
        requestAnimationFrame(() => {
          const translateMode = viewport?.getAttribute('data-transform-mode');
          window.dispatchEvent(new KeyboardEvent('keydown', { key: 'e', bubbles: true }));
          requestAnimationFrame(() => {
            const rotateMode = viewport?.getAttribute('data-transform-mode');
            window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Alt', altKey: true, bubbles: true }));
            requestAnimationFrame(() => {
              const altDisablesSnap = viewport?.getAttribute('data-angle-snap-disabled');
              window.dispatchEvent(new KeyboardEvent('keyup', { key: 'Alt', bubbles: true }));
              resolve({
                translateMode,
                rotateMode,
                altDisablesSnap,
                grabPointCount: viewport?.getAttribute('data-grab-point-count')
              });
            });
          });
        });
      })`);
      const planeResolutionInteraction = await window.webContents.executeJavaScript(`new Promise((resolve) => {
        document.querySelector('.tree-button[data-object-id="audience-plane"]')?.click();
        requestAnimationFrame(() => {
          window.dispatchEvent(new KeyboardEvent('keydown', { key: 'w', bubbles: true }));
          requestAnimationFrame(() => {
            const planeTranslateMode = document.querySelector('.viewport')?.getAttribute('data-transform-mode');
            window.dispatchEvent(new KeyboardEvent('keydown', { key: 'e', bubbles: true }));
            requestAnimationFrame(() => {
              const planeRotateMode = document.querySelector('.viewport')?.getAttribute('data-transform-mode');
              window.dispatchEvent(new KeyboardEvent('keydown', { key: 'r', bubbles: true }));
              const input = document.querySelector('input[aria-label="Plane resolution"]');
              const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
              if (!input || !setter) return resolve({ available: false });
              setter.call(input, '40');
              input.dispatchEvent(new Event('input', { bubbles: true }));
              requestAnimationFrame(() => {
                const result = {
                  available: true,
                  selectedObject: document.querySelector('.inspector-heading strong')?.textContent?.trim(),
                  planeTranslateMode,
                  planeRotateMode,
                  planeScaleMode: document.querySelector('.viewport')?.getAttribute('data-transform-mode'),
                  planeScaleHandleCount: document.querySelector('.viewport')?.getAttribute('data-grab-point-count'),
                  planeProperties: {
                    x: document.querySelector('input[aria-label="X"]')?.value,
                    near: document.querySelector('input[aria-label="Near"]')?.value,
                    height: document.querySelector('input[aria-label="Height"]')?.value,
                    pitch: document.querySelector('input[aria-label="Pitch"]')?.value,
                    yaw: document.querySelector('input[aria-label="Yaw"]')?.value,
                    roll: document.querySelector('input[aria-label="Roll"]')?.value,
                    sizeReadout: document.querySelector('.plane-size-readout output')?.textContent?.trim(),
                    editableSizeInputs: document.querySelectorAll('input[aria-label="Width"], input[aria-label="Depth"]').length
                  },
                  value: input.value,
                  maximum: input.max,
                  shape: document.querySelector('.plane-resolution-row output')?.textContent?.trim(),
                  fieldShape: document.querySelector('.solve-status em')?.textContent?.trim()
                };
                document.querySelector('.tree-button[data-object-id="subwoofer-1"]')?.click();
                requestAnimationFrame(() => resolve(result));
              });
            });
          });
        });
      })`);
      let level2Move = null;
      if (level2Smoke) {
        level2Move = await window.webContents.executeJavaScript(`new Promise((resolve) => {
          document.querySelector('.primary-button')?.click();
          const deadline = Date.now() + 120000;
          const check = () => {
            const status = document.querySelector('.solve-status strong')?.textContent || '';
            const error = document.querySelector('.error-toast span')?.textContent || '';
            if (status.includes('Boundary solution') || error || Date.now() >= deadline) resolve({ status, error });
            else setTimeout(check, 100);
          };
          check();
        })`);
        level2Move = await window.webContents.executeJavaScript(`new Promise((resolve) => {
          const statusPanel = document.querySelector('.solve-status');
          const initialRevision = Number(statusPanel?.getAttribute('data-solve-revision') || 0);
          const input = document.querySelector('.right-panel .number-row input');
          const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
          if (!input || !setter) return resolve({ initialRevision, moved: false });
          setter.call(input, '-1');
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
          const deadline = Date.now() + 30000;
          const check = () => {
            const status = document.querySelector('.solve-status strong')?.textContent || '';
            const revision = Number(statusPanel?.getAttribute('data-solve-revision') || 0);
            const error = document.querySelector('.error-toast span')?.textContent || '';
            if ((status.includes('Boundary solution') && revision > initialRevision) || error || Date.now() >= deadline) {
              resolve({ initialRevision, revision, moved: input.value === '-1' && revision > initialRevision, status, error });
            } else setTimeout(check, 50);
          };
          check();
        })`);
      }
      const snapshot = await window.webContents.executeJavaScript(`({
        title: document.title,
        shell: Boolean(document.querySelector('.app-shell')),
        canvasCount: document.querySelectorAll('canvas').length,
        packageName: document.querySelector('.package-name')?.textContent,
        packageSource: document.querySelector('.package-subtitle')?.textContent,
        metricCount: document.querySelectorAll('.metrics-grid > div').length,
        sourceCount: document.querySelectorAll('.scene-tree .tree-button').length,
        activeFidelity: document.querySelector('.fidelity-switcher button.active span')?.textContent,
        solveButtonEnabled: !document.querySelector('.primary-button')?.disabled,
        solveButtonLabel: document.querySelector('.primary-button')?.textContent?.trim(),
        liveSolveEnabled: document.querySelector('.primary-button')?.getAttribute('aria-pressed'),
        solveStatus: document.querySelector('.solve-status strong')?.textContent,
        solveError: document.querySelector('.error-toast span')?.textContent || null
      })`);
      console.log(JSON.stringify({ ...snapshot, transformInteraction, planeResolutionInteraction, level2Move, consoleErrors }));
      app.quit();
    });
    setTimeout(() => {
      console.error("Deploy desktop smoke test timed out.");
      app.exit(1);
    }, level2Smoke ? 130000 : 15000).unref();
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
  const path = join(here, "../library/S218BP_LOD.blabsp");
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

ipcMain.handle("deploy:solve-level2", async (event, payload) => deployWorker.solve(payload, event.sender));

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => deployWorker.close());
