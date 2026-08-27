const { app, BrowserWindow, dialog, ipcMain } = require("electron");
const { spawn } = require("node:child_process");
const { readFile, unlink, writeFile } = require("node:fs/promises");
const { basename, dirname, isAbsolute, join, resolve } = require("node:path");
const { performance } = require("node:perf_hooks");

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
        const parseStarted = performance.now();
        try {
          const message = JSON.parse(line);
          this.handleMessage(message, {
            jsonParseMs: performance.now() - parseStarted,
            stdoutBytes: Buffer.byteLength(line, "utf8"),
          });
        } catch (error) {
          console.error("Invalid Deploy solve worker response", error, line);
        }
      }
      newline = this.stdoutBuffer.indexOf("\n");
    }
  }

  handleMessage(message, transport = {}) {
    if (message.type === "ready") {
      this.resolveReady?.();
      this.resolveReady = null;
      this.rejectReady = null;
      return;
    }
    const job = this.pending.get(message.id);
    if (!job) return;
    if (message.type === "status" || message.type === "initialized") {
      if (job.sender && !job.sender.isDestroyed()) job.sender.send("deploy:solve-status", message);
    } else if (message.type === "result") {
      job.result = message.result;
      job.resultTransport = transport;
    } else if (message.type === "profile") {
      job.workerProfile = message.metrics;
    } else if (message.type === "completed") {
      this.pending.delete(message.id);
      if (job.kind === "warmup") {
        job.resolve();
        return;
      }
      if (job.result) {
        job.result.pipeline = {
          ...(job.result.pipeline || {}),
          ...(job.workerProfile || {}),
          electron_worker_ready_wait_s: job.workerReadyWaitMs / 1000,
          electron_request_json_encode_s: job.requestJsonEncodeMs / 1000,
          electron_python_stdin_bytes: job.requestBytes,
          electron_worker_result_json_parse_s: (job.resultTransport?.jsonParseMs || 0) / 1000,
          python_electron_stdout_bytes: job.resultTransport?.stdoutBytes || 0,
          electron_worker_roundtrip_s: (performance.now() - job.startedAt) / 1000,
        };
        job.resolve(job.result);
      }
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
    const invokedAt = performance.now();
    await this.ensureStarted();
    const workerReadyWaitMs = performance.now() - invokedAt;
    if (!this.process?.stdin.writable) throw new Error("Deploy solve worker is unavailable.");
    const id = this.nextId++;
    const encodeStarted = performance.now();
    const request = `${JSON.stringify({ id, operation: "solve", payload })}\n`;
    const requestJsonEncodeMs = performance.now() - encodeStarted;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        kind: "solve",
        resolve,
        reject,
        sender,
        result: null,
        workerProfile: null,
        resultTransport: null,
        startedAt: invokedAt,
        workerReadyWaitMs,
        requestJsonEncodeMs,
        requestBytes: Buffer.byteLength(request, "utf8"),
      });
      this.process.stdin.write(request);
    });
  }

  async warmup() {
    await this.ensureStarted();
    if (!this.process?.stdin.writable) throw new Error("Deploy solve worker is unavailable.");
    const id = this.nextId++;
    const request = `${JSON.stringify({ id, operation: "warmup", backend: "cuda" })}\n`;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        kind: "warmup",
        resolve,
        reject,
        sender: null,
        result: null,
      });
      this.process.stdin.write(request);
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
  const benchmarkLevel2 = process.argv.includes("--benchmark-level2");
  const smokeTest = process.argv.includes("--smoke-test") || level2Smoke || benchmarkLevel2;
  const window = new BrowserWindow({
    width: 1540,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: "#101311",
    show: !smokeTest || benchmarkLevel2,
    titleBarStyle: "hiddenInset",
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(here, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      backgroundThrottling: !benchmarkLevel2,
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
      let openProjectInteraction = null;
      if (!benchmarkLevel2 && !level2Smoke) {
        const smokeProjectPath = join(app.getPath("temp"), `boundary-lab-deploy-${process.pid}.blabdeploy.json`);
        const smokeProject = {
          schema: "boundary-lab-deploy-project",
          schema_version: 2,
          name: "Loaded Project Smoke",
          package: {
            id: "smoke-package",
            name: "S218BP",
            source_file: join(here, "../library/S218BP_LOD.blabsp"),
          },
          sources: [
            { id: "subwoofer-1", positionX: -2, positionHeightM: 0.4, positionZ: 0, pitchDeg: 0, yawDeg: 0, rollDeg: 0, levelDb: -3, delayMs: 0, polarity: 1 },
            { id: "subwoofer-2", positionX: 2, positionHeightM: 0.4, positionZ: 0, pitchDeg: 0, yawDeg: 0, rollDeg: 0, levelDb: -3, delayMs: 0, polarity: 1 },
          ],
          observation_plane: { widthM: 18, depthM: 16, centerXM: 1, nearM: 2, heightM: 1.4, pitchDeg: 0, yawDeg: 0, rollDeg: 0, columns: 36, rows: 32, heatmapMinimumDb: 55, heatmapMaximumDb: 135, heatmapBandingDb: 5 },
          selected_frequency_hz: 80,
          requested_fidelity: "pattern",
        };
        await writeFile(smokeProjectPath, `${JSON.stringify(smokeProject, null, 2)}\n`, "utf8");
        const showOpenDialog = dialog.showOpenDialog;
        dialog.showOpenDialog = async (options) => options?.title === "Open Boundary Lab Deploy project"
          ? { canceled: false, filePaths: [smokeProjectPath] }
          : showOpenDialog.call(dialog, options);
        try {
          openProjectInteraction = await window.webContents.executeJavaScript(`new Promise((resolve) => {
            window.confirm = () => true;
            document.querySelector('button[aria-label="Open project"]')?.click();
            const deadline = Date.now() + 10000;
            const check = () => {
              const name = document.querySelector('.project-breadcrumb strong')?.textContent?.trim() || '';
              const error = document.querySelector('.error-toast span')?.textContent || '';
              if (name === 'Loaded Project Smoke' || error || Date.now() >= deadline) {
                resolve({
                  name,
                  error: error || null,
                  sourceCount: document.querySelectorAll('.tree-button[data-object-id^="subwoofer-"]').length,
                  edited: Boolean(document.querySelector('.project-breadcrumb i'))
                });
              } else setTimeout(check, 50);
            };
            check();
          })`);
        } finally {
          dialog.showOpenDialog = showOpenDialog;
          await unlink(smokeProjectPath).catch(() => {});
        }
      }
      await window.webContents.executeJavaScript(`(() => {
        document.querySelectorAll('.fidelity-switcher button')[1]?.click();
        return new Promise((resolve) => setTimeout(resolve, 0));
      })()`);
      if (benchmarkLevel2) {
        const benchmark = await window.webContents.executeJavaScript(`(async () => {
          const waitForProfile = (previousGeneration, actionStarted, label) => new Promise((resolve) => {
            const deadline = performance.now() + 240000;
            const check = () => {
              const profile = window.boundaryLabDeployProfile;
              const error = document.querySelector('.error-toast span')?.textContent || '';
              if (profile?.texture_ready && Number(profile.generation || 0) > previousGeneration) {
                resolve({ label, interaction_to_texture_s: (performance.now() - actionStarted) / 1000, profile });
              } else if (error || performance.now() >= deadline) {
                resolve({ label, error: error || 'benchmark timeout', profile: profile || null });
              } else {
                setTimeout(check, 25);
              }
            };
            check();
          });
          const setNativeValue = (input, value) => {
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
            if (!input || !setter) throw new Error('Benchmark control is unavailable.');
            setter.call(input, String(value));
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
          };

          window.boundaryLabDeployProfile = undefined;
          const coldStarted = performance.now();
          document.querySelector('.primary-button')?.click();
          const cold = await waitForProfile(0, coldStarted, 'cold_54x54');
          if (cold.error) return { cases: [cold] };

          document.querySelector('.tree-button[data-object-id="subwoofer-1"]')?.click();
          await new Promise((resolve) => requestAnimationFrame(resolve));
          const sourceX = document.querySelector('.right-panel input[aria-label="X"]');
          const warmStarted = performance.now();
          setNativeValue(sourceX, Number(sourceX.value) + 0.1);
          const warm = await waitForProfile(Number(cold.profile.generation), warmStarted, 'warm_move_54x54');
          if (warm.error) return { cases: [cold, warm] };

          document.querySelector('.tree-button[data-object-id="audience-plane"]')?.click();
          await new Promise((resolve) => requestAnimationFrame(resolve));
          const resolution = document.querySelector('input[aria-label="Plane resolution"]');
          const highResolutionStarted = performance.now();
          setNativeValue(resolution, 200);
          const highResolution = await waitForProfile(
            Number(warm.profile.generation),
            highResolutionStarted,
            'warm_plane_200x200'
          );
          if (highResolution.error) return { cases: [cold, warm, highResolution] };

          const planeX = document.querySelector('.right-panel input[aria-label="X"]');
          const repeatedPlaneStarted = performance.now();
          setNativeValue(planeX, Number(planeX.value) + 0.1);
          const repeatedPlane = await waitForProfile(
            Number(highResolution.profile.generation),
            repeatedPlaneStarted,
            'warm_plane_repeat_200x200'
          );
          return {
            cases: [cold, warm, highResolution, repeatedPlane],
            final_grid: document.querySelector('.plane-resolution-row output')?.textContent?.trim() || null,
          };
        })()`);
        console.log(JSON.stringify({ benchmark: "deploy-level2-pipeline", ...benchmark, consoleErrors }));
        app.quit();
        return;
      }
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
                    heatmapMinimumDb: document.querySelector('input[aria-label="Scale minimum"]')?.value,
                    heatmapMaximumDb: document.querySelector('input[aria-label="Scale maximum"]')?.value,
                    heatmapBandingDb: document.querySelector('input[aria-label="Banding"]')?.value,
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
      const sceneObjectInteraction = await window.webContents.executeJavaScript(`new Promise((resolve) => {
        const first = document.querySelector('.tree-button[data-object-id="subwoofer-1"]');
        const second = document.querySelector('.tree-button[data-object-id="subwoofer-2"]');
        first?.click();
        second?.dispatchEvent(new MouseEvent('click', { bubbles: true, ctrlKey: true }));
        requestAnimationFrame(() => {
          const viewport = document.querySelector('.viewport');
          const selectedBeforeAdd = Array.from(document.querySelectorAll('.tree-button[aria-selected="true"]'))
            .map((row) => row.getAttribute('data-object-id'));
          const selectedCountBeforeAdd = viewport?.getAttribute('data-selected-object-count');
          const sourceCountBeforeAdd = document.querySelectorAll('.tree-button[data-object-id^="subwoofer-"]').length;
          document.querySelector('button[aria-label="Add speaker"]')?.click();
          requestAnimationFrame(() => {
            const sourceCountAfterAdd = document.querySelectorAll('.tree-button[data-object-id^="subwoofer-"]').length;
            const addedId = document.querySelector('.tree-button[aria-selected="true"]')?.getAttribute('data-object-id');
            const remove = document.querySelector('button[aria-label="Remove selected speakers"]');
            const removeEnabled = !remove?.disabled;
            remove?.click();
            requestAnimationFrame(() => resolve({
              selectedBeforeAdd,
              selectedCountBeforeAdd,
              sourceCountBeforeAdd,
              sourceCountAfterAdd,
              addedId,
              removeEnabled,
              sourceCountAfterRemove: document.querySelectorAll('.tree-button[data-object-id^="subwoofer-"]').length
            }));
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
        openProjectAvailable: Boolean(document.querySelector('button[aria-label="Open project"]')),
        liveSolveEnabled: document.querySelector('.primary-button')?.getAttribute('aria-pressed'),
        solveStatus: document.querySelector('.solve-status strong')?.textContent,
        solveError: document.querySelector('.error-toast span')?.textContent || null
      })`);
      console.log(JSON.stringify({ ...snapshot, openProjectInteraction, transformInteraction, planeResolutionInteraction, sceneObjectInteraction, level2Move, consoleErrors }));
      app.quit();
    });
    setTimeout(() => {
      console.error("Deploy desktop smoke test timed out.");
      app.exit(1);
    }, benchmarkLevel2 ? 720000 : level2Smoke ? 130000 : 15000).unref();
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

ipcMain.handle("deploy:open-project", async () => {
  const selection = await dialog.showOpenDialog({
    title: "Open Boundary Lab Deploy project",
    filters: [
      { name: "Boundary Lab Deploy projects", extensions: ["blabdeploy.json"] },
      { name: "JSON files", extensions: ["json"] },
      { name: "All files", extensions: ["*"] },
    ],
    properties: ["openFile"],
  });
  if (selection.canceled || selection.filePaths.length === 0) return null;
  const projectPath = selection.filePaths[0];
  const contents = await readFile(projectPath, "utf8");
  let project;
  try {
    project = JSON.parse(contents);
  } catch {
    return { name: basename(projectPath), path: projectPath, contents, package: null };
  }

  const sourceFile = typeof project?.package?.source_file === "string"
    ? project.package.source_file
    : null;
  const candidates = sourceFile
    ? [
        isAbsolute(sourceFile) ? sourceFile : resolve(dirname(projectPath), sourceFile),
        join(here, "../library", basename(sourceFile)),
      ]
    : [];
  for (const candidate of [...new Set(candidates)]) {
    try {
      return {
        name: basename(projectPath),
        path: projectPath,
        contents,
        package: await readPackageSelection(candidate),
      };
    } catch {
      // Try the next portable or bundled package location.
    }
  }

  const packageSelection = await dialog.showOpenDialog({
    title: sourceFile
      ? `Locate ${basename(sourceFile)}`
      : "Locate the project speaker package",
    defaultPath: dirname(projectPath),
    filters: [
      { name: "Boundary Lab speaker packages", extensions: ["blabsp"] },
      { name: "All files", extensions: ["*"] },
    ],
    properties: ["openFile"],
  });
  const packageResult = packageSelection.canceled || packageSelection.filePaths.length === 0
    ? null
    : await readPackageSelection(packageSelection.filePaths[0]);
  return { name: basename(projectPath), path: projectPath, contents, package: packageResult };
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
  void deployWorker.warmup().catch((error) => console.error("Deploy worker warmup failed", error));
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => deployWorker.close());
