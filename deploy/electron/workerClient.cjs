const { spawn } = require("node:child_process");
const { join } = require("node:path");
const { performance } = require("node:perf_hooks");

class DeployWorkerClient {
  constructor(repositoryRoot) {
    this.repositoryRoot = repositoryRoot;
    this.process = null;
    this.stdoutBuffer = "";
    this.pending = new Map();
    this.nextId = 1;
    this.readyPromise = null;
    this.resolveReady = null;
    this.rejectReady = null;
    this.warming = false;
  }

  ensureStarted() {
    if (this.process && this.readyPromise) return this.readyPromise;
    const python = process.env.BLAB_PYTHON_EXE || "python";
    const pythonPath = [join(this.repositoryRoot, "src"), process.env.PYTHONPATH].filter(Boolean).join(require("node:path").delimiter);
    this.process = spawn(python, ["-m", "blab.deploy.worker"], {
      cwd: this.repositoryRoot,
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
    } else if (message.type === "microphone-progress") {
      if (job.sender && !job.sender.isDestroyed()) job.sender.send("deploy:microphone-sweep-progress", message);
    } else if (message.type === "result") {
      job.result = message.result;
      job.resultTransport = transport;
    } else if (message.type === "profile") {
      job.workerProfile = message.metrics;
    } else if (message.type === "completed") {
      this.pending.delete(message.id);
      if (job.kind === "warmup" || job.kind === "cancel") {
        job.resolve(job.kind === "cancel" ? Boolean(message.cancelled) : undefined);
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
      else job.reject(new Error(`${job.kind === "microphone-sweep" ? "Microphone sweep" : "Level 2 solve"} completed without returning a field.`));
    } else if (message.type === "cancelled") {
      this.pending.delete(message.id);
      if (job.kind === "microphone-sweep") {
        job.resolve({
          cancelled: true,
          frequencies_hz: [],
          microphone_ids: [],
          spl_db: [],
          pressure: { real: [], imag: [] },
          completed_count: Number(message.completed_count || 0),
          total_count: 0,
        });
      } else {
        job.reject(new Error("Level 2 solve was cancelled."));
      }
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

  async solve(payload, sender, kind = "solve", operation = "solve") {
    const invokedAt = performance.now();
    if (this.warming && sender && !sender.isDestroyed()) {
      sender.send("deploy:solve-status", { type: "status", message: "Waiting for BEAT CUDA warmup" });
    }
    await this.ensureStarted();
    const workerReadyWaitMs = performance.now() - invokedAt;
    if (!this.process?.stdin.writable) throw new Error("Deploy solve worker is unavailable.");
    const id = this.nextId++;
    const encodeStarted = performance.now();
    const request = `${JSON.stringify({ id, operation, payload })}\n`;
    const requestJsonEncodeMs = performance.now() - encodeStarted;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {
        kind,
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

  async microphoneSweep(payload, sender) {
    return this.solve(payload, sender, "microphone-sweep", "microphone_sweep");
  }

  async cancelMicrophoneSweep() {
    await this.ensureStarted();
    if (!this.process?.stdin.writable) throw new Error("Deploy solve worker is unavailable.");
    const active = [...this.pending.entries()].find(([, job]) => job.kind === "microphone-sweep");
    if (!active) return false;
    const [targetId] = active;
    const id = this.nextId++;
    const request = `${JSON.stringify({ id, operation: "cancel", target_id: targetId })}\n`;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { kind: "cancel", resolve, reject, sender: null, result: null });
      this.process.stdin.write(request);
    });
  }

  async warmup() {
    this.warming = true;
    try {
      await this.ensureStarted();
      if (!this.process?.stdin.writable) throw new Error("Deploy solve worker is unavailable.");
      const id = this.nextId++;
      const request = `${JSON.stringify({ id, operation: "warmup", backend: "cuda" })}\n`;
      return await new Promise((resolve, reject) => {
        this.pending.set(id, {
          kind: "warmup",
          resolve,
          reject,
          sender: null,
          result: null,
        });
        this.process.stdin.write(request);
      });
    } finally {
      this.warming = false;
    }
  }

  close() {
    if (this.process && !this.process.killed) this.process.kill();
    this.process = null;
  }
}

module.exports = { DeployWorkerClient };
