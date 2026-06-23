/**
 * Python Service Manager
 *
 * Spawns and manages Python backend processes (piste, precis).
 * Backends auto-start on first tool call and are killed on MCP server exit.
 * Uses stdin/stdout JSON protocol for fast communication after initial import.
 */

import { spawn, execSync, type ChildProcess } from 'node:child_process';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { Logger } from './logging.js';

// ═══════════════════════════════════════════════════════════════════════
// Auto-detect Python — tries multiple candidates, returns first working.
// ═══════════════════════════════════════════════════════════════════════

let _cachedPython: string | null = null;

/** Find a working Python executable. Caches result after first successful find. */
export function findPython(logger?: Logger): string | null {
  if (_cachedPython) return _cachedPython;

  const candidates = buildCandidates();
  logger?.debug(`Python: trying ${candidates.length} candidates...`);

  for (const candidate of candidates) {
    try {
      const result = execSync(`"${candidate}" --version`, {
        stdio: ['ignore', 'pipe', 'ignore'],
        timeout: 5000,
        windowsHide: true,
      });
      const version = result.toString().trim();
      if (version.toLowerCase().includes('python')) {
        _cachedPython = candidate;
        logger?.info(`Python found: ${candidate} (${version})`);
        return candidate;
      }
    } catch {
      // Try next candidate
    }
  }

  logger?.warn('Python not found automatically. Install from https://python.org');
  logger?.warn('Candidates tried: ' + candidates.join(', '));
  return null;
}

/** Reset cached Python path (useful for testing). */
export function resetPythonCache(): void {
  _cachedPython = null;
}

/** Build the list of Python candidates in priority order. */
function buildCandidates(): string[] {
  const isWin = process.platform === 'win32';
  const candidates: string[] = [];

  if (isWin) {
    // Windows: try py launcher first (installed with official Python)
    candidates.push('py', 'python', 'python3');

    const localAppData = process.env.LOCALAPPDATA || '';
    const programFiles = process.env.ProgramFiles || 'C:\\Program Files';
    const programFilesX86 = process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)';

    // 1st: Official Python.org installs (working SSL, short paths)
    for (const ver of ['312', '311', '310', '313', '39', '38']) {
      candidates.push(
        `${programFiles}\\Python${ver}\\python.exe`,
        `${programFilesX86}\\Python${ver}\\python.exe`,
      );
      if (localAppData) {
        candidates.push(`${localAppData}\\Programs\\Python\\Python${ver}\\python.exe`);
      }
    }

    // 2nd: Conda / Miniconda
    const userProfile = process.env.USERPROFILE || '';
    if (userProfile) {
      for (const conda of ['Anaconda3', 'miniconda3', 'Miniconda3']) {
        candidates.push(
          `${userProfile}\\${conda}\\python.exe`,
          `${userProfile}\\${conda}\\Scripts\\python.exe`,
        );
      }
    }

    // 3rd: Chocolatey / winget
    candidates.push('C:\\Python312\\python.exe', 'C:\\Python311\\python.exe', 'C:\\Python310\\python.exe');

    // Last resort: Microsoft Store Python (⚠ broken SSL, sandboxed)
    // Only used if no other Python is found — tools may fail with SSL errors
    if (localAppData) {
      candidates.push(
        `${localAppData}\\Microsoft\\WindowsApps\\python.exe`,
        `${localAppData}\\Microsoft\\WindowsApps\\python3.exe`,
      );
    }
  } else {
    // macOS / Linux
    candidates.push(
      'python3', 'python',
      '/usr/bin/python3', '/usr/bin/python',
      '/usr/local/bin/python3', '/usr/local/bin/python',
      '/opt/homebrew/bin/python3',
    );
  }

  // Deduplicate
  return [...new Set(candidates)];
}

export interface PythonServiceOptions {
  /** Unique name for this service (e.g. "precis", "piste") */
  name: string;
  /** Absolute path to the Python bridge script */
  scriptPath: string;
  /** Working directory for the Python process */
  cwd: string;
  /** Environment variables to pass to Python */
  env?: Record<string, string>;
  /** Timeout in ms for initial health check */
  healthTimeout?: number;
  /** Whether to auto-start on first request */
  autoStart?: boolean;
}

export class PythonService {
  private process: ChildProcess | null = null;
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  private nextId = 1;
  private buffer = '';
  private started = false;
  private healthy = false;

  constructor(
    private options: PythonServiceOptions,
    private logger: Logger,
  ) {}

  get name(): string { return this.options.name; }
  get isHealthy(): boolean { return this.healthy; }

  /** Start the Python process. Idempotent — only starts once. */
  async start(): Promise<void> {
    if (this.started) return;
    this.started = true;

    this.logger.info(`Starting ${this.options.name} backend...`);

    // Prefer uv-managed Python (proper SSL) over system Python
    const uvPath = process.platform === 'win32'
      ? `${process.cwd()}\\.vendor\\uv.exe`
      : `${process.cwd()}/.vendor/uv`;

    let cmd = '';
    let args: string[] = [];

    if (existsSync(uvPath)) {
      // Check if uv has a managed Python with working SSL
      try {
        const managed = execSync(`"${uvPath}" python find 3.11 --no-python-downloads`, {
          encoding: 'utf8', stdio: 'pipe', timeout: 10_000, windowsHide: true,
        }).trim();
        if (managed && !managed.includes('WindowsApps') && existsSync(managed)) {
          cmd = managed;
          args = [this.options.scriptPath];
          this.logger.info(`${this.options.name}: using uv-managed Python → ${managed}`);
        }
      } catch { /* fall through to system Python */ }
    }

    if (!cmd) {
      const pythonCmd = findPython(this.logger);
      if (!pythonCmd) {
        this.logger.error(`${this.options.name}: Python not found. Install from https://python.org`);
        this.healthy = false;
        return;
      }
      cmd = pythonCmd;
      args = [this.options.scriptPath];
    }

    const env = { ...process.env, ...this.options.env, PYTHONUNBUFFERED: '1' };

    // Include .python-packages (created by setup.mjs — short path, no MAX_PATH)
    // pip --target installs directly into the directory
    const targetPackages = `${process.cwd()}${process.platform === 'win32' ? '\\.python-packages' : '/.python-packages'}`;
    if (env.PYTHONPATH) {
      env.PYTHONPATH = process.platform === 'win32'
        ? `${targetPackages};${env.PYTHONPATH}`
        : `${targetPackages}:${env.PYTHONPATH}`;
    } else {
      env.PYTHONPATH = targetPackages;
    }

    this.process = spawn(cmd, args, {
      cwd: this.options.cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
      env,
      windowsHide: true,
    });

    this.process.stdout?.on('data', (chunk: Buffer) => {
      this.buffer += chunk.toString();
      this.processBuffer();
    });

    this.process.stderr?.on('data', (chunk: Buffer) => {
      const msg = chunk.toString().trim();
      if (msg) this.logger.debug(`[${this.options.name}] ${msg}`);
    });

    this.process.on('error', (err) => {
      this.logger.error(`${this.options.name} process error: ${err.message}`);
      this.healthy = false;
    });

    this.process.on('exit', (code) => {
      this.logger.warn(`${this.options.name} process exited with code ${code}`);
      this.healthy = false;
      this.started = false;
    });

    // Wait for health check
    try {
      await this.waitForHealth(this.options.healthTimeout ?? 15000);
      this.healthy = true;
      this.logger.info(`${this.options.name} backend is ready`);
    } catch (err: any) {
      this.logger.warn(`${this.options.name} health check failed: ${err.message}. Will retry on first request.`);
    }
  }

  /** Send a request to the Python backend and get the response. */
  async call(action: string, params: Record<string, unknown> = {}): Promise<any> {
    if (!this.started) {
      await this.start();
    }

    if (!this.healthy || !this.process || this.process.killed) {
      throw new Error(`${this.options.name}: Python backend is not running. Check that setup.mjs completed successfully.`);
    }

    const id = this.nextId++;
    const request = JSON.stringify({ id, action, params });

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${this.options.name}: timeout waiting for response to "${action}"`));
      }, 60000);

      this.pending.set(id, {
        resolve: (v: any) => { clearTimeout(timer); resolve(v); },
        reject: (e: Error) => { clearTimeout(timer); reject(e); },
      });

      try {
        this.process!.stdin!.write(request + '\n');
      } catch (err: any) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(new Error(`${this.options.name}: failed to send request: ${err.message}`));
      }
    });
  }

  /** Stop the Python process. */
  stop(): void {
    if (this.process) {
      this.process.kill('SIGTERM');
      // Force kill after 5s
      setTimeout(() => { if (this.process && !this.process.killed) this.process.kill('SIGKILL'); }, 5000);
      this.process = null;
      this.started = false;
      this.healthy = false;
    }
  }

  // ── Private ──────────────────────────────────────────────────────

  private processBuffer(): void {
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      // Check for health signal
      if (trimmed === '__READY__') {
        this.healthy = true;
        this.resolvePendingHealth();
        continue;
      }

      try {
        const msg = JSON.parse(trimmed);
        if (msg.id && this.pending.has(msg.id)) {
          const { resolve, reject } = this.pending.get(msg.id)!;
          this.pending.delete(msg.id);
          if (msg.error) {
            reject(new Error(msg.error));
          } else {
            resolve(msg.result);
          }
        }
      } catch {
        // Not JSON — might be a log line
      }
    }
  }

  private healthResolve: (() => void) | null = null;

  private async waitForHealth(timeout: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('Health check timed out')), timeout);
      this.healthResolve = () => {
        clearTimeout(timer);
        resolve();
      };
      // If already healthy
      if (this.healthy) {
        clearTimeout(timer);
        resolve();
      }
    });
  }

  private resolvePendingHealth(): void {
    if (this.healthResolve) {
      this.healthResolve();
      this.healthResolve = null;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════

export class PythonServiceManager {
  private services = new Map<string, PythonService>();

  constructor(private logger: Logger) {}

  register(options: PythonServiceOptions): PythonService {
    const service = new PythonService(options, this.logger);
    this.services.set(options.name, service);

    if (options.autoStart !== false) {
      // Auto-start in background — don't block
      service.start().catch(() => {});
    }

    return service;
  }

  get(name: string): PythonService | undefined {
    return this.services.get(name);
  }

  /** Returns the best available Python interpreter path, or throws. */
  getPythonPath(): string {
    const python = findPython(this.logger);
    if (!python) {
      throw new Error(
        'No working Python interpreter found. Install Python 3.10+ from https://python.org ' +
        'and ensure it is on your PATH.'
      );
    }
    return python;
  }

  /** Stop all services. Call on MCP server shutdown. */
  stopAll(): void {
    for (const [name, service] of this.services) {
      this.logger.info(`Stopping ${name} backend...`);
      service.stop();
    }
  }
}
