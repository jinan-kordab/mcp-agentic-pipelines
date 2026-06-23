/**
 * Structured Logger
 *
 * JSON-formatted logging to stderr (safe for stdio MCP transport).
 * stdout is reserved for MCP protocol messages.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

const LOG_LEVEL_PRIORITY: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  tool?: string;
  message: string;
  data?: unknown;
}

export class Logger {
  constructor(private readonly minLevel: LogLevel = 'info') {}

  debug(message: string, tool?: string, data?: unknown): void {
    this.log('debug', message, tool, data);
  }

  info(message: string, tool?: string, data?: unknown): void {
    this.log('info', message, tool, data);
  }

  warn(message: string, tool?: string, data?: unknown): void {
    this.log('warn', message, tool, data);
  }

  error(message: string, tool?: string, data?: unknown): void {
    this.log('error', message, tool, data);
  }

  private log(level: LogLevel, message: string, tool?: string, data?: unknown): void {
    if (LOG_LEVEL_PRIORITY[level] < LOG_LEVEL_PRIORITY[this.minLevel]) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
    };
    if (tool) entry.tool = tool;
    if (data !== undefined) entry.data = data;

    // Write to stderr so stdout stays clean for MCP protocol
    process.stderr.write(JSON.stringify(entry) + '\n');
  }
}

/** Create a logger with the configured minimum level. */
export function createLogger(level: LogLevel): Logger {
  return new Logger(level);
}

/** Default logger instance (info level). */
export const defaultLogger = new Logger('info');
