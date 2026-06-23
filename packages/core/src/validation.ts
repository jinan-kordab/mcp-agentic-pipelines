/**
 * Runtime Input Validation
 *
 * Wraps Zod schemas for validating MCP tool input arguments.
 * Returns structured ValidationErrors on failure.
 */

import { z, ZodError } from 'zod';
import { ValidationError } from './errors.js';

/**
 * Validate tool arguments against a Zod schema.
 * Throws ValidationError with field-level detail on failure.
 *
 * @param schema - Zod schema to validate against.
 * @param args - Raw arguments from MCP tool call.
 * @param toolName - Tool name for error context.
 * @returns Parsed and typed arguments.
 */
export function validateArgs<T extends z.ZodType>(
  schema: T,
  args: unknown,
  toolName: string,
): z.infer<T> {
  try {
    return schema.parse(args ?? {});
  } catch (error) {
    if (error instanceof ZodError) {
      const messages = error.errors
        .map((e) => `${e.path.join('.') || '(root)'}: ${e.message}`)
        .join('; ');
      throw new ValidationError(toolName, messages);
    }
    throw new ValidationError(toolName, String(error));
  }
}

/**
 * Validate and sanitize a string input.
 * Strips null bytes, normalizes Unicode, trims whitespace.
 */
export function sanitizeString(input: unknown, maxLength: number = 4000): string {
  if (typeof input !== 'string') return '';
  return input
    .replace(/\u0000/g, '')          // Strip null bytes
    .normalize('NFKC')                // Unicode normalization
    .trim()
    .slice(0, maxLength);
}

/**
 * Validate that a base64 string is well-formed and within size limits.
 */
export function validateBase64(input: unknown, maxBytes: number = 10 * 1024 * 1024): { valid: true; buffer: Buffer } | { valid: false; error: string } {
  if (typeof input !== 'string') {
    return { valid: false, error: 'Input must be a base64-encoded string' };
  }

  // Estimate decoded size: base64 is ~4/3 of original
  if (input.length > maxBytes * 1.4) {
    return { valid: false, error: `Base64 input exceeds maximum size of ${Math.round(maxBytes / 1024 / 1024)} MB` };
  }

  try {
    const buffer = Buffer.from(input, 'base64');
    if (buffer.length > maxBytes) {
      return { valid: false, error: `Decoded data exceeds maximum size of ${Math.round(maxBytes / 1024 / 1024)} MB` };
    }
    if (buffer.length === 0 && input.length > 0) {
      return { valid: false, error: 'Invalid base64 encoding' };
    }
    return { valid: true, buffer };
  } catch {
    return { valid: false, error: 'Invalid base64 encoding' };
  }
}

/**
 * Clamp an integer value within [min, max], with a fallback for NaN.
 */
export function clampInt(raw: unknown, fallback: number, min: number, max: number): number {
  const n = Number(raw ?? fallback);
  if (Number.isNaN(n)) return fallback;
  return Math.min(Math.max(Math.floor(n), min), max);
}

// ── Common reusable Zod schemas ──────────────────────────────────────

/** Positive integer with bounds. */
export const intSchema = (min: number, max: number, fallback: number) =>
  z.coerce.number().int().min(min).max(max).default(fallback);

/** Non-empty string with max length. */
export const stringSchema = (maxLength: number = 4000) =>
  z.string().min(1).max(maxLength).transform((s) => sanitizeString(s, maxLength));

/** Language locale enum. */
export const localeSchema = z.enum(['en', 'fr']);

/** Base64-encoded data. */
export const base64Schema = z.string().min(1);
