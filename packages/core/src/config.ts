/**
 * Central Configuration Loader
 *
 * Loads and validates all environment variables using Zod schemas.
 * Re-exports the resolved LLM config for convenience.
 */

import { z } from 'zod';
import dotenv from 'dotenv';
import { resolveLLMConfig, llmProviderSchema, type ResolvedLLMConfig } from './llm-config.js';

dotenv.config();

// ── Full configuration schema ────────────────────────────────────────

export const configSchema = z.object({
  // --- LLM Defaults ---
  LLM_DEFAULT_PROVIDER: z.string().default('openai'),
  LLM_DEFAULT_API_KEY: z.string().default(''),
  LLM_DEFAULT_BASE_URL: z.string().default(''),
  LLM_DEFAULT_MODEL: z.string().default(''),

  // --- Azure ---
  AZURE_OPENAI_ENDPOINT: z.string().default(''),
  AZURE_OPENAI_API_KEY: z.string().default(''),
  AZURE_OPENAI_DEPLOYMENT: z.string().default(''),
  AZURE_OPENAI_API_VERSION: z.string().default('2024-08-01-preview'),

  // --- Anthropic ---
  ANTHROPIC_API_KEY: z.string().default(''),
  ANTHROPIC_BASE_URL: z.string().default(''),

  // --- Google ---
  GOOGLE_API_KEY: z.string().default(''),

  // --- Ollama ---
  OLLAMA_HOST: z.string().default('http://localhost:11434'),

  // --- DeepPipe ---
  DEEPPIPE_INDEX_PATH: z.string().default('./data/deeppipe.db'),
  DEEPPIPE_LLM_PROVIDER: z.string().default(''),
  DEEPPIPE_LLM_API_KEY: z.string().default(''),
  DEEPPIPE_LLM_MODEL: z.string().default(''),

  // --- Piste ---
  PISTE_API_URL: z.string().default('http://localhost:8000'),
  PISTE_LLM_PROVIDER: z.string().default(''),
  PISTE_LLM_API_KEY: z.string().default(''),
  TAVILY_API_KEY: z.string().default(''),
  SERPER_API_KEY: z.string().default(''),
  GOOGLE_CSE_API_KEY: z.string().default(''),
  GOOGLE_CSE_ID: z.string().default(''),

  // --- Precis ---
  PRECIS_API_URL: z.string().default('http://localhost:8001'),
  PRECIS_LLM_PROVIDER: z.string().default(''),
  PRECIS_LLM_API_KEY: z.string().default(''),

  // --- Clinical ---
  CLINICAL_LLM_PROVIDER: z.string().default(''),
  CLINICAL_LLM_API_KEY: z.string().default(''),
  CLINICAL_LLM_MODEL: z.string().default(''),
  CLINICAL_STT_PROVIDER: z.string().default('groq'),
  GROQ_API_KEY: z.string().default(''),
  CLINICAL_TTS_PROVIDER: z.string().default('elevenlabs'),
  ELEVENLABS_API_KEY: z.string().default(''),
  ELEVENLABS_VOICE_ID: z.string().default('21m00Tcm4TlvDq8ikWAM'),

  // --- MCP Server ---
  MCP_SERVER_NAME: z.string().default('mcp-agentic-pipelines'),
  MCP_SERVER_VERSION: z.string().default('1.0.0'),
  MCP_TRANSPORT: z.enum(['stdio', 'sse']).default('stdio'),
  MCP_SSE_PORT: z.coerce.number().int().positive().default(3100),
  MCP_LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  MCP_RATE_LIMIT_ENABLED: z.enum(['true', 'false', '1', '0']).default('false').transform(v => v === 'true' || v === '1'),
  MCP_RATE_LIMIT_MAX_RPS: z.coerce.number().int().positive().default(10),
});

export type RawConfig = z.infer<typeof configSchema>;

// ── Enriched config with resolved LLM settings ───────────────────────

export interface Config extends RawConfig {
  /** Resolved LLM config for DeepPipe (chat feature). */
  deepPipeLLM: ResolvedLLMConfig;
  /** Resolved LLM config for Piste. */
  pisteLLM: ResolvedLLMConfig;
  /** Resolved LLM config for Precis. */
  precisLLM: ResolvedLLMConfig;
  /** Resolved LLM config for Clinical Intake. */
  clinicalLLM: ResolvedLLMConfig;
}

/** Singleton config instance, loaded once at startup. */
let _config: Config | null = null;

/**
 * Load and validate configuration from environment variables.
 * Safe to call multiple times — returns cached instance after first call.
 */
export function loadConfig(): Config {
  if (_config) return _config;

  const raw = configSchema.parse(process.env);

  _config = {
    ...raw,
    deepPipeLLM: resolveLLMConfig({
      provider: raw.DEEPPIPE_LLM_PROVIDER || undefined,
      apiKey: raw.DEEPPIPE_LLM_API_KEY || undefined,
      model: raw.DEEPPIPE_LLM_MODEL || undefined,
    }),
    pisteLLM: resolveLLMConfig({
      provider: raw.PISTE_LLM_PROVIDER || undefined,
      apiKey: raw.PISTE_LLM_API_KEY || undefined,
    }),
    precisLLM: resolveLLMConfig({
      provider: raw.PRECIS_LLM_PROVIDER || undefined,
      apiKey: raw.PRECIS_LLM_API_KEY || undefined,
    }),
    clinicalLLM: resolveLLMConfig({
      provider: raw.CLINICAL_LLM_PROVIDER || undefined,
      apiKey: raw.CLINICAL_LLM_API_KEY || undefined,
      model: raw.CLINICAL_LLM_MODEL || undefined,
    }),
  };

  return _config;
}

/**
 * Reset cached config (useful in tests).
 */
export function resetConfig(): void {
  _config = null;
}

export { resolveLLMConfig, type ResolvedLLMConfig } from './llm-config.js';
