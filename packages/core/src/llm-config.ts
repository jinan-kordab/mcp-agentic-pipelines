/**
 * Multi-Provider LLM Configuration
 *
 * Supports: OpenAI, Anthropic, Google Gemini, DeepSeek, Groq,
 *           Ollama, OpenRouter, Azure OpenAI, and any OpenAI-compatible custom endpoint.
 *
 * Each integration package can use a different provider or inherit the default.
 * Environment variables follow a layered pattern:
 *   1. Per-component override (e.g. DEEPPIPE_LLM_PROVIDER)
 *   2. Default LLM config (LLM_DEFAULT_PROVIDER)
 *   3. Hard-coded fallback (openai)
 */

import { z } from 'zod';

// ── Provider definitions ─────────────────────────────────────────────

export const LLM_PROVIDERS = [
  'openai',
  'anthropic',
  'google',
  'deepseek',
  'groq',
  'ollama',
  'openrouter',
  'azure',
  'custom',
] as const;

export type LLMProvider = (typeof LLM_PROVIDERS)[number];

/** Auto-configuration per provider — base URL and default model. */
export const PROVIDER_DEFAULTS: Record<LLMProvider, { baseUrl: string; defaultModel: string }> = {
  openai:     { baseUrl: 'https://api.openai.com/v1',                        defaultModel: 'gpt-4o-mini' },
  anthropic:  { baseUrl: 'https://api.anthropic.com/v1',                      defaultModel: 'claude-3-haiku-20240307' },
  google:     { baseUrl: 'https://generativelanguage.googleapis.com/v1beta',  defaultModel: 'gemini-2.0-flash' },
  deepseek:   { baseUrl: 'https://api.deepseek.com',                          defaultModel: 'deepseek-chat' },
  groq:       { baseUrl: 'https://api.groq.com/openai/v1',                    defaultModel: 'llama-3.3-70b-versatile' },
  ollama:     { baseUrl: 'http://localhost:11434/v1',                         defaultModel: 'llama3.2' },
  openrouter: { baseUrl: 'https://openrouter.ai/api/v1',                      defaultModel: 'openai/gpt-4o-mini' },
  azure:      { baseUrl: '',                                                  defaultModel: 'gpt-4o-mini' },
  custom:     { baseUrl: '',                                                  defaultModel: '' },
};

/** Which providers use the OpenAI-compatible chat completions API? */
export const OPENAI_COMPATIBLE_PROVIDERS: Set<LLMProvider> = new Set([
  'openai', 'deepseek', 'groq', 'ollama', 'openrouter', 'azure', 'custom',
]);

/** Providers that use native SDKs (not OpenAI-compatible). */
export const NATIVE_SDK_PROVIDERS: Set<LLMProvider> = new Set([
  'anthropic', 'google',
]);

// ── Resolved LLM config for a single component ───────────────────────

export interface ResolvedLLMConfig {
  provider: LLMProvider;
  apiKey: string;
  baseUrl: string;
  model: string;
  /** True if this endpoint uses OpenAI-compatible chat completions. */
  isOpenAICompatible: boolean;
  /** Azure-specific fields (only set when provider=azure). */
  azure?: {
    endpoint: string;
    apiVersion: string;
    deployment: string;
  };
  /** Anthropic-specific (only set when provider=anthropic). */
  anthropic?: {
    baseUrl: string;
  };
  /** Google-specific (only set when provider=google). */
  google?: {
    apiKey: string;
  };
}

// ── Resolution logic ─────────────────────────────────────────────────

/**
 * Resolve the effective LLM configuration for a component.
 *
 * Layering (highest to lowest priority):
 *   1. Per-component env vars (e.g. DEEPPIPE_LLM_PROVIDER, DEEPPIPE_LLM_API_KEY)
 *   2. Default LLM env vars (LLM_DEFAULT_PROVIDER, LLM_DEFAULT_API_KEY)
 *   3. Hard-coded fallback (openai, empty key)
 *
 * @param overrides - Per-component environment variable values.
 */
export function resolveLLMConfig(overrides?: {
  provider?: string;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
}): ResolvedLLMConfig {
  // ── Determine provider ──────────────────────────────────
  const rawProvider = (
    overrides?.provider ||
    process.env.LLM_DEFAULT_PROVIDER ||
    'openai'
  ).toLowerCase().trim();

  const provider: LLMProvider = LLM_PROVIDERS.includes(rawProvider as LLMProvider)
    ? (rawProvider as LLMProvider)
    : (rawProvider === 'gemini' ? 'google' : 'custom');

  // ── Resolve base URL ────────────────────────────────────
  const defaults = PROVIDER_DEFAULTS[provider];
  let baseUrl = '';

  if (provider === 'azure') {
    baseUrl = process.env.AZURE_OPENAI_ENDPOINT || '';
  } else if (provider === 'ollama') {
    baseUrl = process.env.OLLAMA_HOST || defaults.baseUrl;
  } else if (provider === 'anthropic') {
    baseUrl = process.env.ANTHROPIC_BASE_URL || defaults.baseUrl;
  } else if (provider === 'google') {
    baseUrl = defaults.baseUrl;
  } else if (provider === 'custom') {
    baseUrl = overrides?.baseUrl || process.env.LLM_DEFAULT_BASE_URL || '';
  } else {
    baseUrl = overrides?.baseUrl || process.env.LLM_DEFAULT_BASE_URL || defaults.baseUrl;
  }
  // Strip trailing slash
  baseUrl = baseUrl.replace(/\/+$/, '');

  // ── Resolve API key ─────────────────────────────────────
  let apiKey = '';
  if (provider === 'anthropic') {
    apiKey = overrides?.apiKey || process.env.ANTHROPIC_API_KEY || process.env.LLM_DEFAULT_API_KEY || '';
  } else if (provider === 'google') {
    apiKey = overrides?.apiKey || process.env.GOOGLE_API_KEY || process.env.LLM_DEFAULT_API_KEY || '';
  } else if (provider === 'azure') {
    apiKey = process.env.AZURE_OPENAI_API_KEY || process.env.LLM_DEFAULT_API_KEY || '';
  } else {
    apiKey = overrides?.apiKey || process.env.LLM_DEFAULT_API_KEY || '';
  }

  // ── Resolve model ───────────────────────────────────────
  let model = '';
  if (provider === 'azure') {
    model = process.env.AZURE_OPENAI_DEPLOYMENT || defaults.defaultModel;
  } else if (provider === 'custom') {
    model = overrides?.model || process.env.LLM_DEFAULT_MODEL || '';
  } else {
    model = overrides?.model || process.env.LLM_DEFAULT_MODEL || defaults.defaultModel;
  }

  // ── Build result ────────────────────────────────────────
  const result: ResolvedLLMConfig = {
    provider,
    apiKey,
    baseUrl,
    model,
    isOpenAICompatible: OPENAI_COMPATIBLE_PROVIDERS.has(provider),
  };

  if (provider === 'azure') {
    result.azure = {
      endpoint: baseUrl,
      apiVersion: process.env.AZURE_OPENAI_API_VERSION || '2024-08-01-preview',
      deployment: model,
    };
  }
  if (provider === 'anthropic') {
    result.anthropic = { baseUrl };
  }
  if (provider === 'google') {
    result.google = { apiKey };
  }

  return result;
}

/**
 * Returns human-readable summary of all available LLM providers.
 * Useful for MCP prompts and help text.
 */
export function listProviders(): string[] {
  return [...LLM_PROVIDERS];
}

/**
 * Validate that a given provider string is recognized.
 */
export function isValidProvider(raw: string): raw is LLMProvider {
  return LLM_PROVIDERS.includes(raw.toLowerCase().trim() as LLMProvider);
}

// ── Zod schema for runtime validation ─────────────────────────────────

export const llmProviderSchema = z.enum(LLM_PROVIDERS);

export const llmConfigSchema = z.object({
  provider: llmProviderSchema,
  apiKey: z.string(),
  baseUrl: z.string(),
  model: z.string(),
  isOpenAICompatible: z.boolean(),
  azure: z.object({
    endpoint: z.string(),
    apiVersion: z.string(),
    deployment: z.string(),
  }).optional(),
  anthropic: z.object({
    baseUrl: z.string(),
  }).optional(),
  google: z.object({
    apiKey: z.string(),
  }).optional(),
});
