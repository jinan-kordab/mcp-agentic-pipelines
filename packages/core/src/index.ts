/**
 * @unified-mcp/core — Barrel Export
 *
 * Shared utilities, types, config, rate limiting, logging,
 * and multi-provider LLM configuration for all integration packages.
 */

// Config
export {
  loadConfig,
  resetConfig,
  resolveLLMConfig,
  configSchema,
  type Config,
  type RawConfig,
  type ResolvedLLMConfig,
} from './config.js';

// Multi-Provider LLM
export {
  LLM_PROVIDERS,
  PROVIDER_DEFAULTS,
  OPENAI_COMPATIBLE_PROVIDERS,
  NATIVE_SDK_PROVIDERS,
  resolveLLMConfig as resolveLLM,
  listProviders,
  isValidProvider,
  llmProviderSchema,
  llmConfigSchema,
  type LLMProvider,
} from './llm-config.js';

// Errors
export {
  MCPToolError,
  ServiceUnavailableError,
  ValidationError,
  RateLimitError,
  LLMNotConfiguredError,
  AuthenticationError,
  NotFoundError,
  InternalError,
} from './errors.js';

// Validation
export {
  validateArgs,
  sanitizeString,
  validateBase64,
  clampInt,
  intSchema,
  stringSchema,
  localeSchema,
  base64Schema,
} from './validation.js';

// Logging
export {
  Logger,
  createLogger,
  defaultLogger,
  type LogLevel,
  type LogEntry,
} from './logging.js';

// Rate Limiting
export {
  TokenBucket,
  RateLimiter,
  createRateLimiter,
  type RateCategory,
} from './rate-limiter.js';

// Python Bridge
export {
  PythonService,
  PythonServiceManager,
  findPython,
  resetPythonCache,
  type PythonServiceOptions,
} from './python-bridge.js';

// Types
export type {
  MCPTextContent,
  MCPToolResponse,
  MCPResourceContent,
  ToolDefinition,
  ResourceDefinition,
  PromptDefinition,
  ServiceHealth,
  LLMProviderInfo,
  AudioTurn,
  ClinicalSession,
  SearchHit,
  SearchResults,
  StoredDocument,
  ChatSource,
  ChatContext,
  FactCheckVerdict,
  VerdictLabel,
  PrecisQueryResult,
  WorkOrder,
} from './types.js';
