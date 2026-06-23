#!/usr/bin/env node
/**
 * MCP Agentic Pipelines — Main Entry Point
 *
 * Integrates piste, clinical-intake, precis-agentic-pipeline,
 * DeepPipe, and @kordabjinan/deeppipe into a single MCP server.
 *
 * Transport: stdio (primary) | SSE (secondary)
 * LLM: Multi-provider — OpenAI, Anthropic, Google, DeepSeek, Groq, Ollama, OpenRouter, Azure, Custom
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ListPromptsRequestSchema,
  ReadResourceRequestSchema,
  GetPromptRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import {
  loadConfig,
  createLogger,
  createRateLimiter,
  listProviders,
  PROVIDER_DEFAULTS,
  PythonServiceManager,
  LLMNotConfiguredError,
  InternalError,
  type Config,
  type Logger,
  type RateLimiter,
  type ToolDefinition,
  type ResourceDefinition,
  type PromptDefinition,
} from '@unified-mcp/core';

// ═══════════════════════════════════════════════════════════════════════
// Bootstrap
// ═══════════════════════════════════════════════════════════════════════

const config: Config = loadConfig();
const logger: Logger = createLogger(config.MCP_LOG_LEVEL);
const rateLimiter: RateLimiter = createRateLimiter(config.MCP_RATE_LIMIT_ENABLED, config.MCP_RATE_LIMIT_MAX_RPS);
const pythonManager = new PythonServiceManager(logger);

logger.info(`MCP Agentic Pipelines v${config.MCP_SERVER_VERSION} starting...`);
logger.info(`Transport: ${config.MCP_TRANSPORT}`);
logger.info(`LLM Default Provider: ${config.LLM_DEFAULT_PROVIDER}`);
logger.info(`Log level: ${config.MCP_LOG_LEVEL}`);

// ═══════════════════════════════════════════════════════════════════════
// Tool, Resource, Prompt registries
// ═══════════════════════════════════════════════════════════════════════

const tools: ToolDefinition[] = [];
const resources: ResourceDefinition[] = [];
const prompts: PromptDefinition[] = [];

// Tool handler map: toolName → handler function
const toolHandlers = new Map<string, (args: unknown) => Promise<{ content: Array<{ type: 'text'; text: string }>; isError?: boolean }>>();

// Resource handler map: uri pattern → handler
const resourceHandlers = new Map<string, (uri: string) => Promise<{ contents: Array<{ uri: string; mimeType: string; text?: string }> }>>();

// Prompt handler map: promptName → handler
const promptHandlers = new Map<string, (args?: Record<string, string>) => Promise<{ messages: Array<{ role: 'user' | 'assistant'; content: { type: 'text'; text: string } }> }>>();

// ═══════════════════════════════════════════════════════════════════════
// Built-in tools (always available)
// ═══════════════════════════════════════════════════════════════════════

// Tool: mcp_health — check server and integration status
tools.push({
  name: 'mcp_health',
  description: 'Check the health status of the MCP server and all integration packages. Returns provider configuration, available tools count, and service reachability.',
  inputSchema: { type: 'object', properties: {} },
});

toolHandlers.set('mcp_health', async () => {
  const health = {
    server: {
      name: config.MCP_SERVER_NAME,
      version: config.MCP_SERVER_VERSION,
      transport: config.MCP_TRANSPORT,
      uptime: process.uptime(),
    },
    tools: {
      total: tools.length,
      registered: Array.from(toolHandlers.keys()),
    },
    resources: resources.length,
    prompts: prompts.length,
    llm: {
      defaultProvider: config.LLM_DEFAULT_PROVIDER,
      deepPipe: {
        provider: config.deepPipeLLM.provider,
        model: config.deepPipeLLM.model,
        configured: !!config.deepPipeLLM.apiKey,
      },
      piste: {
        provider: config.pisteLLM.provider,
        model: config.pisteLLM.model,
        configured: !!config.pisteLLM.apiKey,
      },
      precis: {
        provider: config.precisLLM.provider,
        model: config.precisLLM.model,
        configured: !!config.precisLLM.apiKey,
      },
      clinical: {
        provider: config.clinicalLLM.provider,
        model: config.clinicalLLM.model,
        configured: !!config.clinicalLLM.apiKey,
      },
    },
    services: {
      piste: { configured: !!config.PISTE_API_URL },
      precis: { configured: !!config.PRECIS_API_URL },
      clinical: {
        stt: { provider: config.CLINICAL_STT_PROVIDER, configured: !!config.GROQ_API_KEY },
        tts: { provider: config.CLINICAL_TTS_PROVIDER, configured: !!config.ELEVENLABS_API_KEY },
      },
      deepPipe: { indexPath: config.DEEPPIPE_INDEX_PATH },
    },
  };

  return { content: [{ type: 'text' as const, text: JSON.stringify(health, null, 2) }] };
});

// Tool: mcp_list_providers — list all supported LLM providers with defaults
tools.push({
  name: 'mcp_list_providers',
  description: 'List all supported LLM providers with their default models and base URLs. Use this to see which providers are available for configuration.',
  inputSchema: {
    type: 'object',
    properties: {
      filter: {
        type: 'string',
        description: 'Optional: filter providers by name substring (e.g. "open" matches openai, openrouter)',
      },
    },
  },
});

toolHandlers.set('mcp_list_providers', async (args: unknown) => {
  const filter = typeof (args as any)?.filter === 'string' ? (args as any).filter.toLowerCase() : '';

  const providers = listProviders()
    .filter((p) => !filter || p.includes(filter))
    .map((provider) => {
      const defaults = PROVIDER_DEFAULTS[provider as keyof typeof PROVIDER_DEFAULTS];
      const isDefault = provider === config.LLM_DEFAULT_PROVIDER;

      // Check which components are using this provider
      const usedBy: string[] = [];
      if (config.deepPipeLLM.provider === provider) usedBy.push('deeppipe');
      if (config.pisteLLM.provider === provider) usedBy.push('piste');
      if (config.precisLLM.provider === provider) usedBy.push('precis');
      if (config.clinicalLLM.provider === provider) usedBy.push('clinical');

      return {
        provider,
        defaultModel: defaults.defaultModel,
        baseUrl: defaults.baseUrl || '(user-specified)',
        isDefaultProvider: isDefault,
        usedByComponents: usedBy.length > 0 ? usedBy : ['(none)'],
        howToConfigure: isDefault
          ? `Set LLM_DEFAULT_API_KEY in .env`
          : `Set LLM_DEFAULT_PROVIDER=${provider} and LLM_DEFAULT_API_KEY in .env, or set per-component like DEEPPIPE_LLM_PROVIDER=${provider}`,
      };
    });

  return { content: [{ type: 'text' as const, text: JSON.stringify(providers, null, 2) }] };
});

// ═══════════════════════════════════════════════════════════════════════
// Lazy-load integration packages
// ═══════════════════════════════════════════════════════════════════════

async function loadIntegrations(): Promise<void> {
  // Each integration is loaded with a 10s timeout so one can't hang the server.

  const withTimeout = async <T>(name: string, fn: () => Promise<T>): Promise<T | null> => {
    try {
      return await Promise.race([
        fn(),
        new Promise<null>((_, reject) => setTimeout(() => reject(new Error('TIMEOUT')), 10000)),
      ]);
    } catch (err: any) {
      logger.warn(`Integration skipped: ${name} — ${err?.message || err}`);
      return null;
    }
  };

  // -- DeepPipe (direct engine import) --
  await withTimeout('deeppipe', async () => {
    const { registerDeepPipe } = await import('@unified-mcp/deeppipe');
    registerDeepPipe({ config, logger, rateLimiter, tools, resources, prompts, toolHandlers, resourceHandlers, promptHandlers });
    logger.info('Integration loaded: deeppipe (engine direct)');
  });

  // -- Piste (Python bridge to DSPy pipeline) --
  await withTimeout('piste', async () => {
    const { registerPiste } = await import('@unified-mcp/piste');
    registerPiste({ config, logger, rateLimiter, tools, resources, prompts, toolHandlers, resourceHandlers, promptHandlers, pythonManager });
    logger.info('Integration loaded: piste (Python bridge)');
  });

  // -- Precis (Python bridge to RAG pipeline) --
  await withTimeout('precis', async () => {
    const { registerPrecis } = await import('@unified-mcp/precis');
    registerPrecis({ config, logger, rateLimiter, tools, resources, prompts, toolHandlers, resourceHandlers, promptHandlers, pythonManager });
    logger.info('Integration loaded: precis (Python bridge)');
  });

  // -- Clinical Intake (native TypeScript pipeline) --
  await withTimeout('clinical', async () => {
    const { registerClinical } = await import('@unified-mcp/clinical');
    registerClinical({ config, logger, rateLimiter, tools, resources, prompts, toolHandlers, resourceHandlers, promptHandlers });
    logger.info('Integration loaded: clinical (native pipeline)');
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Dependency guard — verifies setup.mjs was run before starting
// ═══════════════════════════════════════════════════════════════════════

import { existsSync } from 'fs';
import { resolve } from 'path';

function verifyDependencies(): void {
  const targetDir = resolve(process.cwd(), '.python-packages');
  if (!existsSync(targetDir)) {
    process.stderr.write('\n⚠  .python-packages/ not found.\n');
    process.stderr.write('   Run:  node setup.mjs\n\n');
    return;
  }
  // Quick file-system check — setup.mjs already verified everything
  const keyMods = ['dspy', 'litellm', 'fastapi', 'nltk'];
  const missing = keyMods.filter(m => !existsSync(resolve(targetDir, m)));
  if (missing.length > 0) {
    process.stderr.write(`\n⚠  Missing packages: ${missing.join(', ')}\n`);
    process.stderr.write('   Run:  node setup.mjs\n\n');
  }
}

// ═══════════════════════════════════════════════════════════════════════
// MCP Server Setup
// ═══════════════════════════════════════════════════════════════════════

async function main(): Promise<void> {
  verifyDependencies();

  // Load all integration packages
  await loadIntegrations();

  logger.info(`Registered ${tools.length} tools, ${resources.length} resources, ${prompts.length} prompts`);

  // Create MCP server
  const server = new Server(
    {
      name: config.MCP_SERVER_NAME,
      version: config.MCP_SERVER_VERSION,
    },
    {
      capabilities: {
        tools: {},
        resources: {},
        prompts: {},
      },
    },
  );

  // ── List Tools ─────────────────────────────────────────────────
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: tools.map((t) => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    })),
  }));

  // ── Call Tool ──────────────────────────────────────────────────
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    const handler = toolHandlers.get(name);

    if (!handler) {
      return {
        content: [{ type: 'text' as const, text: JSON.stringify({ code: 'UNKNOWN_TOOL', message: `Tool "${name}" is not registered.` }) }],
        isError: true,
      };
    }

    try {
      // Rate limit check
      rateLimiter.check(name);

      logger.debug(`Tool called: ${name}`, name);
      const result = await handler(args ?? {});
      return result;
    } catch (error: any) {
      logger.error(`Tool error: ${name}`, name, error.message);

      // If it's already an MCPToolError, use its response format
      if (error?.toMCPResponse) {
        return error.toMCPResponse();
      }

      return new InternalError(error?.message ?? String(error)).toMCPResponse();
    }
  });

  // ── List Resources ─────────────────────────────────────────────
  server.setRequestHandler(ListResourcesRequestSchema, async () => ({
    resources: resources.map((r) => ({
      uri: r.uri,
      name: r.name,
      description: r.description,
      mimeType: r.mimeType,
    })),
  }));

  // ── Read Resource ──────────────────────────────────────────────
  server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
    const { uri } = request.params;

    // Try exact match first, then pattern match
    for (const [pattern, handler] of resourceHandlers) {
      // Convert pattern like 'deeppipe://documents/{id}' to regex
      const regex = new RegExp('^' + pattern.replace(/\{(\w+)\}/g, '([^/]+)') + '$');
      if (regex.test(uri)) {
        try {
          return await handler(uri);
        } catch (error: any) {
          return {
            contents: [{
              uri,
              mimeType: 'application/json',
              text: JSON.stringify({ error: error?.message ?? 'Resource read failed' }),
            }],
          };
        }
      }
    }

    return {
      contents: [{
        uri,
        mimeType: 'application/json',
        text: JSON.stringify({ error: `Resource not found: ${uri}` }),
      }],
    };
  });

  // ── List Prompts ───────────────────────────────────────────────
  server.setRequestHandler(ListPromptsRequestSchema, async () => ({
    prompts: prompts.map((p) => ({
      name: p.name,
      description: p.description,
      arguments: p.arguments,
    })),
  }));

  // ── Get Prompt ─────────────────────────────────────────────────
  server.setRequestHandler(GetPromptRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    const handler = promptHandlers.get(name);

    if (!handler) {
      return {
        messages: [{
          role: 'user' as const,
          content: { type: 'text' as const, text: `Prompt "${name}" not found.` },
        }],
      };
    }

    try {
      return await handler(args as Record<string, string> | undefined);
    } catch (error: any) {
      return {
        messages: [{
          role: 'user' as const,
          content: { type: 'text' as const, text: `Error loading prompt: ${error?.message ?? error}` },
        }],
      };
    }
  });

  // ── Connect Transport ──────────────────────────────────────────
  if (config.MCP_TRANSPORT === 'sse') {
    // SSE transport requires a running HTTP server; for self-contained usage, default to stdio
    logger.info('SSE transport not yet configured for standalone mode. Using stdio.');
    const transport = new StdioServerTransport();
    await server.connect(transport);
    logger.info('MCP Server connected via stdio (fallback from SSE)');
  } else {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    logger.info('MCP Server connected via stdio');
  }

  logger.info('Ready for requests.');
  process.stderr.write('__MCP_READY__\n');  // signal test client / launchers
}

// ── Graceful shutdown ────────────────────────────────────────────────
process.on('SIGINT', () => {
  logger.info('Shutting down...');
  pythonManager.stopAll();
  process.exit(0);
});
process.on('SIGTERM', () => {
  logger.info('Shutting down...');
  pythonManager.stopAll();
  process.exit(0);
});

// ── Start ────────────────────────────────────────────────────────────
main().catch((err) => {
  logger.error('Fatal startup error', undefined, err);
  process.exit(1);
});
