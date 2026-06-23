/**
 * DeepPipe Integration for Unified MCP Server
 *
 * Direct engine import — no HTTP layer.
 * Exposes 10 tools, 2 resources, 1 prompt.
 *
 * Multi-LLM support: uses config.deepPipeLLM for the chat feature.
 * Any OpenAI-compatible provider (openai, deepseek, groq, ollama, openrouter, azure, custom).
 */

import type { Pipeline } from '@kordabjinan/deeppipe';
import { openPipeline } from '@kordabjinan/deeppipe';
import { extname } from 'node:path';
import { readFile } from 'node:fs/promises';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

import type {
  Config,
  Logger,
  RateLimiter,
  ToolDefinition,
  ResourceDefinition,
  PromptDefinition,
  ResolvedLLMConfig,
} from '@unified-mcp/core';
import {
  MCPToolError,
  ValidationError,
  NotFoundError,
  LLMNotConfiguredError,
  clampInt,
  sanitizeString,
  validateBase64,
} from '@unified-mcp/core';

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════

export interface DeepPipeContext {
  pipeline: Pipeline;
  config: Config;
  logger: Logger;
  rateLimiter: RateLimiter;
}

export interface RegisterContext {
  config: Config;
  logger: Logger;
  rateLimiter: RateLimiter;
  tools: ToolDefinition[];
  resources: ResourceDefinition[];
  prompts: PromptDefinition[];
  toolHandlers: Map<string, (args: unknown) => Promise<{ content: Array<{ type: 'text'; text: string }>; isError?: boolean }>>;
  resourceHandlers: Map<string, (uri: string) => Promise<{ contents: Array<{ uri: string; mimeType: string; text?: string }> }>>;
  promptHandlers: Map<string, (args?: Record<string, string>) => Promise<{ messages: Array<{ role: 'user' | 'assistant'; content: { type: 'text'; text: string } }> }>>;
}

// ═══════════════════════════════════════════════════════════════════════
// Pipeline Lifecycle
// ═══════════════════════════════════════════════════════════════════════

let _pipeline: Pipeline | null = null;

function getPipeline(config: Config, logger: Logger): Pipeline {
  if (_pipeline) return _pipeline;

  const indexPath = config.DEEPPIPE_INDEX_PATH;
  const dir = dirname(indexPath);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }

  const opened = openPipeline({ location: indexPath });
  if (!opened.ok) {
    throw new MCPToolError('ENGINE_ERROR', `Failed to open DeepPipe index: ${opened.error.message}`);
  }

  _pipeline = opened.value;
  logger.info(`DeepPipe index opened: ${indexPath} (${_pipeline.documentCount()} documents)`);
  return _pipeline;
}

// ═══════════════════════════════════════════════════════════════════════
// Tool Schemas
// ═══════════════════════════════════════════════════════════════════════

const SEARCH_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    query: { type: 'string', description: 'Search query. Supports field:value, "exact phrases", prefix*, AND/OR/NOT operators.' },
    limit: { type: 'integer', minimum: 1, maximum: 200, default: 20, description: 'Max results to return.' },
    offset: { type: 'integer', minimum: 0, default: 0, description: 'Pagination offset.' },
    snippets: { type: 'boolean', default: false, description: 'Include highlighted text snippets around matches.' },
  },
  required: ['query'],
};

const INGEST_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    data: { type: 'string', description: 'Base64-encoded document bytes.' },
    source: { type: 'string', description: 'Original filename or source identifier (e.g. "contract.pdf").' },
  },
  required: ['data'],
};

const INGEST_FILE_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    path: { type: 'string', description: 'Absolute or relative path to a document file on disk.' },
  },
  required: ['path'],
};

const CHAT_CONTEXT_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    question: { type: 'string', description: 'Natural language question to answer from your documents.' },
    history: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          role: { type: 'string', enum: ['user', 'assistant'] },
          content: { type: 'string' },
        },
      },
      maxItems: 8,
      description: 'Prior conversation turns for context.',
    },
    maxSources: { type: 'integer', minimum: 1, maximum: 20, default: 5 },
    maxTokens: { type: 'integer', minimum: 100, maximum: 8000, default: 3000 },
  },
  required: ['question'],
};

const LIST_DOCUMENTS_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    limit: { type: 'integer', minimum: 1, maximum: 500, default: 50 },
    offset: { type: 'integer', minimum: 0, default: 0 },
  },
};

const GET_DOCUMENT_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'integer', minimum: 1, description: 'Document ID (from list_documents or ingest results).' },
  },
  required: ['id'],
};

const REMOVE_DOCUMENT_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'integer', minimum: 1, description: 'Document ID to remove from the index.' },
  },
  required: ['id'],
};

const STATS_INPUT_SCHEMA = {
  type: 'object',
  properties: {},
};

// ═══════════════════════════════════════════════════════════════════════
// Tool Handlers
// ═══════════════════════════════════════════════════════════════════════

function handleSearch(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { query, limit, offset, snippets } = args as any;
    const q = sanitizeString(query, 1000);
    if (!q) throw new ValidationError('query', 'Search query must not be empty.');

    const result = ctx.pipeline.search(q, {
      limit: clampInt(limit, 20, 1, 200),
      offset: clampInt(offset, 0, 0, 1_000_000),
      snippets: snippets === true,
    });

    if (!result.ok) {
      return { content: [{ type: 'text' as const, text: JSON.stringify({ error: result.error.toJSON() }) }], isError: true };
    }

    return { content: [{ type: 'text' as const, text: JSON.stringify(result.value) }] };
  };
}

function handleIngest(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { data, source } = args as any;
    const validation = validateBase64(data, 64 * 1024 * 1024);
    if (!validation.valid) throw new ValidationError('data', validation.error);

    const result = ctx.pipeline.ingestBytes(validation.buffer, String(source ?? 'unnamed'));
    if (!result.ok) {
      return { content: [{ type: 'text' as const, text: JSON.stringify({ error: result.error.toJSON() }) }], isError: true };
    }

    ctx.logger.info(`Ingested: ${source || 'unnamed'} (${result.value.wordCount} words)`, 'deeppipe_ingest');
    return { content: [{ type: 'text' as const, text: JSON.stringify(result.value) }] };
  };
}

function handleIngestFile(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { path } = args as any;
    if (!path || typeof path !== 'string') throw new ValidationError('path', 'File path is required.');

    try {
      const result = await ctx.pipeline.ingestFile(path);
      if (!result.ok) {
        return { content: [{ type: 'text' as const, text: JSON.stringify({ error: result.error.toJSON() }) }], isError: true };
      }
      ctx.logger.info(`Ingested file: ${path} (${result.value.wordCount} words)`, 'deeppipe_ingest_file');
      return { content: [{ type: 'text' as const, text: JSON.stringify(result.value) }] };
    } catch (err: any) {
      throw new MCPToolError('IO_ERROR', `Cannot read file: ${err?.message ?? err}`);
    }
  };
}

function handleChatContext(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { question, history } = args as any;
    const q = sanitizeString(question, 4000);
    if (!q) throw new ValidationError('question', 'Question must not be empty.');

    const result = ctx.pipeline.chatContext(q, {
      history: Array.isArray(history) ? history.slice(-8) : [],
    });

    if (!result.ok) {
      return { content: [{ type: 'text' as const, text: JSON.stringify({ error: result.error.toJSON() }) }], isError: true };
    }

    // Include LLM provider info in the response so the client knows what model to use
    const enriched = {
      ...result.value,
      _llm: {
        provider: ctx.config.deepPipeLLM.provider,
        model: ctx.config.deepPipeLLM.model,
        configured: !!ctx.config.deepPipeLLM.apiKey,
        note: ctx.config.deepPipeLLM.apiKey
          ? `Ready for LLM chat. Send messages to ${ctx.config.deepPipeLLM.baseUrl}`
          : `No LLM configured. Set DEEPPIPE_LLM_API_KEY or LLM_DEFAULT_API_KEY to enable chat. Use extractive_answer for no-LLM fallback.`,
      },
    };

    return { content: [{ type: 'text' as const, text: JSON.stringify(enriched) }] };
  };
}

function handleExtractiveAnswer(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { question, history } = args as any;
    const q = sanitizeString(question, 4000);
    if (!q) throw new ValidationError('question', 'Question must not be empty.');

    // Import extractiveAnswer from engine
    const { extractiveAnswer } = await import('@kordabjinan/deeppipe');

    const result = ctx.pipeline.chatContext(q, {
      history: Array.isArray(history) ? history.slice(-8) : [],
    });

    if (!result.ok) {
      return { content: [{ type: 'text' as const, text: JSON.stringify({ error: result.error.toJSON() }) }], isError: true };
    }

    const answer = extractiveAnswer(result.value);
    return { content: [{ type: 'text' as const, text: JSON.stringify({ answer, sources: result.value.sources }) }] };
  };
}

function handleListDocuments(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { limit, offset } = args as any;
    const docs = ctx.pipeline.listDocuments(
      clampInt(limit, 50, 1, 500),
      clampInt(offset, 0, 0, 1_000_000),
    );
    return { content: [{ type: 'text' as const, text: JSON.stringify({ documents: docs }) }] };
  };
}

function handleGetDocument(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { id } = args as any;
    const doc = ctx.pipeline.getDocument(Number(id));
    if (!doc) throw new NotFoundError('Document', String(id));
    return { content: [{ type: 'text' as const, text: JSON.stringify({ document: doc }) }] };
  };
}

function handleGetDocumentText(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { id } = args as any;
    const text = ctx.pipeline.documentText(Number(id));
    if (text === undefined) throw new NotFoundError('Document', String(id));
    return { content: [{ type: 'text' as const, text: JSON.stringify({ text }) }] };
  };
}

function handleRemoveDocument(ctx: DeepPipeContext) {
  return async (args: unknown) => {
    const { id } = args as any;
    const result = ctx.pipeline.removeDocument(Number(id));
    if (!result.ok) {
      return { content: [{ type: 'text' as const, text: JSON.stringify({ error: result.error.toJSON() }) }], isError: true };
    }
    ctx.logger.info(`Removed document ${id}`, 'deeppipe_remove_document');
    return { content: [{ type: 'text' as const, text: JSON.stringify({ ok: true }) }] };
  };
}

function handleStats(ctx: DeepPipeContext) {
  return async (_args: unknown) => {
    const count = ctx.pipeline.documentCount();
    return { content: [{ type: 'text' as const, text: JSON.stringify({ documentCount: count, indexLocation: ctx.config.DEEPPIPE_INDEX_PATH }) }] };
  };
}

// ═══════════════════════════════════════════════════════════════════════
// Resource Handlers
// ═══════════════════════════════════════════════════════════════════════

function handleDocumentResource(ctx: DeepPipeContext) {
  return async (uri: string) => {
    const match = uri.match(/^deeppipe:\/\/documents\/(\d+)(\/text)?$/);
    if (!match) throw new NotFoundError('Resource', uri);

    const id = Number(match[1]);
    if (match[2] === '/text') {
      const text = ctx.pipeline.documentText(id);
      if (text === undefined) throw new NotFoundError('Document', String(id));
      return { contents: [{ uri, mimeType: 'text/plain', text }] };
    }

    const doc = ctx.pipeline.getDocument(id);
    if (!doc) throw new NotFoundError('Document', String(id));
    return { contents: [{ uri, mimeType: 'application/json', text: JSON.stringify(doc, null, 2) }] };
  };
}

// ═══════════════════════════════════════════════════════════════════════
// Prompt Handlers
// ═══════════════════════════════════════════════════════════════════════

function handleChatPrompt(ctx: DeepPipeContext) {
  return async (args?: Record<string, string>) => {
    const question = args?.question ?? '';

    return {
      messages: [{
        role: 'user' as const,
        content: {
          type: 'text' as const,
          text: `You are a helpful assistant answering questions about the user's document library. Every claim must include a citation to the source document. If the answer cannot be found in the provided documents, say so honestly.\n\nQuestion: ${question}\n\nUse the documents retrieved by the DeepPipe search engine to ground your answer. Cite sources by document title and passage.`,
        },
      }],
    };
  };
}

// ═══════════════════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════════════════

export function registerDeepPipe(ctx: RegisterContext): void {
  const { config, logger, rateLimiter, tools, resources, prompts, toolHandlers, resourceHandlers, promptHandlers } = ctx;

  const pipeline = getPipeline(config, logger);
  const deepPipeCtx: DeepPipeContext = { pipeline, config, logger, rateLimiter };

  // ── Tools ──────────────────────────────────────────────────────
  const toolDefs: Array<{ name: string; description: string; inputSchema: any; handler: (args: unknown) => Promise<any> }> = [
    { name: 'deeppipe_search',              description: 'Full-text search across all indexed documents using BM25 ranking (SQLite FTS5). Supports field:value, "exact phrases", prefix*, AND/OR/NOT operators. Returns ranked hits with optional snippet highlighting.', inputSchema: SEARCH_INPUT_SCHEMA, handler: handleSearch(deepPipeCtx) },
    { name: 'deeppipe_ingest',              description: 'Ingest and index a document from base64-encoded bytes. Supports PDF, DOCX, XLSX, HTML, EML, MHT, ZIP, and plain text. Documents are parsed with pure-TypeScript parsers — no external dependencies.', inputSchema: INGEST_INPUT_SCHEMA, handler: handleIngest(deepPipeCtx) },
    { name: 'deeppipe_ingest_file',         description: 'Ingest and index a document from a file path on disk. Same format support as deeppipe_ingest.', inputSchema: INGEST_FILE_INPUT_SCHEMA, handler: handleIngestFile(deepPipeCtx) },
    { name: 'deeppipe_chat_context',        description: 'Build a grounded RAG context for answering a question from your documents. Retrieves the most relevant passages and assembles a cited message sequence ready for any OpenAI-compatible LLM provider.', inputSchema: CHAT_CONTEXT_INPUT_SCHEMA, handler: handleChatContext(deepPipeCtx) },
    { name: 'deeppipe_extractive_answer',   description: 'Answer a question using extractive passage selection (no LLM required). Returns verbatim passages with citations — useful when no LLM is configured.', inputSchema: CHAT_CONTEXT_INPUT_SCHEMA, handler: handleExtractiveAnswer(deepPipeCtx) },
    { name: 'deeppipe_list_documents',      description: 'List all documents currently indexed in DeepPipe with metadata (ID, source filename, word count, indexed date).', inputSchema: LIST_DOCUMENTS_INPUT_SCHEMA, handler: handleListDocuments(deepPipeCtx) },
    { name: 'deeppipe_get_document',        description: 'Get metadata for a specific indexed document by ID.', inputSchema: GET_DOCUMENT_INPUT_SCHEMA, handler: handleGetDocument(deepPipeCtx) },
    { name: 'deeppipe_get_text',            description: 'Get the full reconstructed text of an indexed document by ID.', inputSchema: GET_DOCUMENT_INPUT_SCHEMA, handler: handleGetDocumentText(deepPipeCtx) },
    { name: 'deeppipe_remove_document',     description: 'Remove a document and all its chunks from the index by ID.', inputSchema: REMOVE_DOCUMENT_INPUT_SCHEMA, handler: handleRemoveDocument(deepPipeCtx) },
    { name: 'deeppipe_stats',               description: 'Get stats about the DeepPipe index: total document count and index file location.', inputSchema: STATS_INPUT_SCHEMA, handler: handleStats(deepPipeCtx) },
  ];

  for (const def of toolDefs) {
    tools.push({ name: def.name, description: def.description, inputSchema: def.inputSchema });
    toolHandlers.set(def.name, def.handler);
  }

  // ── Resources ──────────────────────────────────────────────────
  resources.push({
    uri: 'deeppipe://documents/{id}',
    name: 'DeepPipe Document Metadata',
    description: 'Metadata for an indexed document (source, word count, format, indexed date).',
    mimeType: 'application/json',
  });
  resources.push({
    uri: 'deeppipe://documents/{id}/text',
    name: 'DeepPipe Document Text',
    description: 'Reconstructed full text of an indexed document.',
    mimeType: 'text/plain',
  });
  resourceHandlers.set('deeppipe://documents/{id}', handleDocumentResource(deepPipeCtx));

  // ── Prompts ────────────────────────────────────────────────────
  prompts.push({
    name: 'deeppipe/chat',
    description: 'Grounded RAG chat prompt template. Answer questions using only cited evidence from your indexed document library.',
    arguments: [
      { name: 'question', description: 'The question to answer from your documents.', required: true },
    ],
  });
  promptHandlers.set('deeppipe/chat', handleChatPrompt(deepPipeCtx));

  logger.info(`DeepPipe: ${toolDefs.length} tools, 2 resources, 1 prompt registered`);
}
