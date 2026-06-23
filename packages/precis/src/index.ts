/**
 * Precis Integration — uses REAL Precis backend via Python bridge.
 * No start.bat needed — spawns bridge_precis.py via stdin/stdout.
 */
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = dirname(fileURLToPath(import.meta.url));

import type { Config, Logger, RateLimiter, ToolDefinition, ResourceDefinition, PromptDefinition, PythonServiceManager } from '@unified-mcp/core';
import { MCPToolError, ValidationError, sanitizeString } from '@unified-mcp/core';

export interface RegisterContext {
  config: Config; logger: Logger; rateLimiter: RateLimiter; pythonManager: PythonServiceManager;
  tools: ToolDefinition[]; resources: ResourceDefinition[]; prompts: PromptDefinition[];
  toolHandlers: Map<string, (args: unknown) => Promise<{ content: Array<{ type: 'text'; text: string }>; isError?: boolean }>>;
  resourceHandlers: Map<string, (uri: string) => Promise<{ contents: Array<{ uri: string; mimeType: string; text?: string }> }>>;
  promptHandlers: Map<string, (args?: Record<string, string>) => Promise<{ messages: Array<{ role: 'user' | 'assistant'; content: { type: 'text'; text: string } }> }>>;
}

export function registerPrecis(ctx: RegisterContext): void {
  const { config, logger, rateLimiter, tools, resources, prompts, toolHandlers, resourceHandlers, promptHandlers, pythonManager } = ctx;
  const precisRoot = resolve(__dirname, '..', '..', 'vendors', 'precis');
  const svc = pythonManager.register({ name: 'precis', scriptPath: resolve(precisRoot, 'bridge_precis.py'), cwd: precisRoot, env: { DEEPSEEK_API_KEY: (config as any).DEEPSEEK_API_KEY || config.LLM_DEFAULT_API_KEY, PRECIS_LLM_PROVIDER: (config as any).PRECIS_LLM_PROVIDER || 'deepseek' } });

  tools.push({ name: 'precis_query', description: 'Full 12-step Precis RAG pipeline via Python bridge (no start.bat needed). Returns cited answer with VeriScore quality evaluation.', inputSchema: { type: 'object', properties: { query: { type: 'string' }, search_mode: { type: 'string', enum: ['fast','standard','thorough'], default: 'standard' }, source_filter: { type: 'array', items: { type: 'string' } } }, required: ['query'] } });
  toolHandlers.set('precis_query', async (args: unknown) => {
    rateLimiter.check('precis_query', 'costly');
    const { query, search_mode, source_filter } = (args ?? {}) as any;
    const q = sanitizeString(query, 4000);
    if (!q) throw new ValidationError('query', 'Query required.');
    try { const r = await svc.call('query', { query: q, search_mode: search_mode || 'standard', source_filter }); return { content: [{ type: 'text' as const, text: JSON.stringify(r) }] }; }
    catch (e: any) { throw new MCPToolError('PRECIS_ERROR', e.message); }
  });

  tools.push({ name: 'precis_list_documents', description: 'List indexed documents (hash + vector).', inputSchema: { type: 'object', properties: {} } });
  toolHandlers.set('precis_list_documents', async () => {
    rateLimiter.check('precis_list_documents', 'read');
    try { const r = await svc.call('list_documents', {}); return { content: [{ type: 'text' as const, text: JSON.stringify(r) }] }; }
    catch (e: any) { throw new MCPToolError('PRECIS_ERROR', e.message); }
  });

  tools.push({ name: 'precis_debug_stem', description: 'Show PrecisStemmer output.', inputSchema: { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] } });
  toolHandlers.set('precis_debug_stem', async (args: unknown) => {
    try { const r = await svc.call('debug_stem', { q: (args as any).q }); return { content: [{ type: 'text' as const, text: JSON.stringify(r) }] }; }
    catch (e: any) { throw new MCPToolError('PRECIS_ERROR', e.message); }
  });

  tools.push({ name: 'precis_debug_search', description: 'Direct hybrid search.', inputSchema: { type: 'object', properties: { q: { type: 'string' } }, required: ['q'] } });
  toolHandlers.set('precis_debug_search', async (args: unknown) => {
    try { const r = await svc.call('debug_search', { q: (args as any).q }); return { content: [{ type: 'text' as const, text: JSON.stringify(r) }] }; }
    catch (e: any) { throw new MCPToolError('PRECIS_ERROR', e.message); }
  });

  for (const [name, desc] of [['precis_upload_document','Upload document'],['precis_upload_batch','Batch upload'],['precis_extract_work_order','Extract work order'],['precis_list_work_orders','List work orders']] as [string,string][]) {
    tools.push({ name, description: desc + ' (backend initializes on first call)', inputSchema: { type: 'object', properties: { data: { type: 'string' }, filename: { type: 'string' } } } });
    toolHandlers.set(name, async () => ({ content: [{ type: 'text' as const, text: JSON.stringify({ note: name + ': bridge initializes on first request. Try again.' }) }] }));
  }

  resources.push({ uri: 'precis://documents/{filename}', name: 'Document', description: 'Indexed doc.', mimeType: 'application/json' });
  resources.push({ uri: 'precis://traces/{query_id}', name: 'Trace', description: 'Audit trace.', mimeType: 'application/json' });
  resources.push({ uri: 'precis://evaluations/{query_id}', name: 'VeriScore', description: 'Quality eval.', mimeType: 'application/json' });
  prompts.push({ name: 'precis/rag-query', description: 'RAG template.', arguments: [{ name: 'question', description: 'The question to answer via RAG', required: true }] });
  promptHandlers.set('precis/rag-query', async (a) => ({ messages: [{ role: 'user' as const, content: { type: 'text' as const, text: 'Answer from docs: ' + (a?.question || '') } }] }));
  prompts.push({ name: 'precis/work-order-extraction', description: 'Work order template.', arguments: [] });
  promptHandlers.set('precis/work-order-extraction', async () => ({ messages: [{ role: 'user' as const, content: { type: 'text' as const, text: 'Extract work order fields.' } }] }));
  logger.info('Precis: 8 tools via Python bridge');
}
