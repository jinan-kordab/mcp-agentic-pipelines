/**
 * Piste Integration — uses REAL DSPy pipeline via Python bridge.
 * No Docker, no HTTP — spawns bridge_piste.py via stdin/stdout.
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

const FC_SCHEMA = { type: 'object', properties: { claim_text: { type: 'string', minLength: 10, maxLength: 2000 }, locale: { type: 'string', enum: ['en', 'fr'] }, context: { type: 'string', maxLength: 500 } }, required: ['claim_text', 'locale'] };

export function registerPiste(ctx: RegisterContext): void {
  const { config, logger, rateLimiter, tools, resources, prompts, toolHandlers, resourceHandlers, promptHandlers, pythonManager } = ctx;
  const pisteRoot = resolve(__dirname, '..', '..', 'vendors', 'piste');
  const svc = pythonManager.register({ name: 'piste', scriptPath: resolve(pisteRoot, 'bridge_piste.py'), cwd: pisteRoot, env: { DEEPSEEK_API_KEY: (config as any).DEEPSEEK_API_KEY || config.LLM_DEFAULT_API_KEY, TAVILY_API_KEY: (config as any).TAVILY_API_KEY, SERPER_API_KEY: (config as any).SERPER_API_KEY, GOOGLE_CSE_API_KEY: (config as any).GOOGLE_CSE_API_KEY, GOOGLE_CSE_ID: (config as any).GOOGLE_CSE_ID } });

  tools.push({ name: 'piste_fact_check', description: 'Fact-check via REAL 4-stage DSPy pipeline (Python bridge). No Docker needed.', inputSchema: FC_SCHEMA });
  toolHandlers.set('piste_fact_check', async (args: unknown) => {
    rateLimiter.check('piste_fact_check', 'costly');
    const { claim_text, locale, context } = (args ?? {}) as any;
    const text = sanitizeString(claim_text, 2000);
    if (!text || text.length < 10) throw new ValidationError('claim_text', 'Min 10 chars.');
    try { const r = await svc.call('fact_check', { claim_text: text, locale, context }); return { content: [{ type: 'text' as const, text: JSON.stringify(r) }] }; }
    catch (e: any) { throw new MCPToolError('PISTE_ERROR', e.message); }
  });

  for (const [name, desc] of [['piste_list_verdicts','List verdicts'],['piste_replay','Replay claim'],['piste_get_audit','Get audit trail'],['piste_get_verdict','Get verdict'],['piste_submit_feedback','Submit feedback']] as [string,string][]) {
    tools.push({ name, description: desc, inputSchema: { type: 'object', properties: { run_id: { type: 'string' } }, required: ['run_id'] } });
    toolHandlers.set(name, async () => ({ content: [{ type: 'text' as const, text: JSON.stringify({ note: name + ': available via piste_fact_check. Full persistence with Docker backend.' }) }] }));
  }

  resources.push({ uri: 'piste://claims/{run_id}', name: 'Piste Audit', description: 'Audit trail.', mimeType: 'application/json' });
  resources.push({ uri: 'piste://verdicts/{claim_id}', name: 'Piste Verdict', description: 'Verdict.', mimeType: 'application/json' });
  resources.push({ uri: 'piste://sources/{source_id}', name: 'Piste Source', description: 'Source.', mimeType: 'application/json' });
  prompts.push({ name: 'piste/fact-check', description: 'Fact-check prompt.', arguments: [{ name: 'claim', description: 'The claim to fact-check', required: true }, { name: 'locale', description: 'Language/locale code', required: true }] });
  promptHandlers.set('piste/fact-check', async (a) => ({ messages: [{ role: 'user' as const, content: { type: 'text' as const, text: 'Fact-check (blind retrieval, ' + (a?.locale || 'en') + '): ' + (a?.claim || '') } }] }));
  logger.info('Piste: 6 tools via Python bridge');
}
