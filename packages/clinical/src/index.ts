/**
 * Clinical Intake Integration for Unified MCP Server
 *
 * USES THE REAL clinical-intake/pipeline.mjs — NOT A REWRITE.
 * Imports directly from the cloned clinical-intake repo.
 *
 * Multi-provider LLM via the real pipeline's clinicalChat().
 * STT: Groq (whisper-large-v3) via the real pipeline
 * TTS: ElevenLabs via the real pipeline
 *
 * Privacy: Anonymous by design.
 */

import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);

import type { Config, Logger, RateLimiter, ToolDefinition, ResourceDefinition, PromptDefinition } from '@unified-mcp/core';
import { ValidationError, validateBase64 } from '@unified-mcp/core';

// ═══════════════════════════════════════════════════════════════════════
// Dynamic import of the REAL clinical-intake pipeline
// ═══════════════════════════════════════════════════════════════════════

const CLINICAL_PIPELINE_PATH = '../../../vendors/clinical-intake/pipeline.mjs';
let _pipelineModule: any = null;
async function getPipeline() {
  if (!_pipelineModule) { _pipelineModule = await import(CLINICAL_PIPELINE_PATH); }
  return _pipelineModule;
}

// ═══════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════

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
// Tool Schemas
// ═══════════════════════════════════════════════════════════════════════

const SESSION_START_SCHEMA = {
  type: 'object',
  properties: {
    patient_name: { type: 'string', description: 'Patient first name (not persisted).', maxLength: 100 },
    lang: { type: 'string', enum: ['en', 'fr'], default: 'fr', description: 'Interview language.' },
  },
};

const PROCESS_AUDIO_SCHEMA = {
  type: 'object',
  properties: {
    session_id: { type: 'string', description: 'Session ID from clinical_start_session.' },
    audio_data: { type: 'string', description: 'Base64-encoded audio (webm, mp3, wav).' },
  },
  required: ['session_id', 'audio_data'],
};

const GENERATE_PODCAST_SCHEMA = {
  type: 'object',
  properties: {
    session_id: { type: 'string', description: 'Session ID to compile into podcast MP3.' },
  },
  required: ['session_id'],
};

const LIST_SESSIONS_SCHEMA = { type: 'object', properties: {} };

const GET_SESSION_SCHEMA = {
  type: 'object',
  properties: {
    session_id: { type: 'string', description: 'Session ID to retrieve.' },
  },
  required: ['session_id'],
};

// ═══════════════════════════════════════════════════════════════════════
// Registration — ALL logic from clinical-intake/pipeline.mjs
// ═══════════════════════════════════════════════════════════════════════

export function registerClinical(ctx: RegisterContext): void {
  const { config, logger, rateLimiter, tools, resources, prompts, toolHandlers, resourceHandlers, promptHandlers } = ctx;

  const llmConfig = config.clinicalLLM;
  logger.info(`Clinical LLM: provider=${llmConfig.provider}, model=${llmConfig.model}`);
  logger.info('Using REAL pipeline from clinical-intake/pipeline.mjs');

  // ── clinical_start_session ─────────────────────────────────────
  tools.push({
    name: 'clinical_start_session',
    description: 'Start a new anonymous clinical intake session. Uses the real clinical-intake pipeline: bilingual questionnaires (EN: Health Canada, FR: RAMQ), Groq STT, multi-provider LLM, ElevenLabs TTS. Returns session ID, greeting text, and greeting audio.',
    inputSchema: SESSION_START_SCHEMA,
  });

  toolHandlers.set('clinical_start_session', async (args: unknown) => {
    rateLimiter.check('clinical_start_session', 'costly');
    const pipe = await getPipeline();

    const { patient_name, lang } = (args ?? {}) as any;
    const language = lang === 'en' ? 'en' : 'fr';
    const sid = pipe.generateSessionId();
    pipe.getSession(sid, patient_name || '', language);
    const session = pipe.getSession(sid);

    let greetingAudioBase64: string | null = null;
    try {
      const audioBuffer = await pipe.synthesize(session.primingText, config.ELEVENLABS_API_KEY, config.ELEVENLABS_VOICE_ID);
      greetingAudioBase64 = audioBuffer.toString('base64');
    } catch (err: any) {
      logger.warn(`TTS unavailable: ${err.message}`, 'clinical_start_session');
    }

    logger.info(`Clinical session: ${sid} (${language})`, 'clinical_start_session');
    return { content: [{ type: 'text' as const, text: JSON.stringify({
      session_id: sid, greeting_text: session.primingText, greeting_audio_base64: greetingAudioBase64,
      lang: session.lang, total_questions: session.questions.length,
      _llm: { provider: llmConfig.provider, model: llmConfig.model, configured: !!llmConfig.apiKey },
    }) }] };
  });

  // ── clinical_process_audio ─────────────────────────────────────
  tools.push({
    name: 'clinical_process_audio',
    description: 'Process patient audio through the REAL clinical pipeline: Groq STT transcription → Multi-provider LLM clinical reasoning → ElevenLabs TTS voice reply. Returns transcript, LLM response, and audio reply.',
    inputSchema: PROCESS_AUDIO_SCHEMA,
  });

  toolHandlers.set('clinical_process_audio', async (args: unknown) => {
    rateLimiter.check('clinical_process_audio', 'costly');
    const pipe = await getPipeline();

    const { session_id, audio_data } = (args ?? {}) as any;
    const validation = validateBase64(audio_data, 10 * 1024 * 1024);
    if (!validation.valid) throw new ValidationError('audio_data', validation.error);

    const session = pipe.getSession(session_id);
    if (!session?.primingText) throw new ValidationError('session_id', `Session "${session_id}" not found.`);

    const result = await pipe.processClinicalTurn(session_id, validation.buffer, {
      groqApiKey: config.GROQ_API_KEY,
      llmApiKey: llmConfig.apiKey,
      llmBaseUrl: llmConfig.baseUrl,
      llmModel: llmConfig.model,
      elevenlabsApiKey: config.ELEVENLABS_API_KEY,
      elevenlabsVoiceId: config.ELEVENLABS_VOICE_ID,
    });

    logger.info(`Turn ${result.turn_number}: "${result.user_text.slice(0, 60)}..."`, 'clinical_process_audio');
    return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
  });

  // ── clinical_generate_podcast ──────────────────────────────────
  tools.push({
    name: 'clinical_generate_podcast',
    description: 'Compile a complete clinical encounter into an MP3 podcast using the REAL pipeline. All dialog turns + greeting combined.',
    inputSchema: GENERATE_PODCAST_SCHEMA,
  });

  toolHandlers.set('clinical_generate_podcast', async (args: unknown) => {
    rateLimiter.check('clinical_generate_podcast', 'costly');
    const pipe = await getPipeline();

    const { session_id } = (args ?? {}) as any;
    const session = pipe.getSession(session_id);
    if (!session?.primingText) throw new ValidationError('session_id', `Session "${session_id}" not found.`);
    if (session.dialog.length === 0) {
      return { content: [{ type: 'text' as const, text: JSON.stringify({ error: 'No conversation to render.' }) }], isError: true };
    }

    const podcast = await pipe.buildPodcast(session_id, {
      elevenlabsApiKey: config.ELEVENLABS_API_KEY,
      elevenlabsVoiceId: config.ELEVENLABS_VOICE_ID,
    });

    logger.info(`Podcast: ${session_id} (${(podcast.length / 1024).toFixed(1)} KB)`, 'clinical_generate_podcast');
    return { content: [{ type: 'text' as const, text: JSON.stringify({
      podcast_base64: podcast.toString('base64'), session_id, turns: session.dialog.length + 1,
      duration_seconds: Math.round(session.dialog.length * 15), file_size_bytes: podcast.length,
      patient_name: session.patientName || '(anonymous)', lang: session.lang,
    }) }] };
  });

  // ── clinical_list_sessions ─────────────────────────────────────
  tools.push({
    name: 'clinical_list_sessions',
    description: 'List all clinical intake sessions from the real pipeline. Anonymous by design.',
    inputSchema: LIST_SESSIONS_SCHEMA,
  });
  toolHandlers.set('clinical_list_sessions', async () => {
    rateLimiter.check('clinical_list_sessions', 'read');
    const pipe = await getPipeline();
    return { content: [{ type: 'text' as const, text: JSON.stringify({ sessions: pipe.listSessions() }) }] };
  });

  // ── clinical_get_session ───────────────────────────────────────
  tools.push({
    name: 'clinical_get_session',
    description: 'Get full session detail with dialog turns (text only, audio excluded).',
    inputSchema: GET_SESSION_SCHEMA,
  });
  toolHandlers.set('clinical_get_session', async (args: unknown) => {
    rateLimiter.check('clinical_get_session', 'read');
    const pipe = await getPipeline();
    const { session_id } = (args ?? {}) as any;
    const session = pipe.getSession(session_id);
    if (!session?.primingText) throw new ValidationError('session_id', `Session "${session_id}" not found.`);
    return { content: [{ type: 'text' as const, text: JSON.stringify({
      session_id, lang: session.lang,
      turns: session.dialog.map((t: any) => ({ role: t.role, text: t.text })),
      is_complete: session.isComplete, questions_total: session.questions.length, questions_asked: session.questionIndex,
      created_at: session.createdAt,
    }) }] };
  });

  // ── Resources ──────────────────────────────────────────────────
  resources.push({
    uri: 'clinical://questions/{lang}',
    name: 'Clinical Questionnaire',
    description: 'Clinical intake questions from the real pipeline (clinical-intake/questions/).',
    mimeType: 'application/json',
  });
  resourceHandlers.set('clinical://questions/{lang}', async (uri: string) => {
    const pipe = await getPipeline();
    const match = uri.match(/^clinical:\/\/questions\/(en|fr)$/);
    if (!match) throw new ValidationError('uri', `Invalid: ${uri}`);
    const lang = match[1] as 'en' | 'fr';
    const questions = lang === 'fr' ? pipe.QUESTIONS_FR : pipe.QUESTIONS_EN;
    return { contents: [{ uri, mimeType: 'application/json', text: JSON.stringify({ lang, questions, count: questions.length }) }] };
  });

  // ── Prompts ────────────────────────────────────────────────────
  prompts.push({
    name: 'clinical/intake-en',
    description: 'English clinical intake prompt from the real pipeline. Includes all 8 Health Canada questions and clinical safety rules.',
    arguments: [{ name: 'patient_name', description: 'Patient first name.', required: false }],
  });
  promptHandlers.set('clinical/intake-en', async (args?: Record<string, string>) => {
    const pipe = await getPipeline();
    return { messages: [{ role: 'user' as const, content: { type: 'text' as const, text: pipe.buildSystemPrompt('en', args?.patient_name || 'sir/madam') } }] };
  });

  prompts.push({
    name: 'clinical/intake-fr',
    description: 'French clinical intake prompt from the real pipeline (RAMQ). Includes all 8 questions and clinical safety rules.',
    arguments: [{ name: 'patient_name', description: 'Patient first name.', required: false }],
  });
  promptHandlers.set('clinical/intake-fr', async (args?: Record<string, string>) => {
    const pipe = await getPipeline();
    return { messages: [{ role: 'user' as const, content: { type: 'text' as const, text: pipe.buildSystemPrompt('fr', args?.patient_name || 'M./Mme') } }] };
  });

  logger.info('Clinical: 5 tools, 1 resource, 2 prompts — using REAL clinical-intake/pipeline.mjs');
}
