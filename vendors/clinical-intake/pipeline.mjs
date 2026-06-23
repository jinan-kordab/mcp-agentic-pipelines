/**
 * Clinical Intake — Core Pipeline Module
 * Extracted from server.js for reuse by both Express HTTP server and MCP server.
 *
 * Pure functions for:
 *   - Loading clinical questionnaires (EN/FR)
 *   - Building system prompts with clinical safety rules
 *   - Groq STT (speech-to-text)
 *   - Multi-provider LLM chat (OpenAI-compatible)
 *   - ElevenLabs TTS (text-to-speech)
 *   - In-memory session management
 *
 * Privacy: Anonymous by design. No patient identifiers persisted.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { Groq } from 'groq-sdk';
import OpenAI from 'openai';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ═══════════════════════════════════════════════════════════════════════
// Clinical Questionnaires
// ═══════════════════════════════════════════════════════════════════════

const QUESTIONS_DIR = path.resolve(__dirname, 'questions');

/** Load clinical questions from questions/{lang}.txt */
export function loadQuestions(lang) {
  const file = path.join(QUESTIONS_DIR, `${lang}.txt`);
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, 'utf8').split('\n').map(l => l.trim()).filter(l => l.length > 0);
}

export const QUESTIONS_FR = loadQuestions('fr');
export const QUESTIONS_EN = loadQuestions('en');

// ═══════════════════════════════════════════════════════════════════════
// System Prompts
// ═══════════════════════════════════════════════════════════════════════

/**
 * Build the clinical system prompt for a given language and patient name.
 * Includes ALL clinical questions and safety rules.
 */
export function buildSystemPrompt(lang, firstName) {
  const questions = lang === 'fr' ? QUESTIONS_FR : QUESTIONS_EN;
  const name = firstName || (lang === 'fr' ? 'M./Mme' : 'sir/madam');
  const org = lang === 'fr' ? 'RAMQ' : 'Health Canada';
  const numberedQuestions = questions.map((q, i) => `${i + 1}. ${q}`).join('\n');

  if (lang === 'fr') {
    return [
      `Vous êtes un assistant d'accueil clinique pour ${org}.`,
      `Vous effectuez un entretien de pré-consultation structuré.`,
      `Vous vous adressez au patient « ${name} ». Utilisez son prénom.`,
      `Vous êtes professionnel, empathique et efficace. Vous parlez en français.`,
      '',
      `VOICI LES QUESTIONS QUE VOUS DEVEZ POSER, DANS CET ORDRE EXACT :`,
      numberedQuestions,
      '',
      'RÈGLES :',
      '- Posez UNE question à la fois. Attendez la réponse du patient avant de passer à la suivante.',
      '- Si le patient répond brièvement, passez à la question suivante.',
      '- Si le patient divague, redirigez doucement vers la question posée.',
      '- Si le patient mentionne un symptôme urgent (douleur thoracique, essoufflement, perte de conscience), notez-le comme PRIORITAIRE.',
      '- NE JAMAIS diagnostiquer, prescrire ou donner un avis médical.',
      '- Vouvoyez le patient (utilisez « vous »).',
      '- Parlez calmement et clairement.',
    ].join('\n');
  }

  return [
    `You are a Clinical Intake Assistant for ${org}.`,
    `You conduct a structured pre-consultation interview.`,
    `You are speaking to the patient "${name}". Use their first name.`,
    `You are professional, empathetic, and efficient. You speak in English.`,
    '',
    `HERE ARE THE QUESTIONS YOU MUST ASK, IN THIS EXACT ORDER:`,
    numberedQuestions,
    '',
    'RULES:',
    '- Ask ONE question at a time. Wait for the patient\'s response before moving to the next.',
    '- If the patient answers briefly, move to the next question.',
    '- If the patient rambles, gently redirect to the question asked.',
    '- If the patient mentions an urgent symptom (chest pain, shortness of breath, loss of consciousness), flag it as PRIORITY.',
    '- NEVER diagnose, prescribe, or give medical advice.',
    '- Speak calmly and clearly — the patient may be anxious or in pain.',
  ].join('\n');
}

/**
 * Build the priming message (first greeting the patient hears).
 */
export function getPriming(lang, firstName) {
  const questions = lang === 'fr' ? QUESTIONS_FR : QUESTIONS_EN;
  const name = firstName || '';
  const addr = name ? ` ${name},` : '';
  const firstQ = questions[0] || (lang === 'fr'
    ? `Bonjour${addr} qu'est-ce qui vous amène aujourd'hui?`
    : `Hello${addr} what brings you in today?`);

  if (name && firstQ.includes('je suis')) {
    return firstQ;
  }
  return lang === 'fr'
    ? `Bonjour${addr} ${firstQ.charAt(0).toLowerCase() + firstQ.slice(1)}`
    : `Hello${addr} ${firstQ.charAt(0).toLowerCase() + firstQ.slice(1)}`;
}

// ═══════════════════════════════════════════════════════════════════════
// Session Management (in-memory, anonymous by design)
// ═══════════════════════════════════════════════════════════════════════

const sessions = new Map();

export function generateSessionId() {
  const now = new Date();
  const pad = n => String(n).padStart(2, '0');
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}-${String(now.getMilliseconds()).padStart(3, '0')}`;
}

export function getSession(id, patientName = '', lang = 'fr') {
  if (!sessions.has(id)) {
    const name = patientName || '';
    const priming = getPriming(lang, patientName);
    sessions.set(id, {
      turns: [
        { role: 'system', content: buildSystemPrompt(lang, patientName) },
        { role: 'assistant', content: priming }
      ],
      dialog: [],
      primingText: priming,
      patientName: patientName,
      lang: lang,
      title: null,
      createdAt: new Date().toISOString(),
      questionIndex: 0,
      questions: lang === 'fr' ? QUESTIONS_FR : QUESTIONS_EN,
      isComplete: false,
    });
  }
  return sessions.get(id);
}

export function listSessions() {
  return Array.from(sessions.entries()).map(([id, s]) => ({
    session_id: id,
    patient_name: s.patientName,
    lang: s.lang,
    turn_count: s.dialog.length,
    is_complete: s.isComplete,
    created_at: s.createdAt,
  })).sort((a, b) => b.created_at.localeCompare(a.created_at));
}

export function deleteSession(id) {
  sessions.delete(id);
}

// ═══════════════════════════════════════════════════════════════════════
// Groq STT (Speech-to-Text)
// ═══════════════════════════════════════════════════════════════════════

let _groqClient = null;

function getGroqClient(apiKey) {
  if (!_groqClient || process.env.GROQ_API_KEY !== apiKey) {
    _groqClient = new Groq({ apiKey });
  }
  return _groqClient;
}

/**
 * Transcribe an audio buffer to text using Groq's Whisper v3.
 * @param {Buffer} audioBuffer - Raw audio bytes (webm, mp3, wav)
 * @param {string} lang - Language hint: 'fr', 'en', or other ISO code
 * @param {string} apiKey - Groq API key
 * @returns {Promise<string>} Transcribed text
 */
export async function transcribe(audioBuffer, lang = 'fr', apiKey) {
  const groq = getGroqClient(apiKey || process.env.GROQ_API_KEY);
  const file = await Groq.toFile(audioBuffer, 'audio.webm');
  const params = {
    file, model: 'whisper-large-v3', temperature: 0.0, response_format: 'json',
  };
  if (lang && lang !== 'en') {
    params.language = lang === 'fr' ? 'fr' : lang;
  }
  const resp = await groq.audio.transcriptions.create(params);
  return resp.text?.trim() || '';
}

// ═══════════════════════════════════════════════════════════════════════
// LLM Chat — Multi-Provider (OpenAI-compatible)
// ═══════════════════════════════════════════════════════════════════════

/**
 * Create an LLM client for any OpenAI-compatible provider.
 * @param {object} config - { apiKey, baseUrl, model }
 * @returns {OpenAI}
 */
export function createLLMClient(config = {}) {
  const apiKey = config.apiKey || process.env.DEEPSEEK_API_KEY || '';
  const baseURL = config.baseUrl || process.env.DEEPSEEK_BASE_URL || 'https://api.deepseek.com';
  return new OpenAI({ apiKey, baseURL });
}

/**
 * Send messages to an LLM for clinical reasoning.
 * @param {Array<{role: string, content: string}>} messages
 * @param {object} config - { apiKey, baseUrl, model }
 * @returns {Promise<string>}
 */
export async function clinicalChat(messages, config = {}) {
  const client = createLLMClient(config);
  const model = config.model || process.env.DEEPSEEK_MODEL || 'deepseek-chat';
  const resp = await client.chat.completions.create({
    model,
    messages,
    temperature: 0.6,
    max_tokens: 400,
    presence_penalty: 0.3,
    frequency_penalty: 0.3,
  });
  return resp.choices[0]?.message?.content?.trim() || "I'm sorry, I didn't catch that.";
}

// ═══════════════════════════════════════════════════════════════════════
// ElevenLabs TTS (Text-to-Speech)
// ═══════════════════════════════════════════════════════════════════════

/**
 * Synthesize speech from text using ElevenLabs.
 * @param {string} text - Text to speak
 * @param {string} apiKey - ElevenLabs API key
 * @param {string} voiceId - ElevenLabs voice ID
 * @returns {Promise<Buffer>} MP3 audio buffer
 */
export async function synthesize(text, apiKey, voiceId) {
  const key = apiKey || process.env.ELEVENLABS_API_KEY;
  const voice = voiceId || process.env.ELEVENLABS_VOICE_ID || '21m00Tcm4TlvDq8ikWAM';
  const url = `https://api.elevenlabs.io/v1/text-to-speech/${voice}/stream`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'xi-api-key': key },
    body: JSON.stringify({
      text,
      model_id: 'eleven_flash_v2_5',
      voice_settings: { stability: 0.5, similarity_boost: 0.75 },
    }),
  });
  if (!resp.ok) {
    const err = await resp.text().catch(() => '');
    throw new Error(`ElevenLabs ${resp.status}: ${err.slice(0, 200)}`);
  }
  return Buffer.from(await resp.arrayBuffer());
}

// ═══════════════════════════════════════════════════════════════════════
// Full Turn Processing
// ═══════════════════════════════════════════════════════════════════════

/**
 * Process one complete clinical turn: STT → LLM → TTS.
 * This is the core pipeline that the MCP server calls directly.
 *
 * @param {string} sessionId
 * @param {Buffer} audioBuffer - Patient's spoken response
 * @param {object} config - { groqApiKey, llmApiKey, llmBaseUrl, llmModel, elevenlabsApiKey, elevenlabsVoiceId }
 * @returns {Promise<object>} { userText, assistantText, assistantAudioBase64, turnNumber, isComplete }
 */
export async function processClinicalTurn(sessionId, audioBuffer, config = {}) {
  const session = getSession(sessionId);

  // 1. STT — transcribe patient audio
  const userText = await transcribe(audioBuffer, session.lang, config.groqApiKey);
  if (!userText) {
    throw new Error('Could not transcribe audio. Please speak clearly and try again.');
  }

  // Save user turn
  session.turns.push({ role: 'user', content: userText });
  session.dialog.push({ role: 'user', text: userText });

  // 2. LLM — clinical reasoning
  const messages = session.turns.map(t => ({ role: t.role, content: t.content }));
  const replyText = await clinicalChat(messages, {
    apiKey: config.llmApiKey,
    baseUrl: config.llmBaseUrl,
    model: config.llmModel,
  });
  session.turns.push({ role: 'assistant', content: replyText });

  // 3. TTS — synthesize reply
  let audioBase64 = null;
  try {
    const audioBuffer = await synthesize(replyText, config.elevenlabsApiKey, config.elevenlabsVoiceId);
    audioBase64 = audioBuffer.toString('base64');
  } catch (err) {
    console.warn(`[${sessionId}] TTS unavailable: ${err.message}`);
  }

  session.dialog.push({ role: 'assistant', text: replyText, audioBase64 });
  session.questionIndex++;
  if (session.questionIndex >= session.questions.length) {
    session.isComplete = true;
  }

  return {
    user_text: userText,
    assistant_text: replyText,
    assistant_audio_base64: audioBase64,
    turn_number: session.questionIndex,
    is_complete: session.isComplete,
    questions_remaining: session.isComplete ? 0 : session.questions.length - session.questionIndex,
  };
}

/**
 * Compile all dialog turns into a single MP3 podcast buffer.
 */
export async function buildPodcast(sessionId, config = {}) {
  const session = getSession(sessionId);
  const chunks = [];

  // Add greeting
  try {
    chunks.push(await synthesize(session.primingText, config.elevenlabsApiKey, config.elevenlabsVoiceId));
  } catch {}

  // Add all turns
  for (const turn of session.dialog) {
    try {
      if (turn.audioBase64) {
        chunks.push(Buffer.from(turn.audioBase64, 'base64'));
      } else {
        const voiceId = turn.role === 'user'
          ? (config.elevenlabsUserVoiceId || config.elevenlabsVoiceId)
          : config.elevenlabsVoiceId;
        chunks.push(await synthesize(turn.text, config.elevenlabsApiKey, voiceId));
      }
    } catch {}
  }

  return Buffer.concat(chunks);
}
