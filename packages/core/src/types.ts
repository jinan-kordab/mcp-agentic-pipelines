/**
 * Shared TypeScript Types
 *
 * Common types used across all integration packages.
 */

import type { ResolvedLLMConfig } from './llm-config.js';

// ── MCP Tool Types ───────────────────────────────────────────────────

/** Standard MCP tool response content item. */
export interface MCPTextContent {
  type: 'text';
  text: string;
}

/** Standard MCP tool response. */
export interface MCPToolResponse {
  content: MCPTextContent[];
  isError?: boolean;
}

/** MCP resource content. */
export interface MCPResourceContent {
  uri: string;
  mimeType: string;
  text?: string;
  blob?: string;
}

/** Definition of a registered MCP tool. */
export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

/** Definition of a registered MCP resource. */
export interface ResourceDefinition {
  uri: string;
  name: string;
  description: string;
  mimeType: string;
}

/** Definition of a registered MCP prompt. */
export interface PromptDefinition {
  name: string;
  description: string;
  arguments?: Array<{
    name: string;
    description: string;
    required: boolean;
  }>;
}

// ── Health Status ────────────────────────────────────────────────────

export interface ServiceHealth {
  service: string;
  status: 'healthy' | 'unhealthy' | 'disabled';
  provider?: string;
  model?: string;
  error?: string;
}

// ── LLM Provider Info ────────────────────────────────────────────────

export interface LLMProviderInfo {
  provider: string;
  baseUrl: string;
  defaultModel: string;
  isOpenAICompatible: boolean;
  configured: boolean;
}

// ── Audio Types (Clinical Intake) ────────────────────────────────────

export interface AudioTurn {
  role: 'user' | 'assistant';
  text: string;
  audioBase64?: string;
  turnNumber: number;
}

export interface ClinicalSession {
  sessionId: string;
  patientName: string;
  lang: 'en' | 'fr';
  turns: AudioTurn[];
  createdAt: string;
  isComplete: boolean;
}

// ── Search Types (DeepPipe) ──────────────────────────────────────────

export interface SearchHit {
  score: number;
  source: string;
  snippet?: string;
  documentId: number;
}

export interface SearchResults {
  hits: SearchHit[];
  totalHits: number;
  elapsedMs: number;
}

export interface StoredDocument {
  id: number;
  source: string;
  wordCount: number;
  indexedAt: string;
  format?: string;
}

// ── Chat Types (DeepPipe RAG) ────────────────────────────────────────

export interface ChatSource {
  index: number;
  documentId: number;
  title: string;
  sourcePath: string;
  score: number;
}

export interface ChatContext {
  grounded: boolean;
  sources: ChatSource[];
  messages: Array<{ role: string; content: string }>;
}

// ── Fact-Check Types (Piste) ─────────────────────────────────────────

export type VerdictLabel =
  | 'TRUE'
  | 'MOSTLY_TRUE'
  | 'HALF_TRUE'
  | 'MOSTLY_FALSE'
  | 'FALSE'
  | 'PANTS_ON_FIRE'
  | 'UNVERIFIABLE';

export interface FactCheckVerdict {
  runId: string;
  claimId: string;
  verdict: {
    label: VerdictLabel;
    distribution: Record<string, number>;
    explanation: string;
    sources: Array<{
      url: string;
      title: string;
      classification: 'SUPPORTS' | 'REFUTES' | 'UNRELATED';
    }>;
  };
  auditUrl: string;
  elapsedMs: number;
}

// ── Precis Types (Agentic RAG) ───────────────────────────────────────

export interface PrecisQueryResult {
  status: 'success' | 'blocked' | 'error';
  traceId: string;
  plan?: {
    subtasks: Array<{ id: string; type: string; query: string }>;
    reasoning: string;
  };
  report?: Record<string, unknown>;
  evaluation?: {
    relevancy: number;
    trust: number;
    exhaustivity: number;
    hallucinationRate: number;
    citationCoverage: number;
    flaggedIssues: string[];
  };
  guardrail?: {
    action: 'pass' | 'flag' | 'redact' | 'block';
    issues: string[];
    requiresHumanReview: boolean;
  };
  error?: string;
}

// ── Work Order Types (Precis) ────────────────────────────────────────

export interface WorkOrder {
  id: string;
  tailNumber: string;
  workOrderNumber: string;
  date: string;
  aircraftModel: string;
  partNumbers: string[];
  mechanicId: string;
  station: string;
  hoursWorked: number;
  inspectorStamp: string;
  adSbReferences: string[];
  fieldsExtracted: number;
}
