/**
 * Token Bucket Rate Limiter
 *
 * Per-tool rate limiting using the token bucket algorithm.
 * Different rate categories for costly vs read-only operations.
 */

import { RateLimitError } from './errors.js';

// ── Token Bucket ─────────────────────────────────────────────────────

export class TokenBucket {
  private tokens: number;
  private lastRefill: number;

  constructor(
    private readonly maxTokens: number,
    private readonly refillRate: number, // tokens per second
  ) {
    this.tokens = maxTokens;
    this.lastRefill = Date.now();
  }

  /**
   * Attempt to consume tokens from the bucket.
   * @returns true if tokens were available and consumed.
   */
  tryConsume(count: number = 1): boolean {
    this.refill();
    if (this.tokens >= count) {
      this.tokens -= count;
      return true;
    }
    return false;
  }

  /**
   * Estimate milliseconds until the next token is available.
   */
  timeUntilNextToken(): number {
    this.refill();
    if (this.tokens >= 1) return 0;
    // tokens is between 0 and 1 (fractional). Need (1 - tokens) worth of refill.
    const needed = 1 - this.tokens;
    return Math.ceil((needed / this.refillRate) * 1000);
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(this.maxTokens, this.tokens + elapsed * this.refillRate);
    this.lastRefill = now;
  }
}

// ── Rate Limiter ─────────────────────────────────────────────────────

/** Rate limit category — determines the token bucket parameters. */
export type RateCategory = 'costly' | 'write' | 'read';

const RATE_LIMITS: Record<RateCategory, { maxTokens: number; refillRate: number }> = {
  costly: { maxTokens: 1, refillRate: 1 },      // 1 req/s
  write:  { maxTokens: 5, refillRate: 5 },      // 5 req/s
  read:   { maxTokens: 30, refillRate: 30 },    // 30 req/s
};

export class RateLimiter {
  private buckets = new Map<string, TokenBucket>();
  private enabled: boolean;

  constructor(
    private readonly defaultMaxRPS: number = 10,
    enabled: boolean = true,
  ) {
    this.enabled = enabled;
  }

  /**
   * Check and consume a rate limit token for a tool.
   * Throws RateLimitError if limit exceeded.
   *
   * @param toolName - MCP tool name.
   * @param category - Rate category override (auto-detected if omitted).
   */
  check(toolName: string, category?: RateCategory): void {
    if (!this.enabled) return;

    const bucket = this.getBucket(toolName, category);
    if (!bucket.tryConsume()) {
      const retryMs = bucket.timeUntilNextToken();
      throw new RateLimitError(toolName, retryMs);
    }
  }

  /**
   * Create or retrieve a token bucket for a tool.
   */
  private getBucket(toolName: string, category?: RateCategory): TokenBucket {
    if (!this.buckets.has(toolName)) {
      const cat = category ?? this.detectCategory(toolName);
      const limits = RATE_LIMITS[cat];
      this.buckets.set(toolName, new TokenBucket(limits.maxTokens, limits.refillRate));
    }
    return this.buckets.get(toolName)!;
  }

  /**
   * Auto-detect rate category from tool name prefix.
   */
  private detectCategory(toolName: string): RateCategory {
    // Costly: full pipeline runs, audio processing
    if (
      toolName.startsWith('piste_fact_check') ||
      toolName.startsWith('clinical_process') ||
      toolName.startsWith('precis_query')
    ) {
      return 'costly';
    }
    // Write: ingestion, upload
    if (
      toolName.includes('ingest') ||
      toolName.includes('upload') ||
      toolName.includes('remove') ||
      toolName.includes('extract')
    ) {
      return 'write';
    }
    // Read: everything else
    return 'read';
  }
}

/** Create a rate limiter from configuration. */
export function createRateLimiter(enabled: boolean, maxRPS: number): RateLimiter {
  return new RateLimiter(maxRPS, enabled);
}
