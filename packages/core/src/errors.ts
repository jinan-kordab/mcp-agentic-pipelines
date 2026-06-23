/**
 * Structured Error Types
 *
 * MCPToolError maps directly to MCP protocol error responses.
 * All tool handlers should throw or return these.
 */

/** Base error class for all MCP tool errors. */
export class MCPToolError extends Error {
  public readonly name = 'MCPToolError';

  constructor(
    public readonly code: string,
    message: string,
    public readonly statusCode: number = 400,
  ) {
    super(message);
  }

  /** Convert to MCP protocol error response shape. */
  toMCPResponse(): { content: Array<{ type: 'text'; text: string }>; isError: true } {
    return {
      content: [{
        type: 'text' as const,
        text: JSON.stringify({ code: this.code, message: this.message }),
      }],
      isError: true,
    };
  }

  /** Convert to a plain JSON-serializable error object. */
  toJSON(): { code: string; message: string } {
    return { code: this.code, message: this.message };
  }
}

/** Thrown when an external service (piste, precis) is unreachable. */
export class ServiceUnavailableError extends MCPToolError {
  constructor(service: string) {
    super(
      'SERVICE_UNAVAILABLE',
      `${service} is not reachable. Ensure the service is running and PISTE_API_URL / PRECIS_API_URL are correctly configured.`,
      503,
    );
  }
}

/** Thrown when tool input fails validation. */
export class ValidationError extends MCPToolError {
  constructor(field: string, message: string) {
    super('VALIDATION_ERROR', `${field}: ${message}`, 400);
  }
}

/** Thrown when rate limit is exceeded. */
export class RateLimitError extends MCPToolError {
  constructor(toolName: string, retryAfterMs: number = 1000) {
    super(
      'RATE_LIMITED',
      `Rate limit exceeded for tool "${toolName}". Retry after ${Math.ceil(retryAfterMs / 1000)}s.`,
      429,
    );
  }
}

/** Thrown when an LLM provider is not configured. */
export class LLMNotConfiguredError extends MCPToolError {
  constructor(component: string, provider: string) {
    super(
      'LLM_NOT_CONFIGURED',
      `${component}: LLM provider "${provider}" requires an API key. Set the appropriate env variable (e.g. LLM_DEFAULT_API_KEY or ${component.toUpperCase()}_LLM_API_KEY).`,
      400,
    );
  }
}

/** Thrown when authentication fails for an external API. */
export class AuthenticationError extends MCPToolError {
  constructor(service: string) {
    super(
      'AUTHENTICATION_ERROR',
      `${service} rejected the API key. Verify your credentials.`,
      401,
    );
  }
}

/** Thrown when a requested resource is not found. */
export class NotFoundError extends MCPToolError {
  constructor(resourceType: string, id: string) {
    super('NOT_FOUND', `${resourceType} with id "${id}" not found.`, 404);
  }
}

/** Thrown for internal/unexpected errors. */
export class InternalError extends MCPToolError {
  constructor(message: string = 'An unexpected internal error occurred.') {
    super('INTERNAL', message, 500);
  }
}
