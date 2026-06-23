# Unified MCP Server — Docker Image
# Works on Azure, AWS, GCP, or any Docker host.
#
# Build:  docker build -t unified-jk-mcp .
# Run:    docker run -p 3100:3100 --env-file .env unified-jk-mcp
#
# In CI/CD (Azure DevOps / GitHub Actions):
#   docker run -d -p 3100:3100 \
#     -e LLM_DEFAULT_PROVIDER=deepseek \
#     -e LLM_DEFAULT_API_KEY=$DEEPSEEK_API_KEY \
#     -e GROQ_API_KEY=$GROQ_API_KEY \
#     -e ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY \
#     unified-jk-mcp

FROM node:22-alpine AS node-base

# ── Install Python for piste/precis bridges ────────────────────────
FROM node-base AS with-python
RUN apk add --no-cache python3 py3-pip && \
    python3 --version && \
    pip3 install --break-system-packages --no-cache-dir \
      dspy-ai litellm python-dotenv \
      fastapi uvicorn pydantic pydantic-settings \
      numpy nltk sqlalchemy httpx pymupdf

# Optional: FAISS + sentence-transformers (large, comment out if not needed)
# RUN pip3 install --break-system-packages --no-cache-dir faiss-cpu sentence-transformers

# ── Build stage ────────────────────────────────────────────────────
FROM with-python AS builder
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./
COPY packages/core/package.json packages/core/
COPY packages/deeppipe/package.json packages/deeppipe/
COPY packages/piste/package.json packages/piste/
COPY packages/precis/package.json packages/precis/
COPY packages/clinical/package.json packages/clinical/
COPY packages/server/package.json packages/server/

# Install Node dependencies
RUN npm ci --workspaces --include-workspace-root 2>/dev/null || npm install --workspaces --include-workspace-root

# Copy source code
COPY . .

# ── Runtime stage ──────────────────────────────────────────────────
FROM with-python AS runtime
WORKDIR /app

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/packages ./packages
COPY --from=builder /app/start.mjs ./
COPY --from=builder /app/scripts ./scripts

# Copy Python bridge scripts from repos
COPY --from=builder /app/../piste/bridge_piste.py ./bridges/piste/
COPY --from=builder /app/../precis-agentic-pipeline/bridge_precis.py ./bridges/precis/

# Create data directory
RUN mkdir -p /app/data

# Default transport: SSE for cloud, stdio for local
ENV MCP_TRANSPORT=sse
ENV MCP_SSE_PORT=3100
ENV MCP_LOG_LEVEL=info
ENV DEEPPIPE_INDEX_PATH=/app/data/deeppipe.db

EXPOSE 3100

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost:3100/health || exit 1

ENTRYPOINT ["node", "--import", "tsx", "packages/server/src/index.ts"]
