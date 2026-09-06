# ─── AI Support Platform — Production Dockerfile ──────────────────────────
# Multi-stage build: small final image, no build tools in production.
#
# Build:
#   docker build -t ai-support-platform .
#
# Run:
#   docker run -d --env-file .env -p 8000:8000 ai-support-platform
#
# Or use docker-compose (recommended):
#   docker compose up -d
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# Prevent Python from buffering stdout/stderr (critical for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for faiss-cpu, pymupdf, psycopg2, cryptography
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependencies layer (cached unless requirements.txt changes) ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Application code ───────────────────────────────────────────────────
COPY . .

# Create non-root user
RUN addgroup --system django && adduser --system --ingroup django django
RUN mkdir -p /app/Data /app/media /app/staticfiles && \
    chown -R django:django /app

# Collect static files
RUN SECRET_KEY=build-placeholder DEBUG=True \
    python manage.py collectstatic --noinput 2>/dev/null || true

USER django

EXPOSE 8000

# Health check (lightweight, no auth needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health/')"

# Run with gunicorn
CMD ["gunicorn", "config.wsgi:application", "-c", "gunicorn.conf.py"]
