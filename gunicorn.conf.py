"""Production Gunicorn configuration for the AI Support Platform.

Capacity math (see README → Performance):
- Each chat request holds a worker thread while the LLM answers (bounded by
  RAG_MAX_CONCURRENT=8 per process), so threads >> workers is the right shape.
- 4 workers x 24 threads = 96 concurrent request slots; with the per-process
  semaphore and the response cache this comfortably serves 200+ simultaneous
  visitors. Increase ``workers`` only with more CPU cores — FAISS search is
  CPU-bound and the FAISS index is duplicated per worker.
"""

import multiprocessing
import os

# Keep workers modest: every worker loads its own copy of the FAISS index.
# threads carry concurrency; the RAG semaphore bounds LLM fan-out per worker.
workers = int(os.getenv("GUNICORN_WORKERS", str(max(2, multiprocessing.cpu_count() // 2))))
threads = int(os.getenv("GUNICORN_THREADS", "24"))

# A chat request can legitimately take up to the LLM timeout (30s default).
timeout = int(os.getenv("GUNICORN_TIMEOUT", "60"))
graceful_timeout = 30
keepalive = 5

# Tune the request/response buffer sizes (keep responses small).
max_requests = 2000
max_requests_jitter = 100

bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")
worker_tmp_dir = "/dev/shm"

accesslog = os.getenv("GUNICORN_ACCESS_LOG", "-")
errorlog = os.getenv("GUNICORN_ERROR_LOG", "-")
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
