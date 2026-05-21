import os
import time
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, '.env'))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY as prom_registry
from app.core.config import settings
from app.core.logging import setup_logging
from app.routers import chat, auth, files, memory, admin
from app.db import Base, engine, SessionLocal
from app import models  # noqa: F401
from app.core.milvus import get_milvus_client

setup_logging(
    json_format=(settings.env == "production"),
    log_file=settings.log_file or "",
)

import logging
logger = logging.getLogger(__name__)

# ── Prometheus metrics ──────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "path_template", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency",
    ["method", "path_template"],
)
app = FastAPI(title="Chat Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.env == "production" and "http://localhost:5173" in settings.cors_origins:
    logger.warning("CORS origins include localhost:5173 in production environment")


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    from app.core.context import request_id_var, generate_request_id

    rid = request.headers.get("X-Request-ID", generate_request_id())
    request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    path_tpl = request.url.path  # simplified
    REQUEST_COUNT.labels(method=request.method, path_template=path_tpl, status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, path_template=path_tpl).observe(elapsed)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s %s %s", request.method, request.url.path, request.client.host if request.client else "unknown")
    if settings.env == "development":
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


Base.metadata.create_all(bind=engine)

from app.core.admin_seed import ensure_admin_user
try:
    db = SessionLocal()
    try:
        ensure_admin_user(db)
    finally:
        db.close()
except Exception:
    logger.exception("Admin seed failed — you may need to run: ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'user'")

app.include_router(chat.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(files.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")


@app.get("/health")
async def health():
    checks = {"status": "ok", "database": "ok", "milvus": "ok"}

    # Database check
    try:
        with SessionLocal() as db:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
    except Exception as e:
        checks["database"] = str(e)
        checks["status"] = "degraded"

    # Milvus check
    try:
        client = get_milvus_client()
        client.list_collections()
    except Exception as e:
        checks["milvus"] = str(e)
        checks["status"] = "degraded"

    return checks


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(prom_registry), media_type="text/plain; charset=utf-8")
