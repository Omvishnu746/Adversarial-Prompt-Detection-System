"""
PromptGuard – FastAPI Application Entry Point (Phase 1)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import API_TITLE, API_VERSION, API_DESCRIPTION
from app.routes.check_prompt import router as prompt_router


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS – open in Phase 1, tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(prompt_router, prefix="/api/v1", tags=["Detection"])

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health() -> dict:
        return {"status": "ok", "version": API_VERSION, "phase": 1}

    return app


app = create_app()
