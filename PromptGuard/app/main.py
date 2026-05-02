"""
PromptGuard – FastAPI Application Entry Point (Phase 3)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import API_TITLE, API_VERSION, API_DESCRIPTION
from app.routes.check_prompt import router as prompt_router


# ── App factory ───────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from app.services.semantic_engine import initialise_semantic_engine
from app.services.model_loader import initialise_classifier, is_classifier_loaded

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase 2: Initialise the SBERT semantic engine on startup
    initialise_semantic_engine()
    # Phase 3: Eagerly load the fine-tuned DistilBERT classifier
    initialise_classifier()
    yield
    # Cleanup on shutdown (if any)

def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
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
        from app.services.semantic_engine import is_semantic_engine_ready
        semantic_ready = is_semantic_engine_ready()
        classifier_ready = is_classifier_loaded()

        if classifier_ready:
            phase = 3
        elif semantic_ready:
            phase = 2
        else:
            phase = 1

        return {
            "status": "ok",
            "version": API_VERSION,
            "phase": phase,
            "semantic_engine_ready": semantic_ready,
            "classifier_ready": classifier_ready,
        }

    return app


app = create_app()
