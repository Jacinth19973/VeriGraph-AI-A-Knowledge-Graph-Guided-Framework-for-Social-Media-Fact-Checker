"""VeriGraph AI — FastAPI Backend."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import fastapi
import numpy

from backend.api import (
    verification, cases, evidence, narratives, 
    agents, debate, knowledge_graph, explanation, audit, 
    social_media, health, analytics
)
from backend.storage.case_store import ensure_dirs


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    from storage.case_store import mark_interrupted_cases
    interrupted = mark_interrupted_cases()
    if interrupted:
        print(f"[startup] marked {interrupted} case(s) interrupted by a previous restart as failed")
    import pipeline.orchestrator, pipeline.retrieval, pipeline.narrative
    import services.live_search_service, services.faiss_service, config.settings, models.schemas
    print("=" * 70)
    print("VERIGRAPH AI RUNTIME")
    print("Python:", sys.version); print("Python executable:", sys.executable)
    print("NumPy:", numpy.__version__, numpy.__file__)
    print("FastAPI:", fastapi.__version__, fastapi.__file__)
    for name, module in {"orchestrator": pipeline.orchestrator, "retrieval": pipeline.retrieval,
                         "narrative": pipeline.narrative, "live_search": services.live_search_service,
                         "faiss_service": services.faiss_service, "settings": config.settings,
                         "schemas": models.schemas}.items(): print(f"{name}: {module.__file__}")
    print("=" * 70)
    yield


app = FastAPI(
    title="VeriGraph AI API",
    description="Knowledge-Graph-Enhanced Multimodal Social Media Fact Verification",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(verification.router, prefix="/api/verification", tags=["Verification"])
app.include_router(cases.router, prefix="/api/cases", tags=["Cases"])
app.include_router(evidence.router, prefix="/api/evidence", tags=["Evidence"])
app.include_router(narratives.router, prefix="/api/narratives", tags=["Narratives"])
app.include_router(agents.router, prefix="/api/agents", tags=["Specialist Agents"])
app.include_router(debate.router, prefix="/api/debate", tags=["Debate"])
app.include_router(knowledge_graph.router, prefix="/api/knowledge-graph", tags=["Knowledge Graph"])
app.include_router(explanation.router, prefix="/api/explanation", tags=["Explanation"])
app.include_router(audit.router, prefix="/api/audit", tags=["Audit"])
app.include_router(social_media.router, prefix="/api/social-media", tags=["Social Media"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])


@app.get("/")
async def root():
    return {
        "name": "VeriGraph AI",
        "subtitle": "Knowledge-Guided Multimodal Fact Verification",
        "version": "1.0.0",
        "docs": "/docs",
    }
