"""Health check endpoints."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import time

from services.ollama_service import check_ollama_health
from services.neo4j_service import check_neo4j_health, get_kg_stats
from services.faiss_service import init_faiss_indexes

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, Any]
    version: str = "1.0.0"


class ServiceStatus(BaseModel):
    name: str
    status: str  # available, unavailable, degraded, not_configured
    details: str = ""
    latency_ms: float = 0.0


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check of all services."""
    start = time.time()
    
    services = {}
    
    # Ollama
    ollama_start = time.time()
    ollama_ok = check_ollama_health()
    services["ollama"] = ServiceStatus(
        name="Ollama (Qwen3:4B)",
        status="available" if ollama_ok else "unavailable",
        latency_ms=round((time.time() - ollama_start) * 1000, 2),
    ).dict()
    
    # Neo4j
    neo4j_start = time.time()
    neo4j_ok = check_neo4j_health()
    services["neo4j"] = ServiceStatus(
        name="Neo4j Knowledge Graph",
        status="available" if neo4j_ok else "unavailable",
        latency_ms=round((time.time() - neo4j_start) * 1000, 2),
    ).dict()
    
    # FAISS
    faiss_start = time.time()
    try:
        init_faiss_indexes()
        services["faiss"] = ServiceStatus(
            name="FAISS Vector Index",
            status="available",
            latency_ms=round((time.time() - faiss_start) * 1000, 2),
        ).dict()
    except Exception as e:
        services["faiss"] = ServiceStatus(
            name="FAISS Vector Index",
            status="unavailable",
            details=str(e)[:100],
            latency_ms=round((time.time() - faiss_start) * 1000, 2),
        ).dict()
    
    # Live search providers
    live_start = time.time()
    from config.settings import SERPAPI_KEY, NEWSAPI_KEY, GDELT_ENABLED
    services["serpapi"] = ServiceStatus(
        name="SerpApi",
        status="available" if SERPAPI_KEY else "not_configured",
        latency_ms=round((time.time() - live_start) * 1000, 2),
    ).dict()
    
    live_start = time.time()
    services["newsapi"] = ServiceStatus(
        name="NewsAPI",
        status="available" if NEWSAPI_KEY else "not_configured",
        latency_ms=round((time.time() - live_start) * 1000, 2),
    ).dict()
    
    live_start = time.time()
    services["gdelt"] = ServiceStatus(
        name="GDELT",
        status="available" if GDELT_ENABLED else "not_configured",
        latency_ms=round((time.time() - live_start) * 1000, 2),
    ).dict()
    
    overall_status = "healthy" if all(
        s.get("status") == "available" for s in services.values() if s.get("status") != "not_configured"
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        services=services,
    )


@router.get("/neo4j/stats")
async def neo4j_stats():
    """Get Neo4j database statistics."""
    try:
        stats = get_kg_stats()
        return {"nodes": stats.get("nodes", 0), "relationships": stats.get("relationships", 0)}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")