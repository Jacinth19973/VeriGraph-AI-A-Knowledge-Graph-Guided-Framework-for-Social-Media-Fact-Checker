"""
VeriGraph AI — Central Configuration
All paths, models, and service endpoints.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# Project root path resolution
_workspace_root = Path(__file__).resolve().parent.parent
_configured_root = os.getenv("PROJECT_ROOT")
# A .env file is commonly copied along with a workspace.  Therefore an
# override is honoured only when deployment explicitly opts in; local source
# execution always uses the tree containing this settings module.
_allow_root_override = os.getenv("VERIGRAPH_ALLOW_PROJECT_ROOT_OVERRIDE", "false").lower() == "true"
PROJECT_ROOT = (Path(_configured_root).expanduser().resolve()
                if _configured_root and _allow_root_override else _workspace_root)

DATASETS_DIR = PROJECT_ROOT / "Datasets"
PREPROCESSED_DIR = DATASETS_DIR / "preprocessed"
KB_DIR = PREPROCESSED_DIR / "knowledge_base"
KG_DIR = PREPROCESSED_DIR / "knowledge_graph"
RUNTIME_DIR = _workspace_root / "runtime"
CASES_DIR = RUNTIME_DIR / "cases"

# Pipeline artifacts
NB6_DIR = PREPROCESSED_DIR / "evidence_retrieval"
NB7_DIR = NB6_DIR / "narrative_clustering"
NB8_DIR = NB6_DIR / "specialist_verification"
NB9_DIR = NB6_DIR / "debate_judgement"

# Models
TEXT_MODEL_NAME = "BAAI/bge-base-en-v1.5"
TEXT_MODEL_PATH = PROJECT_ROOT / "notebooks" / "saved_bge_base_model"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MODEL_PATH = PROJECT_ROOT / "notebooks" / "saved_clip_model"

TEXT_EMB_DIM = 768
IMAGE_EMB_DIM = 512

# Review Mode vs Full Research Mode (Requirements 41, 42)
VERIGRAPH_REVIEW_MODE = (os.getenv("VERIGRAPH_REVIEW_MODE", "true").strip().lower() != "false")

if VERIGRAPH_REVIEW_MODE:
    MAX_QUERIES = int(os.getenv("MAX_QUERIES", "4"))
    MAX_EXTERNAL_RESULTS_PER_PROVIDER = int(os.getenv("MAX_EXTERNAL_RESULTS_PER_PROVIDER", "5"))
    MAX_FINAL_EVIDENCE = int(os.getenv("MAX_FINAL_EVIDENCE", "12"))
    MAX_EVIDENCE_FOR_AGENTS = int(os.getenv("MAX_EVIDENCE_FOR_AGENTS", "5"))
    DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "1"))
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
else:
    MAX_QUERIES = int(os.getenv("MAX_QUERIES", "8"))
    MAX_EXTERNAL_RESULTS_PER_PROVIDER = int(os.getenv("MAX_EXTERNAL_RESULTS_PER_PROVIDER", "10"))
    MAX_FINAL_EVIDENCE = int(os.getenv("MAX_FINAL_EVIDENCE", "30"))
    MAX_EVIDENCE_FOR_AGENTS = int(os.getenv("MAX_EVIDENCE_FOR_AGENTS", "8"))
    DEBATE_ROUNDS = int(os.getenv("DEBATE_ROUNDS", "2"))
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))

MAX_FINAL_FUSED_EVIDENCE = MAX_FINAL_EVIDENCE

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_DISABLE_THINK = True

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Project@2026")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Live search
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "") or os.getenv("SERPAPI_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "") or os.getenv("NEWSAPI_API_KEY", "")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
GDELT_ENABLED = (os.getenv("GDELT_ENABLED", "false").strip().lower() != "false")

# Social Media API Keys
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "")

# Pipeline config
MAX_TEXT_CHARS_PER_EVIDENCE = 350
MAX_CLUSTERS = 8
MIN_CLUSTERS = 2
REDUNDANCY_SIMILARITY_THRESHOLD = 0.90
CONFIDENCE_DECIMALS = 2

ROUNDS_BY_PRIORITY = {"low": DEBATE_ROUNDS, "medium": max(DEBATE_ROUNDS, 2), "high": max(DEBATE_ROUNDS, 3)}
MAX_DEBATE_HISTORY_CHARS = 5000
MAX_TURN_CHARS = 500

# KG expansion
KG_EXPANSION_MAX_CANDIDATES = 10
KG_EXPANSION_CONFIDENCE_THRESHOLD = 0.60

# Ensure directories
for d in [CASES_DIR, NB6_DIR, NB7_DIR, NB8_DIR, NB9_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[settings] project_root={PROJECT_ROOT}")
print(f"[settings] datasets={DATASETS_DIR} kb={KB_DIR}")
print(f"[settings] models: bge={TEXT_MODEL_PATH} clip={CLIP_MODEL_PATH}")

