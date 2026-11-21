# VeriGraph AI

**Knowledge-Graph-Guided Framework for Social Media Fact-Checking via Credibility-Aware Retrieval and Adaptive Multi-Agent Debate**

> A multimodal fact verification system that combines knowledge graph context, narrative clustering, specialist agents, and structured debate to verify social media claims with explainable reasoning.

---

## Overview

VeriGraph AI is an end-to-end fact-checking framework designed to verify claims from social media posts (text + optional images). It integrates:

- **Multimodal Evidence Retrieval** — Dense vector search (BGE-base + CLIP) across web, news, fact-checks, and knowledge bases
- **Narrative Clustering** — Groups retrieved evidence into coherent storylines with credibility weighting
- **Specialist Agent Panel** — Four LLM agents (Evidence Verification, Claim Verification, Source Credibility, Temporal/Location Consistency)
- **Adversarial Debate** — Supporter vs. Skeptic agents debate over multiple rounds with a Judge arbitrator
- **Knowledge Graph Context** — Neo4j-backed entity/event expansion for grounding and gap detection
- **Explainable Output** — Natural-language summaries, confidence breakdowns, and audit trails

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VERIGRAPH AI PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT (Claim + Optional Image)                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  EMBEDDINGS  │───▶│  RETRIEVAL   │───▶│REVERSE IMAGE │                   │
│  │  (BGE/CLIP)  │    │  (Multi-src) │    │   SEARCH     │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│       │                     │                     │                          │
│       ▼                     ▼                     ▼                          │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │              NARRATIVE CLUSTERING (HDBSCAN + LLM)            │            │
│  │    Grounded narratives + credibility scores + diagnostics    │            │
│  └──────────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  IMAGE CHECK │    │   KG CONTEXT │    │ SPECIALISTS  │                   │
│  │ (CLIP + Rev) │    │   (Neo4j)    │    │  (4 Agents)  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│       │                     │                     │                          │
│       └─────────────────────┼─────────────────────┘                          │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │   UNCERTAINTY   │                                       │
│                    │   PROFILING     │                                       │
│                    └─────────────────┘                                       │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │     DEBATE      │                                       │
│                    │ Supporter vs    │                                       │
│                    │   Skeptic       │                                       │
│                    └─────────────────┘                                       │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │      JUDGE      │                                       │
│                    │  (Arbitration)  │                                       │
│                    └─────────────────┘                                       │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐    ┌─────────────────┐               │
│                    │  EXPLAINABILITY │    │  KG EXPANSION   │               │
│                    │  (Summaries,    │    │  (New entities/ │               │
│                    │   Confidence)   │    │   events → KG)  │               │
│                    └─────────────────┘    └─────────────────┘               │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │    PERSISTENCE  │                                       │
│                    │  (Case Store +  │                                       │
│                    │   Audit Trail)  │                                       │
│                    └─────────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal Input** | Text claims + images (URLs, uploads, data URIs) |
| **Configurable Pipeline** | 5 ablation configs (A–E) for research/benchmarking |
| **Review Mode** | Fast mode (1 debate round, fewer queries) vs Full Research Mode |
| **Live Search** | SerpAPI, NewsAPI, Google CSE, GDELT integration |
| **Social Media APIs** | X/Twitter, Facebook, Instagram post fetching |
| **Explainable AI** | Natural-language verdicts with evidence citations |
| **Audit Trail** | Per-stage timing, status, errors, and metrics |
| **Case Management** | JSON persistence with full history |
| **Benchmarking** | Built-in evaluation harness with metrics |

---

## Quick Start

### Prerequisites

- **Python 3.10 – 3.12** (faiss-cpu/torch wheels unreliable on 3.13+)
- **Node.js 18+** (for frontend)
- **Ollama** running locally with a model (default: `qwen3:4b`)
- **Neo4j** (optional, for knowledge graph features)

### Backend Setup

```bash
# Clone and enter project
git clone https://github.com/Jacinth19973/VeriGraph-AI-A-Knowledge-Graph-Guided-Framework-for-Social-Media-Fact-Checker.git
cd VeriGraph-AI-A-Knowledge-Graph-Guided-Framework-for-Social-Media-Fact-Checker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy .env.example to .env and edit)
cp .env.example .env
# Edit .env with your API keys and settings

# Start Ollama (in separate terminal)
ollama serve
ollama pull qwen3:4b

# Start Neo4j (optional, for KG features)
# docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
#   -e NEO4J_AUTH=neo4j/Project@2026 neo4j:latest

# Run backend
uvicorn backend.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```

### Verify Installation

```bash
# Health check
curl http://localhost:8000/api/health

# API docs
open http://localhost:8000/docs
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# Pipeline Mode
VERIGRAPH_REVIEW_MODE=true        # true = fast review mode, false = full research
MAX_QUERIES=4                     # Search queries per provider (review mode)
MAX_FINAL_EVIDENCE=12             # Max evidence items after fusion
DEBATE_ROUNDS=1                   # Debate rounds (review mode)
OLLAMA_TIMEOUT=180                # LLM timeout (seconds)

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b

# Neo4j (Knowledge Graph)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=Project@2026
NEO4J_DATABASE=neo4j

# Live Search APIs (optional)
SERPAPI_KEY=your_serpapi_key
NEWSAPI_KEY=your_newsapi_key
GOOGLE_CSE_API_KEY=your_google_cse_key
GOOGLE_CSE_ID=your_cse_id
GDELT_ENABLED=false

# Social Media (optional)
FACEBOOK_ACCESS_TOKEN=
INSTAGRAM_ACCESS_TOKEN=
X_BEARER_TOKEN=

# Image Processing
VERIGRAPH_USE_IMAGES=true
```

### Pipeline Configurations (Ablation)

| Config | Narratives | Specialists | Debate | KG | Consensus | Use Case |
|--------|------------|-------------|--------|-----|-----------|----------|
| **A** | ❌ | ❌ | ❌ | ❌ | ❌ | Retrieval + Judge baseline |
| **B** | ✅ | ❌ | ❌ | ❌ | ❌ | + Narrative clustering |
| **C** | ✅ | ✅ | ❌ | ❌ | ❌ | + Specialist agents |
| **D** | ✅ | ✅ | ✅ | ❌ | ❌ | + Adversarial debate |
| **E** | ✅ | ✅ | ✅ | ✅ | ✅ | **Full VeriGraph (default)** |

Set via `config` parameter in API request: `"config": "E"`

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health check |
| GET | `/api/health/detailed` | Detailed component status |

#### Verification
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/verification/verify` | Submit claim for verification |
| GET | `/api/verification/{case_id}` | Get verification result |
| GET | `/api/verification/{case_id}/audit` | Get audit trail |

**Request Body:**
```json
{
  "case_id": "unique-case-id",
  "claim": "The claim text to verify",
  "image_path": "https://example.com/image.jpg",  // optional
  "config": "E",                                   // optional (A-E)
  "metadata": {}                                   // optional
}
```

#### Cases
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cases` | List all cases (paginated) |
| GET | `/api/cases/{case_id}` | Get case details |
| DELETE | `/api/cases/{case_id}` | Delete a case |

#### Evidence
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/evidence/{case_id}` | Get evidence for a case |
| GET | `/api/evidence/{case_id}/{evidence_id}` | Get specific evidence |

#### Narratives
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/narratives/{case_id}` | Get narrative clusters |

#### Specialist Agents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agents/{case_id}` | Get specialist results |

#### Debate
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/debate/{case_id}` | Get debate history and results |

#### Knowledge Graph
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/knowledge-graph/{case_id}` | Get KG context |
| GET | `/api/knowledge-graph/{case_id}/expansion` | Get KG expansion results |

#### Explanation
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/explanation/{case_id}` | Get explainability output |

#### Social Media
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/social-media/fetch` | Fetch post from social media URL |
| GET | `/api/social-media/platforms` | List supported platforms |

#### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/analytics/overview` | System-wide metrics |
| GET | `/api/analytics/performance` | Pipeline performance stats |

---

## Usage Examples

### Verify a Text Claim
```bash
curl -X POST http://localhost:8000/api/verification/verify \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "test-001",
    "claim": "The Eiffel Tower is located in London",
    "config": "E"
  }'
```

### Verify a Claim with Image
```bash
curl -X POST http://localhost:8000/api/verification/verify \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "test-002",
    "claim": "This photo shows a recent earthquake in Tokyo",
    "image_path": "https://example.com/earthquake-photo.jpg",
    "config": "E"
  }'
```

### Verify Social Media Post
```bash
curl -X POST http://localhost:8000/api/social-media/fetch \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://x.com/user/status/1234567890"
  }'
```

### Check Result
```bash
curl http://localhost:8000/api/verification/test-001
```

### Get Explanation
```bash
curl http://localhost:8000/api/explanation/test-001
```

---

## Project Structure

```
VeriGraph-AI/
├── backend/                    # FastAPI backend
│   ├── api/                    # API route modules
│   │   ├── agents.py           # Specialist agent endpoints
│   │   ├── analytics.py        # System analytics
│   │   ├── audit.py            # Audit trail
│   │   ├── cases.py            # Case management
│   │   ├── debate.py           # Debate endpoints
│   │   ├── evidence.py         # Evidence retrieval
│   │   ├── explanation.py      # Explainability
│   │   ├── health.py           # Health checks
│   │   ├── knowledge_graph.py  # KG endpoints
│   │   ├── narratives.py       # Narrative clustering
│   │   ├── social_media.py     # Social media fetching
│   │   └── verification.py     # Main verification endpoint
│   ├── services/               # Backend-specific services
│   ├── storage/                # Case persistence
│   ├── main.py                 # FastAPI app entry
│   └── package-lock.json
├── config/                     # Central configuration
│   ├── settings.py             # All settings & env vars
│   └── __init__.py
├── evaluation/                 # Benchmarking & evaluation
│   ├── run_benchmark_evaluation.py
│   ├── benchmark_manifest.json
│   └── evaluation_report.json
├── frontend/                   # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/              # Page components
│   │   ├── services/           # API client
│   │   └── types/              # TypeScript types
│   ├── package.json
│   └── vite.config.ts
├── models/                     # Pydantic data models
│   ├── schemas.py              # All API/schemas
│   └── __init__.py
├── pipeline/                   # Core verification pipeline
│   ├── debate.py               # Supporter/Skeptic/Judge debate
│   ├── embeddings.py           # BGE/CLIP embeddings
│   ├── evaluate_benchmark.py   # Benchmark runner
│   ├── explainability.py       # Natural-language explanations
│   ├── kg_expansion.py         # Knowledge graph expansion
│   ├── multimodal.py           # Image analysis (CLIP + reverse)
│   ├── narrative.py            # Narrative clustering (HDBSCAN)
│   ├── ollama_client.py        # LLM client wrapper
│   ├── orchestrator.py         # Pipeline orchestration
│   ├── retrieval.py            # Multi-source evidence retrieval
│   ├── specialists.py          # 4 specialist agents
│   ├── uncertainty.py          # Uncertainty profiling
│   ├── visual_filter.py        # Visual evidence filtering
│   └── __init__.py
├── runtime/                    # Runtime data (gitignored)
│   ├── cases/                  # Persisted case JSON files
│   └── uploads/                # Uploaded images
├── scripts/                    # Utility scripts
│   ├── check_image.py
│   └── preflight.py
├── services/                   # Shared services
│   ├── claim_conflicts.py
│   ├── faiss_service.py
│   ├── image_fetch.py
│   ├── json_utils.py
│   ├── live_search_service.py
│   ├── nb8_service.py
│   ├── neo4j_service.py
│   ├── ollama_service.py
│   ├── reverse_image_service.py
│   ├── social_media_service.py
│   ├── source_credibility.py
│   ├── time_service.py
│   ├── url_utils.py
│   └── __init__.py
├── storage/                    # Case storage abstraction
│   ├── case_store.py
│   └── __init__.py
├── requirements.txt            # Python dependencies
├── package-lock.json           # Root npm lock (if any)
├── .env                        # Environment config (gitignored)
├── .gitignore
└── README.md                   # This file
```

---

## Models & Dependencies

### Embedding Models
| Model | Purpose | Dimension | Local Path |
|-------|---------|-----------|------------|
| BGE-base-en-v1.5 | Text embeddings | 768 | `notebooks/saved_bge_base_model/` |
| CLIP-ViT-Base-Patch32 | Image embeddings | 512 | `notebooks/saved_clip_model/` |

### LLM (via Ollama)
- Default: `qwen3:4b` (configurable via `OLLAMA_MODEL`)
- Used for: Specialists, Debate, Judge, Narratives, KG Expansion

### Key Python Packages
```
fastapi, uvicorn, pydantic>=2, python-dotenv, requests,
numpy, pandas, scikit-learn, pillow, beautifulsoup4,
neo4j, faiss-cpu, torch, transformers, sentence-transformers
```

---

## Benchmarking

Run the built-in benchmark evaluation:

```bash
python -m pipeline.evaluate_benchmark
```

Results saved to `evaluation/evaluation_report.json` with metrics:
- Accuracy, Precision, Recall, F1
- Per-config ablation results (A–E)
- Processing time per case
- Evidence retrieval stats

---

## Development

### Run Tests
```bash
# Backend tests
pytest tests/ -v

# Frontend lint
cd frontend && npm run lint
```

### Code Style
- Python: Black + Ruff (configured in `pyproject.toml` if present)
- TypeScript: Oxlint + Prettier (configured in frontend)

### Adding a New Specialist Agent
1. Add agent logic in `pipeline/specialists.py`
2. Extend `SpecialistResultBundle` in `models/schemas.py`
3. Update orchestrator to call new agent
4. Add API endpoint in `backend/api/agents.py`

---

## Deployment

### Docker (Recommended)
```dockerfile
# Dockerfile.backend
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -f Dockerfile.backend -t verigraph-backend .
docker run -d -p 8000:8000 --env-file .env verigraph-backend
```

### Production Checklist
- [ ] Set `VERIGRAPH_REVIEW_MODE=false` for full research mode
- [ ] Configure all API keys in `.env`
- [ ] Run Neo4j with persistent volume
- [ ] Use GPU-enabled Ollama for faster inference
- [ ] Set up reverse proxy (nginx) with SSL
- [ ] Configure CORS for production frontend domain
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Enable log aggregation

---

## Research Background

This implementation is based on the paper:

> **"A Knowledge-Graph-Guided Framework for Social Media Fact-Checking via Credibility-Aware Retrieval and Adaptive Multi-Agent Debate"**

Key innovations:
1. **Credibility-aware retrieval** — Sources weighted by historical reliability
2. **Narrative clustering** — Evidence organized into coherent storylines
3. **Adaptive multi-agent debate** — Supporter/Skeptic with uncertainty-driven round allocation
4. **KG-guided expansion** — Verified claims expand the knowledge graph iteratively

---

## License

MIT License — see `LICENSE` file for details.

---

## Citation

If you use VeriGraph AI in your research, please cite:

```bibtex
@article{verigraph2026,
  title={A Knowledge-Graph-Guided Framework for Social Media Fact-Checking via Credibility-Aware Retrieval and Adaptive Multi-Agent Debate},
  author={Jacinth, M. and et al.},
  journal={ICTCS 2026},
  year={2026}
}
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Support

- **Issues**: [GitHub Issues](https://github.com/Jacinth19973/VeriGraph-AI-A-Knowledge-Graph-Guided-Framework-for-Social-Media-Fact-Checker/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Jacinth19973/VeriGraph-AI-A-Knowledge-Graph-Guided-Framework-for-Social-Media-Fact-Checker/discussions)

---

## Acknowledgments

- **BGE Embeddings**: BAAI
- **CLIP**: OpenAI
- **Ollama**: Local LLM serving
- **Neo4j**: Graph database
- **FAISS**: Vector similarity search
- **HDBSCAN**: Narrative clustering
- **FastAPI**: Modern Python web framework
- **React + Vite**: Frontend tooling