"""VeriGraph AI — Backend services wrapper."""
from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-export main services
from services.ollama_service import *
from services.neo4j_service import *
from services.faiss_service import *
from services.live_search_service import *
from services.url_utils import *
from services.nb8_service import *
from services.social_media_service import *