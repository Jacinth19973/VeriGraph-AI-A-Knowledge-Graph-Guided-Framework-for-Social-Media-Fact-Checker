"""VeriGraph AI preflight check.  Run from the project root:   python scripts/preflight.py

Checks the Python environment, model folders, FAISS indexes, API keys, Ollama (reachable, model
pulled, GPU or CPU, real generation speed) and Neo4j. Exit code 1 if a required check fails.
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

results = []  # (level, text)


def ok(msg): results.append(("OK", msg))
def warn(msg): results.append(("WARN", msg))
def fail(msg): results.append(("FAIL", msg))


def check_python():
    v = sys.version_info
    (ok if (3, 10) <= (v.major, v.minor) <= (3, 12) else warn)(
        f"Python {v.major}.{v.minor}.{v.micro} at {sys.executable}"
        + ("" if (3, 10) <= (v.major, v.minor) <= (3, 12) else "  (3.10-3.12 recommended)"))
    for mod in ("fastapi", "uvicorn", "numpy", "pandas", "sklearn", "pydantic", "requests",
                "faiss", "torch", "transformers", "sentence_transformers", "neo4j", "PIL"):
        try:
            m = importlib.import_module(mod)
            ok(f"import {mod} ({getattr(m, '__version__', 'n/a')})")
        except Exception as exc:
            fail(f"import {mod} failed: {type(exc).__name__}: {str(exc)[:160]}")


def check_files():
    try:
        from config import settings as s
    except Exception as exc:
        fail(f"config.settings could not be imported: {exc}")
        return
    for label, p in (("BGE model folder", s.TEXT_MODEL_PATH), ("CLIP model folder", s.CLIP_MODEL_PATH)):
        if Path(p).exists() and any(Path(p).iterdir()):
            ok(f"{label}: {p}")
        else:
            warn(f"{label} missing or empty ({p}); the model will be downloaded from Hugging Face")
    kb = Path(s.KB_DIR)
    expected = {"text_index_bge.faiss": s.TEXT_EMB_DIM, "image_index_clip.faiss": s.IMAGE_EMB_DIM,
                "text_index_clip.faiss": s.IMAGE_EMB_DIM}
    for name, dim in expected.items():
        path = kb / name
        if not path.exists():
            warn(f"FAISS index missing: {path} (historical retrieval will be empty)")
            continue
        try:
            import faiss
            d = faiss.read_index(str(path)).d
            (ok if d == dim else fail)(f"{name}: dimension {d} (expected {dim})")
        except Exception as exc:
            fail(f"{name}: could not be read ({exc})")
    (ok if (kb / "knowledge_base_metadata.csv").exists() else warn)(f"knowledge_base_metadata.csv in {kb}")
    for key in ("SERPAPI_KEY", "NEWSAPI_KEY"):
        (ok if os.getenv(key) else warn)(f"{key} is {'set' if os.getenv(key) else 'EMPTY - live search will not use it'}")


def check_gpu():
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        warn("nvidia-smi was not found: no NVIDIA driver is installed or it is not on PATH. "
             "Install the latest driver from nvidia.com/drivers and reboot; Ollama cannot use the GPU without it.")
        return
    try:
        lines = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        for line in lines:
            name, driver, total, used = [x.strip() for x in line.split(",")]
            free = int(total) - int(used)
            ok(f"GPU: {name}, driver {driver}, {used}/{total} MiB VRAM in use, {free} MiB free")
            if free < 3200:
                warn(f"only {free} MiB of VRAM is free. qwen3:4b needs about 3.2 GB, so Ollama may split the model with "
                     f"the CPU. Close browsers/games, set OLLAMA_FLASH_ATTENTION=1 and OLLAMA_KV_CACHE_TYPE=q8_0, "
                     f"or use a smaller model such as qwen3:1.7b.")
    except Exception as exc:
        warn(f"could not read nvidia-smi: {type(exc).__name__}: {exc}")


def check_ollama():
    import requests
    try:
        from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL
    except Exception as exc:
        fail(f"cannot read Ollama settings: {exc}")
        return
    try:
        tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).json()
    except Exception as exc:
        fail(f"Ollama not reachable at {OLLAMA_BASE_URL}: {type(exc).__name__}. Start it with 'ollama serve'.")
        return
    names = [m.get("name", "") for m in tags.get("models", [])]
    ok(f"Ollama reachable at {OLLAMA_BASE_URL}")
    wanted = OLLAMA_MODEL if ":" in OLLAMA_MODEL else OLLAMA_MODEL + ":latest"
    if wanted not in names and OLLAMA_MODEL not in names:
        fail(f"model '{OLLAMA_MODEL}' is not pulled (found: {', '.join(names) or 'none'}). Run: ollama pull {OLLAMA_MODEL}")
        return
    ok(f"model '{OLLAMA_MODEL}' is available")
    try:
        from services.ollama_service import call_llm
        start = time.time()
        r = call_llm('Reply with JSON only: {"ok": true}', "Say ok.", num_predict=20, timeout=240)
        took = time.time() - start
        if r.get("error"):
            fail(f"test generation failed: {r.get('error')} after {took:.0f}s")
        else:
            (ok if took < 60 else warn)(f"test generation took {took:.1f}s (first call includes model loading)")
    except Exception as exc:
        fail(f"test generation raised {type(exc).__name__}: {exc}")
        return
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
            "model": OLLAMA_MODEL, "prompt": "Count from 1 to 30 in words.", "stream": False, "think": False,
            "options": {"num_predict": 60}}, timeout=300).json()
        if r.get("eval_count") and r.get("eval_duration"):
            tps = r["eval_count"] / (r["eval_duration"] / 1e9)
            (ok if tps >= 15 else warn)(
                f"generation speed: {tps:.1f} tokens/second "
                + ("(GPU-class speed)" if tps >= 15 else "(CPU-class speed; a GPU gives roughly 25-50 tokens/second on an RTX 3050)"))
    except Exception as exc:
        warn(f"could not measure generation speed: {type(exc).__name__}: {str(exc)[:100]}")
    try:
        ps = requests.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5).json().get("models", [])
        for m in ps:
            size, vram = m.get("size", 0), m.get("size_vram", 0)
            if size and vram >= size:
                ok(f"{m.get('name')} runs fully on the GPU")
            elif vram > 0:
                warn(f"{m.get('name')} is split between GPU and CPU ({vram/size:.0%} on GPU); expect slower answers")
            else:
                warn(f"{m.get('name')} runs on the CPU only; this is why each LLM call takes minutes. "
                     f"Look in %LOCALAPPDATA%\\Ollama\\server.log for 'inference compute' / 'no GPU detected' / 'CUDA' "
                     f"lines, update the NVIDIA driver and Ollama, and see the GPU steps in the notes.")
    except Exception as exc:
        warn(f"could not read /api/ps: {exc}")


def check_neo4j():
    try:
        from services.neo4j_service import check_neo4j_health
        from config.settings import NEO4J_URI, NEO4J_USER
        if check_neo4j_health():
            ok(f"Neo4j reachable at {NEO4J_URI} as '{NEO4J_USER}'")
        else:
            warn(f"Neo4j NOT reachable at {NEO4J_URI}. Start Neo4j Desktop/service and check NEO4J_* in .env. "
                 f"The pipeline still runs, with kg_context.status='unavailable'.")
    except Exception as exc:
        warn(f"Neo4j check could not run: {type(exc).__name__}: {str(exc)[:160]}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    for step in (check_python, check_files, check_gpu, check_ollama, check_neo4j):
        try:
            step()
        except Exception as exc:
            fail(f"{step.__name__} crashed: {type(exc).__name__}: {exc}")
    marks = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
    for level, text in results:
        print(f"{marks[level]} {text}")
    bad = sum(1 for l, _ in results if l == "FAIL")
    print(f"\n{sum(1 for l,_ in results if l=='OK')} ok, {sum(1 for l,_ in results if l=='WARN')} warnings, {bad} failed")
    sys.exit(1 if bad else 0)
