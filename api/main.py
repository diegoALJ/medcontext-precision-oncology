from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.schemas import (
    CompileRequest,
    CompileResponse,
    HealthResponse,
)
from src.clinical_fact_compiler import ClinicalFactCompiler
from src.medcontext_packet import MedContextPacketBuilder


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("medcontext.api")


# ============================================================
# CONFIG
# ============================================================

VOCAB_PATH = (
    ROOT
    / "data"
    / "vocabulary"
    / "oncology_vocabulary.yaml"
)

MODEL_DIR = (
    ROOT
    / "results"
    / "models"
)


# ============================================================
# APPLICATION STATE
# ============================================================

pipeline: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading MedContext inference pipeline")

    compiler = ClinicalFactCompiler(
        vocabulary_path=VOCAB_PATH,
        model_dir=MODEL_DIR,
    )

    pipeline["packet_builder"] = MedContextPacketBuilder(
        compiler
    )

    logger.info("MedContext inference pipeline ready")

    yield

    pipeline.clear()

    logger.info("MedContext API stopped")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="MedContext Precision Oncology API",
    description=(
        "REST API for compiling precision-oncology "
        "reports into structured clinical facts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok"
    )


@app.post(
    "/compile",
    response_model=CompileResponse,
)
def compile_report(
    request: CompileRequest,
) -> CompileResponse:

    start_time = time.perf_counter()

    logger.info(
        "Inference request received | report_id=%s | characters=%d",
        request.report_id,
        len(request.report_text),
    )

    try:
        packet_builder = pipeline["packet_builder"]

        packet = packet_builder.compile_compact_report(
            report_text=request.report_text,
            report_id=request.report_id,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "Inference completed | report_id=%s | facts=%d | latency_ms=%.2f",
            request.report_id,
            len(packet["facts"]),
            latency_ms,
        )

        return CompileResponse(
            **packet
        )

    except Exception:
        logger.exception(
            "Inference failed | report_id=%s",
            request.report_id,
        )

        raise HTTPException(
            status_code=500,
            detail="MedContext inference failed.",
        )