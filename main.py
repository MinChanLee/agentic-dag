"""
FastAPI Server — Agentic DAG Pipeline
Railway 배포용 웹 서버 뼈대
"""

import asyncio
import json
import logging
import os
import sys
import traceback

sys.stdout.reconfigure(line_buffering=True)

from fastapi import FastAPI, Form, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from contextlib import asynccontextmanager

from agentic.api import arun_pipeline
from agentic.models import FileInput
from agentic.db.pool import get_pool, close_pool
from agentic.db.run_logger import RunLogger, _runs_table, _logs_table
from agentic.db.log_handler import DBLogHandler


@asynccontextmanager
async def lifespan(app):
    # Startup
    await get_pool()
    db_handler = DBLogHandler()
    db_handler.setLevel(logging.INFO)
    db_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )
    logging.getLogger().addHandler(db_handler)
    yield
    # Shutdown
    await close_pool()


app = FastAPI(title="Agentic Pipeline API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: 프로젝트에 맞게 파이프라인 경로 수정
PIPELINE_PATH = os.path.join(BASE_DIR, "pipelines", "example.yaml")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/run")
async def run_pipeline(
    # TODO: 프로젝트 입력에 맞게 파라미터 수정
    user_text: str = Form(...),
):
    """run_id를 즉시 반환하고, 파이프라인은 백그라운드에서 실행."""
    run_logger = RunLogger()

    # DB에 run 레코드 생성 (status='running')
    await run_logger.start(
        input_metadata={
            "user_text": user_text,
            # TODO: 프로젝트 입력 필드 추가
        },
    )

    # 백그라운드 태스크
    async def _run():
        try:
            inputs = {
                "user_text": user_text,
                # TODO: 프로젝트 입력 필드 추가
            }

            result = await arun_pipeline(
                pipeline_path=PIPELINE_PATH,
                inputs=inputs,
            )

            outputs = result.outputs_only()

            await run_logger.complete(
                output_data=outputs,
            )
        except Exception as e:
            traceback.print_exc()
            await run_logger.fail(str(e))

    asyncio.create_task(_run())

    return JSONResponse({"run_id": run_logger.run_id, "status": "running"})


# ------------------------------------------------------------------
# Run status & listing endpoints
# ------------------------------------------------------------------


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """단일 run 상세 조회."""
    pool = await get_pool()
    if not pool:
        return JSONResponse(
            {"error": "Database not configured"}, status_code=503
        )

    runs = _runs_table()
    row = await pool.fetchrow(f"SELECT * FROM {runs} WHERE id = $1", run_id)
    if not row:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    data = dict(row)
    for key in ("created_at", "started_at", "completed_at"):
        if data.get(key):
            data[key] = data[key].isoformat()
    for key in ("input_metadata", "output_data"):
        if data.get(key) and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except (json.JSONDecodeError, TypeError):
                pass

    return JSONResponse(data)


@app.get("/runs")
async def list_runs(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
):
    """run 목록 조회 (pagination)."""
    pool = await get_pool()
    if not pool:
        return JSONResponse(
            {"error": "Database not configured"}, status_code=503
        )

    runs = _runs_table()
    offset = (page - 1) * limit

    total_row = await pool.fetchrow(f"SELECT count(*) as cnt FROM {runs}")
    total_count = total_row["cnt"] if total_row else 0

    rows = await pool.fetch(
        f"""SELECT id, status, label, created_at, completed_at
            FROM {runs}
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2""",
        limit,
        offset,
    )

    items = []
    for row in rows:
        item = dict(row)
        for key in ("created_at", "completed_at"):
            if item.get(key):
                item[key] = item[key].isoformat()
        items.append(item)

    return JSONResponse(
        {
            "items": items,
            "total_count": total_count,
            "page": page,
            "limit": limit,
        }
    )


@app.get("/runs/{run_id}/logs")
async def get_run_logs(run_id: str):
    """특정 run의 로그 조회."""
    pool = await get_pool()
    if not pool:
        return JSONResponse(
            {"error": "Database not configured"}, status_code=503
        )

    logs = _logs_table()
    rows = await pool.fetch(
        f"""SELECT id, level, message, prompt, response, created_at
            FROM {logs}
            WHERE run_id = $1
            ORDER BY id""",
        run_id,
    )

    items = []
    for row in rows:
        item = dict(row)
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat()
        items.append(item)

    return JSONResponse({"run_id": run_id, "logs": items})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
