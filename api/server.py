# api/server.py
#
# FastAPI 앱. main.py(Streamlit)를 대체하는 API 서버 — 분석 파이프라인은
# api/pipeline.py(원본 main.py/utils/loader.py 로직을 그대로 옮긴 것)를 그대로 쓰고,
# 화면은 web/ 디렉터리의 정적 HTML/CSS/JS를 서빙한다.
#
# 실행: uv run uvicorn api.server:app --port 8020  (반드시 프로젝트 루트에서 실행할
# 필요는 없음 — pipeline.py가 파일 위치 기준 절대경로를 씀)

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import pipeline

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load_all_resources()
    yield


app = FastAPI(title="한국어 문장 구조 분석 엔진 API", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    results = pipeline.analyze_text(req.text)
    return {"results": results}


@app.get("/api/health")
def health():
    ready = pipeline.PARSER is not None and pipeline.KIWI is not None
    return {"status": "ok" if ready else "loading"}


# 정적 프런트엔드(web/index.html 등) — API 라우트 등록 뒤에 마운트해야
# "/api/*" 경로를 정적 파일 탐색보다 먼저 매칭한다.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
