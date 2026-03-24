"""
Web UI для работы с VLM моделями через vLLM.

Запуск:
    python3 web/app.py
    # или
    uvicorn web.app:app --host 0.0.0.0 --port 7860 --reload
"""

import base64
import io
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.vllm_client import VLMClient


# ============================================================
# Приложение
# ============================================================

app = FastAPI(title="VLM Web UI", description="Web interface for VLM models via vLLM")

WEB_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")

# Глобальный клиент (переиспользуется между запросами)
_client: Optional[VLMClient] = None


def get_client(server_url: str = "http://localhost:8000/v1", model_name: str = "qwen3-vl-8b") -> VLMClient:
    global _client
    if _client is None or _client.model_name != model_name:
        _client = VLMClient(base_url=server_url, model_name=model_name)
    return _client


# ============================================================
# Роуты
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
async def analyze(
    prompt: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    model_name: str = Form("qwen3-vl-8b"),
    server_url: str = Form("http://localhost:8000/v1"),
    max_tokens: int = Form(2048),
    temperature: float = Form(0.3),
):
    """Анализ изображения или текстовый запрос."""
    client = get_client(server_url, model_name)

    images = []

    # Загруженный файл
    if image and image.filename:
        content = await image.read()
        b64 = base64.b64encode(content).decode("utf-8")
        images.append(b64)

    # URL изображения
    if image_url and image_url.strip():
        images.append(image_url.strip())

    start = time.time()

    try:
        result = client.chat(
            prompt=prompt,
            images=images if images else None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.time() - start

        return JSONResponse({
            "result": result,
            "elapsed_seconds": round(elapsed, 2),
            "model": model_name,
        })
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )


@app.get("/api/health")
async def health():
    """Проверка доступности vLLM сервера."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get("http://localhost:8000/health")
            return {"vllm": "ok" if resp.status_code == 200 else "error"}
    except Exception as e:
        return {"vllm": "unavailable", "error": str(e)}


@app.get("/api/models")
async def list_models():
    """Список доступных моделей на vLLM."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get("http://localhost:8000/v1/models")
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Запуск
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
    )
