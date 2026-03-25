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

import httpx
from fastapi import FastAPI, File, Form, UploadFile, Request, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.vllm_client import VLMClient


# ============================================================
# Настройки vLLM (прокси)
# ============================================================

VLLM_BASE_URL = "http://localhost:8000"
vllm_client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=300.0)


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
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get("http://localhost:8000/health")
            return {"vllm": "ok" if resp.status_code == 200 else "error"}
    except Exception as e:
        return {"vllm": "unavailable", "error": str(e)}


@app.get("/api/models")
async def list_models():
    """Список доступных моделей на vLLM."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get("http://localhost:8000/v1/models")
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# OpenAI Совместимость (Прокси)
# ============================================================

@app.post("/api/generate")
@app.post("/v1/chat/completions")
async def openai_proxy(request: Request):
    """
    Проксирует запрос к vLLM OpenAI API.
    Поддерживает стандартный формат OpenAI и стриминг.
    """
    # Получаем тело запроса
    body = await request.json()
    
    # Извлекаем заголовки (кроме хоста и длины контента)
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    # Проверяем параметр stream
    is_streaming = body.get("stream", False)
    
    # Отправляем запрос к vLLM
    try:
        if is_streaming:
            async def stream_generator():
                async with vllm_client.stream(
                    "POST", 
                    "/v1/chat/completions", 
                    json=body, 
                    headers=headers
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            response = await vllm_client.post(
                "/v1/chat/completions",
                json=body,
                headers=headers
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
            
    except Exception as e:
        return JSONResponse(
            {"error": f"vLLM server error: {str(e)}", "type": "proxy_error"},
            status_code=500
        )


@app.post("/v1/completions")
async def completions_proxy(request: Request):
    """Аналогичный прокси для legacy completions API."""
    body = await request.json()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    
    is_streaming = body.get("stream", False)
    
    try:
        if is_streaming:
            async def stream_generator():
                async with vllm_client.stream("POST", "/v1/completions", json=body, headers=headers) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            response = await vllm_client.post("/v1/completions", json=body, headers=headers)
            return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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
