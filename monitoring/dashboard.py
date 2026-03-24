"""
CLI дашборд для мониторинга vLLM сервера и запросов.

Показывает:
- Статус vLLM сервера
- GPU загрузка (через nvidia-smi)
- Статистика запросов из лог-файла
- Последние запросы

Использование:
    python3 monitoring/dashboard.py                   # Одноразовый вывод
    python3 monitoring/dashboard.py --watch 5         # Обновление каждые 5 сек
    python3 monitoring/dashboard.py --log logs/requests.jsonl  # Указать лог
"""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None


def get_vllm_status(server_url: str = "http://localhost:8000") -> dict:
    """Получить статус vLLM сервера."""
    if httpx is None:
        return {"status": "unknown", "error": "httpx not installed"}

    try:
        with httpx.Client(timeout=3.0) as client:
            # Health check
            health = client.get(f"{server_url}/health")
            health_ok = health.status_code == 200

            # Models
            models_resp = client.get(f"{server_url}/v1/models")
            models = []
            if models_resp.status_code == 200:
                data = models_resp.json()
                models = [m["id"] for m in data.get("data", [])]

            return {
                "status": "ok" if health_ok else "error",
                "models": models,
            }
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


def get_gpu_info() -> list[dict]:
    """Получить информацию о GPU через nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "temp_c": int(parts[2]) if parts[2] != "N/A" else None,
                    "util_pct": int(parts[3]) if parts[3] != "N/A" else None,
                    "mem_used_mb": int(parts[4]) if parts[4] != "N/A" else None,
                    "mem_total_mb": int(parts[5]) if parts[5] != "N/A" else None,
                    "power_w": float(parts[6]) if parts[6] != "N/A" else None,
                })
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def parse_log_stats(log_file: str) -> dict:
    """Парсит лог-файл и считает статистику."""
    path = Path(log_file)
    if not path.exists():
        return {"error": f"Log file not found: {log_file}"}

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        return {"total": 0}

    latencies = [e["latency_ms"] for e in entries if "latency_ms" in e]
    errors = [e for e in entries if e.get("error")]
    models = Counter(e.get("model", "unknown") for e in entries)
    image_count = sum(1 for e in entries if e.get("has_images"))

    return {
        "total": len(entries),
        "errors": len(errors),
        "error_rate_pct": round(len(errors) / len(entries) * 100, 1) if entries else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 0) if latencies else 0,
        "min_latency_ms": round(min(latencies), 0) if latencies else 0,
        "max_latency_ms": round(max(latencies), 0) if latencies else 0,
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 0) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 0) if latencies else 0,
        "with_images": image_count,
        "text_only": len(entries) - image_count,
        "models": dict(models),
        "last_request": entries[-1].get("timestamp", "?"),
    }


def format_bar(value: float, max_value: float, width: int = 20) -> str:
    """Рисует ASCII прогресс-бар."""
    if max_value <= 0:
        return " " * width
    filled = int(value / max_value * width)
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


def render_dashboard(
    server_url: str = "http://localhost:8000",
    log_file: str = "logs/requests.jsonl",
):
    """Отрисовать дашборд в терминал."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\033[2J\033[H")  # Clear screen
    print(f"{'=' * 60}")
    print(f"  VLM Dashboard  |  {now}")
    print(f"{'=' * 60}")

    # vLLM Status
    vllm = get_vllm_status(server_url)
    status_icon = "●" if vllm["status"] == "ok" else "○"
    print(f"\n  vLLM Server: {status_icon} {vllm['status']}")
    if vllm.get("models"):
        print(f"  Models: {', '.join(vllm['models'])}")
    if vllm.get("error"):
        print(f"  Error: {vllm['error']}")

    # GPU
    gpus = get_gpu_info()
    if gpus:
        print(f"\n  {'─' * 56}")
        print(f"  GPU Status")
        print(f"  {'─' * 56}")
        for gpu in gpus:
            mem_pct = (
                gpu["mem_used_mb"] / gpu["mem_total_mb"] * 100
                if gpu["mem_total_mb"]
                else 0
            )
            bar = format_bar(mem_pct, 100, 15)
            print(
                f"  GPU {gpu['index']}: {gpu['name']}"
            )
            print(
                f"    Memory: {bar} {gpu['mem_used_mb']}/{gpu['mem_total_mb']} MB ({mem_pct:.0f}%)"
            )
            if gpu["util_pct"] is not None:
                util_bar = format_bar(gpu["util_pct"], 100, 15)
                print(f"    Util:   {util_bar} {gpu['util_pct']}%")
            if gpu["temp_c"] is not None:
                print(f"    Temp: {gpu['temp_c']}°C  Power: {gpu['power_w']:.0f}W")
    else:
        print("\n  GPU: not available (nvidia-smi not found)")

    # Request stats
    stats = parse_log_stats(log_file)
    print(f"\n  {'─' * 56}")
    print(f"  Request Statistics ({log_file})")
    print(f"  {'─' * 56}")

    if stats.get("error"):
        print(f"  {stats['error']}")
    elif stats["total"] == 0:
        print("  No requests logged yet.")
    else:
        print(f"  Total requests:   {stats['total']}")
        print(f"  Errors:           {stats['errors']} ({stats['error_rate_pct']}%)")
        print(f"  With images:      {stats['with_images']}")
        print(f"  Text only:        {stats['text_only']}")
        print(f"  Avg latency:      {stats['avg_latency_ms']}ms")
        print(f"  P50 latency:      {stats['p50_latency_ms']}ms")
        print(f"  P95 latency:      {stats['p95_latency_ms']}ms")
        print(f"  Min/Max:          {stats['min_latency_ms']}/{stats['max_latency_ms']}ms")
        if stats.get("models"):
            print(f"  Models used:      {stats['models']}")
        print(f"  Last request:     {stats['last_request']}")

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="VLM Monitoring Dashboard")
    parser.add_argument(
        "--server", default="http://localhost:8000", help="vLLM server URL"
    )
    parser.add_argument(
        "--log", default="logs/requests.jsonl", help="Log file path"
    )
    parser.add_argument(
        "--watch", type=int, default=None, help="Refresh interval (seconds)"
    )
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                render_dashboard(args.server, args.log)
                print(f"  Refreshing every {args.watch}s (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        render_dashboard(args.server, args.log)


if __name__ == "__main__":
    main()
