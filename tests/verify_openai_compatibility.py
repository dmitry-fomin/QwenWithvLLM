import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from client.vllm_client import VLMClient

class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Стандартный формат ответа OpenAI
            response_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "qwen3-vl-8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is a mock OpenAI-compatible response."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_mock_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, MockOpenAIHandler)
    print(f"Starting mock server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    # Запуск mock-сервера в отдельном потоке
    PORT = 8081
    server_thread = threading.Thread(target=run_mock_server, args=(PORT,), daemon=True)
    server_thread.start()
    
    # Даем серверу время запуститься
    time.sleep(1)
    
    print("\n--- Verifying with VLMClient ---")
    client = VLMClient(base_url=f"http://localhost:{PORT}/v1", model_name="qwen3-vl-8b")
    
    try:
        response = client.chat(prompt="Hello, mock server!")
        print(f"Client response: {response}")
        
        if response == "This is a mock OpenAI-compatible response.":
            print("✅ SUCCESS: Client correctly parsed the OpenAI-style response.")
        else:
            print("❌ FAILURE: Client returned unexpected response.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ ERROR while calling client: {e}")
        sys.exit(1)

    print("\n--- Verifying raw JSON structure (OpenAI spec) ---")
    import httpx
    with httpx.Client() as http:
        resp = http.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "test"}]}
        )
        data = resp.json()
        print(json.dumps(data, indent=2))
        
        # Проверка обязательных полей OpenAI
        required_fields = ["id", "object", "choices", "usage"]
        missing = [f for f in required_fields if f not in data]
        
        if not missing:
            print("✅ SUCCESS: Response contains all required OpenAI fields.")
            
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                 print("✅ SUCCESS: Choice structure is OpenAI-compatible.")
            else:
                 print("❌ FAILURE: Choice structure is missing 'message' or 'content'.")
                 sys.exit(1)
        else:
            print(f"❌ FAILURE: Missing OpenAI fields: {missing}")
            sys.exit(1)

    print("\n--- Summary ---")
    print("Project is confirmed to be compatible with OpenAI API response format via mock testing.")
