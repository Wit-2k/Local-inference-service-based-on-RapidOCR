import json
import socket
import subprocess
import sys
import time

import requests

BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{BASE_URL}/health"
OCR_URL = f"{BASE_URL}/ocr"


def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def is_ocr_server_ready(max_retries=10, interval=1):
    """轮询 /health 端点，确认 OCR 服务是否就绪"""
    for i in range(max_retries):
        try:
            resp = requests.get(HEALTH_URL, timeout=2)
            data = resp.json()
            if data.get("engine_loaded"):
                return True
        except requests.ConnectionError:
            pass
        if i < max_retries - 1:
            time.sleep(interval)
    return False


def wait_for_server(max_retries=30, interval=1):
    """等待服务启动就绪"""
    for i in range(max_retries):
        if is_ocr_server_ready(max_retries=1):
            return True
        if i < max_retries - 1:
            time.sleep(interval)
    return False


if __name__ == "__main__":
    if is_port_in_use(8000):
        # 端口被占用，检查是否为 OCR 服务
        if is_ocr_server_ready():
            print("✅ 检测到 OCR 服务已在运行，直接发送请求")
        else:
            print("❌ 端口 8000 被其他程序占用，退出")
            sys.exit(1)
    else:
        # 端口空闲，启动服务子进程
        print("🚀 启动 OCR 服务...")
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "server:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
        )

        if not wait_for_server():
            print("❌ 服务启动超时，退出")
            sys.exit(1)

        print("✅ OCR 服务已就绪")

    # 发送 OCR 请求
    try:
        with open("sign.jpg", "rb") as f:
            response = requests.post(OCR_URL, files={"file": f})

        response.raise_for_status()
        result = response.json()

        with open("result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(
            f"✅ 识别完成，推理耗时 {result.get('inference_time_ms', '?')} ms，结果已保存至 result.json"
        )
    except requests.RequestException as e:
        print(f"❌ 请求失败: {e}")
        sys.exit(1)
