'''
本文件运行时无法指定要识别的图片，之后将迁移至 Gradio 框架做成 Web 应用
'''

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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # 尝试绑定指定端口
            s.bind(("localhost", port))
            # 如果绑定没报错，说明端口确实空闲可用
            return False
        except OSError:
            # 绑定报错（通常是被占用），说明不可用
            return True


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
    process = None

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
        # 非阻塞启动
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "server:app",
                "--host",
                "0.0.0.0",  # 绑定本机所有 IPv4 地址
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
        with open("stc89.jpg", "rb") as f:
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
    finally:
        if process is not None:
            print("正在关闭服务...")
            process.terminate()
            process.wait()
            print("服务已关闭")
