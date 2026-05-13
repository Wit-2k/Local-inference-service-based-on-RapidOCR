"""
本文件运行时无法指定要识别的图片，之后将迁移至 Gradio 框架做成 Web 应用
"""

import json
import socket
import subprocess
import sys
import time

import requests

BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{BASE_URL}/health"
OCR_URL = f"{BASE_URL}/ocr"
PORT = 8000


def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True


def wait_for_server(max_retries: int = 30, interval: float = 1.0) -> bool:
    """轮询 /health 端点，确认 OCR 服务是否就绪"""
    for i in range(max_retries):
        try:
            if requests.get(HEALTH_URL, timeout=2).json().get("engine_loaded"):
                return True
        except requests.ConnectionError:
            pass
        if i < max_retries - 1:
            time.sleep(interval)
    return False


# 在 Gradio 中调用
def shutdown_server(process: subprocess.Popen | None):
    """关闭服务子进程"""
    if process and process.poll() is None:  # poll() 为 None 表示进程仍在运行
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()  # 强制杀死
        print("👋 OCR 服务已关闭")


def recognize(image_url):
    try:
        with open(image_url, "rb") as f:
            response = requests.post(OCR_URL, files={"file": f})
        response.raise_for_status()
        with open("result.json", "w", encoding="utf-8") as f:
            json.dump(response.json(), f, ensure_ascii=False, indent=2)
        print("✅ 已完成，结果保存至 result.json")
    except requests.RequestException as e:
        print(f"❌ 请求失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    process = None
    try:
        if is_port_in_use(PORT):
            if wait_for_server(max_retries=1):
                print("✅ 检测到 OCR 服务已在运行")
            else:
                print(f"❌ 端口 {PORT} 被其他程序占用，退出")
                sys.exit(1)
        else:
            print("🚀 启动 OCR 服务...")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "ocr_server:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(PORT),
                ]
            )
            if not wait_for_server():
                print("❌ 服务启动超时，退出")
                sys.exit(1)
            print("✅ OCR 服务已就绪")

        # TODO: 在此处添加 OCR 请求逻辑 (如调用 OCR_URL)
        recognize("angled.jpg")

    except KeyboardInterrupt:
        print("\n⚠️ 手动中断")
    finally:
        shutdown_server(process)

