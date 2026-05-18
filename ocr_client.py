"""OCR 服务进程管理与请求客户端。"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from ocr_server import APP_IMPORT_PATH

LOCAL_PROXY_BYPASS = "localhost,127.0.0.1,::1"


def configure_local_proxy_bypass() -> None:
    bypass_hosts = LOCAL_PROXY_BYPASS.split(",")
    for env_key in ("NO_PROXY", "no_proxy"):
        values = [
            value.strip()
            for value in os.environ.get(env_key, "").split(",")
            if value.strip()
        ]
        for host in bypass_hosts:
            if host not in values:
                values.append(host)
        os.environ[env_key] = ",".join(values)


configure_local_proxy_bypass()


BASE_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{BASE_URL}/health"
OCR_URL = f"{BASE_URL}/ocr"
SHUTDOWN_URL = f"{BASE_URL}/shutdown"
PORT = 8000
PROJECT_DIR = Path(__file__).resolve().parent
RESULT_PATH = PROJECT_DIR / "result.json"


def is_port_in_use(port: int) -> bool:
    """检查端口是否被占用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_client:
        try:
            socket_client.bind(("localhost", port))
            return False
        except OSError:
            return True


def get_health(timeout: float = 2.0) -> dict[str, Any]:
    """读取 OCR 服务健康状态。"""
    response = requests.get(HEALTH_URL, timeout=timeout)
    response.raise_for_status()
    return response.json()


def is_ocr_ready() -> bool:
    """判断 OCR 服务是否已可用。"""
    try:
        return bool(get_health().get("engine_loaded"))
    except (requests.RequestException, ValueError):
        return False


def wait_for_server(max_retries: int = 30, interval: float = 1.0) -> bool:
    """轮询 /health 端点，确认 OCR 服务是否就绪。"""
    for attempt_index in range(max_retries):
        if is_ocr_ready():
            return True
        if attempt_index < max_retries - 1:
            time.sleep(interval)
    return False


def start_server(host: str = "0.0.0.0", port: int = PORT) -> subprocess.Popen:
    """启动 OCR FastAPI 服务进程，并返回进程句柄。"""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        APP_IMPORT_PATH,
        "--host",
        host,
        "--port",
        str(port),
    ]
    process_options: dict[str, Any] = {"cwd": PROJECT_DIR}
    if sys.platform.startswith("win"):
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(command, **process_options)


def ensure_ocr_service(
    max_retries: int = 60, interval: float = 1.0
) -> subprocess.Popen | None:
    """确保 OCR 服务运行；新启动时返回进程句柄，已有服务则返回 None。"""
    if is_port_in_use(PORT):
        if wait_for_server(max_retries=1, interval=interval):
            return None
        raise RuntimeError(f"端口 {PORT} 被其他程序占用，无法启动 OCR 服务")

    process = start_server(port=PORT)
    if wait_for_server(max_retries=max_retries, interval=interval):
        return process

    shutdown_server(process)
    raise TimeoutError("OCR 服务启动超时")


def wait_for_port_free(
    port: int = PORT, timeout: float = 8.0, interval: float = 0.2
) -> bool:
    """等待端口释放，用于确认 OCR 服务已经真正退出。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_port_in_use(port):
            return True
        time.sleep(interval)
    return not is_port_in_use(port)


def shutdown_server(process: subprocess.Popen | None) -> bool:
    """强制终止 OCR 服务进程及其整个进程树。"""
    if process is None:
        return False

    pid = process.pid
    poll = process.poll()

    if poll is not None:
        return False
    
    cmd = ["taskkill", "/F", "/T", "/PID", str(pid)]        
    subprocess.run(cmd, capture_output=True, text=True, timeout=10)

    try:
        process.wait(timeout=5)
    except Exception as e:
        print(f"[shutdown] wait 超时: {e}")

    return True


def request_server_shutdown(timeout: float = 2.0, wait_timeout: float = 8.0) -> bool:
    """请求本机 OCR 服务自行关闭，用于页面手动停止常驻服务。"""
    try:
        response = requests.post(SHUTDOWN_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return False
    return wait_for_port_free(timeout=wait_timeout)


def save_result(
    result: dict[str, Any], save_path: str | Path | None = RESULT_PATH
) -> None:
    """按 result.json 的结构保存最近一次 OCR 结果。"""
    if save_path is None:
        return

    result_path = Path(save_path)
    with result_path.open("w", encoding="utf-8") as result_file:
        json.dump(result, result_file, ensure_ascii=False, indent=2)


def request_ocr_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
    timeout: float = 60.0,
    save_path: str | Path | None = RESULT_PATH,
    enhance: bool = False,
) -> dict[str, Any]:
    """向 OCR 服务提交图片字节并返回 result.json 同构结果。"""
    files = {"file": (filename, image_bytes, "application/octet-stream")}
    response = requests.post(
        OCR_URL,
        files=files,
        params={"enhance": str(enhance).lower()},
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    save_result(result, save_path=save_path)
    return result


def recognize_array(
    image: np.ndarray,
    timeout: float = 60.0,
    save_path: str | Path | None = RESULT_PATH,
    enhance: bool = False,
) -> dict[str, Any]:
    """识别 Gradio/摄像头传入的 numpy 图像。"""
    if image.ndim == 2:
        image_to_encode = image
    elif image.shape[2] == 4:
        image_to_encode = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image_to_encode = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    encoded_ok, encoded_image = cv2.imencode(".jpg", image_to_encode)
    if not encoded_ok:
        raise ValueError("无法编码摄像头画面")

    return request_ocr_bytes(
        encoded_image.tobytes(),
        filename="frame.jpg",
        timeout=timeout,
        save_path=save_path,
        enhance=enhance,
    )


def recognize_file(
    image_path: str | Path,
    timeout: float = 60.0,
    save_path: str | Path | None = RESULT_PATH,
    enhance: bool = False,
) -> dict[str, Any]:
    """识别本地图片文件。"""
    image_file_path = Path(image_path)
    with image_file_path.open("rb") as image_file:
        return request_ocr_bytes(
            image_file.read(),
            filename=image_file_path.name,
            timeout=timeout,
            save_path=save_path,
            enhance=enhance,
        )


def recognize(image_url: str | Path, enhance: bool = False) -> dict[str, Any]:
    """兼容旧入口：识别文件并保存到 result.json。"""
    result = recognize_file(image_url, enhance=enhance)
    print("✅ 已完成，结果保存至 result.json")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用本地 OCR 服务识别图片")
    parser.add_argument(
        "image_path", nargs="?", default=R"preprocess_variants\06_laser_dark.jpg", help="输入图片路径"
    )
    parser.add_argument(
        "--enhance",
        action="store_false",
        help="启用芯片激光打标增强，多版本 OCR 后选择最佳结果",
    )
    return parser.parse_args()


def main() -> None:
    process: subprocess.Popen | None = None
    args = parse_args()
    image_path = Path(args.image_path)
    try:
        process = ensure_ocr_service()
        result = recognize(image_path, enhance=args.enhance)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
        requests.RequestException,
    ) as exc:
        print(f"❌ OCR 调用失败: {exc}")
        sys.exit(1)
    finally:
        shutdown_server(process)


if __name__ == "__main__":
    main()
