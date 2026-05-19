"""Windows desktop shell for the Gradio OCR app."""

from __future__ import annotations

import ctypes
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Callable

import requests

from runtime_paths import app_base_dir


APP_TITLE = "芯片 OCR 识别"
GRADIO_HOST = "127.0.0.1"
GRADIO_PORT = 7860
GRADIO_URL = f"http://{GRADIO_HOST}:{GRADIO_PORT}"
OCR_PORT = 8000
OCR_SHUTDOWN_URL = f"http://{GRADIO_HOST}:{OCR_PORT}/shutdown"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 860
SERVER_READY_TIMEOUT_SECONDS = 45.0
LOG_PATH = app_base_dir() / "desktop_app.log"
EDGE_PROFILE_DIR = app_base_dir() / "edge_profile"

cleanup_lock = threading.Lock()
cleanup_started = False
gradio_close: Callable[[], None] | None = None
stdio_handles: list[object] = []


def log_message(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
    except OSError:
        pass


def ensure_standard_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        if getattr(sys, stream_name) is not None:
            continue
        stream = open(os.devnull, "w", encoding="utf-8")
        stdio_handles.append(stream)
        setattr(sys, stream_name, stream)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_client:
        return socket_client.connect_ex((GRADIO_HOST, port)) == 0


def show_error(title: str, message: str) -> None:
    log_message(f"{title}: {message}")
    if sys.platform.startswith("win"):
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
        return
    print(f"{title}: {message}", file=sys.stderr)


def wait_for_gradio_ready(timeout_seconds: float = SERVER_READY_TIMEOUT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = requests.get(GRADIO_URL, timeout=1.0)
            if response.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.25)
    return False


def wait_for_port_free(port: int, timeout_seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not is_port_in_use(port):
            return True
        time.sleep(0.2)
    return not is_port_in_use(port)


def find_edge_executable() -> str | None:
    executable = shutil.which("msedge")
    if executable:
        return executable

    candidate_paths = [
        os.path.join(
            os.environ.get("ProgramFiles(x86)", ""),
            "Microsoft",
            "Edge",
            "Application",
            "msedge.exe",
        ),
        os.path.join(
            os.environ.get("ProgramFiles", ""),
            "Microsoft",
            "Edge",
            "Application",
            "msedge.exe",
        ),
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft",
            "Edge",
            "Application",
            "msedge.exe",
        ),
    ]
    for candidate in candidate_paths:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def run_desktop_window() -> None:
    edge_executable = find_edge_executable()
    if edge_executable is None:
        raise RuntimeError(
            "未找到 Microsoft Edge，无法打开桌面窗口。请安装 Edge 后重试。"
        )

    EDGE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        edge_executable,
        f"--app={GRADIO_URL}",
        f"--user-data-dir={EDGE_PROFILE_DIR}",
        "--no-first-run",
        "--disable-features=Translate",
        f"--window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}",
    ]
    log_message(f"Launching Edge app window: {edge_executable}")
    process = subprocess.Popen(command)
    process.wait()


def start_gradio_server() -> None:
    global gradio_close

    log_message("Importing display module")
    from display import APP_CSS, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, demo

    if GRADIO_SERVER_NAME != GRADIO_HOST or GRADIO_SERVER_PORT != GRADIO_PORT:
        raise RuntimeError("桌面入口和 Gradio 端口配置不一致")

    log_message(f"Launching Gradio on {GRADIO_HOST}:{GRADIO_PORT}")
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False,
        css=APP_CSS,
        inbrowser=False,
        prevent_thread_lock=True,
        quiet=True,
    )
    gradio_close = demo.close

    if not wait_for_gradio_ready():
        raise TimeoutError("Gradio 页面启动超时")
    log_message("Gradio is ready")


def cleanup_services() -> None:
    global cleanup_started

    with cleanup_lock:
        if cleanup_started:
            return
        cleanup_started = True

    try:
        requests.post(OCR_SHUTDOWN_URL, timeout=1.5)
    except requests.RequestException:
        pass
    wait_for_port_free(OCR_PORT, timeout_seconds=6.0)

    if gradio_close is not None:
        try:
            gradio_close()
        except Exception as exc:
            print(f"[desktop] Gradio 关闭失败: {exc}", file=sys.stderr)

    wait_for_port_free(GRADIO_PORT, timeout_seconds=2.0)


def main() -> int:
    ensure_standard_streams()

    if is_port_in_use(GRADIO_PORT):
        show_error(
            APP_TITLE,
            f"端口 {GRADIO_PORT} 已被占用，无法启动桌面应用。\n"
            "请先关闭正在运行的网页服务或其他占用该端口的程序。",
        )
        return 1

    try:
        start_gradio_server()
    except Exception as exc:
        cleanup_services()
        log_message(f"Startup failed: {exc!r}")
        log_message(traceback.format_exc())
        show_error(APP_TITLE, f"启动失败：{exc}")
        return 1

    try:
        run_desktop_window()
    except Exception as exc:
        log_message(f"Desktop window startup failed: {exc!r}")
        log_message(traceback.format_exc())
        show_error(APP_TITLE, f"桌面窗口启动失败：{exc}")
        return 1
    finally:
        cleanup_services()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
