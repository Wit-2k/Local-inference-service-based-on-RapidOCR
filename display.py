"""
Gradio 摄像头实时 OCR 展示页。

操作流程：
1. 点击 Click to Access Webcam
2. 点击“开始实时识别”
3. 点击“录制”
"""

import os
import subprocess
import threading
from typing import Any

import cv2
import gradio as gr
import numpy as np
import requests

from capture import (
    DEFAULT_CAPTURE_INTERVAL_SECONDS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
)
from ocr_client import (
    ensure_ocr_service,
    is_ocr_ready,
    recognize_array,
    request_server_shutdown,
    shutdown_server,
)

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


GRADIO_SERVER_NAME = "127.0.0.1"
GRADIO_SERVER_PORT = 7860
RAW_CAMERA_FPS = 30.0
WEBCAM_CONSTRAINTS = {
    "video": {
        "width": {"ideal": DEFAULT_FRAME_WIDTH},
        "height": {"ideal": DEFAULT_FRAME_HEIGHT},
        "frameRate": {"ideal": RAW_CAMERA_FPS, "max": RAW_CAMERA_FPS},
    }
}
APP_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at 10% 10%, rgba(79, 255, 176, 0.12), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(77, 166, 255, 0.14), transparent 34%),
        linear-gradient(135deg, #07100d 0%, #0b1218 48%, #050607 100%);
    color: #e8fff5;
}
.gradio-container h1 {
    letter-spacing: -0.04em;
    font-weight: 850;
}
.gradio-container .contain, .gradio-container .block {
    border-color: rgba(135, 255, 204, 0.18) !important;
}
button.primary {
    box-shadow: 0 0 22px rgba(83, 255, 181, 0.28) !important;
}
textarea, .json-holder {
    font-family: Consolas, 'Cascadia Mono', monospace !important;
}
"""

server_process: subprocess.Popen | None = None
server_lock = threading.Lock()


def start_background_service() -> str:
    """启动或复用 OCR 服务，并保持后台常驻。"""
    global server_process
    with server_lock:
        if is_ocr_ready():
            return "✅ OCR 服务已在运行"

        if server_process is not None and server_process.poll() is None:
            return "⏳ OCR 服务正在启动"

        try:
            server_process = ensure_ocr_service()
        except (RuntimeError, TimeoutError, OSError) as exc:
            return f"❌ OCR 服务启动失败：{exc}"
        if server_process is None:
            return "✅ 已连接到现有 OCR 服务"
        return "✅ OCR 服务已启动并常驻后台"


def stop_background_service() -> tuple[str, bool]:
    """仅在网页按钮触发时停止 OCR 服务。"""
    global server_process
    with server_lock:
        stopped = shutdown_server(server_process)
        server_process = None

    if stopped:
        return "🛑 OCR 服务已手动停止", False
    if is_ocr_ready():
        if request_server_shutdown():
            return "🛑 已请求现有 OCR 服务停止", False
        return "⚠️ 当前 OCR 服务正在运行，但关闭请求失败", False
    return "ℹ️ OCR 服务未运行", False


def format_result(result: dict[str, Any]) -> str:
    """按 result.json 的结构格式化识别结果。"""
    lines = [f"推理耗时：{result.get('inference_time_ms', 0)} ms"]

    items = result.get("result") or []
    if not items:
        lines.append("未识别到文本")
        return "\n".join(lines)

    for index, item in enumerate(items, start=1):
        text = item.get("text", "")
        score = item.get("score")
        if score is None:
            lines.append(f"{index}. {text}")
        else:
            lines.append(f"{index}. {text}  置信度：{score:.4f}")
    return "\n".join(lines)


def draw_boxes(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """在摄像头画面上绘制 OCR 检测框。"""
    annotated = image.copy()
    for item in result.get("result") or []:
        box = item.get("box")
        if not box:
            continue
        points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            annotated, [points], isClosed=True, color=(0, 255, 0), thickness=2
        )
    return annotated


def recognize_stream_frame(
    enabled: bool, frame_rgb: np.ndarray | None
) -> tuple[Any, str, dict[str, Any]]:
    """识别浏览器 WebRTC 摄像头流抽取的当前帧。"""
    if frame_rgb is None:
        return gr.update(), "等待浏览器摄像头画面...", {}

    if not enabled:
        return gr.update(), "OCR 已暂停，摄像头原始流仍在更新。", {}

    try:
        if not is_ocr_ready():
            start_background_service()
        result = recognize_array(frame_rgb)
    except (
        RuntimeError,
        ValueError,
        requests.RequestException,
        TimeoutError,
        OSError,
    ) as exc:
        return gr.update(), f"❌ 识别失败：{exc}", {}

    annotated = draw_boxes(frame_rgb, result)
    return annotated, format_result(result), result


def set_recognition_enabled(enabled: bool) -> tuple[bool, str]:
    """切换实时 OCR 开关。"""
    if enabled:
        message = start_background_service()
        return True, f"{message}\n▶️ 实时识别已开启"
    return False, "⏸️ 实时识别已暂停，后台服务仍保持运行。"


def initialize_app() -> tuple[bool, str]:
    service_message = start_background_service()
    return True, f"{service_message}\n📷 请在摄像头原始流中允许浏览器访问摄像头"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="实时 OCR 摄像头识别") as demo:
        enabled_state = gr.State(True)

        gr.Markdown(
            "# 实时 OCR 摄像头识别\n"
            "页面运行在 `127.0.0.1:7860`，OCR 服务运行在 `127.0.0.1:8000`。"
            "页面使用浏览器摄像头原始流预览，并按固定间隔抽帧送 OCR。"
        )

        with gr.Row():
            frame_preview = gr.Image(
                type="numpy",
                label="摄像头原始流",
                sources=["webcam"],
                streaming=True,
                interactive=True,
                webcam_options=gr.WebcamOptions(
                    mirror=False,
                    constraints=WEBCAM_CONSTRAINTS,
                ),
            )
            annotated = gr.Image(type="numpy", label="识别框预览")

        with gr.Row():
            start_button = gr.Button("开启实时识别", variant="primary")
            pause_button = gr.Button("暂停实时识别")
            stop_service_button = gr.Button("停止 OCR 服务", variant="stop")

        status = gr.Textbox(
            label="状态", value="正在启动 OCR 服务...", interactive=False
        )
        result_text = gr.Textbox(label="识别结果", lines=8, interactive=False)
        result_json = gr.JSON(label="result.json 同构数据")

        demo.load(
            initialize_app,
            outputs=[enabled_state, status],
        )
        start_button.click(
            lambda: set_recognition_enabled(True),
            outputs=[enabled_state, status],
        )
        pause_button.click(
            lambda: set_recognition_enabled(False),
            outputs=[enabled_state, status],
        )
        stop_service_button.click(
            stop_background_service, outputs=[status, enabled_state]
        )
        frame_preview.stream(
            recognize_stream_frame,
            inputs=[enabled_state, frame_preview],
            outputs=[annotated, result_text, result_json],
            concurrency_limit=1,
            show_progress="hidden",
            stream_every=DEFAULT_CAPTURE_INTERVAL_SECONDS,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=GRADIO_SERVER_PORT,
        share=False,
        css=APP_CSS,
    )
