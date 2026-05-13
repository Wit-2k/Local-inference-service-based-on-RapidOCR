"""Gradio 摄像头实时 OCR 展示页。"""

import subprocess
import threading
import time
from typing import Any

import cv2
import gradio as gr
import numpy as np
import requests

from capture import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CAPTURE_INTERVAL_SECONDS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
    FRAME_DIR,
    bgr_to_rgb,
    capture_frame,
    get_camera_info,
    open_camera,
)
from ocr_client import (
    ensure_ocr_service,
    is_ocr_ready,
    recognize_array,
    request_server_shutdown,
    shutdown_server,
)

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
camera_lock = threading.Lock()
camera_capture: cv2.VideoCapture | None = None


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


def start_camera(
    camera_index: int,
    width: int,
    height: int,
) -> tuple[bool, str]:
    """按 capture.py 的方式打开本机摄像头并设置高分辨率。"""
    global camera_capture
    with camera_lock:
        if camera_capture is not None and camera_capture.isOpened():
            camera_capture.release()
            camera_capture = None

        try:
            camera_capture = open_camera(
                camera_index=int(camera_index),
                width=int(width),
                height=int(height),
            )
            info = get_camera_info(camera_capture)
        except RuntimeError as exc:
            camera_capture = None
            return False, f"❌ {exc}"

    return (
        True,
        f"📷 摄像头已打开：{info.width:.0f} x {info.height:.0f}, FPS {info.fps:.1f}",
    )


def stop_camera() -> tuple[bool, str, Any, Any, Any]:
    """释放本机摄像头。"""
    global camera_capture
    with camera_lock:
        if camera_capture is not None:
            camera_capture.release()
            camera_capture = None
            return False, "📷 摄像头已停止", None, gr.update(), gr.update()
    return False, "ℹ️ 摄像头未运行", None, gr.update(), gr.update()


def format_result(result: dict[str, Any], frame_path: str | None) -> str:
    """按 result.json 的结构格式化识别结果。"""
    lines = [f"推理耗时：{result.get('inference_time_ms', 0)} ms"]
    if frame_path:
        lines.append(f"截图：{frame_path}")

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
        cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return annotated


def capture_and_recognize(enabled: bool) -> tuple[Any, Any, str, dict[str, Any]]:
    """用 capture.py 的 OpenCV 截图方式抓取一帧并送 OCR。"""
    with camera_lock:
        if camera_capture is None or not camera_capture.isOpened():
            return None, gr.update(), "等待启动摄像头...", {}

        try:
            frame_bgr, frame_path = capture_frame(camera_capture, frame_dir=FRAME_DIR, save=True)
        except RuntimeError as exc:
            return None, gr.update(), f"❌ {exc}", {}

    frame_rgb = bgr_to_rgb(frame_bgr)
    if not enabled:
        return frame_rgb, gr.update(), "OCR 已暂停，截图仍在更新。", {}

    try:
        if not is_ocr_ready():
            start_background_service()
        result = recognize_array(frame_rgb)
    except (RuntimeError, ValueError, requests.RequestException, TimeoutError, OSError) as exc:
        return frame_rgb, gr.update(), f"❌ 识别失败：{exc}", {}

    annotated = draw_boxes(frame_rgb, result)
    return frame_rgb, annotated, format_result(result, str(frame_path)), result


def set_recognition_enabled(enabled: bool) -> tuple[bool, str]:
    """切换实时 OCR 开关。"""
    if enabled:
        message = start_background_service()
        return True, f"{message}\n▶️ 实时识别已开启"
    return False, "⏸️ 实时识别已暂停，后台服务仍保持运行。"


def initialize_app(camera_index: int, width: int, height: int) -> tuple[bool, bool, str]:
    service_message = start_background_service()
    camera_enabled, camera_message = start_camera(camera_index, width, height)
    return True, camera_enabled, f"{service_message}\n{camera_message}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="实时 OCR 摄像头识别") as demo:
        enabled_state = gr.State(True)
        camera_enabled_state = gr.State(False)
        timer = gr.Timer(value=DEFAULT_CAPTURE_INTERVAL_SECONDS, active=True)

        gr.Markdown(
            "# 实时 OCR 摄像头识别\n"
            "页面按 `capture.py` 的 OpenCV 方式打开本机摄像头，以高分辨率定时截图，保存到 `frames/` 并送 OCR。"
        )

        with gr.Row():
            camera_index = gr.Number(
                value=DEFAULT_CAMERA_INDEX,
                label="摄像头编号",
                precision=0,
                interactive=True,
            )
            frame_width = gr.Number(
                value=DEFAULT_FRAME_WIDTH,
                label="截图宽度",
                precision=0,
                interactive=True,
            )
            frame_height = gr.Number(
                value=DEFAULT_FRAME_HEIGHT,
                label="截图高度",
                precision=0,
                interactive=True,
            )

        with gr.Row():
            frame_preview = gr.Image(type="numpy", label="OpenCV 摄像头截图")
            annotated = gr.Image(type="numpy", label="识别框预览")

        with gr.Row():
            start_camera_button = gr.Button("启动/重启摄像头", variant="primary")
            stop_camera_button = gr.Button("停止摄像头")
            start_button = gr.Button("开启实时识别", variant="primary")
            pause_button = gr.Button("暂停实时识别")
            stop_service_button = gr.Button("停止 OCR 服务", variant="stop")

        status = gr.Textbox(label="状态", value="正在启动 OCR 服务和摄像头...", interactive=False)
        result_text = gr.Textbox(label="识别结果", lines=8, interactive=False)
        result_json = gr.JSON(label="result.json 同构数据")

        demo.load(
            initialize_app,
            inputs=[camera_index, frame_width, frame_height],
            outputs=[enabled_state, camera_enabled_state, status],
        )
        start_camera_button.click(
            start_camera,
            inputs=[camera_index, frame_width, frame_height],
            outputs=[camera_enabled_state, status],
        )
        stop_camera_button.click(
            stop_camera,
            outputs=[camera_enabled_state, status, frame_preview, result_text, result_json],
        )
        start_button.click(
            lambda: set_recognition_enabled(True),
            outputs=[enabled_state, status],
        )
        pause_button.click(
            lambda: set_recognition_enabled(False),
            outputs=[enabled_state, status],
        )
        stop_service_button.click(stop_background_service, outputs=[status, enabled_state])
        timer.tick(
            capture_and_recognize,
            inputs=[enabled_state],
            outputs=[frame_preview, annotated, result_text, result_json],
            concurrency_limit=1,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(css=APP_CSS)
