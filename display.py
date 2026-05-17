"""Gradio 摄像头实时 OCR 展示页。"""

import base64
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import requests
import platform
import signal


from capture import (
    DEFAULT_CAPTURE_INTERVAL_SECONDS,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_WIDTH,
)
from chip_db import match_ocr_payload
from ocr_client import (
    ensure_ocr_service,
    is_ocr_ready,
    recognize_array,
    shutdown_server,
)
from segment import SegmentedChip, segment_array_with_metadata

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
MAX_OCR_WORKERS = 4
JPEG_QUALITY = 86
WEBCAM_CONSTRAINTS = {
    "video": {
        "width": {"ideal": DEFAULT_FRAME_WIDTH},
        "height": {"ideal": DEFAULT_FRAME_HEIGHT},
        "frameRate": {"ideal": RAW_CAMERA_FPS, "max": RAW_CAMERA_FPS},
    },
    "audio": False,
}
UI_DIR = Path(__file__).resolve().parent
DISPLAY_HTML_PATH = UI_DIR / "display.html"
DISPLAY_CSS_PATH = UI_DIR / "display.css"
DISPLAY_JS_PATH = UI_DIR / "display.js"


def load_text_asset(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_webrtc_js() -> str:
    replacements = {
        "__WEBCAM_CONSTRAINTS__": json.dumps(WEBCAM_CONSTRAINTS, ensure_ascii=False),
        "__CAPTURE_INTERVAL_MS__": str(int(DEFAULT_CAPTURE_INTERVAL_SECONDS * 1000)),
        "__JPEG_QUALITY__": f"{JPEG_QUALITY / 100:.2f}",
    }
    js = load_text_asset(DISPLAY_JS_PATH)
    for token, value in replacements.items():
        js = js.replace(token, value)
    return js


APP_CSS = load_text_asset(DISPLAY_CSS_PATH)

WEBRTC_HTML = load_text_asset(DISPLAY_HTML_PATH)

WEBRTC_JS = build_webrtc_js()

server_process: subprocess.Popen | None = None
server_lock = threading.Lock()
ocr_stop_requested = False


def start_background_service(*, user_requested: bool = False) -> str:
    """启动或复用 OCR 服务。"""
    global server_process, ocr_stop_requested
    with server_lock:
        # ✅ 只有用户主动点击"启动"才清除停止标记
        if user_requested:
            ocr_stop_requested = False
        elif ocr_stop_requested:
            return "OCR 服务已手动停止，需手动重新启动"

        if is_ocr_ready():
            return "OCR 服务已在运行"

        if server_process is not None and server_process.poll() is None:
            return "OCR 服务正在启动"

        try:
            server_process = ensure_ocr_service()
        except (RuntimeError, TimeoutError, OSError) as exc:
            return f"OCR 服务启动失败：{exc}"

        if server_process is None:
            return "已连接到现有 OCR 服务"

        return "OCR 服务已启动"


def start_recognition_service(*args, **kwargs) -> dict[str, str]:
    return {"status": start_background_service(user_requested=True)}


def is_ocr_stop_requested() -> bool:
    with server_lock:
        return ocr_stop_requested


def stop_background_service() -> tuple[str, bool]:
    global server_process, ocr_stop_requested
    with server_lock:
        ocr_stop_requested = True

        server_process = None

        # 兜底：按端口杀
        time.sleep(1)
        if _is_port_in_use(8000):
            _kill_by_port(8000)
            time.sleep(1)

        if _is_port_in_use(8000):
            return "停止失败：端口 8000 仍被占用", False
        return "OCR 服务已停止，端口 8000 已释放", False

def _is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill_by_port(port: int):
    """兜底：直接找到占用端口的 PID 并强杀。"""
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.split()[-1]
                r = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", pid],
                    capture_output=True, text=True, timeout=10,
                )
    except Exception as e:
        print(f"[kill_by_port] 异常: {e}")


def stop_ocr_service_for_ui(*args, **kwargs) -> dict[str, str]:
    """Gradio 回调可能传入参数，用 *args 兜住。"""
    status, _ = stop_background_service()
    return {"status": status}


def normalize_frame_request(
    frame_payload: Any, enabled: bool = True
) -> tuple[str, bool]:
    """兼容 Gradio HTML server function 对多参数的打包方式。"""
    payload = frame_payload
    request_enabled = enabled

    for _ in range(4):
        if isinstance(payload, dict):
            if "enabled" in payload:
                request_enabled = bool(payload["enabled"])
            if "data_url" in payload:
                payload = payload["data_url"]
                continue
            if "data" in payload:
                payload = payload["data"]
                continue
            if "value" in payload:
                payload = payload["value"]
                continue
            break

        if isinstance(payload, (list, tuple)):
            if not payload:
                break
            if len(payload) >= 2:
                request_enabled = bool(payload[1])
            payload = payload[0]
            continue

        break

    if not isinstance(payload, str):
        raise ValueError(f"摄像头帧格式无效：收到 {type(payload).__name__}")

    return payload, request_enabled


def data_url_to_rgb(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("摄像头帧格式无效")

    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("无法解码摄像头帧")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_data_url(image: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        raise ValueError("无法编码识别框预览")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def draw_chip_boxes(image: np.ndarray, chips: list[SegmentedChip]) -> np.ndarray:
    annotated = image.copy()
    line_width = max(2, min(image.shape[:2]) // 280)
    font_scale = max(0.6, min(image.shape[:2]) / 900)
    for index, chip in enumerate(chips, start=1):
        points = np.array(chip.box_points, dtype=np.int32)
        cv2.polylines(
            annotated,
            [points.reshape((-1, 1, 2))],
            isClosed=True,
            color=(255, 0, 0),
            thickness=line_width,
        )
        x = int(points[:, 0].min())
        y = int(points[:, 1].min())
        label = f"chip {index}"
        baseline = 0
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
        )
        label_top = max(0, y - text_h - baseline - 8)
        cv2.rectangle(
            annotated,
            (x, label_top),
            (x + text_w + 10, label_top + text_h + baseline + 8),
            (255, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x + 5, label_top + text_h + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )
    return annotated


def chip_result_payload(
    index: int,
    chip: SegmentedChip,
    ocr_payload: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    texts = []
    if ocr_payload is not None:
        for item in ocr_payload.get("result") or []:
            texts.append({"text": item.get("text", ""), "score": item.get("score")})

    match = match_ocr_payload(ocr_payload or {"result": []})
    x, y, w, h = chip.rect
    return {
        "index": index,
        "rect": {"x": x, "y": y, "w": w, "h": h},
        "box_points": [{"x": px, "y": py} for px, py in chip.box_points],
        "angle": chip.angle,
        "segment_score": chip.score,
        "segment_source": chip.source,
        "texts": texts,
        "match": match,
        "inference_time_ms": (ocr_payload or {}).get("inference_time_ms"),
        "error": error,
    }


def recognize_chip(index: int, chip: SegmentedChip) -> dict[str, Any]:
    try:
        payload = recognize_array(chip.image, save_path=None)
    except (
        RuntimeError,
        ValueError,
        requests.RequestException,
        TimeoutError,
        OSError,
    ) as exc:
        return chip_result_payload(index, chip, None, error=f"识别失败：{exc}")
    return chip_result_payload(index, chip, payload)


def recognize_chips_parallel(
    chips: list[SegmentedChip],
) -> list[dict[str, Any]]:
    if not chips:
        return []

    max_workers = min(MAX_OCR_WORKERS, len(chips))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(recognize_chip, index, chip): index
            for index, chip in enumerate(chips, start=1)
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item["index"])


def recognize_multi_chip_frame(data_url: Any, enabled: bool = True) -> dict[str, Any]:
    try:
        data_url, enabled = normalize_frame_request(data_url, enabled)
    except ValueError as exc:
        return {
            "status": f"摄像头帧无效：{exc}",
            "summary": "未完成识别",
            "chips": [],
            "annotated_image": None,
        }

    if not enabled:
        return {
            "status": "实时识别已暂停",
            "summary": "暂无结果",
            "chips": [],
            "annotated_image": None,
        }

    started_at = time.perf_counter()
    try:
        frame_rgb = data_url_to_rgb(data_url)
    except ValueError as exc:
        return {
            "status": f"摄像头帧无效：{exc}",
            "summary": "未完成识别",
            "chips": [],
            "annotated_image": None,
        }

    try:
        chips = segment_array_with_metadata(frame_rgb, input_color="rgb")
    except RuntimeError as exc:
        return {
            "status": f"分割失败：{exc}",
            "summary": "未完成识别",
            "chips": [],
            "annotated_image": None,
        }

    annotated = draw_chip_boxes(frame_rgb, chips) if chips else frame_rgb
    annotated_image = rgb_to_data_url(annotated)

    if not chips:
        return {
            "status": "未检测到芯片候选区域",
            "summary": "检测到 0 个芯片",
            "chips": [],
            "annotated_image": annotated_image,
        }

    if not is_ocr_ready():
        if is_ocr_stop_requested():
            return {
                "status": "OCR 服务已手动停止",
                "summary": f"检测到 {len(chips)} 个芯片，OCR 服务未运行",
                "chips": [
                    chip_result_payload(index, chip, None, error="OCR 服务已停止")
                    for index, chip in enumerate(chips, start=1)
                ],
                "annotated_image": annotated_image,
            }
        service_status = start_background_service()
        if not is_ocr_ready():
            return {
                "status": service_status,
                "summary": f"检测到 {len(chips)} 个芯片，OCR 服务不可用",
                "chips": [
                    chip_result_payload(index, chip, None, error="OCR 服务不可用")
                    for index, chip in enumerate(chips, start=1)
                ],
                "annotated_image": annotated_image,
            }

    chip_results = recognize_chips_parallel(chips)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    matched_count = sum(1 for item in chip_results if item["match"].get("part_number"))
    return {
        "status": f"检测到 {len(chips)} 个芯片，用时 {elapsed_ms} ms",
        "summary": f"{len(chips)} 个芯片，{matched_count} 个型号匹配",
        "chips": chip_results,
        "annotated_image": annotated_image,
    }


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.HTML(
            WEBRTC_HTML,
            js_on_load=WEBRTC_JS,
            server_functions=[
                start_recognition_service,
                stop_ocr_service_for_ui,
                recognize_multi_chip_frame,
            ],
            container=False,
            padding=False,
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
