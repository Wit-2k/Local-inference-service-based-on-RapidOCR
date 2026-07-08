"""Gradio 摄像头实时 OCR 展示页。"""

import base64
import copy
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

DEFAULT_CAPTURE_INTERVAL_SECONDS = 1.0
DEFAULT_FRAME_HEIGHT = 1440
DEFAULT_FRAME_WIDTH = 2560

from chip_db import match_ocr_payload
from ocr_client import (
    ensure_ocr_service,
    is_ocr_ready,
    recognize_array,
)
from runtime_paths import resource_path
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
REALTIME_MAX_CHIPS = 3
OCR_MAX_IMAGE_SIDE = 512
OCR_LARGE_CHIP_MAX_IMAGE_SIDE = 768
OCR_LARGE_CHIP_SIDE_THRESHOLD = 560
OCR_LARGE_CHIP_ASPECT_THRESHOLD = 3.2
PREVIEW_MAX_IMAGE_SIDE = 1280
OCR_CACHE_IOU_THRESHOLD = 0.72
CHIP_FINGERPRINT_SIZE = (96, 32)
CHIP_FINGERPRINT_MEAN_THRESHOLD = 3.0
CHIP_FINGERPRINT_PIXEL_THRESHOLD = 14
CHIP_FINGERPRINT_CHANGED_RATIO_THRESHOLD = 0.03
FRAME_DIFF_WIDTH = 160
FRAME_DIFF_MEAN_THRESHOLD = 2.5
FRAME_DIFF_PIXEL_THRESHOLD = 12
FRAME_DIFF_CHANGED_RATIO_THRESHOLD = 0.015
JPEG_QUALITY = 86
WEBCAM_CONSTRAINTS = {
    "video": {
        "width": {"ideal": DEFAULT_FRAME_WIDTH},
        "height": {"ideal": DEFAULT_FRAME_HEIGHT},
        "frameRate": {"ideal": RAW_CAMERA_FPS, "max": RAW_CAMERA_FPS},
    },
    "audio": False,
}
DISPLAY_HTML_PATH = resource_path("display.html")
DISPLAY_CSS_PATH = resource_path("display.css")
DISPLAY_JS_PATH = resource_path("display.js")


def load_text_asset(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_webrtc_js() -> str:
    replacements = {
        "__WEBCAM_CONSTRAINTS__": json.dumps(WEBCAM_CONSTRAINTS, ensure_ascii=False),
        "__CAPTURE_INTERVAL_MS__": str(int(DEFAULT_CAPTURE_INTERVAL_SECONDS * 1000)),
        "__JPEG_QUALITY__": f"{JPEG_QUALITY / 100:.2f}",
        "__FRAME_DIFF_WIDTH__": str(FRAME_DIFF_WIDTH),
        "__FRAME_DIFF_MEAN_THRESHOLD__": f"{FRAME_DIFF_MEAN_THRESHOLD:.3f}",
        "__FRAME_DIFF_PIXEL_THRESHOLD__": str(FRAME_DIFF_PIXEL_THRESHOLD),
        "__FRAME_DIFF_CHANGED_RATIO_THRESHOLD__": (
            f"{FRAME_DIFF_CHANGED_RATIO_THRESHOLD:.5f}"
        ),
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
ocr_cache_lock = threading.Lock()
ocr_result_cache: list[dict[str, Any]] = []


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
    clear_ocr_result_cache()
    return {"status": start_background_service(user_requested=True)}


def is_ocr_stop_requested() -> bool:
    with server_lock:
        return ocr_stop_requested


def stop_background_service() -> tuple[str, bool]:
    global server_process, ocr_stop_requested
    with server_lock:
        ocr_stop_requested = True

        server_process = None
        clear_ocr_result_cache()

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
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.split()[-1]
                r = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", pid],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
    except Exception as e:
        print(f"[kill_by_port] 异常: {e}")


def stop_ocr_service_for_ui(*args, **kwargs) -> dict[str, str]:
    """Gradio 回调可能传入参数，用 *args 兜住。"""
    status, _ = stop_background_service()
    return {"status": status}


def normalize_frame_request(
    frame_payload: Any, enabled: bool = True, frame_stable: bool = False
) -> tuple[str, bool, bool]:
    """兼容 Gradio HTML server function 对多参数的打包方式。"""
    payload = frame_payload
    request_enabled = enabled
    request_frame_stable = frame_stable

    for _ in range(4):
        if isinstance(payload, dict):
            if "enabled" in payload:
                request_enabled = bool(payload["enabled"])
            if "frame_stable" in payload:
                request_frame_stable = bool(payload["frame_stable"])
            if "allow_cache" in payload:
                request_frame_stable = bool(payload["allow_cache"])
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
            if len(payload) >= 3:
                request_frame_stable = bool(payload[2])
            payload = payload[0]
            continue

        break

    if not isinstance(payload, str):
        raise ValueError(f"摄像头帧格式无效：收到 {type(payload).__name__}")

    return payload, request_enabled, request_frame_stable


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


def clear_ocr_result_cache() -> None:
    with ocr_cache_lock:
        ocr_result_cache.clear()


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)


def resize_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if max_side <= 0 or long_side <= max_side:
        return image

    scale = max_side / long_side
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def ocr_max_side_for_chip(image: np.ndarray) -> int:
    h, w = image.shape[:2]
    long_side = max(h, w)
    short_side = max(1, min(h, w))
    aspect_ratio = long_side / short_side
    if (
        long_side >= OCR_LARGE_CHIP_SIDE_THRESHOLD
        or aspect_ratio >= OCR_LARGE_CHIP_ASPECT_THRESHOLD
    ):
        return OCR_LARGE_CHIP_MAX_IMAGE_SIDE
    return OCR_MAX_IMAGE_SIDE


def rect_area_dict(rect: dict[str, int]) -> int:
    return max(0, int(rect["w"])) * max(0, int(rect["h"]))


def rect_iou_dict(rect_a: dict[str, int], rect_b: dict[str, int]) -> float:
    ax1, ay1 = int(rect_a["x"]), int(rect_a["y"])
    ax2, ay2 = ax1 + int(rect_a["w"]), ay1 + int(rect_a["h"])
    bx1, by1 = int(rect_b["x"]), int(rect_b["y"])
    bx2, by2 = bx1 + int(rect_b["w"]), by1 + int(rect_b["h"])

    left = max(ax1, bx1)
    top = max(ay1, by1)
    right = min(ax2, bx2)
    bottom = min(ay2, by2)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0

    union = rect_area_dict(rect_a) + rect_area_dict(rect_b) - intersection
    return intersection / union if union > 0 else 0.0


def rect_dict_from_chip(chip: SegmentedChip) -> dict[str, int]:
    x, y, w, h = chip.rect
    return {"x": x, "y": y, "w": w, "h": h}


def box_points_payload(chip: SegmentedChip) -> list[dict[str, int]]:
    return [{"x": px, "y": py} for px, py in chip.box_points]


def chip_fingerprint(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"不支持的芯片图像形状: {image.shape}")

    return cv2.resize(
        gray,
        CHIP_FINGERPRINT_SIZE,
        interpolation=cv2.INTER_AREA,
    )


def fingerprint_difference(
    current: np.ndarray,
    cached: np.ndarray,
) -> tuple[float, float]:
    if current.shape != cached.shape:
        return float("inf"), 1.0

    delta = cv2.absdiff(current, cached)
    mean_delta = float(delta.mean())
    changed_ratio = float(
        np.count_nonzero(delta > CHIP_FINGERPRINT_PIXEL_THRESHOLD) / delta.size
    )
    return mean_delta, changed_ratio


def is_fingerprint_stable(mean_delta: float, changed_ratio: float) -> bool:
    return (
        mean_delta < CHIP_FINGERPRINT_MEAN_THRESHOLD
        and changed_ratio < CHIP_FINGERPRINT_CHANGED_RATIO_THRESHOLD
    )


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
    return {
        "index": index,
        "rect": rect_dict_from_chip(chip),
        "box_points": box_points_payload(chip),
        "angle": chip.angle,
        "segment_score": chip.score,
        "segment_source": chip.source,
        "texts": texts,
        "match": match,
        "inference_time_ms": (ocr_payload or {}).get("inference_time_ms"),
        "error": error,
    }


def cached_chip_result_payload(
    index: int,
    chip: SegmentedChip,
    cached_entry: dict[str, Any],
    *,
    cache_age_ms: float,
    cache_iou: float,
    fingerprint_mean_delta: float,
    fingerprint_changed_ratio: float,
) -> dict[str, Any]:
    return {
        "index": index,
        "rect": rect_dict_from_chip(chip),
        "box_points": box_points_payload(chip),
        "angle": chip.angle,
        "segment_score": chip.score,
        "segment_source": chip.source,
        "texts": copy.deepcopy(cached_entry["texts"]),
        "match": copy.deepcopy(cached_entry["match"]),
        "inference_time_ms": cached_entry.get("inference_time_ms"),
        "error": None,
        "cached": True,
        "timings": {
            "ocr_wall_ms": 0.0,
            "cache_age_ms": cache_age_ms,
            "cache_iou": round(cache_iou, 4),
            "fingerprint_mean_delta": round(fingerprint_mean_delta, 3),
            "fingerprint_changed_ratio": round(fingerprint_changed_ratio, 4),
            "cached": True,
        },
    }


def recognize_chip(index: int, chip: SegmentedChip) -> dict[str, Any]:
    started_at = time.perf_counter()
    ocr_max_side = ocr_max_side_for_chip(chip.image)
    ocr_image = resize_long_side(chip.image, ocr_max_side)
    try:
        payload = recognize_array(ocr_image, save_path=None, enhance=True)
    except (
        RuntimeError,
        ValueError,
        requests.RequestException,
        TimeoutError,
        OSError,
    ) as exc:
        result = chip_result_payload(index, chip, None, error=f"识别失败：{exc}")
        result["timings"] = {"ocr_wall_ms": elapsed_ms(started_at)}
        return result

    result = chip_result_payload(index, chip, payload)
    result["timings"] = {
        "ocr_wall_ms": elapsed_ms(started_at),
        "ocr_input_width": int(ocr_image.shape[1]),
        "ocr_input_height": int(ocr_image.shape[0]),
        "ocr_input_max_side": ocr_max_side,
        "ocr_enhance_enabled": True,
        "ocr_preprocess_variant": payload.get("preprocess_variant"),
        "cached": False,
    }
    return result


def cached_results_for_chips(
    chips: list[SegmentedChip],
) -> dict[int, dict[str, Any]]:
    now = time.monotonic()
    with ocr_cache_lock:
        entries = copy.deepcopy(ocr_result_cache)

    results: dict[int, dict[str, Any]] = {}
    used_cache_indexes: set[int] = set()
    track_updates: list[tuple[int, SegmentedChip, np.ndarray]] = []
    for index, chip in enumerate(chips, start=1):
        rect = rect_dict_from_chip(chip)
        fingerprint = chip_fingerprint(chip.image)
        best_index: int | None = None
        best_iou = 0.0
        best_mean_delta = float("inf")
        best_changed_ratio = 1.0
        for cache_index, entry in enumerate(entries):
            if cache_index in used_cache_indexes:
                continue
            iou = rect_iou_dict(rect, entry["rect"])
            if iou < OCR_CACHE_IOU_THRESHOLD:
                continue
            mean_delta, changed_ratio = fingerprint_difference(
                fingerprint, entry["fingerprint"]
            )
            if not is_fingerprint_stable(mean_delta, changed_ratio):
                continue
            if iou > best_iou:
                best_iou = iou
                best_index = cache_index
                best_mean_delta = mean_delta
                best_changed_ratio = changed_ratio

        if best_index is None:
            continue

        used_cache_indexes.add(best_index)
        track_updates.append((best_index, chip, fingerprint))
        cache_age_ms = round(
            (now - float(entries[best_index]["recognized_at"])) * 1000, 2
        )
        results[index] = cached_chip_result_payload(
            index,
            chip,
            entries[best_index],
            cache_age_ms=cache_age_ms,
            cache_iou=best_iou,
            fingerprint_mean_delta=best_mean_delta,
            fingerprint_changed_ratio=best_changed_ratio,
        )

    if track_updates:
        with ocr_cache_lock:
            for cache_index, chip, fingerprint in track_updates:
                if cache_index >= len(ocr_result_cache):
                    continue
                ocr_result_cache[cache_index]["rect"] = rect_dict_from_chip(chip)
                ocr_result_cache[cache_index]["box_points"] = box_points_payload(chip)
                ocr_result_cache[cache_index]["fingerprint"] = fingerprint.copy()
                ocr_result_cache[cache_index]["updated_at"] = now

    return results


def cache_ocr_result(chip: SegmentedChip, result: dict[str, Any]) -> None:
    if not result.get("match", {}).get("part_number"):
        return

    now = time.monotonic()
    entry = {
        "rect": rect_dict_from_chip(chip),
        "box_points": box_points_payload(chip),
        "fingerprint": chip_fingerprint(chip.image),
        "texts": copy.deepcopy(result.get("texts", [])),
        "match": copy.deepcopy(result.get("match", {})),
        "inference_time_ms": result.get("inference_time_ms"),
        "recognized_at": now,
        "updated_at": now,
    }
    with ocr_cache_lock:
        best_index: int | None = None
        best_iou = 0.0
        for cache_index, existing in enumerate(ocr_result_cache):
            iou = rect_iou_dict(entry["rect"], existing["rect"])
            if iou > best_iou:
                best_iou = iou
                best_index = cache_index

        if best_index is not None and best_iou >= OCR_CACHE_IOU_THRESHOLD:
            ocr_result_cache[best_index] = entry
        else:
            ocr_result_cache.append(entry)
        ocr_result_cache[:] = ocr_result_cache[-REALTIME_MAX_CHIPS:]


def recognize_chips_with_cache(chips: list[SegmentedChip]) -> list[dict[str, Any]]:
    if not chips:
        return []

    results_by_index = cached_results_for_chips(chips)
    chips_to_recognize = [
        (index, chip)
        for index, chip in enumerate(chips, start=1)
        if index not in results_by_index
    ]

    if not chips_to_recognize:
        return [results_by_index[index] for index in sorted(results_by_index)]

    max_workers = min(MAX_OCR_WORKERS, len(chips_to_recognize))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(recognize_chip, index, chip): index
            for index, chip in chips_to_recognize
        }
        for future in as_completed(futures):
            result = future.result()
            index = futures[future]
            results_by_index[index] = result
            cache_ocr_result(chips[index - 1], result)

    return [results_by_index[index] for index in sorted(results_by_index)]


def recognize_multi_chip_frame(
    data_url: Any, enabled: bool = True, frame_stable: bool = False
) -> dict[str, Any]:
    started_at = time.perf_counter()
    timings: dict[str, Any] = {}

    def finish(
        *,
        status: str,
        summary: str,
        chips: list[dict[str, Any]] | None = None,
        annotated_image: str | None = None,
    ) -> dict[str, Any]:
        timings["total_ms"] = elapsed_ms(started_at)
        return {
            "status": status,
            "summary": summary,
            "chips": chips or [],
            "annotated_image": annotated_image,
            "timings": dict(timings),
        }

    try:
        step_started = time.perf_counter()
        data_url, enabled, frame_stable = normalize_frame_request(
            data_url, enabled, frame_stable
        )
        timings["request_ms"] = elapsed_ms(step_started)
        timings["frame_stable"] = frame_stable
    except ValueError as exc:
        return finish(status=f"摄像头帧无效：{exc}", summary="未完成识别")

    if not enabled:
        return finish(status="实时识别已暂停", summary="暂无结果")

    try:
        step_started = time.perf_counter()
        frame_rgb = data_url_to_rgb(data_url)
        timings["decode_ms"] = elapsed_ms(step_started)
    except ValueError as exc:
        return finish(status=f"摄像头帧无效：{exc}", summary="未完成识别")

    try:
        step_started = time.perf_counter()
        chips = segment_array_with_metadata(
            frame_rgb,
            input_color="rgb",
            # max_chips=REALTIME_MAX_CHIPS,
            realtime=True,
        )
        timings["segment_ms"] = elapsed_ms(step_started)
    except RuntimeError as exc:
        return finish(status=f"分割失败：{exc}", summary="未完成识别")

    step_started = time.perf_counter()
    annotated = draw_chip_boxes(frame_rgb, chips) if chips else frame_rgb
    annotated_image = rgb_to_data_url(
        resize_long_side(annotated, PREVIEW_MAX_IMAGE_SIDE)
    )
    timings["preview_ms"] = elapsed_ms(step_started)

    if not chips:
        return finish(
            status="未检测到芯片候选区域",
            summary="检测到 0 个芯片",
            annotated_image=annotated_image,
        )

    step_started = time.perf_counter()
    if not is_ocr_ready():
        if is_ocr_stop_requested():
            timings["service_ms"] = elapsed_ms(step_started)
            return finish(
                status="OCR 服务已手动停止",
                summary=f"检测到 {len(chips)} 个芯片，OCR 服务未运行",
                chips=[
                    chip_result_payload(index, chip, None, error="OCR 服务已停止")
                    for index, chip in enumerate(chips, start=1)
                ],
                annotated_image=annotated_image,
            )
        service_status = start_background_service()
        if not is_ocr_ready():
            timings["service_ms"] = elapsed_ms(step_started)
            return finish(
                status=service_status,
                summary=f"检测到 {len(chips)} 个芯片，OCR 服务不可用",
                chips=[
                    chip_result_payload(index, chip, None, error="OCR 服务不可用")
                    for index, chip in enumerate(chips, start=1)
                ],
                annotated_image=annotated_image,
            )
    timings["service_ms"] = elapsed_ms(step_started)

    step_started = time.perf_counter()
    chip_results = recognize_chips_with_cache(chips)
    timings["ocr_ms"] = elapsed_ms(step_started)
    cached_count = sum(1 for item in chip_results if item.get("cached"))
    timings["cache_hits"] = cached_count
    timings["ocr_real_count"] = len(chip_results) - cached_count
    total_ms = elapsed_ms(started_at)
    matched_count = sum(1 for item in chip_results if item["match"].get("part_number"))
    cache_text = f"，复用 {cached_count}" if cached_count else ""
    status = (
        f"检测到 {len(chips)} 个芯片，用时 {total_ms} ms"
        f"（分割 {timings.get('segment_ms', 0)} ms，"
        f"OCR {timings.get('ocr_ms', 0)} ms{cache_text}）"
    )
    return finish(
        status=status,
        summary=f"{len(chips)} 个芯片，{matched_count} 个型号匹配",
        chips=chip_results,
        annotated_image=annotated_image,
    )


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
