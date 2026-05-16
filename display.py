"""Gradio 摄像头实时 OCR 展示页。"""

import base64
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from chip_db import match_ocr_payload
from ocr_client import (
    ensure_ocr_service,
    is_ocr_ready,
    recognize_array,
    request_server_shutdown,
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
APP_CSS = """
.gradio-container {
    background:
        linear-gradient(135deg, #faf7f2 0%, #f5f0e8 48%, #faf8f5 100%);
    color: #2c2c2c;
}
.gradio-container .contain, .gradio-container .block {
    border-color: rgba(180, 160, 130, 0.25) !important;
}
#chip-ocr-app {
    color: #2c2c2c;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
}
#chip-ocr-app .app-header {
    align-items: end;
    display: flex;
    gap: 18px;
    justify-content: space-between;
    margin-bottom: 18px;
}
#chip-ocr-app h1 {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 0;
    margin: 0;
}
#chip-ocr-app .status-pill {
    background: rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: 999px;
    color: #4a4a4a;
    min-width: 260px;
    padding: 10px 14px;
    text-align: right;
}
#chip-ocr-app .workspace {
    display: grid;
    gap: 22px;
    grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr);
}
#chip-ocr-app .panel {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(180, 160, 130, 0.30);
    border-radius: 8px;
    overflow: hidden;
}
#chip-ocr-app .panel-title {
    align-items: center;
    background: rgba(0, 0, 0, 0.04);
    border-bottom: 1px solid rgba(180, 160, 130, 0.25);
    display: flex;
    font-size: 15px;
    font-weight: 700;
    height: 38px;
    padding: 0 14px;
}
#chip-ocr-app .media-slot {
    align-items: center;
    aspect-ratio: 16 / 9;
    background: #f8faf9;
    display: flex;
    justify-content: center;
    position: relative;
}
#chip-ocr-app video,
#chip-ocr-app .annotated-image {
    height: 100%;
    object-fit: contain;
    width: 100%;
}
#chip-ocr-app .annotated-image.is-empty {
    opacity: 0;
}
#chip-ocr-app .empty-state {
    color: #6e7774;
    font-size: 15px;
    position: absolute;
}
#chip-ocr-app .controls {
    display: grid;
    gap: 18px;
    grid-template-columns: repeat(4, minmax(150px, 1fr));
    margin: 22px 0;
}
#chip-ocr-app button {
    border: 0;
    border-radius: 8px;
    color: #08100d;
    cursor: pointer;
    font-size: 18px;
    font-weight: 800;
    min-height: 58px;
}
#chip-ocr-app button:disabled {
    cursor: wait;
    opacity: 0.58;
}
#chip-ocr-app .camera-button {
    background: #e9f5ef;
}
#chip-ocr-app .start-button {
    background: #ff7a18;
}
#chip-ocr-app .pause-button {
    background: #e4e6e8;
}
#chip-ocr-app .stop-button {
    background: #f2434d;
    color: #fff;
}
#chip-ocr-app .results-panel {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 8px;
    color: #121918;
    padding: 18px;
}
#chip-ocr-app .results-header {
    align-items: center;
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
}
#chip-ocr-app .results-header h2 {
    font-size: 18px;
    letter-spacing: 0;
    margin: 0;
}
#chip-ocr-app .summary {
    color: #54615d;
    font-size: 14px;
}
#chip-ocr-app .chip-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
#chip-ocr-app .chip-card {
    border: 1px solid #d8e0dd;
    border-radius: 8px;
    padding: 14px;
}
#chip-ocr-app .chip-card h3 {
    font-size: 16px;
    letter-spacing: 0;
    margin: 0 0 8px;
}
#chip-ocr-app .match {
    color: #b91c1c;
    font-weight: 800;
}
#chip-ocr-app .muted {
    color: #66736f;
}
#chip-ocr-app .ocr-line {
    font-family: Consolas, 'Cascadia Mono', monospace;
    margin-top: 6px;
}
#chip-ocr-app .history-panel {
    background: rgba(255, 255, 255, 0.96);
    border-radius: 8px;
    color: #121918;
    margin-top: 18px;
    padding: 18px;
}
#chip-ocr-app .history-window {
    display: grid;
    gap: 10px;
    max-height: 340px;
    overflow-y: auto;
    padding-right: 4px;
}
#chip-ocr-app .history-empty {
    border: 1px dashed #cbd5d1;
    border-radius: 8px;
    color: #66736f;
    padding: 22px;
    text-align: center;
}
#chip-ocr-app .history-record {
    align-items: center;
    border: 1px solid #d8e0dd;
    border-radius: 8px;
    display: grid;
    gap: 14px;
    grid-template-columns: 180px minmax(0, 1fr) auto;
    padding: 10px;
}
#chip-ocr-app .history-thumb {
    aspect-ratio: 16 / 9;
    background: #eef1ef;
    border-radius: 6px;
    object-fit: cover;
    width: 100%;
}
#chip-ocr-app .history-models {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
#chip-ocr-app .model-pill {
    background: #fff1e6;
    border: 1px solid #ffc58d;
    border-radius: 999px;
    color: #9a3412;
    font-weight: 800;
    padding: 6px 10px;
}
#chip-ocr-app .history-time {
    color: #54615d;
    font-family: Consolas, 'Cascadia Mono', monospace;
    font-size: 13px;
    white-space: nowrap;
}
@media (max-width: 900px) {
    #chip-ocr-app .app-header,
    #chip-ocr-app .workspace {
        display: block;
    }
    #chip-ocr-app .status-pill {
        margin-top: 12px;
        min-width: 0;
        text-align: left;
    }
    #chip-ocr-app .panel + .panel {
        margin-top: 18px;
    }
    #chip-ocr-app .controls {
        grid-template-columns: 1fr 1fr;
    }
    #chip-ocr-app .history-record {
        align-items: start;
        grid-template-columns: 1fr;
    }
    #chip-ocr-app .history-time {
        white-space: normal;
    }
}
"""

WEBRTC_HTML = """
<div id="chip-ocr-app">
    <div class="app-header">
        <h1 style="color: #2c2c2c;">实时 OCR 摄像头识别</h1>
        <div class="status-pill" data-role="status">正在启动 OCR 服务...</div>
    </div>
    <div class="workspace">
        <section class="panel">
            <div class="panel-title">摄像头原始流</div>
            <div class="media-slot">
                <video data-role="video" autoplay playsinline muted></video>
                <div class="empty-state" data-role="camera-empty">摄像头未启动</div>
            </div>
        </section>
        <section class="panel">
            <div class="panel-title">识别框预览</div>
            <div class="media-slot">
                <img class="annotated-image is-empty" data-role="annotated" alt="识别框预览" />
                <div class="empty-state" data-role="preview-empty">等待识别结果</div>
            </div>
        </section>
    </div>
    <div class="controls">
        <button class="camera-button" data-role="camera-button">启动摄像头</button>
        <button class="start-button" data-role="start-button">开启实时识别</button>
        <button class="pause-button" data-role="pause-button">暂停实时识别</button>
        <button class="stop-button" data-role="stop-button">停止 OCR 服务</button>
    </div>
    <section class="results-panel">
        <div class="results-header">
            <h2>识别结果</h2>
            <span class="summary" data-role="summary">暂无结果</span>
        </div>
        <div class="chip-grid" data-role="results"></div>
    </section>
    <section class="history-panel">
        <div class="results-header">
            <h2>成功识别历史</h2>
            <span class="summary" data-role="history-summary">0 条记录</span>
        </div>
        <div class="history-window" data-role="history">
            <div class="history-empty" data-role="history-empty">暂无匹配记录</div>
        </div>
    </section>
    <canvas data-role="canvas" hidden></canvas>
</div>
"""

WEBRTC_JS = f"""
const app = element.querySelector('#chip-ocr-app');
const video = app.querySelector('[data-role="video"]');
const canvas = app.querySelector('[data-role="canvas"]');
const statusEl = app.querySelector('[data-role="status"]');
const summaryEl = app.querySelector('[data-role="summary"]');
const resultsEl = app.querySelector('[data-role="results"]');
const historyEl = app.querySelector('[data-role="history"]');
const historySummaryEl = app.querySelector('[data-role="history-summary"]');
const historyEmpty = app.querySelector('[data-role="history-empty"]');
const annotatedEl = app.querySelector('[data-role="annotated"]');
const cameraEmpty = app.querySelector('[data-role="camera-empty"]');
const previewEmpty = app.querySelector('[data-role="preview-empty"]');
const cameraButton = app.querySelector('[data-role="camera-button"]');
const startButton = app.querySelector('[data-role="start-button"]');
const pauseButton = app.querySelector('[data-role="pause-button"]');
const stopButton = app.querySelector('[data-role="stop-button"]');
const constraints = {json.dumps(WEBCAM_CONSTRAINTS)};
const intervalMs = {int(DEFAULT_CAPTURE_INTERVAL_SECONDS * 1000)};
const jpegQuality = {JPEG_QUALITY / 100:.2f};
const maxHistoryRecords = 20;
const historyDedupeMs = 3000;
let stream = null;
let recognizeTimer = null;
let recognizing = false;
let requestInFlight = false;
let historyRecordCount = 0;
let lastHistorySignature = '';
let lastHistoryAt = 0;

function setStatus(message) {{
    statusEl.textContent = message;
}}

function normalizeServerPayload(value, fallback = {{}}) {{
    if (value == null) {{
        return {{...fallback}};
    }}
    if (Array.isArray(value)) {{
        if (value.length === 0) return {{...fallback}};
        if (value.length === 1) return normalizeServerPayload(value[0], fallback);
        return {{...fallback, data: value}};
    }}
    if (typeof value === 'object') {{
        if ('data' in value) return normalizeServerPayload(value.data, fallback);
        if ('value' in value) return normalizeServerPayload(value.value, fallback);
        return {{...fallback, ...value}};
    }}
    return {{...fallback, status: String(value)}};
}}

async function callServer(fnName, args = [], fallback = {{}}) {{
    if (!server || typeof server[fnName] !== 'function') {{
        return {{...fallback}};
    }}
    const value = await server[fnName](...args);
    return normalizeServerPayload(value, fallback);
}}

function setPreview(src) {{
    if (src) {{
        annotatedEl.src = src;
        annotatedEl.classList.remove('is-empty');
        previewEmpty.style.display = 'none';
    }} else {{
        annotatedEl.removeAttribute('src');
        annotatedEl.classList.add('is-empty');
        previewEmpty.style.display = 'block';
    }}
}}

function clearResults(message = '暂无结果') {{
    summaryEl.textContent = message;
    resultsEl.replaceChildren();
}}

function addText(parent, tag, text, className = '') {{
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    parent.appendChild(node);
    return node;
}}

function matchedModels(payload) {{
    const models = [];
    for (const chip of payload.chips || []) {{
        const partNumber = chip.match && chip.match.part_number;
        if (partNumber && !models.includes(partNumber)) {{
            models.push(partNumber);
        }}
    }}
    return models;
}}

function formatRecordTime(date) {{
    return date.toLocaleString('zh-CN', {{
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    }});
}}

function updateHistorySummary() {{
    const count = historyEl.querySelectorAll('.history-record').length;
    historySummaryEl.textContent = `${{count}} 条记录`;
    if (historyEmpty) {{
        historyEmpty.style.display = count ? 'none' : 'block';
    }}
}}

function appendMatchedHistory(payload) {{
    const models = matchedModels(payload);
    if (!models.length || !payload.annotated_image) return;

    const now = Date.now();
    const signature = models.join('|');
    if (signature === lastHistorySignature && now - lastHistoryAt < historyDedupeMs) {{
        return;
    }}
    lastHistorySignature = signature;
    lastHistoryAt = now;

    const record = document.createElement('article');
    record.className = 'history-record';
    record.dataset.historyId = String(++historyRecordCount);

    const image = document.createElement('img');
    image.className = 'history-thumb';
    image.alt = '识别框画面';
    image.src = payload.annotated_image;
    record.appendChild(image);

    const modelWrap = document.createElement('div');
    modelWrap.className = 'history-models';
    for (const model of models) {{
        addText(modelWrap, 'span', model, 'model-pill');
    }}
    record.appendChild(modelWrap);

    addText(record, 'time', formatRecordTime(new Date(now)), 'history-time');
    historyEl.prepend(record);

    const records = historyEl.querySelectorAll('.history-record');
    for (let index = maxHistoryRecords; index < records.length; index += 1) {{
        records[index].remove();
    }}
    updateHistorySummary();
}}

function renderResults(payload) {{
    payload = normalizeServerPayload(payload, {{
        status: '未收到识别结果',
        summary: '暂无结果',
        chips: [],
        annotated_image: null,
    }});
    if (payload.annotated_image) setPreview(payload.annotated_image);
    if (payload.status) setStatus(payload.status);
    summaryEl.textContent = payload.summary || '暂无结果';
    resultsEl.replaceChildren();

    for (const chip of payload.chips || []) {{
        const card = document.createElement('article');
        card.className = 'chip-card';
        addText(card, 'h3', `芯片 ${{chip.index}}`);

        const match = chip.match || {{}};
        if (match.part_number) {{
            addText(card, 'div', `匹配型号：${{match.part_number}}`, 'match');
            if (match.description) addText(card, 'div', match.description, 'muted');
        }} else {{
            addText(card, 'div', '未匹配到型号', 'muted');
        }}

        if (chip.error) {{
            addText(card, 'div', chip.error, 'muted');
        }} else if (chip.texts && chip.texts.length) {{
            for (const item of chip.texts) {{
                const score = item.score == null ? '' : `  (${{Number(item.score).toFixed(4)}})`;
                addText(card, 'div', `${{item.text}}${{score}}`, 'ocr-line');
            }}
        }} else {{
            addText(card, 'div', '未识别到文本', 'muted');
        }}

        if (match.normalized_text) {{
            addText(card, 'div', `清洗文本：${{match.normalized_text}}`, 'muted');
        }}
        resultsEl.appendChild(card);
    }}
    appendMatchedHistory(payload);
}}

async function startCamera() {{
    if (stream) return;
    cameraButton.disabled = true;
    try {{
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        await video.play();
        cameraEmpty.style.display = 'none';
        setStatus('摄像头原始流已启动');
    }} catch (error) {{
        stream = null;
        setStatus(`摄像头启动失败：${{error.message || error}}`);
    }} finally {{
        cameraButton.disabled = false;
    }}
}}

function stopRecognitionLoop() {{
    recognizing = false;
    if (recognizeTimer) {{
        window.clearInterval(recognizeTimer);
        recognizeTimer = null;
    }}
}}

async function captureAndRecognize() {{
    if (!recognizing || requestInFlight || !video.videoWidth || !video.videoHeight) {{
        return;
    }}

    requestInFlight = true;
    try {{
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL('image/jpeg', jpegQuality);
        const payload = await callServer(
            'recognize_multi_chip_frame',
            [frame, recognizing],
            {{
                status: '未收到识别结果',
                summary: '暂无结果',
                chips: [],
                annotated_image: null,
            }}
        );
        renderResults(payload);
    }} catch (error) {{
        setStatus(`识别失败：${{error.message || error}}`);
    }} finally {{
        requestInFlight = false;
    }}
}}

async function startRecognition() {{
    startButton.disabled = true;
    try {{
        await startCamera();
        const service = await callServer(
            'start_recognition_service',
            [],
            {{status: 'OCR 服务接口未返回状态'}}
        );
        if (service.status) setStatus(service.status);
        recognizing = true;
        if (!recognizeTimer) {{
            recognizeTimer = window.setInterval(captureAndRecognize, intervalMs);
        }}
        await captureAndRecognize();
    }} catch (error) {{
        setStatus(`启动失败：${{error.message || error}}`);
    }} finally {{
        startButton.disabled = false;
    }}
}}

function pauseRecognition() {{
    stopRecognitionLoop();
    setStatus('实时识别已暂停');
}}

async function stopService() {{
    stopRecognitionLoop();
    stopButton.disabled = true;
    try {{
        const payload = await callServer(
            'stop_ocr_service_for_ui',
            [],
            {{status: 'OCR 服务已停止'}}
        );
        setStatus(payload.status || 'OCR 服务已停止');
    }} catch (error) {{
        setStatus(`停止失败：${{error.message || error}}`);
    }} finally {{
        stopButton.disabled = false;
    }}
}}

cameraButton.addEventListener('click', startCamera);
startButton.addEventListener('click', startRecognition);
pauseButton.addEventListener('click', pauseRecognition);
stopButton.addEventListener('click', stopService);

callServer(
    'start_recognition_service',
    [],
    {{status: 'OCR 服务接口未返回状态'}}
).then((payload) => {{
    if (payload.status) setStatus(payload.status);
}}).catch((error) => {{
    setStatus(`OCR 服务启动失败：${{error.message || error}}`);
}});
"""

server_process: subprocess.Popen | None = None
server_lock = threading.Lock()


def start_background_service() -> str:
    """启动或复用 OCR 服务，并保持后台常驻。"""
    global server_process
    with server_lock:
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


def start_recognition_service() -> dict[str, str]:
    return {"status": start_background_service()}


def stop_background_service() -> tuple[str, bool]:
    """仅在网页按钮触发时停止 OCR 服务。"""
    global server_process
    with server_lock:
        stopped = shutdown_server(server_process)
        server_process = None

    if stopped:
        return "OCR 服务已手动停止", False
    if is_ocr_ready():
        if request_server_shutdown():
            return "已请求现有 OCR 服务停止", False
        return "当前 OCR 服务正在运行，但关闭请求失败", False
    return "OCR 服务未运行", False


def stop_ocr_service_for_ui() -> dict[str, str]:
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
