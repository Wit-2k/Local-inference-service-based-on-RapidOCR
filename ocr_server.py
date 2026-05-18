import os
import signal
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from rapidocr import EngineType, LangDet, LangRec, OCRVersion, RapidOCR

from chip_preprocess import generate_chip_ocr_variants
from config import LIMIT_SIDE_LEN, MODEL_TYPE

APP_IMPORT_PATH = "ocr_server:app"
PREFERRED_ENHANCE_VARIANT = "laser_dark"

engine: RapidOCR | None = None


def build_ocr_params() -> dict[str, Any]:
    params: dict[str, Any] = {}
    for prefix, lang in [("Det", LangDet.CH), ("Rec", LangRec.CH)]:
        params.update(
            {
                f"{prefix}.engine_type": EngineType.OPENVINO,
                f"{prefix}.lang_type": lang,
                f"{prefix}.model_type": MODEL_TYPE,
                f"{prefix}.ocr_version": OCRVersion.PPOCRV5,
                f"{prefix}.enable_hpi": True,
            }
        )
    params["Det.limit_side_len"] = LIMIT_SIDE_LEN
    return params


def create_ocr_engine() -> RapidOCR:
    return RapidOCR(params=build_ocr_params())


def serialize_ocr_result(result: Any, inference_time_ms: float) -> dict[str, Any]:
    boxes = result.boxes.tolist() if result.boxes is not None else []
    txts = result.txts or []
    scores = getattr(result, "scores", None) or [None] * len(txts)

    return {
        "inference_time_ms": round(inference_time_ms, 2),
        "result": [
            {
                "box": box,
                "text": text,
                "score": round(float(score), 6) if score is not None else None,
            }
            for box, text, score in zip(boxes, txts, scores)
        ],
    }


def text_quality(payload: dict[str, Any]) -> float:
    quality = 0.0
    for item in payload.get("result") or []:
        text = "".join(char for char in item.get("text", "") if char.isalnum())
        score = item.get("score") or 0.0
        quality += len(text) * float(score)
    return quality


def run_ocr(img: np.ndarray) -> dict[str, Any]:
    if engine is None:
        raise RuntimeError("OCR 引擎尚未就绪")

    start = time.perf_counter()
    result = engine(img)
    return serialize_ocr_result(result, (time.perf_counter() - start) * 1000)


def recognize_image(img: np.ndarray, enhance: bool = True) -> dict[str, Any]:
    if not enhance:
        return run_ocr(img)

    start = time.perf_counter()
    variants = generate_chip_ocr_variants(img)
    best_payload: dict[str, Any] | None = None
    best_variant = PREFERRED_ENHANCE_VARIANT
    best_quality = -1.0

    preferred = next(
        (
            (variant_name, variant_image)
            for variant_name, variant_image in variants
            if variant_name == PREFERRED_ENHANCE_VARIANT
        ),
        None,
    )
    if preferred is not None:
        best_variant, preferred_image = preferred
        best_payload = run_ocr(preferred_image)
        best_quality = text_quality(best_payload)
        if best_quality > 0:
            best_payload["inference_time_ms"] = round(
                (time.perf_counter() - start) * 1000, 2
            )
            best_payload["preprocess_variant"] = best_variant
            return best_payload

    for variant_name, variant_image in variants:
        if variant_name == PREFERRED_ENHANCE_VARIANT:
            continue
        payload = run_ocr(variant_image)
        quality = text_quality(payload)
        if quality > best_quality:
            best_payload = payload
            best_variant = variant_name
            best_quality = quality

    if best_payload is None:
        best_payload = {"inference_time_ms": 0.0, "result": []}

    best_payload["inference_time_ms"] = round((time.perf_counter() - start) * 1000, 2)
    best_payload["preprocess_variant"] = best_variant
    return best_payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = create_ocr_engine()
    print("✅ OCR 引擎加载完成")
    yield
    engine = None
    print("👋 OCR 引擎已释放")


app = FastAPI(lifespan=lifespan)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), enhance: bool = True):
    """接收上传图片，返回 OCR 识别结果"""
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "OCR 引擎尚未就绪"})

    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "无法解码图片"})

    return recognize_image(img, enhance=enhance)


@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": engine is not None}


@app.post("/shutdown")
async def shutdown(request: Request):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        return JSONResponse(status_code=403, content={"error": "只允许本机关闭"})

    def stop_process() -> None:
        time.sleep(0.3)          # 等待 HTTP 响应发出
        os._exit(0)              # ✅ 跨平台可靠，直接退出

    threading.Thread(target=stop_process, daemon=True).start()
    return {"status": "shutting_down"}
