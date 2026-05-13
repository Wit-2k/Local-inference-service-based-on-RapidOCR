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

from config import LIMIT_SIDE_LEN, MODEL_TYPE

APP_IMPORT_PATH = "ocr_server:app"

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
    # params["Det.thresh"] = 0.1
    # params["Det.box_thresh"] = 0.1
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


def recognize_image(img: np.ndarray) -> dict[str, Any]:
    if engine is None:
        raise RuntimeError("OCR 引擎尚未就绪")

    start = time.perf_counter()
    result = engine(img)
    return serialize_ocr_result(result, (time.perf_counter() - start) * 1000)


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
async def ocr(file: UploadFile = File(...)):
    """接收上传图片，返回 OCR 识别结果"""
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "OCR 引擎尚未就绪"})

    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "无法解码图片"})

    return recognize_image(img)


@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": engine is not None}


@app.post("/shutdown")
async def shutdown(request: Request):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        return JSONResponse(status_code=403, content={"error": "只允许本机关闭 OCR 服务"})

    def stop_process() -> None:
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=stop_process, daemon=True).start()
    return {"status": "shutting_down"}
