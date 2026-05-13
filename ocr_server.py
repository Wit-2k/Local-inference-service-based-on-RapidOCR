import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from rapidocr import EngineType, LangDet, LangRec, OCRVersion, RapidOCR

from config import LIMIT_SIDE_LEN, MODEL_TYPE

engine: RapidOCR | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    params = {}
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
    engine = RapidOCR(params=params)
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

    start = time.perf_counter()
    result = engine(img)

    # 真正重要的只有 txts
    boxes = result.boxes.tolist() if result.boxes is not None else []
    txts = result.txts or []
    scores = getattr(result, "scores", None) or [None] * len(txts)

    return {
        "inference_time_ms": round((time.perf_counter() - start) * 1000, 2),
        "result": [
            {
                "box": b,
                "text": t,
                "score": round(float(s), 6) if s is not None else None,
            }
            for b, t, s in zip(boxes, txts, scores)
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": engine is not None}
