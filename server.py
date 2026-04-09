import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from rapidocr import EngineType, LangDet, LangRec, OCRVersion, RapidOCR

from config import LIMIT_SIDE_LEN, MODEL_TYPE

# ── 全局引擎，启动时加载一次 ──────────────────────────────────
engine: RapidOCR | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    global engine
    engine = RapidOCR(
        params={
            "Det.engine_type": EngineType.OPENVINO,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": MODEL_TYPE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Det.limit_side_len": LIMIT_SIDE_LEN,
            "Rec.engine_type": EngineType.OPENVINO,
            "Rec.lang_type": LangRec.CH,
            "Rec.model_type": MODEL_TYPE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }
    )
    print("✅ OCR 引擎加载完成")
    yield
    # 关闭时清理
    engine = None
    print("👋 OCR 引擎已释放")


app = FastAPI(lifespan=lifespan)


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    接收上传的图片文件，返回 OCR 识别结果。

    返回 JSON 格式:
    {
        "inference_time_ms": 123.45,
        "result": [
            {
                "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
                "text": "识别文字",
                "score": 0.987
            }
        ]
    }
    """
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "OCR 引擎尚未就绪"},
        )

    # 读取上传的图片字节
    contents = await file.read()

    # 解码图片
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "无法解码图片，请确认上传的是有效的图片文件"},
        )

    # 推理计时
    start_time = time.perf_counter()
    result = engine(img)
    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000

    # 构建结构化结果
    boxes = result.boxes.tolist() if result.boxes is not None else []
    txts = list(result.txts) if result.txts is not None else []
    scores = (
        list(result.scores)
        if hasattr(result, "scores") and result.scores is not None
        else [None] * len(txts)
    )

    records = [
        {
            "box": box,
            "text": txt,
            "score": round(float(score), 6) if score is not None else None,
        }
        for box, txt, score in zip(boxes, txts, scores)
    ]

    return {
        "inference_time_ms": round(elapsed_ms, 2),
        "result": records,
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "ok", "engine_loaded": engine is not None}
