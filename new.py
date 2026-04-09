import json
import time

import cv2
from rapidocr import EngineType, LangDet, LangRec, OCRVersion, RapidOCR

from config import LIMIT_SIDE_LEN, MODEL_TYPE


def save_to_json(result):
    """
    提取 RapidOCR 返回结果中的 boxes 和 txts 字段，加上推理时间，将三者保存为 result.json
    """
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

    output = {
        "inference_time_ms": round(elapsed_ms, 2),
        "result": records,
    }

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


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

img_url = R"D:\Default Downloads\page.jpg"
img = cv2.imread(img_url)
if img is None:
    raise ValueError("图片加载失败")

start_time = time.perf_counter()
result = engine(img)
end_time = time.perf_counter()

elapsed_ms = (end_time - start_time) * 1000
print(f"推理时间: {elapsed_ms:.2f} ms")

result.vis("vis_result.jpg")

save_to_json(result)

print("结果已保存至 result.json")
