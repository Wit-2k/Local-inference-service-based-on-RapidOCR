# 配置文件

from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

# MODEL_TYPE = ModelType.SERVER
MODEL_TYPE = ModelType.MOBILE  # 只要准确率过关就用 MOBILE
LIMIT_SIDE_LEN: int = 480  # 默认值 736
