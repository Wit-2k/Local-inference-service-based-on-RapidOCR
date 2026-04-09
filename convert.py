import os
import urllib.request

import cv2
from rapidocr_onnxruntime import RapidOCR

# 1. 准备字典文件 (PP-OCRv3/v4/v5 中文模型通用的字典)
dict_path = "ppocr_keys_v1.txt"
if not os.path.exists(dict_path):
    print("首次运行，正在下载字典文件...")
    # 使用 Gitee 官方源以保证国内下载速度
    url = "https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.7/ppocr/utils/ppocr_keys_v1.txt"
    urllib.request.urlretrieve(url, dict_path)
    print("字典文件下载完成！")

# 2. 初始化 RapidOCR 引擎
engine = RapidOCR(
    det_model_path=R"PP-OCRv5_server_det_infer/model.onnx",
    rec_model_path=R"PP-OCRv5_server_rec_infer/model.onnx",
    rec_keys_path=dict_path,  # ✨ 核心修复：明确传入字典文件路径
)

# 3. 读取图像
img_path = R"sign.jpg"
img = cv2.imread(img_path)

if img is None:
    raise ValueError(f"无法读取图片，请检查路径: {img_path}")

# 4. 执行推理
# result 是一个列表，elapse 是各阶段耗时统计
result, elapse = engine(img)

# 5. 解析并打印结果
print(f"==============\n推理耗时统计: {elapse}")

if result is not None:
    for res in result:
        # RapidOCR 返回的格式是元组：(边界框坐标, 识别文本, 置信度)
        box, text, score = res
        print(f"文本: {text} | 置信度: {score:.4f}")
else:
    print("未检测到任何文本")
