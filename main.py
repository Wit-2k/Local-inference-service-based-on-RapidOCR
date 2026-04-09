import cv2
from rapidocr_onnxruntime import RapidOCR

if __name__ == "__main__":
    dict_path = R".\ppocrv5_dict.txt"

    # 1. 初始化 RapidOCR 引擎，直接加载你的 ONNX 模型文件
    engine = RapidOCR(
        det_model_path=R"PP-OCRv5_server_det_infer/model.onnx",
        rec_model_path=R"PP-OCRv5_server_rec_infer/model.onnx",
        rec_keys_path=dict_path,
    )

    # 2. 读取图像
    img_path = R"sign.jpg"
    img = cv2.imread(img_path)

    # 3. 执行推理
    # 返回的 result 是一个列表，elapse 是各阶段耗时统计
    result, elapse = engine(img)

    # 4. 解析并打印结果
    print(f"推理耗时统计: {elapse}")

    if result is not None:
        for res in result:
            # RapidOCR 返回的格式是元组：(边界框坐标, 识别文本, 置信度)
            box, text, score = res
            print(f"文本: {text} | 置信度: {score:.4f} | 框坐标: {box}")
    else:
        print("未检测到文本")
