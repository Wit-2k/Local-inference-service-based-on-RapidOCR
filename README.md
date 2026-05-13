# 基于 RapidOCR 的本地推理服务

### 初衷

PaddleOCR 的部署十分麻烦且极易失败，因此选择 RapidOCR 封装的调用方式，推理引擎为 OpenVINO.



### 优势

一步解决定位 + 识别，不用调 YOLO 做定位。



### 项目结构

`ocr_server.py` 为 OCR FastAPI 后端服务。

`config.py` 为配置文件，指定 `MODEL_TYPE` `LIMIT_SIDE_LEN` 两项参数。

`ocr_client.py` 为 OCR 请求客户端，可启动/检测本地 8000 端口服务并提交图片识别。



### 目前的问题

准确率高度依赖 SERVER 模型，MOBILE 模型应用场景十分有限。但 SERVER 模型对手机直出相片的识别时间极长，实时场景下不可用。


