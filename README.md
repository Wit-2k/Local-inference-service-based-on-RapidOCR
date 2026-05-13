# 基于 RapidOCR 的本地推理服务

### 初衷

PaddleOCR 的部署十分麻烦且极易失败，因此选择 RapidOCR 封装的调用方式，推理引擎为 OpenVINO.



### 优势

一步解决定位 + 识别，不用调 YOLO 做定位。



### 项目结构

`ocr_server.py` 为 OCR FastAPI 后端服务。

`config.py` 为配置文件，指定 `MODEL_TYPE` `LIMIT_SIDE_LEN` 两项参数。

`ocr_client.py` 为 OCR 请求客户端，可启动/检测本地 8000 端口服务并提交图片识别。

`display.py` 为 Gradio 实时页面，复用 `capture.py` 的 OpenCV 摄像头截图方式，显示实时截图、OCR 检测框和 `result.json` 同构识别结果。



### 启动网页

```bash
python display.py
```

页面启动后会自动启动或复用 8000 端口 OCR 服务，并按 `capture.py` 的高分辨率设置打开本机摄像头，定时截图保存到 `frames/` 后送 OCR。服务会常驻后台，暂停实时识别不会关闭服务；只有点击页面中的“停止 OCR 服务”按钮才会请求关闭当前本机 OCR 服务。



### 目前的问题

准确率高度依赖 SERVER 模型，MOBILE 模型应用场景十分有限。但 SERVER 模型对手机直出相片的识别时间极长，实时场景下不可用。




