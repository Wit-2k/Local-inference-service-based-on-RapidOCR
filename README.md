# 基于 RapidOCR 的本地推理服务

### 初衷

PaddleOCR 的部署十分麻烦且极易失败，因此选择 RapidOCR 封装的调用方式，推理引擎为 OpenVINO.

### 优势

一步解决文本检测和文本识别，不用调 YOLO 做定位。

### 项目结构

`ocr_server.py` 为 OCR FastAPI 后端服务。

`config.py` 为配置文件，指定 `MODEL_TYPE` `LIMIT_SIDE_LEN` 两项参数。

`ocr_client.py` 为 OCR 请求客户端，可启动/检测本地 8000 端口服务并提交图片识别。

`display.py` 为 Gradio 实时页面，复用 `capture.py` 的 OpenCV 摄像头取帧方式，显示实时画面、OCR 检测框和 `result.json` 同构识别结果；识别过程不再保存临时截图文件。

`segment.py` 为芯片分割预处理脚本，可从一张输入图中截取多个芯片；当检测到多个芯片时会输出为 `输出名_01.jpg`、`输出名_02.jpg` 等文件。

`capture.py` 作为独立截图脚本运行时仍会保存截图到 `frames/`，并会定时清理旧截图，避免文件无限增加。

### 分割芯片

```bash
python segment.py images/fourth.jpg chip_crop.jpg
```

默认会输出调试图；不需要调试图时可以使用：

```bash
python segment.py images/fourth.jpg chip_crop.jpg --no-debug
```

### 启动网页

```bash
python display.py
```

页面地址默认是 `http://127.0.0.1:7860`，这是 Gradio 网页端口；`http://127.0.0.1:8000` 是 OCR FastAPI 服务端口，两者不是同一个服务。

页面启动后会自动启动或复用 8000 端口 OCR 服务，并按 `capture.py` 的高分辨率设置打开本机摄像头，定时从内存帧直接送 OCR，不会把每帧保存到 `frames/`。服务会常驻后台，暂停实时识别不会关闭服务；只有点击页面中的“停止 OCR 服务”按钮才会请求关闭当前本机 OCR 服务。

### 目前的问题

只有 SERVER 模型能满足高准确率要求，MOBILE 模型的精度相对有限。但 SERVER 模型对手机直出相片的识别时间极长，实时场景下不可用。
