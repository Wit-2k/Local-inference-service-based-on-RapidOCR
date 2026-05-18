# 项目详细流程与调试说明

本文档面向项目交付、复现和现场调试，说明实时芯片 OCR 识别系统的整体链路、模块职责、调试入口、常见问题和拍摄要求。

## 1. 项目目标

本项目用于本地实时识别摄像头画面中的芯片型号。当前方案不引入 YOLO 等检测模型，而是使用浏览器摄像头原始流、OpenCV 芯片分割、RapidOCR 本地服务和 SQLite/CSV 型号规则库完成识别。

核心目标：

- 浏览器左侧显示摄像头原始流，用作连续预览。
- 后端按固定间隔抽帧识别，不追求每帧 OCR。
- 每帧先检测多个芯片区域，再分别裁剪送 OCR。
- 右侧识别框预览只画芯片级红色旋转框和编号。
- 识别结果按芯片分组显示。
- 只有后端规则库匹配成功的结果才进入底部历史记录。

## 2. 运行入口

常用启动命令：

```bash
uv run display.py
```

页面地址：

```text
http://127.0.0.1:7860
```

OCR 后端服务地址：

```text
http://127.0.0.1:8000
```

两者不是同一个服务：

- `7860` 是 Gradio 页面服务。
- `8000` 是 FastAPI OCR 服务。

健康检查接口：

```text
http://127.0.0.1:8000/health
```

如果返回里 `engine_loaded` 为 `true`，表示 OCR 引擎已加载完成。

## 3. 主要文件职责

### `display.py`

Gradio 页面入口和实时识别总控。

它负责：

- 加载 `display.html`、`display.css`、`display.js`。
- 定义浏览器摄像头参数，例如分辨率和 30fps 预览约束。
- 暴露 Gradio server functions 给前端 JavaScript 调用。
- 接收前端传来的摄像头帧 Data URL。
- 解码摄像头帧为 RGB `numpy.ndarray`。
- 调用 `segment_array_with_metadata()` 做多芯片分割。
- 绘制红色旋转芯片框和编号。
- 并发调用 `ocr_client.recognize_array()`。
- 调用 `chip_db.match_ocr_payload()` 做型号规则匹配。
- 返回页面需要的 JSON 数据。

### `display.html`

页面结构。

当前页面包含：

- 摄像头原始流。
- 识别框预览。
- 启动摄像头、开启实时识别、暂停实时识别、停止 OCR 服务四个按钮。
- 识别结果区。
- 成功识别历史区。
- 一个隐藏 canvas，用于从 `<video>` 抽帧。

### `display.css`

页面样式。

主要控制：

- 双栏预览布局。
- 状态提示 `status-pill`。
- 按钮、结果卡片、历史记录窗口。
- 识别成功型号标签。

### `display.js`

浏览器端摄像头和识别循环。

它负责：

- 使用 `navigator.mediaDevices.getUserMedia()` 获取浏览器摄像头原始流。
- 把摄像头流挂到 `<video>`，左侧连续预览。
- 按 `DEFAULT_CAPTURE_INTERVAL_SECONDS` 定时对 `<video>` 做低分辨率帧差。
- 需要识别时，把 `<video>` 画到隐藏 canvas，并用 `canvas.toDataURL("image/jpeg", quality)` 得到 JPEG Data URL。
- 调用后端 `recognize_multi_chip_frame()`，同时传入本帧是否稳定，供后端记录本次抽帧状态。
- 渲染右侧识别框预览、结果卡片和成功识别历史。

注意：左侧原始流由浏览器直接显示，理论上可以接近摄像头实时帧率；OCR 识别是按固定间隔抽帧，不是 30fps。

### `segment.py`

芯片分割模块。

当前采用纯 OpenCV 方案：

- 灰度化。
- CLAHE 增强。
- 基于暗区阈值和边缘闭运算生成多路候选掩码。
- 对轮廓使用 `cv2.minAreaRect()` 得到旋转候选框。
- 根据旋转面积、长宽比、暗区比例、中心/边缘对比度等规则过滤。
- 使用非极大值去重，尽量保留完整芯片主体，减少只框到引脚的情况。
- 对旋转框做透视裁剪，送给 OCR 的芯片图尽量保持水平。

实时页面使用：

```python
segment_array_with_metadata(image, input_color="rgb", realtime=True)
```

它返回 `SegmentedChip` 列表，每个元素包含：

- `image`：旋转矫正后的芯片裁剪图。
- `rect`：兼容用的水平外接矩形。
- `box_points`：旋转框四点，用于页面绘制红框。
- `angle`：候选角度。
- `score`：分割评分。
- `source`：候选来源掩码。

实时模式会禁用 `edge_*` 边缘候选，只保留较轻的暗区候选，并限制候选数量，目的是减少背景纹理误检和降低每帧耗时。离线 CLI 调试仍使用完整候选模式，方便分析复杂图片。

CLI 调试入口：

```bash
uv run python segment.py images/fourth.jpg chip_crop.jpg --debug
```

带 `--debug` 时会输出：

- 裁剪后的芯片图。
- 画框调试图。
- 各路候选掩码图。

### `ocr_server.py`

FastAPI OCR 服务。

它负责：

- 启动时创建 RapidOCR 引擎。
- 提供 `/health` 健康检查。
- 提供 `/ocr` 上传图片识别接口。
- 提供 `/shutdown` 本机关闭接口。

OCR 参数来自 `config.py`：

- `MODEL_TYPE`：RapidOCR 模型类型。
- `LIMIT_SIDE_LEN`：检测输入边长限制。

`/ocr` 接收图片字节，解码为 OpenCV BGR 图像，然后调用 RapidOCR 返回：

- `inference_time_ms`
- `result`
- 每个文本项的 `box`、`text`、`score`

如果请求参数 `enhance=true`，会调用 `chip_preprocess.py` 生成多种预处理变体，并选取得分最高的 OCR 结果。

### `ocr_client.py`

OCR 服务进程管理和 HTTP 客户端。

它负责：

- 检查 `8000` 端口是否可用。
- 启动 `uvicorn ocr_server:app`。
- 轮询 `/health` 等待 OCR 引擎就绪。
- 把内存中的芯片图编码成 JPEG 并提交给 `/ocr`。
- CLI 模式下识别单张图片并保存 `result.json`。

实时页面调用：

```python
recognize_array(chip.image, save_path=None)
```

这里 `save_path=None` 表示实时流程不保存 `result.json`。

### `chip_db.py` 和 `chip_rules.csv`

芯片型号规则库和 OCR 后处理。

`chip_rules.csv` 是人类可编辑的规则源，交付给别人时优先让对方改这个文件。

格式：

```csv
part_number,pattern,description,priority,enabled
SN74LS00N,74[0-9A-Z]{2}[0O]{2},Quad 2-input NAND gate,10,1
```

字段说明：

- `part_number`：最终显示的芯片型号。
- `pattern`：用于匹配 OCR 清洗文本的正则表达式。
- `description`：型号说明。
- `priority`：优先级，数字越小越先匹配。
- `enabled`：是否启用，支持 `1/0`、`true/false`、`启用/禁用`。

程序启动或匹配时会连接 `chip_rules.sqlite3`。连接时会：

1. 创建 `chip_rules` 表。
2. 读取 `chip_rules.csv`。
3. 比较 CSV 和 SQLite 当前内容。
4. 只有两者不一致时才把 SQLite 同步为 CSV 内容。

因此 SQLite 更像运行时缓存，CSV 才是交付时应维护的规则文件。

匹配前会把 OCR 文本清洗为大写字母数字串，例如：

```text
SN 74LS00 N -> SN74LS00N
```

然后按规则优先级执行正则匹配。

### `chip_preprocess.py`

芯片 OCR 预处理调试模块。

它可以从一张芯片裁剪图生成七种变体：

- `original`
- `body`
- `upscaled`
- `clahe_sharpen`
- `laser_bright`
- `laser_dark`
- `blackhat`

调试入口：

```bash
uv run python chip_preprocess.py chip_crop_03.jpg preprocess_variants/
```

当 OCR 对激光打标、低对比度文字不稳定时，可以先输出这些变体，看哪一种最容易被 OCR 识别。

### `capture.py`

独立 OpenCV 摄像头截图脚本。

当前实时页面已经改为浏览器摄像头流；`capture.py` 主要作为独立调试工具保留。

它负责：

- 使用 OpenCV 打开摄像头。
- 设置默认分辨率。
- 定时保存截图到 `frames/`。
- 清理过期截图。
- 检查摄像头已打开但无法读取的情况，并提示可能被其他程序占用。

## 4. 实时识别完整流程

### 步骤 1：打开页面

运行 `display.py` 后，访问：

```text
http://127.0.0.1:7860
```

页面初始状态不会自动启动 OCR 服务，状态提示为需要手动开启。

### 步骤 2：启动摄像头

点击“启动摄像头”时：

1. `display.js` 调用浏览器 `getUserMedia()`。
2. 浏览器弹出摄像头权限请求。
3. 授权后，左侧 `<video>` 显示摄像头原始流。

这个预览流不经过 Python，不经过 OCR，也不经过 OpenCV，因此应该比后端处理画面更流畅。

### 步骤 3：开启实时识别

点击“开启实时识别”时：

1. 前端确保摄像头已启动。
2. 前端调用 `start_recognition_service()`。
3. 后端检查 OCR 服务是否已经运行。
4. 如果 `8000` 端口没有可用 OCR 服务，则启动 `uvicorn ocr_server:app`。
5. 后端等待 `/health` 返回 OCR 引擎就绪。
6. 前端启动定时识别循环。

定时识别循环不是 30fps，而是按：

```python
DEFAULT_CAPTURE_INTERVAL_SECONDS
```

当前默认值在 `capture.py` 中，为 `1.0` 秒。

### 步骤 4：浏览器抽帧

每次识别循环中：

1. 前端先在低分辨率 motion canvas 上计算帧差。
2. 如果上一帧所有芯片都已匹配型号、且本帧稳定，则直接沿用上次结果，不调用后端。
3. 否则把当前 `<video>` 帧画到隐藏 canvas。
4. 用 JPEG 格式编码为 Data URL。
5. 调用 Python server function：

```text
recognize_multi_chip_frame(frame, recognizing, frame_stable)
```

前端有 `requestInFlight` 保护：如果上一次识别还没返回，不会并发塞入下一帧，避免 OCR 请求堆积。

### 步骤 5：后端解码帧

`display.py` 中：

1. `normalize_frame_request()` 兼容 Gradio 对参数的不同包装方式。
2. `data_url_to_rgb()` 把 Data URL 解码成 OpenCV 图像。
3. 图像转为 RGB，供后续分割使用。

如果 Data URL 格式错误，会返回“摄像头帧无效”之类的状态，而不是直接让页面崩溃。

### 步骤 6：芯片分割

后端调用：

```python
segment_array_with_metadata(frame_rgb, input_color="rgb")
```

输出多个芯片候选。每个候选同时提供：

- 旋转矫正后的裁剪图，用于 OCR。
- 旋转框四点，用于页面红框预览。
- 水平外接矩形，用于兼容字段。

如果没检测到芯片：

- 右侧预览仍会显示当前帧。
- 状态提示“未检测到芯片候选区域”。
- 不会调用 OCR。

### 步骤 7：绘制识别框预览

后端在完整帧副本上绘制：

- 红色旋转四边形。
- `chip 1`、`chip 2` 等编号。

绘制后的图像编码成 Data URL 返回给前端，显示在右侧“识别框预览”。

### 步骤 8：OCR 请求、缓存和服务端排队

如果检测到多个芯片，`display.py` 会先尝试按单芯片 track 复用已成功匹配的 OCR 结果。只有新出现、位置变化过大、芯片图像指纹变化明显，或上一次没有匹配到型号的芯片，才会真实发起 OCR 请求。

需要真实 OCR 的芯片会用线程池并发发起请求：

```python
max_workers = min(4, chip_count)
```

每个芯片调用：

```python
recognize_array(chip.image, save_path=None)
```

实时链路会先把送 OCR 的芯片裁剪图按最长边限制缩放，避免大芯片裁剪图把 RapidOCR 耗时拉得过高。右侧识别框预览也会按最长边限制缩放后再返回前端。

这里的并发是客户端并发请求，不代表服务端会并发推理。当前 `ocr_server.py` 是单个 FastAPI 进程、单个 uvicorn worker、单个全局 RapidOCR engine。`/ocr` 入口虽然是 `async def`，但内部会同步调用 `engine(img)`。因此多个 OCR 请求到达服务端后通常会排队执行。页面状态中的 `OCR xxx ms` 是客户端等待墙钟时间，包含 HTTP 往返、服务端排队等待和实际 RapidOCR 推理时间；三芯片场景下它可能接近多个真实 OCR 请求耗时之和。

后端还有一层单芯片 track 缓存。默认策略是：当前芯片框和历史 track 的重叠度足够高、芯片裁剪图的小尺寸灰度指纹变化足够小、且历史结果已经匹配到型号时，复用该芯片的 OCR 文本和型号匹配结果。缓存命中后，track 会跟随当前芯片框和图像指纹更新；未匹配成功的芯片不会缓存，下一帧会继续真实 OCR。同一位置更换芯片时，图像指纹差异会阻止复用旧型号。

实时流程不会保存每个芯片裁剪图，也不会保存 `result.json`。

### 步骤 9：型号规则匹配

每个芯片 OCR 返回后，后端调用：

```python
match_ocr_payload(ocr_payload)
```

处理流程：

1. 收集 OCR 文本。
2. 拼接并清洗为大写字母数字串。
3. 读取启用的规则。
4. 按 `priority ASC, part_number ASC` 排序匹配。
5. 命中后返回型号、描述、命中的文本和规则。

### 步骤 10：前端渲染结果和历史

前端收到 payload 后：

- 更新状态提示。
- 更新右侧识别框预览。
- 按芯片生成结果卡片。
- 如果某个芯片有 `match.part_number`，则把这次识别加入底部成功识别历史。

历史记录只保存前端内存中的最近记录，刷新页面后会清空。

为了避免同一型号在短时间内刷屏，前端有一个简单去重窗口：

```text
historyDedupeMs = 3000
```

## 5. 按钮行为说明

### 启动摄像头

只启动浏览器摄像头原始流。

不会启动 OCR 服务，也不会开始后端识别。

### 开启实时识别

会执行三件事：

1. 如果摄像头未启动，则先启动摄像头。
2. 请求后端启动或复用 OCR 服务。
3. 启动前端定时抽帧识别循环。

### 暂停实时识别

只停止前端定时抽帧识别循环。

它不会关闭浏览器摄像头，也不会停止 OCR 后端服务。这样再次点击“开启实时识别”时可以更快恢复。

### 停止 OCR 服务

会先停止前端识别循环，然后请求后端释放 `8000` 端口上的 OCR 服务。

摄像头原始流不会被关闭；如果只想释放 OCR 资源，点击这个按钮即可。如果要释放摄像头权限，通常可以关闭页面或刷新后不再启动摄像头。

## 6. 调试建议

### 先分层定位问题

不要一上来同时怀疑摄像头、分割、OCR 和规则库。建议按下面顺序排查：

1. 左侧原始流是否正常。
2. 右侧是否能出现识别框。
3. 识别结果卡片是否有 OCR 文本。
4. OCR 文本清洗后是否接近期望型号。
5. `chip_rules.csv` 的规则是否能匹配清洗文本。

### 摄像头问题

现象：

- 左侧一直显示“摄像头未启动”。
- 浏览器报权限错误。
- 系统相机可用，但网页不可用。

检查：

- 浏览器是否允许 `127.0.0.1:7860` 使用摄像头。
- 摄像头是否被系统相机、会议软件、其他浏览器页面占用。
- 是否选错摄像头设备。
- 页面是否通过 `http://127.0.0.1:7860` 打开，而不是从文件系统打开。

### OCR 服务问题

现象：

- 点击“开启实时识别”后状态长时间停在服务启动中。
- `8000` 端口被占用。
- 停止 OCR 服务后端口没有释放。

检查：

```powershell
netstat -ano | findstr :8000
```

或使用本项目中的端口检查脚本。

也可以直接访问：

```text
http://127.0.0.1:8000/health
```

如果端口被其他 Python 进程占用，需要先确认它是不是当前项目启动的 OCR 服务。不要随手杀掉不确定来源的进程。

### 分割问题

现象：

- 右侧没有红框。
- 只框到引脚。
- 大芯片倾斜后漏检。
- 把背景阴影或杂物误框成芯片。

调试入口：

```bash
uv run python segment.py images/fourth.jpg chip_crop.jpg --debug
```

重点看输出的：

- 原图画框结果。
- `dark` 和 `dark_close_*` 掩码。
- `edge_*` 掩码。

常调参数在 `segment.py`：

- `min_area_ratio`：候选最小面积比例。
- `max_area_ratio`：候选最大面积比例。
- `max_aspect_ratio`：候选最大长宽比。
- `min_score`：候选最低评分。
- `padding_ratio`：旋转框外扩比例。
- `DARK_CLOSE_KERNELS`：暗区闭运算核尺寸。
- `EDGE_CLOSE_KERNELS`：边缘闭运算核尺寸。
- `SHADOW_*`：贴近画面边缘的大块低纹理阴影过滤阈值。

如果大芯片只框到引脚，通常要看暗区闭运算是否把主体连起来，或候选评分是否偏向局部高对比区域。

如果把摄像头、手机或灯架的阴影误框成芯片，常见特征是：候选框贴近画面右边或下边、面积较大、内部纹理平滑、边缘密度很低。此时优先调整光照和摆位，让阴影离开识别区域；代码里的 `is_shadow_like_candidate()` 会作为兜底过滤这类贴边阴影。

### OCR 文本问题

现象：

- 红框正确，但结果显示“未识别到文本”。
- OCR 只识别到个别字符。
- `0` 和 `O`、`1` 和 `I` 混淆。

调试入口：

```bash
uv run python chip_preprocess.py chip_crop_03.jpg preprocess_variants/
```

查看七种预处理图，判断文字在哪种图上最清楚。

如果某种预处理明显更好，可以考虑在 `/ocr?enhance=true` 流程里启用，或调整 `generate_chip_ocr_variants()` 的变体。

### 型号匹配问题

现象：

- OCR 文本看起来已经接近型号，但页面仍显示“未匹配到型号”。

检查：

1. 结果卡片里的“清洗文本”是什么。
2. `chip_rules.csv` 中是否有对应型号。
3. `pattern` 是否能匹配清洗文本。
4. `enabled` 是否为 `1` 或 `启用`。
5. 优先级是否被更早规则抢先命中。

例如 OCR 常把 `00` 识别为 `0O`，规则可以写成：

```regex
74[0-9A-Z]{2}[0O]{2}
```

### 前端参数问题

前端定时抽帧间隔来自：

```python
DEFAULT_CAPTURE_INTERVAL_SECONDS
```

摄像头约束来自 `display.py` 中的：

```python
WEBCAM_CONSTRAINTS
```

JPEG 质量来自：

```python
JPEG_QUALITY
```

实时识别性能相关参数还包括：

```python
REALTIME_MAX_CHIPS
OCR_MAX_IMAGE_SIDE
PREVIEW_MAX_IMAGE_SIDE
OCR_CACHE_IOU_THRESHOLD
CHIP_FINGERPRINT_SIZE
CHIP_FINGERPRINT_MEAN_THRESHOLD
CHIP_FINGERPRINT_PIXEL_THRESHOLD
CHIP_FINGERPRINT_CHANGED_RATIO_THRESHOLD
FRAME_DIFF_WIDTH
FRAME_DIFF_MEAN_THRESHOLD
FRAME_DIFF_PIXEL_THRESHOLD
FRAME_DIFF_CHANGED_RATIO_THRESHOLD
```

前端会先用低分辨率帧差判断画面是否稳定。只有上一帧检测到的所有芯片都已经匹配到型号，并且连续抽帧的平均亮度差和变化像素比例都低于阈值时，本次才不会调用后端识别流程。页面会沿用上一次完整识别结果，包括右侧识别框预览和芯片结果卡片。

注意这里有两种不同的“复用”：

- 前端帧差复用：整帧画面稳定且上一帧所有芯片都已匹配型号时，直接不调用后端，省掉上传、解码、分割、OCR 和预览图编码；只要仍有芯片未匹配成功，就继续调用后端。
- 后端单芯片 track 复用：画面已经进入后端分割后，逐个芯片比较位置和图像指纹。位置稳定、图像内容稳定、且历史结果已匹配型号的芯片会复用 OCR 结果；未匹配成功、新出现、明显移动或内容变化的芯片会真实 OCR。

这些值会在 `build_webrtc_js()` 中注入到 `display.js`。

修改后需要重启 `display.py` 并刷新浏览器页面。

## 7. 拍摄要求

为了让分割和 OCR 稳定，拍摄环境比算法细节更重要。建议按下面要求布置。

### 背景

- 使用浅色、哑光、纹理少的背景。
- 避免木纹、网格纸、反光桌面、强阴影。
- 芯片周围留出空白，不要贴近画面边缘。
- 多个芯片之间留明显间距，避免候选区域粘连。

### 光照

- 使用均匀漫射光。
- 避免强点光源直射芯片表面。
- 避免芯片上的大面积高光、反光和过曝。
- 如果激光打标很暗，可以略微侧光，但不要让引脚阴影盖住文字。

### 角度

- 芯片可以小角度旋转，当前分割支持旋转框。
- 但不建议透视角过大，也就是不要让芯片一端明显远离镜头。
- 摄像头尽量垂直于桌面。
- 如果大芯片一边虚、一边清楚，说明平面角度或景深不合适。

### 对焦

- 文字区域必须清晰，而不是只看芯片轮廓清晰。
- 大芯片和小芯片同时拍摄时，尽量让它们处于同一平面高度。
- 如果摄像头自动对焦来回跳，可以先固定芯片位置和光照，再启动识别。

### 尺寸

- 芯片文字区域不要太小。
- 芯片主体也不要占满画面，建议单个大芯片最长边不超过画面宽度的 60%。
- 多芯片场景下，每个芯片都需要留出完整边缘和引脚区域。

### 摆放

- DIP 芯片尽量让长边接近水平或小角度倾斜。
- 不要让芯片互相遮挡。
- 不要让手、镊子、包装袋进入识别区域。
- 如果只想识别某一个芯片，先把其他深色物体移出画面。

## 8. 常见现象解释

### 左侧很流畅，右侧更新慢

这是正常现象。

左侧是浏览器直接显示摄像头原始流，目标是预览流畅。右侧是后端抽帧、分割、OCR、规则匹配后的结果，受 OCR 模型速度影响，不会是 30fps。

### 暂停后摄像头还在动

这是正常行为。

“暂停实时识别”只停止抽帧识别循环，不关闭摄像头预览，也不停止 OCR 服务。

### 停止 OCR 服务后页面还能看到摄像头

这是正常行为。

摄像头原始流由浏览器管理，OCR 服务由 Python 后端管理。停止 OCR 服务只释放 `8000` 端口和 OCR 资源。

### 有 OCR 文本但没有历史记录

历史记录只记录“型号匹配成功”的结果。

如果 OCR 识别到了文本，但没有命中 `chip_rules.csv` 中的启用规则，不会进入成功识别历史。

### `result.json` 没有更新

实时页面不会保存 `result.json`。

`result.json` 是旧的 CLI 兼容行为，实时流程中调用 OCR 时使用 `save_path=None`。

## 9. 建议的调试流程

### 调试分割

1. 用页面或系统相机截一张清晰图片。
2. 运行：

```bash
uv run python segment.py your_image.jpg chip_crop.jpg --debug
```

3. 查看画框图是否完整框住芯片。
4. 查看掩码图判断芯片主体是否被连成完整区域。
5. 再微调 `segment.py` 参数。

### 调试 OCR

1. 先确认 `chip_crop.jpg` 是完整芯片或完整文字区域。
2. 运行：

```bash
uv run python ocr_client.py chip_crop.jpg
```

3. 如果结果差，再运行：

```bash
uv run python chip_preprocess.py chip_crop.jpg preprocess_variants/
```

4. 对比七种变体，判断是否需要启用增强流程。

### 调试规则匹配

1. 看页面结果卡片中的“清洗文本”。
2. 在 `chip_rules.csv` 添加或调整规则。
3. 重启页面服务或触发下一次数据库连接。
4. 再识别同一芯片。

注意：SQLite 会在连接时和 CSV 同步。交付或调试时不要手工改 `chip_rules.sqlite3` 作为长期配置。

### 调试服务停止

点击“停止 OCR 服务”后，检查端口：

```powershell
netstat -ano | findstr :8000
```

如果仍被占用：

- 先确认 PID 是否为本项目启动的 Python。
- 如果是旧残留进程，再手动结束。
- 如果是其他程序，不要直接杀进程，应先确认来源。

## 10. 交付注意事项

交付给他人时，至少包含：

- Python 项目文件。
- `pyproject.toml` 和 `uv.lock`。
- `display.py`、`display.html`、`display.css`、`display.js`。
- `ocr_server.py`、`ocr_client.py`、`segment.py`、`chip_db.py`、`chip_preprocess.py`。
- `chip_rules.csv`。
- 必要的模型依赖或安装说明。

不要把下面内容当作必须交付的配置源：

- `chip_rules.sqlite3`：运行时缓存，可由 CSV 生成。
- `result.json`：CLI 临时结果。
- `frames/`：独立截图脚本生成的临时截图。
- `preprocess_variants/`：预处理调试输出。
- `*_box.jpg`、`*_dark*.jpg`、`*_edge*.jpg`：分割调试输出。

## 11. 快速自检命令

语法检查：

```bash
uv run python -m compileall segment.py display.py ocr_client.py ocr_server.py chip_db.py chip_preprocess.py
```

分割单图：

```bash
uv run python segment.py images/fourth.jpg chip_crop.jpg --debug
```

输出 OCR 预处理变体：

```bash
uv run python chip_preprocess.py chip_crop.jpg preprocess_variants/
```

启动页面：

```bash
uv run python display.py
```

检查 OCR 服务：

```text
http://127.0.0.1:8000/health
```
