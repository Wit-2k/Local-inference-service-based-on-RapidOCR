# 🔬 ChipOCR — 基于 RapidOCR 的本地实时芯片识别系统

> 使用浏览器摄像头实时识别芯片型号，纯 OpenCV 分割 + RapidOCR 推理，无需部署 PaddleOCR。

![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688?logo=fastapi&logoColor=white)![FastAPI](https://img.shields.io/badge/RapidOCR-3.8+-f44336?logo=rapidocr&logoColor=white)![Gradio](https://img.shields.io/badge/Gradio-6.13+-FF6B00?logo=gradio&logoColor=white)![OpenVINO](https://img.shields.io/badge/OpenVINO-2026+-0068B5?logo=intel&logoColor=white)![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ 项目亮点

| 特性 | 说明 |
|---|---|
| **零模型分割** | 纯 OpenCV 形态学流水线检测芯片，无需 YOLO 等检测模型 |
| **双服务架构** | OCR 推理与 UI 解耦，独立生命周期管理 |
| **双重缓存** | 前端帧差分跳过稳定画面 + 后端芯片指纹复用 OCR 结果 |
| **激光标记增强** | 7 种预处理变体，专治激光蚀刻文字识别难题 |
| **规则引擎** | CSV 可编辑的型号规则库，正则匹配 + 优先级排序 |
| **一键打包** | PyInstaller 打包为 Windows 桌面应用（Edge App 窗口） |

---

## 📐 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  浏览器 (Gradio UI  :7860)                                  │
│  ┌──────────┐    帧差分     ┌──────────────┐                │
│  │ 摄像头    │──────────────▶│ 场景稳定性检测 │──── 稳定? ──▶ 跳过请求│
│  └──────────┘               └──────┬───────┘                │
│                                    │ 不稳定                  │
│                                    ▼                        │
│                            JPEG Data URL                    │
└────────────────────────────────┬────────────────────────────┘
                                 │ HTTP POST
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│  OCR 后端 (FastAPI  :8000)                                  │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐    │
│  │ 芯片分割  │──▶│ 透视校正裁剪  │──▶│ OCR 缓存 (IoU+指纹) │    │
│  │ (OpenCV)  │   │ warpPerspective│   └────────┬─────────┘  │
│  └──────────┘   └──────────────┘            │ 未命中       │
│                                              ▼              │
│                                     ┌──────────────┐        │
│                                     │  RapidOCR    │        │
│                                     │ (OpenVINO)   │        │
│                                     └──────┬───────┘        │
│                                            ▼                │
│  ┌──────────────┐   ┌───────────────────────────────────┐   │
│  │ 激光标记增强  │──▶│ 7 种预处理变体 → 选择最佳识别结果  │   │
│  └──────────────┘   └───────────────┬───────────────────┘   │
│                                     ▼                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ chip_rules.csv → SQLite 规则匹配 (正则 + 优先级)     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python ≥ 3.11（推荐 3.14）
- 摄像头设备（推荐 2560×1440 @ 30fps）
- Windows / macOS / Linux

### 安装依赖

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -e .
```

### 启动系统

```bash
# 启动 Gradio 网页（自动拉起 OCR 后端）
uv run display.py
```

启动后访问：

| 服务 | 地址 | 说明 |
|---|---|---|
| **Gradio 页面** | `http://127.0.0.1:7860` | 摄像头预览 + 识别结果 |
| **OCR 后端** | `http://127.0.0.1:8000` | FastAPI 推理服务 |
| **健康检查** | `http://127.0.0.1:8000/health` | 确认引擎加载状态 |

> 💡 **提示**：页面启动后会自动启动或复用 8000 端口 OCR 服务。暂停实时识别不会关闭服务，只有点击页面中的「停止 OCR 服务」按钮才会关闭。

---

## 📁 项目结构

```
pure-onnx/
├── display.py              # 🖥️  Gradio 网页入口（:7860）
├── ocr_server.py           # ⚡ FastAPI OCR 后端（:8000）
├── ocr_client.py           # 📡 OCR 请求客户端 + 进程管理
├── segment.py              # ✂️  芯片分割模块（纯 OpenCV）
├── chip_preprocess.py      # 🔧 激光标记 OCR 预处理（7 种变体）
├── chip_db.py              # 🗃️  型号规则数据库（CSV → SQLite）
├── display.html            # 🎨 前端页面结构
├── display.css             # 🎨 前端样式
├── display.js              # 🎨 前端逻辑（摄像头、帧差分、渲染）
├── chip_rules.csv          # 📋 芯片型号规则表（可编辑）
├── desktop_app.py          # 🪟 Windows 桌面壳（Edge App 模式）
├── ocr_service_entry.py    # 📦 PyInstaller OCR 服务入口
├── runtime_paths.py        # 🛤️  路径辅助（源码 / 打包模式）
├── my_log.py               # 📝 文件日志
├── pyproject.toml          # 📦 项目元数据与依赖
├── DETAILS.md              # 📖 详细技术文档（800+ 行）
├── WinApp-Instructions.md  # 📖 Windows 打包指南
├── packaging/              # 📦 PyInstaller spec 文件
├── images/                 # 🖼️  测试芯片图片
└── frames/                 # 📸 摄像头截图存档
```

---

## 🧩 核心模块

### 1. 芯片分割 (`segment.py`)

纯 OpenCV 流水线，零模型依赖：

1. **多掩码候选生成** — 暗色阈值 + 多种形态学核尺寸
2. **边缘闭合** — 补全芯片轮廓断裂
3. **旋转边界框** — `cv2.minAreaRect` 拟合 + 透视校正
4. **几何过滤** — 面积 / 宽高比 / 中心对比度 / 边缘密度评分
5. **阴影剔除** — 边缘触碰大平滑区域判定
6. **合并分裂 + NMS** — 处理紧密排列芯片

```bash
# 分割芯片并输出调试图
python segment.py images/fourth.jpg chip_crop.jpg

# 不需要调试图
python segment.py images/fourth.jpg chip_crop.jpg --no-debug
```

### 2. OCR 后端 (`ocr_server.py`)

| 配置项 | 值 |
|---|---|
| 推理引擎 | OpenVINO |
| 模型版本 | PPOCRv6 Small |
| 检测输入限制 | `LIMIT_SIDE_LEN = 480` |
| 性能提示 | LATENCY 模式，1 流，4 线程 |
| 超线程 | 关闭 |

### 3. 型号规则库 (`chip_rules.csv`)

```csv
part_number,pattern,description,priority,enabled
SN74LS00N,SN\s*74LS00,四2输入与非门,100,1
STC89C52RC,STC\s*89C52,STC89C52RC 单片机,90,1
```

- **自动同步**：启动时从 CSV 同步到 SQLite
- **正则匹配**：OCR 文本归一化（大写字母数字）后匹配
- **优先级排序**：数值越大优先级越高

### 4. 激光标记增强 (`chip_preprocess.py`)

针对激光蚀刻文字生成 7 种预处理变体：

| 变体 | 用途 |
|---|---|
| `original` | 原图 |
| `body_crop` | 芯片本体裁剪 |
| `upscaled` | 超分辨率放大 |
| `clahe_sharpen` | CLAHE + 锐化 |
| `laser_bright` | 激光明亮背景增强 |
| `laser_dark` | 激光暗色标记增强（优先） |
| `blackhat` | 黑帽变换提取亮文字 |

> 优先使用 `laser_dark` 变体；若无识别结果，依次尝试所有变体并选择最佳。

---

## ⚙️ 配置参数

### 实时识别 (`display.py`)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DEFAULT_CAPTURE_INTERVAL_SECONDS` | `1.0` | 抽帧间隔（秒） |
| `REALTIME_MAX_CHIPS` | `3` | 单帧最大芯片数 |
| `OCR_MAX_IMAGE_SIDE` | `512` | 普通芯片 OCR 输入尺寸 |
| `OCR_LARGE_CHIP_MAX_IMAGE_SIDE` | `768` | 大芯片 OCR 输入尺寸 |
| `OCR_CACHE_IOU_THRESHOLD` | `0.72` | 芯片跟踪缓存 IoU 阈值 |
| `JPEG_QUALITY` | `86` | JPEG 编码质量 |

### 摄像头设置

推荐：**2560×1440 @ 30fps**（`display.js` 中配置）

---

## 🖥️ Windows 桌面打包

详见 [`WinApp-Instructions.md`](WinApp-Instructions.md)：

```bash
# 安装打包依赖
uv add --dev pyinstaller

# 打包
pyinstaller packaging/chip_ocr_desktop.spec
```

产出：
- `ChipOCR.exe` — GUI 壳（无控制台）
- `ocr_server.exe` — OCR 后端服务

---

## 🔍 健康检查与调试

```bash
# 检查 OCR 服务状态
curl http://127.0.0.1:8000/health

# 查看端口占用
powershell -File Check-Port.ps1

# 查看进程树
powershell -File PID-Check.ps1
```

---

## 📖 更多文档

- [`DETAILS.md`](DETAILS.md) — 完整技术文档（流水线 10 步详解、按钮行为、调试指南、拍摄要求）
- [`WinApp-Instructions.md`](WinApp-Instructions.md) — Windows 桌面打包指南

---

## 🤝 致谢

- [RapidOCR](https://github.com/RapidAI/RapidOCR) — OCR 引擎封装
- [OpenVINO](https://github.com/openvinotoolkit/openvino) — Intel 推理加速
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — OCR 模型
- [Gradio](https://github.com/gradio-app/gradio) — Web UI 框架
