# Windows 桌面应用打包与使用说明

本文档说明如何把本项目打包为 Windows 桌面应用，以及交付、运行和排查问题时需要注意的事项。

## 1. 应用结构

当前 Windows 桌面版采用“本地服务 + Edge App 窗口”的方式：

- `ChipOCR.exe`：桌面入口。启动 Gradio 页面服务，并用 Microsoft Edge 的 App 模式打开窗口。
- `ocr_server.exe`：OCR 后端 sidecar。由页面在需要 OCR 时自动启动。
- `127.0.0.1:7860`：页面服务端口。
- `127.0.0.1:8000`：OCR FastAPI 服务端口。
- `_internal/`：PyInstaller 收集的 Python 运行时、依赖库、模型资源和静态资源。

注意：桌面版不再使用 `pywebview`，因为当前 Python 3.14 环境下 `pythonnet` 不兼容，会导致 `Failed to initialize Python.Runtime.dll`。

## 2. 环境准备

打包机器需要：

- Windows 10/11。
- Microsoft Edge 已安装，并可通过系统路径或常见安装路径找到 `msedge.exe`。
- 已安装 `uv`。
- 项目依赖已同步。

推荐先执行：

```powershell
uv sync --dev
```

如果依赖环境异常，先删除 `.venv` 后重新同步：

```powershell
Remove-Item -Recurse -Force .venv
uv sync --dev
```

## 3. 打包命令

在项目根目录执行：

```powershell
uv run pyinstaller packaging/chip_ocr_desktop.spec --noconfirm
```

打包成功后产物位于：

```text
dist\ChipOCR\
```

主要文件：

```text
dist\ChipOCR\ChipOCR.exe
dist\ChipOCR\ocr_server.exe
dist\ChipOCR\_internal\
```

不要只复制单个 `ChipOCR.exe`。交付时必须复制整个 `dist\ChipOCR` 文件夹。

## 4. 打包前自检

建议打包前先执行：

```powershell
uv run python -m compileall display.py ocr_client.py ocr_server.py desktop_app.py ocr_service_entry.py runtime_paths.py chip_db.py
```

如果刚运行过旧版本桌面应用，先确认没有残留进程：

```powershell
Get-Process | Where-Object { $_.ProcessName -in @("ChipOCR", "ocr_server") }
```

如果存在旧进程，先关闭应用窗口；必要时可结束残留进程：

```powershell
Get-Process | Where-Object { $_.ProcessName -in @("ChipOCR", "ocr_server") } | Stop-Process -Force
```

如果重新打包时报 `PermissionError: 拒绝访问 dist\ChipOCR...`，通常是旧 `ChipOCR.exe`、`ocr_server.exe` 或 Edge App 窗口仍在占用打包目录。关闭后重试即可。

## 5. 运行方法

双击运行：

```text
dist\ChipOCR\ChipOCR.exe
```

或在 PowerShell 中运行：

```powershell
.\dist\ChipOCR\ChipOCR.exe
```

启动后会发生以下流程：

1. `ChipOCR.exe` 启动本地 Gradio 页面服务。
2. 服务监听 `http://127.0.0.1:7860`。
3. 程序用 Microsoft Edge App 模式打开桌面窗口。
4. 点击“开启实时识别”后，页面会自动启动或复用 `ocr_server.exe`。
5. OCR 服务监听 `http://127.0.0.1:8000`。

关闭 Edge App 窗口后，`ChipOCR.exe` 会尝试关闭 OCR 服务并释放端口。

## 6. 使用说明

窗口打开后：

1. 点击“启动摄像头”。
2. 在 Edge 权限提示中允许摄像头访问。
3. 点击“开启实时识别”。
4. 将芯片放入画面，观察右侧识别框和下方识别结果。
5. 如需释放 OCR 资源，点击“停止 OCR 服务”。

摄像头权限是 Edge 针对本地地址保存的权限。若误拒绝，可在 Edge 的站点权限中重置 `127.0.0.1:7860` 的摄像头权限。

## 7. 交付所需文件

交付桌面版时，至少包含整个目录：

```text
dist\ChipOCR\
```

不要遗漏：

- `ChipOCR.exe`
- `ocr_server.exe`
- `_internal\`
- `_internal\display.html`
- `_internal\display.css`
- `_internal\display.js`
- `_internal\chip_rules.csv`
- `_internal\rapidocr\`
- `_internal\gradio\`
- `_internal\openvino\`

首次运行时，程序会把内置 `chip_rules.csv` 复制到 `dist\ChipOCR\chip_rules.csv`，作为可编辑规则文件。之后优先读取 exe 同目录的 `chip_rules.csv`。

## 8. 修改型号规则

交付后的规则文件位置：

```text
dist\ChipOCR\chip_rules.csv
```

字段格式：

```csv
part_number,pattern,description,priority,enabled
SN74LS00N,74[0-9A-Z]{2}[0O]{2},Quad 2-input NAND gate,10,1
```

修改后重启桌面应用，或等待下一次规则库连接时同步。不要把 `chip_rules.sqlite3` 当作长期配置源；它只是运行时缓存。

## 9. 常见问题

### 端口 7860 被占用

现象：

```text
端口 7860 已被占用，无法启动桌面应用
```

处理：

```powershell
netstat -ano | findstr :7860
```

确认占用进程后关闭它。若是旧的 `ChipOCR.exe`，直接结束旧进程即可。

### 端口 8000 被占用

OCR 服务启动失败时检查：

```powershell
netstat -ano | findstr :8000
```

如果是旧的 `ocr_server.exe` 或 Python OCR 服务残留，关闭后重试。不要随意结束不认识的进程。

### Failed to initialize Python.Runtime.dll

这是旧 pywebview 方案在 Python 3.14 下的兼容问题。当前实现已改为 Edge App 模式，不应再出现该错误。

如果仍出现，说明运行的不是最新打包产物。请重新打包并确认 `dist\ChipOCR\_internal` 中不再包含：

```text
pythonnet\runtime\Python.Runtime.dll
```

### 启动失败但弹窗信息不够详细

查看日志：

```text
dist\ChipOCR\desktop_app.log
```

这里会记录 Gradio 启动、Edge App 窗口启动和异常 traceback。

### 缺少 version.txt、link.svg 或 blocks_events.py

这类错误表示 PyInstaller 没有收齐依赖包的运行时资源。当前 spec 已显式收集：

- `rapidocr`
- `gradio`
- `safehttpx`
- `groovy`
- `openvino` 动态库

如升级依赖后再次出现类似错误，需要在 `packaging/chip_ocr_desktop.spec` 中补充对应包的 `collect_data_files(...)` 或 hidden import。

### OCR sidecar 无法读取 ONNX 模型

如果日志出现 OpenVINO 无法读取 `.onnx`，通常是缺少 OpenVINO ONNX frontend。当前 spec 已包含：

```python
collect_dynamic_libs("openvino")
"openvino.frontend.onnx"
```

如升级 OpenVINO 后复现，优先检查 `_internal\openvino\libs\openvino_onnx_frontend.dll` 是否存在。

## 10. 验证命令

验证打包后的 OCR sidecar：

```powershell
.\dist\ChipOCR\ocr_server.exe --help
```

启动桌面应用后，检查页面服务：

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:7860" -UseBasicParsing
```

开启 OCR 后，检查 OCR 服务：

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing
```

正常返回中应包含：

```json
{"status":"ok","engine_loaded":true}
```

## 11. 开发模式运行

不打包时可直接运行桌面入口：

```powershell
uv run python desktop_app.py
```

也可以继续运行原 Web 入口：

```powershell
uv run python display.py
```

原 Web 入口地址仍为：

```text
http://127.0.0.1:7860
```

## 12. 维护注意事项

- 打包方式是 one-folder，不是 one-file。
- `dist\ChipOCR` 是完整应用目录，内部文件不要随意移动。
- `desktop_app.py` 使用系统 Edge 打开 App 窗口，目标机器必须安装 Edge。
- `ocr_server.exe` 是控制台程序，便于单独调试；由 `ChipOCR.exe` 启动时会隐藏窗口。
- `ChipOCR.exe` 是无控制台窗口程序，启动错误会弹窗并写入 `desktop_app.log`。
- 重新打包前建议关闭所有旧应用窗口，避免文件占用。
- 如果升级 Gradio、RapidOCR、OpenVINO、Uvicorn 或 Python 版本，必须重新做一次完整打包验证。
