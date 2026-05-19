# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

project_root = Path(SPECPATH).parent

datas = (
    collect_data_files("rapidocr")
    + collect_data_files("gradio")
    + collect_data_files("safehttpx")
    + collect_data_files("groovy")
    + [
    (str(project_root / "display.html"), "."),
    (str(project_root / "display.css"), "."),
    (str(project_root / "display.js"), "."),
    (str(project_root / "chip_rules.csv"), "."),
    ]
)

binaries = collect_dynamic_libs("openvino")

hiddenimports = [
    "openvino.frontend.onnx",
    "uvicorn.logging",
    "uvicorn.lifespan.on",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
]

a = Analysis(
    [str(project_root / "desktop_app.py"), str(project_root / "ocr_service_entry.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    module_collection_mode={"gradio": "pyz+py"},
    optimize=0,
)
pyz = PYZ(a.pure)
runtime_scripts = a.scripts[:-2]
desktop_scripts = runtime_scripts + [a.scripts[-2]]
ocr_service_scripts = runtime_scripts + [a.scripts[-1]]

chip_ocr = EXE(
    pyz,
    desktop_scripts,
    [],
    exclude_binaries=True,
    name="ChipOCR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

ocr_server = EXE(
    pyz,
    ocr_service_scripts,
    [],
    exclude_binaries=True,
    name="ocr_server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    chip_ocr,
    ocr_server,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ChipOCR",
)
