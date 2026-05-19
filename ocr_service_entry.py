"""PyInstaller sidecar entrypoint for the OCR FastAPI service."""

from __future__ import annotations

import argparse
import multiprocessing

import uvicorn

from ocr_server import APP_IMPORT_PATH

DEFAULT_OCR_PORT = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动芯片 OCR 后端服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=DEFAULT_OCR_PORT, help="监听端口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(
        APP_IMPORT_PATH,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
