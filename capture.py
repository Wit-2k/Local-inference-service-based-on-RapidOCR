import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

FRAME_DIR = Path("frames")
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 2560
DEFAULT_FRAME_HEIGHT = 1440
DEFAULT_CAPTURE_INTERVAL_SECONDS = 1.0
DEFAULT_FRAME_RETENTION_SECONDS = 10 * 60


@dataclass(frozen=True)
class CameraInfo:
    width: float
    height: float
    fps: float


def ensure_frame_dir(frame_dir: str | Path = FRAME_DIR) -> Path:
    path = Path(frame_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_camera(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    width: int = DEFAULT_FRAME_WIDTH,
    height: int = DEFAULT_FRAME_HEIGHT,
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_camera_info(cap: cv2.VideoCapture) -> CameraInfo:
    return CameraInfo(
        width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        fps=cap.get(cv2.CAP_PROP_FPS),
    )


def read_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("读取画面失败")
    return frame


def iter_frame_files(frame_dir: str | Path = FRAME_DIR) -> Iterable[Path]:
    path = Path(frame_dir)
    if not path.exists():
        return []
    return path.glob("*.jpg")


def cleanup_old_frames(
    frame_dir: str | Path = FRAME_DIR,
    retention_seconds: float = DEFAULT_FRAME_RETENTION_SECONDS,
    now: float | None = None,
) -> int:
    """删除超过保留时间的截图，避免 frames/ 无限增长。"""
    current_time = time.time() if now is None else now
    removed_count = 0
    for frame_path in iter_frame_files(frame_dir):
        try:
            if current_time - frame_path.stat().st_mtime > retention_seconds:
                frame_path.unlink()
                removed_count += 1
        except OSError:
            continue
    return removed_count


def save_frame(frame: np.ndarray, frame_dir: str | Path = FRAME_DIR) -> Path:
    ensure_frame_dir(frame_dir)
    filename = time.strftime("%Y%m%d_%H%M%S.jpg")
    frame_path = Path(frame_dir) / filename
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"保存截图失败: {frame_path}")
    return frame_path


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def capture_frame(
    cap: cv2.VideoCapture,
    frame_dir: str | Path = FRAME_DIR,
    save: bool = True,
) -> tuple[np.ndarray, Path | None]:
    frame = read_frame(cap)
    frame_path = save_frame(frame, frame_dir) if save else None
    return frame, frame_path


def main() -> None:
    os.makedirs(FRAME_DIR, exist_ok=True)

    # 目前一次只能识别一个芯片
    cap = open_camera()

    info = get_camera_info(cap)
    print("当前宽度:", info.width)
    print("当前高度:", info.height)
    print("当前帧率:", info.fps)

    last_save_time = 0.0
    last_cleanup_time = 0.0

    try:
        while True:
            frame = read_frame(cap)
            now = time.time()

            if now - last_save_time >= DEFAULT_CAPTURE_INTERVAL_SECONDS:
                filename = save_frame(frame)
                print(f"保存:{filename}")
                last_save_time = now

            if now - last_cleanup_time >= DEFAULT_FRAME_RETENTION_SECONDS:
                removed_count = cleanup_old_frames(now=now)
                if removed_count:
                    print(f"清理旧截图:{removed_count} 张")
                last_cleanup_time = now

            # 如果不需要预览，可以删掉下面这几行
            cv2.imshow("camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("❌ 被手动打断")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
