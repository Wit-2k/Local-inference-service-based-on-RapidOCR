"""芯片激光打标 OCR 预处理。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def crop_dark_body(image: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("输入图像为空")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_h, image_w = gray.shape[:2]
    min_area = image_h * image_w * min_area_ratio
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        return image

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    padding = 4
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image_w - x, w + 2 * padding)
    h = min(image_h - y, h + 2 * padding)
    return image[y : y + h, x : x + w]


def upscale(image: np.ndarray, scale: float = 3.0) -> np.ndarray:
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def clahe_sharpen(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.2)
    sharpened = cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def enhance_laser_marking(image: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    background = cv2.GaussianBlur(gray, (0, 0), 15)
    enhanced = cv2.subtract(background, gray)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(enhanced)
    if invert:
        enhanced = cv2.bitwise_not(enhanced)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def blackhat_enhance(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 7))
    enhanced = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(enhanced)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def generate_chip_ocr_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    body = crop_dark_body(image)
    upscaled = upscale(body)
    return [
        ("original", image),
        ("body", body),
        ("upscaled", upscaled),
        ("clahe_sharpen", clahe_sharpen(upscaled)),
        ("laser_bright", enhance_laser_marking(upscaled, invert=False)),
        ("laser_dark", enhance_laser_marking(upscaled, invert=True)),
        ("blackhat", blackhat_enhance(upscaled)),
    ]


def output_dir_for_image(image_path: str | Path) -> Path:
    path = Path(image_path)
    return path.with_name(f"{path.stem}_preprocess_variants")


def save_chip_ocr_variants(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    extension: str = ".jpg",
) -> list[Path]:
    src = cv2.imread(str(image_path))
    if src is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    output = (
        Path(output_dir) if output_dir is not None else output_dir_for_image(image_path)
    )
    output.mkdir(parents=True, exist_ok=True)

    suffix = extension if extension.startswith(".") else f".{extension}"
    saved_paths: list[Path] = []
    for index, (name, variant) in enumerate(generate_chip_ocr_variants(src), start=1):
        path = output / f"{index:02d}_{name}{suffix}"
        ok = cv2.imwrite(str(path), variant)
        if not ok:
            raise RuntimeError(f"保存失败: {path}")
        saved_paths.append(path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出芯片 OCR 预处理的七种调试变体")
    parser.add_argument(
        "image_path", nargs="?", default=R"images\dark.png", help="输入芯片图片路径"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="preprocess_variants/",
        help="输出目录；默认使用 <输入文件名>_preprocess_variants",
    )
    parser.add_argument(
        "--ext",
        default=".jpg",
        help="输出图片扩展名，例如 .jpg 或 .png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_paths = save_chip_ocr_variants(args.image_path, args.output_dir, args.ext)
    print(f"已输出 {len(saved_paths)} 个预处理变体:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
