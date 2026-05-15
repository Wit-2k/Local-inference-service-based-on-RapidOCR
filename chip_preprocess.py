"""芯片激光打标 OCR 预处理。"""

from __future__ import annotations

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
