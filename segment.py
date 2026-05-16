"""
本文件为预处理，从原图中分割出各个芯片，目的是保证多目标识别率。
使用 OpenCV 库实现的纯视觉方案而非 YOLO，目的是压缩处理时间。
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

Rect = tuple[int, int, int, int]
CandidateMask = tuple[str, np.ndarray]

EDGE_CLOSE_KERNELS: tuple[tuple[int, int], ...] = ((7, 7), (15, 15), (31, 15))
MIN_DARK_RATIO = 0.22
MIN_CENTER_BORDER_CONTRAST = 25.0
MAX_BORDER_TOUCHING_AREA_RATIO = 0.08


@dataclass(frozen=True)
class ChipCandidate:
    rect: Rect
    score: float
    source: str


def to_gray(image: np.ndarray, input_color: str = "rgb") -> np.ndarray:
    """
    灰度化函数
    - input_color='rgb': 输入按RGB三通道解释
    - input_color='bgr': 输入按BGR三通道解释（OpenCV默认）
    - 已经是单通道时直接返回
    """
    if image is None or image.size == 0:
        raise ValueError("输入图像为空")

    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"不支持的图像形状: {image.shape}")

    mode = input_color.lower()
    if mode == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mode == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError("input_color 仅支持 'rgb' 或 'bgr'")


def clahe_enhance(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    CLAHE 对比度受限自适应直方图均衡化（用于灰度图）
    参数:
        gray: 单通道灰度图, uint8
        clip_limit: 对比度限制，常用 2.0~4.0
        tile_grid_size: 分块大小，常用 (8,8)
    返回:
        增强后的灰度图（uint8）
    """
    if gray is None or gray.size == 0:
        raise ValueError("输入灰度图为空")
    if gray.ndim != 2:
        raise ValueError("clahe_enhance 仅支持单通道灰度图")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    二值化：假设芯片整体偏黑，背景偏亮
    1) OTSU自动阈值
    2) 反色，让芯片(黑色区域)在二值图中变为白色，便于后续轮廓检测
    """
    if gray is None or gray.size == 0:
        raise ValueError("输入灰度图为空")

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_inv = cv2.bitwise_not(bw)
    return bw_inv


def denoise(binary_img: np.ndarray) -> np.ndarray:
    """
    降噪：开运算去小白点，闭运算填小孔洞。
    """
    if binary_img is None or binary_img.size == 0:
        raise ValueError("输入二值图为空")

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return cleaned


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def build_candidate_masks(gray: np.ndarray) -> list[CandidateMask]:
    """
    构造多路候选掩码：
    - dark: 暗区域阈值，适合黑色芯片主体清晰的场景
    - edge_*: 边缘闭合，适合背景纹理导致暗区域粘连的场景
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    enhanced = clahe_enhance(blur, clip_limit=2.0, tile_grid_size=(8, 8))

    dark = denoise(binarize(enhanced))
    masks: list[CandidateMask] = [("dark", dark)]

    edges = auto_canny(blur)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for kernel_size in EDGE_CLOSE_KERNELS:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        closed = cv2.dilate(closed, close_kernel, iterations=1)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)
        masks.append((f"edge_{kernel_size[0]}x{kernel_size[1]}", closed))

    return masks


def clip_rect(rect: Rect, image_shape: tuple[int, int]) -> Rect:
    x, y, w, h = rect
    image_h, image_w = image_shape[:2]
    x = max(0, min(int(x), image_w - 1))
    y = max(0, min(int(y), image_h - 1))
    w = max(1, min(int(w), image_w - x))
    h = max(1, min(int(h), image_h - y))
    return x, y, w, h


def pad_rect(
    rect: Rect, image_shape: tuple[int, int], padding_ratio: float = 0.06
) -> Rect:
    x, y, w, h = rect
    pad_x = int(round(w * padding_ratio))
    pad_y = int(round(h * padding_ratio))
    return clip_rect((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), image_shape)


def rect_area(rect: Rect) -> int:
    return rect[2] * rect[3]


def rect_intersection_area(rect_a: Rect, rect_b: Rect) -> int:
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    return max(0, right - left) * max(0, bottom - top)


def rect_iou(rect_a: Rect, rect_b: Rect) -> float:
    intersection = rect_intersection_area(rect_a, rect_b)
    if intersection == 0:
        return 0.0

    union = rect_area(rect_a) + rect_area(rect_b) - intersection
    return intersection / union if union > 0 else 0.0


def rect_containment_ratio(inner: Rect, outer: Rect) -> float:
    inner_area = rect_area(inner)
    if inner_area == 0:
        return 0.0
    return rect_intersection_area(inner, outer) / inner_area


def center_border_contrast(roi: np.ndarray) -> float:
    h, w = roi.shape[:2]
    border_x = max(1, w // 10)
    border_y = max(1, h // 10)
    if h <= border_y * 2 or w <= border_x * 2:
        return 0.0

    border = np.concatenate(
        [
            roi[:border_y, :].ravel(),
            roi[-border_y:, :].ravel(),
            roi[:, :border_x].ravel(),
            roi[:, -border_x:].ravel(),
        ]
    )
    center = roi[border_y:-border_y, border_x:-border_x]
    return float(border.mean() - center.mean())


def score_rect(gray: np.ndarray, rect: Rect) -> tuple[float, float, float, float]:
    x, y, w, h = rect
    roi = gray[y : y + h, x : x + w]
    dark_threshold = float(np.percentile(gray, 35))
    dark_ratio = float((roi < dark_threshold).mean())
    contrast = center_border_contrast(roi)

    edge_map = auto_canny(roi)
    edge_density = float((edge_map > 0).mean())
    contrast_score = min(max(contrast, 0.0) / 80.0, 1.0)
    edge_score = min(edge_density * 8.0, 1.0)
    score = 0.55 * dark_ratio + 0.35 * contrast_score + 0.10 * edge_score
    return score, dark_ratio, contrast, edge_density


def rect_touches_border(
    rect: Rect, image_shape: tuple[int, int], margin_ratio: float = 0.01
) -> bool:
    x, y, w, h = rect
    image_h, image_w = image_shape[:2]
    margin_x = max(1, int(round(image_w * margin_ratio)))
    margin_y = max(1, int(round(image_h * margin_ratio)))
    return (
        x <= margin_x
        or y <= margin_y
        or x + w >= image_w - margin_x
        or y + h >= image_h - margin_y
    )


def contour_rects(mask: np.ndarray) -> Iterable[tuple[Rect, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area <= 0:
            continue
        yield cv2.boundingRect(contour), contour_area


def is_duplicate_candidate(
    candidate: ChipCandidate,
    selected: ChipCandidate,
    *,
    iou_threshold: float,
    containment_threshold: float,
) -> bool:
    if rect_iou(candidate.rect, selected.rect) >= iou_threshold:
        return True
    return (
        rect_containment_ratio(candidate.rect, selected.rect) >= containment_threshold
    )


def non_max_suppression(
    candidates: list[ChipCandidate],
    iou_threshold: float = 0.35,
    containment_threshold: float = 0.75,
) -> list[ChipCandidate]:
    selected: list[ChipCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if not any(
            is_duplicate_candidate(
                candidate,
                item,
                iou_threshold=iou_threshold,
                containment_threshold=containment_threshold,
            )
            for item in selected
        ):
            selected.append(candidate)
    return selected


def has_plausible_geometry(
    rect: Rect,
    contour_area: float,
    *,
    image_area: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_aspect_ratio: float,
    min_side: int,
) -> bool:
    _, _, w, h = rect
    area = rect_area(rect)
    area_ratio = area / image_area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False
    if min(w, h) < min_side:
        return False

    aspect_ratio = max(w / h, h / w)
    if aspect_ratio > max_aspect_ratio:
        return False

    fill_ratio = contour_area / max(float(area), 1.0)
    return fill_ratio >= 0.08


def score_plausible_chip(
    gray: np.ndarray, rect: Rect, min_score: float
) -> float | None:
    score, dark_ratio, contrast, _ = score_rect(gray, rect)
    if dark_ratio < MIN_DARK_RATIO and contrast < MIN_CENTER_BORDER_CONTRAST:
        return None
    if score < min_score:
        return None
    return score


def sort_rects_reading_order(candidates: list[ChipCandidate]) -> list[ChipCandidate]:
    if not candidates:
        return []

    heights = [candidate.rect[3] for candidate in candidates]
    row_tolerance = max(20, int(np.median(heights) * 0.55))
    rows: list[list[ChipCandidate]] = []

    for candidate in sorted(candidates, key=lambda item: item.rect[1]):
        x, y, _, _ = candidate.rect
        for row in rows:
            row_y = int(np.mean([item.rect[1] for item in row]))
            if abs(y - row_y) <= row_tolerance:
                row.append(candidate)
                break
        else:
            rows.append([candidate])

    ordered: list[ChipCandidate] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item.rect[0]))
    return ordered


def find_chip_candidates(
    gray: np.ndarray,
    *,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.18,
    max_aspect_ratio: float = 5.5,
    min_side_ratio: float = 0.025,
    min_score: float = 0.35,
    nms_iou_threshold: float = 0.35,
    padding_ratio: float = 0.06,
    max_chips: int | None = None,
) -> list[ChipCandidate]:
    """
    检测多个芯片候选框。
    过滤逻辑强调“暗色矩形主体 + 中心比边缘更暗 + 合理面积/长宽比”，
    用来减少木纹、阴影、画面边缘暗物体造成的误检。
    """
    image_h, image_w = gray.shape[:2]
    image_area = float(image_h * image_w)
    min_side = max(20, int(round(min(image_h, image_w) * min_side_ratio)))
    candidates: list[ChipCandidate] = []

    for source, mask in build_candidate_masks(gray):
        for rect, contour_area in contour_rects(mask):
            clipped_rect = clip_rect(rect, gray.shape)
            if not has_plausible_geometry(
                clipped_rect,
                contour_area,
                image_area=image_area,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                max_aspect_ratio=max_aspect_ratio,
                min_side=min_side,
            ):
                continue

            score = score_plausible_chip(gray, clipped_rect, min_score)
            if score is None:
                continue

            area_ratio = rect_area(clipped_rect) / image_area
            if (
                rect_touches_border(clipped_rect, gray.shape)
                and area_ratio > MAX_BORDER_TOUCHING_AREA_RATIO
            ):
                continue

            padded = pad_rect(clipped_rect, gray.shape, padding_ratio=padding_ratio)
            candidates.append(ChipCandidate(rect=padded, score=score, source=source))

    selected = non_max_suppression(candidates, iou_threshold=nms_iou_threshold)
    selected = sort_rects_reading_order(selected)
    if max_chips is not None:
        selected = selected[:max_chips]
    return selected


def segment_array(
    image: np.ndarray,
    input_color: str = "rgb",
    max_chips: int | None = None,
) -> list[tuple[np.ndarray, Rect]]:
    """
    从内存图像中分割多个芯片，不保存任何本地文件。
    input_color: 'rgb' 或 'bgr'，用于候选检测的灰度转换。
    返回: [(芯片图像, 矩形), ...]
    """
    if image is None or image.size == 0:
        raise ValueError("输入图像为空")

    gray = to_gray(image, input_color=input_color)
    candidates = find_chip_candidates(gray, max_chips=max_chips)
    if not candidates:
        return []

    chips: list[tuple[np.ndarray, Rect]] = []
    for candidate in candidates:
        x, y, w, h = candidate.rect
        chips.append((image[y : y + h, x : x + w].copy(), candidate.rect))
    return chips


def output_path_for_index(output_path: str | Path, index: int, total: int) -> Path:
    path = Path(output_path)
    if total == 1:
        return path

    suffix = path.suffix or ".jpg"
    return path.with_name(f"{path.stem}_{index:02d}{suffix}")


def save_debug_images(
    src: np.ndarray,
    masks: list[CandidateMask],
    candidates: list[ChipCandidate],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".jpg"
    stem = output.stem

    vis = src.copy()
    for index, candidate in enumerate(candidates, start=1):
        x, y, w, h = candidate.rect
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            vis,
            f"{index}:{candidate.score:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output.with_name(f"{stem}_box{suffix}")), vis)
    for source, mask in masks:
        cv2.imwrite(str(output.with_name(f"{stem}_{source}{suffix}")), mask)


def segment(
    image_path: str = "input.jpg",
    output_path: str = "chip_crop.jpg",
    debug: bool = True,
    input_color: str = "bgr",
    max_chips: int | None = None,
) -> tuple[list[np.ndarray], list[Rect]]:
    """
    从图中分割出多个黑色矩形芯片并保存。
    input_color: 'rgb' 或 'bgr'。cv2.imread 读取的文件通常应使用 'bgr'。
    返回: (芯片图像列表, 矩形列表)
    """
    src = cv2.imread(str(image_path))
    if src is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    gray = to_gray(src, input_color=input_color)
    candidates = find_chip_candidates(gray, max_chips=max_chips)
    if not candidates:
        raise RuntimeError("未检测到芯片候选区域")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    chips: list[np.ndarray] = []
    rects: list[Rect] = []
    for index, candidate in enumerate(candidates, start=1):
        x, y, w, h = candidate.rect
        chip = src[y : y + h, x : x + w]
        crop_path = output_path_for_index(output, index, len(candidates))
        ok = cv2.imwrite(str(crop_path), chip)
        if not ok:
            raise RuntimeError(f"保存失败: {crop_path}")

        chips.append(chip)
        rects.append(candidate.rect)

    if debug:
        save_debug_images(src, build_candidate_masks(gray), candidates, output)

    return chips, rects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从输入图片中分割出一个或多个芯片")
    parser.add_argument(
        "image_path", nargs="?", default=R"images\big.jpg", help="输入图片路径"
    )
    parser.add_argument(
        "output_path", nargs="?", default="chip_crop.jpg", help="输出裁剪图片路径"
    )
    parser.add_argument(
        "--input-color",
        choices=["bgr", "rgb"],
        default="bgr",
        help="输入图像通道顺序；cv2.imread 读取文件时使用 bgr",
    )
    parser.add_argument(
        "--max-chips", type=int, default=None, help="最多输出的芯片数量"
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否输出候选掩码和画框调试图",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    start_time = time.perf_counter()

    chips, rects = segment(
        args.image_path,
        args.output_path,
        debug=args.debug,
        input_color=args.input_color,
        max_chips=args.max_chips,
    )

    end_time = time.perf_counter()

    print(f"检测到 {len(chips)} 个芯片: {rects}")
    print(f"用时 {(end_time - start_time) * 1000} 毫秒")
