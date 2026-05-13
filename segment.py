'''
本文件为预处理，从原图片中分割出芯片，不一定能压缩推理时间，但能保证多目标识别率
'''
from pathlib import Path
import cv2
import numpy as np


def to_gray(image: np.ndarray, input_color: str = 'rgb') -> np.ndarray:
    """
    灰度化函数
    - input_color='rgb': 输入按RGB三通道解释
    - input_color='bgr': 输入按BGR三通道解释（OpenCV默认）
    - 已经是单通道时直接返回
    """
    if image is None or image.size == 0:
        raise ValueError('输入图像为空')

    # 已是灰度图
    if image.ndim == 2:
        return image

    # (H, W, 1) -> squeeze 成单通道
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'不支持的图像形状: {image.shape}')

    mode = input_color.lower()
    if mode == 'rgb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if mode == 'bgr':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError("input_color 仅支持 'rgb' 或 'bgr'")

def clahe_enhance(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
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
        raise ValueError('输入灰度图为空')
    if gray.ndim != 2:
        raise ValueError('clahe_enhance 仅支持单通道灰度图')
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    enhanced = clahe.apply(gray)
    return enhanced

def binarize(gray: np.ndarray) -> np.ndarray:
    """
    二值化：假设芯片整体偏黑，背景偏亮
    1) OTSU自动阈值
    2) 反色，让芯片(黑色区域)在二值图中变为白色，便于后续轮廓检测
    """
    if gray is None or gray.size == 0:
        raise ValueError('输入灰度图为空')

    # OTSU 二值化（先得到“亮=白，暗=黑”）
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反色：让黑色芯片区域变为白色目标区域，利于 findContours
    bw_inv = cv2.bitwise_not(bw)
    return bw_inv


def denoise(binary_img: np.ndarray) -> np.ndarray:
    """
    降噪：开运算去小白点，闭运算填小孔洞
    """
    if binary_img is None or binary_img.size == 0:
        raise ValueError('输入二值图为空')

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    return cleaned


def find_largest_chip_rect(cleaned_binary: np.ndarray, min_area_ratio: float = 0.01):
    """
    轮廓检测，找最大的近似矩形区域
    返回：(x, y, w, h)
    """
    h, w = cleaned_binary.shape[:2]
    img_area = float(h * w)

    contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('未检测到任何轮廓')

    best_rect = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_ratio:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # 优先使用四边形；否则退化为外接矩形
        if len(approx) == 4:
            x, y, rw, rh = cv2.boundingRect(approx)
        else:
            x, y, rw, rh = cv2.boundingRect(cnt)

        rect_area = rw * rh
        if rect_area > best_area:
            best_area = rect_area
            best_rect = (x, y, rw, rh)

    if best_rect is None:
        # 如果没有满足面积阈值的，退化为最大轮廓
        cnt = max(contours, key=cv2.contourArea)
        x, y, rw, rh = cv2.boundingRect(cnt)
        best_rect = (x, y, rw, rh)

    return best_rect


def segment(
    image_path: str = 'input.jpg',
    output_path: str = 'chip_crop.jpg',
    debug: bool = True,
    input_color: str = 'rgb',
):
    """
    从图中分割出最大的黑色矩形（芯片）并保存
    input_color: 'rgb' 或 'bgr'
    """
    image_path = str(image_path)
    output_path = str(output_path)

    src = cv2.imread(image_path)
    if src is None:
        raise FileNotFoundError(f'无法读取图片: {image_path}')

    # 注意：若你的数据是RGB排列，请使用 input_color='rgb'
    gray = to_gray(src, input_color=input_color)
    gray = clahe_enhance(gray, clip_limit=2.0, tile_grid_size=(8, 8))

    # 1) 二值化
    bw = binarize(gray)

    # 2) 降噪
    cleaned = denoise(bw)

    # 3) 找最大矩形并裁剪
    x, y, w, h = find_largest_chip_rect(cleaned)

    # 边界保护
    H, W = src.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    chip = src[y:y + h, x:x + w]

    # 保存结果
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(output_path, chip)
    if not ok:
        raise RuntimeError(f'保存失败: {output_path}')

    if debug:
        # 画框调试图
        vis = src.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        stem = Path(output_path).stem
        suffix = Path(output_path).suffix or '.jpg'
        debug_bin = str(Path(output_path).with_name(f'{stem}_binary{suffix}'))
        debug_clean = str(Path(output_path).with_name(f'{stem}_clean{suffix}'))
        debug_box = str(Path(output_path).with_name(f'{stem}_box{suffix}'))

        cv2.imwrite(debug_bin, bw)
        cv2.imwrite(debug_clean, cleaned)
        cv2.imwrite(debug_box, vis)

    return chip, (x, y, w, h)


if __name__ == '__main__':
    # 测试图为RGB时，传入 input_color='rgb'
    segment('sample.jpg', 'chip_crop.jpg', debug=True, input_color='rgb')
