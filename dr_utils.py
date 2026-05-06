"""
dr_utils.py — Disease Ranger: funções utilitárias (I/O, imagem, geometria, área).
Sem dependências de Streamlit nem estado de sessão.
"""
import cv2
import hashlib
import io
import numpy as np
import os
from pathlib import Path
from PIL import Image


# =====================================================
# ESCALA / ÁREA
# =====================================================
def load_scale_from_tif(tif_path):
    with Image.open(tif_path) as img:
        dpi = img.info.get("dpi", (96, 96))
        return 2.54 / dpi[0]


def calculate_area(mask, scale):
    bin_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0
    for c in contours:
        a = cv2.contourArea(c)
        if scale:
            total_area += a * (scale ** 2)
    return total_area


# =====================================================
# CORES
# =====================================================
def class_color_bgr(cls):
    color_map = {
        "plant": "green", "raiz": "brown", "soil": "brown",
        "background": "white", "healthy": "green", "lesion": "black",
        "clorosis": "yellow", "haste": "green", "cyst": "red", "dirt": "brown",
        "wbackground": "white", "bbackground": "blue", "referencesquare": "orange",
    }
    table = {
        "white": (255, 255, 255), "green": (0, 255, 0), "brown": (42, 42, 165),
        "black": (0, 0, 0), "yellow": (0, 255, 255), "red": (0, 0, 255),
        "blue": (255, 0, 0), "orange": (0, 165, 255),
    }
    return table.get(color_map.get(cls, "white"), (255, 255, 255))


# =====================================================
# CANVAS / DISPLAY
# =====================================================
def fit_to_square_display(img_rgb, zoom, x0, y0, canvas_size=800):
    h, w, _ = img_rgb.shape
    s_fit = min(canvas_size / w, canvas_size / h)
    content_w = max(1, int(round(w * s_fit)))
    content_h = max(1, int(round(h * s_fit)))

    crop_w = max(1, int(round(w / zoom)))
    crop_h = max(1, int(round(h / zoom)))

    x0 = max(0, min(w - crop_w, int(x0)))
    y0 = max(0, min(h - crop_h, int(y0)))

    crop = img_rgb[y0:y0 + crop_h, x0:x0 + crop_w]
    content_resized = cv2.resize(crop, (content_w, content_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((canvas_size, canvas_size, 3), 200, dtype=np.uint8)
    offset_x = (canvas_size - content_w) // 2
    offset_y = (canvas_size - content_h) // 2
    canvas[offset_y:offset_y + content_h, offset_x:offset_x + content_w] = content_resized

    mapping = {
        "canvas_size": canvas_size, "offset_x": offset_x, "offset_y": offset_y,
        "s_fit": s_fit, "zoom": float(zoom), "x0": int(x0), "y0": int(y0),
        "crop_w": int(crop_w), "crop_h": int(crop_h),
        "content_w": int(content_w), "content_h": int(content_h),
    }
    return canvas, mapping


def display_to_original_coords(xd, yd, mapping, img_shape):
    h, w, _ = img_shape
    offset_x, offset_y = mapping["offset_x"], mapping["offset_y"]
    content_w, content_h = mapping["content_w"], mapping["content_h"]
    s_fit, zoom = mapping["s_fit"], mapping["zoom"]
    x0, y0 = mapping["x0"], mapping["y0"]

    if not (offset_x <= xd < offset_x + content_w and offset_y <= yd < offset_y + content_h):
        return None

    x = int(round(x0 + (xd - offset_x) / (s_fit * zoom)))
    y = int(round(y0 + (yd - offset_y) / (s_fit * zoom)))
    return (max(0, min(w - 1, x)), max(0, min(h - 1, y)))


def draw_marker_on_canvas(canvas, xd, yd, color_rgb):
    cv2.circle(canvas, (int(xd), int(yd)), radius=6, color=color_rgb, thickness=-1)
    cv2.circle(canvas, (int(xd), int(yd)), radius=10, color=color_rgb, thickness=2)


def draw_markers_on_display(canvas, points_per_class, mapping):
    out = np.ascontiguousarray(canvas.copy())
    offset_x, offset_y = mapping["offset_x"], mapping["offset_y"]
    s_fit, zoom = mapping["s_fit"], mapping["zoom"]
    x0, y0 = mapping["x0"], mapping["y0"]
    crop_w, crop_h = mapping["crop_w"], mapping["crop_h"]

    for cls, pts in points_per_class.items():
        color_bgr = class_color_bgr(cls)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        for (x, y) in pts:
            if x0 <= x < x0 + crop_w and y0 <= y < y0 + crop_h:
                xd = offset_x + int(round((x - x0) * s_fit * zoom))
                yd = offset_y + int(round((y - y0) * s_fit * zoom))
                draw_marker_on_canvas(out, xd, yd, color_rgb)
    return out


# =====================================================
# THUMBNAIL
# =====================================================
def thumbnail_bgr(img_bgr, max_side=256):
    h, w, _ = img_bgr.shape
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# =====================================================
# CROP
# =====================================================
def crop_bounds(h, w, top, bottom, left, right):
    top = max(0, int(top))
    bottom = max(0, int(bottom))
    left = max(0, int(left))
    right = max(0, int(right))
    if top + bottom >= h:
        bottom = max(0, h - top - 1)
    if left + right >= w:
        right = max(0, w - left - 1)
    return top, h - bottom, left, w - right


def apply_crop(img_rgb, top, bottom, left, right):
    h, w, _ = img_rgb.shape
    y0, y1, x0, x1 = crop_bounds(h, w, top, bottom, left, right)
    return np.ascontiguousarray(img_rgb[y0:y1, x0:x1]), (y0, y1, x0, x1)


# =====================================================
# SAVE / I/O
# =====================================================
def safe_save_colorized_bgr(colorized_bgr: np.ndarray, out_path) -> bool:
    try:
        if not isinstance(out_path, Path):
            out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = colorized_bgr
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        ok = cv2.imwrite(str(out_path), img)
        if ok and out_path.exists():
            return True

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(str(out_path))
        return out_path.exists()
    except Exception:
        return False


def bgr_to_png_bytes(colorized_bgr: np.ndarray) -> bytes:
    img = colorized_bgr
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bio = io.BytesIO()
    Image.fromarray(img_rgb).save(bio, format="PNG")
    bio.seek(0)
    return bio.getvalue()


def make_key(prefix: str, path: str) -> str:
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    return f"{prefix}_{h}"
