"""
dr_annotation.py — Disease Ranger: página de anotação interativa (cliques RGB).

Responsabilidades:
- Renderizar canvas com zoom/pan e marcadores coloridos
- Registrar/desfazer cliques por classe
- Gerenciar session_state relacionado à anotação

Requer: dr_utils.py
"""
import io
import streamlit as st
import numpy as np
from PIL import Image as _PILImage
from streamlit_image_coordinates import streamlit_image_coordinates

from dr_utils import (
    fit_to_square_display,
    draw_markers_on_display,
    display_to_original_coords,
    draw_marker_on_canvas,
    class_color_bgr,
)


class _PreEncodedImage:
    """Wraps pre-encoded JPEG bytes to skip re-encoding inside streamlit_image_coordinates."""
    def __init__(self, data: bytes):
        self._data = data
    def save(self, buf, **kwargs):
        buf.write(self._data)


# =====================================================
# SESSION STATE HELPERS
# =====================================================
def ensure_points_struct(image_key, classes):
    if "points" not in st.session_state:
        st.session_state.points = {}
    if image_key not in st.session_state.points:
        st.session_state.points[image_key] = {c: [] for c in classes}
    else:
        existing = st.session_state.points[image_key]
        for c in classes:
            existing.setdefault(c, [])
        for c in list(existing.keys()):
            if c not in classes:
                existing.pop(c, None)
    if "click_log" not in st.session_state:
        st.session_state.click_log = {}
    st.session_state.click_log.setdefault(image_key, [])


def samples_ready_total(points_per_class, min_total=10):
    total = sum(len(pts) for pts in points_per_class.values())
    return total >= min_total, total


def enter_annotation(mode_key, image_key, img_rgb, classes_for_image):
    st.session_state.annotate_mode = mode_key
    st.session_state.annotate_image_key = image_key
    st.session_state.images_cache[image_key] = np.ascontiguousarray(img_rgb.copy().astype(np.uint8))
    st.session_state.classes_map[image_key] = classes_for_image
    ensure_points_struct(image_key, classes_for_image)
    # Clear stale component state to prevent spurious point on re-entry
    for _stale_key in (f"coords_{image_key}", f"last_added_orig_{image_key}",
                       "_anno_canvas_base", "_anno_canvas_marked",
                       "_anno_canvas_jpeg", "_anno_canvas_jpeg_params"):
        st.session_state.pop(_stale_key, None)


def exit_annotation():
    st.session_state.annotate_mode = None
    st.session_state.annotate_image_key = None


def make_skip_callback(image_key):
    def _cb():
        st.session_state[f"skip_click_{image_key}"] = True
    return _cb


# =====================================================
# PÁGINA DE ANOTAÇÃO
# =====================================================
@st.fragment
def annotation_page(active_class_key):
    """Renderiza a UI completa de anotação de pontos."""
    image_key = st.session_state.annotate_image_key
    if not image_key or image_key not in st.session_state.images_cache:
        st.error("Annotation image not found.")
        if st.button("Back"):
            exit_annotation()
            st.rerun(scope="app")
        st.stop()

    img_rgb = st.session_state.images_cache[image_key]
    classes_for_image = st.session_state.classes_map.get(image_key, [])
    if not classes_for_image:
        st.error("Could not retrieve classes for annotation of this image.")
        if st.button("Back"):
            exit_annotation()
            st.rerun(scope="app")
        st.stop()

    ensure_points_struct(image_key, classes_for_image)

    skip_key = f"skip_click_{image_key}"
    if skip_key not in st.session_state:
        st.session_state[skip_key] = False
    last_added_key = f"last_added_orig_{image_key}"
    if last_added_key not in st.session_state:
        st.session_state[last_added_key] = None

    st.subheader("🖱️ RGB Point Annotation")

    if active_class_key not in st.session_state:
        st.session_state[active_class_key] = classes_for_image[0]
    _active_change_cb = make_skip_callback(image_key)
    active_class = st.selectbox(
        "Active class to mark",
        classes_for_image,
        index=classes_for_image.index(st.session_state[active_class_key]),
        key=active_class_key,
        on_change=_active_change_cb
    )

    canvas_key = f"canvas_{image_key}"
    zoom_key   = f"zoom_{image_key}"
    step_key   = f"step_{image_key}"
    _cb = make_skip_callback(image_key)

    h, w, _ = img_rgb.shape
    pan_x_ctrl_key = f"pan_x_ctrl_{image_key}"
    pan_y_ctrl_key = f"pan_y_ctrl_{image_key}"
    if pan_x_ctrl_key not in st.session_state:
        st.session_state[pan_x_ctrl_key] = 0
    if pan_y_ctrl_key not in st.session_state:
        st.session_state[pan_y_ctrl_key] = 0
    if step_key not in st.session_state:
        st.session_state[step_key] = 50

    canvas_size = st.slider("Canvas size (px)", 400, 1200, 800, 50, key=canvas_key, on_change=_cb)
    zoom = st.slider("Zoom", 1.0, 8.0, 1.0, 0.1, key=zoom_key, on_change=_cb)

    pan_cols = st.columns([1, 1, 2])
    with pan_cols[0]:
        pan_x_val = st.number_input(
            "Pan X (px)", min_value=0, max_value=max(0, w-1),
            value=int(st.session_state[pan_x_ctrl_key]), step=50, on_change=_cb
        )
    with pan_cols[1]:
        pan_y_val = st.number_input(
            "Pan Y (px)", min_value=0, max_value=max(0, h-1),
            value=int(st.session_state[pan_y_ctrl_key]), step=50, on_change=_cb
        )
    with pan_cols[2]:
        st.slider("Arrow step (px)", 10, 200, st.session_state[step_key], 10, key=step_key, on_change=_cb)

    if pan_x_val != st.session_state[pan_x_ctrl_key]:
        st.session_state[pan_x_ctrl_key] = int(pan_x_val)
        st.session_state[skip_key] = True
    if pan_y_val != st.session_state[pan_y_ctrl_key]:
        st.session_state[pan_y_ctrl_key] = int(pan_y_val)
        st.session_state[skip_key] = True

    arrow_cols = st.columns(4)
    step_val = int(st.session_state[step_key])
    if arrow_cols[0].button("←", key=f"left_{image_key}"):
        st.session_state[pan_x_ctrl_key] = max(0, int(st.session_state[pan_x_ctrl_key]) - step_val)
        st.session_state[skip_key] = True
    if arrow_cols[1].button("→", key=f"right_{image_key}"):
        st.session_state[pan_x_ctrl_key] = min(max(0, w-1), int(st.session_state[pan_x_ctrl_key]) + step_val)
        st.session_state[skip_key] = True
    if arrow_cols[2].button("↑", key=f"up_{image_key}"):
        st.session_state[pan_y_ctrl_key] = max(0, int(st.session_state[pan_y_ctrl_key]) - step_val)
        st.session_state[skip_key] = True
    if arrow_cols[3].button("↓", key=f"down_{image_key}"):
        st.session_state[pan_y_ctrl_key] = min(max(0, h-1), int(st.session_state[pan_y_ctrl_key]) + step_val)
        st.session_state[skip_key] = True

    pan_x_cur = int(st.session_state[pan_x_ctrl_key])
    pan_y_cur = int(st.session_state[pan_y_ctrl_key])

    # Cache canvas base (cv2.resize) — só recomputa quando view params mudam
    _base_params = (image_key, int(st.session_state[zoom_key] * 1000),
                    pan_x_cur, pan_y_cur, int(st.session_state[canvas_key]))
    _base_cached = st.session_state.get("_anno_canvas_base")
    if _base_cached is None or _base_cached[0] != _base_params:
        canvas_img, mapping = fit_to_square_display(
            img_rgb, st.session_state[zoom_key], pan_x_cur, pan_y_cur,
            canvas_size=st.session_state[canvas_key]
        )
        st.session_state["_anno_canvas_base"] = (_base_params, canvas_img, mapping)
    else:
        canvas_img, mapping = _base_cached[1], _base_cached[2]

    # Cache canvas marcado — só redesenha quando pontos mudam
    _pts_sig = tuple((c, len(pts)) for c, pts in st.session_state.points[image_key].items())
    _marked_params = (_base_params, _pts_sig)
    _marked_cached = st.session_state.get("_anno_canvas_marked")
    if _marked_cached is None or _marked_cached[0] != _marked_params:
        canvas_marked = draw_markers_on_display(canvas_img, st.session_state.points[image_key], mapping)
        st.session_state["_anno_canvas_marked"] = (_marked_params, canvas_marked)
    else:
        canvas_marked = _marked_cached[1]

    # Cache JPEG encoding — skip re-encoding on reruns where canvas hasn't changed
    # PNG compress_level=0 (default) = ~2.6 MB/click; JPEG quality=75 = ~80 KB/click (30× less)
    _jpeg_params_key = "_anno_canvas_jpeg_params"
    if st.session_state.get(_jpeg_params_key) != _marked_params:
        _buf = io.BytesIO()
        _PILImage.fromarray(canvas_marked).save(_buf, format="JPEG", quality=75)
        st.session_state["_anno_canvas_jpeg"] = _buf.getvalue()
        st.session_state[_jpeg_params_key] = _marked_params
    _canvas_src = _PreEncodedImage(st.session_state["_anno_canvas_jpeg"])

    coords = streamlit_image_coordinates(
        _canvas_src,
        key=f"coords_{image_key}",
        width=st.session_state[canvas_key],
        image_format="JPEG",
        jpeg_quality=75,
        cursor="crosshair",
    )

    undo_cols = st.columns(3)
    if undo_cols[0].button("Undo (active class)", key=f"undo_cls_{image_key}"):
        pts = st.session_state.points[image_key][active_class]
        if pts:
            removed = pts.pop()
            log = st.session_state.click_log[image_key]
            for i in range(len(log) - 1, -1, -1):
                if log[i] == (active_class, removed):
                    log.pop(i)
                    break
            st.success(f"Removed last point from class {active_class}: {removed}")
        else:
            st.info("No points to remove for the active class.")

    if undo_cols[1].button("Undo (last)", key=f"undo_last_{image_key}"):
        log = st.session_state.click_log[image_key]
        if log:
            cls_log, pt_log = log.pop()
            pts = st.session_state.points[image_key][cls_log]
            if pt_log in pts:
                pts.remove(pt_log)
            st.success(f"Removed last point [{cls_log}] {pt_log}")
        else:
            st.info("No points to undo.")

    if undo_cols[2].button("Clear all points", key=f"clear_{image_key}"):
        st.session_state.points[image_key] = {c: [] for c in classes_for_image}
        st.session_state.click_log[image_key] = []
        # Clear stale state so no spurious point is added after reset
        for _ck in (last_added_key, f"coords_{image_key}",
                    "_anno_canvas_marked", "_anno_canvas_jpeg", "_anno_canvas_jpeg_params"):
            st.session_state.pop(_ck, None)
        st.rerun(scope="fragment")

    if st.session_state[skip_key]:
        st.session_state[skip_key] = False
    else:
        if coords is not None and "x" in coords and "y" in coords:
            xd = float(coords["x"])
            yd = float(coords["y"])
            orig = display_to_original_coords(xd, yd, mapping, img_rgb.shape)
            if orig is not None:
                x, y = orig
                if st.session_state[last_added_key] != (x, y):
                    st.session_state.points[image_key][active_class].append((x, y))
                    st.session_state.click_log[image_key].append((active_class, (x, y)))
                    st.session_state[last_added_key] = (x, y)
                    st.toast(f"Ponto adicionado em **{active_class}**: ({x}, {y})", icon="✅")

    _, total_pts = samples_ready_total(st.session_state.points[image_key])
    st.write(f"Total clicks: {total_pts}")
    st.write("### Collected points (per class)")
    for c in classes_for_image:
        st.write(f"- {c}: {len(st.session_state.points[image_key][c])}")

    cols_footer = st.columns(2)
    if cols_footer[0].button("Save"):
        exit_annotation()
        st.rerun(scope="app")
    if cols_footer[1].button("Cancel"):
        exit_annotation()
        st.rerun(scope="app")
