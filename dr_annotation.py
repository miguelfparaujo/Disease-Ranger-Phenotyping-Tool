"""
dr_annotation.py — Disease Ranger: página de anotação interativa (cliques RGB).

Responsabilidades:
- Renderizar canvas com zoom/pan e marcadores coloridos
- Registrar/desfazer cliques por classe
- Gerenciar session_state relacionado à anotação

Requer: dr_utils.py
"""
import streamlit as st
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

from dr_utils import (
    fit_to_square_display,
    draw_markers_on_display,
    display_to_original_coords,
)


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
    for _stale_key in (f"coords_{image_key}", f"last_display_click_{image_key}"):
        st.session_state.pop(_stale_key, None)


def exit_annotation():
    st.session_state.annotate_mode = None
    st.session_state.annotate_image_key = None


# =====================================================
# PÁGINA DE ANOTAÇÃO
# =====================================================
def annotation_page(active_class_key):
    """Renderiza a UI completa de anotação de pontos."""
    image_key = st.session_state.annotate_image_key
    if not image_key or image_key not in st.session_state.images_cache:
        st.error("Annotation image not found.")
        if st.button("Back"):
            exit_annotation()
            st.rerun()
        st.stop()

    img_rgb = st.session_state.images_cache[image_key]
    classes_for_image = st.session_state.classes_map.get(image_key, [])
    if not classes_for_image:
        st.error("Could not retrieve classes for annotation of this image.")
        if st.button("Back"):
            exit_annotation()
            st.rerun()
        st.stop()

    ensure_points_struct(image_key, classes_for_image)

    st.subheader("🖱️ RGB Point Annotation")

    if active_class_key not in st.session_state:
        st.session_state[active_class_key] = classes_for_image[0]
    active_class = st.selectbox(
        "Active class to mark",
        classes_for_image,
        index=classes_for_image.index(st.session_state[active_class_key]),
        key=active_class_key
    )

    canvas_size = st.slider("Canvas size (px)", 400, 1200, 800, 50, key=f"canvas_{image_key}")
    zoom = st.slider("Zoom", 1.0, 8.0, 1.0, 0.1, key=f"zoom_{image_key}")

    h, w, _ = img_rgb.shape
    pan_x_key = f"pan_x_{image_key}"
    pan_y_key = f"pan_y_{image_key}"
    if pan_x_key not in st.session_state:
        st.session_state[pan_x_key] = 0
    if pan_y_key not in st.session_state:
        st.session_state[pan_y_key] = 0

    pan_cols = st.columns([1, 1, 2])
    st.session_state[pan_x_key] = pan_cols[0].number_input(
        "Pan X (px)", min_value=0, max_value=max(0, w - 1),
        value=st.session_state[pan_x_key], step=50
    )
    st.session_state[pan_y_key] = pan_cols[1].number_input(
        "Pan Y (px)", min_value=0, max_value=max(0, h - 1),
        value=st.session_state[pan_y_key], step=50
    )
    step = pan_cols[2].slider("Arrow step (px)", 10, 200, 50, 10, key=f"step_{image_key}")

    arrow_cols = st.columns(4)
    if arrow_cols[0].button("←", key=f"left_{image_key}"):
        st.session_state[pan_x_key] = max(0, st.session_state[pan_x_key] - step)
    if arrow_cols[1].button("→", key=f"right_{image_key}"):
        st.session_state[pan_x_key] = min(max(0, w - 1), st.session_state[pan_x_key] + step)
    if arrow_cols[2].button("↑", key=f"up_{image_key}"):
        st.session_state[pan_y_key] = max(0, st.session_state[pan_y_key] - step)
    if arrow_cols[3].button("↓", key=f"down_{image_key}"):
        st.session_state[pan_y_key] = min(max(0, h - 1), st.session_state[pan_y_key] + step)

    canvas_img, mapping = fit_to_square_display(
        img_rgb, zoom, st.session_state[pan_x_key], st.session_state[pan_y_key],
        canvas_size=canvas_size
    )
    canvas_marked = draw_markers_on_display(canvas_img, st.session_state.points[image_key], mapping)

    coords = streamlit_image_coordinates(canvas_marked, key=f"coords_{image_key}", width=canvas_size)

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
        st.session_state.pop(f"last_display_click_{image_key}", None)
        st.rerun()

    if coords is not None and "x" in coords and "y" in coords:
        xd = float(coords["x"])
        yd = float(coords["y"])
        _display_key = f"last_display_click_{image_key}"
        if st.session_state.get(_display_key) != (xd, yd):
            orig = display_to_original_coords(xd, yd, mapping, img_rgb.shape)
            if orig is not None:
                x, y = orig
                st.session_state.points[image_key][active_class].append((x, y))
                st.session_state.click_log[image_key].append((active_class, (x, y)))
                st.session_state[_display_key] = (xd, yd)
                st.success(f"Point added to **{active_class}**: ({x}, {y})")
            else:
                st.session_state[_display_key] = (xd, yd)
                st.info("Click outside the image area. Adjust pan/zoom and try again.")

    _, total_pts = samples_ready_total(st.session_state.points[image_key])
    st.write(f"Total clicks: {total_pts}")
    st.write("### Collected points (per class)")
    for c in classes_for_image:
        st.write(f"- {c}: {len(st.session_state.points[image_key][c])}")

    cols_footer = st.columns(2)
    if cols_footer[0].button("Save"):
        exit_annotation()
        st.rerun()
    if cols_footer[1].button("Cancel"):
        exit_annotation()
        st.rerun()
