"""
main.py - Disease Ranger Web Edition
Arquivo principal (Streamlit). Importa modulos dr_*.py para manter este arquivo enxuto.

Modulos:
  dr_utils.py       - I/O, geometria, escala, area, canvas
  dr_classifier.py  - PDF (PlantCV Naive Bayes) + Interativo (Random Forest)
  dr_results.py     - Logica de resultado por doenca, Vagem, Stink Bug
  dr_annotation.py  - Pagina interativa de anotacao de pontos
"""
import streamlit as st
import numpy as np
import os
import cv2
import tempfile
import math
import io
import base64
from PIL import Image
from pathlib import Path

import pandas as pd
try:
    import tkinter as tk
    from tkinter import filedialog
    _TKINTER_AVAILABLE = True
except Exception:
    _TKINTER_AVAILABLE = False

from dr_utils import (
    load_scale_from_tif, calculate_area, make_key,
    safe_save_colorized_bgr, bgr_to_png_bytes,
    apply_crop, crop_bounds, thumbnail_bgr,
)
from dr_classifier import (
    pdf_classes_from_image, bayes, colorize,
    collect_samples_from_points_with_xy, train_classifier,
    classify_image_with_xy, counts_from_mask, percentages_for_disease,
)
from dr_results import (
    split_vagens_components, sort_components_left_to_right,
    compute_vagem_percentages, draw_vagem_boxes, process_image_result,
)
from dr_annotation import (
    ensure_points_struct, samples_ready_total,
    enter_annotation, exit_annotation, annotation_page,
)

st.set_page_config(page_title="Disease Ranger", page_icon="🧬", layout="wide")

_LOGO_PATH      = Path(__file__).parent / "bayer_logo.png"
_LOGO_DARK_PATH = Path(__file__).parent / "logobranca.png"

@st.cache_data(show_spinner=False)
def _logo_base64() -> str | None:
    """Lê a logo como base64 para renderização nativa no browser (preserva transparência PNG)."""
    try:
        with open(_LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _logo_dark_base64() -> str | None:
    """Lê a logo branca (modo escuro) como base64."""
    try:
        with open(_LOGO_DARK_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


with st.sidebar:
    _b64      = _logo_base64()
    _b64_dark = _logo_dark_base64()
    if _b64 is not None:
        if _b64_dark is not None:
            st.markdown(
                f'<a href="/" target="_self" title="Back to home" style="display:block; margin-bottom:6px;">'
                f'<picture>'
                f'<source media="(prefers-color-scheme: dark)" srcset="data:image/png;base64,{_b64_dark}">'
                f'<img src="data:image/png;base64,{_b64}" width="150" style="cursor:pointer; display:block;"/>'
                f'</picture></a>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<a href="/" target="_self" title="Back to home">'
                f'<img src="data:image/png;base64,{_b64}" width="150" '
                f'style="cursor:pointer; display:block; margin-bottom:6px;"/></a>',
                unsafe_allow_html=True
            )
    else:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    st.divider()


class Options:
    def __init__(self):
        self.disease = None

options = Options()


def pick_folder():
    if _TKINTER_AVAILABLE:
        try:
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory()
            root.destroy()
            return folder
        except Exception:
            pass
    return None


st.title("🧬 Disease Ranger - Image Analysis")

options.disease = st.sidebar.selectbox(
    "Select disease",
    ["PRR", "TLS", "FLS", "Vagem", "PGR", "CHR", "SCN", "Stink Bug"]
)

classification_mode = st.sidebar.radio("Classification mode", ["Automatic (PDF)", "Interactive (clicks)"])

mode = st.sidebar.radio("Operation mode", ["Single image", "Folder"])

st.sidebar.markdown("### Crop (px)")
crop_top    = st.sidebar.number_input("Top crop (px)", min_value=0, value=0, step=10, key="crop_top")
crop_bottom = st.sidebar.number_input("Bottom crop (px)", min_value=0, value=0, step=10, key="crop_bottom")
crop_left   = st.sidebar.number_input("Left crop (px)", min_value=0, value=0, step=10, key="crop_left")
crop_right  = st.sidebar.number_input("Right crop (px)", min_value=0, value=0, step=10, key="crop_right")
crop_tuple  = (crop_top, crop_bottom, crop_left, crop_right)
st.sidebar.caption(
    "Powered by [PlantCV](https://plantcv.org/) © Donald Danforth Plant Science Center "
    "([MPL-2.0](https://github.com/danforthcenter/plantcv/blob/main/LICENSE))  \n"
    "Cite: Schuhl et al. 2026, *Plant Phenome J.* "
    "[10.1002/ppj2.70065](https://doi.org/10.1002/ppj2.70065)  \n"
    "Naive Bayes: Abbasi & Fahlgren 2016, *IEEE WNYISPW* "
    "[10.1109/WNYIPW.2016.7904790](https://doi.org/10.1109/WNYIPW.2016.7904790)"
)

use_xy_features  = False
use_pdf_fallback = False
if classification_mode == "Interactive (clicks)":
    use_xy_features = st.sidebar.checkbox(
        "Include (x,y) coordinates in clicks (Interactive)",
        value=True,
        help="Adds normalized x,y as classifier features in addition to R,G,B."
    )
    use_pdf_fallback = st.sidebar.checkbox(
        "Use PDF as fallback if points are insufficient (Interactive)",
        value=True,
        help="If fewer than 2 classes have points, automatically uses PDF in folder analysis."
    )

pdf_files = {
    "PRR": "Nayve/prr2026_pdf.txt",
    "FLS": "Nayve/naive_bayes_pdfsFLS.txt",
    "TLS": "Nayve/ASR2026_pdf.txt",
    "Vagem": "Nayve/naive_bayes_pdfsVagem.txt",
    "CHR": "Nayve/naive_bayes_canopyout25pdf.txt",
    "PGR": "Nayve/pgr_pdf.txt",
    "SCN": "Nayve/nematodes_pdf.txt",
    "Stink Bug": "Nayve/stinkbug_pdf.txt",
}
pdf_file = pdf_files.get(options.disease)

st.divider()

for _key, _default in [
    ("annotate_mode", None),
    ("annotate_image_key", None),
    ("images_cache", {}),
    ("classes_map", {}),
    ("selected_files", []),
    ("last_folder_used", None),
    ("single_scale", None),
    ("last_single_image_key", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

if st.session_state.annotate_mode is not None:
    active_class_key = (
        "active_class_single" if st.session_state.annotate_mode == "single"
        else "active_class_folder"
    )
    annotation_page(active_class_key)
    st.stop()

if mode == "Single image":
    uploaded     = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp"], key="single_uploader")
    uploaded_tif = st.file_uploader("Upload scale TIF (optional)", type=["tif"], key="single_scale_uploader")

    image_key = None
    img_rgb   = None

    if uploaded is not None:
        image_key = f"single::{uploaded.name}"
        st.session_state.last_single_image_key = image_key
        pil_img = Image.open(uploaded).convert("RGB")
        img_rgb = np.ascontiguousarray(np.array(pil_img).astype(np.uint8))
        st.session_state.images_cache[image_key] = img_rgb
    else:
        image_key = st.session_state.last_single_image_key
        if image_key and image_key in st.session_state.images_cache:
            img_rgb = st.session_state.images_cache[image_key]
        else:
            st.info("Upload an image to continue.")
            st.stop()

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    interactive_classes = st.session_state.classes_map.get(image_key)
    if not interactive_classes:
        interactive_classes = []
        if pdf_file and os.path.exists(pdf_file):
            interactive_classes = pdf_classes_from_image(img_bgr, pdf_file)
        if not interactive_classes:
            interactive_classes = ["background", "plant", "lesion"]
        st.session_state.classes_map[image_key] = interactive_classes
    ensure_points_struct(image_key, interactive_classes)

    if classification_mode == "Interactive (clicks)" and st.button("Mark class"):
        enter_annotation("single", image_key, img_rgb, interactive_classes)
        st.rerun()

    scale = st.session_state.single_scale
    if uploaded_tif is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_tif.read())
            st.session_state.single_scale = load_scale_from_tif(tmp.name)
        scale = st.session_state.single_scale

    img_rgb_cropped, _ = apply_crop(img_rgb, crop_top, crop_bottom, crop_left, crop_right)

    if classification_mode == "Automatic (PDF)":
        if pdf_file and os.path.exists(pdf_file):
            percentages, mask, _ = bayes(
                cv2.cvtColor(img_rgb_cropped, cv2.COLOR_RGB2BGR), pdf_file, options.disease
            )
        else:
            st.error("Disease PDF file not found.")
            st.stop()
    else:
        pts_per_class    = st.session_state.points[image_key]
        samples_all      = collect_samples_from_points_with_xy(
            img_rgb, pts_per_class, use_xy=use_xy_features, crop=crop_tuple
        )
        classes_with_pts = [c for c in interactive_classes if len(samples_all.get(c, [])) > 0]
        if len(classes_with_pts) >= 2:
            train_samp = {c: samples_all[c] for c in classes_with_pts}
            model = train_classifier(train_samp)
            model._use_xy = use_xy_features
            model._classes_to_train = classes_with_pts
            mask = classify_image_with_xy(img_rgb_cropped, model, classes_with_pts)
            counts = counts_from_mask(mask)
            percentages, _ = percentages_for_disease(options.disease, counts)
        else:
            if use_pdf_fallback and pdf_file and os.path.exists(pdf_file):
                st.info("Insufficient points (minimum 2 classes). Using PDF as fallback.")
                percentages, mask, _ = bayes(
                    cv2.cvtColor(img_rgb_cropped, cv2.COLOR_RGB2BGR), pdf_file, options.disease
                )
            else:
                st.warning("Insufficient points and PDF fallback disabled or unavailable.")
                st.stop()

    colorized_bgr = np.ascontiguousarray(colorize(mask, options.disease).astype(np.uint8))

    base_name = uploaded.name if uploaded is not None else "single_image.png"
    base_root, base_ext = os.path.splitext(base_name)
    download_filename = f"{base_root}_colorized.png"

    if options.disease == "Vagem":
        components        = split_vagens_components(mask)
        components_sorted = sort_components_left_to_right(components)
        draw_vagem_boxes(colorized_bgr, components_sorted)
        png_bytes = bgr_to_png_bytes(colorized_bgr)

        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        col1.image(Image.fromarray(img_rgb_cropped), caption="Original (cropped)", width='stretch')
        col2.image(colorized_rgb, caption="Classification (numbered pods)", width='stretch')
        st.download_button("Download colorized image (PNG)", data=png_bytes,
                           file_name=download_filename, mime="image/png")

        rows = []
        if not components_sorted:
            rows.append({"pod_image": f"{base_root}_vagem01{base_ext}", "pod_id": 1,
                         "healthy (%)": 0.0, "lesion (%)": 0.0, "bbox": ""})
        else:
            for i, comp in enumerate(components_sorted, start=1):
                _, _, _, hpct, lpct = compute_vagem_percentages(mask, comp['mask_bool'])
                x0, y0, x1, y1 = comp['bbox']
                rows.append({
                    "pod_image": f"{base_root}_vagem{i:02d}{base_ext}",
                    "pod_id": i,
                    "healthy (%)": round(hpct, 2),
                    "lesion (%)": round(lpct, 2),
                    "bbox": f"{x0},{y0},{x1},{y1}"
                })
        st.subheader("Results per pod")
        st.dataframe(pd.DataFrame(rows))

    elif options.disease == "Stink Bug":
        png_bytes = bgr_to_png_bytes(colorized_bgr)
        h2, w2    = img_rgb_cropped.shape[:2]
        total_px  = h2 * w2
        wbg  = int(np.count_nonzero(mask.get('wbackground', 0)))
        bbg  = int(np.count_nonzero(mask.get('bbackground', 0)))
        hl_h = int(np.count_nonzero(mask.get('healthy', 0)))
        hl_l = int(np.count_nonzero(mask.get('lesion', 0)))
        ref  = int(np.count_nonzero(mask.get('referencesquare', 0)))
        total_hl = hl_h + hl_l
        _z   = np.zeros((h2, w2), np.uint8)

        wbg_pct = round(wbg / total_px * 100, 2) if total_px else 0.0
        bbg_pct = round(bbg / total_px * 100, 2) if total_px else 0.0
        h_pct   = round(hl_h / total_hl * 100, 2) if total_hl else 0.0
        l_pct   = round(hl_l / total_hl * 100, 2) if total_hl else 0.0
        ref_pct = round(ref  / total_px * 100, 2) if total_px else 0.0
        _area_keys = {
            "wbackground_area (cm2)": "wbackground",
            "bbackground_area (cm2)": "bbackground",
            "healthy_area (cm2)": "healthy",
            "lesion_area (cm2)": "lesion",
            "referencesquare_area (cm2)": "referencesquare",
        }
        areas = {k: (round(calculate_area(mask.get(_area_keys[k], _z), scale), 2) if scale else "N/A")
                 for k in _area_keys}

        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        col1.image(Image.fromarray(img_rgb_cropped), caption="Original (cropped)", width='stretch')
        col2.image(colorized_rgb, caption="Classification (Stink Bug)", width='stretch')
        st.download_button("Download colorized image (PNG)", data=png_bytes,
                           file_name=download_filename, mime="image/png")
        st.subheader("Results (Stink Bug)")
        st.json({
            "wbackground (%)": wbg_pct, "bbackground (%)": bbg_pct,
            "healthy (%)": h_pct, "lesion (%)": l_pct,
            "referencesquare (%)": ref_pct,
            "healthy+lesion (%)": round(h_pct + l_pct, 2),
            **{k: v for k, v in areas.items()}
        })

    else:
        png_bytes = bgr_to_png_bytes(colorized_bgr)
        areas = {f"{c} area (cm2)": round(calculate_area(mask[c], scale), 2) for c in mask}
        colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
        col1, col2 = st.columns(2)
        col1.image(Image.fromarray(img_rgb_cropped), caption="Original (cropped)", width='stretch')
        col2.image(colorized_rgb, caption="Classification (cropped)", width='stretch')
        st.download_button("Download colorized image (PNG)", data=png_bytes,
                           file_name=download_filename, mime="image/png")
        st.subheader("Results")
        st.json({**percentages, **areas})


else:
    cols = st.columns([1, 1, 2])
    if _TKINTER_AVAILABLE:
        if cols[0].button("Select folder"):
            folder = pick_folder()
            if folder:
                st.session_state.folder = folder
    else:
        typed = cols[0].text_input(
            "Folder path", value=st.session_state.get("folder", ""),
            placeholder="/path/to/images", key="_folder_text"
        )
        if typed:
            st.session_state.folder = typed

    folder = st.session_state.get("folder")
    if not folder:
        st.info("Enter a folder path to continue.")
        st.stop()

    if st.session_state.last_folder_used != folder:
        st.session_state.selected_files = []
        st.session_state.last_folder_used = folder

    st.write(f"Selected folder: {folder}")

    colorized_dir = Path(folder) / "Colorized"
    colorized_dir.mkdir(exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_imgs = sorted([str(Path(folder) / f) for f in os.listdir(folder) if f.lower().endswith(exts)])
    if not all_imgs:
        st.warning("No supported images found in folder.")
        st.stop()

    page_size   = st.sidebar.number_input("Page size (thumbnails)", min_value=6, max_value=60, value=12, step=6)
    total_pages = max(1, math.ceil(len(all_imgs) / page_size))
    page        = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start       = (page - 1) * page_size
    page_items  = all_imgs[start:min(len(all_imgs), start + page_size)]

    select_all_toggle = st.checkbox("Select all images", value=False)
    if select_all_toggle:
        st.session_state.selected_files = all_imgs.copy()

    prev_sel    = st.session_state.get("selected_files", [])
    default_sel = [p for p in prev_sel if p in all_imgs]
    selected_files = st.multiselect(
        "Select images for annotation and analysis",
        options=all_imgs, default=default_sel, key="selected_files"
    )

    mass_cols = st.columns(3)
    if mass_cols[0].button("Add page to selection"):
        st.session_state.selected_files = sorted(list(set(st.session_state.selected_files).union(set(page_items))))
        st.rerun()
    if mass_cols[1].button("Select all"):
        st.session_state.selected_files = all_imgs.copy()
        st.rerun()
    if mass_cols[2].button("Clear selection"):
        st.session_state.selected_files = []
        st.rerun()

    seed_path    = (st.session_state.selected_files[0] if st.session_state.selected_files else all_imgs[0])
    pil_seed     = Image.open(seed_path).convert("RGB")
    img_bgr_seed = cv2.cvtColor(np.ascontiguousarray(np.array(pil_seed).astype(np.uint8)), cv2.COLOR_RGB2BGR)

    interactive_classes = []
    if pdf_file and os.path.exists(pdf_file):
        interactive_classes = pdf_classes_from_image(img_bgr_seed, pdf_file)
    if not interactive_classes:
        interactive_classes = ["background", "plant", "lesion"]

    for p in all_imgs:
        st.session_state.classes_map[p] = interactive_classes
        ensure_points_struct(p, interactive_classes)

    st.write("### Thumbnails (current page)")
    num_cols = 4
    for i in range(0, len(page_items), num_cols):
        row_items = page_items[i:i + num_cols]
        cols_row  = st.columns(len(row_items))
        for col, p in zip(cols_row, row_items):
            with col:
                try:
                    pil_thumb = Image.open(p).convert("RGB")
                except Exception:
                    st.warning(f"Failed to open: {os.path.basename(p)}")
                    continue
                col.image(pil_thumb, caption=os.path.basename(p), width='stretch')
                pts_count = sum(len(st.session_state.points[p][c]) for c in interactive_classes)
                st.caption(f"Points collected: {pts_count}")
                btn_key = make_key("annotate", p)
                if st.button("Annotate", key=btn_key):
                    img_rgb_cur = np.ascontiguousarray(np.array(pil_thumb).astype(np.uint8))
                    enter_annotation("folder", p, img_rgb_cur, interactive_classes)
                    if p not in st.session_state.selected_files:
                        st.session_state.selected_files.append(p)
                    st.rerun()

    st.divider()

    annotated_files  = [p for p in all_imgs if sum(len(st.session_state.points[p][c]) for c in interactive_classes) > 0]
    files_to_process = sorted(set(st.session_state.selected_files).union(set(annotated_files)))

    if not files_to_process:
        st.info("Select images in the multiselect or annotate points in some images to proceed.")
        st.stop()

    tif_candidates = [str(Path(folder) / f) for f in os.listdir(folder) if f.lower().endswith(".tif")]
    scale_global   = load_scale_from_tif(tif_candidates[0]) if tif_candidates else None
    if options.disease == "Stink Bug":
        for _sb in (Path(folder) / "stinkbug.tif", Path(folder) / "stinkbug.TIF"):
            if _sb.exists():
                scale_global = load_scale_from_tif(str(_sb))
                break

    analyze_btn_label = (
        "Analyze folder and save (Interactive)"
        if classification_mode == "Interactive (clicks)"
        else "Analyze folder and save (Automatic PDF)"
    )

    if st.button(analyze_btn_label):
        results_rows = []
        n = len(files_to_process)
        progress_bar = st.progress(0)
        status = st.empty()

        use_pdf_mode = True
        model = None

        if classification_mode == "Automatic (PDF)":
            if not (pdf_file and os.path.exists(pdf_file)):
                st.error("Disease PDF file not found.")
                st.stop()
        else:
            aggregated_samples = {c: [] for c in interactive_classes}
            for p in files_to_process:
                try:
                    pil_img_p = Image.open(p).convert("RGB")
                    img_rgb_p = np.ascontiguousarray(np.array(pil_img_p).astype(np.uint8))
                    samp = collect_samples_from_points_with_xy(
                        img_rgb_p, st.session_state.points[p],
                        use_xy=use_xy_features, crop=crop_tuple
                    )
                    for c in interactive_classes:
                        aggregated_samples[c].extend(samp.get(c, []))
                except Exception as e:
                    st.warning(f"Error preparing samples from {os.path.basename(p)}: {e}")

            classes_to_train = [c for c in interactive_classes if len(aggregated_samples[c]) > 0]

            if len(classes_to_train) >= 2:
                train_samp = {c: aggregated_samples[c] for c in classes_to_train}
                model = train_classifier(train_samp)
                model._use_xy = use_xy_features
                model._classes_to_train = classes_to_train
                use_pdf_mode = False
            else:
                if use_pdf_fallback and pdf_file and os.path.exists(pdf_file):
                    st.info("Insufficient points to train. Using PDF as fallback.")
                else:
                    st.warning("Insufficient points and PDF fallback disabled or unavailable.")
                    st.stop()

        mode_label = "Automatic PDF" if use_pdf_mode else "Interactive"
        st.subheader(f"Processing selected images ({mode_label})")

        for i, p in enumerate(files_to_process, start=1):
            status.write(f"Processing {i}/{n}: {os.path.basename(p)}")
            try:
                pil_img_p    = Image.open(p).convert("RGB")
                img_rgb_p    = np.ascontiguousarray(np.array(pil_img_p).astype(np.uint8))
                img_rgb_c, _ = apply_crop(img_rgb_p, crop_top, crop_bottom, crop_left, crop_right)

                if use_pdf_mode:
                    img_bgr_c = cv2.cvtColor(img_rgb_c, cv2.COLOR_RGB2BGR)
                    _, mask, _ = bayes(img_bgr_c, pdf_file, options.disease)
                else:
                    mask = classify_image_with_xy(img_rgb_c, model, model._classes_to_train)

                colorized_bgr = np.ascontiguousarray(colorize(mask, options.disease).astype(np.uint8))

                rows, _ = process_image_result(
                    img_rgb_c, mask, colorized_bgr,
                    options.disease, p, scale_global, colorized_dir
                )
                results_rows.extend(rows)

            except Exception as e:
                st.warning(f"Error processing {os.path.basename(p)}: {e}")
            progress_bar.progress(i / n)

        status.write("Done.")

        if results_rows:
            df       = pd.DataFrame(results_rows)
            csv_path = Path(folder) / f"results_{options.disease}_{Path(folder).name}.csv"
            df.to_csv(csv_path, index=False, sep=';')

            xlsx_path = Path(folder) / f"results_{options.disease}_{Path(folder).name}.xlsx"
            excel_ok  = False
            for engine in ("openpyxl", "xlsxwriter"):
                try:
                    with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
                        df.to_excel(writer, index=False, sheet_name="Results")
                    excel_ok = True
                    break
                except Exception:
                    pass

            st.success(f"Colorized images saved at: {colorized_dir}")
            if excel_ok:
                st.success(f"Spreadsheets saved at: {csv_path} and {xlsx_path}")
            else:
                st.warning("Could not save Excel (.xlsx). Install 'openpyxl' or 'xlsxwriter'. CSV was saved.")
                st.info(f"CSV saved at: {csv_path}")
