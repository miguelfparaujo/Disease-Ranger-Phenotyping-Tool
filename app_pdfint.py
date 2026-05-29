piuimport streamlit as st
import numpy as np
import os
import cv2
import tempfile
from PIL import Image
from plantcv import plantcv as pcv

from streamlit_image_coordinates import streamlit_image_coordinates
from sklearn.ensemble import RandomForestClassifier

# Tkinter para seleção de pasta
import tkinter as tk
from tkinter import filedialog

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Disease Ranger", page_icon="🧬", layout="wide")

with st.sidebar:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.divider()

# =====================================================
# OPTIONS
# =====================================================
class Options:
    def __init__(self):
        self.disease = None

options = Options()

# =====================================================
# UTILS
# =====================================================
def pick_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory()
    root.destroy()
    return folder


def load_scale_from_tif(tif_path):
    with Image.open(tif_path) as img:
        dpi = img.info.get("dpi", (96, 96))
        return 2.54 / dpi[0]


def calculate_area(mask, scale):
    bin_mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = 0
    for c in contours:
        a = cv2.contourArea(c)
        if scale:
            total_area += a * (scale ** 2)
    return total_area


def colorize(mask):
    colors = {
        "plant": "green",
        "raiz": "brown",
        "soil": "brown",
        "background": "white",
        "healthy": "green",
        "lesion": "black",
        "clorosis": "yellow",
        "haste": "green",
        "cyst": "red",
        "dirt": "brown",
    }

    masks, cols = [], []
    for k in mask:
        masks.append(mask[k])
        cols.append(colors.get(k, "white"))

    return pcv.visualize.colorize_masks(masks=masks, colors=cols)


def class_color_bgr(cls):
    color_map = {
        "plant": "green",
        "raiz": "brown",
        "soil": "brown",
        "background": "white",
        "healthy": "green",
        "lesion": "black",
        "clorosis": "yellow",
        "haste": "green",
        "cyst": "red",
        "dirt": "brown",
    }
    table = {
        "white": (255, 255, 255),
        "green": (0, 255, 0),
        "brown": (42, 42, 165),
        "black": (0, 0, 0),
        "yellow": (0, 255, 255),
        "red": (0, 0, 255),
    }
    return table.get(color_map.get(cls, "white"), (255, 255, 255))


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

    # Log de cliques para desfazer
    if "click_log" not in st.session_state:
        st.session_state.click_log = {}
    st.session_state.click_log.setdefault(image_key, [])


def collect_samples_from_points(img_rgb, points_per_class):
    samples = {}
    for cls, pts in points_per_class.items():
        pixels = []
        for (x, y) in pts:
            pixels.append(img_rgb[y, x])
        samples[cls] = pixels
    return samples


def samples_ready_total(points_per_class, min_total=20):
    total = sum(len(pts) for pts in points_per_class.values())
    return total >= min_total, total


def draw_marker_on_canvas(canvas, xd, yd, color_rgb):
    cv2.circle(canvas, (int(xd), int(yd)), radius=6, color=color_rgb, thickness=-1)
    cv2.circle(canvas, (int(xd), int(yd)), radius=10, color=color_rgb, thickness=2)


def fit_to_square_display(img_rgb, zoom, x0, y0, canvas_size=800):
    """
    Ajusta a imagem para caber inteira em um quadrado (canvas_size x canvas_size) quando zoom=1.
    Para zoom>1, faz crop no original em (x0, y0) com janela proporcional e redimensiona para preencher
    o mesmo retângulo de conteúdo dentro do canvas (letterboxing nas bordas).
    Retorna:
      - display_canvas (RGB uint8) de tamanho canvas_size x canvas_size
      - mapping dict com informações para converter coordenadas de clique para original
    """
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
        "canvas_size": canvas_size,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "s_fit": s_fit,
        "zoom": float(zoom),
        "x0": int(x0),
        "y0": int(y0),
        "crop_w": int(crop_w),
        "crop_h": int(crop_h),
        "content_w": int(content_w),
        "content_h": int(content_h),
    }
    return canvas, mapping


def display_to_original_coords(xd, yd, mapping, img_shape):
    """
    Converte coordenadas de clique no canvas (letterboxed) para coordenadas originais.
    Retorna (x, y) no original ou None se clique fora do conteúdo.
    """
    h, w, _ = img_shape
    offset_x = mapping["offset_x"]
    offset_y = mapping["offset_y"]
    content_w = mapping["content_w"]
    content_h = mapping["content_h"]
    s_fit = mapping["s_fit"]
    zoom = mapping["zoom"]
    x0 = mapping["x0"]
    y0 = mapping["y0"]

    if not (offset_x <= xd < offset_x + content_w and offset_y <= yd < offset_y + content_h):
        return None

    x_rel_disp = xd - offset_x
    y_rel_disp = yd - offset_y

    x_rel_orig = x_rel_disp / (s_fit * zoom)
    y_rel_orig = y_rel_disp / (s_fit * zoom)

    x = int(round(x0 + x_rel_orig))
    y = int(round(y0 + y_rel_orig))

    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    return (x, y)


def draw_markers_on_display(canvas, points_per_class, mapping):
    out = np.ascontiguousarray(canvas.copy())
    offset_x = mapping["offset_x"]
    offset_y = mapping["offset_y"]
    s_fit = mapping["s_fit"]
    zoom = mapping["zoom"]
    x0 = mapping["x0"]
    y0 = mapping["y0"]
    crop_w = mapping["crop_w"]
    crop_h = mapping["crop_h"]

    for cls, pts in points_per_class.items():
        color_bgr = class_color_bgr(cls)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        for (x, y) in pts:
            if x0 <= x < x0 + crop_w and y0 <= y < y0 + crop_h:
                xd = offset_x + int(round((x - x0) * s_fit * zoom))
                yd = offset_y + int(round((y - y0) * s_fit * zoom))
                draw_marker_on_canvas(out, xd, yd, color_rgb)
    return out


def thumbnail_bgr(img_bgr, max_side=256):
    h, w, _ = img_bgr.shape
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    thumb = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return thumb


def pdf_classes_from_image(img_bgr, pdf_file):
    try:
        thumb = thumbnail_bgr(img_bgr, max_side=128)
        mask = pcv.naive_bayes_classifier(rgb_img=thumb, pdf_file=pdf_file)
        return list(mask.keys())
    except Exception:
        return []


# =====================================================
# PLANTCV (PDF)
# =====================================================
def bayes(img_bgr, pdf_file):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if options.disease == "PRR":
        img = img[:, 400:]
    elif options.disease == "CHR":
        img = img[480:img.shape[0]-136, 360:img.shape[1]-766]

    mask = pcv.naive_bayes_classifier(rgb_img=img, pdf_file=pdf_file)

    class_counts = {k: np.count_nonzero(mask[k]) for k in mask}
    total = sum(class_counts.values())

    percentages = {
        f"{k} (%)": round(class_counts[k] * 100 / total, 2) if total else 0
        for k in class_counts
    }

    return percentages, mask, total


# =====================================================
# INTERATIVO (treino e classificação)
# =====================================================
def train_classifier(samples):
    X, y = [], []
    for cls, pixels in samples.items():
        for p in pixels:
            X.append(p)
            y.append(cls)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def classify_image(img_rgb, model, classes):
    h, w, _ = img_rgb.shape
    pixels = img_rgb.reshape(-1, 3)
    preds = model.predict(pixels)
    pred_img = preds.reshape(h, w)
    return {cls: (pred_img == cls).astype(np.uint8) for cls in classes}


# =====================================================
# UI STATE
# =====================================================
st.title("🧬 Disease Ranger – Web Edition")

options.disease = st.sidebar.selectbox(
    "Selecione a doença",
    ["PRR", "TLS", "FLS", "Vagem", "PGR", "CHR", "SCN"]
)

classification_mode = st.sidebar.radio("Modo de classificação", ["Automático (PDF)", "Interativo (cliques)"])

pdf_files = {
    "PRR": "Nayve/naive_bayes_pdfs_newPRR.txt",
    "FLS": "Nayve/naive_bayes_pdfsFLS.txt",
    "TLS": "Nayve/naive_bayes_TLS0825pdf.txt",
    "Vagem": "Nayve/naive_bayes_pdfsVagem.txt",
    "CHR": "Nayve/naive_bayes_canopyout25pdf.txt",
    "PGR": "Nayve/pgr_pdf.txt",
    "SCN": "Nayve/nematodes_pdf.txt"
}
pdf_file = pdf_files.get(options.disease)

mode = st.sidebar.radio("Modo de operação", ["Imagem única", "Pasta"])
st.divider()

# Estados
if "annotate_mode" not in st.session_state:
    st.session_state.annotate_mode = None  # None, "single", "folder"
if "annotate_image_key" not in st.session_state:
    st.session_state.annotate_image_key = None
if "images_cache" not in st.session_state:
    st.session_state.images_cache = {}
if "classes_map" not in st.session_state:
    st.session_state.classes_map = {}  # image_key -> list of classes

# Estados extras para persistência entre reruns (Single)
if "last_single_image_key" not in st.session_state:
    st.session_state.last_single_image_key = None
if "single_scale" not in st.session_state:
    st.session_state.single_scale = None


def enter_annotation(mode_key, image_key, img_rgb, classes_for_image):
    st.session_state.annotate_mode = mode_key
    st.session_state.annotate_image_key = image_key
    st.session_state.images_cache[image_key] = np.ascontiguousarray(img_rgb.copy().astype(np.uint8))
    st.session_state.classes_map[image_key] = classes_for_image
    ensure_points_struct(image_key, classes_for_image)


def exit_annotation():
    st.session_state.annotate_mode = None
    st.session_state.annotate_image_key = None


# =====================================================
# PÁGINA DEDICADA DE ANOTAÇÃO (com canvas quadrado, zoom e desfazer)
# =====================================================
def annotation_page(active_class_key):
    image_key = st.session_state.annotate_image_key
    if not image_key or image_key not in st.session_state.images_cache:
        st.error("Imagem de anotação não encontrada.")
        if st.button("Voltar"):
            exit_annotation()
            st.rerun()
        st.stop()

    img_rgb = st.session_state.images_cache[image_key]
    classes_for_image = st.session_state.classes_map.get(image_key, [])
    if not classes_for_image:
        st.error("Não foi possível obter as classes para anotação desta imagem.")
        if st.button("Voltar"):
            exit_annotation()
            st.rerun()
        st.stop()

    ensure_points_struct(image_key, classes_for_image)

    st.subheader("🖱️ Anotação de pontos RGB")

    # Seleção da classe ativa na própria página de anotação
    if active_class_key not in st.session_state:
        st.session_state[active_class_key] = classes_for_image[0]
    active_class = st.selectbox(
        "Classe ativa para marcar",
        classes_for_image,
        index=classes_for_image.index(st.session_state[active_class_key]),
        key=active_class_key
    )

    canvas_size = st.slider("Tamanho do canvas (px)", min_value=400, max_value=1200, value=800, step=50, key=f"canvas_{image_key}")
    zoom = st.slider("Zoom", min_value=1.0, max_value=8.0, value=1.0, step=0.1, key=f"zoom_{image_key}")

    h, w, _ = img_rgb.shape
    # Pan inicial
    pan_x_key = f"pan_x_{image_key}"
    pan_y_key = f"pan_y_{image_key}"
    if pan_x_key not in st.session_state:
        st.session_state[pan_x_key] = 0
    if pan_y_key not in st.session_state:
        st.session_state[pan_y_key] = 0

    # Controles de pan
    pan_cols = st.columns([1, 1, 2])
    st.session_state[pan_x_key] = pan_cols[0].number_input(
        "Pan X (px)", min_value=0, max_value=max(0, w-1), value=st.session_state[pan_x_key], step=50
    )
    st.session_state[pan_y_key] = pan_cols[1].number_input(
        "Pan Y (px)", min_value=0, max_value=max(0, h-1), value=st.session_state[pan_y_key], step=50
    )
    step = pan_cols[2].slider("Passo das setas (px)", min_value=10, max_value=200, value=50, step=10, key=f"step_{image_key}")
    arrow_cols = st.columns(4)
    if arrow_cols[0].button("←", key=f"left_{image_key}"):
        st.session_state[pan_x_key] = max(0, st.session_state[pan_x_key] - step)
    if arrow_cols[1].button("→", key=f"right_{image_key}"):
        st.session_state[pan_x_key] = min(max(0, w-1), st.session_state[pan_x_key] + step)
    if arrow_cols[2].button("↑", key=f"up_{image_key}"):
        st.session_state[pan_y_key] = max(0, st.session_state[pan_y_key] - step)
    if arrow_cols[3].button("↓", key=f"down_{image_key}"):
        st.session_state[pan_y_key] = min(max(0, h-1), st.session_state[pan_y_key] + step)

    # Renderização no canvas quadrado com letterboxing
    canvas_img, mapping = fit_to_square_display(
        img_rgb, zoom, st.session_state[pan_x_key], st.session_state[pan_y_key], canvas_size=canvas_size
    )

    # Desenha marcadores no canvas
    canvas_marked = draw_markers_on_display(canvas_img, st.session_state.points[image_key], mapping)

    # Exibe imagem clicável e captura coordenadas
    coords = streamlit_image_coordinates(canvas_marked, key=f"coords_{image_key}", width=canvas_size)

    # Ações de desfazer
    undo_cols = st.columns(3)
    if undo_cols[0].button("Desfazer (classe ativa)", key=f"undo_cls_{image_key}"):
        if st.session_state.points[image_key][active_class]:
            removed = st.session_state.points[image_key][active_class].pop()
            if st.session_state.click_log[image_key]:
                for i in range(len(st.session_state.click_log[image_key]) - 1, -1, -1):
                    (cls_log, pt_log) = st.session_state.click_log[image_key][i]
                    if cls_log == active_class and pt_log == removed:
                        st.session_state.click_log[image_key].pop(i)
                        break
            st.success(f"Removido último ponto da classe {active_class}: {removed}")
        else:
            st.info("Nenhum ponto para remover na classe ativa.")
    if undo_cols[1].button("Desfazer (último)", key=f"undo_last_{image_key}"):
        if st.session_state.click_log[image_key]:
            cls_log, pt_log = st.session_state.click_log[image_key].pop()
            if pt_log in st.session_state.points[image_key][cls_log]:
                st.session_state.points[image_key][cls_log].remove(pt_log)
            st.success(f"Removido último ponto [{cls_log}] {pt_log}")
        else:
            st.info("Nenhum ponto para desfazer.")
    if undo_cols[2].button("Limpar todos os pontos", key=f"clear_{image_key}"):
        st.session_state.points[image_key] = {c: [] for c in classes_for_image}
        st.session_state.click_log[image_key] = []
        st.rerun()

    # Registrar clique
    if coords is not None and "x" in coords and "y" in coords:
        xd = float(coords["x"])
        yd = float(coords["y"])
        orig = display_to_original_coords(xd, yd, mapping, img_rgb.shape)
        if orig is None:
            st.info("Clique fora da área da imagem. Ajuste pan/zoom e tente novamente.")
        else:
            x, y = orig
            st.session_state.points[image_key][active_class].append((x, y))
            st.session_state.click_log[image_key].append((active_class, (x, y)))
            st.success(f"[{image_key}] Ponto ({x}, {y}) adicionado à classe {active_class}")

    # Status dos pontos
    ready_total, total_pts = samples_ready_total(st.session_state.points[image_key], min_total=20)
    st.write(f"Total de cliques: {total_pts}")
    st.write("### Pontos coletados (por classe)")
    for c in classes_for_image:
        st.write(f"- {c}: {len(st.session_state.points[image_key][c])}")

    # Rodapé com salvar/cancelar
    cols_footer = st.columns(2)
    if ready_total and cols_footer[0].button("Salvar"):
        exit_annotation()
        st.rerun()
    if cols_footer[1].button("Cancelar"):
        exit_annotation()
        st.rerun()

    if not ready_total:
        st.info("Para habilitar 'Salvar', colecione pelo menos 20 cliques no total (recomendado distribuir entre as classes).")


# Se estamos no modo anotação, renderiza página dedicada e interrompe fluxo normal
if st.session_state.annotate_mode is not None:
    active_class_key = "active_class_single" if st.session_state.annotate_mode == "single" else "active_class_folder"
    annotation_page(active_class_key)
    st.stop()

# =====================================================
# IMAGEM ÚNICA
# =====================================================
if mode == "Imagem única":
    uploaded = st.file_uploader("Envie uma imagem", type=["jpg", "png", "jpeg", "bmp"], key="single_uploader")
    uploaded_tif = st.file_uploader("Envie o TIF de escala (opcional)", type=["tif"], key="single_scale_uploader")

    # Definição do image_key e recuperação da imagem carregada
    image_key = None
    pil_img = None
    img_rgb = None

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
            pil_img = Image.fromarray(img_rgb)
        else:
            st.info("Envie uma imagem para continuar.")
            st.stop()

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Classes do PDF para ESTA imagem (persistidas em classes_map)
    interactive_classes = st.session_state.classes_map.get(image_key)
    if not interactive_classes:
        interactive_classes = []
        if pdf_file and os.path.exists(pdf_file):
            interactive_classes = pdf_classes_from_image(img_bgr, pdf_file)
        if not interactive_classes:
            interactive_classes = ["background", "plant", "lesion"]
        st.session_state.classes_map[image_key] = interactive_classes
    ensure_points_struct(image_key, interactive_classes)

    # Botão único para entrar na anotação no modo interativo
    if classification_mode == "Interativo (cliques)":
        if st.button("Marcar classe"):
            enter_annotation("single", image_key, img_rgb, interactive_classes)
            st.rerun()
    else:
        # No modo automático, esconder o botão e continuar com análise automática
        pass

    # Escala (opcional) persistida
    scale = st.session_state.single_scale
    if uploaded_tif is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_tif.read())
            st.session_state.single_scale = load_scale_from_tif(tmp.name)
        scale = st.session_state.single_scale

    # CLASSIFICAÇÃO
    if classification_mode == "Automático (PDF)":
        if pdf_file and os.path.exists(pdf_file):
            percentages, mask, _ = bayes(img_bgr, pdf_file)
        else:
            st.error("Arquivo de PDF da doença não encontrado.")
            st.stop()
    else:
        ready_total, total_pts = samples_ready_total(st.session_state.points[image_key], min_total=20)
        if not ready_total:
            st.warning("Colete ao menos 20 cliques (total) na página de anotação e clique em 'Salvar'.")
            st.stop()

        # Interativo: treino estritamente com os pontos marcados
        samples = collect_samples_from_points(img_rgb, st.session_state.points[image_key])
        model = train_classifier(samples)
        mask = classify_image(img_rgb, model, interactive_classes)

        counts = {c: np.count_nonzero(mask[c]) for c in mask}
        total = sum(counts.values())
        percentages = {f"{c} (%)": round(counts[c] * 100 / total, 2) if total else 0 for c in counts}

    # Áreas
    areas = {f"{c} area (cm²)": round(calculate_area(mask[c], scale), 2) for c in mask}

    colorized = colorize(mask)
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    col1.image(pil_img, caption="Original", use_container_width=True)
    col2.image(colorized_rgb, caption="Classificação", use_container_width=True)

    st.subheader("Resultados")
    st.json({**percentages, **areas})

# =====================================================
# PASTA (múltiplas imagens)
# =====================================================
else:
    cols = st.columns([1, 1, 2])
    if cols[0].button("Selecionar pasta"):
        folder = pick_folder()
        if folder:
            st.session_state.folder = folder

    folder = st.session_state.get("folder")
    if not folder:
        st.info("Selecione uma pasta para continuar.")
        st.stop()

    st.write(f"Pasta selecionada: {folder}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    if not all_imgs:
        st.warning("Nenhuma imagem suportada encontrada na pasta.")
        st.stop()

    selected_files = st.multiselect(
        "Selecione uma ou mais imagens para anotação e análise",
        all_imgs,
        default=st.session_state.get("selected_files", all_imgs[:1]),
        key="selected_files"
    )
    if not selected_files:
        st.info("Selecione pelo menos uma imagem.")
        st.stop()

    # Tenta localizar TIF de escala (opcional)
    tif_candidates = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".tif")]
    scale_tif_path = tif_candidates[0] if len(tif_candidates) == 1 else None
    scale_global = load_scale_from_tif(scale_tif_path) if scale_tif_path else None

    # Define classes com base na primeira imagem selecionada
    first_img_path = selected_files[0]
    pil_first = Image.open(first_img_path).convert("RGB")
    img_rgb_first = np.ascontiguousarray(np.array(pil_first).astype(np.uint8))
    img_bgr_first = cv2.cvtColor(img_rgb_first, cv2.COLOR_RGB2BGR)

    interactive_classes = []
    if pdf_file and os.path.exists(pdf_file):
        interactive_classes = pdf_classes_from_image(img_bgr_first, pdf_file)
    if not interactive_classes:
        interactive_classes = ["background", "plant", "lesion"]

    # Inicializa pontos e classes por imagem
    for p in selected_files:
        st.session_state.classes_map[p] = interactive_classes
        ensure_points_struct(p, interactive_classes)

    # Seleção de imagem ativa permanece; classe ativa foi movida para a página de anotação
    current_img_path = st.selectbox("Imagem ativa para anotação", selected_files, key="active_image_path")

    # Botão único para entrar na anotação no modo interativo
    if classification_mode == "Interativo (cliques)":
        if st.button("Marcar classe"):
            pil_img_cur = Image.open(current_img_path).convert("RGB")
            img_rgb_cur = np.ascontiguousarray(np.array(pil_img_cur).astype(np.uint8))
            enter_annotation("folder", current_img_path, img_rgb_cur, interactive_classes)
            st.rerun()

    # Classificação por pasta
    if classification_mode == "Automático (PDF)":
        if not (pdf_file and os.path.exists(pdf_file)):
            st.error("Arquivo de PDF da doença não encontrado.")
            st.stop()

        st.subheader("Análise automática (PDF) nas imagens selecionadas")
        for p in selected_files:
            pil_img_p = Image.open(p).convert("RGB")
            img_rgb_p = np.ascontiguousarray(np.array(pil_img_p).astype(np.uint8))
            img_bgr_p = cv2.cvtColor(img_rgb_p, cv2.COLOR_RGB2BGR)

            percentages, mask, _ = bayes(img_bgr_p, pdf_file)
            areas = {f"{c} area (cm²)": round(calculate_area(mask[c], scale_global), 2) for c in mask}

            colorized = colorize(mask)
            colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            col1.image(pil_img_p, caption=f"Original – {os.path.basename(p)}", use_container_width=True)
            col2.image(colorized_rgb, caption=f"Classificação – {os.path.basename(p)}", use_container_width=True)

            st.json({**percentages, **areas})

    else:
        aggregated_samples = {c: [] for c in interactive_classes}
        for p in selected_files:
            pil_img_p = Image.open(p).convert("RGB")
            img_rgb_p = np.ascontiguousarray(np.array(pil_img_p).astype(np.uint8))
            samples_p = collect_samples_from_points(img_rgb_p, st.session_state.points[p])
            for c in interactive_classes:
                aggregated_samples[c].extend(samples_p.get(c, []))

        ready_total = sum(len(aggregated_samples[c]) for c in interactive_classes) >= 20
        st.write("### Pontos agregados (todas as imagens selecionadas)")
        for c in interactive_classes:
            st.write(f"{c}: {len(aggregated_samples[c])}")
        st.write(f"Total: {sum(len(aggregated_samples[c]) for c in interactive_classes)}")

        if not ready_total:
            st.warning("Colete ao menos 20 cliques no total nas imagens selecionadas e clique em 'Salvar' na página de anotação.")
            st.stop()

        # Interativo: treino estritamente com os pontos marcados (agregados)
        model = train_classifier(aggregated_samples)

        st.subheader("Classificação interativa nas imagens selecionadas")
        for p in selected_files:
            pil_img_p = Image.open(p).convert("RGB")
            img_rgb_p = np.ascontiguousarray(np.array(pil_img_p).astype(np.uint8))

            mask = classify_image(img_rgb_p, model, interactive_classes)
            counts = {c: np.count_nonzero(mask[c]) for c in mask}
            total = sum(counts.values())
            percentages = {f"{c} (%)": round(counts[c] * 100 / total, 2) if total else 0 for c in counts}

            areas = {f"{c} area (cm²)": round(calculate_area(mask[c], scale_global), 2) for c in mask}

            colorized = colorize(mask)
            colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            col1.image(pil_img_p, caption=f"Original – {os.path.basename(p)}", use_container_width=True)
            col2.image(colorized_rgb, caption=f"Classificação – {os.path.basename(p)}", use_container_width=True)

            st.json({**percentages, **areas})