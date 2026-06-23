"""
dr_classifier.py — Disease Ranger: lógica de classificação (PDF Naive Bayes + Interativo RF).

Melhorias de performance no modo Interativo:
- RandomForestClassifier com n_estimators reduzido (30 ao invés de 100) — 3x mais rápido,
  mantendo precisão suficiente para classificação RGB/5-features.
- features_for_image usa float32 em vez de float64 — metade da memória.
- predict_proba desabilitado (predict direto) — mais rápido.
- Colorização determinística para Stink Bug evita chamadas desnecessárias ao PlantCV.
"""
import cv2
import numpy as np
from plantcv import plantcv as pcv
from sklearn.ensemble import RandomForestClassifier

from dr_utils import crop_bounds, thumbnail_bgr


# =====================================================
# PRR — auxiliares de pós-processamento espacial
# (portado do version08_09.py para restaurar precisão da versão offline)
# =====================================================
def _estimate_root_cut(mask_plant):
    """Estima a linha de corte entre parte aérea e raiz pela última linha com planta consistente."""
    projection = np.sum(mask_plant > 0, axis=1)
    if np.max(projection) == 0:
        return mask_plant.shape[0] // 2
    norm = projection / np.max(projection)
    for i in range(len(norm) - 1, -1, -1):
        if norm[i] > 0.25:
            return i
    return mask_plant.shape[0] // 2


def _enforce_root_connectivity(mask_root, bottom_ratio=0.08):
    """Mantém apenas regiões de raiz conectadas à faixa inferior da imagem."""
    h, w = mask_root.shape
    band_start = int(h * (1 - bottom_ratio))
    num_labels, labels = cv2.connectedComponents(mask_root.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask_root
    valid = np.zeros_like(mask_root, dtype=np.uint8)
    bottom_labels = np.unique(labels[band_start:h, :])
    bottom_labels = bottom_labels[bottom_labels != 0]
    for lbl in bottom_labels:
        valid[labels == lbl] = 1
    return valid


def _find_closest_class(color, class_colors):
    """Retorna a classe cuja cor de referência é a mais próxima da cor fornecida."""
    closest = None
    min_dist = float('inf')
    for cls, ref in class_colors.items():
        dist = float(np.linalg.norm(np.array(color, dtype=np.float32) - np.array(ref, dtype=np.float32)))
        if dist < min_dist:
            min_dist = dist
            closest = cls
    return closest


def _apply_prr_spatial_processing(img_bgr, mask):
    """
    Aplica pós-processamento espacial específico de PRR:
    - Separa parte aérea (plant/lesion) da raiz usando corte dinâmico
    - Reforça conectividade da raiz à faixa inferior
    - Reclassifica pixels não atribuídos pelo Naive Bayes
    """
    h = img_bgr.shape[0]

    # --- Linha de corte dinâmica ---
    if 'plant' in mask:
        kernel = np.ones((9, 9), np.uint8)
        plant_clean = cv2.morphologyEx(mask['plant'], cv2.MORPH_CLOSE, kernel)
        cut_line = _estimate_root_cut(plant_clean)
    else:
        cut_line = h // 2

    cut_line = int(np.clip(cut_line, h * 0.15, h * 0.85))
    cut_line = max(cut_line, int(h * 0.4))

    mask_upper = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    mask_upper[:cut_line, :] = 1
    mask_lower = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    mask_lower[cut_line:, :] = 1

    # --- Restrições espaciais por classe ---
    if 'raiz' in mask:
        mask['raiz'] = mask['raiz'] * mask_lower
        mask['raiz'] = cv2.erode(mask['raiz'], np.ones((3, 3), np.uint8), iterations=1)
        raiz_before = mask['raiz'].copy()
        mask['raiz'] = _enforce_root_connectivity(mask['raiz'])
        if np.count_nonzero(mask['raiz']) < 0.1 * np.count_nonzero(raiz_before):
            mask['raiz'] = raiz_before

    if 'lesion' in mask:
        mask['lesion'] = mask['lesion'] * mask_upper

    # fallback: se raiz ficou vazia, usa pixels escuros abaixo do corte
    if 'raiz' in mask and np.count_nonzero(mask['raiz']) == 0:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        dark_mask = (hsv[:, :, 2] < 90)
        mask['raiz'] = (dark_mask & mask_lower.astype(bool)).astype(np.uint8)

    # --- Reclassificação de pixels não atribuídos ---
    total_classified = np.zeros(img_bgr.shape[:2], dtype=bool)
    for m in mask.values():
        total_classified |= m.astype(bool)
    unclassified = ~total_classified

    if np.any(unclassified):
        # Cores de referência em BGR (ordem dos canais do cv2)
        lower_colors = {'raiz': (19, 69, 139), 'plant': (0, 128, 0), 'background': (255, 255, 255)}
        upper_colors = {'lesion': (0, 255, 255), 'plant': (0, 128, 0), 'background': (255, 255, 255)}

        unc_lower = unclassified & mask_lower.astype(bool)
        if np.any(unc_lower):
            temp = unc_lower.astype(np.uint8)
            region_color = cv2.mean(img_bgr, mask=temp)[:3]
            cls = _find_closest_class(region_color, lower_colors)
            mask.setdefault(cls, np.zeros(img_bgr.shape[:2], dtype=np.uint8))
            mask[cls] = np.clip(mask[cls].astype(np.int32) + temp, 0, 1).astype(np.uint8)

        unc_upper = unclassified & mask_upper.astype(bool)
        if np.any(unc_upper):
            temp = unc_upper.astype(np.uint8)
            region_color = cv2.mean(img_bgr, mask=temp)[:3]
            cls = _find_closest_class(region_color, upper_colors)
            mask.setdefault(cls, np.zeros(img_bgr.shape[:2], dtype=np.uint8))
            mask[cls] = np.clip(mask[cls].astype(np.int32) + temp, 0, 1).astype(np.uint8)

    return mask


# =====================================================
# PDF (Naive Bayes PlantCV)
# =====================================================
def pdf_classes_from_image(img_bgr, pdf_file):
    """Lê as classes do arquivo PDF classificando um thumbnail pequeno."""
    try:
        thumb = thumbnail_bgr(img_bgr, max_side=128)
        # PlantCV usa cv2.COLOR_BGR2HSV internamente — passa BGR diretamente
        mask = pcv.naive_bayes_classifier(rgb_img=thumb, pdf_file=pdf_file)
        return list(mask.keys())
    except Exception:
        return []


def bayes(img_bgr, pdf_file, disease: str):
    """Classifica imagem com Naive Bayes (PDF) e retorna percentuais, mask e denominador.
    PlantCV converte internamente com cv2.COLOR_BGR2HSV, portanto img_bgr é passado
    diretamente sem converter para RGB — os PDFs foram treinados com imagens BGR.
    """
    mask = pcv.naive_bayes_classifier(rgb_img=img_bgr, pdf_file=pdf_file)
    if disease == "PRR":
        mask = _apply_prr_spatial_processing(img_bgr, mask)
    counts = counts_from_mask(mask)
    percentages, denom = percentages_for_disease(disease, counts)
    return percentages, mask, denom


# =====================================================
# INTERATIVO (Random Forest)
# =====================================================
def collect_samples_from_points_with_xy(img_rgb, points_per_class, use_xy=True, crop=(0, 0, 0, 0)):
    """
    Coleta amostras RGB (+ xy normalizados opcionais) dos pontos anotados dentro do recorte.
    """
    h, w, _ = img_rgb.shape
    top, bottom, left, right = crop
    y0, y1, x0, x1 = crop_bounds(h, w, top, bottom, left, right)
    ch = y1 - y0
    cw = x1 - x0

    samples = {}
    for cls, pts in points_per_class.items():
        feats = []
        for (x, y) in pts:
            if x0 <= x < x1 and y0 <= y < y1:
                R, G, B = img_rgb[y, x]
                if use_xy:
                    x_norm = (x - x0) / (cw - 1) if cw > 1 else 0.0
                    y_norm = (y - y0) / (ch - 1) if ch > 1 else 0.0
                    feats.append([int(R), int(G), int(B), float(x_norm), float(y_norm)])
                else:
                    feats.append([int(R), int(G), int(B)])
        samples[cls] = feats
    return samples


def train_classifier(samples):
    """
    Treina RandomForest para classificação por pixel.
    n_estimators=30: 3x mais rápido que 100 com precisão muito próxima para esta tarefa.
    n_jobs=-1: paraleliza no número de CPUs disponíveis.
    """
    X, y = [], []
    for cls, feats in samples.items():
        for v in feats:
            X.append(v)
            y.append(cls)
    model = RandomForestClassifier(
        n_estimators=30,
        random_state=42,
        n_jobs=-1,          # usa todos os núcleos disponíveis
        max_depth=None,     # mantém profundidade irrestrita
    )
    model.fit(X, y)
    return model


def features_for_image(img_rgb, use_xy=True):
    """
    Constrói matriz de atributos para todos os pixels (float32 — metade da memória vs float64).
    Shape: (h*w, 3) ou (h*w, 5)
    """
    h, w, _ = img_rgb.shape
    R = img_rgb[:, :, 0].reshape(-1).astype(np.float32)
    G = img_rgb[:, :, 1].reshape(-1).astype(np.float32)
    B = img_rgb[:, :, 2].reshape(-1).astype(np.float32)
    if not use_xy:
        return np.column_stack([R, G, B])

    xs = np.tile(np.arange(w, dtype=np.float32), h)
    ys = np.repeat(np.arange(h, dtype=np.float32), w)
    x_norm = xs / (w - 1) if w > 1 else np.zeros(h * w, dtype=np.float32)
    y_norm = ys / (h - 1) if h > 1 else np.zeros(h * w, dtype=np.float32)

    return np.column_stack([R, G, B, x_norm, y_norm])


def classify_image_with_xy(img_rgb_cropped, model, classes):
    """
    Classifica por pixel usando o modelo treinado. Recebe a imagem JÁ recortada.
    Usa predict em batch (toda a imagem de uma vez) — eficiente no scikit-learn.
    """
    h, w, _ = img_rgb_cropped.shape
    use_xy = bool(getattr(model, "_use_xy", False))
    X = features_for_image(img_rgb_cropped, use_xy=use_xy)
    preds = model.predict(X)
    pred_img = np.array(preds).reshape(h, w)
    return {cls: (pred_img == cls).astype(np.uint8) for cls in classes}


# =====================================================
# COLORIZAÇÃO
# =====================================================
_STINKBUG_COLORS_RGB = {
    "wbackground": (255, 255, 255),
    "bbackground": (0, 0, 255),
    "healthy": (0, 255, 0),
    "lesion": (0, 0, 0),
    "referencesquare": (255, 165, 0),
}
_STINKBUG_PRIORITY = ["wbackground", "bbackground", "healthy", "lesion", "referencesquare"]

_COLORIZE_NAMES = {
    "plant": "green", "raiz": "brown", "soil": "brown",
    "background": "white", "healthy": "green", "lesion": "black",
    "clorosis": "yellow", "haste": "green", "cyst": "red", "dirt": "brown",
    "wbackground": "white", "bbackground": "blue", "referencesquare": "yellow",
}


def colorize(mask, disease):
    """
    Retorna imagem BGR colorizada para a máscara classificada.
    Para Stink Bug usa lógica determinística (sem PlantCV) — mais rápido e correto.
    """
    if disease == "Stink Bug":
        first = next(iter(mask.values()))
        h, w = first.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in _STINKBUG_PRIORITY:
            m = mask.get(cls)
            if m is not None:
                rgb[(m > 0)] = _STINKBUG_COLORS_RGB[cls]
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    masks, cols = [], []
    for k in mask:
        masks.append(mask[k])
        cols.append(_COLORIZE_NAMES.get(k, "white"))
    return pcv.visualize.colorize_masks(masks=masks, colors=cols)


# =====================================================
# PERCENTUAIS
# =====================================================
def counts_from_mask(mask: dict) -> dict:
    return {k: int(np.count_nonzero(mask.get(k, 0))) for k in mask}


def percentages_for_disease(disease: str, counts: dict) -> tuple:
    """
    Retorna (dict_percentuais, denominador) seguindo regras por doença:
    - PRR:         denom = plant + raiz + background; background% calculado sobre total de pixels
    - Vagem / FLS: denom = healthy + lesion
    - TLS:         denom = healthy + clorosis + lesion
    - Outros:      denom = total de todos os pixels classificados
    """
    d = disease or ""
    h = counts.get("healthy", 0)
    l = counts.get("lesion", 0)
    c = counts.get("clorosis", 0)
    if d == "PRR":
        plant  = counts.get("plant", 0)
        raiz   = counts.get("raiz", 0)
        bg     = counts.get("background", 0)
        total_px = sum(counts.values())
        denom  = plant + raiz + bg
        if denom == 0:
            return {"background (%)": 0.0, "plant (%)": 0.0, "raiz (%)": 0.0, "lesion (%)": 0.0}, denom
        return {
            "background (%)": round(100 * bg / total_px, 2) if total_px else 0.0,
            "plant (%)":      round(100 * plant / denom, 2),
            "raiz (%)":       round(100 * raiz  / denom, 2),
            "lesion (%)":     round(100 * l     / denom, 2),
        }, denom
    elif d in ("Vagem", "FLS"):
        denom = h + l
        if denom == 0:
            return {"healthy (%)": 0.0, "lesion (%)": 0.0}, denom
        return {
            "healthy (%)": round(100 * h / denom, 2),
            "lesion (%)": round(100 * l / denom, 2),
        }, denom
    elif d == "TLS":
        denom = h + c + l
        if denom == 0:
            return {"healthy (%)": 0.0, "clorosis (%)": 0.0, "lesion (%)": 0.0}, denom
        return {
            "healthy (%)": round(100 * h / denom, 2),
            "clorosis (%)": round(100 * c / denom, 2),
            "lesion (%)": round(100 * l / denom, 2),
        }, denom
    else:
        total = sum(counts.values())
        if total == 0:
            return {f"{k} (%)": 0.0 for k in counts}, total
        return {f"{k} (%)": round(100 * counts[k] / total, 2) for k in counts}, total
