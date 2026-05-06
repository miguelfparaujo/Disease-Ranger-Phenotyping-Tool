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
# PDF (Naive Bayes PlantCV)
# =====================================================
def pdf_classes_from_image(img_bgr, pdf_file):
    """Lê as classes do arquivo PDF classificando um thumbnail pequeno."""
    try:
        thumb = thumbnail_bgr(img_bgr, max_side=128)
        mask = pcv.naive_bayes_classifier(rgb_img=thumb, pdf_file=pdf_file)
        return list(mask.keys())
    except Exception:
        return []


def bayes(img_bgr, pdf_file, disease: str):
    """Classifica imagem com Naive Bayes (PDF) e retorna percentuais, mask e denominador."""
    mask = pcv.naive_bayes_classifier(rgb_img=img_bgr, pdf_file=pdf_file)
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
    - Vagem / FLS: denom = healthy + lesion
    - TLS:         denom = healthy + clorosis + lesion
    - Outros:      denom = total de todos os pixels classificados
    """
    d = disease or ""
    h = counts.get("healthy", 0)
    l = counts.get("lesion", 0)
    c = counts.get("clorosis", 0)
    if d in ("Vagem", "FLS"):
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
