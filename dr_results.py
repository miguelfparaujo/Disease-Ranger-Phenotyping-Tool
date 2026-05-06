"""
dr_results.py — Disease Ranger: lógica de resultado por doença.

Centraliza:
- Detecção e métricas de vagens (Vagem / PGR)
- Métricas Stink Bug
- Desenho de caixas/numeração nas colorizadas
- Construção de linhas para o DataFrame de resultados
- Salvar imagem colorizada + retornar linha de resultado

Sem duplicação: o mesmo código serve para modo PDF, Interativo e Fallback.
"""
import cv2
import numpy as np
import os
from pathlib import Path

from dr_utils import calculate_area, safe_save_colorized_bgr
from dr_classifier import counts_from_mask, percentages_for_disease


# =====================================================
# VAGEM — detecção de componentes
# =====================================================
def _pca_aspect_ratio(comp_bool):
    ys, xs = np.where(comp_bool)
    if xs.size < 5:
        return 1.0
    pts = np.stack([xs, ys], axis=0).astype(np.float64)
    pts -= pts.mean(axis=1, keepdims=True)
    cov = np.cov(pts)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    if eigvals[1] <= 1e-8:
        return 999.0
    return float(np.sqrt(eigvals[0] / eigvals[1]))


def _split_by_watershed_if_needed(comp_bool_full, bbox, area, q75, ar, seed_thresh=0.26):
    likely_multi = (area > 1.30 * q75) and (ar < 1.25)
    if not likely_multi:
        return [comp_bool_full]

    x0, y0, x1, y1 = bbox
    roi = comp_bool_full[y0:y1+1, x0:x1+1].astype(np.uint8) * 255
    if roi.sum() == 0:
        return [comp_bool_full]

    roi_dil = cv2.dilate(roi, np.ones((3, 3), np.uint8), iterations=1)
    dist = cv2.distanceTransform(roi_dil, cv2.DIST_L2, 5)
    maxd = dist.max()
    if maxd < 1e-3:
        return [comp_bool_full]

    seeds = (dist > (seed_thresh * maxd)).astype(np.uint8)
    num_seeds, seed_labels = cv2.connectedComponents(seeds)
    if num_seeds <= 2:
        return [comp_bool_full]

    markers = seed_labels.astype(np.int32)
    markers[roi_dil == 0] = 0
    ws_img = cv2.cvtColor(roi_dil, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_img, markers)

    subcomponents = []
    for lbl in range(2, markers.max() + 1):
        submask_roi = (markers == lbl) & (roi_dil > 0)
        if np.count_nonzero(submask_roi) == 0:
            continue
        sub_full = np.zeros_like(comp_bool_full, dtype=bool)
        sub_full[y0:y1+1, x0:x1+1] = submask_roi
        subcomponents.append(sub_full)

    return subcomponents if subcomponents else [comp_bool_full]


def split_vagens_components(mask, min_area=None, pad=6, max_pods=10,
                             seed_thresh=0.26, min_fill=0.10, min_aspect=1.15):
    healthy = (mask['healthy'] > 0).astype(np.uint8)
    lesion  = (mask['lesion']  > 0).astype(np.uint8)
    pod_mask = cv2.bitwise_or(healthy, lesion)

    h, w = pod_mask.shape
    k_open  = max(3, int(round(min(h, w) * 0.002)))
    k_close = max(5, int(round(min(h, w) * 0.003)))
    pod_mask = cv2.morphologyEx(pod_mask, cv2.MORPH_OPEN,  np.ones((k_open,  k_open),  np.uint8))
    pod_mask = cv2.morphologyEx(pod_mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    pod_mask = cv2.GaussianBlur(pod_mask, (3, 3), 0)

    num_labels_all, _, stats_all, _ = cv2.connectedComponentsWithStats(pod_mask, connectivity=8)
    areas_all = [int(stats_all[lbl, cv2.CC_STAT_AREA]) for lbl in range(1, num_labels_all)]
    q75 = np.percentile(areas_all, 75) if areas_all else 0
    largest_area = int(max(areas_all)) if areas_all else 0

    if areas_all and q75 > 0:
        min_area_adapt = max(3000, int(0.30 * q75))
    else:
        min_area_adapt = max(3000, int(0.0008 * h * w))
    if min_area is None:
        min_area = min_area_adapt

    min_bbox_h = max(40, int(0.04 * h))
    min_bbox_w = max(25, int(0.015 * w))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pod_mask, connectivity=8)

    comps_raw = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        w_box = int(stats[label, cv2.CC_STAT_WIDTH])
        h_box = int(stats[label, cv2.CC_STAT_HEIGHT])
        if h_box < min_bbox_h or w_box < min_bbox_w:
            continue

        x1 = min(w - 1, x0 + w_box - 1 + pad)
        y1 = min(h - 1, y0 + h_box - 1 + pad)
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        bbox = (x0, y0, x1, y1)
        comp_bool = (labels == label)
        bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1)
        fill_ratio = area / max(1, bbox_area)
        touches_border = (x0 == 0) or (y0 == 0) or (x1 == w - 1) or (y1 == h - 1)

        if not touches_border and fill_ratio < min_fill:
            continue
        ar = _pca_aspect_ratio(comp_bool)
        if not touches_border and ar < min_aspect:
            continue
        comps_raw.append({'mask_bool': comp_bool, 'bbox': bbox, 'area_px': area, 'ar': ar})

    components = []
    for comp in comps_raw:
        sub_list = _split_by_watershed_if_needed(comp['mask_bool'], comp['bbox'],
                                                  comp['area_px'], q75, comp['ar'],
                                                  seed_thresh=seed_thresh)
        for sub in sub_list:
            ys, xs = np.where(sub)
            if xs.size == 0:
                continue
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(w - 1, x1 + pad); y1 = min(h - 1, y1 + pad)
            h_box = (y1 - y0 + 1); w_box = (x1 - x0 + 1)
            if h_box < min_bbox_h or w_box < min_bbox_w or h_box < 3 or w_box < 3:
                continue
            area_sub = int(np.count_nonzero(sub))
            min_area_sub = max(min_area, int(0.25 * largest_area))
            if area_sub < min_area_sub:
                continue
            bbox_area = h_box * w_box
            fill_ratio = area_sub / max(1, bbox_area)
            touches_border = (x0 == 0) or (y0 == 0) or (x1 == w - 1) or (y1 == h - 1)
            if not touches_border and fill_ratio < min_fill:
                continue
            ar_sub = _pca_aspect_ratio(sub)
            if not touches_border and ar_sub < min_aspect:
                continue
            components.append({'mask_bool': sub, 'bbox': (x0, y0, x1, y1), 'area_px': area_sub})

    if components:
        components = sorted(components, key=lambda c: c['area_px'], reverse=True)[:max_pods]
    return components


def sort_components_left_to_right(components):
    def centroid_xy(comp):
        x0, y0, x1, y1 = comp['bbox']
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return sorted(components, key=centroid_xy)


def compute_vagem_percentages(mask, comp_bool):
    healthy_bool = (mask['healthy'] > 0)
    lesion_bool  = (mask['lesion']  > 0)
    healthy_i = np.count_nonzero(healthy_bool & comp_bool)
    lesion_i  = np.count_nonzero(lesion_bool  & comp_bool)
    total_i   = healthy_i + lesion_i
    if total_i > 0:
        hpct = (healthy_i / total_i) * 100
        lpct = (lesion_i  / total_i) * 100
    else:
        hpct = lpct = 0.0
    return healthy_i, lesion_i, total_i, hpct, lpct


# =====================================================
# PROCESSAMENTO UNIFICADO POR IMAGEM
# =====================================================
def draw_vagem_boxes(colorized_bgr, components_sorted):
    """Desenha caixas e numeração nas vagens (in-place)."""
    for j, comp in enumerate(components_sorted, start=1):
        x0, y0, x1, y1 = comp['bbox']
        if (x1 - x0) >= 3 and (y1 - y0) >= 3:
            cv2.rectangle(colorized_bgr, (x0, y0), (x1, y1), (0, 0, 255), thickness=1)
            cv2.putText(colorized_bgr, f"{j:02d}", (x0+3, y0+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def process_image_result(img_rgb_cropped, mask, colorized_bgr, disease, filepath,
                          scale, colorized_dir):
    """
    Processa resultado de UMA imagem classificada. Salva colorized e retorna lista de rows.

    Retorna: list[dict] — linhas para o DataFrame (1 linha para maioria, N para Vagem).
    """
    filepath = Path(filepath)
    base = filepath.stem
    name = filepath.stem
    ext  = filepath.suffix

    colorized_bgr = np.ascontiguousarray(colorized_bgr.astype(np.uint8))
    results_rows = []

    if disease == "Vagem":
        components = split_vagens_components(
            mask, min_area=None, pad=6, max_pods=10,
            seed_thresh=0.26, min_fill=0.10, min_aspect=1.15
        )
        components_sorted = sort_components_left_to_right(components)
        draw_vagem_boxes(colorized_bgr, components_sorted)

        out_path_img = Path(colorized_dir) / f"{base}_colorized.png"
        safe_save_colorized_bgr(colorized_bgr, out_path_img)

        if not components_sorted:
            results_rows.append({
                "arquivo": f"{name}_vagem01{ext}", "vagem_id": 1,
                "healthy (%)": 0.0, "lesion (%)": 0.0, "bbox": ""
            })
        else:
            for j, comp in enumerate(components_sorted, start=1):
                _, _, _, hpct, lpct = compute_vagem_percentages(mask, comp['mask_bool'])
                x0, y0, x1, y1 = comp['bbox']
                results_rows.append({
                    "arquivo": f"{name}_vagem{j:02d}{ext}",
                    "vagem_id": j,
                    "healthy (%)": round(hpct, 2),
                    "lesion (%)": round(lpct, 2),
                    "bbox": f"{x0},{y0},{x1},{y1}"
                })

    elif disease == "Stink Bug":
        h2, w2 = img_rgb_cropped.shape[:2]
        total_px = h2 * w2
        wbg       = int(np.count_nonzero(mask.get('wbackground', 0)))
        bbg       = int(np.count_nonzero(mask.get('bbackground', 0)))
        hl_h      = int(np.count_nonzero(mask.get('healthy', 0)))
        hl_l      = int(np.count_nonzero(mask.get('lesion', 0)))
        ref       = int(np.count_nonzero(mask.get('referencesquare', 0)))
        total_hl  = hl_h + hl_l
        _z = np.zeros((h2, w2), np.uint8)

        wbg_pct   = round(wbg / total_px * 100, 2) if total_px else 0.0
        bbg_pct   = round(bbg / total_px * 100, 2) if total_px else 0.0
        h_pct     = round(hl_h / total_hl * 100, 2) if total_hl else 0.0
        l_pct     = round(hl_l / total_hl * 100, 2) if total_hl else 0.0
        ref_pct   = round(ref  / total_px * 100, 2) if total_px else 0.0

        areas = {}
        for key in ('wbackground', 'bbackground', 'healthy', 'lesion', 'referencesquare'):
            a = calculate_area(mask.get(key, _z), scale)
            areas[f"{key}_area (cm²)"] = round(a, 2) if scale else 'N/A'

        out_path_img = Path(colorized_dir) / f"{base}_colorized.png"
        safe_save_colorized_bgr(colorized_bgr, out_path_img)

        row = {
            "arquivo": filepath.name,
            "wbackground (%)": wbg_pct, "bbackground (%)": bbg_pct,
            "healthy (%)": h_pct, "lesion (%)": l_pct,
            "referencesquare (%)": ref_pct,
            "healthy+lesion (%)": round(h_pct + l_pct, 2),
            **areas
        }
        results_rows.append(row)

    else:
        counts = counts_from_mask(mask)
        percentages, _ = percentages_for_disease(disease, counts)
        areas = {f"{c} area (cm²)": round(calculate_area(mask[c], scale), 2) for c in mask}

        out_path_img = Path(colorized_dir) / f"{base}_colorized.png"
        safe_save_colorized_bgr(colorized_bgr, out_path_img)

        row = {"arquivo": filepath.name}
        row.update(percentages)
        row.update(areas)
        results_rows.append(row)

    return results_rows, colorized_bgr
