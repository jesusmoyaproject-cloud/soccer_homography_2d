"""
Módulo de Apariencia y Reconocimiento de Jugador (Pasada 1):
1. TeamClusterer: Clasificación de equipo en espacio CIE-LAB con máscara anti-césped (y soporte opcional CLIP),
   con bloqueo temporal por historial para evitar parpadeo de color de equipo.
2. JerseyNumberRecognizer: Lectura oportunista de dorsales con EasyOCR sobre recortes de torso pre-procesados
   con CLAHE y Gating de Calidad (tamaño mínimo, sin oclusión, intervalo de cuadros y bloqueo por consenso).
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _extract_non_grass_torso_lab(
    frame_bgr: np.ndarray,
    bbox: List[float],
) -> Optional[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """
    Recorta la región del torso de la camiseta, elimina píxeles de césped en HSV y devuelve:
    - Vector medio CIE-LAB ponderado [0.35*L, A, B] para clustering robusto a sombras.
    - Color mediano BGR real de la camiseta.
    """
    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 16:
        return None

    cx1 = max(0, int(round(x1 + 0.22 * bw)))
    cx2 = min(w_img, int(round(x2 - 0.22 * bw)))
    cy1 = max(0, int(round(y1 + 0.15 * bh)))
    cy2 = min(h_img, int(round(y1 + 0.50 * bh)))

    if cx2 <= cx1 + 2 or cy2 <= cy1 + 2:
        return None

    crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # Máscara amplia de césped verde y sombra de césped (Hue 24..96, Sat >= 22)
    grass_mask = cv2.inRange(hsv, np.array([24, 22, 20]), np.array([96, 255, 255]))
    valid_mask = grass_mask == 0

    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid < 6:
        # Si casi todo el recorte es césped (caja muy pequeña o descentrada), no contaminar el buffer
        return None
    valid_bgr = crop_bgr[valid_mask]

    med_bgr = np.median(valid_bgr, axis=0)
    bgr_tuple = (int(med_bgr[0]), int(med_bgr[1]), int(med_bgr[2]))

    lab_patch = cv2.cvtColor(valid_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    med_lab = np.median(lab_patch, axis=0).astype(np.float32)
    # Reducir peso del canal de luminancia L para invarianza a sombras en el estadio
    weighted_lab = np.array([med_lab[0] * 0.35, med_lab[1], med_lab[2]], dtype=np.float32)
    return weighted_lab, bgr_tuple


class TeamClusterer:
    """
    Clasificador de equipos en tiempo real basado en K-Means sobre CIE-LAB sin césped,
    con votación histórica por raw_track_id para garantizar estabilidad en cruces.
    Equipos:
      - 1: Equipo A
      - 2: Equipo B
      - 3: Arquero (GK)
      - 4: Árbitro (Referee)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("tracking_2d", {}).get("team_filter", {})
        self.min_samples_to_fit = int(cfg.get("min_samples_to_fit", 12))
        self.lock_votes = int(cfg.get("lock_votes", 8))

        self.centroids_lab: Optional[np.ndarray] = None  # Shape (2, 3) para Equipo A y Equipo B
        self.team_bgr_palette: Dict[int, Tuple[int, int, int]] = {
            1: (255, 190, 0),    # Equipo A (Celeste/Azul por defecto hasta calibrar)
            2: (50, 60, 245),    # Equipo B (Rojo por defecto hasta calibrar)
            3: (0, 230, 120),    # GK (Verde neón)
            4: (0, 215, 255),    # Referee (Amarillo)
        }
        self._buffer_lab: List[np.ndarray] = []
        self._buffer_bgr: List[Tuple[int, int, int]] = []
        self._track_votes: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._locked_team: Dict[int, int] = {}
        self._track_lab_sum: Dict[int, List[np.ndarray]] = defaultdict(list)

    def reset(self) -> None:
        self.centroids_lab = None
        self._buffer_lab.clear()
        self._buffer_bgr.clear()
        self._track_votes.clear()
        self._locked_team.clear()
        self._track_lab_sum.clear()

    def _fit_kmeans_2(self) -> None:
        """Ajusta 2 centroides de equipo (Equipo A y Equipo B) usando cv2.kmeans sobre el buffer CIE-LAB."""
        if len(self._buffer_lab) < 4:
            return
        data = np.array(self._buffer_lab, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
        _, labels, centers = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

        # Ordenar determinísticamente los 2 clusters por el canal A (rojo-verde) para que Team 1 y Team 2 no permuten
        order = np.argsort(centers[:, 1])
        self.centroids_lab = centers[order]

        labels_flat = labels.ravel()
        for team_idx, cluster_idx in enumerate(order, start=1):
            member_bgrs = [
                self._buffer_bgr[i] for i in range(len(labels_flat)) if labels_flat[i] == cluster_idx
            ]
            if member_bgrs:
                med = np.median(np.array(member_bgrs), axis=0)
                self.team_bgr_palette[team_idx] = (int(med[0]), int(med[1]), int(med[2]))

    def classify_players(self, frame_bgr: np.ndarray, players: List[Dict[str, Any]]) -> None:
        """
        Asigna in-place a cada jugador:
          - 'team_id': 1 (Equipo A), 2 (Equipo B), 3 (GK), 4 (Referee)
          - 'team_label': 'Team_A', 'Team_B', 'GK', 'Referee'
          - 'lab_feature': vector CIE-LAB [3]
          - 'color_bgr': color oficial representativo del equipo
        """
        # 1. Extraer características LAB de todos los jugadores de campo
        frame_features: List[Tuple[int, np.ndarray, Tuple[int, int, int]]] = []
        for idx, p in enumerate(players):
            role = p.get("role", "Player")
            if role == "Referee" or p.get("class_id") == 3:
                p["team_id"] = 4
                p["team_label"] = "Referee"
                p["color_bgr"] = self.team_bgr_palette[4]
                p["lab_feature"] = None
                continue
            if role == "GK" or p.get("class_id") == 1:
                p["team_id"] = 3
                p["team_label"] = "GK"
                p["color_bgr"] = self.team_bgr_palette[3]
                p["lab_feature"] = None
                continue

            tid = int(p["id"])
            # Si el jugador ya tiene su equipo bloqueado y el K-Means ya está calibrado,
            # reutilizar directamente sin volver a recortar ni convertir HSV/LAB en CPU.
            if tid in self._locked_team and self.centroids_lab is not None and len(self._buffer_lab) >= 40:
                team_id = self._locked_team[tid]
                p["team_id"] = team_id
                p["team_label"] = "Team_A" if team_id == 1 else "Team_B"
                p["color_bgr"] = self.team_bgr_palette.get(team_id, (255, 190, 0))
                cached_labs = self._track_lab_sum.get(tid)
                p["lab_feature"] = cached_labs[-1].tolist() if cached_labs else None
                continue

            res = _extract_non_grass_torso_lab(frame_bgr, p["bbox"])
            if res is not None:
                lab_vec, bgr_val = res
                p["lab_feature"] = lab_vec.tolist()
                frame_features.append((idx, lab_vec, bgr_val))
                if len(self._buffer_lab) < 240:
                    self._buffer_lab.append(lab_vec)
                    self._buffer_bgr.append(bgr_val)
            else:
                p["lab_feature"] = None

        # 2. Entrenar o actualizar K-Means si aún no está calibrado
        if self.centroids_lab is None and len(self._buffer_lab) >= max(4, min(self.min_samples_to_fit, len(frame_features))):
            self._fit_kmeans_2()
        elif self.centroids_lab is not None and len(self._buffer_lab) in (30, 60, 120):
            self._fit_kmeans_2()

        # 3. Asignar cluster de equipo con bloqueo por historial de track_id
        for idx, lab_vec, bgr_val in frame_features:
            p = players[idx]
            tid = int(p["id"])
            self._track_lab_sum[tid].append(lab_vec)

            if tid in self._locked_team:
                team_id = self._locked_team[tid]
            elif self.centroids_lab is not None:
                d0 = float(np.linalg.norm(lab_vec - self.centroids_lab[0]))
                d1 = float(np.linalg.norm(lab_vec - self.centroids_lab[1]))
                instant_team = 1 if d0 <= d1 else 2
                self._track_votes[tid][instant_team] += 1

                votes = self._track_votes[tid]
                best_team = 1 if votes[1] >= votes[2] else 2
                if votes[best_team] >= self.lock_votes and votes[best_team] >= 2 * votes[3 - best_team]:
                    self._locked_team[tid] = best_team
                team_id = best_team
            else:
                team_id = 1

            p["team_id"] = team_id
            p["team_label"] = "Team_A" if team_id == 1 else "Team_B"
            p["color_bgr"] = self.team_bgr_palette.get(team_id, bgr_val)

        # Fallback para jugadores cuyo recorte era muy pequeño
        for p in players:
            if "team_id" not in p:
                tid = int(p["id"])
                team_id = self._locked_team.get(tid, 1)
                p["team_id"] = team_id
                p["team_label"] = "Team_A" if team_id == 1 else "Team_B"
                p["color_bgr"] = self.team_bgr_palette.get(team_id, (255, 190, 0))


class JerseyNumberRecognizer:
    """
    Reconocedor oportunista de dorsales basado en EasyOCR con Gating de Calidad:
    - Solo evalúa bboxes con altura suficiente (>= min_bbox_height) y sin solapamiento severo.
    - Ejecuta OCR cada `ocr_interval_frames` por cada track_id y se detiene cuando el dorsal
      de ese track_id alcanza consenso fuerte (ahorrando >85% de cómputo de OCR).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, use_gpu: bool = True):
        cfg = (config or {}).get("tracking_2d", {}).get("jersey_ocr", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.min_bbox_height = int(cfg.get("min_bbox_height_px", 60))
        self.min_conf = float(cfg.get("min_ocr_conf", 0.45))
        self.interval_frames = max(1, int(cfg.get("ocr_interval_frames", 10)))
        self.lock_conf_sum = float(cfg.get("lock_confidence_sum", 1.8))
        self.max_crops_per_frame = int(cfg.get("max_crops_per_frame", 1))
        self.use_gpu = use_gpu

        self._reader = None
        self._reader_failed = False
        self._last_eval_frame: Dict[int, int] = {}
        self._jersey_scores: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._locked_jersey: Dict[int, str] = {}

    def reset(self) -> None:
        self._last_eval_frame.clear()
        self._jersey_scores.clear()
        self._locked_jersey.clear()

    def _get_reader(self):
        if not self.enabled or self._reader_failed:
            return None
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
            except Exception as e:
                logger.warning(f"[JerseyNumberRecognizer] EasyOCR no disponible ({e}). Continuando sin OCR.")
                self._reader_failed = True
                return None
        return self._reader

    @staticmethod
    def _preprocess_torso_crop(frame_bgr: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extrae el dorsal (espalda/pecho) y mejora el contraste local con CLAHE para lectura de dígitos."""
        h_img, w_img = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        cx1 = max(0, int(round(x1 + 0.15 * bw)))
        cx2 = min(w_img, int(round(x2 - 0.15 * bw)))
        cy1 = max(0, int(round(y1 + 0.14 * bh)))
        cy2 = min(h_img, int(round(y1 + 0.55 * bh)))

        if cx2 - cx1 < 12 or cy2 - cy1 < 16:
            return None

        crop = frame_bgr[cy1:cy2, cx1:cx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        return clahe.apply(upscaled)

    @staticmethod
    def _is_occluded(idx: int, players: List[Dict[str, Any]]) -> bool:
        """Verifica si la caja del jugador tiene solapamiento alto (IoU > 0.25) con otro jugador."""
        bx1, by1, bx2, by2 = players[idx]["bbox"]
        area_a = max(1.0, (bx2 - bx1) * (by2 - by1))
        for j, other in enumerate(players):
            if j == idx:
                continue
            ox1, oy1, ox2, oy2 = other["bbox"]
            ix1, iy1 = max(bx1, ox1), max(by1, oy1)
            ix2, iy2 = min(bx2, ox2), min(by2, oy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if (inter / area_a) > 0.25:
                    return True
        return False

    def recognize_frame_jerseys(
        self,
        frame_bgr: np.ndarray,
        players: List[Dict[str, Any]],
        frame_idx: int,
    ) -> None:
        """
        Evalúa oportunistamente el dorsal de los jugadores elegibles y anota en cada jugador:
          - 'jersey_ocr_raw': lectura instantánea en este frame (si se ejecutó OCR)
          - 'jersey_ocr_conf': confianza de la lectura instantánea
          - 'jersey_number': mejor dorsal acumulado hasta el momento para ese track_id
        """
        for p in players:
            tid = int(p["id"])
            p["jersey_ocr_raw"] = None
            p["jersey_ocr_conf"] = 0.0
            p["jersey_number"] = self._locked_jersey.get(tid, self._get_best_candidate(tid))

        if not self.enabled:
            return

        # Seleccionar candidatos óptimos en este frame (no bloqueados, altos, sin oclusión)
        candidates: List[Tuple[float, int]] = []
        for idx, p in enumerate(players):
            if p.get("role") == "Referee":
                continue
            tid = int(p["id"])
            if tid in self._locked_jersey:
                continue
            last_f = self._last_eval_frame.get(tid, -999)
            if (frame_idx - last_f) < self.interval_frames:
                continue

            x1, y1, x2, y2 = p["bbox"]
            bh = y2 - y1
            if bh < self.min_bbox_height:
                continue
            if self._is_occluded(idx, players):
                continue
            candidates.append((bh, idx))

        if not candidates:
            return

        reader = self._get_reader()
        if reader is None:
            return

        # Priorizar los recortes más cercanos/grandes hasta max_crops_per_frame
        candidates.sort(reverse=True)
        for _, idx in candidates[: self.max_crops_per_frame]:
            p = players[idx]
            tid = int(p["id"])
            self._last_eval_frame[tid] = frame_idx

            crop_img = self._preprocess_torso_crop(frame_bgr, p["bbox"])
            if crop_img is None:
                continue

            try:
                results = reader.readtext(crop_img, allowlist="0123456789", detail=1, paragraph=False)
            except Exception:
                continue

            for _, text, conf in results:
                clean_digits = re.sub(r"\D", "", str(text))
                score = float(conf)
                if not clean_digits or len(clean_digits) > 2 or clean_digits == "0" or clean_digits == "00":
                    continue
                if score < self.min_conf:
                    continue

                num_str = str(int(clean_digits))
                p["jersey_ocr_raw"] = num_str
                p["jersey_ocr_conf"] = round(score, 3)

                self._jersey_scores[tid][num_str] += score
                if self._jersey_scores[tid][num_str] >= self.lock_conf_sum:
                    self._locked_jersey[tid] = num_str
                p["jersey_number"] = self._locked_jersey.get(tid, self._get_best_candidate(tid))
                break

    def _get_best_candidate(self, tid: int) -> Optional[str]:
        scores = self._jersey_scores.get(tid)
        if not scores:
            return None
        best_num, best_val = max(scores.items(), key=lambda kv: kv[1])
        return best_num if best_val >= self.min_conf else None

