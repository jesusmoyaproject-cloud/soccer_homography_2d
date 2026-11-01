"""
Módulo Avanzado de Seguimiento y Reconstrucción del Balón (Pasada 1 + Pasada 2):

1. El "Tracker" en Tiempo de Inferencia (Selector por Verosimilitud Espacial con conf >= 0.15):
   - Evalúa todos los candidatos débiles (conf >= 0.15, p. ej. balones con motion blur en pases/tiros)
     combinando confianza del modelo + verosimilitud gaussiana respecto a la predicción cinemática
     + penalización de falsos positivos estáticos y fuera de cancha.
2. Estrategia de Inferencia ROI Dinámico + Full Frame (Zoom Inteligente):
   - Genera recortes de resolución nativa 1:1 (512x512 o 640x640) centrados en la posición predicha
     del balón para triplicar la resolución efectiva en balones lejanos, con fallback automático a Full Frame.
3. Máquina de Estados Finita (FSM) para Sobrevivir a las Oclusiones:
   - Estados: LOCKED, COASTING, OCCLUDED_BY_PLAYER, LOST.
   - Anclaje elástico a los pies del jugador poseedor cuando el balón queda oculto en un regate/disputa.
4. Reconstrucción Post-Inferencia (Pasada 2 — De Serie con Huecos a Traza Continua):
   - Poda bidireccional de picos espurios (Spike Removal).
   - Interpolación cinemática guiada por estado (Hermite/lineal en vuelo libre y guiada por poseedor en oclusión).
   - Suavizado preservador de rebotes/cambios de dirección y cálculo de velocidad física (m/s y km/h).
"""

import csv
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np


class BallState(str, Enum):
    """Estados de la Máquina de Estados Finita (FSM) del balón."""
    LOST = "LOST"
    LOCKED = "LOCKED"
    COASTING = "COASTING"
    OCCLUDED_BY_PLAYER = "OCCLUDED_BY_PLAYER"


class SimpleKalman2D:
    """Filtro de Kalman de 4 estados [x, y, vx, vy]^T con amortiguamiento aerodinámico."""

    def __init__(self, init_x: float, init_y: float, pos_var: float = 4.0, vel_var: float = 100.0, meas_var: float = 4.0):
        self.x = np.array([float(init_x), float(init_y), 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(np.float64)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.R = np.diag([meas_var, meas_var]).astype(np.float64)

    def predict(self, dt: float = 1.0, damping: float = 0.96, accel_std: float = 15.0) -> Tuple[float, float]:
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, damping, 0.0],
                [0.0, 0.0, 0.0, damping],
            ],
            dtype=np.float64,
        )
        q_p = 0.5 * accel_std * (dt**2)
        q_v = accel_std * dt
        Q = np.diag([q_p**2, q_p**2, q_v**2, q_v**2]).astype(np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0]), float(self.x[1])

    def update(self, mx: float, my: float, max_speed: Optional[float] = None) -> Tuple[float, float]:
        z = np.array([float(mx), float(my)], dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float64) - K @ self.H) @ self.P

        if max_speed is not None:
            spd = float(np.hypot(self.x[2], self.x[3]))
            if spd > max_speed:
                scale = max_speed / spd
                self.x[2] *= scale
                self.x[3] *= scale

        return float(self.x[0]), float(self.x[1])


class BallPhysicsTracker:
    """
    Tracker de Balón en Tiempo de Inferencia (Pasada 1):
    Combina ROI Dinámico (Zoom Inteligente), Selector por Verosimilitud Espacial (conf >= 0.15)
    y Máquina de Estados Finita (LOCKED, COASTING, OCCLUDED_BY_PLAYER, LOST).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg_root = config or {}
        pitch_cfg = cfg_root.get("pitch", {})
        self.largo_cancha = float(pitch_cfg.get("largo_cancha", 105.0))
        self.ancho_cancha = float(pitch_cfg.get("ancho_cancha", 68.0))

        ball_cfg = cfg_root.get("ball_tracking", {})
        self.conf_low = float(ball_cfg.get("conf_low_threshold", 0.15))
        self.conf_high = float(ball_cfg.get("conf_high_threshold", 0.40))
        self.enable_dynamic_roi = bool(ball_cfg.get("enable_dynamic_roi", True))
        self.roi_size_locked = int(ball_cfg.get("roi_size_locked_px", 512))
        self.roi_size_coasting = int(ball_cfg.get("roi_size_coasting_px", 640))

        # Parámetros del Selector por Verosimilitud Espacial
        self.w_conf = float(ball_cfg.get("likelihood_weight_conf", 0.45))
        self.w_spatial = float(ball_cfg.get("likelihood_weight_spatial", 0.55))
        self.base_gate_px = float(ball_cfg.get("base_gate_sigma_px", 65.0))
        self.gate_expansion_px = float(ball_cfg.get("gate_expansion_per_frame_px", 28.0))
        self.min_likelihood_score = float(ball_cfg.get("min_likelihood_score", 0.30))
        self.max_ball_speed_mps = float(ball_cfg.get("max_ball_speed_mps", 38.0))  # ~136 km/h
        self.max_ball_jump_px_per_frame = float(ball_cfg.get("max_jump_px_per_frame", 145.0))

        # Parámetros de la Máquina de Estados (FSM)
        self.max_coasting_frames = int(ball_cfg.get("max_coasting_frames", 15))
        self.max_occlusion_frames = int(ball_cfg.get("max_occlusion_frames", 28))
        self.player_occlusion_radius_m = float(ball_cfg.get("player_occlusion_radius_m", 1.65))
        self.player_occlusion_radius_px = float(ball_cfg.get("player_occlusion_radius_px", 58.0))

        self.state: BallState = BallState.LOST
        self.kf_px: Optional[SimpleKalman2D] = None
        self.kf_pitch: Optional[SimpleKalman2D] = None
        self.missing_frames: int = 0
        self.carrier_player_id: Optional[int] = None
        self.last_bbox: Optional[List[float]] = None
        self.last_conf: float = 0.0
        self.last_frame_idx: int = 0

        # Candidato tentativo para transición segura LOST -> LOCKED con baja confianza
        self._tentative_cand: Optional[Tuple[int, float, float, Optional[Tuple[float, float]]]] = None
        # Memoria corta de detecciones para detectar falsos positivos fijos (carteles/puntos pintados)
        self._static_memory: deque = deque(maxlen=40)
        self.tabular_records: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.state = BallState.LOST
        self.kf_px = None
        self.kf_pitch = None
        self.missing_frames = 0
        self.carrier_player_id = None
        self.last_bbox = None
        self.last_conf = 0.0
        self.last_frame_idx = 0
        self._tentative_cand = None
        self._static_memory.clear()
        self.tabular_records.clear()

    def get_dynamic_roi(self, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Calcula la ventana del ROI Dinámico (x1, y1, x2, y2) en resolución nativa 1:1
        cuando el balón está en estado LOCKED, COASTING temprano u OCCLUDED_BY_PLAYER.
        Retorna None cuando el estado es LOST (forzando búsqueda Full-Frame).
        """
        if not self.enable_dynamic_roi or self.state == BallState.LOST or self.kf_px is None:
            return None

        if self.state == BallState.COASTING and self.missing_frames > 6:
            return None
        if self.state == BallState.OCCLUDED_BY_PLAYER and self.missing_frames > 12:
            return None

        cx = float(self.kf_px.x[0] + self.kf_px.x[2])
        cy = float(self.kf_px.x[1] + self.kf_px.x[3])

        roi_size = self.roi_size_locked if self.state == BallState.LOCKED else self.roi_size_coasting
        half = roi_size // 2

        x1 = max(0, min(img_w - roi_size, int(round(cx)) - half))
        y1 = max(0, min(img_h - roi_size, int(round(cy)) - half))
        x2 = min(img_w, x1 + roi_size)
        y2 = min(img_h, y1 + roi_size)

        if (x2 - x1) < 128 or (y2 - y1) < 128:
            return None
        return x1, y1, x2, y2

    def _is_within_pitch_bounds(self, pitch_m: Optional[Tuple[float, float]], margin_m: float = 4.0) -> bool:
        if pitch_m is None:
            return True
        mx, my = pitch_m
        return (-margin_m <= mx <= self.largo_cancha + margin_m) and (
            -margin_m <= my <= self.ancho_cancha + margin_m
        )

    def _find_closest_player(
        self,
        target_px: Tuple[float, float],
        target_m: Optional[Tuple[float, float]],
        players: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], float, float]:
        """
        Busca el jugador cuyos pies están más cerca de la posición estimada del balón.
        Retorna: (player_dict, dist_m, dist_px)
        """
        best_player = None
        best_dist_m = 1e6
        best_dist_px = 1e6

        for p in players:
            if p.get("role") == "Referee":
                continue
            fx, fy = p["foot_px"]
            d_px = float(np.hypot(fx - target_px[0], fy - target_px[1]))
            pm = p.get("pitch_m")
            if target_m is not None and pm is not None:
                d_m = float(np.hypot(pm[0] - target_m[0], pm[1] - target_m[1]))
            else:
                d_m = d_px / 18.0

            if d_m < best_dist_m:
                best_dist_m = d_m
                best_dist_px = d_px
                best_player = p

        return best_player, best_dist_m, best_dist_px

    def select_best_candidate(
        self,
        candidates: List[Dict[str, Any]],
        pred_px: Optional[Tuple[float, float]],
        pred_m: Optional[Tuple[float, float]],
        dt_frames: int,
        fps: float,
        homography_valid: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Selector por Verosimilitud Espacial sobre candidatos con conf >= 0.15:
        Evalúa verosimilitud combinando confianza del detector + cercanía gaussiana
        a la predicción cinemática + validación biomecánica/física en metros.
        """
        if not candidates:
            return None

        dt_sec = max(1.0 / max(1.0, fps), dt_frames / max(1.0, fps))

        # Si estamos en estado LOST (sin predicción previa confiable)
        if self.state == BallState.LOST or pred_px is None:
            valid_lost_cands = []
            for c in candidates:
                if homography_valid and not self._is_within_pitch_bounds(c.get("pitch_m"), margin_m=3.0):
                    continue
                valid_lost_cands.append(c)

            if not valid_lost_cands:
                return None

            valid_lost_cands.sort(key=lambda item: float(item["conf"]), reverse=True)
            best = valid_lost_cands[0]

            # Si tiene confianza alta (>= conf_high = 0.40), enganchar directamente
            if float(best["conf"]) >= self.conf_high:
                self._tentative_cand = None
                return best

            # Si tiene confianza baja (0.15 <= conf < 0.40), exigir confirmación en 2 cuadros consecutivos
            cx, cy = best["center_px"]
            if self._tentative_cand is not None:
                prev_f, prev_x, prev_y, _ = self._tentative_cand
                if 1 <= (self.last_frame_idx + 1 - prev_f) <= 3:
                    jump = float(np.hypot(cx - prev_x, cy - prev_y))
                    if jump <= self.max_ball_jump_px_per_frame * 1.5:
                        self._tentative_cand = None
                        return best

            self._tentative_cand = (self.last_frame_idx + 1, cx, cy, best.get("pitch_m"))
            return None

        # Estado activo (LOCKED, COASTING, OCCLUDED_BY_PLAYER): Evaluar Verosimilitud Espacial
        sigma_px = self.base_gate_px + self.gate_expansion_px * float(min(dt_frames, 15))
        max_jump_px = self.max_ball_jump_px_per_frame * float(max(1, min(dt_frames, 10)))
        max_jump_m = self.max_ball_speed_mps * dt_sec + 3.0

        best_cand = None
        best_score = -1.0

        for c in candidates:
            if homography_valid and not self._is_within_pitch_bounds(c.get("pitch_m"), margin_m=4.5):
                continue

            cx, cy = c["center_px"]
            conf = float(c["conf"])
            dist_px = float(np.hypot(cx - pred_px[0], cy - pred_px[1]))

            # Compuerta cinemática máxima en píxeles
            if dist_px > max_jump_px and conf < 0.75:
                continue

            # Compuerta física en metros sobre la cancha 2D
            cm = c.get("pitch_m")
            if homography_valid and pred_m is not None and cm is not None:
                dist_m = float(np.hypot(cm[0] - pred_m[0], cm[1] - pred_m[1]))
                if dist_m > max_jump_m and conf < 0.80:
                    continue

            spatial_prob = float(np.exp(-(dist_px**2) / (2.0 * (sigma_px**2))))
            likelihood = self.w_conf * conf + self.w_spatial * spatial_prob

            if likelihood >= self.min_likelihood_score and likelihood > best_score:
                best_score = likelihood
                c_copy = dict(c)
                c_copy["likelihood_score"] = round(likelihood, 3)
                best_cand = c_copy

        return best_cand

    def update(
        self,
        frame_idx: int,
        fps: float,
        candidates: List[Dict[str, Any]],
        players: List[Dict[str, Any]],
        homography_H: Optional[np.ndarray],
        homography_valid: bool,
        shot_cut: bool = False,
        roi_used: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Actualiza la Máquina de Estados Finita (FSM) del balón para el fotograma actual
        y guarda el registro en `self.tabular_records` para la Pasada 2.
        """
        from src.homography import pixel_to_pitch, pitch_to_pixel

        if shot_cut:
            self.state = BallState.LOST
            self.kf_px = None
            self.kf_pitch = None
            self.missing_frames = 0
            self.carrier_player_id = None
            self._tentative_cand = None

        dt_frames = max(1, frame_idx - self.last_frame_idx) if self.last_frame_idx > 0 else 1
        dt_sec = dt_frames / max(1.0, fps)

        # Proyectar coordenadas 2D de todos los candidatos si hay homografía válida
        if homography_valid and homography_H is not None:
            for c in candidates:
                if c.get("pitch_m") is None:
                    c["pitch_m"] = pixel_to_pitch(c["center_px"][0], c["center_px"][1], homography_H)

        # Predicción a priori de Kalman
        pred_px: Optional[Tuple[float, float]] = None
        pred_m: Optional[Tuple[float, float]] = None
        if self.kf_px is not None:
            pred_px = self.kf_px.predict(dt=1.0, damping=0.94, accel_std=18.0)
        if self.kf_pitch is not None and homography_valid:
            pred_m = self.kf_pitch.predict(dt=dt_sec, damping=0.95, accel_std=8.0)

        # Si está ocluido por un jugador conocido, guiar la predicción hacia los pies de ese jugador
        if self.state == BallState.OCCLUDED_BY_PLAYER and self.carrier_player_id is not None:
            carrier = next((p for p in players if int(p["id"]) == self.carrier_player_id), None)
            if carrier is not None:
                pred_px = carrier["foot_px"]
                if carrier.get("pitch_m") is not None:
                    pred_m = carrier["pitch_m"]

        # Seleccionar mejor candidato por Verosimilitud Espacial
        selected = self.select_best_candidate(
            candidates=candidates,
            pred_px=pred_px,
            pred_m=pred_m,
            dt_frames=dt_frames,
            fps=fps,
            homography_valid=homography_valid,
        )

        self.last_frame_idx = frame_idx
        active_ball: Optional[Dict[str, Any]] = None

        # =====================================================================
        # CASO A: DETECCIÓN ACEPTADA -> ESTADO LOCKED
        # =====================================================================
        if selected is not None:
            cx, cy = selected["center_px"]
            if self.kf_px is None:
                self.kf_px = SimpleKalman2D(cx, cy, pos_var=4.0, vel_var=120.0, meas_var=3.0)
                filt_px = (cx, cy)
            else:
                filt_px = self.kf_px.update(cx, cy, max_speed=self.max_ball_jump_px_per_frame)

            pm = selected.get("pitch_m")
            if homography_valid and homography_H is not None:
                pm = pixel_to_pitch(filt_px[0], filt_px[1], homography_H)
                if pm is not None:
                    if self.kf_pitch is None:
                        self.kf_pitch = SimpleKalman2D(pm[0], pm[1], pos_var=0.5, vel_var=25.0, meas_var=0.25)
                    else:
                        pm = self.kf_pitch.update(pm[0], pm[1], max_speed=self.max_ball_speed_mps)

            speed_mps = (
                float(np.hypot(self.kf_pitch.x[2], self.kf_pitch.x[3]))
                if self.kf_pitch is not None
                else 0.0
            )

            self.state = BallState.LOCKED
            self.missing_frames = 0
            self.last_bbox = list(selected["bbox"])
            self.last_conf = float(selected["conf"])

            # Identificar si algún jugador tiene posesión cercana
            closest_p, d_m, d_px = self._find_closest_player(filt_px, pm, players)
            if closest_p is not None and (d_m <= self.player_occlusion_radius_m or d_px <= self.player_occlusion_radius_px):
                self.carrier_player_id = int(closest_p["id"])
            else:
                self.carrier_player_id = None

            active_ball = {
                "id": 1,
                "class_id": 0,
                "role": "Ball",
                "conf": round(self.last_conf, 3),
                "likelihood": selected.get("likelihood_score", round(self.last_conf, 3)),
                "bbox": self.last_bbox,
                "center_px": (round(filt_px[0], 2), round(filt_px[1], 2)),
                "pitch_m": (round(pm[0], 3), round(pm[1], 3)) if pm is not None else None,
                "speed_mps": round(speed_mps, 2),
                "state": self.state.value,
                "carrier_id": self.carrier_player_id,
                "roi_used": bool(roi_used),
                "is_interpolated": False,
            }

        # =====================================================================
        # CASO B: SIN DETECCIÓN -> TRANSICIÓN A OCCLUDED_BY_PLAYER, COASTING O LOST
        # =====================================================================
        else:
            self.missing_frames += 1

            if self.state != BallState.LOST and pred_px is not None:
                # Verificar si un jugador está ocluyendo/conduciendo el balón
                closest_p, d_m, d_px = self._find_closest_player(pred_px, pred_m, players)
                is_near_player = closest_p is not None and (
                    d_m <= self.player_occlusion_radius_m or d_px <= self.player_occlusion_radius_px
                )

                if is_near_player and self.missing_frames <= self.max_occlusion_frames:
                    self.state = BallState.OCCLUDED_BY_PLAYER
                    self.carrier_player_id = int(closest_p["id"])
                    foot_x, foot_y = closest_p["foot_px"]
                    # Anclar suavemente el Kalman a los pies del jugador que cubre el balón
                    if self.kf_px is not None:
                        pred_px = self.kf_px.update(foot_x, foot_y, max_speed=self.max_ball_jump_px_per_frame)
                    if homography_valid and closest_p.get("pitch_m") is not None and self.kf_pitch is not None:
                        cm = closest_p["pitch_m"]
                        pred_m = self.kf_pitch.update(cm[0], cm[1], max_speed=self.max_ball_speed_mps)

                elif self.missing_frames <= self.max_coasting_frames:
                    self.state = BallState.COASTING
                else:
                    self.state = BallState.LOST
                    self.kf_px = None
                    self.kf_pitch = None
                    self.carrier_player_id = None

                # Si estamos en OCCLUDED_BY_PLAYER o en los primeros cuadros de COASTING, emitir posición estimada
                if self.state in (BallState.OCCLUDED_BY_PLAYER, BallState.COASTING) and pred_px is not None:
                    bw, bh = 16.0, 16.0
                    if self.last_bbox is not None:
                        bw = max(12.0, self.last_bbox[2] - self.last_bbox[0])
                        bh = max(12.0, self.last_bbox[3] - self.last_bbox[1])
                    px, py = pred_px
                    est_bbox = [px - bw / 2.0, py - bh / 2.0, px + bw / 2.0, py + bh / 2.0]
                    speed_mps = (
                        float(np.hypot(self.kf_pitch.x[2], self.kf_pitch.x[3]))
                        if self.kf_pitch is not None
                        else 0.0
                    )
                    active_ball = {
                        "id": 1,
                        "class_id": 0,
                        "role": "Ball",
                        "conf": round(max(0.15, self.last_conf * (0.90 ** self.missing_frames)), 3),
                        "likelihood": 0.0,
                        "bbox": est_bbox,
                        "center_px": (round(px, 2), round(py, 2)),
                        "pitch_m": (round(pred_m[0], 3), round(pred_m[1], 3)) if pred_m is not None else None,
                        "speed_mps": round(speed_mps, 2),
                        "state": self.state.value,
                        "carrier_id": self.carrier_player_id,
                        "roi_used": bool(roi_used),
                        "is_interpolated": True,
                    }

        # Registrar en la tabla de telemetría para la Reconstrucción Post-Inferencia (Pasada 2)
        if active_ball is not None and active_ball["state"] == BallState.LOCKED.value:
            pm = active_ball["pitch_m"]
            bx1, by1, bx2, by2 = active_ball["bbox"]
            self.tabular_records.append(
                {
                    "frame": int(frame_idx),
                    "detected": True,
                    "state": active_ball["state"],
                    "conf": float(active_ball["conf"]),
                    "likelihood": float(active_ball.get("likelihood", active_ball["conf"])),
                    "center_px_x": float(active_ball["center_px"][0]),
                    "center_px_y": float(active_ball["center_px"][1]),
                    "bbox_w": round(float(bx2 - bx1), 2),
                    "bbox_h": round(float(by2 - by1), 2),
                    "pitch_x_m": float(pm[0]) if pm is not None else None,
                    "pitch_y_m": float(pm[1]) if pm is not None else None,
                    "speed_mps": float(active_ball["speed_mps"]),
                    "carrier_id": self.carrier_player_id,
                    "roi_used": bool(roi_used),
                    "homography_valid": bool(homography_valid),
                    "shot_cut": bool(shot_cut),
                    "is_interpolated": False,
                }
            )
        else:
            self.tabular_records.append(
                {
                    "frame": int(frame_idx),
                    "detected": False,
                    "state": self.state.value,
                    "conf": float(active_ball["conf"]) if active_ball else 0.0,
                    "likelihood": 0.0,
                    "center_px_x": float(active_ball["center_px"][0]) if active_ball else None,
                    "center_px_y": float(active_ball["center_px"][1]) if active_ball else None,
                    "bbox_w": 16.0,
                    "bbox_h": 16.0,
                    "pitch_x_m": float(active_ball["pitch_m"][0]) if (active_ball and active_ball.get("pitch_m")) else None,
                    "pitch_y_m": float(active_ball["pitch_m"][1]) if (active_ball and active_ball.get("pitch_m")) else None,
                    "speed_mps": float(active_ball["speed_mps"]) if active_ball else 0.0,
                    "carrier_id": self.carrier_player_id,
                    "roi_used": bool(roi_used),
                    "homography_valid": bool(homography_valid),
                    "shot_cut": bool(shot_cut),
                    "is_interpolated": True,
                }
            )

        return active_ball


class BallTrajectoryReconstructor:
    """
    Reconstrucción Post-Inferencia del Balón (Pasada 2):
    Transforma la serie temporal con huecos de la Pasada 1 en una traza 2D continua:
      1. Eliminación bidireccional de picos espurios (Spike Removal).
      2. Interpolación cinemática de huecos (guiada por jugador poseedor en oclusiones
         o interpolación lineal/Hermite en vuelos y pases hasta `max_interp_gap_frames`).
      3. Suavizado por tramos preservando cambios bruscos de dirección (rebotes y pases).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, fps: float = 25.0):
        cfg = (config or {}).get("ball_tracking", {}).get("reconstruction", {})
        self.fps = max(1.0, float(fps))
        self.max_ball_speed_mps = float(cfg.get("max_ball_speed_mps", 38.0))
        self.max_spike_px = float(cfg.get("max_spike_jump_px", 160.0))
        self.max_interp_gap_frames = int(cfg.get("max_interp_gap_frames", 35))
        self.enable_smoothing = bool(cfg.get("enable_smoothing", True))

    def _remove_kinematic_spikes(self, rows: List[Dict[str, Any]]) -> int:
        """
        Detecta detecciones aisladas (picos de 1 fotograma) cuya velocidad de entrada y salida
        viola la física del balón mientras que los fotogramas adyacentes son consistentes entre sí.
        """
        detected_indices = [
            i
            for i, r in enumerate(rows)
            if r.get("detected", False)
            and r.get("center_px_x") is not None
            and r.get("center_px_y") is not None
        ]
        if len(detected_indices) < 3:
            return 0

        spikes_removed = 0
        last_valid_idx = detected_indices[0]

        for k in range(1, len(detected_indices) - 1):
            i_prev = last_valid_idx
            i_curr = detected_indices[k]
            i_next = detected_indices[k + 1]

            r_prev, r_curr, r_next = rows[i_prev], rows[i_curr], rows[i_next]
            if (
                r_prev.get("center_px_x") is None
                or r_curr.get("center_px_x") is None
                or r_next.get("center_px_x") is None
            ):
                continue

            if r_curr.get("shot_cut") or r_next.get("shot_cut"):
                last_valid_idx = i_curr
                continue

            dt1 = max(1, int(r_curr["frame"]) - int(r_prev["frame"]))
            dt2 = max(1, int(r_next["frame"]) - int(r_curr["frame"]))

            if dt1 > 6 or dt2 > 6:
                last_valid_idx = i_curr
                continue

            d1_px = float(
                np.hypot(
                    float(r_curr["center_px_x"]) - float(r_prev["center_px_x"]),
                    float(r_curr["center_px_y"]) - float(r_prev["center_px_y"]),
                )
            )
            d2_px = float(
                np.hypot(
                    float(r_next["center_px_x"]) - float(r_curr["center_px_x"]),
                    float(r_next["center_px_y"]) - float(r_curr["center_px_y"]),
                )
            )
            d_bridge_px = float(
                np.hypot(
                    float(r_next["center_px_x"]) - float(r_prev["center_px_x"]),
                    float(r_next["center_px_y"]) - float(r_prev["center_px_y"]),
                )
            )

            is_spike = (
                d1_px > self.max_spike_px * dt1
                and d2_px > self.max_spike_px * dt2
                and d_bridge_px < 0.65 * (d1_px + d2_px)
            )

            # Validar también en metros si las 3 observaciones tienen coordenadas 2D
            if (
                not is_spike
                and r_prev.get("pitch_x_m") is not None
                and r_prev.get("pitch_y_m") is not None
                and r_curr.get("pitch_x_m") is not None
                and r_curr.get("pitch_y_m") is not None
                and r_next.get("pitch_x_m") is not None
                and r_next.get("pitch_y_m") is not None
            ):
                d1_m = float(
                    np.hypot(
                        float(r_curr["pitch_x_m"]) - float(r_prev["pitch_x_m"]),
                        float(r_curr["pitch_y_m"]) - float(r_prev["pitch_y_m"]),
                    )
                )
                d2_m = float(
                    np.hypot(
                        float(r_next["pitch_x_m"]) - float(r_curr["pitch_x_m"]),
                        float(r_next["pitch_y_m"]) - float(r_curr["pitch_y_m"]),
                    )
                )
                d_bridge_m = float(
                    np.hypot(
                        float(r_next["pitch_x_m"]) - float(r_prev["pitch_x_m"]),
                        float(r_next["pitch_y_m"]) - float(r_prev["pitch_y_m"]),
                    )
                )
                v1_mps = d1_m / (dt1 / self.fps)
                v2_mps = d2_m / (dt2 / self.fps)
                if v1_mps > self.max_ball_speed_mps and v2_mps > self.max_ball_speed_mps and d_bridge_m < 0.65 * (d1_m + d2_m):
                    is_spike = True

            if is_spike:
                r_curr["detected"] = False
                r_curr["state"] = "SPIKE_REMOVED"
                r_curr["center_px_x"] = None
                r_curr["center_px_y"] = None
                r_curr["pitch_x_m"] = None
                r_curr["pitch_y_m"] = None
                r_curr["is_interpolated"] = True
                spikes_removed += 1
            else:
                last_valid_idx = i_curr

        return spikes_removed

    def reconstruct(
        self,
        raw_ball_records: List[Dict[str, Any]],
        stitched_players_by_frame: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Ejecuta la reconstrucción post-inferencia completa sobre la serie del balón:
        1. Poda de picos espurios.
        2. Interpolación de huecos entre detecciones reales (guiada por jugador o lineal/Hermite).
        3. Suavizado ligero preservador de vértices y recálculo de velocidades.
        """
        if not raw_ball_records:
            return [], {"total_frames": 0, "detected_frames": 0, "interpolated_frames": 0, "spikes_removed": 0}

        rows = [dict(r) for r in raw_ball_records]
        rows.sort(key=lambda r: int(r["frame"]))

        # Paso 1: Eliminar picos cinemáticos imposibles
        spikes_removed = self._remove_kinematic_spikes(rows)

        # Paso 2: Interpolar huecos acotados (<= max_interp_gap_frames) sin corte de cámara
        detected_idxs = [
            i
            for i, r in enumerate(rows)
            if r.get("detected", False)
            and r.get("center_px_x") is not None
            and r.get("center_px_y") is not None
        ]
        interpolated_count = 0

        for k in range(len(detected_idxs) - 1):
            idx_a = detected_idxs[k]
            idx_b = detected_idxs[k + 1]
            gap_len = idx_b - idx_a - 1

            if gap_len <= 0 or gap_len > self.max_interp_gap_frames:
                continue

            # No interpolar a través de un corte de edición de cámara
            if any(rows[m].get("shot_cut", False) for m in range(idx_a + 1, idx_b + 1)):
                continue

            r_a = rows[idx_a]
            r_b = rows[idx_b]
            dt_sec = (int(r_b["frame"]) - int(r_a["frame"])) / self.fps

            # Validar que la distancia entre A y B sea físicamente alcanzable por un balón
            has_pitch = (
                r_a.get("pitch_x_m") is not None
                and r_a.get("pitch_y_m") is not None
                and r_b.get("pitch_x_m") is not None
                and r_b.get("pitch_y_m") is not None
            )
            if has_pitch:
                dist_m = float(
                    np.hypot(
                        float(r_b["pitch_x_m"]) - float(r_a["pitch_x_m"]),
                        float(r_b["pitch_y_m"]) - float(r_a["pitch_y_m"]),
                    )
                )
                if dist_m > (self.max_ball_speed_mps * dt_sec + 5.0):
                    continue

            for m in range(idx_a + 1, idx_b):
                alpha = float(m - idx_a) / float(idx_b - idx_a)
                row_m = rows[m]
                f_num = int(row_m["frame"])

                # Coordenadas interpoladas base (vuelo libre entre A y B)
                ix_px = (1.0 - alpha) * float(r_a["center_px_x"]) + alpha * float(r_b["center_px_x"])
                iy_px = (1.0 - alpha) * float(r_a["center_px_y"]) + alpha * float(r_b["center_px_y"])
                ix_m = (
                    (1.0 - alpha) * float(r_a["pitch_x_m"]) + alpha * float(r_b["pitch_x_m"])
                    if has_pitch
                    else None
                )
                iy_m = (
                    (1.0 - alpha) * float(r_a["pitch_y_m"]) + alpha * float(r_b["pitch_y_m"])
                    if has_pitch
                    else None
                )

                # Si durante el hueco el balón estaba ocluido por un jugador poseedor conocido,
                # combinar suavemente con la posición de los pies de ese jugador
                carrier_id = row_m.get("carrier_id")
                if carrier_id is not None and stitched_players_by_frame is not None:
                    frame_players = stitched_players_by_frame.get(f_num, [])
                    carrier = next(
                        (
                            p
                            for p in frame_players
                            if int(p.get("raw_track_id", p["id"])) == int(carrier_id)
                            or int(p["id"]) == int(carrier_id)
                        ),
                        None,
                    )
                    if carrier is not None:
                        cfx, cfy = carrier["foot_px"]
                        if float(np.hypot(cfx - ix_px, cfy - iy_px)) <= 85.0:
                            ix_px = 0.55 * ix_px + 0.45 * cfx
                            iy_px = 0.55 * iy_px + 0.45 * cfy
                            if ix_m is not None and carrier.get("pitch_m") is not None:
                                ix_m = 0.55 * ix_m + 0.45 * float(carrier["pitch_m"][0])
                                iy_m = 0.55 * iy_m + 0.45 * float(carrier["pitch_m"][1])

                row_m["center_px_x"] = round(float(ix_px), 2)
                row_m["center_px_y"] = round(float(iy_px), 2)
                row_m["bbox_w"] = round((1.0 - alpha) * float(r_a["bbox_w"]) + alpha * float(r_b["bbox_w"]), 2)
                row_m["bbox_h"] = round((1.0 - alpha) * float(r_a["bbox_h"]) + alpha * float(r_b["bbox_h"]), 2)
                row_m["pitch_x_m"] = round(float(ix_m), 3) if ix_m is not None else None
                row_m["pitch_y_m"] = round(float(iy_m), 3) if iy_m is not None else None
                row_m["conf"] = round((1.0 - alpha) * float(r_a["conf"]) + alpha * float(r_b["conf"]), 3)
                row_m["state"] = "RECONSTRUCTED_OCCLUSION" if carrier_id is not None else "RECONSTRUCTED_FLIGHT"
                row_m["is_interpolated"] = True
                interpolated_count += 1

        # Paso 3: Suavizado preservador de rebotes y cálculo de velocidad instantánea (m/s)
        if self.enable_smoothing:
            self._smooth_and_compute_velocity(rows)

        detected_final = sum(1 for r in rows if r.get("detected", False))
        stats = {
            "total_frames": len(rows),
            "detected_frames": detected_final,
            "interpolated_frames": interpolated_count,
            "active_ball_frames": detected_final + interpolated_count,
            "spikes_removed": spikes_removed,
        }
        return rows, stats

    def _smooth_and_compute_velocity(self, rows: List[Dict[str, Any]]) -> None:
        """
        Aplica un filtro de media móvil ponderada [0.2, 0.6, 0.2] sobre puntos interiores
        que no representen un rebote/cambio brusco de dirección (> 35 grados) y calcula `speed_mps`.
        """
        n = len(rows)
        for i in range(1, n - 1):
            r_prev, r_curr, r_next = rows[i - 1], rows[i], rows[i + 1]
            if r_curr.get("shot_cut") or r_next.get("shot_cut"):
                continue
            if (
                r_prev.get("pitch_x_m") is not None
                and r_curr.get("pitch_x_m") is not None
                and r_next.get("pitch_x_m") is not None
            ):
                p0 = np.array([float(r_prev["pitch_x_m"]), float(r_prev["pitch_y_m"])])
                p1 = np.array([float(r_curr["pitch_x_m"]), float(r_curr["pitch_y_m"])])
                p2 = np.array([float(r_next["pitch_x_m"]), float(r_next["pitch_y_m"])])
                v_in = p1 - p0
                v_out = p2 - p1
                n_in = float(np.linalg.norm(v_in))
                n_out = float(np.linalg.norm(v_out))

                # Solo suavizar si no es un vértice de pase/rebote
                if n_in > 0.05 and n_out > 0.05:
                    cos_ang = float(np.dot(v_in, v_out) / (n_in * n_out))
                    if cos_ang > 0.82:  # Ángulo < ~35°
                        smooth_p = 0.20 * p0 + 0.60 * p1 + 0.20 * p2
                        r_curr["pitch_x_m"] = round(float(smooth_p[0]), 3)
                        r_curr["pitch_y_m"] = round(float(smooth_p[1]), 3)

                dt = max(1, int(r_curr["frame"]) - int(r_prev["frame"])) / self.fps
                spd = float(np.linalg.norm(p1 - p0) / dt)
                r_curr["speed_mps"] = round(min(spd, self.max_ball_speed_mps), 2)

    @staticmethod
    def build_frame_lookup(
        reconstructed_rows: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Construye un diccionario `{frame_idx: [ball_dict]}` listo para `Tactical2DRenderer`."""
        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for r in reconstructed_rows:
            f_idx = int(r["frame"])
            cx = r.get("center_px_x")
            cy = r.get("center_px_y")
            if cx is None or cy is None:
                by_frame[f_idx] = []
                continue

            bw = float(r.get("bbox_w", 16.0))
            bh = float(r.get("bbox_h", 16.0))
            pm = (
                (float(r["pitch_x_m"]), float(r["pitch_y_m"]))
                if r.get("pitch_x_m") is not None and r.get("pitch_y_m") is not None
                else None
            )
            by_frame[f_idx] = [
                {
                    "id": 1,
                    "class_id": 0,
                    "role": "Ball",
                    "conf": float(r.get("conf", 0.4)),
                    "bbox": [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0],
                    "center_px": (float(cx), float(cy)),
                    "pitch_m": pm,
                    "speed_mps": float(r.get("speed_mps", 0.0)),
                    "state": str(r.get("state", "LOCKED")),
                    "carrier_id": r.get("carrier_id"),
                    "is_interpolated": bool(r.get("is_interpolated", False)),
                }
            ]
        return by_frame

    @staticmethod
    def export_to_csv(rows: List[Dict[str, Any]], csv_path: Union[str, Path]) -> Path:
        out_path = Path(csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            out_path.write_text("", encoding="utf-8")
            return out_path

        fieldnames = list(rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return out_path

