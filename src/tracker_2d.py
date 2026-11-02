"""
Módulo de Seguimiento en Coordenadas 2D de Cancha (Pasada 1):
Implementa BoT-SORT modificado con Filtro de Kalman en el plano métrico 2D (X_m, Y_m, Vx, Vy),
Gating por Equipo (CIE-LAB) y reconexión inmediata de pistas en coordenadas de cancha.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.spatial_geometry import apollonius_reach_cost


class KalmanTrack2D:
    """
    Filtro de Kalman lineal en coordenadas métricas de cancha:
      Estado x = [X_m, Y_m, Vx_mps, Vy_mps]^T
    Inmune al movimiento de paneo/zoom de la cámara de TV.
    """

    def __init__(
        self,
        track_id: int,
        init_pos_m: Tuple[float, float],
        frame_idx: int,
        team_id: int,
        role: str,
        bbox: List[float],
        lab_feature: Optional[List[float]] = None,
        jersey_number: Optional[str] = None,
    ):
        self.track_id = int(track_id)
        self.last_frame = int(frame_idx)
        self.age = 1
        self.hits = 1
        self.missing_frames = 0
        self.team_id = int(team_id)
        self.role = role
        self.last_bbox = list(bbox)
        self.lab_feature = np.array(lab_feature, dtype=np.float32) if lab_feature is not None else None
        self.jersey_number = jersey_number

        # Estado [x, y, vx, vy]
        self.x = np.array([float(init_pos_m[0]), float(init_pos_m[1]), 0.0, 0.0], dtype=np.float64)
        # Matriz de covarianza P
        self.P = np.diag([1.0, 1.0, 16.0, 16.0]).astype(np.float64)
        # Matriz de observación H_kf (2x4)
        self.H_kf = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        # Ruido de medición en metros (~0.6m std)
        self.R = np.diag([0.36, 0.36]).astype(np.float64)

    def predict(self, dt: float) -> Tuple[float, float]:
        """Predice la posición 2D en metros tras un intervalo dt (segundos)."""
        if dt <= 0.0:
            return float(self.x[0]), float(self.x[1])

        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.92, 0.0],  # Amortiguamiento ligero para evitar extrapolaciones desbocadas
                [0.0, 0.0, 0.0, 0.92],
            ],
            dtype=np.float64,
        )
        # Ruido de proceso escalado por aceleración humana (~4.5 m/s^2)
        sigma_a = 4.5
        q_pos = 0.5 * sigma_a * (dt**2)
        q_vel = sigma_a * dt
        Q = np.diag([q_pos**2, q_pos**2, q_vel**2, q_vel**2]).astype(np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0]), float(self.x[1])

    def update(
        self,
        meas_pos_m: Tuple[float, float],
        frame_idx: int,
        fps: float,
        bbox: List[float],
        team_id: Optional[int] = None,
        lab_feature: Optional[List[float]] = None,
        jersey_number: Optional[str] = None,
        max_speed_mps: float = 12.0,
    ) -> Tuple[float, float, float]:
        """Actualiza el filtro de Kalman con una nueva observación (X_m, Y_m) en la cancha."""
        dt = max(1.0 / max(1.0, fps), (frame_idx - self.last_frame) / max(1.0, fps))
        if frame_idx > self.last_frame:
            self.predict(dt)

        z = np.array([float(meas_pos_m[0]), float(meas_pos_m[1])], dtype=np.float64)
        y = z - (self.H_kf @ self.x)
        S = self.H_kf @ self.P @ self.H_kf.T + self.R
        K = self.P @ self.H_kf.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ self.H_kf) @ self.P

        # Acotar velocidad máxima físicamente posible (sprint humano <= 12.0 m/s)
        speed = float(np.hypot(self.x[2], self.x[3]))
        if speed > max_speed_mps:
            scale = max_speed_mps / speed
            self.x[2] *= scale
            self.x[3] *= scale
            speed = max_speed_mps

        self.last_frame = int(frame_idx)
        self.missing_frames = 0
        self.hits += 1
        self.age += 1
        self.last_bbox = list(bbox)

        if team_id is not None and self.team_id in (1, 2) and team_id in (1, 2):
            self.team_id = int(team_id)
        if lab_feature is not None:
            vec = np.array(lab_feature, dtype=np.float32)
            self.lab_feature = vec if self.lab_feature is None else (0.8 * self.lab_feature + 0.2 * vec)
        if jersey_number:
            self.jersey_number = jersey_number

        return float(self.x[0]), float(self.x[1]), speed


class PitchBoTSORT2D:
    """
    Gestor de seguimiento híbrido (Pasada 1) que combina los IDs de BoT-SORT con:
    1. Filtro de Kalman en coordenadas 2D de cancha (metros).
    2. Gating estricto por equipo (evita intercambios de ID entre rivales en duelos).
    3. Reconexión inmediata de pistas perdidas en el plano 2D (hasta `memory_frames`).
    4. Registro tabular cuadro a cuadro para el Cosido Global en la Pasada 2.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("tracking_2d", {})
        self.max_speed_mps = float(cfg.get("max_speed_mps", 12.0))
        self.reconnect_radius_m = float(cfg.get("reconnect_radius_m", 3.8))
        self.memory_frames = int(cfg.get("memory_frames", 45))

        self.tracks_2d: Dict[int, KalmanTrack2D] = {}
        self.raw_to_canonical_id: Dict[int, int] = {}
        self.tabular_records: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.tracks_2d.clear()
        self.raw_to_canonical_id.clear()
        self.tabular_records.clear()

    def update_frame(
        self,
        frame_idx: int,
        fps: float,
        players: List[Dict[str, Any]],
        homography_valid: bool,
        shot_cut: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Actualiza el estado 2D de todos los jugadores detectados en el fotograma actual
        y almacena las filas en el buffer tabular de la Pasada 1.
        """
        if shot_cut:
            # En un corte de cámara reiniciamos la asociación inmediata online;
            # la Pasada 2 (GlobalTrackletStitcher) se encargará de coser a través del corte.
            self.raw_to_canonical_id.clear()
            for trk in self.tracks_2d.values():
                trk.missing_frames = self.memory_frames + 1

        active_canonical_ids_in_frame = set()
        unmatched_players: List[Dict[str, Any]] = []

        # 1. Intentar asociar jugadores cuyo raw_track_id ya tiene una pista 2D activa y coherente
        for p in players:
            raw_id = int(p["id"])
            can_id = self.raw_to_canonical_id.get(raw_id, raw_id)
            pm = p.get("pitch_m")
            team_id = int(p.get("team_id", 1))

            if can_id in self.tracks_2d and homography_valid and pm is not None:
                trk = self.tracks_2d[can_id]
                dt = max(1.0 / max(1.0, fps), (frame_idx - trk.last_frame) / max(1.0, fps))
                pred_x, pred_y = float(trk.x[0] + trk.x[2] * dt), float(trk.x[1] + trk.x[3] * dt)
                dist_m = float(np.hypot(pm[0] - pred_x, pm[1] - pred_y))
                max_allowed_jump = self.max_speed_mps * dt + 2.5

                # Validar que BoT-SORT no haya hecho un ID switch hacia un rival o un salto imposible
                team_conflict = (
                    trk.hits >= 6
                    and trk.team_id in (1, 2)
                    and team_id in (1, 2)
                    and trk.team_id != team_id
                    and dist_m > 1.8
                )
                if dist_m <= max_allowed_jump and not team_conflict:
                    kx, ky, spd = trk.update(
                        pm,
                        frame_idx=frame_idx,
                        fps=fps,
                        bbox=p["bbox"],
                        team_id=team_id,
                        lab_feature=p.get("lab_feature"),
                        jersey_number=p.get("jersey_number"),
                        max_speed_mps=self.max_speed_mps,
                    )
                    p["raw_track_id"] = raw_id
                    p["id"] = can_id
                    p["pitch_m"] = (kx, ky)
                    p["speed_mps"] = round(spd, 2)
                    p["vx_mps"] = round(float(trk.x[2]), 3)
                    p["vy_mps"] = round(float(trk.x[3]), 3)
                    if trk.jersey_number and not p.get("jersey_number"):
                        p["jersey_number"] = trk.jersey_number
                    active_canonical_ids_in_frame.add(can_id)
                    continue

            unmatched_players.append(p)

        # 2. Reconectar en coordenadas 2D de cancha los jugadores con nuevo raw_id frente a pistas recién perdidas
        candidate_tracks = [
            trk
            for cid, trk in self.tracks_2d.items()
            if cid not in active_canonical_ids_in_frame
            and 1 <= (frame_idx - trk.last_frame) <= self.memory_frames
        ]

        if unmatched_players and candidate_tracks and homography_valid:
            n_det = len(unmatched_players)
            n_trk = len(candidate_tracks)
            cost_mat = np.full((n_det, n_trk), 1e6, dtype=np.float64)

            for i, p in enumerate(unmatched_players):
                pm = p.get("pitch_m")
                if pm is None:
                    continue
                p_team = int(p.get("team_id", 1))
                p_jersey = p.get("jersey_number")
                p_lab = np.array(p["lab_feature"], dtype=np.float32) if p.get("lab_feature") is not None else None

                for j, trk in enumerate(candidate_tracks):
                    # Gating por equipo
                    if p_team in (1, 2) and trk.team_id in (1, 2) and p_team != trk.team_id:
                        continue
                    if (p_team in (3, 4) or trk.team_id in (3, 4)) and p_team != trk.team_id:
                        continue
                    # Gating por dorsal si ambos tienen dorsal distinto
                    if p_jersey and trk.jersey_number and p_jersey != trk.jersey_number:
                        continue

                    dt = max(1.0 / max(1.0, fps), (frame_idx - trk.last_frame) / max(1.0, fps))
                    pred_x = float(trk.x[0] + trk.x[2] * min(dt, 1.5))
                    pred_y = float(trk.x[1] + trk.x[3] * min(dt, 1.5))
                    dist_m = float(np.hypot(pm[0] - pred_x, pm[1] - pred_y))

                    max_gate_m = min(self.reconnect_radius_m + 3.0 * dt, self.max_speed_mps * dt + 2.0)
                    if dist_m > max_gate_m:
                        continue

                    # Costo anisotrópico de alcance de Apolonio
                    apol_dist = apollonius_reach_cost(
                        (float(trk.x[0]), float(trk.x[1])),
                        (float(trk.x[2]), float(trk.x[3])),
                        pm,
                        dt_sec=dt,
                        max_speed_mps=self.max_speed_mps,
                    )

                    color_cost = 0.0
                    if p_lab is not None and trk.lab_feature is not None:
                        color_cost = float(np.linalg.norm(p_lab - trk.lab_feature)) / 50.0

                    jersey_bonus = -1.5 if (p_jersey and trk.jersey_number and p_jersey == trk.jersey_number) else 0.0
                    cost_mat[i, j] = apol_dist + 0.5 * color_cost + jersey_bonus

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            matched_det_indices = set()
            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < 1e5:
                    p = unmatched_players[r]
                    trk = candidate_tracks[c]
                    raw_id = int(p["id"])
                    can_id = trk.track_id
                    self.raw_to_canonical_id[raw_id] = can_id

                    kx, ky, spd = trk.update(
                        p["pitch_m"],
                        frame_idx=frame_idx,
                        fps=fps,
                        bbox=p["bbox"],
                        team_id=int(p.get("team_id", 1)),
                        lab_feature=p.get("lab_feature"),
                        jersey_number=p.get("jersey_number"),
                        max_speed_mps=self.max_speed_mps,
                    )
                    p["raw_track_id"] = raw_id
                    p["id"] = can_id
                    p["pitch_m"] = (kx, ky)
                    p["speed_mps"] = round(spd, 2)
                    p["vx_mps"] = round(float(trk.x[2]), 3)
                    p["vy_mps"] = round(float(trk.x[3]), 3)
                    if trk.jersey_number and not p.get("jersey_number"):
                        p["jersey_number"] = trk.jersey_number
                    active_canonical_ids_in_frame.add(can_id)
                    matched_det_indices.add(r)

            unmatched_players = [
                p for idx, p in enumerate(unmatched_players) if idx not in matched_det_indices
            ]

        # 3. Inicializar nuevas pistas 2D para los jugadores no emparejados
        for p in unmatched_players:
            raw_id = int(p["id"])
            can_id = raw_id
            if can_id in active_canonical_ids_in_frame:
                can_id = max(list(self.tracks_2d.keys()) + [raw_id]) + 1
            self.raw_to_canonical_id[raw_id] = can_id
            p["raw_track_id"] = raw_id
            p["id"] = can_id
            p["speed_mps"] = 0.0
            p["vx_mps"] = 0.0
            p["vy_mps"] = 0.0

            pm = p.get("pitch_m")
            if homography_valid and pm is not None:
                self.tracks_2d[can_id] = KalmanTrack2D(
                    track_id=can_id,
                    init_pos_m=pm,
                    frame_idx=frame_idx,
                    team_id=int(p.get("team_id", 1)),
                    role=p.get("role", "Player"),
                    bbox=p["bbox"],
                    lab_feature=p.get("lab_feature"),
                    jersey_number=p.get("jersey_number"),
                )
            active_canonical_ids_in_frame.add(can_id)

        # Incrementar contador de ausencia para pistas no vistas en este frame
        for cid, trk in self.tracks_2d.items():
            if cid not in active_canonical_ids_in_frame:
                trk.missing_frames += 1

        # 4. Registrar fila tabular por cada jugador para la Pasada 2 (Global Tracklet Stitching)
        for p in players:
            pm = p.get("pitch_m")
            x1, y1, x2, y2 = p["bbox"]
            lab = p.get("lab_feature")
            bgr = p.get("color_bgr", (255, 190, 0))
            self.tabular_records.append(
                {
                    "frame": int(frame_idx),
                    "raw_track_id": int(p.get("raw_track_id", p["id"])),
                    "online_track_id": int(p["id"]),
                    "role": str(p.get("role", "Player")),
                    "class_id": int(p.get("class_id", 2)),
                    "conf": round(float(p.get("conf", 0.5)), 4),
                    "bbox_x1": round(float(x1), 2),
                    "bbox_y1": round(float(y1), 2),
                    "bbox_x2": round(float(x2), 2),
                    "bbox_y2": round(float(y2), 2),
                    "foot_px_x": round(float(p["foot_px"][0]), 2),
                    "foot_px_y": round(float(p["foot_px"][1]), 2),
                    "pitch_x_m": round(float(pm[0]), 3) if pm is not None else None,
                    "pitch_y_m": round(float(pm[1]), 3) if pm is not None else None,
                    "speed_mps": round(float(p.get("speed_mps", 0.0)), 2),
                    "vx_mps": round(float(p.get("vx_mps", 0.0)), 3),
                    "vy_mps": round(float(p.get("vy_mps", 0.0)), 3),
                    "team_id": int(p.get("team_id", 1)),
                    "team_label": str(p.get("team_label", "Team_A")),
                    "color_b": int(bgr[0]),
                    "color_g": int(bgr[1]),
                    "color_r": int(bgr[2]),
                    "lab_l": round(float(lab[0]), 2) if lab is not None else None,
                    "lab_a": round(float(lab[1]), 2) if lab is not None else None,
                    "lab_b": round(float(lab[2]), 2) if lab is not None else None,
                    "jersey_ocr_raw": p.get("jersey_ocr_raw"),
                    "jersey_ocr_conf": round(float(p.get("jersey_ocr_conf", 0.0)), 3),
                    "jersey_online": p.get("jersey_number"),
                    "homography_valid": bool(homography_valid),
                    "shot_cut": bool(shot_cut),
                }
            )

        return players

