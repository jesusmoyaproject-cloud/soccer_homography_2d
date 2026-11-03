"""
Módulo de Sinceramiento Homográfico (Pasada 2) y Exportación de Contratos JSON
para `scoutingData_FrontEnd` (`match_metadata.json`, `tracking_output.json`,
`identity_manifest.json` y `consolidation_report.json`).

Implementa:
1. `reconcile_homographies_pass2`:
   - Interpola y sincera matrices H y H^-1 en breves caídas de detección de keypoints
     dentro de un mismo plano de cámara (respetando cortes de edición y tomas cerradas).
   - Retro-proyecta (pixel -> metros) registros de jugadores y balón en los cuadros
     donde la homografía fue recuperada en la Pasada 2.
2. `ScoutingJsonExporter`:
   - Homologa IDs y colores de equipos (`Team 1 -> 0 / team_a`, `Team 2 -> 1 / team_b`,
     `Referee -> -1 / referee`) con paleta BGR, RGB y HEX.
   - Reproyecta vía H^-1 (`pitch_to_pixel`) las coordenadas de jugadores y balón
     interpolados en 2D para garantizar continuidad visual tanto en el radar 2D como
     sobre el reproductor de video del Frontend.
   - Exporta los 4 archivos JSON compatibles con `scoutingData_FrontEnd`.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from src.homography import (
    HomographyFrameResult,
    KeypointObservation,
    pixel_to_pitch,
    pitch_to_pixel,
)
from src.pitch_config import (
    LARGO_CANCHA,
    ANCHO_CANCHA,
    PITCH_COORDS_METERS,
)
from src.spatial_geometry import compute_voronoi_and_apollonius_surface


def _bgr_to_rgb_and_hex(bgr: Tuple[int, int, int]) -> Tuple[List[int], List[int], str]:
    """Convierte una tupla (B, G, R) a listas BGR, RGB y string hexadecimal #RRGGBB."""
    b = int(np.clip(bgr[0], 0, 255))
    g = int(np.clip(bgr[1], 0, 255))
    r = int(np.clip(bgr[2], 0, 255))
    hex_str = f"#{r:02X}{g:02X}{b:02X}"
    return [b, g, r], [r, g, b], hex_str


def _matrix_to_nested_list(mat: Optional[np.ndarray], decimals: int = 6) -> Optional[List[List[float]]]:
    """Serializa una matriz numpy 3x3 a lista anidada de floats redondeados."""
    if mat is None:
        return None
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        return None
    return [[round(float(arr[r, c]), decimals) for c in range(3)] for r in range(3)]


def reconcile_homographies_pass2(
    frame_cache_meta: Dict[int, Dict[str, Any]],
    raw_player_records: Optional[List[Dict[str, Any]]] = None,
    raw_ball_records: Optional[List[Dict[str, Any]]] = None,
    max_gap_frames: int = 20,
    pitch_length: float = LARGO_CANCHA,
    pitch_width: float = ANCHO_CANCHA,
) -> Dict[str, int]:
    """
    Sincera la secuencia de homografías en la Pasada 2:
    1. Identifica breves huecos (1..max_gap_frames) donde `is_valid == False` entre dos
       fotogramas con homografía válida del mismo plano de cámara (sin corte de edición).
    2. Interpola linealmente la matriz proyectiva normalizada H (con H[2,2] = 1.0) y calcula H^-1.
    3. Sintetiza puntos de referencia proyectados en los cuadros interpolados si no tenían >= 4
       keypoints, para que el calibrador interactivo del Frontend disponga de pines válidos.
    4. Actualiza las coordenadas `pitch_x_m, pitch_y_m` de jugadores y balón en esos cuadros
       recuperados antes de ejecutar el cosido global.
    """
    if not frame_cache_meta:
        return {"valid_pass1": 0, "interpolated_pass2": 0, "total_valid_pass2": 0}

    sorted_frames = sorted(frame_cache_meta.keys())
    valid_indices = [
        idx
        for idx in sorted_frames
        if frame_cache_meta[idx].get("homography") is not None
        and frame_cache_meta[idx]["homography"].is_valid
        and frame_cache_meta[idx]["homography"].H is not None
    ]

    interpolated_frames_set = set()

    for k in range(len(valid_indices) - 1):
        f_a = valid_indices[k]
        f_b = valid_indices[k + 1]
        gap = f_b - f_a
        if gap <= 1 or gap > max_gap_frames:
            continue

        h_a: HomographyFrameResult = frame_cache_meta[f_a]["homography"]
        h_b: HomographyFrameResult = frame_cache_meta[f_b]["homography"]
        if h_a.H is None or h_b.H is None:
            continue

        # Verificar que en medio del hueco no haya ocurrido un corte de cámara o toma cerrada real
        intermediate_frames = [f for f in range(f_a + 1, f_b) if f in frame_cache_meta]
        has_cut_or_closeup = False
        for f_m in intermediate_frames:
            hm: HomographyFrameResult = frame_cache_meta[f_m]["homography"]
            if hm.shot_cut_histogram or hm.proximity_switch_triggered or hm.green_ratio < 0.14:
                has_cut_or_closeup = True
                break

        if has_cut_or_closeup:
            continue

        # Normalizar H_a y H_b para que H[2, 2] == 1.0 antes de interpolar
        denom_a = float(h_a.H[2, 2]) if abs(float(h_a.H[2, 2])) > 1e-9 else 1.0
        denom_b = float(h_b.H[2, 2]) if abs(float(h_b.H[2, 2])) > 1e-9 else 1.0
        mat_a = h_a.H.astype(np.float64) / denom_a
        mat_b = h_b.H.astype(np.float64) / denom_b

        for f_m in intermediate_frames:
            alpha = float(f_m - f_a) / float(gap)
            mat_interp = (1.0 - alpha) * mat_a + alpha * mat_b
            if abs(mat_interp[2, 2]) > 1e-9:
                mat_interp = mat_interp / mat_interp[2, 2]

            try:
                inv_interp = np.linalg.inv(mat_interp)
            except np.linalg.LinAlgError:
                continue

            hm = frame_cache_meta[f_m]["homography"]
            hm.H = mat_interp
            hm.H_inv = inv_interp
            hm.is_valid = True
            hm.is_closeup_or_cut = False
            hm.status_message = "OK: Homografía sincerada en Pasada 2 (Interpolación Temporal)"

            # Si el cuadro tenía menos de 4 keypoints detectados, proyectar referencias canónicas visibles
            if len([kp for kp in hm.keypoints if kp.is_inlier]) < 4:
                ref_ids = ["15", "16", "17", "18", "10", "13", "20", "23"]
                synth_kps: List[KeypointObservation] = []
                for idx_kp, kp_id in enumerate(ref_ids):
                    if kp_id not in PITCH_COORDS_METERS:
                        continue
                    mx, my = PITCH_COORDS_METERS[kp_id]
                    px_pt = pitch_to_pixel(mx, my, inv_interp)
                    if px_pt is not None:
                        px_x, px_y = px_pt
                        if 0.0 <= px_x <= 3840.0 and 0.0 <= px_y <= 2160.0:
                            synth_kps.append(
                                KeypointObservation(
                                    kp_index=idx_kp,
                                    kp_id=kp_id,
                                    raw_px=(px_x, px_y),
                                    stabilized_px=(px_x, px_y),
                                    pitch_m=(mx, my),
                                    confidence=0.75,
                                    is_inlier=True,
                                    stabilization_note="PASS2_INTERP",
                                )
                            )
                if len(synth_kps) >= 4:
                    hm.keypoints = synth_kps
                    hm.num_inliers = len(synth_kps)

            interpolated_frames_set.add(f_m)

    # Retro-proyectar jugadores y balón en los fotogramas cuya homografía fue sincerada
    if interpolated_frames_set:
        if raw_player_records:
            for rec in raw_player_records:
                f_idx = int(rec.get("frame", -1))
                if f_idx in interpolated_frames_set and rec.get("pitch_x_m") is None:
                    hm = frame_cache_meta[f_idx]["homography"]
                    if hm.H is not None and rec.get("foot_px_x") is not None and rec.get("foot_px_y") is not None:
                        pm = pixel_to_pitch(float(rec["foot_px_x"]), float(rec["foot_px_y"]), hm.H)
                        if pm is not None:
                            rec["pitch_x_m"] = round(float(np.clip(pm[0], -5.0, pitch_length + 5.0)), 3)
                            rec["pitch_y_m"] = round(float(np.clip(pm[1], -5.0, pitch_width + 5.0)), 3)
                            rec["homography_valid"] = True

        if raw_ball_records:
            for b_rec in raw_ball_records:
                f_idx = int(b_rec.get("frame", -1))
                if f_idx in interpolated_frames_set and b_rec.get("pitch_x_m") is None:
                    hm = frame_cache_meta[f_idx]["homography"]
                    if hm.H is not None and b_rec.get("center_px_x") is not None and b_rec.get("center_px_y") is not None:
                        pm = pixel_to_pitch(float(b_rec["center_px_x"]), float(b_rec["center_px_y"]), hm.H)
                        if pm is not None:
                            b_rec["pitch_x_m"] = round(float(np.clip(pm[0], -5.0, pitch_length + 5.0)), 3)
                            b_rec["pitch_y_m"] = round(float(np.clip(pm[1], -5.0, pitch_width + 5.0)), 3)
                            b_rec["homography_valid"] = True

    return {
        "valid_pass1": len(valid_indices),
        "interpolated_pass2": len(interpolated_frames_set),
        "total_valid_pass2": len(valid_indices) + len(interpolated_frames_set),
    }


class ScoutingJsonExporter:
    """
    Exportador oficial del contrato JSON para `scoutingData_FrontEnd`.
    Genera `match_metadata.json`, `tracking_output.json`, `identity_manifest.json`
    y `consolidation_report.json` a partir de las inferencias sinceradas en la Pasada 2.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        pitch_cfg = self.config.get("pitch", {})
        self.pitch_length = float(pitch_cfg.get("largo_cancha", LARGO_CANCHA))
        self.pitch_width = float(pitch_cfg.get("ancho_cancha", ANCHO_CANCHA))

    @staticmethod
    def _map_team_and_role(
        internal_team_id: int,
        internal_role: str,
        pitch_x: Optional[float],
        pitch_length: float = 105.0,
    ) -> Tuple[int, str, str]:
        """
        Convierte la codificación interna de `soccer_homography_2d` al contrato de `scoutingData_FrontEnd`:
          - Interno 1 (Team_A) -> team_id: 0, team: "team_a", role: "player"
          - Interno 2 (Team_B) -> team_id: 1, team: "team_b", role: "player"
          - Interno 3 (GK)     -> team_id: 0 o 1 (según mitad de cancha), role: "goalkeeper"
          - Interno 4 (Referee)-> team_id: -1, team: "referee", role: "referee"
        """
        role_upper = str(internal_role).upper()
        if internal_team_id == 4 or role_upper == "REFEREE":
            return -1, "referee", "referee"
        if internal_team_id == 3 or role_upper in ("GK", "GOALKEEPER"):
            if pitch_x is not None and float(pitch_x) > (pitch_length / 2.0):
                return 1, "team_b", "goalkeeper"
            return 0, "team_a", "goalkeeper"
        if internal_team_id == 2:
            return 1, "team_b", "player"
        return 0, "team_a", "player"

    def export_all(
        self,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        fps: float,
        total_frames: int,
        frame_width: int,
        frame_height: int,
        frame_cache_meta: Dict[int, Dict[str, Any]],
        stitched_by_frame: Dict[int, List[Dict[str, Any]]],
        reconstructed_balls_by_frame: Dict[int, List[Dict[str, Any]]],
        team_bgr_palette: Dict[int, Tuple[int, int, int]],
        stitch_stats: Dict[str, Any],
        ball_stats: Dict[str, Any],
        homog_reconcile_stats: Dict[str, int],
        timings: Dict[str, float],
        device_str: str = "cuda",
        model_format: str = "pt",
        cuda_available: bool = True,
    ) -> Dict[str, Path]:
        """
        Construye y escribe en disco los archivos JSON del partido en `output_dir`:
          - `match_metadata.json`
          - `tracking_output.json`
          - `identity_manifest.json`
          - `consolidation_report.json`
        """
        out_folder = Path(output_dir)
        out_folder.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).name

        # 1. Construir paleta de equipos homologada (0: team_a, 1: team_b, -1: referee)
        bgr_a = team_bgr_palette.get(1, (255, 190, 0))
        bgr_b = team_bgr_palette.get(2, (50, 60, 245))
        bgr_ref = team_bgr_palette.get(4, (0, 215, 255))

        bgr_a_list, rgb_a_list, hex_a = _bgr_to_rgb_and_hex(bgr_a)
        bgr_b_list, rgb_b_list, hex_b = _bgr_to_rgb_and_hex(bgr_b)
        bgr_r_list, rgb_r_list, hex_r = _bgr_to_rgb_and_hex(bgr_ref)

        team_hex_lookup = {0: hex_a, 1: hex_b, -1: hex_r}

        # Mapa auxiliar `raw_track_id -> global_track_id` por frame para traducir `carrier_id` del balón
        frames_json: List[Dict[str, Any]] = []
        possession_counts = {0: 0, 1: 0}
        voronoi_t0_samples: List[float] = []
        apollonius_t0_samples: List[float] = []

        sorted_frame_indices = sorted(frame_cache_meta.keys())

        for f_idx in sorted_frame_indices:
            meta = frame_cache_meta[f_idx]
            homog_res: HomographyFrameResult = meta["homography"]
            H_mat = homog_res.H if homog_res.is_valid else None
            H_inv = homog_res.H_inv if homog_res.is_valid else None

            raw_to_global_in_frame: Dict[int, int] = {}
            global_to_team_in_frame: Dict[int, int] = {}
            global_to_player_obj: Dict[int, Dict[str, Any]] = {}

            # --- A. Keypoints Sincerados ---
            keypoints_json: List[Dict[str, Any]] = []
            if homog_res.keypoints:
                for kp in homog_res.keypoints:
                    keypoints_json.append(
                        {
                            "id": str(kp.kp_id).zfill(2),
                            "pixel_x": round(float(kp.stabilized_px[0]), 2),
                            "pixel_y": round(float(kp.stabilized_px[1]), 2),
                            "pitch_x": round(float(kp.pitch_m[0]), 2),
                            "pitch_y": round(float(kp.pitch_m[1]), 2),
                            "conf": round(float(kp.confidence), 3),
                            "is_inlier": bool(kp.is_inlier),
                        }
                    )

            # --- B. Jugadores Cosidos y Sincerados (Pasada 2) ---
            players_stitched = stitched_by_frame.get(f_idx, [])
            players_json: List[Dict[str, Any]] = []

            for p in players_stitched:
                g_id = int(p["id"])
                r_id = int(p.get("raw_track_id", g_id))
                raw_to_global_in_frame[r_id] = g_id
                raw_to_global_in_frame[g_id] = g_id

                pm = p.get("pitch_m")
                px_x, px_y = float(p["foot_px"][0]), float(p["foot_px"][1])
                bbox = [round(float(v), 2) for v in p["bbox"]]
                is_interp = bool(p.get("is_interpolated", False))

                # Si fue interpolado en 2D y tenemos H^-1 válida, reproyectar pies y ajustar bbox al video
                if is_interp and pm is not None and H_inv is not None:
                    reproj_px = pitch_to_pixel(float(pm[0]), float(pm[1]), H_inv)
                    if reproj_px is not None:
                        dx_px = reproj_px[0] - px_x
                        dy_px = reproj_px[1] - px_y
                        px_x, px_y = reproj_px[0], reproj_px[1]
                        bbox = [
                            round(bbox[0] + dx_px, 2),
                            round(bbox[1] + dy_px, 2),
                            round(bbox[2] + dx_px, 2),
                            round(bbox[3] + dy_px, 2),
                        ]

                # Si no tenía coordenada 2D pero la homografía fue recuperada en Pasada 2, proyectar ahora
                if pm is None and H_mat is not None:
                    proj_m = pixel_to_pitch(px_x, px_y, H_mat)
                    if proj_m is not None:
                        pm = (
                            float(np.clip(proj_m[0], 0.0, self.pitch_length)),
                            float(np.clip(proj_m[1], 0.0, self.pitch_width)),
                        )

                pitch_x_val = round(float(np.clip(pm[0], 0.0, self.pitch_length)), 2) if pm is not None else None
                pitch_y_val = round(float(np.clip(pm[1], 0.0, self.pitch_width)), 2) if pm is not None else None

                fe_team_id, fe_team_str, fe_role = self._map_team_and_role(
                    int(p.get("team_id", 1)),
                    str(p.get("role", "Player")),
                    pitch_x_val,
                    self.pitch_length,
                )
                global_to_team_in_frame[g_id] = fe_team_id

                jersey_val = p.get("jersey_number")
                shirt_str = str(jersey_val).strip() if (jersey_val is not None and str(jersey_val).strip() != "") else None

                p_obj = {
                    "track_id": g_id,
                    "raw_track_id": r_id,
                    "identity_id": f"ID_{g_id:04d}",
                    "team": fe_team_str,
                    "team_id": fe_team_id,
                    "team_color": team_hex_lookup.get(fe_team_id, hex_a),
                    "role": fe_role,
                    "shirt_number": shirt_str,
                    "confidence": round(float(p.get("conf", 0.75)), 4),
                    "is_interpolated": is_interp,
                    "bbox": bbox,
                    "pixel_x": round(px_x, 2),
                    "pixel_y": round(px_y, 2),
                    "pitch_x": pitch_x_val,
                    "pitch_y": pitch_y_val,
                    "speed_mps": round(float(p.get("speed_mps", 0.0)), 2),
                    "vx_mps": round(float(p.get("vx_mps", 0.0)), 2),
                    "vy_mps": round(float(p.get("vy_mps", 0.0)), 2),
                }
                players_json.append(p_obj)
                global_to_player_obj[g_id] = p_obj

            # Muestrear control territorial Voronoi/Apolonio cada 15 cuadros válidos
            if f_idx % 15 == 0 and len(players_stitched) >= 2:
                ctrl = compute_voronoi_and_apollonius_surface(
                    players_stitched,
                    largo_cancha=self.pitch_length,
                    ancho_cancha=self.pitch_width,
                    grid_w=52,
                    grid_h=34,
                )
                if ctrl is not None:
                    voronoi_t0_samples.append(float(ctrl["team_a_voronoi_pct"]))
                    apollonius_t0_samples.append(float(ctrl["team_a_apollonius_pct"]))

            # --- C. Balón Reconstruido y Sincerado (Pasada 2) ---
            balls_list = reconstructed_balls_by_frame.get(f_idx, [])
            ball_json: Optional[Dict[str, Any]] = None
            possessor_global_id: Optional[int] = None
            possessor_team_id: Optional[int] = None
            possessor_dist_m: Optional[float] = None
            possessor_rel_speed: Optional[float] = None
            possession_confirmed = False

            if balls_list:
                b = balls_list[0]
                b_pm = b.get("pitch_m")
                b_cx, b_cy = float(b["center_px"][0]), float(b["center_px"][1])
                b_bbox = [round(float(v), 2) for v in b["bbox"]]
                b_interp = bool(b.get("is_interpolated", False))

                # Si el balón fue reconstruido en 2D (vuelo/Apolonio/oclusión) y hay H^-1, reproyectar a píxel
                if b_interp and b_pm is not None and H_inv is not None:
                    reproj_b = pitch_to_pixel(float(b_pm[0]), float(b_pm[1]), H_inv)
                    if reproj_b is not None:
                        b_cx, b_cy = reproj_b[0], reproj_b[1]
                        b_bbox = [
                            round(b_cx - 7.0, 2),
                            round(b_cy - 7.0, 2),
                            round(b_cx + 7.0, 2),
                            round(b_cy + 7.0, 2),
                        ]

                if b_pm is None and H_mat is not None:
                    proj_bm = pixel_to_pitch(b_cx, b_cy, H_mat)
                    if proj_bm is not None:
                        b_pm = (
                            float(np.clip(proj_bm[0], 0.0, self.pitch_length)),
                            float(np.clip(proj_bm[1], 0.0, self.pitch_width)),
                        )

                b_pitch_x = round(float(np.clip(b_pm[0], 0.0, self.pitch_length)), 2) if b_pm is not None else None
                b_pitch_y = round(float(np.clip(b_pm[1], 0.0, self.pitch_width)), 2) if b_pm is not None else None

                raw_carrier = b.get("carrier_id")
                if raw_carrier is not None:
                    possessor_global_id = raw_to_global_in_frame.get(int(raw_carrier), int(raw_carrier))

                # Si no había carrier explícito pero hay jugadores cerca en 2D (< 2.2m), asociar al más cercano
                if possessor_global_id is None and b_pitch_x is not None and b_pitch_y is not None:
                    best_d = 2.2
                    for p_cand in players_json:
                        if p_cand["team_id"] in (0, 1) and p_cand["pitch_x"] is not None and p_cand["pitch_y"] is not None:
                            d_m = float(np.hypot(p_cand["pitch_x"] - b_pitch_x, p_cand["pitch_y"] - b_pitch_y))
                            if d_m < best_d:
                                best_d = d_m
                                possessor_global_id = int(p_cand["track_id"])

                if possessor_global_id is not None and possessor_global_id in global_to_player_obj:
                    carrier_p = global_to_player_obj[possessor_global_id]
                    if carrier_p["team_id"] in (0, 1):
                        possessor_team_id = int(carrier_p["team_id"])
                        possession_counts[possessor_team_id] += 1
                        possession_confirmed = True
                        if b_pitch_x is not None and carrier_p["pitch_x"] is not None:
                            possessor_dist_m = round(
                                float(np.hypot(carrier_p["pitch_x"] - b_pitch_x, carrier_p["pitch_y"] - b_pitch_y)),
                                2,
                            )
                        possessor_rel_speed = round(
                            abs(float(b.get("speed_mps", 0.0)) - float(carrier_p.get("speed_mps", 0.0))),
                            2,
                        )

                ball_json = {
                    "detected": not b_interp,
                    "confidence": round(float(b.get("conf", 0.5)), 4),
                    "bbox": b_bbox,
                    "pixel_x": round(b_cx, 2),
                    "pixel_y": round(b_cy, 2),
                    "pitch_x": b_pitch_x,
                    "pitch_y": b_pitch_y,
                    "is_interpolated": b_interp,
                    "state": str(b.get("state", "LOCKED")),
                    "speed_mps": round(float(b.get("speed_mps", 0.0)), 2),
                    "carrier_track_id": possessor_global_id,
                }

            tot_poss_so_far = possession_counts[0] + possession_counts[1]
            if tot_poss_so_far > 0:
                t0_pct_running = round((possession_counts[0] / tot_poss_so_far) * 100.0, 1)
                t1_pct_running = round(100.0 - t0_pct_running, 1)
            else:
                t0_pct_running, t1_pct_running = 50.0, 50.0

            is_h_interp = "Pasada 2" in homog_res.status_message
            homog_json = {
                "valid": bool(homog_res.is_valid and H_mat is not None),
                "is_interpolated": is_h_interp,
                "status": str(homog_res.status_message),
                "matrix": _matrix_to_nested_list(H_mat) if homog_res.is_valid else None,
                "inv_matrix": _matrix_to_nested_list(H_inv) if homog_res.is_valid else None,
            }

            frames_json.append(
                {
                    "frame": int(f_idx),
                    "timestamp": round(float(f_idx) / max(1.0, float(fps)), 3),
                    "keypoints": keypoints_json,
                    "players": players_json,
                    "ball": ball_json,
                    "possession": {
                        "possessor_id": possessor_global_id,
                        "candidate_id": possessor_global_id,
                        "team_id": possessor_team_id,
                        "distance_m": possessor_dist_m,
                        "relative_speed_mps": possessor_rel_speed,
                        "is_confirmed": possession_confirmed,
                        "team_0_pct": t0_pct_running,
                        "team_1_pct": t1_pct_running,
                    },
                    "homography": homog_json,
                    "homography_valid": bool(homog_res.is_valid and H_mat is not None),
                }
            )

        # 2. Estadísticas consolidadas de posesión y dominio espacial (Voronoi / Apolonio)
        total_poss = possession_counts[0] + possession_counts[1]
        voronoi_t0 = round(float(np.mean(voronoi_t0_samples)), 1) if voronoi_t0_samples else 50.0
        voronoi_t1 = round(100.0 - voronoi_t0, 1)
        apollonius_t0 = round(float(np.mean(apollonius_t0_samples)), 1) if apollonius_t0_samples else 50.0
        apollonius_t1 = round(100.0 - apollonius_t0, 1)

        if total_poss > 0:
            team_0_pct = round((possession_counts[0] / total_poss) * 100.0, 1)
            team_1_pct = round(100.0 - team_0_pct, 1)
        else:
            # Si no hubo contacto directo registrado, usar el dominio cinemático de Apolonio como estimador
            team_0_pct = apollonius_t0
            team_1_pct = apollonius_t1

        det_cfg = self.config.get("detection", {})
        hom_cfg = self.config.get("homography", {})
        processed_frames_count = len(frames_json)
        duration_sec = round(processed_frames_count / max(1.0, float(fps)), 2)

        metadata_dict: Dict[str, Any] = {
            "match_info": {
                "video_name": video_name,
                "fps": round(float(fps), 2),
                "total_frames": int(processed_frames_count),
                "duration_seconds": duration_sec,
                "resolution": {
                    "width": int(frame_width),
                    "height": int(frame_height),
                    "original_width": int(frame_width),
                    "original_height": int(frame_height),
                },
                "pitch_dimensions_m": {
                    "length": self.pitch_length,
                    "width": self.pitch_width,
                    "penalty_box_length": 16.5,
                    "penalty_box_width": 40.32,
                    "goal_box_length": 5.5,
                    "goal_box_width": 18.32,
                    "centre_circle_radius": 9.15,
                    "penalty_spot_distance": 11.0,
                },
            },
            "teams": {
                "team_a": {
                    "id": 0,
                    "team_id": 0,
                    "name": "Equipo A",
                    "color_hex": hex_a,
                    "primary_color_hex": hex_a,
                    "color_rgb": rgb_a_list,
                    "primary_color_rgb": rgb_a_list,
                    "color_bgr": bgr_a_list,
                    "primary_color_bgr": bgr_a_list,
                    "color_is_assigned": True,
                },
                "team_b": {
                    "id": 1,
                    "team_id": 1,
                    "name": "Equipo B",
                    "color_hex": hex_b,
                    "primary_color_hex": hex_b,
                    "color_rgb": rgb_b_list,
                    "primary_color_rgb": rgb_b_list,
                    "color_bgr": bgr_b_list,
                    "primary_color_bgr": bgr_b_list,
                    "color_is_assigned": True,
                },
                "referee": {
                    "id": -1,
                    "team_id": -1,
                    "name": "Árbitro / Referee",
                    "color_hex": hex_r,
                    "primary_color_hex": hex_r,
                    "color_rgb": rgb_r_list,
                    "primary_color_rgb": rgb_r_list,
                    "color_bgr": bgr_r_list,
                    "primary_color_bgr": bgr_r_list,
                    "color_is_assigned": True,
                },
            },
            "possession_stats": {
                "team_0_pct": team_0_pct,
                "team_1_pct": team_1_pct,
                "voronoi_control_team_0_pct": voronoi_t0,
                "voronoi_control_team_1_pct": voronoi_t1,
                "apollonius_control_team_0_pct": apollonius_t0,
                "apollonius_control_team_1_pct": apollonius_t1,
            },
            "inference": {
                "device": str(device_str),
                "format": str(model_format),
                "cuda_available": bool(cuda_available),
                "environment": "CUDA_TENSORRT" if "engine" in str(model_format).lower() else "PYTORCH",
                "stabilization_method": str(hom_cfg.get("stabilization_method", "both")),
                "two_pass_enabled": True,
                "spatial_geometry": "voronoi_apollonius",
                "thresholds": {
                    "conf_player": float(det_cfg.get("conf_player", 0.50)),
                    "conf_ball": float(det_cfg.get("conf_ball", 0.40)),
                    "conf_keypoint": float(det_cfg.get("conf_keypoint", 0.35)),
                },
                "reconciliation_summary": {
                    "stitched_players": int(stitch_stats.get("stitched_players", 0)),
                    "raw_tracklets": int(stitch_stats.get("raw_tracklets", 0)),
                    "tracklet_merges": int(stitch_stats.get("merges_performed", 0)),
                    "ball_active_frames": int(ball_stats.get("active_ball_frames", 0)),
                    "ball_interpolated_frames": int(ball_stats.get("interpolated_frames", 0)),
                    "homography_valid_pass1": int(homog_reconcile_stats.get("valid_pass1", 0)),
                    "homography_interpolated_pass2": int(homog_reconcile_stats.get("interpolated_pass2", 0)),
                },
            },
            "performance": {
                "total_time_s": round(float(timings.get("total_s", 0.0)), 2),
                "fps_average": round(
                    processed_frames_count / max(1e-3, float(timings.get("total_s", 1.0))), 2
                ),
                "pass1_inference_s": round(float(timings.get("pass1_s", 0.0)), 2),
                "pass2_stitching_s": round(float(timings.get("pass2_s", 0.0)), 2),
                "render_s": round(float(timings.get("render_s", 0.0)), 2),
            },
        }

        tracking_output_dict: Dict[str, Any] = {
            "metadata_file": "match_metadata.json",
            "metadata": metadata_dict,
            "frames": frames_json,
        }

        # 3. Construir `identity_manifest.json` y `consolidation_report.json` para el panel IdentityEngine
        identities_list = stitch_stats.get("identities_manifest", [])
        for id_item in identities_list:
            id_item["match_id"] = video_name

        consolidated_count = sum(1 for item in identities_list if item.get("segment_count", 1) > 1)
        standalone_count = len(identities_list) - consolidated_count
        needs_review_count = sum(1 for item in identities_list if item.get("status") == "NEEDS_REVIEW")

        identity_manifest_dict = {
            "version": "1.0",
            "match_id": video_name,
            "total_identities": len(identities_list),
            "identities": identities_list,
        }

        consolidation_report_dict = {
            "execution_summary": {
                "processing_time_seconds": round(float(timings.get("pass2_s", 0.0)), 2),
                "total_input_tracks": int(stitch_stats.get("raw_tracklets", 0)),
                "total_output_identities": len(identities_list),
                "candidates_evaluated": int(stitch_stats.get("raw_tracklets", 0)) ** 2,
                "total_links_accepted": int(stitch_stats.get("merges_performed", 0)),
                "consolidated_identities_count": consolidated_count,
                "standalone_identities_count": standalone_count,
                "needs_review_count": needs_review_count,
                "conflicts_count": 0,
            },
            "conflicts": [],
        }

        metadata_path = out_folder / "match_metadata.json"
        tracking_path = out_folder / "tracking_output.json"
        manifest_path = out_folder / "identity_manifest.json"
        report_path = out_folder / "consolidation_report.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        with open(tracking_path, "w", encoding="utf-8") as f:
            json.dump(tracking_output_dict, f, indent=2, ensure_ascii=False)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(identity_manifest_dict, f, indent=2, ensure_ascii=False)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(consolidation_report_dict, f, indent=2, ensure_ascii=False)

        return {
            "match_metadata": metadata_path,
            "tracking_output": tracking_path,
            "identity_manifest": manifest_path,
            "consolidation_report": report_path,
        }
