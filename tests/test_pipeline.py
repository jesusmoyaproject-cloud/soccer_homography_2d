"""
Suite de pruebas para verificar:
1. Diccionario exacto de 30 keypoints (KP_MAP y PITCH_COORDS_METERS).
2. Estabilización Anti-Parpadeo:
   - Método 1: Ventana de Proximidad en Píxeles + Compensación de Paneo Global de Cámara.
   - Método 2: Proximidad por Proyección Inversa (pred=(450,300), yolo=(450,292) -> (450,298)).
3. Detección de Tomas Cerradas y Blindaje Geométrico/Topológico:
   - Método 1: Interruptor por pérdida de proximidad (>= 80% de puntos violan el umbral).
   - Método 2: Detección de corte por histogramas de color.
   - Método 3: Validación geométrica de la cancha (inliers, error inverso en píxeles, convexidad y horizonte).
   - Pre-filtro topológico de keypoints invertidos en el círculo central (14, 16, 17, 19).
"""

import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pitch_config import KP_MAP, PITCH_COORDS_METERS, LARGO_CANCHA, ANCHO_CANCHA
from src.homography import (
    HomographyStabilizer,
    filter_topological_outliers,
    validate_projected_pitch_topology,
)
from src.tracker_2d import PitchBoTSORT2D
from src.tracklet_stitcher import GlobalTrackletStitcher
from src.ball_tracker import BallPhysicsTracker, BallTrajectoryReconstructor, BallState


def test_kp_map_and_pitch_coords():
    assert len(KP_MAP) == 30
    assert len(PITCH_COORDS_METERS) == 30
    mx, my = LARGO_CANCHA / 2.0, ANCHO_CANCHA / 2.0
    assert PITCH_COORDS_METERS["01"] == (0.0, 0.0)
    assert PITCH_COORDS_METERS["10"] == (16.5, my - 20.16)
    assert PITCH_COORDS_METERS["14"] == (mx - 9.15, my)
    assert PITCH_COORDS_METERS["19"] == (mx + 9.15, my)
    assert PITCH_COORDS_METERS["32"] == (LARGO_CANCHA, ANCHO_CANCHA)


def test_method_1_pixel_window():
    """
    Verifica el ejemplo exacto del Método 1:
    - Cuadro N-1 en (450, 300), radio = 10 px.
    - Cuadro N en (451, 302) -> dentro de la ventana -> aceptado en (451, 302).
    - Cuadro N en (450, 285) -> salto de 15 px -> fuera de la ventana -> retenido o limitado a 10 px.
    """
    cfg_keep = {
        "homography": {
            "method_1_pixel_window": {
                "enabled": True,
                "tolerance_radius_px": 10.0,
                "out_of_window_mode": "keep_previous",
            }
        }
    }
    stab = HomographyStabilizer(cfg_keep)
    stab.prev_keypoints_px["10"] = (450.0, 300.0)

    # Caso dentro de la ventana: (451, 302)
    xy_ok, note_ok, dist_ok = stab.apply_method_1_pixel_window("10", (451.0, 302.0))
    assert xy_ok == (451.0, 302.0)
    assert note_ok == "M1_IN_WINDOW"
    assert dist_ok < 10.0

    # Caso fuera de la ventana (salto de 15 píxeles hacia arriba: 450, 285)
    xy_noisy, note_noisy, dist_noisy = stab.apply_method_1_pixel_window("10", (450.0, 285.0))
    assert abs(dist_noisy - 15.0) < 1e-5
    assert xy_noisy == (450.0, 300.0)
    assert note_noisy == "M1_HELD_PREV"

    # Modo borde de ventana (clamp_border): debe quedar en (450, 290)
    stab.m1_mode = "clamp_border"
    xy_clamp, note_clamp, _ = stab.apply_method_1_pixel_window("10", (450.0, 285.0))
    assert abs(xy_clamp[0] - 450.0) < 1e-5
    assert abs(xy_clamp[1] - 290.0) < 1e-5
    assert note_clamp == "M1_CLAMPED_BORDER"


def test_camera_pan_compensation_prevents_lag():
    """
    Verifica que cuando la cámara hace un paneo coherente de (+20, -4) px en todos los
    puntos, los puntos existentes acompañan el paneo sin quedar retrasados respecto a puntos nuevos.
    """
    stab = HomographyStabilizer()
    stab.prev_keypoints_px = {
        "14": (200.0, 500.0),
        "16": (260.0, 440.0),
        "17": (260.0, 560.0),
        "19": (320.0, 500.0),
    }
    raw_pan = {
        "14": (220.0, 496.0, 0.9),
        "16": (281.0, 436.0, 0.9),
        "17": (279.0, 556.0, 0.9),
        "19": (340.0, 496.0, 0.9),
    }
    dx, dy = stab.estimate_camera_pan_delta(raw_pan)
    assert abs(dx - 20.0) <= 1.0
    assert abs(dy - (-4.0)) <= 1.0

    # Al aplicar Método 1 con pan_delta, '14' entra como M1_IN_WINDOW en (220, 496) sin retraso
    xy_14, note_14, _ = stab.apply_method_1_pixel_window("14", (220.0, 496.0), pan_delta=(dx, dy))
    assert xy_14 == (220.0, 496.0)
    assert note_14 == "M1_IN_WINDOW"


def test_method_2_inverse_projection():
    """
    Verifica el ejemplo exacto del Método 2:
    - Predicción matemática vía M_anterior^-1 dice (450, 300).
    - YOLO detecta en (450, 292) (8 píxeles de error).
    - Con peso de proximidad (0.25), devuelve el punto intermedio ponderado (450, 298).
    """
    stab = HomographyStabilizer({"homography": {"method_2_inverse_projection": {"yolo_weight": 0.25}}})
    mx, my = PITCH_COORDS_METERS["10"]
    H_inv = np.array(
        [
            [1.0, 0.0, 450.0 - mx],
            [0.0, 1.0, 300.0 - my],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    stab.M_anterior_inv = H_inv
    stab.M_anterior = np.linalg.inv(H_inv)

    xy_corr, note, err_px = stab.apply_method_2_inverse_projection("10", (450.0, 292.0))
    assert abs(err_px - 8.0) < 1e-4
    assert abs(xy_corr[0] - 450.0) < 1e-4
    assert abs(xy_corr[1] - 298.0) < 1e-4
    assert note == "M2_INV_WEIGHTED"


def test_topological_filter_and_closeup_methods():
    """Verifica el pre-filtro topológico y los 3 métodos de seguridad."""
    # Pre-filtro topológico: si KP 17 (abajo del círculo) aparece por error arriba de KP 14 con menor conf
    raw_swapped = {
        "15": (250.0, 200.0, 0.92, 12),
        "14": (180.0, 500.0, 0.90, 28),
        "19": (320.0, 500.0, 0.91, 29),
        "17": (250.0, 420.0, 0.45, 14),  # Incorrecto: y(17)=420 < y(14)=500
    }
    filtered = filter_topological_outliers(raw_swapped, PITCH_COORDS_METERS)
    assert "17" not in filtered
    assert "14" in filtered and "19" in filtered

    stab = HomographyStabilizer()

    # 1. Método 1 de Toma Cerrada: >= 80% de puntos saltan simultáneamente (>45px)
    stab.prev_keypoints_px = {
        "01": (100.0, 100.0),
        "02": (100.0, 200.0),
        "03": (100.0, 300.0),
        "04": (100.0, 400.0),
        "05": (100.0, 500.0),
    }
    jumped_raw = {
        "01": (350.0, 400.0, 0.9),
        "02": (350.0, 500.0, 0.9),
        "03": (350.0, 600.0, 0.9),
        "04": (350.0, 700.0, 0.9),
        "05": (102.0, 501.0, 0.9),
    }
    triggered, ratio = stab.check_closeup_method_1_proximity_switch(jumped_raw)
    assert triggered is True
    assert abs(ratio - 0.80) < 1e-5

    # 2. Método 2 de Toma Cerrada: Corte por histograma de color
    green_pitch = np.full((240, 320, 3), (35, 140, 40), dtype=np.uint8)
    closeup_face = np.full((240, 320, 3), (180, 90, 210), dtype=np.uint8)
    cut1, _, green1 = stab.check_closeup_method_2_histogram(green_pitch)
    cut2, diff2, green2 = stab.check_closeup_method_2_histogram(closeup_face)
    assert cut1 is False
    assert green1 > 0.90
    assert cut2 is True
    assert diff2 >= stab.c2_hist_threshold
    assert green2 < 0.05

    # 3. Método 3 de Toma Cerrada: Validación Geométrica (8 puntos detectados, 5 descartados)
    src_pts = np.zeros((8, 2), dtype=np.float32)
    dst_pts = np.zeros((8, 2), dtype=np.float32)
    mask_8_with_5_discarded = np.array([[1], [1], [1], [0], [0], [0], [0], [0]], dtype=np.uint8)
    H_dummy = np.eye(3, dtype=np.float64)

    is_ok, reason, n_in, n_out, in_ratio = stab.check_closeup_method_3_geometry(
        H_dummy, mask_8_with_5_discarded, src_pts, dst_pts
    )
    assert is_ok is False
    assert "BANDERA ROJA" in reason
    assert n_in == 3
    assert n_out == 5
    assert abs(in_ratio - (3.0 / 8.0)) < 1e-5

    # 4. Validación Topológica: Matriz que invierte el eje Y (banda superior debajo de la inferior)
    H_flipped_y = np.diag([1.0, -1.0, 1.0])
    inliers_sample = np.array([[43.35, 34.0], [52.5, 0.0], [52.5, 43.15], [61.65, 34.0]], dtype=np.float32)
    topo_ok, topo_reason = validate_projected_pitch_topology(H_flipped_y, H_flipped_y, inliers_sample)
    assert topo_ok is False
    assert "BANDERA ROJA" in topo_reason


def test_pass1_kalman_2d_and_pass2_global_tracklet_stitching():
    """
    Verifica el flujo completo de Pasada 1 (Kalman 2D + Gating de Equipo) y
    Pasada 2 (Propagación retroactiva de dorsal + Global Tracklet Stitching tras corte de cámara).
    """
    tracker_2d = PitchBoTSORT2D()
    fps = 25.0

    # Simular un jugador (raw_track_id=10, Team 1) corriendo de X=40m a X=42m entre frames 1..10
    # El dorsal "9" recién es leído por EasyOCR en el frame 8.
    for f in range(1, 11):
        p = {
            "id": 10,
            "role": "Player",
            "class_id": 2,
            "conf": 0.88,
            "bbox": [300.0, 400.0, 340.0, 490.0],
            "foot_px": (320.0, 490.0),
            "pitch_m": (40.0 + 0.2 * f, 30.0),
            "team_id": 1,
            "team_label": "Team_A",
            "color_bgr": (255, 190, 0),
            "lab_feature": [50.0, 120.0, 140.0],
            "jersey_ocr_raw": "9" if f == 8 else None,
            "jersey_ocr_conf": 0.91 if f == 8 else 0.0,
            "jersey_number": "9" if f >= 8 else None,
        }
        tracker_2d.update_frame(frame_idx=f, fps=fps, players=[p], homography_valid=True, shot_cut=False)

    # En el frame 15 ocurre un corte de cámara (shot_cut=True) y el mismo jugador reaparece con
    # un nuevo raw_track_id=245 cerca de (43.5m, 30.2m), mismo equipo (Team 1), y otro rival (Team 2) cerca.
    for f in range(15, 22):
        p_same = {
            "id": 245,
            "role": "Player",
            "class_id": 2,
            "conf": 0.90,
            "bbox": [350.0, 400.0, 390.0, 490.0],
            "foot_px": (370.0, 490.0),
            "pitch_m": (43.0 + 0.2 * (f - 15), 30.2),
            "team_id": 1,
            "team_label": "Team_A",
            "color_bgr": (255, 190, 0),
            "lab_feature": [51.0, 121.0, 139.0],
            "jersey_ocr_raw": None,
            "jersey_ocr_conf": 0.0,
            "jersey_number": None,
        }
        p_rival = {
            "id": 246,
            "role": "Player",
            "class_id": 2,
            "conf": 0.85,
            "bbox": [355.0, 405.0, 395.0, 495.0],
            "foot_px": (375.0, 495.0),
            "pitch_m": (42.8, 30.1),
            "team_id": 2,  # Equipo rival: jamás debe coserse con raw_track_id=10
            "team_label": "Team_B",
            "color_bgr": (40, 50, 240),
            "lab_feature": [50.0, 175.0, 150.0],
            "jersey_ocr_raw": None,
            "jersey_ocr_conf": 0.0,
            "jersey_number": None,
        }
        tracker_2d.update_frame(
            frame_idx=f,
            fps=fps,
            players=[p_same, p_rival],
            homography_valid=True,
            shot_cut=(f == 15),
        )

    # Ejecutar Pasada 2: Global Tracklet Stitching
    stitcher = GlobalTrackletStitcher(fps=fps)
    stitched_rows, stats = stitcher.stitch_and_propagate(tracker_2d.tabular_records)

    # Verificar que raw_track_id=10 y raw_track_id=245 fueron unidos en el mismo global_track_id=1,
    # mientras que el rival (raw_track_id=246, Team 2) recibió un global_track_id distinto (2).
    assert stats["stitched_players"] == 2
    assert stats["merges_performed"] == 1

    by_frame = stitcher.build_frame_lookup(stitched_rows)
    # Verificar que el dorsal "9" leído en el frame 8 se propagó retroactivamente al Frame 1 y hacia adelante al Frame 20
    frame_1_player = by_frame[1][0]
    frame_20_players = {p["id"]: p for p in by_frame[20]}
    assert frame_1_player["id"] == 1
    assert frame_1_player["jersey_number"] == "9"
    assert frame_20_players[1]["jersey_number"] == "9"
    assert frame_20_players[2]["team_id"] == 2


def test_ball_spatial_likelihood_dynamic_roi_fsm_and_reconstruction():
    """
    Verifica los 4 pilares del seguimiento de balón:
    1. Selector por Verosimilitud Espacial (conf >= 0.15 prefiere candidato cercano al Kalman frente a FP lejano con mayor conf).
    2. ROI Dinámico (512x512 en LOCKED, 640x640 en COASTING/OCCLUDED, None en LOST).
    3. Máquina de Estados (LOCKED -> OCCLUDED_BY_PLAYER -> LOCKED).
    4. Reconstrucción Post-Inferencia (Spike Removal + Interpolación de Huecos a Traza Continua).
    """
    bt = BallPhysicsTracker()
    fps = 25.0
    H_ident = np.array([[0.05, 0.0, 10.0], [0.0, 0.05, 5.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # En LOST: sin ROI dinámico
    assert bt.state == BallState.LOST
    assert bt.get_dynamic_roi(1920, 1080) is None

    # Frame 1: Detección inicial clara (conf = 0.72) -> Pasa a LOCKED
    c1 = [{"id": 1, "class_id": 0, "role": "Ball", "conf": 0.72, "bbox": [492, 392, 508, 408], "center_px": (500.0, 400.0), "pitch_m": None}]
    b1 = bt.update(frame_idx=1, fps=fps, candidates=c1, players=[], homography_H=H_ident, homography_valid=True)
    assert b1 is not None
    assert bt.state == BallState.LOCKED

    # En LOCKED: el ROI dinámico debe ser de 512x512 alrededor de (500, 400)
    roi_locked = bt.get_dynamic_roi(1920, 1080)
    assert roi_locked is not None
    assert (roi_locked[2] - roi_locked[0]) == 512

    # Frame 2: El balón real tiene motion blur (conf = 0.21 en (512, 402)), mientras que en la tribuna/media lejana
    # aparece un falso positivo con conf = 0.62 en (1400, 150). El selector espacial debe elegir el balón real (0.21).
    c2 = [
        {"id": 1, "class_id": 0, "role": "Ball", "conf": 0.62, "bbox": [1392, 142, 1408, 158], "center_px": (1400.0, 150.0), "pitch_m": None},
        {"id": 2, "class_id": 0, "role": "Ball", "conf": 0.21, "bbox": [504, 394, 520, 410], "center_px": (512.0, 402.0), "pitch_m": None},
    ]
    b2 = bt.update(frame_idx=2, fps=fps, candidates=c2, players=[], homography_H=H_ident, homography_valid=True)
    assert b2 is not None
    assert abs(b2["center_px"][0] - 512.0) < 10.0

    # Frame 3 y 4: El balón queda oculto detrás de los pies del jugador ID=7 en (520, 404) -> OCCLUDED_BY_PLAYER
    player_7 = [{"id": 7, "role": "Player", "foot_px": (520.0, 404.0), "pitch_m": (36.0, 25.2)}]
    b3 = bt.update(frame_idx=3, fps=fps, candidates=[], players=player_7, homography_H=H_ident, homography_valid=True)
    b4 = bt.update(frame_idx=4, fps=fps, candidates=[], players=player_7, homography_H=H_ident, homography_valid=True)
    assert bt.state == BallState.OCCLUDED_BY_PLAYER
    assert bt.carrier_player_id == 7
    assert b3 is not None and b4 is not None

    # Frame 5: El balón sale del pie del jugador y vuelve a detectarse en (545, 408) -> vuelve a LOCKED
    c5 = [{"id": 1, "class_id": 0, "role": "Ball", "conf": 0.68, "bbox": [537, 400, 553, 416], "center_px": (545.0, 408.0), "pitch_m": None}]
    b5 = bt.update(frame_idx=5, fps=fps, candidates=c5, players=player_7, homography_H=H_ident, homography_valid=True)
    assert bt.state == BallState.LOCKED
    assert b5 is not None

    # Probar Reconstrucción Post-Inferencia (Pasada 2):
    # Inyectar un pico espurio imposible en el frame 6 y una detección limpia en el frame 7
    bt.tabular_records.append({
        "frame": 6,
        "detected": True,
        "state": "LOCKED",
        "conf": 0.50,
        "likelihood": 0.50,
        "center_px_x": 1500.0,  # Salto espurio de ~950 px en 1 frame
        "center_px_y": 900.0,
        "bbox_w": 16.0,
        "bbox_h": 16.0,
        "pitch_x_m": 85.0,
        "pitch_y_m": 50.0,
        "speed_mps": 45.0,
        "carrier_id": None,
        "roi_used": False,
        "homography_valid": True,
        "shot_cut": False,
        "is_interpolated": False,
    })
    bt.tabular_records.append({
        "frame": 7,
        "detected": True,
        "state": "LOCKED",
        "conf": 0.75,
        "likelihood": 0.80,
        "center_px_x": 575.0,
        "center_px_y": 412.0,
        "bbox_w": 16.0,
        "bbox_h": 16.0,
        "pitch_x_m": 38.75,
        "pitch_y_m": 25.6,
        "speed_mps": 12.0,
        "carrier_id": None,
        "roi_used": True,
        "homography_valid": True,
        "shot_cut": False,
        "is_interpolated": False,
    })
    bt.tabular_records.append({
        "frame": 8,
        "detected": True,
        "state": "LOCKED",
        "conf": 0.78,
        "likelihood": 0.82,
        "center_px_x": 590.0,
        "center_px_y": 414.0,
        "bbox_w": 16.0,
        "bbox_h": 16.0,
        "pitch_x_m": 39.5,
        "pitch_y_m": 25.7,
        "speed_mps": 12.0,
        "carrier_id": None,
        "roi_used": True,
        "homography_valid": True,
        "shot_cut": False,
        "is_interpolated": False,
    })

    reconstructor = BallTrajectoryReconstructor(fps=fps)
    recon_rows, stats = reconstructor.reconstruct(bt.tabular_records)
    assert stats["spikes_removed"] == 1
    assert stats["active_ball_frames"] == 8
    lookup = reconstructor.build_frame_lookup(recon_rows)
    # El frame 6 (antes pico en 1500.0) ahora debe estar interpolado suavemente cerca de ~560 px
    assert 540.0 <= lookup[6][0]["center_px"][0] <= 580.0


def test_voronoi_and_apollonius_geometry_player_and_ball():
    """
    Verifica las mejoras de Geometría de Voronoi y Apolonio:
    1. En cancha 2D: un jugador que corre a alta velocidad expande su área de Apolonio frente a Voronoi estático.
    2. En oclusión de balón: Apolonio elige al jugador que conduce en el sentido del balón frente a un rival estático ligeramente más cerca.
    3. En reconstrucción de huecos: Apolonio detecta un toque/rebote intermedio en los pies de un jugador (A -> M -> B).
    """
    from src.spatial_geometry import (
        compute_voronoi_and_apollonius_surface,
        select_apollonius_ball_carrier,
    )

    # 1. Comparar Voronoi estático vs Apolonio dinámico cuando el Jugador 1 corre a 8.5 m/s hacia el campo rival
    players_field = [
        {"id": 1, "team_id": 1, "role": "Player", "pitch_m": (45.0, 34.0), "vx_mps": 8.5, "vy_mps": 0.0, "foot_px": (500.0, 400.0)},
        {"id": 2, "team_id": 2, "role": "Player", "pitch_m": (60.0, 34.0), "vx_mps": 0.0, "vy_mps": 0.0, "foot_px": (700.0, 400.0)},
    ]
    surf = compute_voronoi_and_apollonius_surface(players_field)
    assert surf is not None
    # En Voronoi estático están simétricos respecto a X=52.5m (50% cada uno),
    # pero en Apolonio el Jugador 1 corriendo a 8.5 m/s hacia la derecha domina > 53% de la cancha
    assert surf["team_a_apollonius_pct"] > surf["team_a_voronoi_pct"]
    assert surf["apollonius_areas_m2"][1] > surf["voronoi_areas_m2"][1]

    # 2. Selección de portador de balón en oclusión mediante Apolonio
    # Balón en (50.0, 34.0) viajando a +7.0 m/s en X.
    # Defensor estático (ID=20) a 1.05m en dirección opuesta vs Delantero (ID=10) a 1.20m corriendo con el balón (+7.5 m/s)
    duel_players = [
        {"id": 20, "team_id": 2, "role": "Player", "pitch_m": (48.95, 34.0), "vx_mps": -1.0, "vy_mps": 0.0, "foot_px": (590.0, 400.0)},
        {"id": 10, "team_id": 1, "role": "Player", "pitch_m": (51.20, 34.0), "vx_mps": 7.5, "vy_mps": 0.0, "foot_px": (612.0, 400.0)},
    ]
    carrier, _, _ = select_apollonius_ball_carrier(
        ball_pos_px=(600.0, 400.0),
        ball_pos_m=(50.0, 34.0),
        ball_vel_mps=(7.0, 0.0),
        players=duel_players,
    )
    assert carrier is not None
    assert carrier["id"] == 10

    # 3. Verificar que Tactical2DRenderer.render_2d_plane dibuja el mapa de Voronoi/Apolonio y flechas sin error
    from src.renderer import Tactical2DRenderer
    from src.homography import HomographyFrameResult
    renderer = Tactical2DRenderer()
    dummy_h = HomographyFrameResult(H=np.eye(3), H_inv=np.eye(3), is_valid=True, is_closeup_or_cut=False, status_message="OK")
    plane_img = renderer.render_2d_plane(players_field, [], dummy_h, update_trails=True)
    assert plane_img is not None and plane_img.shape[0] > 100





