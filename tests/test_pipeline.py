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
