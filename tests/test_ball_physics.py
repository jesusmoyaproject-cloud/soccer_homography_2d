"""
tests/test_ball_physics.py — Pruebas de Calibración de Cámara, Proyección 3D y Efecto Magnus.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ball_physics_3d import (
    estimate_camera_pose_from_homography,
    correct_ball_ground_projection,
    classify_and_fit_flight_trajectory,
    CameraPose,
)
from src.ball_tracker import BallTrajectoryReconstructor


def test_camera_pose_decomposition():
    """
    Verifica que la descomposición de H_inv recupere una posición de cámara verosímil:
    elevación Z_c positiva y posición lateral Y_c por fuera de las líneas de banda.
    """
    f = 1800.0
    cx, cy = 960.0, 540.0
    K = np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0],
    ])

    # Rotación con ángulo de inclinación (pitch tilt ~ 25 grados hacia abajo)
    theta = np.radians(25.0)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])

    # Cámara a mitad de cancha (X=52.5), 25 m detrás de la banda (Y=-25.0), 18 m de altura (Z=18.0)
    C = np.array([52.5, -25.0, 18.0])
    t = -R @ C

    # H_inv ~ K [r1, r2, t]
    H_inv = K @ np.column_stack([R[:, 0], R[:, 1], t])

    pose = estimate_camera_pose_from_homography(H_inv, img_w=1920, img_h=1080)
    assert pose is not None
    assert pose.pitch_elevation_m > 5.0
    assert pose.optical_center_m[2] > 5.0


def test_ground_projection_correction_removes_parallax():
    """
    Verifica que `correct_ball_ground_projection` elimine el paralaje:
    Un balón en (52.5, 30.0) a 5 metros de altura con cámara en (52.5, -20.0, 20.0)
    se proyecta ópticamente en Y = 46.67 m. La corrección debe retornar (52.5, 30.0).
    """
    cam_center = np.array([52.5, -20.0, 20.0])
    true_ground = (52.5, 30.0)
    ball_z = 5.0

    # Rayo óptico intersectando Z=0: P_ray = P_ground + Z * (P_ground - C_xy) / (C_z - Z)
    # P_ray_y = 30.0 + 5.0 * (30.0 - (-20.0)) / (20.0 - 5.0) = 30.0 + 5.0 * (50.0 / 15.0) = 46.6667
    ray_proj_y = 30.0 + ball_z * (30.0 - (-20.0)) / (20.0 - ball_z)
    optical_pos = (52.5, ray_proj_y)

    corrected = correct_ball_ground_projection(
        optical_pitch_m=optical_pos,
        estimated_height_m=ball_z,
        camera_optical_center=cam_center,
    )

    assert abs(corrected[0] - true_ground[0]) < 1e-4
    assert abs(corrected[1] - true_ground[1]) < 1e-4


def test_parabola_flight_classification():
    """
    Verifica que un tramo aéreo a alta velocidad sin jugador sea clasificado
    como AERIAL_PARABOLA y que se calcule height_m > 0.
    """
    n_frames = 20
    fps = 25.0
    records = []
    for i in range(n_frames):
        t = i / fps
        records.append({
            "frame": i + 1,
            "detected": True,
            "center_px_x": 500.0 + i * 15.0,
            "center_px_y": 400.0 - i * 5.0,
            "pitch_x_m": 40.0 + 12.0 * t,
            "pitch_y_m": 25.0 + 4.0 * t,
            "speed_mps": 12.6,
            "carrier_id": None,
            "shot_cut": False,
        })

    fitted = classify_and_fit_flight_trajectory(records, fps=fps)
    assert len(fitted) == n_frames
    mid_item = fitted[n_frames // 2]
    assert mid_item["flight_phase"] == "AERIAL_PARABOLA"
    assert mid_item["height_m"] > 0.5
    assert "optical_pitch_x_m" in mid_item


def test_curved_shot_magnus_classification():
    """
    Verifica que un tiro con comba (aceleración transversal cuadrática en el plano)
    sea reconocido como CURVED_SHOT.
    """
    n_frames = 25
    fps = 25.0
    records = []
    for i in range(n_frames):
        t = i / fps
        # Curva con término t^2 pronunciado (Efecto Magnus)
        records.append({
            "frame": i + 1,
            "detected": True,
            "center_px_x": 400.0 + i * 18.0,
            "center_px_y": 500.0 - i * 8.0,
            "pitch_x_m": 35.0 + 15.0 * t,
            "pitch_y_m": 20.0 + 5.0 * t + 3.0 * (t ** 2),
            "speed_mps": 18.0,
            "carrier_id": None,
            "shot_cut": False,
        })

    fitted = classify_and_fit_flight_trajectory(records, fps=fps)
    mid_item = fitted[n_frames // 2]
    assert mid_item["flight_phase"] == "CURVED_SHOT"
    assert mid_item["height_m"] > 0.5


def test_ball_reconstructor_full_pass():
    """
    Verifica que BallTrajectoryReconstructor ejecute el ciclo completo
    de reconstrucción balística 3D e inserte campos 3D en build_frame_lookup.
    """
    reconstructor = BallTrajectoryReconstructor(fps=25.0)
    raw_records = [
        {
            "frame": 1,
            "detected": True,
            "center_px_x": 500.0,
            "center_px_y": 450.0,
            "bbox_w": 16.0,
            "bbox_h": 16.0,
            "pitch_x_m": 40.0,
            "pitch_y_m": 25.0,
            "speed_mps": 12.0,
            "conf": 0.85,
            "carrier_id": None,
            "shot_cut": False,
        },
        {
            "frame": 2,
            "detected": False,  # Gap a interpolar
            "center_px_x": None,
            "center_px_y": None,
            "bbox_w": 16.0,
            "bbox_h": 16.0,
            "pitch_x_m": None,
            "pitch_y_m": None,
            "speed_mps": 0.0,
            "conf": 0.0,
            "carrier_id": None,
            "shot_cut": False,
        },
        {
            "frame": 3,
            "detected": True,
            "center_px_x": 540.0,
            "center_px_y": 440.0,
            "bbox_w": 16.0,
            "bbox_h": 16.0,
            "pitch_x_m": 41.0,
            "pitch_y_m": 25.4,
            "speed_mps": 12.0,
            "conf": 0.80,
            "carrier_id": None,
            "shot_cut": False,
        },
    ]

    reconstructed_rows, stats = reconstructor.reconstruct(raw_records)
    assert stats["total_frames"] == 3
    assert stats["interpolated_frames"] == 1
    assert reconstructed_rows[1]["pitch_x_m"] is not None

    lookup = reconstructor.build_frame_lookup(reconstructed_rows)
    assert 2 in lookup
    assert "height_m" in lookup[2][0]
    assert "speed_3d_mps" in lookup[2][0]
    assert "flight_phase" in lookup[2][0]


def test_frame_to_frame_jump_filter_and_pixel_smoothing():
    """Verifica que saltos no físicos (outliers) sean filtrados y los píxeles sean suavizados en Pasada 2."""
    config = {
        "ball_tracking": {
            "reconstruction": {
                "max_ball_speed_mps": 35.0,
                "max_spike_jump_px": 120.0,
                "max_interp_gap_frames": 25,
                "enable_smoothing": True,
            }
        }
    }
    reconstructor = BallTrajectoryReconstructor(config=config, fps=30.0)

    # Simular una secuencia continua con un falso positivo que salta 25 metros en 1 frame
    raw_records = [
        {"frame": 1, "detected": True, "center_px_x": 100.0, "center_px_y": 200.0, "pitch_x_m": 50.0, "pitch_y_m": 30.0, "conf": 0.8},
        {"frame": 2, "detected": True, "center_px_x": 105.0, "center_px_y": 202.0, "pitch_x_m": 51.0, "pitch_y_m": 30.2, "conf": 0.8},
        # Falso positivo en el frame 3 (salta bruscamente a 400px y 80m de cancha)
        {"frame": 3, "detected": True, "center_px_x": 450.0, "center_px_y": 600.0, "pitch_x_m": 85.0, "pitch_y_m": 15.0, "conf": 0.4},
        # Vuelve a la trayectoria real en el frame 4
        {"frame": 4, "detected": True, "center_px_x": 115.0, "center_px_y": 206.0, "pitch_x_m": 53.0, "pitch_y_m": 30.6, "conf": 0.8},
        {"frame": 5, "detected": True, "center_px_x": 120.0, "center_px_y": 208.0, "pitch_x_m": 54.0, "pitch_y_m": 30.8, "conf": 0.8},
    ]

    reconstructed, stats = reconstructor.reconstruct(raw_records)
    assert stats["spikes_removed"] >= 1
    # El frame 3 debe haber sido corregido cinemáticamente hacia la interpolación suave de 105 a 115 px
    f3 = next(r for r in reconstructed if r["frame"] == 3)
    assert abs(f3["center_px_x"] - 110.0) < 5.0
    assert abs(f3["pitch_x_m"] - 52.0) < 1.0


def test_prune_isolated_phantom_tracklets_prevents_bridges():
    """Verifica que clusters fantasma aislados de 2 frames sean podados sin lanzar TypeError."""
    reconstructor = BallTrajectoryReconstructor(fps=25.0)
    raw_records = [
        # Cluster 1: Detección fantasma de 2 frames en el círculo central
        {"frame": 10, "detected": True, "center_px_x": 900.0, "center_px_y": 250.0, "pitch_x_m": 52.0, "pitch_y_m": 25.0, "conf": 0.35},
        {"frame": 11, "detected": True, "center_px_x": 905.0, "center_px_y": 252.0, "pitch_x_m": 52.2, "pitch_y_m": 25.1, "conf": 0.38},
        # Brecha sin detección
        {"frame": 12, "detected": False, "center_px_x": None, "center_px_y": None, "pitch_x_m": None, "pitch_y_m": None, "conf": 0.0},
        {"frame": 13, "detected": False, "center_px_x": None, "center_px_y": None, "pitch_x_m": None, "pitch_y_m": None, "conf": 0.0},
        # Cluster 2: Balón real en la banda inferior (distante > 400px y > 30m)
        {"frame": 20, "detected": True, "center_px_x": 400.0, "center_px_y": 800.0, "pitch_x_m": 53.0, "pitch_y_m": 60.0, "conf": 0.85},
        {"frame": 21, "detected": True, "center_px_x": 415.0, "center_px_y": 800.0, "pitch_x_m": 54.0, "pitch_y_m": 60.0, "conf": 0.88},
        {"frame": 22, "detected": True, "center_px_x": 430.0, "center_px_y": 800.0, "pitch_x_m": 55.0, "pitch_y_m": 60.0, "conf": 0.89},
        {"frame": 23, "detected": True, "center_px_x": 445.0, "center_px_y": 800.0, "pitch_x_m": 56.0, "pitch_y_m": 60.0, "conf": 0.90},
    ]

    reconstructed, stats = reconstructor.reconstruct(raw_records)
    assert stats["spikes_removed"] >= 2
    # El cluster 1 debe estar podado
    f10 = next(r for r in reconstructed if r["frame"] == 10)
    assert f10["detected"] is False
    assert f10["center_px_x"] is None



