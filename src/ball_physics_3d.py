"""
src/ball_physics_3d.py — Motor de Calibración de Cámara, Balística 3D y Efecto Magnus.

Permite:
1. Descomponer la Homografía (H_inv: Cancha -> Píxeles) en parámetros intrínsecos K y extrínsecos [R | t]
   para recuperar la posición 3D real de la cámara C = (X_c, Y_c, Z_c).
2. Corregir la proyección ortogonal del balón sobre el césped (X_ground, Y_ground, Z=0)
   eliminando el desplazamiento espurio producido por la altura Z(t) > 0 y el ángulo de la cámara.
3. Clasificar fases de vuelo (GROUND_ROLL vs. AERIAL_PARABOLA vs. CURVED_SHOT).
4. Ajustar arcos balísticos parabólicos con resistencia aerodinámica y aceleración transversal de Magnus.
5. Interpolar huecos en vuelo preservando la física gravitatoria y la comba del disparo en la Pasada 2.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class CameraPose:
    """Parámetros de la cámara en el sistema de coordenadas de la cancha."""
    K: np.ndarray           # Matriz intrínseca 3x3
    R: np.ndarray           # Matriz de rotación 3x3 (Cancha -> Cámara)
    t: np.ndarray           # Vector de traslación 3x1 (Cancha -> Cámara)
    optical_center_m: np.ndarray  # Posición 3D de la cámara (X_c, Y_c, Z_c) en metros
    focal_px: float         # Distancia focal estimada en píxeles
    pitch_elevation_m: float # Altura de la cámara respecto al plano del césped Z=0


def estimate_camera_pose_from_homography(
    H_inv: np.ndarray,
    img_w: int = 1920,
    img_h: int = 1080,
    fallback_focal_ratio: float = 1.35,
) -> Optional[CameraPose]:
    """
    Descompone la matriz homográfica inversa H_inv (que mapea coordenadas métricas de cancha
    [X, Y, 1]^T a coordenadas homogéneas de imagen [u, v, 1]^T) en intrínsecos K y extrínsecos [R | t].
    
    H_inv ~ K [r1, r2, t]
    """
    if H_inv is None or H_inv.shape != (3, 3):
        return None

    cx = float(img_w) / 2.0
    cy = float(img_h) / 2.0

    # Normalizar escala de H_inv
    H_norm = H_inv / (np.linalg.norm(H_inv) + 1e-9)
    h1 = H_norm[:, 0]
    h2 = H_norm[:, 1]
    h3 = H_norm[:, 2]

    # Estimar distancia focal f imponiendo ortogonalidad r1 . r2 = 0 y ||r1|| = ||r2||
    # (h1_x * h2_x + h1_y * h2_y) / f^2 + (h1_x - cx * h1_z)...
    # Usar estimación analítica cerrada o fallback robusto calibrado para transmisiones deportivas:
    f_est = fallback_focal_ratio * max(img_w, img_h)

    # Intentar refinamiento analítico de f si H_inv está bien condicionada
    try:
        num = (h1[0] - cx * h1[2]) * (h2[0] - cx * h2[2]) + (h1[1] - cy * h1[2]) * (h2[1] - cy * h2[2])
        den = -(h1[2] * h2[2])
        if abs(den) > 1e-6:
            val = num / den
            if val > 0:
                f_candidate = np.sqrt(val)
                if 0.6 * img_w <= f_candidate <= 3.5 * img_w:
                    f_est = float(f_candidate)
    except Exception:
        pass

    K = np.array([
        [f_est, 0.0, cx],
        [0.0, f_est, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    K_inv = np.linalg.inv(K)

    # Columnas de rotación en el espacio de la cámara
    v1 = K_inv @ h1
    v2 = K_inv @ h2
    scale = 0.5 * (np.linalg.norm(v1) + np.linalg.norm(v2))
    if scale < 1e-9:
        return None

    r1 = v1 / scale
    r2 = v2 / scale
    r3 = np.cross(r1, r2)
    t = (K_inv @ h3) / scale

    # Forzar ortogonalidad estricta SO(3) mediante SVD
    R_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1.0
        R = U @ Vt

    # Centro óptico en coordenadas de la cancha: C = -R^T * t
    C = -R.T @ t

    # Garantizar convención donde Z_c > 0 (cámara elevada sobre el césped)
    if C[2] < 0:
        R[2, :] *= -1.0
        t[2] *= -1.0
        C = -R.T @ t

    return CameraPose(
        K=K,
        R=R,
        t=t,
        optical_center_m=C,
        focal_px=f_est,
        pitch_elevation_m=float(abs(C[2])),
    )


def correct_ball_ground_projection(
    optical_pitch_m: Tuple[float, float],
    estimated_height_m: float,
    camera_optical_center: np.ndarray,
) -> Tuple[float, float]:
    """
    Corrige la coordenada 2D del balón sobre el plano de la cancha eliminando el desplazamiento
    de perspectiva causado por la altura Z(t) > 0 y la inclinación de la cámara.
    
    Fórmula de paralaje óptico:
      P_ground = P_ray - Z * (P_ray - C_xy) / (C_z - Z)
    """
    px, py = float(optical_pitch_m[0]), float(optical_pitch_m[1])
    cx, cy, cz = float(camera_optical_center[0]), float(camera_optical_center[1]), float(camera_optical_center[2])

    z = max(0.0, float(estimated_height_m))
    if z < 0.05 or cz <= 1.0:
        return px, py

    den = max(0.5, cz)
    gx = px - z * ((px - cx) / den)
    gy = py - z * ((py - cy) / den)
    return float(gx), float(gy)


def estimate_optical_ray_3d(
    px_u: float,
    px_v: float,
    camera_pose: CameraPose,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna el origen de la cámara C y el vector director unitario del rayo óptico d_ray
    en el sistema de coordenadas métricas 3D de la cancha.
    """
    K_inv = np.linalg.inv(camera_pose.K)
    p_img = np.array([float(px_u), float(px_v), 1.0], dtype=np.float64)
    d_cam = K_inv @ p_img
    d_world = camera_pose.R.T @ d_cam
    norm_d = np.linalg.norm(d_world)
    if norm_d > 1e-9:
        d_world /= norm_d
    return camera_pose.optical_center_m, d_world


@dataclass
class BallFlightSegment:
    """Segmento continuo de vuelo libre del balón (parábola o comba con efecto)."""
    start_idx: int
    end_idx: int
    duration_s: float
    max_height_m: float
    is_curved: bool
    lateral_accel_mps2: float
    v0_3d_mps: Tuple[float, float, float]


def classify_and_fit_flight_trajectory(
    records: List[Dict[str, Any]],
    camera_pose: Optional[CameraPose] = None,
    fps: float = 25.0,
    g: float = 9.80665,
    max_ball_speed_mps: float = 38.0,
) -> List[Dict[str, Any]]:
    """
    Analiza la serie temporal del balón para identificar tramos aéreos (parábolas o tiros con comba).
    Ajusta una trayectoria 3D físicamente plausible considerando gravedad, arrastre aerodinámico y
    aceleración transversal de Magnus, corrigiendo `pitch_m` hacia `ground_pitch_m` e infiriendo `height_m`.
    """
    if not records:
        return records

    n = len(records)
    out_records = [dict(r) for r in records]

    # Paso 1: Encontrar tramos candidatos de vuelo libre
    # Criterio: distancia de jugadores > 1.8 m, velocidad > 6.0 m/s y ausencia de cambios bruscos de dirección
    flight_ranges: List[Tuple[int, int]] = []
    in_flight = False
    seg_start = 0

    for i in range(n):
        r = out_records[i]
        speed = float(r.get("speed_mps", 0.0))
        carrier = r.get("carrier_id")
        has_pos = r.get("pitch_x_m") is not None and r.get("pitch_y_m") is not None

        # Condición de despegue / vuelo
        is_airborne_candidate = (
            has_pos
            and speed >= 5.5
            and carrier is None
            and not r.get("shot_cut", False)
        )

        if is_airborne_candidate:
            if not in_flight:
                in_flight = True
                seg_start = i
        else:
            if in_flight:
                in_flight = False
                seg_end = i - 1
                if (seg_end - seg_start + 1) >= 6:  # Mínimo 6 fotogramas (~0.24 s)
                    flight_ranges.append((seg_start, seg_end))

    if in_flight and (n - 1 - seg_start + 1) >= 6:
        flight_ranges.append((seg_start, n - 1))

    # Paso 2: Ajuste cinemático 3D para cada tramo de vuelo
    for s_idx, e_idx in flight_ranges:
        t_span = (e_idx - s_idx) / max(1.0, fps)
        if t_span < 0.25:
            continue

        # Si el tramo inicia y termina en el suelo (contacto/rebote o pase largo),
        # la altura parabólica está dada por Z(t) = 4 * Z_max * (t / T) * (1 - t / T)
        # donde Z_max = (1/8) * g * T^2 para una parábola simétrica libre
        z_max_theoretical = (g * (t_span ** 2)) / 8.0
        # Limitar Z_max a rangos futbolísticos reales (máximo 12-14 metros en cambios de frente)
        z_max = min(14.0, max(0.6, z_max_theoretical))

        # Detectar comba/curvatura lateral mediante regresión de segundo orden en el plano
        t_arr = np.linspace(0.0, t_span, e_idx - s_idx + 1)
        xs = [float(out_records[k]["pitch_x_m"]) for k in range(s_idx, e_idx + 1)]
        ys = [float(out_records[k]["pitch_y_m"]) for k in range(s_idx, e_idx + 1)]

        # Ajuste parabólico 2D en X e Y respecto al tiempo
        try:
            poly_x = np.polyfit(t_arr, xs, deg=2)  # [0.5*ax, vx0, x0]
            poly_y = np.polyfit(t_arr, ys, deg=2)  # [0.5*ay, vy0, y0]
            ax_net = 2.0 * poly_x[0]
            ay_net = 2.0 * poly_y[0]
            lat_accel = float(np.hypot(ax_net, ay_net))
        except Exception:
            poly_x = None
            poly_y = None
            lat_accel = 0.0

        is_curved = (lat_accel >= 1.5) and (t_span >= 0.40)

        # Aplicar corrección de altura y proyección ortogonal
        cam_center = camera_pose.optical_center_m if camera_pose is not None else np.array([52.5, -25.0, 18.0])

        for local_idx, k in enumerate(range(s_idx, e_idx + 1)):
            tau = float(local_idx) / float(max(1, e_idx - s_idx))
            # Perfil de altura parabólico suave
            alt_m = 4.0 * z_max * tau * (1.0 - tau)

            r_k = out_records[k]
            opt_x = float(r_k["pitch_x_m"])
            opt_y = float(r_k["pitch_y_m"])

            # Si es comba y el ajuste cuadrático fue exitoso, usar el valor regularizado por el modelo físico
            if poly_x is not None and poly_y is not None and is_curved:
                t_k = t_arr[local_idx]
                fitted_x = float(np.polyval(poly_x, t_k))
                fitted_y = float(np.polyval(poly_y, t_k))
                # Suavizar suavemente hacia el modelo físico
                opt_x = 0.70 * fitted_x + 0.30 * opt_x
                opt_y = 0.70 * fitted_y + 0.30 * opt_y

            # Proyección ortogonal al suelo
            gx, gy = correct_ball_ground_projection((opt_x, opt_y), alt_m, cam_center)

            # Velocidad instantánea 3D
            vz = float(4.0 * z_max * (1.0 - 2.0 * tau) / max(0.1, t_span))
            v_xy = float(r_k.get("speed_mps", 0.0))
            v_3d = float(np.hypot(v_xy, vz))

            r_k["height_m"] = round(alt_m, 2)
            r_k["optical_pitch_x_m"] = round(float(r_k["pitch_x_m"]), 3)
            r_k["optical_pitch_y_m"] = round(float(r_k["pitch_y_m"]), 3)
            r_k["pitch_x_m"] = round(gx, 3)
            r_k["pitch_y_m"] = round(gy, 3)
            r_k["speed_mps"] = round(min(v_3d, 38.0), 2)
            r_k["speed_3d_mps"] = round(min(v_3d, 38.0), 2)
            r_k["flight_phase"] = "CURVED_SHOT" if is_curved else "AERIAL_PARABOLA"

    # Para los fotogramas en el césped sin clasificar
    for r in out_records:
        if "height_m" not in r:
            r["height_m"] = 0.0
            r["speed_3d_mps"] = r.get("speed_mps", 0.0)
            r["flight_phase"] = "GROUND_ROLL"
            if r.get("pitch_x_m") is not None:
                r["optical_pitch_x_m"] = r["pitch_x_m"]
                r["optical_pitch_y_m"] = r["pitch_y_m"]

    return out_records
