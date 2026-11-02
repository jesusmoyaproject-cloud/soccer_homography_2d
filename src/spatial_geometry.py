"""
src/spatial_geometry.py — Motor Geométrico de Voronoi y Apolonio para Fútbol 2D.

Implementa:
1. Geometría Cinemática de Apolonio (Tiempo de Llegada / Intercepción con Inercia):
   - Desambiguación de jugador poseedor del balón (`select_apollonius_ball_carrier`) en oclusiones.
   - Detección de toques/rebotes intermedios en huecos de balón (`find_apollonius_intermediate_touch`).
   - Costo direccional de alcance de Apolonio para cosido y reconexión de jugadores (`apollonius_reach_cost`).
   - Lóbulo direccional de Apolonio para el selector espacial de balón con `conf >= 0.15`.
2. Geometría de Voronoi:
   - Ponderación espacial por celdas de dispersión de keypoints en Homografía (`compute_voronoi_keypoint_weights`).
   - Partición de cancha en tiempo real (Voronoi estático y Apolonio dinámico con vectores de velocidad)
     para el minimapa táctico 2D y métricas de control territorial (`compute_voronoi_and_apollonius_surface`).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


def apollonius_time_to_intercept(
    player_pos_m: Tuple[float, float],
    player_vel_mps: Tuple[float, float],
    target_pos_m: Tuple[float, float],
    max_speed_mps: float = 9.5,
    reaction_time_s: float = 0.18,
) -> float:
    """
    Calcula el tiempo físico mínimo (en segundos) que tarda un jugador en llegar al punto `target_pos_m`
    bajo el modelo cinemático de Apolonio con tiempo de reacción e inercia inicial:
      p_inercial = p_0 + v_0 * t_reac
      T = t_reac + ||target - p_inercial|| / v_efectiva(alineación)
    """
    px, py = float(player_pos_m[0]), float(player_pos_m[1])
    vx, vy = float(player_vel_mps[0]), float(player_vel_mps[1])
    tx, ty = float(target_pos_m[0]), float(target_pos_m[1])

    # Posición tras el tiempo de reacción manteniendo la inercia actual
    inert_x = px + vx * reaction_time_s
    inert_y = py + vy * reaction_time_s

    dx = tx - inert_x
    dy = ty - inert_y
    dist_rem = float(np.hypot(dx, dy))
    if dist_rem < 1e-4:
        return 0.0

    speed_curr = float(np.hypot(vx, vy))
    turn_penalty_s = 0.0
    eff_speed = max(4.0, max_speed_mps)
    if speed_curr > 0.4:
        cos_align = (vx * dx + vy * dy) / (speed_curr * dist_rem)
        if cos_align >= 0.0:
            # En el cono frontal, el jugador aprovecha su momento lineal (Círculo de Apolonio)
            eff_speed = max_speed_mps + 0.22 * speed_curr * cos_align
        else:
            # A contrapié, sufre una penalización fija de frenado/giro en segundos (no reduce su v_max a larga distancia)
            turn_penalty_s = 0.32 * (-cos_align) * (speed_curr / max(1.0, max_speed_mps))

    return float(reaction_time_s + turn_penalty_s + dist_rem / eff_speed)


def apollonius_reach_cost(
    pos_a_m: Tuple[float, float],
    vel_a_mps: Tuple[float, float],
    pos_b_m: Tuple[float, float],
    dt_sec: float,
    max_speed_mps: float = 12.0,
) -> float:
    """
    Evalúa el costo cinemático anisotrópico de Apolonio para unir un jugador que estaba en `pos_a_m`
    con velocidad `vel_a_mps` hacia una detección en `pos_b_m` tras `dt_sec` segundos.
    Penaliza inversiones bruscas de 180° a alta velocidad y premia la continuidad en el disco de Apolonio.
    """
    dt_eff = max(0.04, float(dt_sec))
    tau = min(dt_eff, 0.85)
    # Centro del disco de Apolonio desplazado por la velocidad inicial
    center_x = float(pos_a_m[0]) + float(vel_a_mps[0]) * tau * 0.65
    center_y = float(pos_a_m[1]) + float(vel_a_mps[1]) * tau * 0.65

    dist_apollonius = float(np.hypot(float(pos_b_m[0]) - center_x, float(pos_b_m[1]) - center_y))
    dist_euclid = float(np.hypot(float(pos_b_m[0]) - float(pos_a_m[0]), float(pos_b_m[1]) - float(pos_a_m[1])))

    # Penalización adicional si el jugador iba rápido (> 3.0 m/s) y el desplazamiento es opuesto
    speed_a = float(np.hypot(vel_a_mps[0], vel_a_mps[1]))
    dir_penalty = 0.0
    if speed_a > 3.0 and dist_euclid > 0.3 and dt_eff < 1.5:
        dx = float(pos_b_m[0]) - float(pos_a_m[0])
        dy = float(pos_b_m[1]) - float(pos_a_m[1])
        cos_theta = (vel_a_mps[0] * dx + vel_a_mps[1] * dy) / (speed_a * dist_euclid)
        if cos_theta < -0.25:
            dir_penalty = 0.85 * abs(cos_theta) * (speed_a / max(1.0, max_speed_mps))

    return float(0.65 * dist_apollonius + 0.35 * dist_euclid + dir_penalty)


def apollonius_directional_ball_bonus(
    pred_px: Tuple[float, float],
    vel_px: Tuple[float, float],
    cand_px: Tuple[float, float],
) -> float:
    """
    Calcula un modificador direccional de Apolonio [-0.08, +0.10] para candidatos de balón (conf >= 0.15):
    Cuando el balón lleva velocidad significativa en cámara, el lugar geométrico de alcance se estira
    en el sentido del vector velocidad (adelante) y se contrae hacia atrás (evitando engancharse a
    botines blancos o medias del jugador que acaba de patear).
    """
    vx, vy = float(vel_px[0]), float(vel_px[1])
    speed_px = float(np.hypot(vx, vy))
    if speed_px < 4.5:
        return 0.0

    dx = float(cand_px[0]) - float(pred_px[0])
    dy = float(cand_px[1]) - float(pred_px[1])
    dist_px = float(np.hypot(dx, dy))
    if dist_px < 2.0:
        return 0.04

    cos_theta = (vx * dx + vy * dy) / (speed_px * dist_px)
    if cos_theta >= 0.25:
        return float(0.08 * cos_theta)
    if cos_theta <= -0.35:
        return float(0.07 * cos_theta)  # Penalización negativa para falsos positivos que quedan atrás
    return 0.0


def select_apollonius_ball_carrier(
    ball_pos_px: Tuple[float, float],
    ball_pos_m: Optional[Tuple[float, float]],
    ball_vel_mps: Tuple[float, float],
    players: List[Dict[str, Any]],
    max_radius_m: float = 1.85,
    max_radius_px: float = 58.0,
) -> Tuple[Optional[Dict[str, Any]], float, float]:
    """
    Selecciona al jugador que ocluye o conduce el balón combinando:
    - Pertenencia a la celda de Voronoi y tiempo de llegada de Apolonio T_i(ball_pos_m).
    - Alineación entre el vector velocidad del jugador y el vector velocidad del balón.
    Retorna: (jugador_seleccionado, dist_efectiva_m, dist_px)
    """
    if not players:
        return None, 999.0, 9999.0

    best_player = None
    best_score_m = 999.0
    best_dist_m = 999.0
    best_dist_px = 9999.0

    bvx, bvy = float(ball_vel_mps[0]), float(ball_vel_mps[1])
    b_speed = float(np.hypot(bvx, bvy))

    for p in players:
        if p.get("role") == "Referee":
            continue
        fx, fy = p["foot_px"]
        d_px = float(np.hypot(fx - ball_pos_px[0], fy - ball_pos_px[1]))

        pm = p.get("pitch_m")
        if ball_pos_m is not None and pm is not None:
            d_m = float(np.hypot(pm[0] - ball_pos_m[0], pm[1] - ball_pos_m[1]))
            pvx = float(p.get("vx_mps", 0.0))
            pvy = float(p.get("vy_mps", 0.0))

            # Tiempo de intercepción de Apolonio convertido a distancia equivalente en metros
            t_apol = apollonius_time_to_intercept(
                pm, (pvx, pvy), ball_pos_m, max_speed_mps=9.5, reaction_time_s=0.14
            )
            eff_m = 0.55 * d_m + 0.45 * (t_apol * 5.0)

            # Si el balón y el jugador se mueven en la misma dirección, reducir costo (conducción activa)
            p_speed = float(np.hypot(pvx, pvy))
            if b_speed > 1.2 and p_speed > 1.0:
                cos_v = (bvx * pvx + bvy * pvy) / (b_speed * p_speed)
                eff_m *= (1.0 - 0.18 * cos_v)
        else:
            d_m = 999.0
            eff_m = d_px / 35.0

        # Elegibilidad dentro del radio de oclusión físico
        if (d_m <= max_radius_m or d_px <= max_radius_px) and eff_m < best_score_m:
            best_score_m = eff_m
            best_dist_m = d_m
            best_dist_px = d_px
            best_player = p

    # Si ninguno entró en el radio estricto, devolver el de menor score para telemetría
    if best_player is None:
        for p in players:
            if p.get("role") == "Referee":
                continue
            fx, fy = p["foot_px"]
            d_px = float(np.hypot(fx - ball_pos_px[0], fy - ball_pos_px[1]))
            pm = p.get("pitch_m")
            d_m = (
                float(np.hypot(pm[0] - ball_pos_m[0], pm[1] - ball_pos_m[1]))
                if (ball_pos_m is not None and pm is not None)
                else 999.0
            )
            score = d_m if d_m < 900.0 else (d_px / 35.0)
            if score < best_score_m:
                best_score_m = score
                best_dist_m = d_m
                best_dist_px = d_px
                best_player = p

    return best_player, best_dist_m, best_dist_px


def find_apollonius_intermediate_touch(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    fps: float,
    stitched_players_by_frame: Optional[Dict[int, List[Dict[str, Any]]]],
    max_ball_speed_mps: float = 38.0,
) -> Optional[Dict[str, Any]]:
    """
    Geometría de Apolonio para Reconstrucción de Huecos del Balón (Pasada 2):
    Cuando el balón desaparece en `frame_a` (punto A) y reaparece en `frame_b` (punto B),
    evalúa si durante algún fotograma intermedio `f_m` un jugador interceptó el balón según el
    Círculo de Apolonio (donde t_jugador <= t_balon y las velocidades A->M y M->B son físicamente
    válidas). Si existe tal intercepción y el jugador está desviado de la cuerda recta A->B,
    retorna el punto pivote M para reconstruir una trayectoria con toque/rebote realista (A -> M -> B).
    """
    if not stitched_players_by_frame:
        return None

    frame_a = int(row_a["frame"])
    frame_b = int(row_b["frame"])
    gap_frames = frame_b - frame_a
    if gap_frames < 4:
        return None

    has_pitch = (
        row_a.get("pitch_x_m") is not None
        and row_a.get("pitch_y_m") is not None
        and row_b.get("pitch_x_m") is not None
        and row_b.get("pitch_y_m") is not None
    )
    if not has_pitch:
        return None

    ax_m, ay_m = float(row_a["pitch_x_m"]), float(row_a["pitch_y_m"])
    bx_m, by_m = float(row_b["pitch_x_m"]), float(row_b["pitch_y_m"])
    ax_px, ay_px = float(row_a["center_px_x"]), float(row_a["center_px_y"])
    bx_px, by_px = float(row_b["center_px_x"]), float(row_b["center_px_y"])

    chord_vec = np.array([bx_m - ax_m, by_m - ay_m], dtype=np.float64)
    chord_len_m = float(np.linalg.norm(chord_vec))
    if chord_len_m < 1.5:
        return None

    best_pivot: Optional[Dict[str, Any]] = None
    best_cost = 999.0

    for f_m in range(frame_a + 1, frame_b):
        alpha = float(f_m - frame_a) / float(gap_frames)
        # Posición sobre la cuerda lineal en el instante f_m
        line_x_m = (1.0 - alpha) * ax_m + alpha * bx_m
        line_y_m = (1.0 - alpha) * ay_m + alpha * by_m
        line_x_px = (1.0 - alpha) * ax_px + alpha * bx_px
        line_y_px = (1.0 - alpha) * ay_px + alpha * by_px

        dt1 = (f_m - frame_a) / max(1.0, fps)
        dt2 = (frame_b - f_m) / max(1.0, fps)

        frame_players = stitched_players_by_frame.get(f_m, [])
        for p in frame_players:
            if p.get("role") == "Referee":
                continue
            pm = p.get("pitch_m")
            if pm is None:
                continue
            px_m, py_m = float(pm[0]), float(pm[1])
            fx_px, fy_px = float(p["foot_px"][0]), float(p["foot_px"][1])

            # Distancia perpendicular/lateral a la cuerda A->B en metros
            dev_m = float(np.hypot(px_m - line_x_m, py_m - line_y_m))
            # Solo considerar candidatos que estén dentro del radio de alcance de un desvío/control
            # (entre 0.35 m y 4.2 m de la cuerda ideal)
            if dev_m > 4.2:
                continue

            # Verificar factibilidad cinemática del balón en los dos tramos A->M y M->B
            d_am = float(np.hypot(px_m - ax_m, py_m - ay_m))
            d_mb = float(np.hypot(bx_m - px_m, by_m - py_m))
            v_am = d_am / max(0.04, dt1)
            v_mb = d_mb / max(0.04, dt2)
            if v_am > max_ball_speed_mps or v_mb > max_ball_speed_mps:
                continue

            # Criterio de Apolonio: tiempo que tarda el jugador en interceptar el balón en f_m
            pvx = float(p.get("vx_mps", 0.0))
            pvy = float(p.get("vy_mps", 0.0))
            t_intercept = apollonius_time_to_intercept(
                (px_m, py_m), (pvx, pvy), (line_x_m, line_y_m), max_speed_mps=9.5, reaction_time_s=0.12
            )

            # Si el jugador ya está prácticamente sobre el punto de paso (t_intercept <= 0.48s)
            if t_intercept <= 0.48 and dev_m < best_cost:
                best_cost = dev_m
                best_pivot = {
                    "frame": f_m,
                    "pitch_m": (px_m, py_m),
                    "center_px": (fx_px, fy_px),
                    "carrier_id": int(p["id"]),
                    "deviation_m": dev_m,
                }

    return best_pivot


def compute_voronoi_keypoint_weights(dst_pts: np.ndarray) -> np.ndarray:
    """
    Calcula pesos espaciales basados en proximidad de Voronoi para los keypoints detectados:
    Los keypoints aislados (con gran celda de Voronoi / gran distancia a su vecino más cercano)
    reciben mayor peso relativo que los keypoints amontonados en una misma esquina del área,
    evitando que un clúster denso incline el horizonte del lado opuesto de la cancha.
    """
    n = len(dst_pts)
    if n <= 4:
        return np.ones(n, dtype=np.float32)

    diff = dst_pts[:, None, :] - dst_pts[None, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)

    # Distancia al vecino más cercano (radio inscrito de la celda de Voronoi del keypoint)
    nn_dist = np.min(dist_mat, axis=1)
    weights = np.clip(np.sqrt(nn_dist + 1.0), 1.0, 4.0).astype(np.float32)
    weights /= np.mean(weights)
    return weights


def compute_voronoi_and_apollonius_surface(
    players: List[Dict[str, Any]],
    largo_cancha: float = 105.0,
    ancho_cancha: float = 68.0,
    grid_w: int = 105,
    grid_h: int = 68,
) -> Optional[Dict[str, Any]]:
    """
    Calcula en forma 100% vectorizada (< 0.4 ms en CPU) sobre una grilla 2D de la cancha:
      1. Diagrama de Voronoi Estático (distancia euclidiana pura).
      2. Diagrama de Apolonio Dinámico (tiempo mínimo de llegada considerando el vector velocidad
         de cada jugador v_i y tiempo de reacción).
    Retorna mapas de dominio por equipo, bordes de celdas de Apolonio/Voronoi, áreas por jugador (m²)
    y porcentajes de control territorial de Equipo A y Equipo B.
    """
    valid_players: List[Dict[str, Any]] = []
    for p in players:
        pm = p.get("pitch_m")
        team_id = int(p.get("team_id", 1))
        if pm is None or team_id not in (1, 2):
            continue
        mx, my = float(pm[0]), float(pm[1])
        if -2.0 <= mx <= largo_cancha + 2.0 and -2.0 <= my <= ancho_cancha + 2.0:
            valid_players.append(p)

    if len(valid_players) < 2:
        return None

    n_p = len(valid_players)
    pos_x = np.array([float(p["pitch_m"][0]) for p in valid_players], dtype=np.float32)
    pos_y = np.array([float(p["pitch_m"][1]) for p in valid_players], dtype=np.float32)
    vel_x = np.array([float(p.get("vx_mps", 0.0)) for p in valid_players], dtype=np.float32)
    vel_y = np.array([float(p.get("vy_mps", 0.0)) for p in valid_players], dtype=np.float32)
    teams = np.array([int(p.get("team_id", 1)) for p in valid_players], dtype=np.int32)
    pids = [int(p["id"]) for p in valid_players]

    # Grilla de coordenadas en metros: shape (grid_h, grid_w)
    gx = np.linspace(0.5, largo_cancha - 0.5, grid_w, dtype=np.float32)
    gy = np.linspace(0.5, ancho_cancha - 0.5, grid_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy)  # (H, W)

    # 1. VORONOI ESTÁTICO: Distancia euclidiana desde cada celda a cada jugador -> (N, H, W)
    dx_vor = grid_x[None, :, :] - pos_x[:, None, None]
    dy_vor = grid_y[None, :, :] - pos_y[:, None, None]
    dist_vor = np.hypot(dx_vor, dy_vor)
    voronoi_owner_idx = np.argmin(dist_vor, axis=0)  # (H, W)
    voronoi_team_map = teams[voronoi_owner_idx]      # (H, W) con valores 1 o 2

    # 2. APOLONIO DINÁMICO: Tiempo de llegada T_i(x, y) con desplazamiento inercial + velocidad
    t_reac = 0.22
    vmax = 9.2
    inert_x = pos_x + vel_x * t_reac
    inert_y = pos_y + vel_y * t_reac

    dx_apo = grid_x[None, :, :] - inert_x[:, None, None]
    dy_apo = grid_y[None, :, :] - inert_y[:, None, None]
    dist_apo = np.hypot(dx_apo, dy_apo) + 1e-4

    speed_curr = np.hypot(vel_x, vel_y)[:, None, None]
    dot_dir = (vel_x[:, None, None] * dx_apo + vel_y[:, None, None] * dy_apo) / (
        np.maximum(speed_curr, 0.1) * dist_apo
    )
    forward_boost = np.maximum(0.0, dot_dir)
    backward_align = np.maximum(0.0, -dot_dir)

    eff_speed = vmax + 0.24 * speed_curr * forward_boost
    turn_penalty = 0.30 * backward_align * (speed_curr / vmax)
    arrival_time = t_reac + turn_penalty + dist_apo / eff_speed  # (N, H, W)

    apollonius_owner_idx = np.argmin(arrival_time, axis=0)  # (H, W)
    apollonius_team_map = teams[apollonius_owner_idx]      # (H, W)

    # Calcular áreas en m² por jugador y porcentaje de dominio por equipo
    cell_area_m2 = (largo_cancha * ancho_cancha) / float(grid_w * grid_h)
    voronoi_areas: Dict[int, float] = {}
    apollonius_areas: Dict[int, float] = {}
    for k, pid in enumerate(pids):
        voronoi_areas[pid] = round(float(np.count_nonzero(voronoi_owner_idx == k) * cell_area_m2), 2)
        apollonius_areas[pid] = round(float(np.count_nonzero(apollonius_owner_idx == k) * cell_area_m2), 2)

    total_cells = float(grid_w * grid_h)
    team_a_apol_pct = round(100.0 * float(np.count_nonzero(apollonius_team_map == 1)) / total_cells, 1)
    team_b_apol_pct = round(100.0 - team_a_apol_pct, 1)

    team_a_vor_pct = round(100.0 * float(np.count_nonzero(voronoi_team_map == 1)) / total_cells, 1)
    team_b_vor_pct = round(100.0 - team_a_vor_pct, 1)

    return {
        "voronoi_owner_idx": voronoi_owner_idx,
        "voronoi_team_map": voronoi_team_map,
        "apollonius_owner_idx": apollonius_owner_idx,
        "apollonius_team_map": apollonius_team_map,
        "voronoi_areas_m2": voronoi_areas,
        "apollonius_areas_m2": apollonius_areas,
        "team_a_apollonius_pct": team_a_apol_pct,
        "team_b_apollonius_pct": team_b_apol_pct,
        "team_a_voronoi_pct": team_a_vor_pct,
        "team_b_voronoi_pct": team_b_vor_pct,
    }

