"""
Módulo de Homografía 2D con MAGSAC++, Estabilización Anti-Parpadeo y Seguridad para Tomas Cerradas.

Implementa:
- Estimación de Homografía con MAGSAC++ (cv2.USAC_MAGSAC) y refinamiento por inliers.
- Estabilización Anti-Parpadeo y Anti-Deformación con compensación de paneo de cámara:
    * Método 1: La Ventana de Proximidad en Píxeles (radio de tolerancia en píxeles respecto a N-1).
    * Método 2: Proximidad por Proyección Inversa (usando M_anterior^-1 desde PITCH_COORDS_METERS).
- Detección de Tomas Cerradas, Cortes de Edición y Blindaje Topológico:
    * Método 1: El "Interruptor" por Pérdida de Proximidad (>= 80% de puntos saltan simultáneamente).
    * Método 2: Detección de Histogramas de Color (cv2.compareHist entre fotograma N y N-1).
    * Método 3: Validación Geométrica y Topológica de la Cancha (Inliers MAGSAC++, error inverso en píxeles,
      convexidad 2D, orientación horaria y horizonte fuera del campo).
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any
import cv2
import numpy as np

from src.pitch_config import (
    KP_MAP,
    PITCH_COORDS_METERS,
    LARGO_CANCHA,
    ANCHO_CANCHA,
    build_pitch_coords_meters,
)

# Seleccionar MAGSAC++ en OpenCV
HOMOGRAPHY_METHOD_MAGSAC = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)

# Grupos verticales ordenados de arriba hacia abajo (Y creciente en metros -> Y creciente en imagen)
VERTICAL_ORDERED_GROUPS = [
    ["01", "02", "03", "04", "05", "06"],
    ["07", "08"],
    ["10", "11", "12", "13"],
    ["15", "16", "17", "18"],
    ["20", "21", "22", "23"],
    ["25", "26"],
    ["27", "28", "29", "30", "31", "32"],
]


@dataclass
class KeypointObservation:
    """Representa un keypoint detectado y su estado tras los filtros de estabilización."""
    kp_index: int
    kp_id: str
    raw_px: Tuple[float, float]
    stabilized_px: Tuple[float, float]
    pitch_m: Tuple[float, float]
    confidence: float
    is_inlier: bool = True
    stabilization_note: str = "RAW"
    jump_px: float = 0.0
    reproj_error_px: float = 0.0


@dataclass
class HomographyFrameResult:
    """Resultado completo del cálculo y validación de homografía en un fotograma."""
    H: Optional[np.ndarray]                         # Matriz 3x3 (Píxeles -> Metros)
    H_inv: Optional[np.ndarray]                     # Matriz inversa 3x3 (Metros -> Píxeles)
    is_valid: bool                                  # True si la homografía es válida y segura
    is_closeup_or_cut: bool                         # True si se detectó toma cerrada o corte inválido
    status_message: str                             # Estado legible para HUD y diagnóstico
    red_flag_triggered: bool = False                # Bandera roja del Método 3 (Validación Geométrica)
    shot_cut_histogram: bool = False                # Corte detectado por Método 2 (Histograma de Color)
    proximity_switch_triggered: bool = False        # Interruptor activado por Método 1 (>=80% violación)
    histogram_diff: float = 0.0                     # Diferencia de histograma respecto a N-1
    green_ratio: float = 0.0                        # Proporción de césped en pantalla
    num_detected: int = 0                           # Total de keypoints detectados sobre el umbral
    num_inliers: int = 0                            # Keypoints aceptados por MAGSAC++
    num_discarded: int = 0                          # Keypoints descartados por MAGSAC++
    inlier_ratio: float = 0.0                       # Proporción de inliers (inliers / detectados)
    mean_reproj_error_px: float = 0.0               # Error medio de reproyección inversa en píxeles
    keypoints: List[KeypointObservation] = field(default_factory=list)


def pixel_to_pitch(
    px_x: float,
    px_y: float,
    H: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Proyecta un punto (x, y) en píxeles de imagen al plano 2D de la cancha en metros."""
    if H is None:
        return None
    pt = np.array([[[float(px_x), float(px_y)]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H)
    mx, my = float(proj[0, 0, 0]), float(proj[0, 0, 1])
    if np.isnan(mx) or np.isnan(my) or np.isinf(mx) or np.isinf(my):
        return None
    return mx, my


def pitch_to_pixel(
    meter_x: float,
    meter_y: float,
    H_inv: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Proyecta una coordenada métrica (X, Y) de la cancha a píxeles de pantalla usando M^-1."""
    if H_inv is None:
        return None
    pt = np.array([[[float(meter_x), float(meter_y)]]], dtype=np.float32)
    proj = cv2.perspectiveTransform(pt, H_inv)
    px, py = float(proj[0, 0, 0]), float(proj[0, 0, 1])
    if np.isnan(px) or np.isnan(py) or np.isinf(px) or np.isinf(py):
        return None
    return px, py


def filter_topological_outliers(
    mapped_raw: Dict[str, Tuple[float, float, float, int]],
    pitch_coords: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float, float, int]]:
    """
    Descarta keypoints que violan el orden físico topológico en la imagen (por ejemplo,
    swaps de YOLO en el círculo central entre KP 14, 16, 17, 19 o inversiones verticales).
    """
    if len(mapped_raw) < 2:
        return mapped_raw

    valid = dict(mapped_raw)

    # 1. Validar orden vertical dentro de cada línea vertical de la cancha (y_img debe crecer con Y_m)
    for group in VERTICAL_ORDERED_GROUPS:
        present = [kp for kp in group if kp in valid]
        if len(present) >= 2:
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    kp_top, kp_bot = present[i], present[j]
                    if kp_top not in valid or kp_bot not in valid:
                        continue
                    y_top, c_top = valid[kp_top][1], valid[kp_top][2]
                    y_bot, c_bot = valid[kp_bot][1], valid[kp_bot][2]
                    # El punto superior en metros debe estar más arriba en píxeles (menor Y)
                    if y_top >= y_bot - 4.0:
                        if c_top < c_bot:
                            valid.pop(kp_top, None)
                        else:
                            valid.pop(kp_bot, None)

    # 2. Reglas específicas del Círculo Central (14: izq, 16: arriba, 17: abajo, 19: der)
    if "14" in valid and "19" in valid:
        if valid["14"][0] >= valid["19"][0] - 8.0:
            # Están invertidos horizontalmente; conservar el de mayor confianza
            if valid["14"][2] < valid["19"][2]:
                valid.pop("14", None)
            else:
                valid.pop("19", None)

    for mid_kp in ("14", "19"):
        if mid_kp in valid and "16" in valid:
            # KP 16 (arriba del círculo) debe tener menor Y que 14 y 19 (ecuador del círculo)
            if valid["16"][1] >= valid[mid_kp][1] - 3.0:
                if valid["16"][2] < valid[mid_kp][2]:
                    valid.pop("16", None)
                else:
                    valid.pop(mid_kp, None)
        if mid_kp in valid and "17" in valid:
            # KP 17 (abajo del círculo) debe tener mayor Y que 14 y 19 (ecuador del círculo)
            if valid["17"][1] <= valid[mid_kp][1] + 3.0:
                if valid["17"][2] < valid[mid_kp][2]:
                    valid.pop("17", None)
                else:
                    valid.pop(mid_kp, None)

    # 3. Consistencia global de orden horizontal X_m vs x_px
    # Si hay >= 4 puntos, la correlación entre X_m (cuando difieren >= 8m) y x_px debe ser positiva
    items = list(valid.items())
    if len(items) >= 4:
        violations_count: Dict[str, int] = {k: 0 for k in valid}
        for i in range(len(items)):
            k1, (x1, _, _, _) = items[i]
            mx1, _ = pitch_coords[k1]
            for j in range(i + 1, len(items)):
                k2, (x2, _, _, _) = items[j]
                mx2, _ = pitch_coords[k2]
                if abs(mx1 - mx2) >= 9.0:
                    if (mx1 < mx2 and x1 > x2 + 15.0) or (mx1 > mx2 and x1 < x2 - 15.0):
                        violations_count[k1] += 1
                        violations_count[k2] += 1

        max_v = max(violations_count.values(), default=0)
        while max_v >= 2 and len(valid) >= 4:
            worst_kp = max(violations_count, key=lambda k: (violations_count[k], -valid[k][2]))
            valid.pop(worst_kp, None)
            violations_count.pop(worst_kp, None)
            break

    return valid


def validate_projected_pitch_topology(
    H: np.ndarray,
    H_inv: np.ndarray,
    inlier_dst_pts: np.ndarray,
    largo_cancha: float = LARGO_CANCHA,
    ancho_cancha: float = ANCHO_CANCHA,
) -> Tuple[bool, str]:
    """
    Verifica que la proyección inversa (M^-1) preserve la geometría física de una cancha:
    1. La línea del horizonte (w <= 0) NO puede cruzar la región visible del campo.
    2. Un rectángulo canónico sobre la zona detectada debe proyectarse como un cuadrilátero
       estrictamente convexo y con orientación horaria positiva (arriba < abajo, izq < der).
    """
    mx, my = largo_cancha / 2.0, ancho_cancha / 2.0
    x_min = max(0.0, float(np.min(inlier_dst_pts[:, 0])) - 5.0)
    x_max = min(largo_cancha, float(np.max(inlier_dst_pts[:, 0])) + 5.0)
    y_min = max(0.0, float(np.min(inlier_dst_pts[:, 1])) - 5.0)
    y_max = min(ancho_cancha, float(np.max(inlier_dst_pts[:, 1])) + 5.0)

    if x_max - x_min < 10.0:
        x_min, x_max = max(0.0, mx - 16.5), min(largo_cancha, mx + 16.5)
    if y_max - y_min < 10.0:
        y_min, y_max = max(0.0, my - 20.16), min(ancho_cancha, my + 20.16)

    # Cuadrilátero de prueba en sentido horario: TL -> TR -> BR -> BL
    rect_m = np.array(
        [
            [x_min, y_min, 1.0],
            [x_max, y_min, 1.0],
            [x_max, y_max, 1.0],
            [x_min, y_max, 1.0],
        ],
        dtype=np.float64,
    )

    cam_homo = rect_m @ H_inv.T
    # Normalizar con el signo del centro del campo
    center_homo = np.array([mx, my, 1.0], dtype=np.float64) @ H_inv.T
    if center_homo[2] < 0:
        cam_homo = -cam_homo

    w_vals = cam_homo[:, 2]
    if np.any(w_vals <= 0.02):
        return False, "BANDERA ROJA: Línea de horizonte cruza dentro de la cancha (perspectiva invertida)"

    quad_px = cam_homo[:, :2] / w_vals[:, None]
    tl, tr, br, bl = quad_px[0], quad_px[1], quad_px[2], quad_px[3]

    # Verificar convexidad y orientación horaria en coordenadas de imagen (x derecha, y abajo)
    edges = [tr - tl, br - tr, bl - br, tl - bl]
    cross_products = []
    for i in range(4):
        e1 = edges[i]
        e2 = edges[(i + 1) % 4]
        cross_z = float(e1[0] * e2[1] - e1[1] * e2[0])
        cross_products.append(cross_z)

    if not all(c > 0.0 for c in cross_products):
        return False, "BANDERA ROJA: Cuadrilátero de cancha no convexo o volteado"

    # La banda superior debe proyectarse más arriba (menor Y) que la banda inferior
    if tl[1] >= bl[1] - 15.0 or tr[1] >= br[1] - 15.0:
        return False, "BANDERA ROJA: Banda superior proyectada por debajo de banda inferior"

    # El lado izquierdo debe proyectarse a la izquierda del lado derecho
    if tl[0] >= tr[0] - 15.0 or bl[0] >= br[0] - 15.0:
        return False, "BANDERA ROJA: Eje longitudinal X invertido"

    # Verificar relación de aspecto / escorzo razonable de cámara de TV
    top_w = float(np.linalg.norm(tr - tl))
    bot_w = float(np.linalg.norm(br - bl))
    if top_w < 1e-3 or (bot_w / top_w) > 8.0 or (top_w / bot_w) > 2.5:
        return False, "BANDERA ROJA: Deformación extrema de perspectiva en bandas"

    return True, "TOPOLOGIA_OK"


class HomographyStabilizer:
    """
    Motor de Homografía con MAGSAC++, estabilización temporal de keypoints (Métodos 1 y 2)
    con compensación de paneo de cámara, y protección contra tomas cerradas (Métodos 1, 2 y 3).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        pitch_cfg = cfg.get("pitch", {})
        self.largo_cancha = float(pitch_cfg.get("largo_cancha", LARGO_CANCHA))
        self.ancho_cancha = float(pitch_cfg.get("ancho_cancha", ANCHO_CANCHA))
        self.pitch_coords = build_pitch_coords_meters(self.largo_cancha, self.ancho_cancha)

        homog_cfg = cfg.get("homography", {})
        self.ransac_threshold = float(homog_cfg.get("ransac_reproj_threshold", 1.5))
        self.ema_alpha = float(homog_cfg.get("ema_alpha", 0.40))
        self.stabilization_method = str(homog_cfg.get("stabilization_method", "both")).lower()
        self.enable_pan_comp = bool(homog_cfg.get("enable_camera_pan_compensation", True))

        # Configuración Anti-Parpadeo Método 1: Ventana de Proximidad en Píxeles
        m1_cfg = homog_cfg.get("method_1_pixel_window", {})
        self.m1_enabled = bool(m1_cfg.get("enabled", True))
        self.m1_radius_px = float(m1_cfg.get("tolerance_radius_px", 10.0))
        self.m1_mode = str(m1_cfg.get("out_of_window_mode", "clamp_border")).lower()

        # Configuración Anti-Parpadeo Método 2: Proximidad por Proyección Inversa
        m2_cfg = homog_cfg.get("method_2_inverse_projection", {})
        self.m2_enabled = bool(m2_cfg.get("enabled", True))
        self.m2_yolo_weight = float(m2_cfg.get("yolo_weight", 0.25))
        self.m2_max_error_px = float(m2_cfg.get("max_error_px", 35.0))

        # Configuración Tomas Cerradas
        closeup_cfg = homog_cfg.get("closeup_detection", {})

        # Tomas Cerradas Método 1: Interruptor por Pérdida de Proximidad
        c1_cfg = closeup_cfg.get("method_1_proximity_switch", {})
        self.c1_enabled = bool(c1_cfg.get("enabled", True))
        self.c1_violation_ratio = float(c1_cfg.get("violation_ratio_threshold", 0.80))
        self.c1_jump_px = float(c1_cfg.get("jump_distance_px", max(15.0, self.m1_radius_px * 2.5)))
        self.c1_min_shared = int(c1_cfg.get("min_shared_points", 3))

        # Tomas Cerradas Método 2: Histogramas de Color
        c2_cfg = closeup_cfg.get("method_2_color_histogram", {})
        self.c2_enabled = bool(c2_cfg.get("enabled", True))
        self.c2_hist_threshold = float(c2_cfg.get("bhattacharyya_threshold", 0.32))
        self.c2_min_green_ratio = float(c2_cfg.get("min_green_ratio", 0.12))

        # Tomas Cerradas Método 3: Validación Geométrica de la Cancha
        c3_cfg = closeup_cfg.get("method_3_geometric_validation", {})
        self.c3_enabled = bool(c3_cfg.get("enabled", True))
        self.c3_min_inliers = int(c3_cfg.get("min_inliers", 4))
        self.c3_min_inlier_ratio = float(c3_cfg.get("min_inlier_ratio", 0.50))
        self.c3_max_reproj_px = float(c3_cfg.get("max_reproj_error_px", 16.0))
        self.c3_max_point_px = float(c3_cfg.get("max_point_error_px", 32.0))
        self.c3_max_cond = float(c3_cfg.get("max_condition_number", 5e6))
        self.c3_topological_filter = bool(c3_cfg.get("enable_topological_filter", True))

        # Estado temporal entre fotogramas (Cuadro N-1)
        self.prev_keypoints_px: Dict[str, Tuple[float, float]] = {}
        self.M_anterior: Optional[np.ndarray] = None
        self.M_anterior_inv: Optional[np.ndarray] = None
        self.prev_hist: Optional[np.ndarray] = None

    def reset_temporal_state(self) -> None:
        """Limpia la memoria temporal cuando se detecta un corte de edición o cambio de toma."""
        self.prev_keypoints_px.clear()
        self.M_anterior = None
        self.M_anterior_inv = None

    def estimate_camera_pan_delta(
        self, raw_detections: Dict[str, Tuple[float, float, float]]
    ) -> Tuple[float, float]:
        """
        Estima el desplazamiento global (dx, dy) por paneo coherente de cámara entre N-1 y N.
        Si todos los keypoints se mueven juntos en la misma dirección (paneo normal de TV,
        <= 40 px/frame y baja dispersión entre vectores), devuelve la mediana (dx, dy) para
        que la ventana de 10 px evalúe solo el parpadeo individual sin arrastrar puntos viejos.
        Si el salto es masivo (> 45 px) o incoherente, devuelve (0, 0) para que el Interruptor
        del Método 1 detecte el corte de toma.
        """
        if not self.enable_pan_comp or not self.prev_keypoints_px:
            return 0.0, 0.0

        shared_ids = [kp_id for kp_id in raw_detections if kp_id in self.prev_keypoints_px]
        if len(shared_ids) < 3:
            return 0.0, 0.0

        dxs = [raw_detections[k][0] - self.prev_keypoints_px[k][0] for k in shared_ids]
        dys = [raw_detections[k][1] - self.prev_keypoints_px[k][1] for k in shared_ids]

        med_dx = float(np.median(dxs))
        med_dy = float(np.median(dys))
        pan_mag = float(np.hypot(med_dx, med_dy))

        # Un paneo físico entre 2 cuadros consecutivos no supera 42 px y es coherente en todos los puntos
        if pan_mag > 42.0:
            return 0.0, 0.0

        residuals = [float(np.hypot(dx - med_dx, dy - med_dy)) for dx, dy in zip(dxs, dys)]
        coherent_count = sum(1 for r in residuals if r <= self.m1_radius_px)
        if coherent_count / float(len(shared_ids)) < 0.65:
            return 0.0, 0.0

        return med_dx, med_dy

    # =========================================================================
    # TOMAS CERRADAS - MÉTODO 2: DETECCIÓN DE HISTOGRAMAS DE COLOR
    # =========================================================================
    def check_closeup_method_2_histogram(
        self, frame_bgr: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Compara el histograma de color del fotograma actual con el anterior para
        detectar cortes de edición y evalúa la proporción de césped verde.
        Usa submuestreo ::4 en imágenes HD/FHD para reducir el tiempo en CPU de ~5.5 ms a ~0.4 ms.
        Retorna: (is_cut_detected, hist_diff, green_ratio)
        """
        sample_bgr = frame_bgr[::4, ::4] if frame_bgr.shape[0] > 360 else frame_bgr
        hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)

        # Proporción de píxeles con tono de césped (Hue entre 28 y 92 en OpenCV [0..180])
        grass_mask = cv2.inRange(hsv, np.array([28, 35, 35]), np.array([92, 255, 255]))
        green_ratio = float(np.count_nonzero(grass_mask)) / float(grass_mask.size)

        # Histograma 2D Hue-Saturation normalizado
        curr_hist = cv2.calcHist([hsv], [0, 1], None, [36, 32], [0, 180, 0, 256])
        cv2.normalize(curr_hist, curr_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_diff = 0.0
        is_cut = False
        if self.prev_hist is not None and self.c2_enabled:
            hist_diff = float(cv2.compareHist(self.prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))
            if hist_diff >= self.c2_hist_threshold:
                is_cut = True

        self.prev_hist = curr_hist
        return is_cut, hist_diff, green_ratio

    # =========================================================================
    # TOMAS CERRADAS - MÉTODO 1: EL "INTERRUPTOR" POR PÉRDIDA DE PROXIMIDAD
    # =========================================================================
    def check_closeup_method_1_proximity_switch(
        self,
        raw_detections: Dict[str, Tuple[float, float, float]],
        pan_delta: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[bool, float]:
        """
        Evalúa si el >= 80% de los puntos clave violan el umbral de proximidad al mismo
        tiempo o desaparecen de un fotograma a otro.
        Retorna: (switch_triggered, violation_ratio)
        """
        if not self.c1_enabled or not self.prev_keypoints_px:
            return False, 0.0

        # Si antes teníamos >= 5 puntos clave estables y de golpe caen a <= 1 (desaparición brusca)
        if len(self.prev_keypoints_px) >= 5 and len(raw_detections) <= 1:
            return True, 1.0

        shared_ids = [kp_id for kp_id in raw_detections if kp_id in self.prev_keypoints_px]
        if len(shared_ids) < self.c1_min_shared:
            return False, 0.0

        dx_cam, dy_cam = pan_delta
        violations = 0
        for kp_id in shared_ids:
            curr_x, curr_y, _ = raw_detections[kp_id]
            prev_x, prev_y = self.prev_keypoints_px[kp_id]
            dist = float(np.hypot(curr_x - (prev_x + dx_cam), curr_y - (prev_y + dy_cam)))
            if dist > self.m1_radius_px:
                violations += 1

        violation_ratio = float(violations) / float(len(shared_ids))
        if violation_ratio >= self.c1_violation_ratio:
            return True, violation_ratio

        return False, violation_ratio

    # =========================================================================
    # ANTI-PARPADEO - MÉTODO 1: LA VENTANA DE PROXIMIDAD EN PÍXELES
    # =========================================================================
    def apply_method_1_pixel_window(
        self,
        kp_id: str,
        curr_xy: Tuple[float, float],
        pan_delta: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[Tuple[float, float], str, float]:
        """
        Método 1: La Ventana de Proximidad en Píxeles.
        - Si en el Cuadro N-1 el punto estaba en (x_prev, y_prev), define un radio de tolerancia (ej. 10 px).
        - Si en el Cuadro N está dentro del radio, lo acepta y actualiza.
        - Si salta fuera del radio, ignora el parpadeo manteniendo el píxel anterior o
          forzándolo a quedarse en el borde de la ventana tolerable.
        """
        if kp_id not in self.prev_keypoints_px:
            return curr_xy, "M1_NEW", 0.0

        dx_cam, dy_cam = pan_delta
        prev_x = self.prev_keypoints_px[kp_id][0] + dx_cam
        prev_y = self.prev_keypoints_px[kp_id][1] + dy_cam
        curr_x, curr_y = curr_xy
        dist = float(np.hypot(curr_x - prev_x, curr_y - prev_y))

        if dist <= self.m1_radius_px:
            # Está dentro de la ventana de proximidad: se acepta la detección
            return (curr_x, curr_y), "M1_IN_WINDOW", dist

        # Está fuera de la ventana de proximidad (parpadeo de YOLO)
        if self.m1_mode == "keep_previous":
            return (prev_x, prev_y), "M1_HELD_PREV", dist

        # Forzar a quedarse en el borde de la ventana tolerable
        ux = (curr_x - prev_x) / max(dist, 1e-6)
        uy = (curr_y - prev_y) / max(dist, 1e-6)
        clamped_x = prev_x + ux * self.m1_radius_px
        clamped_y = prev_y + uy * self.m1_radius_px
        return (float(clamped_x), float(clamped_y)), "M1_CLAMPED_BORDER", dist

    # =========================================================================
    # ANTI-PARPADEO - MÉTODO 2: PROXIMIDAD POR PROYECCIÓN INVERSA
    # =========================================================================
    def apply_method_2_inverse_projection(
        self,
        kp_id: str,
        yolo_xy: Tuple[float, float],
        pan_delta: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[Tuple[float, float], str, float]:
        """
        Método 2: Proximidad por Proyección Inversa (El método profesional).
        1. Toma la posición fija en metros de PITCH_COORDS_METERS[kp_id].
        2. Usa M_anterior^-1 para calcular en qué píxel exacto debería estar el punto (x_pred, y_pred).
        3. Compara con la inferencia de YOLO (x_yolo, y_yolo) y calcula la distancia de error.
        4. Calcula un punto intermedio ponderado por proximidad (ej: pred=300, yolo=292 -> 298).
        """
        if self.M_anterior_inv is None or kp_id not in self.pitch_coords:
            return yolo_xy, "M2_NO_PRIOR_H", 0.0

        meter_x, meter_y = self.pitch_coords[kp_id]
        pred_px = pitch_to_pixel(meter_x, meter_y, self.M_anterior_inv)
        if pred_px is None:
            return yolo_xy, "M2_INV_FAIL", 0.0

        dx_cam, dy_cam = pan_delta
        pred_x = pred_px[0] + dx_cam
        pred_y = pred_px[1] + dy_cam
        yolo_x, yolo_y = yolo_xy
        err_px = float(np.hypot(yolo_x - pred_x, yolo_y - pred_y))

        if err_px > self.m2_max_error_px:
            # Si el salto de YOLO es excesivo respecto a la geometría inversa, retenemos la predicción matemática
            return (pred_x, pred_y), "M2_REJECTED_NOISE", err_px

        # Punto intermedio ponderado por proximidad:
        # Ej: pred = 300, yolo = 292 (8px error) con w_yolo = 0.25 -> 0.75 * 300 + 0.25 * 292 = 298.0
        w_yolo = float(np.clip(self.m2_yolo_weight, 0.05, 0.95))
        weighted_x = (1.0 - w_yolo) * pred_x + w_yolo * yolo_x
        weighted_y = (1.0 - w_yolo) * pred_y + w_yolo * yolo_y
        return (float(weighted_x), float(weighted_y)), "M2_INV_WEIGHTED", err_px

    # =========================================================================
    # TOMAS CERRADAS - MÉTODO 3: VALIDACIÓN GEOMÉTRICA DE LA CANCHA
    # =========================================================================
    def check_closeup_method_3_geometry(
        self,
        H: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> Tuple[bool, str, int, int, float]:
        """
        Método 3: Validación Geométrica de la Cancha (Filtro de Seguridad).
        Verifica que MAGSAC++/RANSAC no haya descartado la mayoría de puntos clave
        por inconsistencia de perspectiva (ej. 5 descartados de 8 detectados) y que
        la matriz resultante mantenga paralelismo/convexidad geométrica válida.
        Retorna: (is_geometrically_valid, reason, num_inliers, num_discarded, inlier_ratio)
        """
        num_detected = len(src_pts)
        if H is None or mask is None or num_detected < self.c3_min_inliers:
            return False, f"BANDERA ROJA: Puntos insuficientes ({num_detected} < {self.c3_min_inliers})", 0, num_detected, 0.0

        mask_flat = mask.ravel().astype(bool)
        num_inliers = int(np.sum(mask_flat))
        num_discarded = int(num_detected - num_inliers)
        inlier_ratio = float(num_inliers) / float(max(1, num_detected))

        if not self.c3_enabled:
            return True, "GEOM_VALIDATION_DISABLED", num_inliers, num_discarded, inlier_ratio

        # Regla 1: Mínimo absoluto de inliers
        if num_inliers < self.c3_min_inliers:
            return (
                False,
                f"BANDERA ROJA: Inliers insuficientes ({num_inliers}/{num_detected})",
                num_inliers,
                num_discarded,
                inlier_ratio,
            )

        # Regla 2: Proporción de inliers (Ej: detecta 8 y descarta 5 -> 3/8 = 37.5% < 50%)
        if inlier_ratio < self.c3_min_inlier_ratio:
            return (
                False,
                f"BANDERA ROJA: {num_discarded}/{num_detected} puntos descartados por MAGSAC++ (ratio={inlier_ratio:.0%})",
                num_inliers,
                num_discarded,
                inlier_ratio,
            )

        # Regla 3: No colinealidad de los inliers en la cancha (deben formar área 2D real, no una sola línea)
        inlier_dst = dst_pts[mask_flat]
        span_x = float(np.max(inlier_dst[:, 0]) - np.min(inlier_dst[:, 0]))
        span_y = float(np.max(inlier_dst[:, 1]) - np.min(inlier_dst[:, 1]))
        if span_x < 3.0 or span_y < 3.0:
            return (
                False,
                f"BANDERA ROJA: Puntos colineales sin apertura 2D (dx={span_x:.1f}m, dy={span_y:.1f}m)",
                num_inliers,
                num_discarded,
                inlier_ratio,
            )

        # Regla 4: Estabilidad algebraica y número de condición
        if np.isnan(H).any() or np.isinf(H).any():
            return False, "BANDERA ROJA: Matriz H contiene NaN/Inf", num_inliers, num_discarded, inlier_ratio

        fro = float(np.linalg.norm(H, "fro"))
        if fro < 1e-9:
            return False, "BANDERA ROJA: Matriz H nula", num_inliers, num_discarded, inlier_ratio

        H_norm = H / fro
        try:
            cond = float(np.linalg.cond(H_norm))
            det = float(np.linalg.det(H_norm))
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return False, "BANDERA ROJA: Matriz H singular", num_inliers, num_discarded, inlier_ratio

        if abs(det) < 1e-11 or cond > self.c3_max_cond:
            return (
                False,
                f"BANDERA ROJA: Cancha deformada / matriz mal condicionada (cond={cond:.1e})",
                num_inliers,
                num_discarded,
                inlier_ratio,
            )

        # Regla 5: Error de reproyección inversa en píxeles (evita la trampa de compresión de escala en metros)
        inlier_src = src_pts[mask_flat]
        proj_px = cv2.perspectiveTransform(inlier_dst.reshape(-1, 1, 2).astype(np.float32), H_inv).reshape(-1, 2)
        errs_px = np.linalg.norm(proj_px - inlier_src, axis=1)
        mean_err_px = float(np.mean(errs_px))
        max_err_px = float(np.max(errs_px))

        if mean_err_px > self.c3_max_reproj_px or max_err_px > self.c3_max_point_px:
            return (
                False,
                f"BANDERA ROJA: Desajuste en píxeles (err_medio={mean_err_px:.1f}px, max={max_err_px:.1f}px)",
                num_inliers,
                num_discarded,
                inlier_ratio,
            )

        # Regla 6: Validación Topológica del Cuadrilátero de Cancha Proyectado (Horizonte y Convexidad)
        topo_ok, topo_reason = validate_projected_pitch_topology(
            H, H_inv, inlier_dst, self.largo_cancha, self.ancho_cancha
        )
        if not topo_ok:
            return False, topo_reason, num_inliers, num_discarded, inlier_ratio

        return True, "GEOMETRIA_OK", num_inliers, num_discarded, inlier_ratio

    def _smooth_homography_control_points(
        self,
        H_cand: np.ndarray,
        H_cand_inv: np.ndarray,
        inlier_dst: np.ndarray,
        pan_delta: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Suaviza la transición entre M_anterior y H_cand en el espacio de 4 puntos de control
        sobre el plano de la cancha (en lugar de promediar matrices H directamente, lo que
        distorsiona la línea del horizonte).
        """
        if self.M_anterior_inv is None:
            return H_cand, H_cand_inv

        mx, my = self.largo_cancha / 2.0, self.ancho_cancha / 2.0
        x0 = max(0.0, float(np.min(inlier_dst[:, 0])) - 4.0)
        x1 = min(self.largo_cancha, float(np.max(inlier_dst[:, 0])) + 4.0)
        y0 = max(0.0, float(np.min(inlier_dst[:, 1])) - 4.0)
        y1 = min(self.ancho_cancha, float(np.max(inlier_dst[:, 1])) + 4.0)
        if x1 - x0 < 12.0:
            x0, x1 = mx - 16.5, mx + 16.5
        if y1 - y0 < 12.0:
            y0, y1 = my - 20.16, my + 20.16

        ctrl_m = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        try:
            prev_px = cv2.perspectiveTransform(ctrl_m.reshape(-1, 1, 2), self.M_anterior_inv).reshape(-1, 2)
            curr_px = cv2.perspectiveTransform(ctrl_m.reshape(-1, 1, 2), H_cand_inv).reshape(-1, 2)
            prev_px[:, 0] += pan_delta[0]
            prev_px[:, 1] += pan_delta[1]

            # Si las esquinas proyectadas son consistentes (< 60 px de diferencia), suavizar
            corner_diff = float(np.max(np.linalg.norm(curr_px - prev_px, axis=1)))
            if corner_diff > 60.0:
                return H_cand, H_cand_inv

            blend_px = ((1.0 - self.ema_alpha) * prev_px + self.ema_alpha * curr_px).astype(np.float32)
            H_smooth = cv2.getPerspectiveTransform(blend_px, ctrl_m)
            if abs(H_smooth[2, 2]) > 1e-9:
                H_smooth = H_smooth / H_smooth[2, 2]
            H_smooth_inv = np.linalg.inv(H_smooth)
            return H_smooth, H_smooth_inv
        except Exception:
            return H_cand, H_cand_inv

    # =========================================================================
    # PIPELINE PRINCIPAL DE PROCESAMIENTO POR FOTOGRAMA
    # =========================================================================
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        raw_keypoints: Dict[int, Tuple[float, float, float]],
        is_video: bool = True,
        precomputed_hist: Optional[Tuple[bool, float, float]] = None,
    ) -> HomographyFrameResult:
        """
        Procesa las detecciones de keypoints de un fotograma ejecutando:
        1. Detección de cortes de edición / tomas cerradas (Métodos 1 y 2).
        2. Pre-filtrado topológico de keypoints invertidos.
        3. Estabilización de keypoints contra parpadeo con compensación de paneo (Métodos 1 y 2).
        4. Cálculo de Homografía con MAGSAC++ y poda iterativa de outliers en píxeles.
        5. Validación Geométrica y Topológica de la Cancha (Método 3 - Filtro de Seguridad).
        """
        mapped_raw: Dict[str, Tuple[float, float, float, int]] = {}
        for kp_idx, (x, y, conf) in raw_keypoints.items():
            if kp_idx in KP_MAP:
                kp_id = KP_MAP[kp_idx]
                if kp_id in self.pitch_coords:
                    mapped_raw[kp_id] = (float(x), float(y), float(conf), int(kp_idx))

        # Pre-filtro topológico para descartar swaps de keypoints (ej. 14/16/17/19 en el círculo central)
        if self.c3_topological_filter:
            mapped_raw = filter_topological_outliers(mapped_raw, self.pitch_coords)

        # ── Paso 1: Verificar Tomas Cerradas / Cortes (Métodos 1 y 2) ────────
        shot_cut_hist = False
        hist_diff = 0.0
        green_ratio = 1.0
        proximity_switch = False
        pan_delta = (0.0, 0.0)

        if is_video and frame_bgr is not None:
            if precomputed_hist is not None:
                shot_cut_hist, hist_diff, green_ratio = precomputed_hist
            else:
                shot_cut_hist, hist_diff, green_ratio = self.check_closeup_method_2_histogram(frame_bgr)
            raw_for_switch = {k: (v[0], v[1], v[2]) for k, v in mapped_raw.items()}
            pan_delta = self.estimate_camera_pan_delta(raw_for_switch)
            proximity_switch, _ = self.check_closeup_method_1_proximity_switch(raw_for_switch, pan_delta=pan_delta)

            if shot_cut_hist or proximity_switch:
                self.reset_temporal_state()
                pan_delta = (0.0, 0.0)

        if is_video and green_ratio < self.c2_min_green_ratio and len(mapped_raw) < 4:
            self.reset_temporal_state()
            return HomographyFrameResult(
                H=None,
                H_inv=None,
                is_valid=False,
                is_closeup_or_cut=True,
                status_message=f"TOMA CERRADA (Cesped={green_ratio:.0%}, KPs={len(mapped_raw)})",
                red_flag_triggered=False,
                shot_cut_histogram=shot_cut_hist,
                proximity_switch_triggered=proximity_switch,
                histogram_diff=hist_diff,
                green_ratio=green_ratio,
                num_detected=len(mapped_raw),
            )

        # ── Paso 2: Estabilización Anti-Parpadeo (Métodos 1 y 2) ─────────────
        observations: List[KeypointObservation] = []
        for kp_id, (rx, ry, conf, kp_idx) in sorted(mapped_raw.items(), key=lambda item: item[1][3]):
            stab_x, stab_y = rx, ry
            note = "RAW"
            jump_val = 0.0

            if is_video:
                use_m1 = self.m1_enabled and self.stabilization_method in ("method_1", "both")
                use_m2 = self.m2_enabled and self.stabilization_method in ("method_2", "both")

                if use_m1:
                    (stab_x, stab_y), note_m1, jump_m1 = self.apply_method_1_pixel_window(
                        kp_id, (stab_x, stab_y), pan_delta=pan_delta
                    )
                    note = note_m1
                    jump_val = jump_m1

                if use_m2 and self.M_anterior_inv is not None:
                    (stab_x, stab_y), note_m2, err_m2 = self.apply_method_2_inverse_projection(
                        kp_id, (stab_x, stab_y), pan_delta=pan_delta
                    )
                    note = f"{note}+{note_m2}" if use_m1 else note_m2
                    jump_val = max(jump_val, err_m2)

            observations.append(
                KeypointObservation(
                    kp_index=kp_idx,
                    kp_id=kp_id,
                    raw_px=(rx, ry),
                    stabilized_px=(stab_x, stab_y),
                    pitch_m=self.pitch_coords[kp_id],
                    confidence=conf,
                    stabilization_note=note,
                    jump_px=jump_val,
                )
            )

        num_detected = len(observations)
        if num_detected < 4:
            if is_video and self.M_anterior is not None and not (shot_cut_hist or proximity_switch):
                return HomographyFrameResult(
                    H=self.M_anterior,
                    H_inv=self.M_anterior_inv,
                    is_valid=True,
                    is_closeup_or_cut=False,
                    status_message=f"M_ANTERIOR RETENIDA (KPs={num_detected})",
                    shot_cut_histogram=shot_cut_hist,
                    proximity_switch_triggered=proximity_switch,
                    histogram_diff=hist_diff,
                    green_ratio=green_ratio,
                    num_detected=num_detected,
                    num_inliers=num_detected,
                    inlier_ratio=1.0,
                    keypoints=observations,
                )
            return HomographyFrameResult(
                H=None,
                H_inv=None,
                is_valid=False,
                is_closeup_or_cut=True,
                status_message=f"SIN HOMOGRAFIA ({num_detected}/4 KPs)",
                shot_cut_histogram=shot_cut_hist,
                proximity_switch_triggered=proximity_switch,
                histogram_diff=hist_diff,
                green_ratio=green_ratio,
                num_detected=num_detected,
                keypoints=observations,
            )

        # ── Paso 3: Estimar Homografía con MAGSAC++ y Poda de Outliers ───────
        src_pts = np.array([obs.stabilized_px for obs in observations], dtype=np.float32)
        dst_pts = np.array([obs.pitch_m for obs in observations], dtype=np.float32)

        try:
            H_cand, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                method=HOMOGRAPHY_METHOD_MAGSAC,
                ransacReprojThreshold=self.ransac_threshold,
            )
            # Solo ejecutar el segundo MAGSAC++ inverso si la primera hipótesis falla en topología
            if H_cand is not None:
                inliers_m = mask.ravel().astype(bool) if mask is not None else np.ones(len(src_pts), dtype=bool)
                topo_m, _ = validate_projected_pitch_topology(
                    H_cand, np.linalg.inv(H_cand), dst_pts[inliers_m], self.largo_cancha, self.ancho_cancha
                )
                if not topo_m:
                    H_inv_direct, mask_px = cv2.findHomography(
                        dst_pts,
                        src_pts,
                        method=HOMOGRAPHY_METHOD_MAGSAC,
                        ransacReprojThreshold=self.c3_max_reproj_px,
                    )
                    if H_inv_direct is not None and mask_px is not None and int(np.sum(mask_px)) >= 4:
                        H_from_px = np.linalg.inv(H_inv_direct)
                        inliers_p = mask_px.ravel().astype(bool)
                        topo_p, _ = validate_projected_pitch_topology(
                            H_from_px, H_inv_direct, dst_pts[inliers_p], self.largo_cancha, self.ancho_cancha
                        )
                        if topo_p:
                            H_cand, mask = H_from_px, mask_px
        except Exception:
            H_cand, mask = None, None

        # ── Paso 4: Método 3 - Validación Geométrica y Topológica ────────────
        is_geom_ok, geom_reason, n_inliers, n_discarded, inlier_ratio = self.check_closeup_method_3_geometry(
            H_cand, mask, src_pts, dst_pts
        )

        mean_reproj_px = 0.0
        if mask is not None:
            mask_flat = mask.ravel().astype(bool)
            for idx, obs in enumerate(observations):
                if idx < len(mask_flat):
                    obs.is_inlier = bool(mask_flat[idx])

        if not is_geom_ok or H_cand is None:
            # Si falló en este cuadro pero tenemos M_anterior muy reciente y no hubo corte, mantener M_anterior por seguridad
            if is_video and self.M_anterior is not None and not (shot_cut_hist or proximity_switch):
                for obs in observations:
                    obs.is_inlier = False
                return HomographyFrameResult(
                    H=self.M_anterior,
                    H_inv=self.M_anterior_inv,
                    is_valid=True,
                    is_closeup_or_cut=False,
                    status_message=f"M_ANTERIOR PROTEGIDA ({geom_reason[:32]})",
                    red_flag_triggered=True,
                    shot_cut_histogram=shot_cut_hist,
                    proximity_switch_triggered=proximity_switch,
                    histogram_diff=hist_diff,
                    green_ratio=green_ratio,
                    num_detected=num_detected,
                    num_inliers=n_inliers,
                    num_discarded=n_discarded,
                    inlier_ratio=inlier_ratio,
                    keypoints=observations,
                )

            return HomographyFrameResult(
                H=None,
                H_inv=None,
                is_valid=False,
                is_closeup_or_cut=True,
                status_message=geom_reason,
                red_flag_triggered=True,
                shot_cut_histogram=shot_cut_hist,
                proximity_switch_triggered=proximity_switch,
                histogram_diff=hist_diff,
                green_ratio=green_ratio,
                num_detected=num_detected,
                num_inliers=n_inliers,
                num_discarded=n_discarded,
                inlier_ratio=inlier_ratio,
                keypoints=observations,
            )

        if abs(H_cand[2, 2]) > 1e-9:
            H_cand = H_cand / H_cand[2, 2]

        try:
            H_cand_inv = np.linalg.inv(H_cand)
        except np.linalg.LinAlgError:
            return HomographyFrameResult(
                H=None,
                H_inv=None,
                is_valid=False,
                is_closeup_or_cut=True,
                status_message="BANDERA ROJA: H no invertible",
                red_flag_triggered=True,
                num_detected=num_detected,
                keypoints=observations,
            )

        inlier_mask_bool = mask.ravel().astype(bool)
        inlier_dst = dst_pts[inlier_mask_bool]

        # Suavizado en espacio de puntos de control virtuales (sin deformar el horizonte)
        if is_video and self.M_anterior is not None and not (shot_cut_hist or proximity_switch):
            H_final, H_final_inv = self._smooth_homography_control_points(
                H_cand, H_cand_inv, inlier_dst, pan_delta
            )
        else:
            H_final, H_final_inv = H_cand, H_cand_inv

        # Calcular error en píxeles de cada punto respecto a H_final_inv
        reproj_all = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2), H_final_inv).reshape(-1, 2)
        errs_all = np.linalg.norm(reproj_all - src_pts, axis=1)
        for i, obs in enumerate(observations):
            obs.reproj_error_px = float(errs_all[i])
        mean_reproj_px = float(np.mean(errs_all[inlier_mask_bool]))

        # Actualizar memoria temporal (solo con keypoints inliers verificados)
        self.M_anterior = H_final
        self.M_anterior_inv = H_final_inv
        for obs in observations:
            if obs.is_inlier:
                self.prev_keypoints_px[obs.kp_id] = obs.stabilized_px

        cut_tag = ""
        if shot_cut_hist:
            cut_tag = " [CORTE HISTOGRAMA]"
        elif proximity_switch:
            cut_tag = " [INTERRUPTOR PROX.]"

        status = f"MAGSAC++ OK | Inliers: {n_inliers}/{num_detected} ({inlier_ratio:.0%}) | Err: {mean_reproj_px:.1f}px{cut_tag}"
        return HomographyFrameResult(
            H=H_final,
            H_inv=H_final_inv,
            is_valid=True,
            is_closeup_or_cut=False,
            status_message=status,
            red_flag_triggered=False,
            shot_cut_histogram=shot_cut_hist,
            proximity_switch_triggered=proximity_switch,
            histogram_diff=hist_diff,
            green_ratio=green_ratio,
            num_detected=num_detected,
            num_inliers=n_inliers,
            num_discarded=n_discarded,
            inlier_ratio=inlier_ratio,
            mean_reproj_error_px=mean_reproj_px,
            keypoints=observations,
        )
