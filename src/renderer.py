"""
Módulo de Visualización y Plano 2D Táctico.
Dibuja sobre la imagen o video:
  - Inferencias de Jugadores (bbox, elipse en pies, ID detectado, rol y conf).
  - Inferencias de Balón (bbox, marcador, ID detectado y conf).
  - Inferencias de Keypoints de Homografía (ID de KP_MAP, conf, estado inlier/outlier).
  - Proyección de las líneas de la cancha sobre la cámara mediante M^-1.
  - Plano 2D (Minimapa táctico a escala) con las posiciones y trayectorias 2D de jugadores y balón.
"""

from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from src.pitch_config import (
    LARGO_CANCHA,
    ANCHO_CANCHA,
    build_pitch_coords_meters,
    get_pitch_line_segments,
)
from src.homography import HomographyFrameResult
from src.spatial_geometry import compute_voronoi_and_apollonius_surface


class Tactical2DRenderer:
    """
    Renderizador de inferencias en cámara y plano 2D de movimientos de jugadores y balón.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        pitch_cfg = cfg.get("pitch", {})
        vis_cfg = cfg.get("visualization", {})

        self.largo_cancha = float(pitch_cfg.get("largo_cancha", LARGO_CANCHA))
        self.ancho_cancha = float(pitch_cfg.get("ancho_cancha", ANCHO_CANCHA))
        self.scale = float(vis_cfg.get("minimap_scale", 7.0))
        self.padding_m = float(vis_cfg.get("minimap_padding_m", 5.0))
        self.trail_length = int(vis_cfg.get("trail_length", 35))
        self.clean_view = bool(vis_cfg.get("clean_view", True))
        self.draw_projected_lines = bool(vis_cfg.get("draw_projected_lines", not self.clean_view))
        self.draw_keypoints = bool(vis_cfg.get("draw_keypoints", True))
        self.draw_hud_banner = bool(vis_cfg.get("draw_hud_banner", not self.clean_view))
        self.draw_camera_frustum = bool(vis_cfg.get("draw_camera_frustum", True))
        self.enable_voronoi_apollonius = bool(vis_cfg.get("enable_voronoi_apollonius", False))
        self.spatial_overlay_alpha = float(vis_cfg.get("spatial_overlay_alpha", 0.22))

        self.pitch_coords = build_pitch_coords_meters(self.largo_cancha, self.ancho_cancha)
        self.pitch_segments = get_pitch_line_segments(self.largo_cancha, self.ancho_cancha)
        self.base_pitch_2d, self.pad_px = self._create_2d_pitch_template()

        # Historial de trayectorias 2D (en metros) y en cámara para videos
        self.player_trails_2d: Dict[int, deque] = {}
        self.ball_trail_2d: deque = deque(maxlen=self.trail_length)
        self.ball_trail_cam: deque = deque(maxlen=20)

    def reset_trails(self) -> None:
        """Reinicia las estelas de movimiento 2D cuando hay cambio de toma."""
        self.player_trails_2d.clear()
        self.ball_trail_2d.clear()
        self.ball_trail_cam.clear()

    def _m_to_px(self, xm: float, ym: float) -> Tuple[int, int]:
        """Convierte coordenadas en metros (0..105, 0..68) a píxeles del plano 2D."""
        px = int(round(xm * self.scale)) + self.pad_px
        py = int(round(ym * self.scale)) + self.pad_px
        return px, py

    def _create_2d_pitch_template(self) -> Tuple[np.ndarray, int]:
        """Construye la plantilla base de la cancha 2D con sus líneas y los 30 keypoints."""
        pad_px = int(round(self.padding_m * self.scale))
        w_pitch = int(round(self.largo_cancha * self.scale))
        h_pitch = int(round(self.ancho_cancha * self.scale))
        canvas_w = w_pitch + 2 * pad_px
        canvas_h = h_pitch + 2 * pad_px

        canvas = np.full((canvas_h, canvas_w, 3), (32, 105, 32), dtype=np.uint8)

        # Franjas de césped estilizadas
        num_stripes = 10
        stripe_w = w_pitch / num_stripes
        for i in range(num_stripes):
            if i % 2 == 0:
                x1 = pad_px + int(round(i * stripe_w))
                x2 = pad_px + int(round((i + 1) * stripe_w))
                canvas[pad_px : pad_px + h_pitch, x1:x2] = (38, 118, 38)

        line_col = (240, 240, 240)
        L, W = self.largo_cancha, self.ancho_cancha
        mx, my = L / 2.0, W / 2.0

        def to_p(x_m: float, y_m: float) -> Tuple[int, int]:
            return int(round(x_m * self.scale)) + pad_px, int(round(y_m * self.scale)) + pad_px

        # Contorno y línea central
        cv2.rectangle(canvas, to_p(0, 0), to_p(L, W), line_col, 2, lineType=cv2.LINE_AA)
        cv2.line(canvas, to_p(mx, 0), to_p(mx, W), line_col, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, to_p(mx, my), int(round(9.15 * self.scale)), line_col, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, to_p(mx, my), 3, line_col, -1, lineType=cv2.LINE_AA)

        # Áreas grandes
        cv2.rectangle(canvas, to_p(0, my - 20.16), to_p(16.5, my + 20.16), line_col, 2, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, to_p(L - 16.5, my - 20.16), to_p(L, my + 20.16), line_col, 2, lineType=cv2.LINE_AA)

        # Áreas chicas
        cv2.rectangle(canvas, to_p(0, my - 9.16), to_p(5.5, my + 9.16), line_col, 2, lineType=cv2.LINE_AA)
        cv2.rectangle(canvas, to_p(L - 5.5, my - 9.16), to_p(L, my + 9.16), line_col, 2, lineType=cv2.LINE_AA)

        # Puntos de penalti y arcos
        cv2.circle(canvas, to_p(11.0, my), 3, line_col, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, to_p(L - 11.0, my), 3, line_col, -1, lineType=cv2.LINE_AA)
        r_arc = int(round(9.15 * self.scale))
        cv2.ellipse(canvas, to_p(11.0, my), (r_arc, r_arc), 0, -53, 53, line_col, 2, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, to_p(L - 11.0, my), (r_arc, r_arc), 0, 127, 233, line_col, 2, lineType=cv2.LINE_AA)

        # Dibujar suavemente la ubicación de los 30 keypoints de referencia
        for kp_id, (kx, ky) in self.pitch_coords.items():
            pt = to_p(kx, ky)
            cv2.circle(canvas, pt, 3, (120, 180, 120), -1, lineType=cv2.LINE_AA)

        return canvas, pad_px

    def draw_projected_pitch_lines(self, frame_bgr: np.ndarray, H_inv: Optional[np.ndarray]) -> None:
        """Proyecta las líneas canónicas 2D sobre la imagen de la cámara usando M^-1."""
        if H_inv is None or not self.draw_projected_lines:
            return

        h_img, w_img = frame_bgr.shape[:2]
        for seg in self.pitch_segments:
            pts = seg.reshape(-1, 2)
            homo = np.column_stack([pts, np.ones(len(pts), dtype=np.float32)])
            cam = homo @ H_inv.T
            w_coords = cam[:, 2]
            valid = w_coords > 0.05
            if not np.any(valid):
                continue

            px_x = cam[:, 0] / np.maximum(w_coords, 1e-6)
            px_y = cam[:, 1] / np.maximum(w_coords, 1e-6)

            poly: List[List[int]] = []
            for i in range(len(pts)):
                if valid[i] and -2500 < px_x[i] < (w_img + 2500) and -2500 < px_y[i] < (h_img + 2500):
                    poly.append([int(round(px_x[i])), int(round(px_y[i]))])
                else:
                    if len(poly) >= 2:
                        cv2.polylines(
                            frame_bgr,
                            [np.array(poly, dtype=np.int32)],
                            isClosed=False,
                            color=(255, 255, 0),
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )
                    poly = []
            if len(poly) >= 2:
                cv2.polylines(
                    frame_bgr,
                    [np.array(poly, dtype=np.int32)],
                    isClosed=False,
                    color=(255, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

    def draw_camera_annotations(
        self,
        frame_bgr: np.ndarray,
        players: List[Dict[str, Any]],
        balls: List[Dict[str, Any]],
        homog_res: HomographyFrameResult,
        stabilization_mode: str = "both",
    ) -> np.ndarray:
        """
        Dibuja todas las inferencias sobre la imagen de entrada:
        - Líneas de cancha proyectadas (si homografía es válida)
        - Keypoints con ID ('01'..'32') y conf
        - Jugadores con bbox, elipse en pies, ID detectado, rol y conf
        - Balón con bbox, marcador, ID detectado y conf
        - HUD de estado de homografía y seguridad
        """
        out = frame_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Líneas de cancha proyectadas si la homografía es válida
        if homog_res.is_valid and homog_res.H_inv is not None:
            self.draw_projected_pitch_lines(out, homog_res.H_inv)

        # 2. Keypoints detectados con ID numérico limpio
        if self.draw_keypoints:
            for obs in homog_res.keypoints:
                if not obs.is_inlier and obs.confidence < 0.25:
                    continue
                sx, sy = int(round(obs.stabilized_px[0])), int(round(obs.stabilized_px[1]))
                if not (0 <= sx < out.shape[1] and 0 <= sy < out.shape[0]):
                    continue

                kp_col = (0, 255, 255) if obs.is_inlier else (0, 180, 255)

                # Círculo amarillo nítido con borde negro de contraste
                cv2.circle(out, (sx, sy), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(out, (sx, sy), 4, kp_col, -1, lineType=cv2.LINE_AA)

                kp_label = str(obs.kp_id)
                (tw, th), _ = cv2.getTextSize(kp_label, font, 0.40, 1)
                bx1, by1 = max(2, sx - tw // 2 - 3), max(2, sy - th - 8)
                bx2, by2 = bx1 + tw + 6, by1 + th + 4
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (18, 24, 38), -1)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), kp_col, 1)
                cv2.putText(out, kp_label, (bx1 + 3, by2 - 2), font, 0.40, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # 3. Jugadores detectados (solo IDs detectados y visualización limpia)
        for p in players:
            x1, y1, x2, y2 = [int(round(v)) for v in p["bbox"]]
            col = tuple(p.get("color_bgr", (255, 180, 0)))

            # Bounding box nítido de 2 px
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2, lineType=cv2.LINE_AA)

            jersey_num = p.get("jersey_number")
            id_text = f"#{jersey_num}" if jersey_num else str(p["id"])

            if self.clean_view:
                # Pastilla sólida minimalista con solo el ID detectado
                label = id_text
                font_scale = 0.44
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
                bx1 = x1
                by2 = max(th + 6, y1 - 3)
                by1 = by2 - th - 5
                bx2 = bx1 + tw + 8

                cv2.rectangle(out, (bx1, by1), (bx2, by2), col, -1, lineType=cv2.LINE_AA)
                cv2.putText(out, label, (bx1 + 4, by2 - 3), font, font_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            else:
                fx, fy = int(round(p["foot_px"][0])), int(round(p["foot_px"][1]))
                axes = (max(6, int((x2 - x1) * 0.45)), max(4, int((y2 - y1) * 0.10)))
                cv2.ellipse(out, (fx, fy), axes, 0, 0, 360, col, 2, lineType=cv2.LINE_AA)
                jersey_tag = f" [#{jersey_num}]" if jersey_num else ""
                label = f"ID:{p['id']}{jersey_tag} {p['role']} {p['conf']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
                bx1 = max(2, fx - tw // 2 - 4)
                by2 = max(th + 6, y1 - 3)
                by1 = by2 - th - 6
                bx2 = bx1 + tw + 8
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), col, 1, lineType=cv2.LINE_AA)
                cv2.putText(out, label, (bx1 + 4, by2 - 4), font, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # 4. Balón detectado o reconstruido
        if homog_res.shot_cut_histogram or homog_res.proximity_switch_triggered:
            self.ball_trail_cam.clear()

        for b in balls:
            cx, cy = int(round(b["center_px"][0])), int(round(b["center_px"][1]))
            is_interp = bool(b.get("is_interpolated", False))
            b_col = (0, 215, 255) if is_interp else (0, 140, 255)

            # Solo añadir a la estela de cámara si es una detección real o reconstrucción en vuelo
            # (evita dibujar líneas erráticas hacia los pies de jugadores en oclusiones especulativas)
            if not (is_interp and b.get("state") == "OCCLUDED_BY_PLAYER"):
                if len(self.ball_trail_cam) >= 1:
                    last_c = self.ball_trail_cam[-1]
                    jump_px = float(np.hypot(cx - last_c[0], cy - last_c[1]))
                    if jump_px > 85.0:
                        self.ball_trail_cam.clear()
                self.ball_trail_cam.append((cx, cy))

            if len(self.ball_trail_cam) >= 2:
                c_pts = np.array(list(self.ball_trail_cam), dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [c_pts], isClosed=False, color=b_col, thickness=2, lineType=cv2.LINE_AA)

            if self.clean_view:
                # Marcador limpio de balón: círculos concéntricos y pastilla compacta
                cv2.circle(out, (cx, cy), 6, b_col, -1, lineType=cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                label = "Ball"
                (tw, th), _ = cv2.getTextSize(label, font, 0.38, 1)
                bx1 = max(2, cx - tw // 2 - 3)
                by2 = max(th + 6, cy - 10)
                by1 = by2 - th - 4
                bx2 = bx1 + tw + 6
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (20, 20, 20), -1, lineType=cv2.LINE_AA)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), b_col, 1, lineType=cv2.LINE_AA)
                cv2.putText(out, label, (bx1 + 3, by2 - 3), font, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            else:
                x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
                cv2.rectangle(out, (x1, y1), (x2, y2), b_col, 2, lineType=cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 6, b_col, -1, lineType=cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                state_tag = " [REC]" if is_interp else ""
                b_label = f"Ball ID:{b['id']}{state_tag} {b['conf']:.2f}"
                (tw, th), _ = cv2.getTextSize(b_label, font, 0.45, 1)
                bx1 = max(2, cx - tw // 2 - 4)
                by2 = max(th + 6, y1 - 4)
                by1 = by2 - th - 6
                bx2 = bx1 + tw + 8
                cv2.rectangle(out, (bx1, by1), (bx2, by2), (15, 15, 15), -1)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), b_col, 1, lineType=cv2.LINE_AA)
                cv2.putText(out, b_label, (bx1 + 4, by2 - 4), font, 0.45, (0, 230, 255), 1, lineType=cv2.LINE_AA)

        # 5. Banner superior de estado (solo en modo diagnóstico, omitido en clean_view)
        if not self.clean_view and self.draw_hud_banner:
            status_col = (0, 210, 80) if homog_res.is_valid else (0, 50, 255)
            hud_text = (
                f"Estab: {stabilization_mode.upper()} | {homog_res.status_message} | "
                f"Players(>=0.5): {len(players)} | Ball(>=0.4): {len(balls)}"
            )
            (hud_w, hud_h), _ = cv2.getTextSize(hud_text, font, 0.58, 2)
            box_w = min(out.shape[1] - 10, hud_w + 30)
            cv2.rectangle(out, (10, 10), (box_w, 48), (15, 20, 30), -1)
            cv2.rectangle(out, (10, 10), (box_w, 48), status_col, 2)
            cv2.putText(out, hud_text, (20, 35), font, 0.58, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        return out

    def compute_camera_frustum_polygon(
        self,
        H: Optional[np.ndarray],
        cam_w: int = 1920,
        cam_h: int = 1080,
    ) -> Optional[np.ndarray]:
        """
        Calcula el polígono 2D en píxeles del canvas de la cancha que representa
        el área proyectada visible por la cámara según la homografía (frustum de cámara).
        """
        if H is None:
            return None

        sample_pts = []
        for u in np.linspace(0.0, float(cam_w), 25):
            sample_pts.append((u, float(cam_h - 2)))
        for v in np.linspace(float(cam_h - 2), 2.0, 20):
            sample_pts.append((float(cam_w - 2), v))
        for u in np.linspace(float(cam_w - 2), 0.0, 25):
            sample_pts.append((u, 2.0))
        for v in np.linspace(2.0, float(cam_h - 2), 20):
            sample_pts.append((2.0, v))

        canvas_pts = []
        margin_m = 6.0
        for u, v in sample_pts:
            P = H @ np.array([u, v, 1.0], dtype=np.float64)
            w = float(P[2])
            if w <= 1e-4:
                continue
            xm = float(P[0] / w)
            ym = float(P[1] / w)
            if (-margin_m <= xm <= self.largo_cancha + margin_m) and (-margin_m <= ym <= self.ancho_cancha + margin_m):
                px, py = self._m_to_px(xm, ym)
                canvas_pts.append([px, py])

        if len(canvas_pts) < 4:
            return None

        hull = cv2.convexHull(np.array(canvas_pts, dtype=np.int32))
        return hull.reshape(-1, 2)

    def render_2d_plane(
        self,
        players: List[Dict[str, Any]],
        balls: List[Dict[str, Any]],
        homog_res: HomographyFrameResult,
        update_trails: bool = True,
    ) -> np.ndarray:
        """
        Genera la imagen del Plano 2D de la cancha con el área proyectada de la cámara
        (trapezoide de cobertura visual) y las posiciones y movimientos limpios de jugadores y balón.
        """
        plane = self.base_pitch_2d.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        if homog_res.shot_cut_histogram or homog_res.proximity_switch_triggered:
            self.reset_trails()

        # 1. Base del Plano 2D: Área Proyectada de la Cámara (Frustum según Homografía)
        if self.draw_camera_frustum and homog_res.is_valid and homog_res.H is not None:
            frustum_poly = self.compute_camera_frustum_polygon(homog_res.H)
            if frustum_poly is not None and len(frustum_poly) >= 3:
                # Cancha oscurecida para la zona exterior fuera de la vista de la cámara
                plane_dark = (self.base_pitch_2d.astype(np.float32) * 0.36).astype(np.uint8)

                # Mantener las líneas de la cancha suavemente visibles en la zona exterior
                pad = self.pad_px
                L, W_m = self.largo_cancha, self.ancho_cancha
                line_dim = (90, 115, 90)
                def to_p(x_m, y_m):
                    return int(round(x_m * self.scale)) + pad, int(round(y_m * self.scale)) + pad

                cv2.rectangle(plane_dark, to_p(0, 0), to_p(L, W_m), line_dim, 1, lineType=cv2.LINE_AA)
                cv2.line(plane_dark, to_p(L/2, 0), to_p(L/2, W_m), line_dim, 1, lineType=cv2.LINE_AA)
                cv2.circle(plane_dark, to_p(L/2, W_m/2), int(round(9.15 * self.scale)), line_dim, 1, lineType=cv2.LINE_AA)
                cv2.rectangle(plane_dark, to_p(0, W_m/2 - 20.16), to_p(16.5, W_m/2 + 20.16), line_dim, 1, lineType=cv2.LINE_AA)
                cv2.rectangle(plane_dark, to_p(L - 16.5, W_m/2 - 20.16), to_p(L, W_m/2 + 20.16), line_dim, 1, lineType=cv2.LINE_AA)

                # Máscara del área proyectada
                mask = np.zeros(plane.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [frustum_poly], 255)

                # Combinar: zona visible activa (brillante) + zona exterior (oscurecida)
                plane = np.where(mask[..., None] == 255, self.base_pitch_2d, plane_dark)

                # Borde nítido y fino del área proyectada (tal cual la segunda imagen de referencia)
                cv2.polylines(plane, [frustum_poly], isClosed=True, color=(225, 245, 225), thickness=2, lineType=cv2.LINE_AA)

        # 2. Resaltar keypoints sobre el plano 2D (inliers y detectados)
        if self.draw_keypoints:
            for obs in homog_res.keypoints:
                if not obs.is_inlier and obs.confidence < 0.25:
                    continue
                if obs.pitch_m is None:
                    continue
                kx, ky = self._m_to_px(obs.pitch_m[0], obs.pitch_m[1])
                col = (0, 255, 255) if obs.is_inlier else (0, 180, 255)
                cv2.circle(plane, (kx, ky), 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(plane, (kx, ky), 4, col, -1, lineType=cv2.LINE_AA)
                cv2.putText(plane, str(obs.kp_id), (kx + 6, ky - 2), font, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # Actualizar y dibujar trayectorias, vectores de velocidad de Apolonio y posiciones 2D de los jugadores
        for p in players:
            pm = p.get("pitch_m")
            if pm is None:
                continue
            mx, my = pm
            # Filtrar proyecciones fuera del margen razonable de la cancha
            if not (-4.0 <= mx <= self.largo_cancha + 4.0 and -4.0 <= my <= self.ancho_cancha + 4.0):
                continue

            px, py = self._m_to_px(mx, my)
            pid = int(p["id"])
            col = tuple(p.get("color_bgr", (255, 190, 0)))

            if update_trails:
                if pid not in self.player_trails_2d:
                    self.player_trails_2d[pid] = deque(maxlen=self.trail_length)
                self.player_trails_2d[pid].append((px, py))

            trail = self.player_trails_2d.get(pid)
            if trail and len(trail) >= 2:
                pts = np.array(list(trail), dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(plane, [pts], isClosed=False, color=col, thickness=2, lineType=cv2.LINE_AA)

            # Dibujar vector de velocidad (flecha cinemática de Apolonio) si el jugador está en movimiento
            vx_m = float(p.get("vx_mps", 0.0))
            vy_m = float(p.get("vy_mps", 0.0))
            spd_m = float(np.hypot(vx_m, vy_m))
            if spd_m >= 1.2:
                tip_x, tip_y = self._m_to_px(mx + vx_m * 0.65, my + vy_m * 0.65)
                cv2.arrowedLine(
                    plane,
                    (px, py),
                    (tip_x, tip_y),
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                    0,
                    0.32,
                )

            cv2.circle(plane, (px, py), 8, col, -1, lineType=cv2.LINE_AA)
            cv2.circle(plane, (px, py), 9, (15, 15, 15), 1, lineType=cv2.LINE_AA)

            jersey_num = p.get("jersey_number")
            id_txt = f"#{jersey_num}" if jersey_num else str(pid)
            (tw, th), _ = cv2.getTextSize(id_txt, font, 0.35, 1)
            cv2.putText(
                plane,
                id_txt,
                (px - tw // 2, py + th // 2),
                font,
                0.35,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        # Actualizar y dibujar trayectoria y posición 2D del balón (con física 3D y altura)
        if balls:
            best_ball = balls[0]
            bm = best_ball.get("pitch_m")
            if bm is not None:
                bmx, bmy = bm
                if -4.0 <= bmx <= self.largo_cancha + 4.0 and -4.0 <= bmy <= self.ancho_cancha + 4.0:
                    bpx, bpy = self._m_to_px(bmx, bmy)
                    height_m = float(best_ball.get("height_m", 0.0))
                    flight_phase = str(best_ball.get("flight_phase", "GROUND_ROLL"))

                    if update_trails:
                        if len(self.ball_trail_2d) >= 1:
                            last_2d = self.ball_trail_2d[-1]
                            jump_2d = float(np.hypot(bpx - last_2d[0], bpy - last_2d[1]))
                            if jump_2d > (15.0 * self.scale):
                                self.ball_trail_2d.clear()
                        self.ball_trail_2d.append((bpx, bpy))

                    if len(self.ball_trail_2d) >= 2:
                        bpts = np.array(list(self.ball_trail_2d), dtype=np.int32).reshape(-1, 1, 2)
                        trail_col = (0, 215, 255) if height_m > 0.4 else (0, 165, 255)
                        cv2.polylines(plane, [bpts], isClosed=False, color=trail_col, thickness=2, lineType=cv2.LINE_AA)

                    if height_m > 0.4:
                        # 1. Sombra real sobre el césped
                        cv2.ellipse(plane, (bpx, bpy), (7, 4), 0, 0, 360, (20, 50, 20), -1, lineType=cv2.LINE_AA)
                        # 2. Vástago vertical de altitud
                        stem_px = min(45, int(round(height_m * 4.2)))
                        ball_draw_pt = (bpx, bpy - stem_px)
                        cv2.line(plane, (bpx, bpy), ball_draw_pt, (200, 230, 255), 1, lineType=cv2.LINE_AA)
                        # 3. Marcador aéreo con escala de radio por altura
                        r_ball = 6 + min(3, int(round(height_m * 0.4)))
                        cv2.circle(plane, ball_draw_pt, r_ball, (0, 220, 255), -1, lineType=cv2.LINE_AA)
                        cv2.circle(plane, ball_draw_pt, r_ball + 2, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                        alt_txt = f"{height_m:.1f}m"
                        if flight_phase == "CURVED_SHOT":
                            alt_txt += " (Comba)"
                        cv2.putText(plane, alt_txt, (bpx + 9, bpy - stem_px + 4), font, 0.32, (240, 255, 240), 1, lineType=cv2.LINE_AA)
                    else:
                        cv2.circle(plane, (bpx, bpy), 6, (0, 140, 255), -1, lineType=cv2.LINE_AA)
                        cv2.circle(plane, (bpx, bpy), 8, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        # Si se levantó bandera roja o es toma cerrada, indicarlo claramente en el plano 2D
        if not homog_res.is_valid:
            msg = "BANDERA ROJA / TOMA CERRADA" if homog_res.red_flag_triggered else "SIN HOMOGRAFIA VALIDA"
            cv2.rectangle(plane, (10, 10), (plane.shape[1] - 10, 38), (0, 0, 160), -1)
            cv2.putText(plane, msg, (18, 29), font, 0.50, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        return plane

    @staticmethod
    def compose_side_by_side(camera_annotated: np.ndarray, plane_2d: np.ndarray) -> np.ndarray:
        """
        Une horizontalmente el fotograma etiquetado de la cámara con el Plano 2D de la cancha
        ajustando la altura del plano 2D proporcionalmente.
        """
        h_cam, w_cam = camera_annotated.shape[:2]
        h_2d, w_2d = plane_2d.shape[:2]
        new_w_2d = int(round(w_2d * (float(h_cam) / float(h_2d))))
        resized_2d = cv2.resize(plane_2d, (new_w_2d, h_cam), interpolation=cv2.INTER_AREA)
        return np.hstack([camera_annotated, resized_2d])

