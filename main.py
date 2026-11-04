"""
main.py — Orquestador Principal de Inferencia de Jugador, Balón, Homografía 2D (MAGSAC++)
y Seguimiento/Reconocimiento en 2 Pasadas (Kalman 2D + EasyOCR + Global Tracklet Stitching).

Uso por línea de comandos:
  # Procesar una imagen:
  python main.py --input ruta/a/imagen.jpg --output outputs/resultado.jpg

  # Procesar un video con Arquitectura de 2 Pasadas (Pasada 1 Online + Pasada 2 Cosido Global):
  python main.py --input ruta/a/video.mp4 --stabilization both --max-frames 3000
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import SoccerInferenceEngine, load_yaml_config
from src.homography import HomographyStabilizer, HomographyFrameResult, pixel_to_pitch
from src.appearance import TeamClusterer, JerseyNumberRecognizer
from src.tracker_2d import PitchBoTSORT2D
from src.tracklet_stitcher import GlobalTrackletStitcher
from src.ball_tracker import BallPhysicsTracker, BallTrajectoryReconstructor
from src.renderer import Tactical2DRenderer
from src.video_reader import ThreadedVideoReader
from src.json_exporter import reconcile_homographies_pass2, ScoutingJsonExporter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class Soccer2DTrackingPipeline:
    """
    Pipeline integral de 2 Pasadas:
      - Pasada 1 (Frame a Frame):
          1. Inferencia de Jugadores (conf=0.5), Balón (conf=0.4) y 30 Keypoints desde modelos .pt.
          2. Homografía 2D con MAGSAC++ + Estabilización (Métodos 1 y 2) + Tomas Cerradas (Métodos 1, 2 y 3).
          3. Clasificación de Equipo por color en espacio CIE-LAB sin césped (`TeamClusterer`).
          4. Reconocimiento oportunista de dorsales con EasyOCR (`JerseyNumberRecognizer`).
          5. BoT-SORT modificado con Filtro de Kalman en coordenadas 2D de cancha (`PitchBoTSORT2D`).
      - Pasada 2 (Post-procesamiento Tabular y Global Tracklet Stitching):
          1. Propaga los dorsales leídos a todo el tracklet.
          2. Detecta cortes de cámara y une fragmentos rotos usando distancia 2D + tiempo + equipo + dorsal.
          3. Renderiza el video final etiquetado con IDs globales cosidos, dorsales y el Plano 2D.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        stabilization_override: Optional[str] = None,
        conf_player_override: Optional[float] = None,
        conf_ball_override: Optional[float] = None,
        two_pass_override: Optional[bool] = None,
        imgsz_override: Optional[int] = None,
    ):
        self.config = load_yaml_config(config_path)

        if stabilization_override is not None:
            self.config.setdefault("homography", {})["stabilization_method"] = stabilization_override
        if conf_player_override is not None:
            self.config.setdefault("detection", {})["conf_player"] = float(conf_player_override)
        if conf_ball_override is not None:
            self.config.setdefault("detection", {})["conf_ball"] = float(conf_ball_override)
        if two_pass_override is not None:
            self.config.setdefault("tracking_2d", {})["enable_two_pass"] = bool(two_pass_override)
        if imgsz_override is not None:
            self.config.setdefault("detection", {})["imgsz"] = int(imgsz_override)

        self.enable_two_pass = bool(
            self.config.get("tracking_2d", {}).get("enable_two_pass", True)
        )

        self.detector = SoccerInferenceEngine(config=self.config)
        self.stabilizer = HomographyStabilizer(config=self.config)
        self.team_clusterer = TeamClusterer(config=self.config)
        self.jersey_recognizer = JerseyNumberRecognizer(
            config=self.config,
            use_gpu=("cuda" in str(self.detector.device).lower()),
        )
        self.tracker_2d = PitchBoTSORT2D(config=self.config)
        self.ball_tracker = BallPhysicsTracker(config=self.config)
        self.renderer = Tactical2DRenderer(config=self.config)
        self._last_homog_res: Optional[HomographyFrameResult] = None

    def _reset_all_states(self) -> None:
        self.stabilizer.reset_temporal_state()
        self.team_clusterer.reset()
        self.jersey_recognizer.reset()
        self.tracker_2d.reset()
        self.ball_tracker.reset()
        self.renderer.reset_trails()
        self._last_homog_res = None

    def infer_single_frame_pass1(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int = 1,
        fps: float = 25.0,
        is_video: bool = False,
    ) -> Dict[str, Any]:
        """
        Ejecuta la Pasada 1 sobre un fotograma:
        Inferencia YOLO + ROI Dinámico de Balón (conf>=0.15) + Homografía MAGSAC++
        + Equipo CIE-LAB + EasyOCR + Kalman 2D (Jugadores) + Máquina de Estados (Balón).
        Incluye:
          - Pre-chequeo ultra-rápido de histograma/césped (submuestreado ::4) para omitir
            inferencias innecesarias en tomas cerradas/intros.
          - Stride adaptativo de Keypoints cuando la homografía ya está enganchada y estable.
        """
        h_img, w_img = frame_bgr.shape[:2]

        # 0. Pre-evaluar histograma/césped (< 0.15 ms) en video para detectar cortes o tomas sin cancha
        pre_hist: Optional[tuple] = None
        shot_cut_hist = False
        green_ratio = 1.0
        if is_video:
            shot_cut_hist, hist_diff, green_ratio = self.stabilizer.check_closeup_method_2_histogram(frame_bgr)
            pre_hist = (shot_cut_hist, hist_diff, green_ratio)

            # Si casi no hay césped en pantalla (< 12%), es una toma cerrada/intro:
            # salir de inmediato sin disparar los 3 modelos en GPU/CPU.
            if green_ratio < self.stabilizer.c2_min_green_ratio:
                self.stabilizer.reset_temporal_state()
                self._last_homog_res = None
                homog_closeup = HomographyFrameResult(
                    H=None,
                    H_inv=None,
                    is_valid=False,
                    is_closeup_or_cut=True,
                    status_message=f"TOMA CERRADA (Cesped={green_ratio:.0%})",
                    shot_cut_histogram=shot_cut_hist,
                    histogram_diff=hist_diff,
                    green_ratio=green_ratio,
                )
                self.tracker_2d.update_frame(
                    frame_idx=frame_idx, fps=fps, players=[], homography_valid=False, shot_cut=True
                )
                self.ball_tracker.update(
                    frame_idx=frame_idx,
                    fps=fps,
                    candidates=[],
                    players=[],
                    homography_H=None,
                    homography_valid=False,
                    shot_cut=True,
                    roi_used=False,
                )
                return {"players": [], "balls": [], "homography": homog_closeup}

        # 1. Determinar de antemano si en este cuadro corremos Keypoints y/o Balón
        #    para despachar los motores (1 sola subida PCIe + LetterBox en GPU).
        #    Stride adaptativo: mientras la homografía tenga >= 4 inliers estables sin corte de cámara,
        #    se reutiliza H durante 3 cuadros (ejecuta KPs cada 4 cuadros), reduciendo el cómputo de GPU.
        kp_stride = 4 if fps >= 20.0 else 2
        can_reuse_homog = (
            is_video
            and not shot_cut_hist
            and self._last_homog_res is not None
            and self._last_homog_res.is_valid
            and self._last_homog_res.H is not None
            and self._last_homog_res.num_inliers >= 4
            and (frame_idx % kp_stride != 0)
        )

        roi_box: Optional[ tuple ] = None
        run_ball = True
        if is_video:
            force_full_scan = shot_cut_hist or (
                self.ball_tracker.missing_frames >= 3 and self.ball_tracker.missing_frames % 3 == 0
            )
            roi_box = None if force_full_scan else self.ball_tracker.get_dynamic_roi(w_img, h_img)
            # En estado LOST, escanear Full-Frame alternando cuadros (cada 2 frames) para ahorrar GPU
            skip_lost_full_scan = (
                roi_box is None
                and self.ball_tracker.state.value == "LOST"
                and (frame_idx % 2 == 0)
            )
            run_ball = not skip_lost_full_scan

        if is_video and getattr(self.detector, "is_cuda", False):
            players, raw_kps, ball_cands, roi_used = self.detector.infer_frame_parallel_gpu(
                frame_bgr,
                run_keypoints=(not can_reuse_homog),
                run_ball=run_ball,
                ball_conf_low=self.ball_tracker.conf_low,
                ball_roi_box=roi_box,
            )
        else:
            players = self.detector.infer_players(frame_bgr, is_video=is_video)
            raw_kps = {}
            ball_cands, roi_used = [], False

        # Si no se detectó ningún jugador en toda la imagen en video, tampoco es una toma abierta de juego:
        # evitar correr Keypoints y Balón innecesariamente.
        if is_video and len(players) == 0:
            self._last_homog_res = None
            homog_empty = HomographyFrameResult(
                H=None,
                H_inv=None,
                is_valid=False,
                is_closeup_or_cut=True,
                status_message="SIN JUGADORES EN CANCHA (0 KPs)",
                shot_cut_histogram=shot_cut_hist,
                green_ratio=green_ratio,
            )
            self.tracker_2d.update_frame(
                frame_idx=frame_idx, fps=fps, players=[], homography_valid=False, shot_cut=shot_cut_hist
            )
            self.ball_tracker.update(
                frame_idx=frame_idx,
                fps=fps,
                candidates=[],
                players=[],
                homography_H=None,
                homography_valid=False,
                shot_cut=shot_cut_hist,
                roi_used=False,
            )
            return {"players": [], "balls": [], "homography": homog_empty}

        # 2. Resolver Homografía 2D con MAGSAC++ (reutilizada o calculada desde raw_kps)
        if can_reuse_homog:
            prev_h = self._last_homog_res
            homog_res = HomographyFrameResult(
                H=prev_h.H,
                H_inv=prev_h.H_inv,
                is_valid=True,
                is_closeup_or_cut=False,
                status_message=prev_h.status_message,
                shot_cut_histogram=False,
                proximity_switch_triggered=False,
                histogram_diff=pre_hist[1] if pre_hist else 0.0,
                green_ratio=green_ratio,
                num_detected=prev_h.num_detected,
                num_inliers=prev_h.num_inliers,
                num_discarded=prev_h.num_discarded,
                inlier_ratio=prev_h.inlier_ratio,
                mean_reproj_error_px=prev_h.mean_reproj_error_px,
                keypoints=prev_h.keypoints,
            )
        else:
            if not (is_video and getattr(self.detector, "is_cuda", False)):
                player_boxes = [p["bbox"] for p in players]
                raw_kps = self.detector.infer_keypoints(frame_bgr, exclude_player_boxes=player_boxes)
            homog_res = self.stabilizer.process_frame(
                frame_bgr, raw_kps, is_video=is_video, precomputed_hist=pre_hist
            )
            self._last_homog_res = homog_res if homog_res.is_valid else None

        shot_cut = bool(homog_res.shot_cut_histogram or homog_res.proximity_switch_triggered)

        # 3. Proyección vectorizada de posiciones crudas de pies de jugadores a coordenadas 2D (metros)
        if homog_res.is_valid and homog_res.H is not None and players:
            feet_arr = np.array([[p["foot_px"]] for p in players], dtype=np.float32)
            proj_arr = cv2.perspectiveTransform(feet_arr, homog_res.H).reshape(-1, 2)
            for idx_p, p in enumerate(players):
                mx, my = float(proj_arr[idx_p, 0]), float(proj_arr[idx_p, 1])
                if np.isfinite(mx) and np.isfinite(my):
                    p["pitch_m"] = (mx, my)

        # 4. Pre-asociar IDs online (IoU en píxeles + Kalman 2D) ANTES de clasificar equipo/dorsal
        #    para que las pistas ya bloqueadas omitan el recorte y conversión HSV+LAB en CPU.
        if is_video:
            self.tracker_2d.pre_associate_frame(
                frame_idx=frame_idx,
                fps=fps,
                players=players,
                homography_valid=homog_res.is_valid,
                shot_cut=shot_cut,
            )

        # 5. Clasificación de Equipo (CIE-LAB sin césped) y Lectura Oportunista de Dorsal (EasyOCR)
        self.team_clusterer.classify_players(frame_bgr, players)
        self.jersey_recognizer.recognize_frame_jerseys(frame_bgr, players, frame_idx=frame_idx)

        # 6. Actualizar Filtro de Kalman 2D en cancha, reconexión de Apolonio y registro tabular
        if is_video:
            players = self.tracker_2d.update_frame(
                frame_idx=frame_idx,
                fps=fps,
                players=players,
                homography_valid=homog_res.is_valid,
                shot_cut=shot_cut,
            )

        # 7. Seguimiento Avanzado de Balón: ROI Dinámico + Verosimilitud (conf>=0.15) + FSM
        if is_video:
            if not getattr(self.detector, "is_cuda", False):
                if not run_ball:
                    ball_cands, roi_used = [], False
                else:
                    ball_cands, roi_used = self.detector.infer_ball_candidates(
                        frame_bgr,
                        conf_low=self.ball_tracker.conf_low,
                        roi_box=roi_box,
                    )

            tracked_ball = self.ball_tracker.update(
                frame_idx=frame_idx,
                fps=fps,
                candidates=ball_cands,
                players=players,
                homography_H=homog_res.H,
                homography_valid=homog_res.is_valid,
                shot_cut=shot_cut,
                roi_used=roi_used,
            )
            balls = [tracked_ball] if tracked_ball is not None else []
        else:
            balls = self.detector.infer_ball(frame_bgr)
            if homog_res.is_valid and homog_res.H is not None:
                for b in balls:
                    bx, by = b["center_px"]
                    b["pitch_m"] = pixel_to_pitch(bx, by, homog_res.H)

        return {
            "players": players,
            "balls": balls,
            "homography": homog_res,
        }

    def process_frame_array(
        self,
        frame_bgr: np.ndarray,
        is_video: bool = False,
        frame_idx: int = 1,
        fps: float = 25.0,
    ) -> Dict[str, Any]:
        """Procesa un fotograma BGR y devuelve la imagen compuesta (cámara etiquetada + plano 2D)."""
        step = self.infer_single_frame_pass1(
            frame_bgr, frame_idx=frame_idx, fps=fps, is_video=is_video
        )
        players = step["players"]
        balls = step["balls"]
        homog_res = step["homography"]

        stab_mode = self.stabilizer.stabilization_method
        annotated_cam = self.renderer.draw_camera_annotations(
            frame_bgr, players, balls, homog_res, stabilization_mode=stab_mode
        )
        plane_2d = self.renderer.render_2d_plane(
            players, balls, homog_res, update_trails=is_video
        )
        combined_frame = self.renderer.compose_side_by_side(annotated_cam, plane_2d)

        return {
            "combined_frame": combined_frame,
            "annotated_camera": annotated_cam,
            "plane_2d": plane_2d,
            "players": players,
            "balls": balls,
            "homography": homog_res,
        }

    def process_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Procesa un archivo de imagen y guarda la imagen etiquetada con su plano 2D."""
        in_path = Path(image_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {in_path}")

        frame_bgr = cv2.imread(str(in_path))
        if frame_bgr is None:
            raise ValueError(f"No se pudo decodificar la imagen: {in_path}")

        out_path = self._resolve_output_path(in_path, output_path, is_video=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self._reset_all_states()
        res = self.process_frame_array(frame_bgr, is_video=False)
        cv2.imwrite(str(out_path), res["combined_frame"])

        homog: Any = res["homography"]
        print("=" * 72)
        print("✅ IMAGEN PROCESADA EXITOSAMENTE")
        print(f"   Entrada:            {in_path}")
        print(f"   Salida Etiquetada:  {out_path}")
        print(f"   Jugadores (c>=0.5): {len(res['players'])} detectados")
        print(f"   Balón (c>=0.4):     {len(res['balls'])} detectado(s)")
        print(f"   Keypoints:          {homog.num_detected} detectados | {homog.num_inliers} inliers MAGSAC++")
        print(f"   Estado Homografía:  {homog.status_message}")
        print("=" * 72)
        return out_path

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        render_video: bool = True,
    ) -> Path:
        """
        Procesa un archivo de video (restringido por defecto a 3000 frames) ejecutando:
          - Pasada 1: Inferencia frame a frame + Homografía MAGSAC++ + Kalman 2D + Equipo + OCR.
          - Pasada 2: Sinceramiento Homográfico + Global Tracklet Stitching + Reconstrucción Balón
                      + Exportación JSON (`match_metadata.json`, `tracking_output.json`) + Renderizado.
        """
        in_path = Path(video_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {in_path}")

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {in_path}")

        if max_frames is None:
            cfg_max = self.config.get("video", {}).get("max_frames")
            max_frames = int(cfg_max) if (cfg_max is not None and int(cfg_max) > 0) else None
        elif int(max_frames) <= 0:
            max_frames = None

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        target_frames = (
            min(total_frames, max_frames)
            if (max_frames and total_frames > 0)
            else (max_frames or total_frames)
        )

        out_path = self._resolve_output_path(in_path, output_path, is_video=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._reset_all_states()

        print("=" * 72)
        print("🚀 INICIANDO PIPELINE DE VIDEO (2 PASADAS: KALMAN 2D + GLOBAL STITCHING)")
        print(f"   Video Entrada:      {in_path.name} ({target_frames} cuadros @ {fps:.1f} FPS)")
        print(f"   Método Estabiliz.:  {self.stabilizer.stabilization_method.upper()}")
        print(f"   Confianza Jugador:  {self.detector.conf_player:.2f} | Balón: {self.detector.conf_ball:.2f}")
        print(f"   Modo 2 Pasadas:     {'ACTIVADO (Pasada 1 + Pasada 2 Tabular)' if self.enable_two_pass else 'DESACTIVADO (1 Pasada)'}")
        print("=" * 72)

        # Si el usuario desactiva explícitamente las 2 pasadas, procesar en 1 sola pasada directa
        if not self.enable_two_pass:
            return self._process_video_single_pass(cap, out_path, target_frames, fps, max_frames)

        cap.release()

        # =====================================================================
        # PASADA 1: Inferencia Frame a Frame + Kalman 2D + Equipo + Dorsal OCR
        # =====================================================================
        print("▶ [PASADA 1/2] Inferencia Frame a Frame + Homografía 2D + Kalman 2D + OCR (Asíncrono)...")
        frame_cache_meta: Dict[int, Dict[str, Any]] = {}
        t_pass1 = time.time()
        frame_idx = 0

        # Lector asíncrono con buffer multihilo en RAM (384 cuadros ~ 2.4 GB RAM, ideal para los 53 GB de Colab)
        reader = ThreadedVideoReader(in_path, queue_size=384, max_frames=max_frames)

        try:
            for frame_idx, frame_bgr in reader:
                step = self.infer_single_frame_pass1(
                    frame_bgr, frame_idx=frame_idx, fps=fps, is_video=True
                )
                frame_cache_meta[frame_idx] = {
                    "balls": step["balls"],
                    "homography": step["homography"],
                }

                if frame_idx % 10 == 0 or frame_idx == target_frames:
                    elapsed = max(1e-6, time.time() - t_pass1)
                    curr_fps = frame_idx / elapsed
                    pct = (frame_idx / max(1, target_frames)) * 100.0
                    print(
                        f"\r   [Pasada 1] {frame_idx}/{target_frames} ({pct:5.1f}%) | "
                        f"{curr_fps:4.1f} FPS | {step['homography'].status_message[:40]}",
                        end="",
                        flush=True,
                    )
        finally:
            reader.release()
            print()

        pass1_elapsed = time.time() - t_pass1

        # =====================================================================
        # PASADA 2: Sinceramiento Homográfico + Cosido Global + Balón + JSONs
        # =====================================================================
        print("▶ [PASADA 2/2] Ejecutando Sinceramiento Homográfico, Global Tracklet Stitching y Balón...")
        t_stitch = time.time()

        # 2.1 Sincerar Homografía en huecos cortos entre planos válidos antes del cosido
        homog_reconcile_stats = reconcile_homographies_pass2(
            frame_cache_meta=frame_cache_meta,
            raw_player_records=self.tracker_2d.tabular_records,
            raw_ball_records=self.ball_tracker.tabular_records,
            max_gap_frames=20,
        )

        # 2.2 Cosido Global de Jugadores (con región cinemática de Apolonio)
        stitcher = GlobalTrackletStitcher(config=self.config, fps=fps)
        raw_records = self.tracker_2d.tabular_records
        stitched_records, stitch_stats = stitcher.stitch_and_propagate(raw_records)
        stitched_by_frame = stitcher.build_frame_lookup(stitched_records)

        # 2.3 Reconstrucción Post-Inferencia de la Traza Continua del Balón
        ball_reconstructor = BallTrajectoryReconstructor(config=self.config, fps=fps)
        reconstructed_ball_rows, ball_stats = ball_reconstructor.reconstruct(
            self.ball_tracker.tabular_records,
            stitched_players_by_frame=stitched_by_frame,
        )
        reconstructed_balls_by_frame = ball_reconstructor.build_frame_lookup(reconstructed_ball_rows)

        raw_csv_path = out_path.with_name(f"{out_path.stem}_raw_tracks.csv")
        stitched_csv_path = out_path.with_name(f"{out_path.stem}_stitched_tracks.csv")
        ball_csv_path = out_path.with_name(f"{out_path.stem}_ball_trajectory.csv")
        stitcher.export_to_csv(raw_records, raw_csv_path)
        stitcher.export_to_csv(stitched_records, stitched_csv_path)
        ball_reconstructor.export_to_csv(reconstructed_ball_rows, ball_csv_path)
        stitch_elapsed = time.time() - t_stitch

        print(
            f"   ✓ Homografía sincerada: {homog_reconcile_stats['valid_pass1']} válidas Pasada 1 + "
            f"{homog_reconcile_stats['interpolated_pass2']} interpoladas Pasada 2 = "
            f"{homog_reconcile_stats['total_valid_pass2']} cuadros con matriz 3x3 activa."
        )
        print(
            f"   ✓ Jugadores cosidos en {stitch_elapsed:.2f}s: "
            f"{stitch_stats['raw_tracklets']} fragmentos -> {stitch_stats['stitched_players']} identidades globales "
            f"({stitch_stats['merges_performed']} uniones, {stitch_stats['jerseys_assigned']} dorsales propagados)."
        )
        print(
            f"   ✓ Balón reconstruido: {ball_stats['detected_frames']} detectados + "
            f"{ball_stats['interpolated_frames']} reconstruidos = {ball_stats['active_ball_frames']}/{ball_stats['total_frames']} cuadros activos "
            f"({ball_stats['spikes_removed']} picos espurios eliminados)."
        )

        # Renderizado del video final con los IDs globales cosidos y la traza continua del balón
        render_elapsed = 0.0
        if render_video:
            print("▶ Renderizando video final con trayectorias 2D cosidas y traza continua de balón...")
            reader_render = ThreadedVideoReader(in_path, queue_size=256, max_frames=frame_idx)
            writer = None
            render_idx = 0
            self.renderer.reset_trails()
            stab_mode = self.stabilizer.stabilization_method
            t_rend = time.time()

            try:
                for render_idx, frame_bgr in reader_render:
                    meta = frame_cache_meta.get(render_idx)
                    if meta is None:
                        continue

                    players_stitched = stitched_by_frame.get(render_idx, [])
                    balls_reconstructed = reconstructed_balls_by_frame.get(render_idx, meta["balls"])
                    homog_res = meta["homography"]

                    annotated_cam = self.renderer.draw_camera_annotations(
                        frame_bgr, players_stitched, balls_reconstructed, homog_res, stabilization_mode=stab_mode
                    )
                    plane_2d = self.renderer.render_2d_plane(
                        players_stitched, balls_reconstructed, homog_res, update_trails=True
                    )
                    combined = self.renderer.compose_side_by_side(annotated_cam, plane_2d)

                    if writer is None:
                        h_out, w_out = combined.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_out, h_out))

                    writer.write(combined)

                    if render_idx % 20 == 0 or render_idx == frame_idx:
                        pct = (render_idx / max(1, frame_idx)) * 100.0
                        print(f"\r   [Render 2D] {render_idx}/{frame_idx} ({pct:5.1f}%)", end="", flush=True)
            finally:
                reader_render.release()
                if writer is not None:
                    writer.release()
                print()
            render_elapsed = time.time() - t_rend

        total_t = time.time() - t_pass1

        # 2.4 Exportar Contratos JSON para `scoutingData_FrontEnd`
        print("▶ Exportando contratos JSON (`match_metadata.json` y `tracking_output.json`)...")
        model_player_cfg = str(self.config.get("models", {}).get("player", "models/modelPlayer.pt"))
        model_fmt = Path(model_player_cfg).suffix.lstrip(".") or "pt"
        dev_str = str(getattr(self.detector, "device", "cuda"))
        json_exporter = ScoutingJsonExporter(config=self.config)

        # Guardar tanto en subcarpeta `<out_dir>/<video_stem>/` como en `<out_dir>/`
        json_subdir = out_path.parent / in_path.stem
        exported_jsons = json_exporter.export_all(
            video_path=in_path,
            output_dir=json_subdir,
            fps=fps,
            total_frames=frame_idx,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_cache_meta=frame_cache_meta,
            stitched_by_frame=stitched_by_frame,
            reconstructed_balls_by_frame=reconstructed_balls_by_frame,
            team_bgr_palette=self.team_clusterer.team_bgr_palette,
            stitch_stats=stitch_stats,
            ball_stats=ball_stats,
            homog_reconcile_stats=homog_reconcile_stats,
            timings={
                "total_s": total_t,
                "pass1_s": pass1_elapsed,
                "pass2_s": stitch_elapsed,
                "render_s": render_elapsed,
            },
            device_str=dev_str,
            model_format=model_fmt,
            cuda_available=("cuda" in dev_str.lower()),
        )

        # Copiar también en el directorio raíz de salida (`outputs/match_metadata.json` y `outputs/tracking_output.json`)
        import shutil
        for k_name, src_json_path in exported_jsons.items():
            dst_root_path = out_path.parent / src_json_path.name
            if dst_root_path.resolve() != src_json_path.resolve():
                shutil.copy2(src_json_path, dst_root_path)

        print("=" * 72)
        print("✅ VIDEO, COSIDO GLOBAL Y EXPORTACIÓN JSON COMPLETADOS EXITOSAMENTE")
        if render_video:
            print(f"   Video Etiquetado:   {out_path}")
        print(f"   JSON Metadata:      {exported_jsons['match_metadata']}")
        print(f"   JSON Inferencias:   {exported_jsons['tracking_output']}")
        print(f"   JSON Identidades:   {exported_jsons['identity_manifest']}")
        print(f"   CSV Pasada 1 Crudo: {raw_csv_path}")
        print(f"   CSV Pasada 2 Cosido:{stitched_csv_path}")
        print(f"   CSV Traza Balón:    {ball_csv_path}")
        print(f"   Cuadros Procesados: {frame_idx} en {total_t:.1f}s")
        print("=" * 72)
        return out_path

    def _process_video_single_pass(
        self,
        cap: cv2.VideoCapture,
        out_path: Path,
        target_frames: int,
        fps: float,
        max_frames: Optional[int],
    ) -> Path:
        writer = None
        frame_idx = 0
        t0 = time.time()
        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break
                if max_frames and frame_idx >= max_frames:
                    break
                frame_idx += 1
                res = self.process_frame_array(frame_bgr, is_video=True, frame_idx=frame_idx, fps=fps)
                combined = res["combined_frame"]
                if writer is None:
                    h_out, w_out = combined.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_out, h_out))
                writer.write(combined)
                if frame_idx % 10 == 0 or frame_idx == target_frames:
                    elapsed = max(1e-6, time.time() - t0)
                    curr_fps = frame_idx / elapsed
                    pct = (frame_idx / max(1, target_frames)) * 100.0
                    print(
                        f"\r   Progreso: {frame_idx}/{target_frames} ({pct:5.1f}%) | {curr_fps:4.1f} FPS",
                        end="",
                        flush=True,
                    )
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            print()
        return out_path

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
        render_video: bool = True,
    ) -> Path:
        """Detecta automáticamente si la entrada es imagen o video y ejecuta el procesamiento."""
        in_path = Path(input_path)
        ext = in_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return self.process_image(in_path, output_path=output_path)
        if ext in VIDEO_EXTENSIONS:
            return self.process_video(
                in_path,
                output_path=output_path,
                max_frames=max_frames,
                render_video=render_video,
            )
        raise ValueError(
            f"Formato de archivo no soportado ({ext}). Use imagen {sorted(IMAGE_EXTENSIONS)} o video {sorted(VIDEO_EXTENSIONS)}."
        )

    def _resolve_output_path(
        self,
        in_path: Path,
        output_path: Optional[Union[str, Path]],
        is_video: bool,
    ) -> Path:
        if output_path is not None:
            return Path(output_path)
        out_dir_name = self.config.get("visualization", {}).get("output_dir", "outputs")
        out_dir = PROJECT_ROOT / out_dir_name
        ext = ".mp4" if is_video else in_path.suffix
        stab = self.stabilizer.stabilization_method
        return out_dir / f"{in_path.stem}_labeled_{stab}{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia de Jugador (conf=0.5), Balón (conf=0.4), Homografía 2D (MAGSAC++) y Cosido Global en 2 Pasadas"
    )
    parser.add_argument(
        "input_pos",
        nargs="?",
        default=None,
        help="Ruta a la imagen o video de entrada (posicional opcional)",
    )
    parser.add_argument(
        "--input", "-i", "--image", "--video",
        dest="input_flag",
        default=None,
        help="Ruta a la imagen o video de entrada",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Ruta de salida para la imagen o video etiquetado",
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/config.yaml",
        help="Ruta al archivo de configuración .yaml (por defecto: configs/config.yaml)",
    )
    parser.add_argument(
        "--stabilization", "-s",
        choices=["method_1", "method_2", "both", "none"],
        default=None,
        help="Método anti-parpadeo de homografía a utilizar (method_1: Ventana Píxeles, method_2: Proyección Inversa, both: Ambos)",
    )
    parser.add_argument(
        "--conf-player",
        type=float,
        default=None,
        help="Umbral de confianza para jugador (por defecto: 0.5 en YAML)",
    )
    parser.add_argument(
        "--conf-ball",
        type=float,
        default=None,
        help="Umbral de confianza para balón (por defecto: 0.4 en YAML)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Tamaño de entrada para inferencia en píxeles (por defecto: 1280 en YAML o auto-detectado del modelo)",
    )
    parser.add_argument(
        "--max-frames", "-m",
        type=int,
        default=None,
        help="Límite máximo de fotogramas a procesar en video (por defecto: sin límite, procesa todo el video)",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        default=False,
        help="Ejecutar en 1 sola pasada sin el post-procesamiento tabular de la Pasada 2",
    )
    parser.add_argument(
        "--no-render-video",
        action="store_true",
        default=False,
        help="Omitir el codificado del video MP4 y exportar únicamente los JSONs y CSVs sincerados",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_source = args.input_flag or args.input_pos
    if not input_source:
        raise SystemExit(
            "Error: Debe proporcionar una imagen o video de entrada.\n"
            "Ejemplo: python main.py --input ruta/a/imagen_o_video.jpg"
        )

    pipeline = Soccer2DTrackingPipeline(
        config_path=args.config,
        stabilization_override=args.stabilization,
        conf_player_override=args.conf_player,
        conf_ball_override=args.conf_ball,
        two_pass_override=(False if args.single_pass else None),
        imgsz_override=args.imgsz,
    )
    pipeline.process(
        input_path=input_source,
        output_path=args.output,
        max_frames=args.max_frames,
        render_video=(not args.no_render_video),
    )


if __name__ == "__main__":
    main()
