"""
main.py — Orquestador Principal de Inferencia de Jugador, Balón y Homografía 2D (MAGSAC++).

Uso por línea de comandos:
  # Procesar una imagen:
  python main.py --input ruta/a/imagen.jpg --output outputs/resultado.jpg

  # Procesar un video probando Método 1 (Ventana de Proximidad en Píxeles):
  python main.py --input ruta/a/video.mp4 --stabilization method_1

  # Procesar un video probando Método 2 (Proximidad por Proyección Inversa):
  python main.py --input ruta/a/video.mp4 --stabilization method_2

  # Procesar combinando ambos métodos y los 3 filtros de tomas cerradas:
  python main.py --input ruta/a/video.mp4 --stabilization both
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import SoccerInferenceEngine, load_yaml_config
from src.homography import HomographyStabilizer, pixel_to_pitch
from src.renderer import Tactical2DRenderer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class Soccer2DTrackingPipeline:
    """
    Pipeline integral que ejecuta:
      1. Detección de Jugadores (conf=0.5), Balón (conf=0.4) y 30 Keypoints desde modelos .pt.
      2. Homografía con MAGSAC++ + Estabilización (Método 1 y Método 2) + Tomas Cerradas (Métodos 1, 2 y 3).
      3. Proyección al plano 2D en metros (105m x 68m) de los movimientos de jugadores y balón.
      4. Etiquetado visual con inferencias, ID detectado, conf y plano 2D.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        stabilization_override: Optional[str] = None,
        conf_player_override: Optional[float] = None,
        conf_ball_override: Optional[float] = None,
    ):
        self.config = load_yaml_config(config_path)

        if stabilization_override is not None:
            self.config.setdefault("homography", {})["stabilization_method"] = stabilization_override
        if conf_player_override is not None:
            self.config.setdefault("detection", {})["conf_player"] = float(conf_player_override)
        if conf_ball_override is not None:
            self.config.setdefault("detection", {})["conf_ball"] = float(conf_ball_override)

        self.detector = SoccerInferenceEngine(config=self.config)
        self.stabilizer = HomographyStabilizer(config=self.config)
        self.renderer = Tactical2DRenderer(config=self.config)

    def process_frame_array(
        self,
        frame_bgr: np.ndarray,
        is_video: bool = False,
    ) -> Dict[str, Any]:
        """
        Procesa un fotograma BGR y devuelve la imagen compuesta (cámara etiquetada + plano 2D)
        junto con los datos estructurados de las inferencias y la matriz de homografía.
        """
        # 1. Inferencia de Jugadores (conf = 0.5) y Balón (conf = 0.4)
        players = self.detector.infer_players(frame_bgr, is_video=is_video)
        balls = self.detector.infer_ball(frame_bgr)

        # Extraer bboxes de jugadores para filtrar keypoints falsos sobre camisetas/medias
        player_boxes = [p["bbox"] for p in players]

        # 2. Inferencia de los 30 Keypoints de Homografía
        raw_kps = self.detector.infer_keypoints(frame_bgr, exclude_player_boxes=player_boxes)

        # 3. Estimación y Estabilización de Homografía con MAGSAC++
        homog_res = self.stabilizer.process_frame(frame_bgr, raw_kps, is_video=is_video)

        # 4. Proyectar posiciones de Jugadores y Balón al Plano 2D en metros (si H es válida)
        if homog_res.is_valid and homog_res.H is not None:
            for p in players:
                fx, fy = p["foot_px"]
                p["pitch_m"] = pixel_to_pitch(fx, fy, homog_res.H)
            for b in balls:
                bx, by = b["center_px"]
                b["pitch_m"] = pixel_to_pitch(bx, by, homog_res.H)

        # 5. Renderizar cámara etiquetada (inferencias, ID, conf) y Plano 2D de movimientos
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

        self.stabilizer.reset_temporal_state()
        self.renderer.reset_trails()

        res = self.process_frame_array(frame_bgr, is_video=False)
        cv2.imwrite(str(out_path), res["combined_frame"])

        homog: Any = res["homography"]
        print("=" * 70)
        print("✅ IMAGEN PROCESADA EXITOSAMENTE")
        print(f"   Entrada:            {in_path}")
        print(f"   Salida Etiquetada:  {out_path}")
        print(f"   Jugadores (c>=0.5): {len(res['players'])} detectados")
        print(f"   Balón (c>=0.4):     {len(res['balls'])} detectado(s)")
        print(f"   Keypoints:          {homog.num_detected} detectados | {homog.num_inliers} inliers MAGSAC++")
        print(f"   Estado Homografía:  {homog.status_message}")
        print("=" * 70)
        return out_path

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Procesa un archivo de video cuadro a cuadro (restringido por defecto a 3000 frames) y devuelve el video etiquetado con el plano 2D."""
        in_path = Path(video_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {in_path}")

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {in_path}")

        if max_frames is None:
            max_frames = int(self.config.get("video", {}).get("max_frames", 3000))

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frames = min(total_frames, max_frames) if (max_frames and total_frames > 0) else (max_frames or total_frames)

        out_path = self._resolve_output_path(in_path, output_path, is_video=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.stabilizer.reset_temporal_state()
        self.renderer.reset_trails()

        writer = None
        frame_idx = 0
        t0 = time.time()

        print("=" * 70)
        print("🚀 INICIANDO PROCESAMIENTO DE VIDEO (JUGADOR + BALÓN + HOMOGRAFÍA 2D)")
        print(f"   Video Entrada:      {in_path.name} ({target_frames} cuadros @ {fps:.1f} FPS)")
        print(f"   Método Estabiliz.:  {self.stabilizer.stabilization_method.upper()}")
        print(f"   Confianza Jugador:  {self.detector.conf_player:.2f} | Balón: {self.detector.conf_ball:.2f}")
        print("=" * 70)

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    break
                if max_frames and frame_idx >= max_frames:
                    break

                frame_idx += 1
                res = self.process_frame_array(frame_bgr, is_video=True)
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
                        f"\r   Progreso: {frame_idx}/{target_frames} ({pct:5.1f}%) | "
                        f"{curr_fps:4.1f} FPS | {res['homography'].status_message[:42]}",
                        end="",
                        flush=True,
                    )
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            print()

        total_t = time.time() - t0
        print("=" * 70)
        print("✅ VIDEO PROCESADO EXITOSAMENTE")
        print(f"   Video Etiquetado:   {out_path}")
        print(f"   Cuadros Procesados: {frame_idx} en {total_t:.1f}s")
        print("=" * 70)
        return out_path

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Detecta automáticamente si la entrada es imagen o video y ejecuta el procesamiento."""
        in_path = Path(input_path)
        ext = in_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return self.process_image(in_path, output_path=output_path)
        if ext in VIDEO_EXTENSIONS:
            return self.process_video(in_path, output_path=output_path, max_frames=max_frames)
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
        description="Inferencia de Jugador (conf=0.5), Balón (conf=0.4) y Homografía 2D con MAGSAC++"
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
        "--max-frames", "-m",
        type=int,
        default=3000,
        help="Límite máximo de fotogramas a procesar en video (por defecto: 3000 frames)",
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
    )
    pipeline.process(
        input_path=input_source,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

