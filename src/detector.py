"""
Módulo de Inferencia Deep Learning (Ultralytics YOLO .pt).
Carga los modelos .pt definidos en el archivo .yaml:
  - Jugador (conf = 0.5)
  - Balón (conf = 0.4)
  - Homografía / 30 Keypoints (YOLO Pose)
"""

import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import cv2
import numpy as np
import yaml

warnings.filterwarnings("ignore")
try:
    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
    ULTRALYTICS_LOGGER.setLevel(logging.ERROR)
except Exception:
    pass

import torch
import torch.nn.functional as F
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carga el archivo de configuración .yaml del proyecto."""
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute() and not config_path.exists():
            config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración YAML no encontrado: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_jersey_bgr(frame_bgr: np.ndarray, bbox: List[float]) -> Tuple[int, int, int]:
    """
    Extrae el color representativo BGR de la camiseta del jugador (tercio superior-medio del bbox)
    descartando píxeles de césped verde para colorear su marcador en el plano 2D.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 + 0.25 * bw))
    cx2 = min(w, int(x2 - 0.25 * bw))
    cy1 = max(0, int(y1 + 0.18 * bh))
    cy2 = min(h, int(y1 + 0.50 * bh))

    if cx2 <= cx1 or cy2 <= cy1:
        return (255, 190, 0)

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.cvtColor_BGR2HSV if hasattr(cv2, "cvtColor_BGR2HSV") else cv2.COLOR_BGR2HSV)
    # Excluir verde césped
    grass = cv2.inRange(hsv, np.array([28, 35, 35]), np.array([92, 255, 255]))
    non_grass = crop[grass == 0]
    if len(non_grass) < 10:
        mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    else:
        mean_bgr = np.median(non_grass, axis=0)
    return int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])


class SoccerInferenceEngine:
    """
    Motor de inferencia unificado para Jugadores (conf=0.5), Balón (conf=0.4) y 30 Keypoints (.pt).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        self.config = config if config is not None else load_yaml_config(config_path)

        # Seleccionar dispositivo (CUDA o CPU)
        req_device = str(self.config.get("device", "auto")).lower()
        if req_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif "cuda" in req_device and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = req_device

        self.half = bool(self.config.get("half", True)) and ("cuda" in self.device)

        # Resolver rutas a los modelos .pt desde el .yaml
        models_cfg = self.config.get("models", {})
        self.player_model_path = self._resolve_model_path(models_cfg.get("player", "models/modelPlayer.pt"))
        self.ball_model_path = self._resolve_model_path(models_cfg.get("ball", "models/modelBall.pt"))
        self.kp_model_path = self._resolve_model_path(models_cfg.get("keypoints", "models/modelKeypoint.pt"))

        # Umbrales de confianza (Jugador = 0.5, Balón = 0.4, Keypoint = 0.35)
        det_cfg = self.config.get("detection", {})
        self.conf_player = float(det_cfg.get("conf_player", 0.50))
        self.conf_ball = float(det_cfg.get("conf_ball", 0.40))
        self.conf_keypoint = float(det_cfg.get("conf_keypoint", 0.35))
        self.imgsz = int(det_cfg.get("imgsz", 640))
        self.player_classes = det_cfg.get("player_classes", [1, 2, 3])
        self.class_names = det_cfg.get("class_names", {0: "Ball", 1: "GK", 2: "Player", 3: "Referee"})
        tracker_rel = det_cfg.get("tracker_config", "configs/botsort_sports.yaml")
        tracker_cand = PROJECT_ROOT / tracker_rel
        self.tracker_config = str(tracker_cand) if tracker_cand.exists() else "botsort.yaml"

        # Cargar los 3 modelos (.pt o .engine TensorRT)
        self.player_model = self._load_model(self.player_model_path, task="detect")
        self.ball_model = self._load_model(self.ball_model_path, task="detect")
        self.keypoint_model = self._load_model(self.kp_model_path, task="pose")

        # Inicializar CUDA Streams paralelos y pre-calentar motores TensorRT/CUDA en VRAM
        self.is_cuda = ("cuda" in self.device) and torch.cuda.is_available()
        if self.is_cuda:
            self.stream_player = torch.cuda.Stream()
            self.stream_kp = torch.cuda.Stream()
            self.stream_ball = torch.cuda.Stream()
            self._gpu_pool = ThreadPoolExecutor(max_workers=3)
            self._warmup_models()
        else:
            self.stream_player = None
            self.stream_kp = None
            self.stream_ball = None
            self._gpu_pool = None

    def _warmup_models(self) -> None:
        """
        Pre-calienta (Warmup) los 3 modelos en GPU durante la inicialización para que
        los contextos de ejecución de TensorRT (.engine) se alojen en VRAM antes del Frame 1.
        """
        try:
            dummy = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device, dtype=torch.float32)
            self.player_model.predict(
                source=dummy, imgsz=self.imgsz, conf=self.conf_player, classes=self.player_classes,
                device=self.device, half=self.half, verbose=False
            )
            self.keypoint_model.predict(
                source=dummy, imgsz=self.imgsz, conf=self.conf_keypoint,
                device=self.device, half=self.half, verbose=False
            )
            self.ball_model.predict(
                source=dummy, imgsz=self.imgsz, conf=self.conf_ball,
                device=self.device, half=self.half, verbose=False
            )
            torch.cuda.synchronize()
        except Exception:
            pass

    @staticmethod
    def _resolve_model_path(rel_or_abs: Union[str, Path]) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute() and p.exists():
            return p
        cand = PROJECT_ROOT / p
        if cand.exists():
            return cand
        raise FileNotFoundError(f"Modelo (.pt/.engine) no encontrado en: {cand}")

    def _load_model(self, path: Path, task: str) -> YOLO:
        model = YOLO(str(path), task=task)
        # Solo aplicar .to(device) si es formato PyTorch (.pt);
        # los motores TensorRT (.engine) manejan internamente los buffers en CUDA.
        if path.suffix.lower() == ".pt" and "cuda" in self.device:
            try:
                model.to(self.device)
            except Exception:
                pass
        return model

    def _upload_frame_to_gpu(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Sube el fotograma BGR uint8 a VRAM una única vez por cuadro y lo convierte a
        tensor RGB float32 normalizado [0.0, 1.0] de forma (1, 3, H, W) directamente en GPU.
        """
        t_bgr = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
        # BGR -> RGB en GPU, HWC -> BCHW y normalización /255.0
        t_rgb = t_bgr[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        t_rgb.mul_(1.0 / 255.0)
        return t_rgb

    @staticmethod
    def _gpu_letterbox(
        gpu_rgb_bchw: torch.Tensor,
        imgsz: int = 640,
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        Ejecuta LetterBox (resize bilinear manteniendo aspecto + padding gris 114/255)
        íntegramente en la GPU en < 0.12 ms, evitando cv2.resize en CPU.
        Retorna: (tensor_bchw_640x640, scale, pad_x, pad_y)
        """
        _, _, h, w = gpu_rgb_bchw.shape
        scale = min(float(imgsz) / float(h), float(imgsz) / float(w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if new_w != w or new_h != h:
            resized = F.interpolate(
                gpu_rgb_bchw, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        else:
            resized = gpu_rgb_bchw

        dw = (imgsz - new_w) / 2.0
        dh = (imgsz - new_h) / 2.0
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padded = F.pad(resized, (left, right, top, bottom), mode="constant", value=114.0 / 255.0)
        return padded.contiguous(), scale, float(left), float(top)

    def infer_frame_parallel_gpu(
        self,
        frame_bgr: np.ndarray,
        run_keypoints: bool = True,
        run_ball: bool = True,
        ball_conf_low: float = 0.15,
        ball_roi_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Tuple[float, float, float]], List[Dict[str, Any]], bool]:
        """
        Super-pipeline de inferencia en GPU (NVIDIA L4):
        1. Sube `frame_bgr` por PCIe a VRAM una sola vez por fotograma.
        2. Calcula el tensor `LetterBox` 640x640 en GPU una sola vez y lo comparte entre
           `player_model`, `keypoint_model` y `ball_model` (o recorta el ROI de balón directo en VRAM).
        3. Ejecuta los 3 motores TensorRT/PyTorch en paralelo sobre 3 `torch.cuda.Stream` concurrentes.
        Retorna: (players, raw_kps, ball_cands, roi_used)
        """
        if not self.is_cuda or self._gpu_pool is None:
            players = self.infer_players(frame_bgr, is_video=True)
            if not players:
                return [], {}, [], False
            raw_kps = (
                self.infer_keypoints(frame_bgr, exclude_player_boxes=[p["bbox"] for p in players])
                if run_keypoints
                else {}
            )
            if run_ball:
                ball_cands, roi_used = self.infer_ball_candidates(
                    frame_bgr, conf_low=ball_conf_low, roi_box=ball_roi_box
                )
            else:
                ball_cands, roi_used = [], False
            return players, raw_kps, ball_cands, roi_used

        h_orig, w_orig = frame_bgr.shape[:2]
        gpu_rgb = self._upload_frame_to_gpu(frame_bgr)
        full_padded, scale, pad_x, pad_y = self._gpu_letterbox(gpu_rgb, imgsz=self.imgsz)

        # Preparar tensor de entrada para el balón (ROI en VRAM o reutilizar full_padded)
        roi_used = False
        b_scale, b_pad_x, b_pad_y = scale, pad_x, pad_y
        rx1, ry1, rx2, ry2 = 0, 0, w_orig, h_orig
        ball_input_tensor = full_padded

        if run_ball and ball_roi_box is not None:
            rx1, ry1, rx2, ry2 = ball_roi_box
            if (rx2 - rx1) > 8 and (ry2 - ry1) > 8:
                gpu_roi = gpu_rgb[:, :, ry1:ry2, rx1:rx2]
                ball_input_tensor, b_scale, b_pad_x, b_pad_y = self._gpu_letterbox(
                    gpu_roi, imgsz=self.imgsz
                )
                roi_used = True

        # Sincronizar el stream por defecto antes de abrir las 3 ramas paralelas
        prep_event = torch.cuda.Event()
        prep_event.record()

        def _task_players():
            with torch.cuda.stream(self.stream_player):
                self.stream_player.wait_event(prep_event)
                res = self.player_model.predict(
                    source=full_padded,
                    imgsz=self.imgsz,
                    conf=self.conf_player,
                    classes=self.player_classes,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
                self.stream_player.synchronize()
                return res

        def _task_keypoints():
            with torch.cuda.stream(self.stream_kp):
                self.stream_kp.wait_event(prep_event)
                res = self.keypoint_model.predict(
                    source=full_padded,
                    imgsz=self.imgsz,
                    conf=self.conf_keypoint,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
                self.stream_kp.synchronize()
                return res

        def _task_ball():
            with torch.cuda.stream(self.stream_ball):
                self.stream_ball.wait_event(prep_event)
                res = self.ball_model.predict(
                    source=ball_input_tensor,
                    imgsz=self.imgsz,
                    conf=float(ball_conf_low),
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
                self.stream_ball.synchronize()
                return res

        fut_p = self._gpu_pool.submit(_task_players)
        fut_kp = self._gpu_pool.submit(_task_keypoints) if run_keypoints else None
        fut_b = self._gpu_pool.submit(_task_ball) if run_ball else None

        res_p = fut_p.result()
        res_kp = fut_kp.result() if fut_kp is not None else None
        res_b = fut_b.result() if fut_b is not None else None

        # 1. Decodificar jugadores (deshaciendo LetterBox a coordenadas originales w_orig x h_orig)
        players: List[Dict[str, Any]] = []
        if res_p.boxes is not None and len(res_p.boxes) > 0:
            boxes_net = res_p.boxes.xyxy.cpu().numpy()
            confs = res_p.boxes.conf.cpu().numpy()
            classes = res_p.boxes.cls.cpu().numpy().astype(int)

            for idx, (box, score, cls_id) in enumerate(zip(boxes_net, confs, classes)):
                if float(score) < self.conf_player:
                    continue
                x1 = float(np.clip((box[0] - pad_x) / scale, 0.0, float(w_orig)))
                y1 = float(np.clip((box[1] - pad_y) / scale, 0.0, float(h_orig)))
                x2 = float(np.clip((box[2] - pad_x) / scale, 0.0, float(w_orig)))
                y2 = float(np.clip((box[3] - pad_y) / scale, 0.0, float(h_orig)))
                foot_x = (x1 + x2) / 2.0
                foot_y = y2
                role = self.class_names.get(int(cls_id), self.class_names.get(str(cls_id), f"Cls{cls_id}"))
                color_bgr = (0, 215, 255) if int(cls_id) == 3 else (255, 190, 0)

                players.append({
                    "id": idx + 1,
                    "class_id": int(cls_id),
                    "role": role,
                    "conf": float(score),
                    "bbox": [x1, y1, x2, y2],
                    "foot_px": (foot_x, foot_y),
                    "pitch_m": None,
                    "color_bgr": color_bgr,
                })

        if not players:
            return [], {}, [], False

        # 2. Decodificar keypoints (aplicando filtro de exclusión sobre cajas de jugadores)
        raw_kps: Dict[int, Tuple[float, float, float]] = {}
        if res_kp is not None and res_kp.keypoints is not None and res_kp.keypoints.xy is not None and len(res_kp.keypoints.xy) > 0:
            kps_xy_net = res_kp.keypoints.xy[0].cpu().numpy()
            kps_conf = (
                res_kp.keypoints.conf[0].cpu().numpy()
                if res_kp.keypoints.conf is not None
                else np.ones(len(kps_xy_net), dtype=np.float32)
            )
            margin = 3.0
            player_boxes = [p["bbox"] for p in players]

            for idx, ((kx_net, ky_net), kconf) in enumerate(zip(kps_xy_net, kps_conf)):
                score = float(kconf)
                if score < self.conf_keypoint:
                    continue
                kx = float((kx_net - pad_x) / scale)
                ky = float((ky_net - pad_y) / scale)
                if kx <= margin or kx >= (w_orig - margin) or ky <= margin or ky >= (h_orig - margin):
                    continue

                inside_player = False
                for bx1, by1, bx2, by2 in player_boxes:
                    if bx1 <= kx <= bx2 and by1 <= ky <= (by2 - 4.0):
                        inside_player = True
                        break
                if inside_player:
                    continue

                raw_kps[int(idx)] = (kx, ky, score)

        # 3. Decodificar candidatos de balón
        ball_cands: List[Dict[str, Any]] = []
        if res_b is not None and res_b.boxes is not None and len(res_b.boxes) > 0:
            boxes_b = res_b.boxes.xyxy.cpu().numpy()
            confs_b = res_b.boxes.conf.cpu().numpy()
            classes_b = res_b.boxes.cls.cpu().numpy().astype(int)
            order = np.argsort(-confs_b)
            roi_w = float(rx2 - rx1)
            roi_h = float(ry2 - ry1)

            for rank, idx in enumerate(order):
                score = float(confs_b[idx])
                if score < ball_conf_low:
                    continue
                b_box = boxes_b[idx]
                lx1 = float(np.clip((b_box[0] - b_pad_x) / b_scale, 0.0, roi_w)) + rx1
                ly1 = float(np.clip((b_box[1] - b_pad_y) / b_scale, 0.0, roi_h)) + ry1
                lx2 = float(np.clip((b_box[2] - b_pad_x) / b_scale, 0.0, roi_w)) + rx1
                ly2 = float(np.clip((b_box[3] - b_pad_y) / b_scale, 0.0, roi_h)) + ry1
                ball_cands.append({
                    "id": rank + 1,
                    "class_id": int(classes_b[idx]),
                    "role": "Ball",
                    "conf": score,
                    "bbox": [lx1, ly1, lx2, ly2],
                    "center_px": ((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0),
                    "pitch_m": None,
                    "from_roi": roi_used,
                })

        return players, raw_kps, ball_cands, roi_used

    def infer_players(
        self,
        frame_bgr: np.ndarray,
        is_video: bool = False,
        conf: Optional[float] = None,
        use_internal_tracker: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta detección de jugadores con conf=0.5 (por defecto).
        Por defecto usa `predict()` y delega el asociación multicriterio a `PitchBoTSORT2D`,
        eliminando la sobrecarga en CPU del tracker interno de Ultralytics.
        """
        c_thresh = float(conf if conf is not None else self.conf_player)
        if is_video and use_internal_tracker:
            try:
                res = self.player_model.track(
                    source=frame_bgr,
                    conf=c_thresh,
                    persist=True,
                    tracker=self.tracker_config,
                    classes=self.player_classes,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
            except Exception:
                res = self.player_model.predict(
                    source=frame_bgr,
                    conf=c_thresh,
                    classes=self.player_classes,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
        else:
            res = self.player_model.predict(
                source=frame_bgr,
                conf=c_thresh,
                classes=self.player_classes,
                device=self.device,
                half=self.half,
                verbose=False,
            )[0]

        players: List[Dict[str, Any]] = []
        if res.boxes is None or len(res.boxes) == 0:
            return players

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            res.boxes.id.cpu().numpy().astype(int)
            if res.boxes.id is not None
            else None
        )

        for idx, (box, score, cls_id) in enumerate(zip(boxes, confs, classes)):
            if float(score) < c_thresh:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            det_id = int(track_ids[idx]) if track_ids is not None else (idx + 1)
            role = self.class_names.get(int(cls_id), self.class_names.get(str(cls_id), f"Cls{cls_id}"))
            color_bgr = (0, 215, 255) if int(cls_id) == 3 else (255, 190, 0)

            players.append({
                "id": det_id,
                "class_id": int(cls_id),
                "role": role,
                "conf": float(score),
                "bbox": [x1, y1, x2, y2],
                "foot_px": (foot_x, foot_y),
                "pitch_m": None,
                "color_bgr": color_bgr,
            })

        return players

    def infer_ball(
        self,
        frame_bgr: np.ndarray,
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta detección de balón con conf=0.4 (por defecto).
        Retorna las detecciones de balón ordenadas por confianza descendente.
        """
        c_thresh = float(conf if conf is not None else self.conf_ball)
        res = self.ball_model.predict(
            source=frame_bgr,
            conf=c_thresh,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        balls: List[Dict[str, Any]] = []
        if res.boxes is None or len(res.boxes) == 0:
            return balls

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        order = np.argsort(-confs)

        for rank, idx in enumerate(order):
            score = float(confs[idx])
            if score < c_thresh:
                continue
            x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            balls.append({
                "id": rank + 1,
                "class_id": int(classes[idx]),
                "role": "Ball",
                "conf": score,
                "bbox": [x1, y1, x2, y2],
                "center_px": (cx, cy),
                "pitch_m": None,
            })

        return balls

    def infer_ball_candidates(
        self,
        frame_bgr: np.ndarray,
        conf_low: float = 0.15,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        fallback_full_frame: bool = False,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Estrategia de Inferencia de Balón: ROI Dinámico (Zoom Inteligente) o Full Frame
        recolectando todos los candidatos desde `conf >= 0.15` en una sola pasada de GPU por fotograma.
        Retorna: (lista_candidatos, se_uso_roi)
        """
        if roi_box is not None:
            rx1, ry1, rx2, ry2 = roi_box
            crop = frame_bgr[ry1:ry2, rx1:rx2]
            if crop.size > 0:
                res_roi = self.ball_model.predict(
                    source=crop,
                    conf=float(conf_low),
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )[0]
                roi_cands: List[Dict[str, Any]] = []
                if res_roi.boxes is not None and len(res_roi.boxes) > 0:
                    boxes = res_roi.boxes.xyxy.cpu().numpy()
                    confs = res_roi.boxes.conf.cpu().numpy()
                    classes = res_roi.boxes.cls.cpu().numpy().astype(int)
                    order = np.argsort(-confs)
                    for rank, idx in enumerate(order):
                        score = float(confs[idx])
                        if score < conf_low:
                            continue
                        x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
                        gx1, gy1, gx2, gy2 = x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1
                        roi_cands.append({
                            "id": rank + 1,
                            "class_id": int(classes[idx]),
                            "role": "Ball",
                            "conf": score,
                            "bbox": [gx1, gy1, gx2, gy2],
                            "center_px": ((gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0),
                            "pitch_m": None,
                            "from_roi": True,
                        })
                if roi_cands or not fallback_full_frame:
                    return roi_cands, True

        # Inferencia en Full Frame cuando no hay ROI (estado LOST o barrido periódico)
        full_cands = self.infer_ball(frame_bgr, conf=conf_low)
        for c in full_cands:
            c["from_roi"] = False
        return full_cands, False

    def infer_keypoints(
        self,
        frame_bgr: np.ndarray,
        conf: Optional[float] = None,
        exclude_player_boxes: Optional[List[List[float]]] = None,
    ) -> Dict[int, Tuple[float, float, float]]:
        """
        Ejecuta inferencia de los 30 keypoints de homografía con YOLO Pose.
        Retorna diccionario {kp_index: (x_px, y_px, conf)} para índices 0..29.
        """
        c_thresh = float(conf if conf is not None else self.conf_keypoint)
        res = self.keypoint_model.predict(
            source=frame_bgr,
            conf=c_thresh,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        raw_kps: Dict[int, Tuple[float, float, float]] = {}
        if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
            return raw_kps

        kps_xy = res.keypoints.xy[0].cpu().numpy()
        kps_conf = (
            res.keypoints.conf[0].cpu().numpy()
            if res.keypoints.conf is not None
            else np.ones(len(kps_xy), dtype=np.float32)
        )

        h, w = frame_bgr.shape[:2]
        margin = 3.0

        for idx, ((kx, ky), kconf) in enumerate(zip(kps_xy, kps_conf)):
            score = float(kconf)
            if score < c_thresh:
                continue
            if kx <= margin or kx >= (w - margin) or ky <= margin or ky >= (h - margin):
                continue

            # Evitar keypoints espurios detectados sobre el cuerpo de un jugador
            if exclude_player_boxes:
                inside_player = False
                for bx1, by1, bx2, by2 in exclude_player_boxes:
                    if bx1 <= kx <= bx2 and by1 <= ky <= (by2 - 4.0):
                        inside_player = True
                        break
                if inside_player:
                    continue

            raw_kps[int(idx)] = (float(kx), float(ky), score)

        return raw_kps

