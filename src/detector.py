"""
Módulo de Inferencia Deep Learning (Ultralytics YOLO .pt).
Carga los modelos .pt definidos en el archivo .yaml:
  - Jugador (conf = 0.5)
  - Balón (conf = 0.4)
  - Homografía / 30 Keypoints (YOLO Pose)
"""

import logging
import warnings
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
        self.player_classes = det_cfg.get("player_classes", [1, 2, 3])
        self.class_names = det_cfg.get("class_names", {0: "Ball", 1: "GK", 2: "Player", 3: "Referee"})
        tracker_rel = det_cfg.get("tracker_config", "configs/botsort_sports.yaml")
        tracker_cand = PROJECT_ROOT / tracker_rel
        self.tracker_config = str(tracker_cand) if tracker_cand.exists() else "botsort.yaml"

        # Cargar los 3 modelos PyTorch (.pt)
        self.player_model = self._load_model(self.player_model_path, task="detect")
        self.ball_model = self._load_model(self.ball_model_path, task="detect")
        self.keypoint_model = self._load_model(self.kp_model_path, task="pose")

    @staticmethod
    def _resolve_model_path(rel_or_abs: Union[str, Path]) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute() and p.exists():
            return p
        cand = PROJECT_ROOT / p
        if cand.exists():
            return cand
        raise FileNotFoundError(f"Modelo .pt no encontrado en: {cand}")

    def _load_model(self, path: Path, task: str) -> YOLO:
        model = YOLO(str(path), task=task)
        if path.suffix.lower() == ".pt" and "cuda" in self.device:
            try:
                model.to(self.device)
            except Exception:
                pass
        return model

    def infer_players(
        self,
        frame_bgr: np.ndarray,
        is_video: bool = False,
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta detección/tracking de jugadores con conf=0.5 (por defecto).
        Retorna lista de diccionarios con id, class_id, role, conf, bbox y punto de apoyo (pies).
        """
        c_thresh = float(conf if conf is not None else self.conf_player)
        if is_video:
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
            color_bgr = (0, 215, 255) if int(cls_id) == 3 else extract_jersey_bgr(frame_bgr, [x1, y1, x2, y2])

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
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Estrategia de Inferencia de Balón: ROI Dinámico (Zoom Inteligente) + Full Frame Fallback
        recolectando todos los candidatos desde `conf >= 0.15`.
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
                if res_roi.boxes is not None and len(res_roi.boxes) > 0:
                    boxes = res_roi.boxes.xyxy.cpu().numpy()
                    confs = res_roi.boxes.conf.cpu().numpy()
                    classes = res_roi.boxes.cls.cpu().numpy().astype(int)
                    order = np.argsort(-confs)
                    roi_cands: List[Dict[str, Any]] = []
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
                    if roi_cands:
                        return roi_cands, True

        # Fallback a Full Frame si no había ROI o el ROI no devolvió candidatos
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

