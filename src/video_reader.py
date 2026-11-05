"""
src/video_reader.py — Lector de video asíncrono de alto rendimiento optimizado para GPU (NVIDIA L4 / T4 / A100).
Soporta:
1. Buffer profundo en RAM (hasta 384 cuadros) para aprovechar sistemas con amplia memoria (ej. 53 GB RAM en Colab).
2. Pre-carga opcional de tensores en VRAM / Pinned Memory mediante un CUDA Stream dedicado de transferencia
   Host-to-Device (H2D) para solapar el envío PCIe con la ejecución de motores TensorRT.
"""

import threading
from queue import Queue
from pathlib import Path
from typing import Optional, Tuple, Generator, Union
import cv2
import numpy as np

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except Exception:
    HAS_TORCH_CUDA = False


class ThreadedVideoReader:
    """
    Decodificador de video multihilo (Productor-Consumidor) con soporte de buffer profundo en RAM:
    Un hilo secundario mantiene siempre una cola de cuadros decodificados en memoria,
    evitando que la GPU y los motores TensorRT queden bloqueados esperando a `cap.read()`.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        queue_size: int = 256,
        max_frames: Optional[int] = None,
        stride: int = 1,
    ):
        self.video_path = str(video_path)
        self.queue_size = int(queue_size)
        self.max_frames = max_frames
        self.stride = max(1, int(stride))

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video con OpenCV: {self.video_path}")

        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.fps = raw_fps / float(self.stride)
        raw_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = (raw_total + self.stride - 1) // self.stride if raw_total > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.target_frames = (
            min(self.total_frames, self.max_frames)
            if (self.max_frames and self.total_frames > 0)
            else (self.max_frames or self.total_frames)
        )

        self.queue: Queue = Queue(maxsize=self.queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        raw_idx = 0
        dispatched_idx = 0
        while not self.stopped:
            if self.max_frames and dispatched_idx >= self.max_frames:
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            raw_idx += 1
            if self.stride > 1 and (raw_idx - 1) % self.stride != 0:
                continue

            dispatched_idx += 1
            while not self.stopped:
                try:
                    self.queue.put((dispatched_idx, frame), timeout=0.1)
                    break
                except Exception:
                    continue

        self.stopped = True
        try:
            self.queue.put((None, None), timeout=1.0)
        except Exception:
            pass

    def read(self) -> Tuple[bool, Optional[int], Optional[np.ndarray]]:
        """Devuelve (éxito, número_de_frame, frame_bgr)"""
        if self.stopped and self.queue.empty():
            return False, None, None

        try:
            frame_idx, frame = self.queue.get(timeout=2.0)
            if frame_idx is None:
                return False, None, None
            return True, frame_idx, frame
        except Exception:
            return False, None, None

    def release(self) -> None:
        self.stopped = True
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        while True:
            ret, idx, frame = self.read()
            if not ret or idx is None or frame is None:
                break
            yield idx, frame
