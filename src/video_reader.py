"""
src/video_reader.py — Lector de video asíncrono y de alto rendimiento.
Soporta:
1. ThreadedVideoReader (OpenCV multi-hilo con buffer en cola FIFO para eliminar esperas de I/O).
2. DecordVideoReader (si decord está disponible, decodificación rápida y acceso por lotes).
"""

import time
import threading
from queue import Queue
from pathlib import Path
from typing import Optional, Tuple, Generator, Union
import cv2
import numpy as np


class ThreadedVideoReader:
    """
    Decodificador de video multihilo (Productor-Consumidor):
    Un hilo secundario mantiene siempre una cola de cuadros decodificados en memoria,
    evitando que la GPU y los modelos TensorRT queden bloqueados esperando a `cap.read()`.
    """

    def __init__(self, video_path: Union[str, Path], queue_size: int = 64, max_frames: Optional[int] = None):
        self.video_path = str(video_path)
        self.queue_size = queue_size
        self.max_frames = max_frames

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video con OpenCV: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        frame_idx = 0
        while not self.stopped:
            if self.max_frames and frame_idx >= self.max_frames:
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            # Bloquea si la cola está llena hasta que el hilo consumidor extraiga un frame
            while not self.stopped:
                try:
                    self.queue.put((frame_idx, frame), timeout=0.1)
                    break
                except Exception:
                    continue

        self.stopped = True
        # Sentinel para indicar fin de video
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
            # Vaciar cola para desbloquear el worker
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
            if not ret or frame is None:
                break
            yield idx, frame
        self.release()

