"""
Definición de los 30 Keypoints del modelo de Homografía y coordenadas métricas FIFA.
Contiene el diccionario exacto KP_MAP y PITCH_COORDS_METERS especificado para el proyecto.
"""

from typing import Dict, Tuple, List
import numpy as np

# Dimensiones por defecto de cancha FIFA en metros
LARGO_CANCHA: float = 105.0
ANCHO_CANCHA: float = 68.0

# Mapeo de los 30 keypoints del modelo YOLO Pose a su identificador de cancha
KP_MAP: Dict[int, str] = {
    0: '01',  1: '02',  2: '03',  3: '04',  4: '05',  5: '06',
    6: '07',  7: '08',  8: '10',  9: '11', 10: '12', 11: '13',
    12: '15', 13: '16', 14: '17', 15: '18',
    16: '20', 17: '21', 18: '22', 19: '23',
    20: '25', 21: '26', 22: '27', 23: '28',
    24: '29', 25: '30', 26: '31', 27: '32',
    28: '14', 29: '19'
}

MX, MY = LARGO_CANCHA / 2.0, ANCHO_CANCHA / 2.0

PITCH_COORDS_METERS: Dict[str, Tuple[float, float]] = {
    '01': (0.0, 0.0),                        '02': (0.0, MY - 20.16),
    '03': (0.0, MY - 9.16),                  '04': (0.0, MY + 9.16),
    '05': (0.0, MY + 20.16),                 '06': (0.0, ANCHO_CANCHA),
    '07': (5.5, MY - 9.16),                  '08': (5.5, MY + 9.16),
    '10': (16.5, MY - 20.16),                '11': (16.5, MY - 7.31),
    '12': (16.5, MY + 7.31),                 '13': (16.5, MY + 20.16),
    '14': (MX - 9.15, MY),                   '15': (MX, 0.0),
    '16': (MX, MY - 9.15),                   '17': (MX, MY + 9.15),
    '18': (MX, ANCHO_CANCHA),                '19': (MX + 9.15, MY),
    '20': (LARGO_CANCHA - 16.5, MY - 20.16), '21': (LARGO_CANCHA - 16.5, MY - 7.31),
    '22': (LARGO_CANCHA - 16.5, MY + 7.31),  '23': (LARGO_CANCHA - 16.5, MY + 20.16),
    '25': (LARGO_CANCHA - 5.5, MY - 9.16),   '26': (LARGO_CANCHA - 5.5, MY + 9.16),
    '27': (LARGO_CANCHA, 0.0),               '28': (LARGO_CANCHA, MY - 20.16),
    '29': (LARGO_CANCHA, MY - 9.16),         '30': (LARGO_CANCHA, MY + 9.16),
    '31': (LARGO_CANCHA, MY + 20.16),        '32': (LARGO_CANCHA, ANCHO_CANCHA),
}


def build_pitch_coords_meters(
    largo_cancha: float = LARGO_CANCHA,
    ancho_cancha: float = ANCHO_CANCHA,
) -> Dict[str, Tuple[float, float]]:
    """
    Construye el diccionario de coordenadas métricas para dimensiones personalizadas
    manteniendo exactamente la formulación geométrica de los 30 keypoints.
    """
    mx, my = largo_cancha / 2.0, ancho_cancha / 2.0
    return {
        '01': (0.0, 0.0),                        '02': (0.0, my - 20.16),
        '03': (0.0, my - 9.16),                  '04': (0.0, my + 9.16),
        '05': (0.0, my + 20.16),                 '06': (0.0, ancho_cancha),
        '07': (5.5, my - 9.16),                  '08': (5.5, my + 9.16),
        '10': (16.5, my - 20.16),                '11': (16.5, my - 7.31),
        '12': (16.5, my + 7.31),                 '13': (16.5, my + 20.16),
        '14': (mx - 9.15, my),                   '15': (mx, 0.0),
        '16': (mx, my - 9.15),                   '17': (mx, my + 9.15),
        '18': (mx, ancho_cancha),                '19': (mx + 9.15, my),
        '20': (largo_cancha - 16.5, my - 20.16), '21': (largo_cancha - 16.5, my - 7.31),
        '22': (largo_cancha - 16.5, my + 7.31),  '23': (largo_cancha - 16.5, my + 20.16),
        '25': (largo_cancha - 5.5, my - 9.16),   '26': (largo_cancha - 5.5, my + 9.16),
        '27': (largo_cancha, 0.0),               '28': (largo_cancha, my - 20.16),
        '29': (largo_cancha, my - 9.16),         '30': (largo_cancha, my + 9.16),
        '31': (largo_cancha, my + 20.16),        '32': (largo_cancha, ancho_cancha),
    }


def get_pitch_line_segments(
    largo_cancha: float = LARGO_CANCHA,
    ancho_cancha: float = ANCHO_CANCHA,
) -> List[np.ndarray]:
    """
    Genera los polilíneas canónicos de la cancha en metros para proyectarlos
    en el plano de la cámara mediante M^-1.
    """
    L, W = largo_cancha, ancho_cancha
    mx, my = L / 2.0, W / 2.0
    segments: List[np.ndarray] = []

    def add_seg(*pts):
        segments.append(np.array(pts, dtype=np.float32).reshape(-1, 1, 2))

    # Contorno exterior y línea de medio campo
    add_seg((0.0, 0.0), (L, 0.0), (L, W), (0.0, W), (0.0, 0.0))
    add_seg((mx, 0.0), (mx, W))

    # Círculo central
    angles = np.linspace(0.0, 2.0 * np.pi, 73)
    circle_pts = np.column_stack([
        mx + 9.15 * np.cos(angles),
        my + 9.15 * np.sin(angles),
    ]).astype(np.float32).reshape(-1, 1, 2)
    segments.append(circle_pts)

    # Área grande izquierda y derecha
    add_seg((0.0, my - 20.16), (16.5, my - 20.16), (16.5, my + 20.16), (0.0, my + 20.16))
    add_seg((L, my - 20.16), (L - 16.5, my - 20.16), (L - 16.5, my + 20.16), (L, my + 20.16))

    # Área chica izquierda y derecha
    add_seg((0.0, my - 9.16), (5.5, my - 9.16), (5.5, my + 9.16), (0.0, my + 9.16))
    add_seg((L, my - 9.16), (L - 5.5, my - 9.16), (L - 5.5, my + 9.16), (L, my + 9.16))

    # Medialunas del área (arcos de penalti)
    arc_angle = np.arccos(5.5 / 9.15)  # ~53.05 grados
    left_arc_angles = np.linspace(-arc_angle, arc_angle, 37)
    left_arc = np.column_stack([
        11.0 + 9.15 * np.cos(left_arc_angles),
        my + 9.15 * np.sin(left_arc_angles),
    ]).astype(np.float32).reshape(-1, 1, 2)
    segments.append(left_arc)

    right_arc_angles = np.linspace(np.pi - arc_angle, np.pi + arc_angle, 37)
    right_arc = np.column_stack([
        (L - 11.0) + 9.15 * np.cos(right_arc_angles),
        my + 9.15 * np.sin(right_arc_angles),
    ]).astype(np.float32).reshape(-1, 1, 2)
    segments.append(right_arc)

    return segments

