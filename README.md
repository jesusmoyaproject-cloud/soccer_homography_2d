# Proyecto de Detección de Jugador, Balón y Homografía 2D (MAGSAC++)

Sistema de visión artificial y Deep Learning (Ultralytics YOLO `.pt` + OpenCV `USAC_MAGSAC`) para detectar jugadores, balón y los 30 keypoints de la cancha, estabilizar la homografía contra parpadeos/deformaciones y proyectar el plano 2D de movimientos.

## Estructura del Proyecto

```text
soccer_homography_2d/
├── configs/
│   └── config.yaml          # Configuración central que invoca los modelos .pt y define hiperparámetros
├── models/                  # Carpeta que alberga los modelos PyTorch (.pt)
│   ├── modelPlayer.pt       # Modelo de detección de jugadores (conf = 0.5)
│   ├── modelBall.pt         # Modelo de detección de balón (conf = 0.4)
│   └── modelKeypoint.pt     # Modelo YOLO Pose de 30 keypoints de cancha
├── src/
│   ├── pitch_config.py      # KP_MAP (30 keypoints) y PITCH_COORDS_METERS (105m x 68m)
│   ├── homography.py        # MAGSAC++, Estabilización (Métodos 1 y 2) y Tomas Cerradas (Métodos 1, 2 y 3)
│   ├── detector.py          # Motor de inferencia para los 3 modelos .pt
│   └── renderer.py          # Etiquetado de inferencias (ID + conf) y generación del Plano 2D
├── outputs/                 # Carpeta donde se guardan las imágenes y videos procesados
├── tests/
│   └── test_pipeline.py     # Pruebas unitarias de los algoritmos de homografía y estabilización
└── main.py                  # Script principal ejecutable para imágenes o videos
```

## Ejecución (`main.py`)

### 1. Procesar una imagen
```bash
python main.py --input "../soccer_field_sample.jpg" --output "outputs/resultado_imagen.jpg"
```

### 2. Procesar un video probando los métodos de estabilización de homografía
- **Método 1 (Ventana de Proximidad en Píxeles)**:
  ```bash
  python main.py --input ruta/al/video.mp4 --stabilization method_1
  ```
- **Método 2 (Proximidad por Proyección Inversa con $M_{anterior}^{-1}$)**:
  ```bash
  python main.py --input ruta/al/video.mp4 --stabilization method_2
  ```
- **Ambos métodos combinados (`both`, por defecto en `configs/config.yaml`)**:
  ```bash
  python main.py --input ruta/al/video.mp4 --stabilization both
  ```

