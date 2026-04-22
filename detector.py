"""
Detector de Personas con Trayectoria de Desplazamiento
Modelo: YOLO11s (best.pt) entrenado con Roboflow
Autor: Proyecto Fundamentos - Semestre 8
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import sys
import ctypes
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────

RUTA_MODELO   = "best.pt"          # Coloca best.pt en la misma carpeta que este script
MAX_PUNTOS    = 60                 # Cuántos puntos de trayectoria se conservan por persona
CONFIANZA_MIN = 0.40               # Umbral mínimo de confianza (0.0 – 1.0)
TRACKER       = "botsort.yaml"     # Tracker: botsort.yaml  o  bytetrack.yaml

# Colores (BGR)
COLOR_BBOX        = (0, 200, 255)  # Naranja para el bounding box
COLOR_ID          = (255, 255, 255)
COLOR_TRAYECTORIA = None           # None = color único por persona (auto)

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def color_por_id(track_id: int) -> tuple:
    """Genera un color BGR reproducible y vibrante para cada ID."""
    np.random.seed(int(track_id) * 7 + 13)
    h = np.random.randint(0, 180)
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def dibujar_frame(frame, resultados, historial: dict) -> np.ndarray:
    """
    Recibe el frame y los resultados del tracker.
    Dibuja bounding boxes, IDs y trayectorias.
    """
    if resultados[0].boxes is None:
        return frame

    boxes = resultados[0].boxes

    # Solo procesar detecciones con ID asignado
    for box in boxes:
        if box.id is None:
            continue

        track_id = int(box.id.item())
        conf     = float(box.conf.item())

        if conf < CONFIANZA_MIN:
            continue

        # Coordenadas del bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Centroide
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Actualizar historial de puntos
        historial[track_id].append((cx, cy))
        if len(historial[track_id]) > MAX_PUNTOS:
            historial[track_id].pop(0)

        color = COLOR_TRAYECTORIA if COLOR_TRAYECTORIA else color_por_id(track_id)

        # ── Trayectoria ──────────────────────────────
        puntos = historial[track_id]
        for i in range(1, len(puntos)):
            alpha = i / len(puntos)             # más brillante cuanto más reciente
            c = tuple(int(v * alpha) for v in color)
            grosor = max(1, int(3 * alpha))
            cv2.line(frame, puntos[i - 1], puntos[i], c, grosor, cv2.LINE_AA)

        # Punto actual (círculo)
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)

        # ── Bounding box ─────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2, cv2.LINE_AA)

        # ── Etiqueta ID + confianza ───────────────────
        etiqueta = f"ID {track_id}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BBOX, -1)
        cv2.putText(frame, etiqueta, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ID, 1, cv2.LINE_AA)

    return frame


def hud(frame, n_personas: int, fps_real: float) -> np.ndarray:
    """Dibuja el HUD (personas activas + FPS) en la esquina superior izquierda."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (260, 68), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"Personas: {n_personas}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps_real:.1f}", (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return frame


def nombre_salida_con_timestamp(base: str, extension: str = ".mp4") -> str:
    """Devuelve un nombre de archivo con fecha/hora agregado.

    Se usa un formato válido en Windows evitando dos puntos en el nombre.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    return f"{base}_{timestamp}{extension}"


def calcular_dimensiones_display(ancho_video: int, alto_video: int,
                                  margen: float = 0.95) -> tuple:
    """
    Calcula las dimensiones de display manteniendo la relación de aspecto.
    Usa ctypes.windll directamente: sin tkinter, sin ventanas extra.
    Retorna (nuevo_ancho, nuevo_alto).
    """
    try:
        # GetSystemMetrics(0) = ancho de pantalla, (1) = alto de pantalla.
        # No se llama a SetProcessDpiAwareness para no interferir con OpenCV.
        user32         = ctypes.windll.user32
        ancho_pantalla = user32.GetSystemMetrics(0)
        alto_pantalla  = user32.GetSystemMetrics(1)
    except Exception:
        ancho_pantalla, alto_pantalla = 1920, 1080

    max_ancho = int(ancho_pantalla * margen)
    max_alto  = int(alto_pantalla  * margen)
    escala    = min(max_ancho / ancho_video, max_alto / alto_video, 1.0)

    nuevo_ancho = int(ancho_video * escala)
    nuevo_alto  = int(alto_video  * escala)

    print(f"[INFO] Pantalla detectada: {ancho_pantalla}x{alto_pantalla}")
    print(f"[INFO] Video original:     {ancho_video}x{alto_video}")
    print(f"[INFO] Ventana de display: {nuevo_ancho}x{nuevo_alto}")

    return nuevo_ancho, nuevo_alto


# ─────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────

def ejecutar(fuente, guardar: bool = False, ruta_salida: str = None,
             ajustar_pantalla: bool = False):
    """
    fuente           : 0 para webcam, o ruta a un archivo de video (str)
    guardar          : True para guardar el video procesado
    ruta_salida      : nombre final del archivo de salida
    ajustar_pantalla : True para escalar la ventana al tamaño de la pantalla
                       manteniendo la relación de aspecto (solo modo video)
    """
    # Verificar que el modelo existe
    if not os.path.exists(RUTA_MODELO):
        print(f"\n[ERROR] No se encontró el modelo en: {RUTA_MODELO}")
        print("Asegúrate de colocar 'best.pt' en la misma carpeta que este script.\n")
        sys.exit(1)

    print(f"\n[INFO] Cargando modelo: {RUTA_MODELO}")
    modelo = YOLO(RUTA_MODELO)

    print(f"[INFO] Abriendo fuente: {'Cámara web' if fuente == 0 else fuente}")
    cap = cv2.VideoCapture(fuente)

    if not cap.isOpened():
        print(f"\n[ERROR] No se pudo abrir la fuente de video: {fuente}")
        sys.exit(1)

    ancho  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    # ── Calcular dimensiones de display ──────────────────────────────────────
    # Se calcula ANTES del bucle para no hacerlo cada frame.
    # No se llama a namedWindow aquí: cv2.imshow crea la ventana sola con el
    # tamaño exacto del frame que recibe, evitando la ventana gris fantasma.
    nombre_ventana = "Detector de Personas - Trayectoria  [Q para salir]"

    if ajustar_pantalla:
        disp_w, disp_h = calcular_dimensiones_display(ancho, alto)
    else:
        disp_w, disp_h = ancho, alto

    # ── Escritor de video (opcional) ─────────────────────────────────────────
    # El video guardado siempre usa la resolución original del video fuente.
    writer = None
    if guardar:
        if not ruta_salida:
            ruta_salida = nombre_salida_con_timestamp("resultado")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(ruta_salida, fourcc, fps_in, (ancho, alto))
        print(f"[INFO] Guardando resultado en: {ruta_salida}")

    historial   = defaultdict(list)   # {track_id: [(x,y), ...]}
    tick_prev   = cv2.getTickCount()
    fps_display = 0.0

    print("\n[LISTO] Presiona  Q  para salir.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Fin del video o no se pudo leer el frame.")
            break

        # frame = cv2.flip(frame, 1)

        # ── Inferencia + tracking ─────────────────
        resultados = modelo.track(
            frame,
            persist   = True,        # mantiene IDs entre frames
            tracker   = TRACKER,
            conf      = CONFIANZA_MIN,
            device    = 0,           # GPU (usa 'cpu' si falla)
            verbose   = False,
            imgsz     = 640,
        )

        # ── Dibujar ───────────────────────────────
        frame = dibujar_frame(frame, resultados, historial)

        # Contar personas activas en este frame
        n_activas = 0
        if resultados[0].boxes is not None:
            n_activas = sum(
                1 for b in resultados[0].boxes
                if b.id is not None and float(b.conf.item()) >= CONFIANZA_MIN
            )

        # ── FPS real ──────────────────────────────
        tick_actual = cv2.getTickCount()
        fps_display = cv2.getTickFrequency() / (tick_actual - tick_prev)
        tick_prev   = tick_actual

        frame = hud(frame, n_activas, fps_display)

        # ── Guardar (resolución original) ─────────
        if writer:
            writer.write(frame)

        # ── Mostrar ───────────────────────────────
        # Si se activa ajustar_pantalla, el frame se redimensiona ANTES de
        # pasarlo a imshow. De este modo imshow crea una sola ventana del
        # tamaño exacto del frame, sin necesidad de namedWindow previo.
        if ajustar_pantalla and (disp_w != ancho or disp_h != alto):
            frame_display = cv2.resize(frame, (disp_w, disp_h),
                                       interpolation=cv2.INTER_LINEAR)
        else:
            frame_display = frame

        cv2.imshow(nombre_ventana, frame_display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Salida solicitada por el usuario.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Recursos liberados. ¡Hasta luego!")


# ─────────────────────────────────────────────
# MENÚ DE INICIO
# ─────────────────────────────────────────────

def menu():
    print("=" * 50)
    print("  DETECTOR DE PERSONAS — YOLO11s + BotSORT")
    print("=" * 50)
    print("\n¿Qué fuente de video deseas usar?")
    print("  [1]  Cámara web (en vivo)")
    print("  [2]  Archivo de video (prueba)")
    print()

    opcion = input("Selecciona una opción (1 o 2): ").strip()

    if opcion == "1":
        # ── Modo cámara ──────────────────────────
        guardar = input("\n¿Guardar el video resultante? (s/n): ").strip().lower() == "s"
        salida  = nombre_salida_con_timestamp("resultado_camara") if guardar else None
        ejecutar(fuente=0, guardar=guardar, ruta_salida=salida, ajustar_pantalla=True)

    elif opcion == "2":
        # ── Modo video ───────────────────────────
        ruta = input("\nEscribe la ruta del archivo de video\n"
                     "(o presiona Enter para usar 'videos/prueba.mp4'): ").strip()
        if not ruta:
            ruta = os.path.join("videos", "prueba.mp4")

        # Normalizar ruta (acepta barras invertidas de Windows)
        ruta = os.path.normpath(ruta)

        guardar = input("¿Guardar el video resultante? (s/n): ").strip().lower() == "s"
        nombre  = os.path.splitext(os.path.basename(ruta))[0]
        salida  = nombre_salida_con_timestamp(f"resultado_{nombre}") if guardar else None
        ejecutar(fuente=ruta, guardar=guardar, ruta_salida=salida, ajustar_pantalla=True)

    else:
        print("[ERROR] Opción no válida. Vuelve a ejecutar el script.")
        sys.exit(1)


if __name__ == "__main__":
    menu()