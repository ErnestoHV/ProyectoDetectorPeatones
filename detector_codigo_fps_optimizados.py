"""
Detector de Personas con Trayectoria de Desplazamiento
Modelo: YOLO11s (best.pt) entrenado con Roboflow
Autor: Proyecto Fundamentos - Semestre 8

OPTIMIZACIONES APLICADAS:
  - Tracker cambiado de BotSORT → ByteTrack (elimina Re-ID por persona)
  - Historial de puntos con deque(maxlen=N) en lugar de lista + pop(0) O(n)
  - Dibujo de trayectorias con cv2.polylines en lugar de N llamadas a cv2.line
  - HUD sin frame.copy() innecesario
  - Verificación explícita de uso de GPU al inicio
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import os
import sys

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────

RUTA_MODELO   = "best.pt"          # Coloca best.pt en la misma carpeta que este script
MAX_PUNTOS    = 60                 # Cuántos puntos de trayectoria se conservan por persona
CONFIANZA_MIN = 0.40               # Umbral mínimo de confianza (0.0 – 1.0)
TRACKER       = "bytetrack.yaml"   # ✅ OPTIMIZACIÓN: ByteTrack es más ligero que BotSORT (sin Re-ID)

# Colores (BGR)
COLOR_BBOX        = (0, 200, 255)  # Naranja para el bounding box
COLOR_ID          = (255, 255, 255)
COLOR_TRAYECTORIA = None           # None = color único por persona (auto)

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def verificar_gpu():
    """Verifica disponibilidad de GPU e imprime información relevante."""
    if torch.cuda.is_available():
        nombre = torch.cuda.get_device_name(0)
        memoria = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] ✅ Detectada: {nombre} ({memoria:.1f} GB)")
        return 0
    else:
        print("[GPU] ⚠️  CUDA no disponible. Se usará CPU (rendimiento reducido).")
        return "cpu"


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

        # ✅ OPTIMIZACIÓN: deque con maxlen evita pop(0) O(n)
        historial[track_id].append((cx, cy))

        color = COLOR_TRAYECTORIA if COLOR_TRAYECTORIA else color_por_id(track_id)

        # ── Trayectoria ──────────────────────────────
        puntos = list(historial[track_id])
        if len(puntos) >= 2:
            # ✅ OPTIMIZACIÓN: una sola llamada cv2.polylines en lugar de N cv2.line
            pts = np.array(puntos, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color,
                          thickness=2, lineType=cv2.LINE_AA)

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
    # ✅ OPTIMIZACIÓN: dibuja directo sobre frame, sin frame.copy() + addWeighted
    cv2.rectangle(frame, (8, 8), (260, 68), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas: {n_personas}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps_real:.1f}", (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────

def ejecutar(fuente, guardar: bool = False, ruta_salida: str = "resultado.mp4"):
    """
    fuente  : 0 para webcam, o ruta a un archivo de video (str)
    guardar : True para guardar el video procesado
    """
    if not os.path.exists(RUTA_MODELO):
        print(f"\n[ERROR] No se encontró el modelo en: {RUTA_MODELO}")
        print("Asegúrate de colocar 'best.pt' en la misma carpeta que este script.\n")
        sys.exit(1)

    # ✅ OPTIMIZACIÓN: verificar GPU antes de cargar el modelo
    device = verificar_gpu()

    print(f"[INFO] Cargando modelo: {RUTA_MODELO}")
    modelo = YOLO(RUTA_MODELO)

    print(f"[INFO] Abriendo fuente: {'Cámara web' if fuente == 0 else fuente}")
    cap = cv2.VideoCapture(fuente)

    if not cap.isOpened():
        print(f"\n[ERROR] No se pudo abrir la fuente de video: {fuente}")
        sys.exit(1)

    ancho  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if guardar:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(ruta_salida, fourcc, fps_in, (ancho, alto))
        print(f"[INFO] Guardando resultado en: {ruta_salida}")

    # ✅ OPTIMIZACIÓN: deque con maxlen=MAX_PUNTOS, elimina pop(0) O(n)
    historial   = defaultdict(lambda: deque(maxlen=MAX_PUNTOS))
    tick_prev   = cv2.getTickCount()
    fps_display = 0.0

    print("\n[LISTO] Presiona  Q  para salir.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Fin del video o no se pudo leer el frame.")
            break

        # ── Inferencia + tracking ─────────────────
        resultados = modelo.track(
            frame,
            persist   = True,
            tracker   = TRACKER,       # ✅ ByteTrack
            conf      = CONFIANZA_MIN,
            device    = device,        # ✅ GPU verificada al inicio
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

        # ── Mostrar ───────────────────────────────
        cv2.imshow("Detector de Personas — Trayectoria  [Q para salir]", frame)

        if writer:
            writer.write(frame)

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
    print("  DETECTOR DE PERSONAS — YOLO11s + ByteTrack")
    print("=" * 50)
    print("\n¿Qué fuente de video deseas usar?")
    print("  [1]  Cámara web (en vivo)")
    print("  [2]  Archivo de video (prueba)")
    print()

    opcion = input("Selecciona una opción (1 o 2): ").strip()

    if opcion == "1":
        guardar = input("\n¿Guardar el video resultante? (s/n): ").strip().lower() == "s"
        salida  = "resultado_camara.mp4" if guardar else ""
        ejecutar(fuente=0, guardar=guardar, ruta_salida=salida)

    elif opcion == "2":
        ruta = input("\nEscribe la ruta del archivo de video\n"
                     "(o presiona Enter para usar 'videos/prueba.mp4'): ").strip()
        if not ruta:
            ruta = os.path.join("videos", "prueba.mp4")

        ruta = os.path.normpath(ruta)

        guardar = input("¿Guardar el video resultante? (s/n): ").strip().lower() == "s"
        nombre  = os.path.splitext(os.path.basename(ruta))[0]
        salida  = f"resultado_{nombre}.mp4" if guardar else ""
        ejecutar(fuente=ruta, guardar=guardar, ruta_salida=salida)

    else:
        print("[ERROR] Opción no válida. Vuelve a ejecutar el script.")
        sys.exit(1)


if __name__ == "__main__":
    menu()