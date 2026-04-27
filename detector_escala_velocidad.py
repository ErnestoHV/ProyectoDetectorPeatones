"""
Detector de Personas con Trayectoria de Desplazamiento y Velocidad
Modelo: YOLO11s (best.pt) entrenado con Roboflow
Autor: Proyecto Fundamentos - Semestre 8

OPTIMIZACIONES APLICADAS:
  - Tracker cambiado de BotSORT → ByteTrack (elimina Re-ID por persona)
  - Historial de puntos con deque(maxlen=N) en lugar de lista + pop(0) O(n)
  - Dibujo de trayectorias con cv2.polylines en lugar de N llamadas a cv2.line
  - HUD sin frame.copy() innecesario
  - Verificación explícita de uso de GPU al inicio
  - Frame skipping para videos de alto FPS
  - Limpieza automática de IDs huérfanos en historial
  - Preprocesamiento CLAHE para baja iluminación
  - Escalado previo para videos 4K

NUEVAS FUNCIONES:
  - calcular_velocidad(): estima la velocidad de cada persona en km/h
  - guardar_csv(): exporta promedios de velocidad por persona al finalizar
  - Etiqueta de velocidad visible sobre cada bounding box
"""

import csv
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
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
TRACKER       = "bytetrack.yaml"   # ByteTrack es más ligero que BotSORT (sin Re-ID)

# Frame skipping: 1 = procesa 1 de cada 2 frames, 0 = procesa todos
SALTAR_FRAMES  = 1

# Frames sin actividad antes de eliminar un ID del historial
FRAMES_LIMITE  = 90

# ── Calibración de velocidad ─────────────────────────────────────────────────
#
# PIXELS_POR_METRO indica cuántos píxeles del video equivalen a 1 metro real.
# Cómo calcularlo:
#   1. Pausa el video en un frame donde veas un objeto de tamaño conocido
#      (ej. un automóvil estándar ≈ 4.5 m, una baldosa ≈ 0.30 m).
#   2. Mide cuántos píxeles ocupa ese objeto en la imagen.
#   3. Divide: PIXELS_POR_METRO = píxeles_medidos / metros_reales
#
# Ejemplo: un auto de 4.5 m ocupa 180 px → 180 / 4.5 = 40
#
# Si no puedes calibrar, deja el valor por defecto (40) y la velocidad
# mostrada será aproximada.
#
PIXELS_POR_METRO = 40.0            # ← Ajusta este valor según tu escena

# Ventana de puntos para calcular la velocidad instantánea.
# Un valor mayor suaviza más pero reacciona más lento a cambios.
VENTANA_VEL  = 10                  # Últimos N puntos usados para la estimación

# Colores (BGR)
COLOR_BBOX        = (0, 200, 255)  # Naranja para el bounding box
COLOR_ID          = (255, 255, 255)
COLOR_TRAYECTORIA = None           # None = color único por persona (auto)
COLOR_VEL         = (180, 255, 100)  # Verde lima para la etiqueta de velocidad

# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def verificar_gpu() -> int | str:
    """Verifica disponibilidad de GPU e imprime información relevante."""
    if torch.cuda.is_available():
        nombre  = torch.cuda.get_device_name(0)
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


def mejorar_frame(frame: np.ndarray) -> np.ndarray:
    """
    Aplica ecualización adaptativa del histograma (CLAHE) sobre el canal
    de luminosidad en espacio LAB. Mejora la detección en videos oscuros
    sin alterar artificialmente los colores.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def escalar_frame(frame: np.ndarray, ancho_max: int = 1280) -> np.ndarray:
    """
    Reduce el frame si supera el ancho máximo indicado, manteniendo la
    relación de aspecto original. Útil para videos 4K.
    """
    h, w = frame.shape[:2]
    if w > ancho_max:
        escala = ancho_max / w
        return cv2.resize(
            frame,
            (int(w * escala), int(h * escala)),
            interpolation=cv2.INTER_AREA
        )
    return frame


# ── NUEVA FUNCIÓN ────────────────────────────────────────────────────────────

def calcular_velocidad(historial: dict, track_id: int, fps_efectivo: float) -> float:
    """
    Estima la velocidad instantánea de una persona en km/h.

    Parámetros
    ----------
    historial     : diccionario {track_id: deque[(x, y), ...]}
    track_id      : ID de la persona a calcular
    fps_efectivo  : FPS reales con los que se procesan frames
                    (fps_video / (SALTAR_FRAMES + 1))

    Retorna
    -------
    Velocidad en km/h. Devuelve 0.0 si no hay suficientes puntos.

    Método
    ------
    Se toman los últimos VENTANA_VEL puntos del historial.
    Se calcula la distancia euclidiana total recorrida en esos puntos,
    se convierte de píxeles a metros con PIXELS_POR_METRO y se divide
    por el tiempo transcurrido (puntos / fps_efectivo) para obtener m/s.
    Finalmente se convierte a km/h multiplicando por 3.6.
    """
    puntos = list(historial[track_id])
    if len(puntos) < 2:
        return 0.0

    # Usar solo los últimos VENTANA_VEL puntos
    ventana = puntos[-VENTANA_VEL:]
    if len(ventana) < 2:
        return 0.0

    # Distancia total en píxeles dentro de la ventana
    distancia_px = sum(
        np.hypot(ventana[i][0] - ventana[i - 1][0],
                 ventana[i][1] - ventana[i - 1][1])
        for i in range(1, len(ventana))
    )

    # Tiempo transcurrido en segundos
    n_intervalos = len(ventana) - 1
    tiempo_s     = n_intervalos / fps_efectivo if fps_efectivo > 0 else 1.0

    # Convertir píxeles → metros → km/h
    distancia_m  = distancia_px / PIXELS_POR_METRO
    velocidad_ms = distancia_m / tiempo_s
    return velocidad_ms * 3.6  # km/h


# ── NUEVA FUNCIÓN ────────────────────────────────────────────────────────────

def guardar_csv(velocidades: dict, ruta_csv: str) -> None:
    """
    Exporta el promedio de velocidad de cada persona detectada a un CSV.

    Parámetros
    ----------
    velocidades : diccionario {track_id: [vel1, vel2, ...]} con todas las
                  mediciones de velocidad (km/h) registradas por frame.
    ruta_csv    : ruta completa del archivo de salida.

    Formato del CSV
    ---------------
    ID_Persona, Velocidad_Promedio_kmh
    1, 4.32
    2, 2.87
    ...

    Solo se incluyen personas con al menos una medición de velocidad > 0
    para evitar registros de IDs que aparecieron un solo frame.
    """
    filas = []
    for tid, mediciones in velocidades.items():
        mediciones_validas = [v for v in mediciones if v > 0.0]
        if not mediciones_validas:
            continue
        promedio = sum(mediciones_validas) / len(mediciones_validas)
        filas.append((tid, round(promedio, 2)))

    # Ordenar por ID para facilitar lectura
    filas.sort(key=lambda x: x[0])

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID_Persona", "Velocidad_Promedio_kmh"])
        writer.writerows(filas)

    print(f"[CSV] ✅ Resultados guardados en: {ruta_csv}  ({len(filas)} persona(s))")


# ── FUNCIÓN MODIFICADA ───────────────────────────────────────────────────────

def dibujar_frame(frame, resultados, historial: dict,
                  velocidades: dict, fps_efectivo: float) -> tuple:
    """
    Recibe el frame y los resultados del tracker.
    Dibuja bounding boxes, IDs, velocidad y trayectorias.

    Parámetros nuevos respecto a la versión anterior
    -------------------------------------------------
    velocidades  : diccionario {track_id: [lista de mediciones km/h]}
                   se actualiza aquí en cada frame.
    fps_efectivo : FPS reales de procesamiento para el cálculo de velocidad.

    Retorna
    -------
    (frame, ids_activos)
      frame       : frame con todos los elementos dibujados
      ids_activos : set de track_ids visibles en este frame
    """
    ids_activos = set()

    if resultados[0].boxes is None:
        return frame, ids_activos

    boxes = resultados[0].boxes

    for box in boxes:
        if box.id is None:
            continue

        track_id = int(box.id.item())
        conf     = float(box.conf.item())

        if conf < CONFIANZA_MIN:
            continue

        ids_activos.add(track_id)

        # Coordenadas del bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Centroide
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        historial[track_id].append((cx, cy))

        # ── Velocidad ─────────────────────────────
        vel_kmh = calcular_velocidad(historial, track_id, fps_efectivo)
        velocidades[track_id].append(vel_kmh)   # Acumular para el CSV

        color = COLOR_TRAYECTORIA if COLOR_TRAYECTORIA else color_por_id(track_id)

        # ── Trayectoria ───────────────────────────
        puntos = list(historial[track_id])
        if len(puntos) >= 2:
            pts = np.array(puntos, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color,
                          thickness=2, lineType=cv2.LINE_AA)

        # Punto actual (círculo)
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)

        # ── Bounding box ──────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2, cv2.LINE_AA)

        # ── Etiqueta superior: ID + confianza ─────
        etiqueta_id  = f"ID {track_id}  {conf:.0%}"
        (tw, th), _  = cv2.getTextSize(etiqueta_id, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), COLOR_BBOX, -1)
        cv2.putText(frame, etiqueta_id, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ID, 1, cv2.LINE_AA)

        # ── Etiqueta inferior: velocidad ──────────
        etiqueta_vel  = f"{vel_kmh:.1f} km/h"
        (vw, vh), _   = cv2.getTextSize(etiqueta_vel, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        # Se dibuja justo debajo del bounding box
        cv2.rectangle(frame, (x1, y2), (x1 + vw + 6, y2 + vh + 8), (30, 30, 30), -1)
        cv2.putText(frame, etiqueta_vel, (x1 + 3, y2 + vh + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_VEL, 1, cv2.LINE_AA)

    return frame, ids_activos


def hud(frame, n_personas: int, fps_real: float) -> np.ndarray:
    """Dibuja el HUD (personas activas + FPS) en la esquina superior izquierda."""
    cv2.rectangle(frame, (8, 8), (260, 68), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas: {n_personas}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps_real:.1f}", (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return frame


def nombre_salida_con_timestamp(base: str, extension: str = ".mp4") -> str:
    """Devuelve un nombre de archivo con fecha/hora agregado."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    return f"{base}_{timestamp}{extension}"


def calcular_dimensiones_display(ancho_video: int, alto_video: int,
                                  margen: float = 0.95) -> tuple:
    """
    Calcula las dimensiones de display manteniendo la relación de aspecto.
    Usa ctypes.windll directamente: sin tkinter, sin ventanas extra.
    """
    try:
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
    """
    if not os.path.exists(RUTA_MODELO):
        print(f"\n[ERROR] No se encontró el modelo en: {RUTA_MODELO}")
        print("Asegúrate de colocar 'best.pt' en la misma carpeta que este script.\n")
        sys.exit(1)

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

    # FPS efectivo: los frames realmente procesados por segundo tras el skipping
    fps_efectivo = fps_in / (SALTAR_FRAMES + 1)
    print(f"[INFO] FPS del video: {fps_in:.1f}  |  FPS efectivo (con skipping): {fps_efectivo:.1f}")

    nombre_ventana = "Detector de Personas - Trayectoria  [Q para salir]"

    if ajustar_pantalla:
        disp_w, disp_h = calcular_dimensiones_display(ancho, alto)
    else:
        disp_w, disp_h = ancho, alto

    # Escritor de video (resolución original)
    writer = None
    if guardar:
        if not ruta_salida:
            ruta_salida = nombre_salida_con_timestamp("resultado")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(ruta_salida, fourcc, fps_in, (ancho, alto))
        print(f"[INFO] Guardando resultado en: {ruta_salida}")

    historial      = defaultdict(lambda: deque(maxlen=MAX_PUNTOS))
    frames_sin_ver = defaultdict(int)
    frame_count    = 0
    tick_prev      = cv2.getTickCount()
    fps_display    = 0.0

    # ── NUEVO: acumula todas las mediciones de velocidad por ID ──────────────
    velocidades = defaultdict(list)   # {track_id: [vel_kmh, vel_kmh, ...]}

    print("\n[LISTO] Presiona  Q  para salir.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] Fin del video o no se pudo leer el frame.")
            break

        # ── Frame skipping ────────────────────────
        frame_count += 1
        if frame_count % (SALTAR_FRAMES + 1) != 0:
            continue

        # ── Escalado para videos 4K ───────────────
        frame = escalar_frame(frame)

        # ── Mejora de iluminación (solo inferencia)
        frame_mejorado = mejorar_frame(frame)

        # ── Inferencia + tracking ─────────────────
        resultados = modelo.track(
            frame_mejorado,
            persist   = True,
            tracker   = TRACKER,
            conf      = CONFIANZA_MIN,
            device    = device,
            verbose   = False,
            imgsz     = 640,
        )

        # ── Dibujar + calcular velocidad ──────────
        # NUEVO: se pasa velocidades y fps_efectivo
        frame, ids_activos = dibujar_frame(
            frame, resultados, historial, velocidades, fps_efectivo
        )

        # ── Limpieza de IDs huérfanos ─────────────
        for tid in list(historial.keys()):
            if tid not in ids_activos:
                frames_sin_ver[tid] += 1
                if frames_sin_ver[tid] > FRAMES_LIMITE:
                    del historial[tid]
                    del frames_sin_ver[tid]
            else:
                frames_sin_ver[tid] = 0

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

    # ── NUEVO: exportar promedios de velocidad al CSV ─────────────────────────
    ruta_csv = nombre_salida_con_timestamp("velocidades", extension=".csv")
    guardar_csv(velocidades, ruta_csv)

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
        salida  = nombre_salida_con_timestamp("resultado_camara") if guardar else None
        ejecutar(fuente=0, guardar=guardar, ruta_salida=salida, ajustar_pantalla=True)

    elif opcion == "2":
        ruta = input("\nEscribe la ruta del archivo de video\n"
                     "(o presiona Enter para usar 'videos/prueba.mp4'): ").strip()
        if not ruta:
            ruta = os.path.join("videos", "prueba.mp4")

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