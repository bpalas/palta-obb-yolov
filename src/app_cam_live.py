import os
from pathlib import Path
import time
import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

# ---- CONFIGURACIÓN FIJA ----
DEVICE = "0" if torch.cuda.is_available() else "cpu"
IMGSZ = 1024
CONF = 0.25
IOU = 0.50

# -------------------------------------------------------------------
# ---- CONFIGURACIÓN DE CÁMARA ----
#
# Elige tu fuente de video:
# 1. Para usar tu celular:
#    - Pon la URL que te da tu app de IP Webcam.
#    - Ejemplo: "http://192.168.1.35:8080/video"
# 2. Para usar una webcam USB:
#    - Usa un número: 0 para la primera, 1 para la segunda, etc.
#
CAMERA_SOURCE = "http://192.168.0.23:8080/video"
# CAMERA_SOURCE = 0  # Descomenta esta línea si quieres volver a usar la webcam
#
# -------------------------------------------------------------------

# ---- Utilidades ----
def plot_image_from_result(result) -> np.ndarray:
    img_bgr = result.plot()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def check_gpu(device_str: str) -> None:
    if device_str != "cpu" and not torch.cuda.is_available():
        st.error("No se detecta GPU con CUDA.")
        st.stop()

# ---- Configuración de la página ----
st.set_page_config(page_title="Palta OBB - Demo", layout="wide")
st.title("🥑 Palta OBB - Demo en vivo")

# ---- Lógica de pesos ----
env_weights = os.getenv("PALTA_WEIGHTS", "").strip()
weights_path = Path(env_weights) if env_weights and Path(env_weights).exists() else Path("outputs/nb_train_obb6/weights/best.pt")
st.caption(f"Usando pesos: `{weights_path.relative_to(Path.cwd())}`")

if not weights_path.exists():
    st.error(f"No se encontró el archivo de pesos en: {weights_path}")
    st.stop()

check_gpu(DEVICE)

# ---- Carga del modelo ----
@st.cache_resource(show_spinner="Cargando modelo de paltas...")
def load_model(wp: Path, dev: str) -> YOLO:
    model = YOLO(wp)
    if dev != "cpu":
        device_str = f"cuda:{dev}"
        if torch.cuda.is_available():
            model.to(device_str)
    return model

model = load_model(weights_path, DEVICE)

# ---- Lógica de la cámara ----
video_placeholder = st.empty()

# Determinar si la fuente es una URL o un índice de webcam
source = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else str(CAMERA_SOURCE)

# Iniciar captura de video
cap = cv2.VideoCapture(source)

# Si la fuente es una webcam USB en Windows, intentar con CAP_DSHOW
if isinstance(source, int) and os.name == "nt":
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

if not cap.isOpened():
    st.error(f"Error al abrir la fuente de video: '{source}'. Verifica la URL o el índice de la cámara.")
    st.stop()

st.success("Cámara conectada. Mostrando video en vivo...")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        st.warning("No se pudo recibir el frame. Intentando reconectar...")
        time.sleep(2)
        cap.release()
        cap = cv2.VideoCapture(source)
        continue

    # Predicción
    with torch.inference_mode():
        results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF, iou=IOU, device=DEVICE, verbose=False)
    
    # Mostrar resultado
    if results:
        rendered_frame = plot_image_from_result(results[0])
        video_placeholder.image(rendered_frame, channels="RGB", use_container_width=True)

cap.release()