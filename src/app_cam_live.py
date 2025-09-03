import time
import os
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image

# ---- detección automática de pesos por defecto ----
def _find_latest_best(root: Path) -> Path | None:
    patterns = [
        "notebooks/outputs/nb_train_obb*/weights/best.pt",
        "outputs/nb_train_obb*/weights/best.pt",
        "outputs/train_obb*/weights/best.pt",
    ]
    cands = []
    for pat in patterns:
        for p in root.glob(pat):
            try:
                if p.is_file():
                    cands.append(p)
            except OSError:
                pass
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

st.set_page_config(page_title="Palta OBB - Cámara en vivo", layout="wide")

# ---- utilidades ----
def plot_image_from_result(result):
    img_bgr = result.plot()            # array BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def check_gpu(device_str):
    if device_str != "cpu" and not torch.cuda.is_available():
        st.error("No se detecta GPU CUDA. Instala PyTorch con CUDA o usa device='cpu'.")
        st.stop()

# ---- sidebar ----
st.sidebar.header("Configuración")
# valor por defecto para pesos: env PALTA_WEIGHTS o último best.pt encontrado
_repo_root = Path(__file__).resolve().parent.parent
_env_weights = os.getenv("PALTA_WEIGHTS", "").strip()
if _env_weights and Path(_env_weights).exists():
    _default_weights = _env_weights
else:
    _best = _find_latest_best(_repo_root)
    _default_weights = str(_best) if _best else "notebooks/outputs/nb_train_obb/weights/best.pt"

weights_path = st.sidebar.text_input("Pesos (.pt) entrenados", _default_weights)
device = st.sidebar.text_input("Dispositivo", "0")  # 0 = primera GPU | "cpu"
imgsz = st.sidebar.selectbox("imgsz", [640, 832, 1024], index=2)
conf = st.sidebar.slider("conf", 0.01, 0.95, 0.15, 0.01)
iou  = st.sidebar.slider("iou", 0.10, 0.95, 0.50, 0.05)
camera_index = st.sidebar.number_input("Índice de cámara", value=0, step=1)
cam_res = st.sidebar.selectbox("Resolucion camara", ["auto", "640x480", "1280x720", "1920x1080"], index=2)
show_fps = st.sidebar.checkbox("Mostrar FPS", value=True)
st.sidebar.caption("Tip: si usas webcam externa, prueba índices 1, 2…")

st.title("🥑 Palta OBB — Demo en cámara en vivo (YOLOv8-OBB)")

# ---- carga de modelo ----
if not Path(weights_path).exists():
    st.error("No se encontró el archivo de pesos .pt. Verifica la ruta.")
    st.stop()

check_gpu(device)

@st.cache_resource(show_spinner=True)
def load_model(wp, dev):
    model = YOLO(wp)
    return model

model = load_model(weights_path, device)

# ---- UI principal ----
colL, colR = st.columns([3,1])
video_placeholder = colL.empty()
status_placeholder = colR.empty()
btn_start = colR.button("▶️ Iniciar", type="primary")
btn_stop  = colR.button("⏹️ Detener")

if "running" not in st.session_state:
    st.session_state.running = False

if btn_start:
    st.session_state.running = True
if btn_stop:
    st.session_state.running = False

# ---- loop de cámara ----
if st.session_state.running:
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)  # DSHOW ayuda en Windows
    # Try to set capture resolution if requested
    try:
        if cam_res != "auto":
            w, h = map(int, cam_res.split("x"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    except Exception:
        pass
    if not cap.isOpened():
        st.error("No se pudo abrir la cámara. Prueba con otro índice.")
        st.session_state.running = False

    prev = time.time()
    frame_count = 0
    avg_fps = 0.0

    while st.session_state.running and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            st.warning("Frame no disponible.")
            break

        # inferencia OBB
        with torch.inference_mode():
            # asegurar memoria contigua (algunas capturas tienen stride irregular)
            frame_bgr = np.ascontiguousarray(frame)
            results = model.predict(
                task="obb",
                source=frame_bgr,      # numpy BGR
                imgsz=int(imgsz),
                conf=float(conf),
                iou=float(iou),
                device=device,
                verbose=False,
                max_det=200,
                persist=True,
            )

        # dibujar y mostrar
        for r in results:
            rendered = plot_image_from_result(r)  # RGB
            video_placeholder.image(rendered, channels="RGB", use_container_width=True)

        # FPS
        if show_fps:
            frame_count += 1
            now = time.time()
            dt = now - prev
            if dt >= 1.0:
                fps = frame_count / dt
                avg_fps = 0.7*avg_fps + 0.3*fps if avg_fps > 0 else fps
                status_placeholder.info(f"FPS: {fps:.1f} (avg {avg_fps:.1f}) • imgsz={imgsz} • conf={conf:.2f} • iou={iou:.2f}")
                prev = now
                frame_count = 0

        # permite detener con el botón
        if not st.session_state.running:
            break

    cap.release()
    status_placeholder.success("Cámara detenida.")
else:
    st.info("Pulsa **Iniciar** para comenzar la demo con la webcam.")
    st.caption("Requisitos: pesos entrenados (.pt) y PyTorch CUDA instalado.")
