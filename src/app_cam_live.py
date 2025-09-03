import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO


# ---- utilidades ----
def _find_latest_best(root: Path) -> Path | None:
    patterns = [
        "notebooks/outputs/nb_train_obb*/weights/best.pt",
        "outputs/nb_train_obb*/weights/best.pt",
        "outputs/train_obb*/weights/best.pt",
    ]
    cands: list[Path] = []
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


def plot_image_from_result(result) -> np.ndarray:
    img_bgr = result.plot()  # BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def check_gpu(device_str: str) -> None:
    if device_str != "cpu" and not torch.cuda.is_available():
        st.error("No se detecta GPU CUDA. Instala PyTorch con CUDA o usa device='cpu'.")
        st.stop()


st.set_page_config(page_title="Palta OBB - Camara en vivo", layout="wide")


# ---- sidebar ----
st.sidebar.header("Configuracion")
repo_root = Path(__file__).resolve().parent.parent
env_weights = os.getenv("PALTA_WEIGHTS", "").strip()
if env_weights and Path(env_weights).exists():
    default_weights = env_weights
else:
    best = _find_latest_best(repo_root)
    default_weights = str(best) if best else "notebooks/outputs/nb_train_obb/weights/best.pt"

weights_path = st.sidebar.text_input("Pesos (.pt) entrenados", default_weights)
device_default = "0" if torch.cuda.is_available() else "cpu"
device = st.sidebar.text_input("Dispositivo (0/cpu)", device_default)
imgsz = st.sidebar.selectbox("imgsz", [640, 832, 1024], index=0)
conf = st.sidebar.slider("conf", 0.01, 0.95, 0.15, 0.01)
iou = st.sidebar.slider("iou", 0.10, 0.95, 0.50, 0.05)
camera_index = st.sidebar.number_input("Indice de camara", value=0, step=1)
cam_res = st.sidebar.selectbox("Resolucion camara", ["auto", "640x480", "1280x720", "1920x1080"], index=2)
backend_opt = st.sidebar.selectbox(
    "Backend de camara (Windows)", ["auto", "dshow (recomendado)", "msmf"], index=1,
    help="Si no ves imagen o va lento, prueba otro backend"
)
preview_only = st.sidebar.checkbox(
    "Solo previsualizar (sin inferencia)", value=False,
    help="Para probar camara y backend sin el costo del modelo"
)
use_fp16 = st.sidebar.checkbox(
    "FP16 (solo GPU)", value=bool(torch.cuda.is_available()),
    help="Acelera en GPU. No usar en CPU"
)
show_fps = st.sidebar.checkbox("Mostrar FPS", value=True)
st.sidebar.caption("Tip: si usas webcam externa, prueba indices 1, 2...")

st.title("Palta OBB - Demo camara en vivo (YOLOv8-OBB)")


# ---- carga de modelo ----
if not Path(weights_path).exists():
    st.error("No se encontro el archivo de pesos .pt. Verifica la ruta.")
    st.stop()

check_gpu(device)


@st.cache_resource(show_spinner=True)
def load_model(wp: str, dev: str) -> YOLO:
    model = YOLO(wp)
    try:
        if dev != "cpu":
            dev_str = dev if dev.startswith("cuda") else f"cuda:{dev}"
            if torch.cuda.is_available():
                model.to(dev_str)
                torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    return model


model = load_model(weights_path, device)


# ---- UI principal ----
colL, colR = st.columns([3, 1])
video_placeholder = colL.empty()
status_placeholder = colR.empty()
btn_start = colR.button("Iniciar", type="primary")
btn_stop = colR.button("Detener")

if "running" not in st.session_state:
    st.session_state.running = False

if btn_start:
    st.session_state.running = True
if btn_stop:
    st.session_state.running = False


# ---- loop de camara ----
if st.session_state.running:
    # Backend de camara (Windows)
    api = 0
    if os.name == "nt":
        if backend_opt.startswith("dshow"):
            api = cv2.CAP_DSHOW
        elif backend_opt.startswith("msmf"):
            api = cv2.CAP_MSMF

    cap = cv2.VideoCapture(int(camera_index), api) if api != 0 else cv2.VideoCapture(int(camera_index))

    # Ajustes de resolucion y MJPG para subir FPS
    try:
        if cam_res != "auto":
            w, h = map(int, cam_res.split("x"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
    except Exception:
        pass

    if not cap.isOpened():
        st.error("No se pudo abrir la camara. Prueba con otro indice o backend.")
        st.session_state.running = False

    prev = time.time()
    frame_count = 0
    avg_fps = 0.0

    # Info de la camara
    try:
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        status_placeholder.info(f"Camara: {cam_w}x{cam_h} @ ~{cam_fps:.0f} FPS")
    except Exception:
        pass

    while st.session_state.running and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            st.warning("Frame no disponible.")
            break

        if preview_only:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")
        else:
            with torch.inference_mode():
                frame_bgr = np.ascontiguousarray(frame)
                half = bool(use_fp16 and device != "cpu" and torch.cuda.is_available())
                results = model.predict(
                    source=frame_bgr,
                    imgsz=int(imgsz),
                    conf=float(conf),
                    iou=float(iou),
                    device=device,
                    verbose=False,
                    max_det=200,
                    half=half,
                )
            for r in results:
                rendered = plot_image_from_result(r)
                video_placeholder.image(rendered, channels="RGB")

        # FPS
        if show_fps:
            frame_count += 1
            now = time.time()
            dt = now - prev
            if dt >= 1.0:
                fps = frame_count / dt
                avg_fps = 0.7 * avg_fps + 0.3 * fps if avg_fps > 0 else fps
                status_placeholder.info(
                    f"FPS: {fps:.1f} (avg {avg_fps:.1f}) | imgsz={imgsz} | conf={conf:.2f} | iou={iou:.2f}"
                )
                prev = now
                frame_count = 0

        if not st.session_state.running:
            break

    cap.release()
    status_placeholder.success("Camara detenida.")
else:
    st.info("Pulsa Iniciar para comenzar la demo con la webcam.")
    st.caption("Requisitos: pesos entrenados (.pt) y PyTorch CUDA instalado.")
