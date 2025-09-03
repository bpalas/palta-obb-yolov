import argparse, os, random
import numpy as np
import torch
from ultralytics import YOLO

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data.yaml", help="Ruta al data.yaml")
    ap.add_argument("--model", default="yolov8n-obb.pt", help="n/s/m/l/x-obb.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    ap.add_argument("--device", default="0", help="GPU index, e.g., 0 o '0,1'")
    ap.add_argument("--project", default="outputs", help="Carpeta raíz de resultados")
    ap.add_argument("--name", default="train_obb", help="Subcarpeta del experimento")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30, help="Early stopping (epochs)")
    args = ap.parse_args()

    set_seed(args.seed)

    device_str = str(args.device).strip().lower()
    if device_str != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError("No hay GPU CUDA disponible. Usa --device cpu o instala PyTorch con CUDA.")
        # Intento de mostrar la GPU seleccionada (primer índice)
        try:
            first_idx = int(device_str.split(',')[0]) if device_str and device_str[0].isdigit() else 0
        except Exception:
            first_idx = 0
        print(">> GPU:", torch.cuda.get_device_name(first_idx))
    else:
        print(">> Ejecutando en CPU (entrenamiento más lento)")

    model = YOLO(args.model)  # p.ej. yolov8n-obb.pt
    model.train(
        task="obb",
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,         # usa GPU
        project=args.project,
        name=args.name,
        workers=os.cpu_count()//2 if os.cpu_count() else 4,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,                   # lr base de Ultralytics
        patience=args.patience,     # early stopping
        verbose=True
    )

if __name__ == "__main__":
    main()
