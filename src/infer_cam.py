import argparse
from ultralytics import YOLO
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="0", help="GPU index (0,1,...)")
    ap.add_argument("--cam", type=int, default=0, help="índice de cámara")
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    # Permitir CPU si se especifica --device cpu
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA requerida. Usa --device cpu o instala PyTorch con CUDA.")
    model = YOLO(args.weights)
    model.predict(
        task="obb",
        source=args.cam,
        show=True,          # ventana con preview en vivo
        imgsz=args.imgsz,
        device=args.device
    )

if __name__ == "__main__":
    main()
