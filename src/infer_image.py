import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ruta a best.pt")
    ap.add_argument("--source", required=True, help="imagen o carpeta")
    ap.add_argument("--outdir", default="outputs/predict", help="carpeta salida")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="0", help="GPU index (0,1,...)")
    args = ap.parse_args()

    # Permitir CPU si se especifica --device cpu
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA requerida. Usa --device cpu o instala PyTorch con CUDA.")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.predict(
        task="obb",
        source=args.source,
        save=True,                  # guarda imágenes con OBB dibujadas
        save_txt=True,              # guarda predicciones .txt
        imgsz=args.imgsz,
        device=args.device,
        project=args.outdir,
        name="run"
    )
    print(f"Listo. Resultados en: {Path(args.outdir)/'run'}")

if __name__ == "__main__":
    main()
