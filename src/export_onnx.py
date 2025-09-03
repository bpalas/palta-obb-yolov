import argparse
from ultralytics import YOLO
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="ruta a best.pt")
    ap.add_argument("--format", default="onnx", choices=["onnx","torchscript","openvino"])
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    model = YOLO(args.weights)
    out = model.export(task="obb", format=args.format, imgsz=args.imgsz)
    print("Exportado:", out)
    if args.format == "onnx":
        print("Sugerencia: onnxruntime-gpu para inferencia acelerada.")

if __name__ == "__main__":
    main()
