# palta-obb-yolov




palta-obb-yolov8/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ LICENSE                        # opcional
├─ configs/
│  ├─ data.yaml                   # el que viene de Roboflow (ajustado a rutas locales)
│  └─ training.yaml               # hiperparámetros opcionales (epochs, imgsz, etc.)
├─ data/                          # NO versionar dataset completo
│  ├─ roboflow/                   # aquí descomprimes el zip (train/valid/test + data.yaml)
│  └─ samples/                    # 5–10 imgs para pruebas rápidas (sí versionar)
├─ outputs/                       # modelos entrenados, logs, métricas (NO versionar)
├─ src/
│  ├─ train_obb.py                # wrapper de entrenamiento YOLOv8-OBB (llama a ultralytics)
│  ├─ infer_image.py              # inferencia sobre imagen/carpeta → guarda resultados
│  ├─ infer_cam.py                # inferencia con webcam/cámara USB
│  ├─ eval_obb.py                 # eval: mAP, confusion, etc. usando ultralytics
│  ├─ export_onnx.py              # export a ONNX/TorchScript
│  ├─ visualize.py                # utils de visualización de OBB y métricas
│  └─ utils.py                    # utilidades comunes (ruta, check gpu, seed, etc.)
└─ app/
   └─ streamlit_app.py            # demo local simple (opcional)


# 0) entorno
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip


pip install torch torchvision torchaudio 

pip install -r requirements.txt

# entreno;

python src/train_obb.py --data configs/data.yaml --model yolov8n-obb.pt --epochs 80 --imgsz 1024 --device 0
 

# launch app

streamlit run src/app_cam_live.py
