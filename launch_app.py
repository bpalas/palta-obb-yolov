#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_latest_best(root: Path) -> Path | None:
    """Busca el best.pt más reciente en rutas conocidas."""
    patterns = [
        "notebooks/outputs/nb_train_obb*/weights/best.pt",
        "outputs/nb_train_obb*/weights/best.pt",
        "outputs/train_obb*/weights/best.pt",
        "**/outputs/nb_train_obb*/weights/best.pt",
    ]
    candidates = []
    for pat in patterns:
        for p in root.glob(pat):
            try:
                if p.is_file():
                    candidates.append(p)
            except OSError:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def copy_to_clipboard(text: str) -> None:
    """Copia texto al portapapeles si es posible (Windows/macOS/Linux)."""
    try:
        if os.name == "nt":  # Windows: usar 'clip'
            # 'clip' espera UTF-16LE
            proc = subprocess.Popen("clip", stdin=subprocess.PIPE, shell=True)
            proc.communicate(input=text.encode("utf-16le"))
        elif shutil.which("pbcopy"):
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=False)
        elif shutil.which("xclip"):
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=False)
    except Exception:
        pass


def detect_cuda_msg() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "CUDA detectada: en la app usa Dispositivo='0' (o el índice de tu GPU)."
        return "Sin CUDA: en la app usa Dispositivo='cpu'."
    except Exception:
        return "(No se pudo verificar CUDA; si falla, usa Dispositivo='cpu')."


def main() -> None:
    parser = argparse.ArgumentParser(description="Lanza la app Streamlit de Palta OBB")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8501")), help="Puerto del servidor Streamlit")
    parser.add_argument("--headless", action="store_true", help="Ejecutar sin abrir navegador")
    parser.add_argument("--weights", type=str, default="auto", help="Ruta al .pt o 'auto' para detectar el último")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    app_path = repo_root / "src" / "app_cam_live.py"
    if not app_path.exists():
        print(f"ERROR: No se encontró la app en {app_path}")
        sys.exit(1)

    # Detectar pesos
    best_path: Path | None
    if args.weights.lower() == "auto":
        best_path = find_latest_best(repo_root)
    else:
        cand = Path(args.weights)
        if not cand.exists():
            cand = (repo_root / args.weights).resolve()
        best_path = cand if cand.exists() else None

    if best_path:
        print("Pesos detectados:", best_path)
        print("Pégalos en la barra lateral (campo 'Pesos (.pt) entrenados').")
        try:
            copy_to_clipboard(str(best_path))
            print("(La ruta se copió al portapapeles)")
        except Exception:
            pass
    else:
        print("No se detectaron pesos automáticamente. Podrás indicar la ruta en la app.")

    # Mensaje de dispositivo
    print(detect_cuda_msg())

    # Verificar Streamlit instalado
    try:
        import streamlit  # noqa: F401
    except Exception:
        print("Streamlit no está instalado. Instala dependencias con:\n  pip install -r requirements.txt")
        sys.exit(1)

    # Construir y lanzar comando
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)]
    if args.headless:
        cmd += ["--server.headless", "true"]

    # preparar variables de entorno para la app
    env = os.environ.copy()
    if best_path:
        env["PALTA_WEIGHTS"] = str(best_path)

    print("Lanzando Streamlit…")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(repo_root), check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Fallo al lanzar Streamlit (código {e.returncode}).")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
