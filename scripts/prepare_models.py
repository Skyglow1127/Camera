"""Export official Ultralytics CPU segmentation models to fixed-shape ONNX.
Run with the isolated .venv-models Python; no camera images leave this machine.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "models" / ".ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--models", nargs="+", default=["yolo26n-seg", "yolo11n-seg"])
    args = parser.parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    torch.set_num_threads(4)
    settings.update({"sync": False})
    directory = ROOT / "models"
    directory.mkdir(exist_ok=True)
    for name in args.models:
        model = YOLO(str(directory / f"{name}.pt"))
        exported = Path(model.export(format="onnx", imgsz=args.size, opset=17, simplify=True,
                                     dynamic=False, batch=1, half=False, device="cpu", nms=False))
        names = directory / f"{name}.names"
        names.write_text("\n".join(model.names[i] for i in range(len(model.names))) + "\n")
        manifest = {"model": name, "ultralytics": ultralytics.__version__, "input_size": args.size,
                    "source": "https://github.com/ultralytics/assets/releases",
                    "license": "https://www.ultralytics.com/license", "onnx": exported.name,
                    "sha256": hashlib.sha256(exported.read_bytes()).hexdigest()}
        (directory / f"{name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest))

if __name__ == "__main__":
    main()
