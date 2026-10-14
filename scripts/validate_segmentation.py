"""Compare C++ ONNX boxes/masks against Ultralytics' reference postprocessing."""
import argparse
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "models" / ".ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--cpp-output", required=True)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from ultralytics import YOLO, settings
    settings.update({"sync": False})
    model = YOLO(args.model, task="segment")
    ref = model.predict(args.image, imgsz=416, rect=False, retina_masks=True,
                        conf=0.35, iou=0.45, device="cpu", verbose=False)[0]
    directory = Path(args.cpp_output)
    cpp = json.loads((directory / "replay_result.json").read_text())
    report = []
    for i, candidate in enumerate(cpp["candidates"]):
        ids = [j for j, cls in enumerate(ref.boxes.cls.tolist()) if ref.names[int(cls)] == candidate["class"]]
        if not ids:
            raise AssertionError(f"No reference detection for {candidate['class']}")
        x, y, w, h = candidate["box_xywh"]
        box = np.array([x, y, x + w, y + h])
        j = min(ids, key=lambda j: np.abs(ref.boxes.xyxy[j].cpu().numpy() - box).sum())
        box_error = float(np.abs(ref.boxes.xyxy[j].cpu().numpy() - box).max())
        score_error = abs(float(ref.boxes.conf[j]) - candidate["confidence"])
        mask = cv2.imread(str(directory / f"replay_mask_{i}.png"), 0) > 0
        truth = ref.masks.data[j].cpu().numpy() > 0.5
        overlap = float(np.logical_and(mask, truth).sum() / max(1, np.logical_or(mask, truth).sum()))
        report.append({"class": candidate["class"], "max_box_error_px": box_error,
                       "score_error": score_error, "mask_iou": overlap})
        assert box_error <= 2, report[-1]
        assert score_error < 1e-4, report[-1]
        assert overlap >= 0.90, report[-1]
    assert report, "No detections; this scene cannot validate segmentation decoding"
    (directory / "reference_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
