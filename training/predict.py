import argparse
import sys
import yaml
from pathlib import Path

def load_config(cfg_path: Path):
    if not cfg_path.is_file():
        sys.exit(f"[❌] Configuration file not found: {cfg_path}")
    try:
        with cfg_path.open('r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        sys.exit(f"[❌] Failed to parse YAML config ({cfg_path}): {e}")
    return cfg

def main():
    parser = argparse.ArgumentParser(description='Run inference with a trained YOLO model on one or more images')
    parser.add_argument('--weights', type=str, required=True, help='Path to the .pt weights file produced by training')
    parser.add_argument('--source', type=str, required=True, help='Path to image file or directory of images to predict on')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('[❌] Ultralytics YOLO package not installed. Run: pip install ultralytics')

    print('\n[🔧] Prediction configuration:')
    print(f'   Weights   : {args.weights}')
    print(f'   Source    : {args.source}')
    print(f'   Confidence: {args.conf}')
    print(f'   Device    : {args.device}')

    model = YOLO(args.weights)
    # Run prediction; results are saved in a folder named "predict" inside the source directory by default
    results = model.predict(source=args.source, conf=args.conf, device=args.device, save=True)
    print('[✅] Prediction completed. Check the "runs/predict" folder for output images with bounding boxes.')

if __name__ == '__main__':
    main()
