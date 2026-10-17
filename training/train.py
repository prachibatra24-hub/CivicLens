import argparse
import os
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

def validate_dataset(dataset_dir: Path, cfg: dict):
    yaml_path = dataset_dir / 'data.yaml'
    if not yaml_path.is_file():
        sys.exit(f"[❌] data.yaml not found in dataset directory: {yaml_path}")
    for split in ['train', 'val']:
        split_path = cfg.get(split)
        if not split_path:
            print(f"[⚠] '{split}' entry missing in data.yaml; skipping validation for this split.")
            continue
        img_dir = (dataset_dir / split_path) if not Path(split_path).is_absolute() else Path(split_path)
        if not img_dir.is_dir():
            sys.exit(f"[❌] {split} image folder does not exist: {img_dir}")
        label_dir = img_dir.parent / 'labels' / split
        if not label_dir.is_dir():
            print(f"[⚠] Expected labels folder not found: {label_dir}")
    print("[✅] Dataset structure appears valid.")

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for CivicLens pothole detection')
    parser.add_argument('--data', type=str, default='dataset', help='Path to dataset directory containing data.yaml')
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='Pre‑trained YOLO checkpoint (Ultralytics model name or local .pt file)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    args = parser.parse_args()

    dataset_dir = Path(args.data)
    if not dataset_dir.is_dir():
        sys.exit(f"[❌] Dataset directory does not exist: {dataset_dir}")

    data_yaml_path = dataset_dir / 'data.yaml'
    data_cfg = load_config(data_yaml_path)
    validate_dataset(dataset_dir, data_cfg)

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('[❌] Ultralytics YOLO package not installed. Run: pip install ultralytics')

    output_dir = Path('models') / 'civiclens'
    output_dir.mkdir(parents=True, exist_ok=True)

    print('\n[🔧] Training configuration:')
    print(f'   Dataset          : {dataset_dir}')
    print(f'   data.yaml        : {data_yaml_path}')
    print(f'   Pre‑trained      : {args.weights}')
    print(f'   Epochs           : {args.epochs}')
    print(f'   Image size       : {args.imgsz}')
    print(f'   Batch size       : {args.batch}')
    print(f'   Device           : {args.device}')
    print(f'   Output folder    : {output_dir}')

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_dir),
        name='run',
        exist_ok=True,
    )
    best_path = results.best
    print(f"[✅] Training finished. Best weights saved to: {best_path}")
    cfg_record_path = output_dir / 'training_config_used.yaml'
    record = {
        'data_dir': str(dataset_dir),
        'data_yaml': str(data_yaml_path),
        'pretrained': args.weights,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'best_weights': str(best_path),
    }
    with cfg_record_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(record, f)
    print(f"[ℹ] Training configuration saved to: {cfg_record_path}")

if __name__ == '__main__':
    main()
