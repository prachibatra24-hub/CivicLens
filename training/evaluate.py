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
    parser = argparse.ArgumentParser(description='Evaluate a trained YOLO model on the validation set')
    parser.add_argument('--data', type=str, default='dataset', help='Path to dataset directory containing data.yaml')
    parser.add_argument('--weights', type=str, required=True, help='Path to the .pt weights file produced by training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    args = parser.parse_args()

    dataset_dir = Path(args.data)
    data_yaml_path = dataset_dir / 'data.yaml'
    if not data_yaml_path.is_file():
        sys.exit(f"[❌] data.yaml not found: {data_yaml_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit('[❌] Ultralytics YOLO package not installed. Run: pip install ultralytics')

    print('\n[🔧] Evaluation configuration:')
    print(f'   Dataset   : {dataset_dir}')
    print(f'   data.yaml : {data_yaml_path}')
    print(f'   Weights   : {args.weights}')
    print(f'   Batch     : {args.batch}')
    print(f'   Device    : {args.device}')

    model = YOLO(args.weights)
    results = model.val(data=str(data_yaml_path), batch=args.batch, device=args.device)
    print('\n[✅] Evaluation completed. Metrics:')
    print(results)

if __name__ == '__main__':
    main()
