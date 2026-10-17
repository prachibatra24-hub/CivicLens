import os
import sys
import glob
import json

def load_yaml(path):
    """Load a YAML file using PyYAML if available.
    Returns a dict or None if the file cannot be parsed.
    """
    try:
        import yaml
    except ImportError:
        print("[!] PyYAML is not installed. Install it with `pip install pyyaml` to read data.yaml.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[!] Failed to parse {path}: {e}")
        return None

def find_images(folder):
    """Return a list of image file paths (common extensions)."""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat), recursive=False))
    return files

def check_labels(image_path, label_folder):
    """Check that a label file exists for the given image and has a valid YOLO format.
    Returns (exists, valid, msg).
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_folder, base + ".txt")
    if not os.path.isfile(label_path):
        return False, False, f"Label file missing: {label_path}"
    # Validate each line – should contain 5 numbers (class_id and 4 bbox values)
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                return True, False, f"Invalid format in {label_path} line {i}: expected 5 numbers, got {len(parts)}"
            try:
                _ = [float(p) for p in parts]
            except ValueError:
                return True, False, f"Non‑numeric value in {label_path} line {i}"
    return True, True, "OK"

def main():
    # Assume this script lives in <project_root>/training
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(project_root, "dataset")
    print("🔎 Checking YOLO dataset in:", dataset_dir)

    # 1️⃣ data.yaml
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        print("[❌] data.yaml not found in the dataset folder.")
        sys.exit(1)
    print("[✔] data.yaml found.")
    yaml_data = load_yaml(yaml_path)
    if yaml_data is None:
        sys.exit(1)

    # Extract useful entries (train/val paths, class names)
    train_path = yaml_data.get("train") or yaml_data.get("train_path")
    val_path = yaml_data.get("val") or yaml_data.get("val_path")
    names = yaml_data.get("names")

    if names and isinstance(names, (list, dict)):
        class_names = list(names) if isinstance(names, list) else list(names.values())
        print(f"[ℹ] Classes ({len(class_names)}): {', '.join(class_names)}")
    else:
        print("[⚠] No class names ("names") found in data.yaml.")
        class_names = []

    # Helper to resolve relative paths
    def resolve(p):
        if not p:
            return None
        if not os.path.isabs(p):
            return os.path.normpath(os.path.join(dataset_dir, p))
        return p

    # 2️⃣ Image / label folders
    for split, path in [("train", train_path), ("val", val_path)]:
        if not path:
            print(f"[⚠] '{split}' path not defined in data.yaml.")
            continue
        img_dir = resolve(path)
        if not os.path.isdir(img_dir):
            print(f"[❌] {split} image folder does not exist: {img_dir}")
            continue
        print(f"[✔] {split} images folder: {img_dir}")
        # Expect a sibling 'labels' folder – YOLO convention
        label_dir = os.path.join(os.path.dirname(img_dir), "labels")
        if not os.path.isdir(label_dir):
            print(f"[⚠] Expected label folder not found next to images: {label_dir}")
            # Continue checking image files but will report missing labels later
        images = find_images(img_dir)
        print(f"    • {len(images)} image file(s) found.")
        if os.path.isdir(label_dir):
            label_files = glob.glob(os.path.join(label_dir, "*.txt"))
            print(f"    • {len(label_files)} label file(s) found.")
        # Per‑image label check
        missing = 0
        invalid = 0
        for img in images:
            exists, valid, msg = check_labels(img, label_dir)
            if not exists:
                missing += 1
                print(f"    [❌] {msg}")
            elif not valid:
                invalid += 1
                print(f"    [❌] {msg}")
        if missing == 0 and invalid == 0:
            print(f"    ✅ All {len(images)} images have matching, well‑formed label files.")
        else:
            print(f"    ⚠️ {missing} missing and {invalid} invalid label files in {split} set.")

    print("\nSummary finished. Review the messages above for any issues.")

if __name__ == "__main__":
    main()
