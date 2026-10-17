# CivicLens Training Scripts

This folder contains helper scripts that let you **train**, **evaluate**, and **run predictions** with the Ultralytics YOLO model for pothole detection.

---
## 📁 Project layout (relevant to training)
```
CivicLens/
├─ dataset/          # <-- place your downloaded / labeled YOLO dataset here
│   ├─ data.yaml    # defines paths, class names, etc.
│   ├─ images/      # optional; can contain train/ and val/ sub‑folders
│   └─ labels/      # optional; matching label files
├─ training/         # <-- you are here
│   ├─ check_dataset.py   # verifies the dataset structure
│   ├─ train.py           # start training (transfer learning)
│   ├─ evaluate.py        # run validation on the test/val split
│   ├─ predict.py         # run inference on new images
│   ├─ requirements.txt   # pip dependencies for the scripts
│   └─ README.md          # you are reading it now
├─ models/           # after training, best.pt and logs will be stored here
└─ …
```

---
## ✅ Step 0 – Verify the dataset
Before doing any heavy training, make sure the dataset is correctly formatted:
```bash
# Install the tiny helper dependencies first (only once)
python -m venv venv && source venv/Scripts/activate   # Windows PowerShell
pip install -r training/requirements.txt

# Run the checker (it will print class names, image/label counts, and any problems)
python training/check_dataset.py
```
If the script reports **[✅] Dataset structure appears valid** and shows matching image/label counts, you can proceed.

---
## 🛠️ Step 1 – Train the model
```bash
# Activate the same virtual environment (if not already active)
source venv/Scripts/activate

# Example command – you can change the arguments as needed
python training/train.py \
    --data dataset \
    --weights yolov8n.pt \   # any Ultralytics checkpoint, e.g. yolov8s.pt, yolov8m.pt
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --device cpu   # or "cuda" if you have a GPU
```
The script will:
1. Validate the dataset again.
2. Start training using the chosen checkpoint.
3. Store all results under `models/civiclens/` (including `best.pt`).
4. Write a `training_config_used.yaml` file there that records the exact hyper‑parameters and the path to the best weights – useful for reproducibility.

---
## 📊 Step 2 – Evaluate the trained model
```bash
python training/evaluate.py \
    --data dataset \
    --weights models/civiclens/run/weights/best.pt \
    --batch 16 \
    --device cpu   # change to cuda if you have a GPU
```
The script prints the validation metrics (mAP, precision, recall, …) returned by Ultralytics.

---
## 🎯 Step 3 – Run inference on new images
```bash
python training/predict.py \
    --weights models/civiclens/run/weights/best.pt \
    --source test_images   # can be a single image or a folder
    --conf 0.25            # confidence threshold (adjust as you like)
    --device cpu
```
Outputs are saved in a `runs/predict/` directory automatically created by Ultralytics.

---
## ☁️ Running on Google Colab
1. Open a new notebook and mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Clone the repository (or upload the folder) into the Drive.
3. Install the requirements:
   ```bash
   !pip install ultralytics pyyaml
   ```
4. Adjust the script arguments to point to the dataset path inside the Drive, e.g. `--data /content/drive/MyDrive/CivicLens/dataset`.
5. Run the same commands shown above (train/eval/predict).

---
## 🔧 Common error handling
* **Missing `data.yaml`** – The checker will stop with a clear `[❌]` message. Ensure the file exists directly inside `dataset/`.
* **Labels not matching images** – The checker prints which images are missing label files or have malformed lines.
* **Package not installed** – All scripts exit with a helpful reminder to run `pip install -r training/requirements.txt`.
* **GPU not available** – Pass `--device cpu` (default). If you request `cuda` on a machine without a GPU, Ultralytics will fall back to CPU and print a warning.

---
## 📌 Quick validation before a long run
To make sure everything is wired correctly *without* starting a full training run, you can execute a **dry‑run** using the `--epochs 1` flag:
```bash
python training/train.py \
    --data dataset \
    --weights yolov8n.pt \
    --epochs 1 \
    --imgsz 640 \
    --batch 1 \
    --device cpu
```
If this finishes without errors, you are ready for the full training.

---
*Happy coding! 🚀*
