# CivicLens Project Structure

This repository will hold the code and data for the **CivicLens** AI model that detects local problems (potholes, garbage piles, etc.) from images.

## Folder Overview

- **`dataset/`** – Will store the raw images and the corresponding annotation files (e.g., YOLO `.txt` labels). This is where you place the data you collect or download.
- **`training/`** – Contains scripts, configuration files, and any auxiliary resources needed to train the model. For now it will be empty, but later you will add a training script that uses PyTorch/YOLO.
- **`models/`** – After training, the exported model files (e.g., `best.pt`, `weights/`) will be saved here so the application can load them for inference.
- **`test_images/`** – A small set of images you can use to quickly test the trained model and see predictions without running the full training pipeline.
- **`app/`** – The web or desktop application code that will allow users to upload images, view predictions, and report issues.

## Next Steps
1. Verify that Python is installed on your machine.
2. Set up a virtual environment.
3. Install the required libraries (e.g., PyTorch, Ultralytics YOLO).
4. Prepare your labeled dataset inside the `dataset/` folder.
5. Write a training script in the `training/` folder.

---
*This README was generated automatically to guide you through the first steps of building CivicLens.*