# Hardware-Vision: Project Process Tracker

This file documents the step-by-step progress of the Hardware-Vision PC component recognition and assembly diagnostics pipeline.

## Stage 1: Environment & Project Setup
- [x] Create project subdirectories (`dataset_creation` and `src`).
- [x] Initialize Python virtual environment.
- [x] Define and install dependencies (`requirements.txt`).

## Stage 2: Data Acquisition & Preprocessing (`dataset_creation`)
- [x] Implement query mappings for 11 hardware classes in `dataset_creation/config_data.py`.
- [x] Build the web crawler using `icrawler.builtin.DuckDuckGoImageCrawler` to download 500 images per class.
- [x] Incorporate rate-limiting bounds to the crawler to handle throttling.
- [x] Deduplicate data via MD5 hashing.
- [x] Validate integrity of downloaded images (remove corrupted images via `PIL`).
- [x] Format and Normalize dataset using ImageNet parameters ($\mu$, $\sigma$).
- [x] Split dataset linearly: 80% Train, 10% Validation, 10% Test.

## Stage 3: Multi-Model Architecture Build (`src`)
- [x] Establish configuration for training loop (`src/config.py`).
- [x] Pull and configure architectures: `ResNet50`, `MobileNetV2`, and `EfficientNet-B0` from PyTorch hub.
- [x] Attach custom dense heads for 11 class output after global average pooling and dropout.
- [x] Define Softmax layer and implement transfer learning (freeze backbones).
- [x] Configure `EarlyStopping` and `ReduceLROnPlateau`.

## Stage 4: Advanced Evaluation & Grad-CAM (`src`)
- [x] Design pipeline for classification report and seaborn confusion matrix.
- [x] Establish the 'Confidence Rejection Threshold' logic (< 0.70 $\rightarrow$ "Uncertain").
- [x] Apply Grad-CAM logic pointing at "Bad Cable Management" elements dynamically.

## Stage 5: Final Review & Integration
- [x] Complete robust logging across all modules.
- [x] End-to-end dry run.
