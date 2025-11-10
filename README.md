# Dog's Breed Identification 🐕 – Full Project

## 📘 Overview

Dog Breed Identification is a deep learning project designed to classify dog breeds from images using convolutional neural networks (CNNs). This project replicates and enhances the **Kaggle Dog Breed Identification** challenge by offering a modular, research‑grade implementation in **PyTorch**.

The model uses transfer learning with architectures like **ResNet‑18**, **ResNet‑34**, and **ResNet‑50**, pre‑trained on ImageNet. The framework supports flexible configuration through YAML files and can be extended for other animal species or fine‑grained classification tasks.

> **Author:** [Mobin Yousefi](https://github.com/mobinyousefi-cs)  
> **License:** MIT  
> **Created:** November 2025

---

## 🧠 Key Features

✅ Clean and professional modular code structure  
✅ Configurable model and training parameters via YAML  
✅ Transfer learning using ResNet backbones  
✅ Train/Validation split with LabelEncoder and stratification  
✅ Checkpointing, logging, and reproducibility  
✅ Easy inference on single or multiple images  
✅ Ready for CI/CD integration and testing  

---

## 🏗️ Project Structure

```text
dogs-breed-identification/
├── LICENSE
├── README.md
├── pyproject.toml
├── .gitignore
├── .editorconfig
├── configs/
│   └── training_config.yaml
├── src/
│   └── dogs_breed_identification/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── models.py
│       ├── train.py
│       ├── evaluate.py
│       ├── infer.py
│       └── utils/
│           ├── __init__.py
│           ├── logging_utils.py
│           ├── seed_utils.py
│           └── metrics.py
└── tests/
    ├── test_data_loader.py
    └── test_models.py
```

---

## 💾 Dataset

Dataset: [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)

After downloading, structure your data as:

```text
data/
  ├── train/           # training images (.jpg)
  ├── test/            # test images (.jpg)
  └── labels.csv       # Kaggle labels file
```

If you need to automatically download and extract via Kaggle API, run:

```bash
kaggle competitions download -c dog-breed-identification
unzip dog-breed-identification.zip -d data/
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mobinyousefi-cs/dogs-breed-identification.git
   cd dogs-breed-identification
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .[dev]
   ```

---

## 🚀 Training the Model

You can start training using:

```bash
python -m dogs_breed_identification.train --config configs/training_config.yaml
```

Training logs and checkpoints will be saved in the `logs/` and `models/` directories respectively.

You can modify the model, epochs, batch size, and more in `configs/training_config.yaml`.

Example config section:

```yaml
model:
  name: "resnet50"
  pretrained: true
  freeze_backbone: false

training:
  batch_size: 32
  num_epochs: 20
  learning_rate: 0.0003
```

---

## 📊 Evaluation

After training, evaluate the model with:

```bash
python -m dogs_breed_identification.evaluate \
  --config configs/training_config.yaml \
  --checkpoint models/resnet50_dogs_breed_baseline_best.pt
```

This will compute the final validation accuracy and log it in `logs/evaluate.log`.

---

## 🔍 Inference (Breed Prediction)

Run inference on one or more dog images:

```bash
python -m dogs_breed_identification.infer \
  --checkpoint models/resnet50_dogs_breed_baseline_best.pt \
  --image path/to/dog1.jpg path/to/dog2.jpg
```

Expected output:

```text
path/to/dog1.jpg: golden_retriever (0.9821)
path/to/dog2.jpg: german_shepherd (0.9453)
```

---

## 🧩 Configuration System

All configurations (paths, hyperparameters, augmentations) are defined in YAML.
Example:

```yaml
paths:
  train_dir: data/train
  test_dir: data/test
  labels_csv: data/labels.csv
  model_dir: models
  logs_dir: logs
```

You can create multiple config files for different experiments.

---

## 🧪 Testing

Unit tests are provided for the data loader and model.
Run them via:

```bash
pytest -v
```

---

## 🧰 Automation (Windows Users)

To make execution easier, you can create a **`run_project.bat`** file in the root directory:

**File:** `run_project.bat`

```bat
@echo off
:: ==============================================================
:: Project: Dog's Breed Identification
:: Author: Mobin Yousefi
:: GitHub: https://github.com/mobinyousefi-cs
:: Description: Automates environment setup and model training.
:: ==============================================================

REM Activate virtual environment
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
) else (
    echo Virtual environment not found. Creating one...
    python -m venv .venv
    call .venv\Scripts\activate
)

REM Install dependencies
pip install -e .[dev]

REM Train model
python -m dogs_breed_identification.train --config configs/training_config.yaml

echo Training complete! Press any key to exit.
pause >nul
```

You can double‑click `run_project.bat` to automatically create a virtual environment, install dependencies, and start training.

---

## 📈 Future Improvements

- ✅ Add MobileNetV3 or EfficientNet backbones for lightweight models
- ✅ Add Grad‑CAM visualizations for interpretability
- ✅ Add TensorBoard or WandB integration
- ✅ Add automatic dataset downloader script

---

## 📚 References

- Kaggle Dog Breed Identification Challenge: [https://www.kaggle.com/c/dog-breed-identification](https://www.kaggle.com/c/dog-breed-identification)  
- PyTorch Official Documentation: [https://pytorch.org/](https://pytorch.org/)  
- Torchvision Models: [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)

---

## 🧑‍💻 Author

**Mobin Yousefi**  
GitHub: [mobinyousefi-cs](https://github.com/mobinyousefi-cs)  
LinkedIn: [Mobin Yousefi](https://linkedin.com/in/mobin-yousefi)

> *This project is part of Mobin Yousefi’s deep learning series showcasing professional‑grade machine learning projects for academic and research purposes.*

