# End-to-End Multi-Modal Deep Learning Framework for Intelligent Skin Lesion Screening
### Fusion of Dermoscopic Images and Clinical Data using MobileViT

![Project Poster](poster.png)

## 📌 Project Overview
This repository provides an end-to-end, lightweight, and explainable multi-modal deep learning framework designed for early screening of skin lesions using the benchmark **HAM10000** dataset. 

The framework fuses **Dermoscopic Images** with **21-Dimensional Patient Clinical Metadata** (Age, Sex, Anatomical Site) using a Late Fusion strategy on top of a hybrid **MobileViT-S** backbone.

---

## 🚀 Key Highlights
* **Lightweight Architecture:** Hybrid MobileViT-S (CNN + Vision Transformer) with ~60MB footprint, optimized for offline local and mobile deployment.
* **Multi-Modal Late Fusion:** Fuses visual feature embeddings ($d=384$) with clinical feature vectors ($d=64$) to resolve borderline diagnostic ambiguities.
* **Explainable AI (XAI):** Integrated **Grad-CAM** attention maps highlighting pathological lesion boundaries while ignoring hair, healthy skin, and artifacts.
* **Uncertainty Calibration:** Temperature Scaling for realistic confidence estimation in clinical decisions.
* **Peak Performance:** Achieved **91.31%** test accuracy on the 4 primary clinical categories.

---

## 📊 Experimental Results & Accuracy Progression

| Phase / Model Version | Modality | Target Classes | Key Strategy | Test Accuracy |
| :--- | :--- | :---: | :--- | :---: |
| **Baseline (CNN / Default)** | Image Only | 7 Classes | Default Cross-Entropy | 83.0% |
| **Balanced (v2)** | Image Only | 7 Classes | Cost-Sensitive Class Weighting + Augmentation | 87.4% |
| **Vision Backbone** | Image Only | 4 Classes | Strategic clinical focus + 12-hour deep training | 91.05% |
| **Multi-Modal Fusion (Ours)** | **Image + 21-D Metadata** | **4 Classes** | **Late Fusion MLP + Frozen Vision Backbone** | **91.31%** |

### Target Categories:
1. **MEL:** Melanoma (Malignant)
2. **BCC:** Basal Cell Carcinoma
3. **BKL:** Benign Keratosis-like Lesions
4. **NV:** Melanocytic Nevi (Benign)

---

## 📓 Interactive Training Notebooks (Kaggle)
You can explore and run the full training pipelines directly on Kaggle:
* **[Notebook 1: MultiModal Training Pipeline](https://www.kaggle.com/code/behrooz03/finalproject)** — Contains data preprocessing, MobileViT backbone training, metadata engineering, and Late Fusion.
* **[Notebook 2: Inference & Deployment Pipeline](https://www.kaggle.com/code/behrooz03/skincancerapp)** — Includes model inference, Grad-CAM integration, and the interactive application.

---

## 📦 Pretrained Model Checkpoints
All trained PyTorch weights across different experimental phases are publicly available on Kaggle:

👉 **[Download All Weights on Kaggle (my-skin-cancer-weights)](https://www.kaggle.com/datasets/behrooz03/my-skin-cancer-weights)**

* `best_model.pth` (Baseline 7-class model)
* `best_model_v2.pth` (Balanced 7-class model)
* `best_model_4classes.pth` (12-hour trained 4-class vision backbone)
* `best_multimodal_model.pth` (**Final Multi-Modal model — 91.31% Accuracy**)

---

## 🛠️ Quick Start & Local Execution

1. **Clone Repository:**
   git clone [https://github.com/Farnam03/MultiModal-Skin-Lesion-MobileViT.git](https://github.com/Farnam03/MultiModal-Skin-Lesion-MobileViT.git)
   cd MultiModal-Skin-Lesion-MobileViT

2. **Install Requirements:**
   pip install -r requirements.txt

3. **Download Model Weights:**
   Download `best_multimodal_model.pth` from the Kaggle Weights link above and place it in the root directory.

4. **Run Offline Application:**
   python app.py
