# Image Classification with Neural Architecture Search (NAS)

Repository ini berisi eksperimen klasifikasi gambar menggunakan dataset MNIST (digit 0–9) dengan pendekatan **Neural Architecture Search (NAS)** untuk menemukan arsitektur CNN optimal. Model dilatih untuk mengenali angka dari dataset MNIST dengan augmentasi data dan hyperparameter tuning otomatis.

> Image Classification project using Neural Architecture Search (NAS) to automatically optimize CNN architectures for MNIST digits recognition (0–9).

---

## Dataset

- **Source**: Kaggle MNIST CSV dataset  
- **Size**: 42000 rows (train), 42000 rows (test)  
- **Features**: `pixel0` … `pixel783`  
- **Target**: `label` (0–9)  
- **Format**: CSV, diubah menjadi array 28x28 untuk CNN input  
- **Preprocessing**: Normalisasi ke [0,1], reshape `(28,28,1)`, one-hot encoding target  

> Catatan: Semua preprocessing dilakukan di awal agar data siap untuk CNN, termasuk normalisasi piksel dan reshaping ke bentuk citra 2D.

---

## Model & Neural Architecture Search (NAS)

- **Model**: Convolutional Neural Network (CNN)  
- **NAS Framework**: `keras-tuner` (Hyperband)  
- **Tuning Scope**:
  - Jumlah layer konvolusi tambahan: 1–2
  - Filter per layer: 32–64
  - Kernel size: 3 atau 5
  - Aktivasi: ReLU, ELU
  - Dense units: 32–64
  - Dropout rate: 0.2–0.4
  - Learning rate: 0.02, 0.04
- **Callbacks**:
  - EarlyStopping (monitor `val_loss`, patience=5)
  - ReduceLROnPlateau (monitor `val_loss`, factor=0.5, patience=3, min_lr=1e-5)

> NAS digunakan untuk mencari kombinasi hyperparameter terbaik secara otomatis tanpa perlu trial-and-error manual.

---

## Training & Evaluation

 - **Epochs per trial**: 20 
- **Metrics**: Accuracy, Loss  
- **Evaluation**:
  - Top 3 models dievaluasi pada data test  
  - Visualisasi loss & accuracy selama training  
---

## Data Augmentation

- **Augmentation Methods**:
  - Rotasi: ±10°
  - Zoom: ±10%
  - Perpindahan width/height: ±10%
- **Tujuan**: Meningkatkan variasi data untuk generalisasi model lebih baik.

---

## Model Download

- Model terbaik disimpan dalam format `.h5`  
---

## Notes

- NAS mempercepat pencarian arsitektur dibanding trial manual  
- Model CNN lebih ringan daripada ensemble tree-based untuk dataset citra  
- Dataset MNIST dapat diganti dataset citra lain dengan preprocessing serupa

---
