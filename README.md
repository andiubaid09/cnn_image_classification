# Image Classification with CNN: NAS vs No NAS  

Repository ini berisi dua pendekatan dalam klasifikasi gambar menggunakan dataset **MNIST (digit 0–9)**:  
1. **No NAS** → CNN dengan arsitektur tetap (fixed architecture)  
2. **NAS** → CNN dengan Neural Architecture Search menggunakan **keras-tuner (Hyperband)** untuk menemukan kombinasi terbaik filter, kernel, dropout, dll.  

Tujuan repo ini adalah membandingkan hasil model CNN **tanpa pencarian otomatis** vs **dengan pencarian otomatis (NAS)**.  

---

## Dataset  
- **Source**: Kaggle MNIST CSV dataset  
- **Size**: 42,000 rows (train), 42,000 rows (test)  
- **Features**: `pixel0 … pixel783`  
- **Target**: `label` (0–9)  
- **Format**: CSV, diubah menjadi array 28x28 untuk CNN input  
- **Preprocessing**:  
  - Normalisasi ke [0,1]  
  - Reshape ke `(28,28,1)`  
  - One-hot encoding target  

---

## Pendekatan  

### 1. No NAS (Fixed CNN)  
- **Arsitektur CNN**:  
  - Beberapa blok Conv2D + BatchNorm + Dropout  
  - Total ~19 layer termasuk Dense output  
- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Data Augmentation**: rotasi, zoom, shift untuk memperbaiki generalisasi  

### 2. NAS (CNN + keras-tuner)  
- **Framework NAS**: keras-tuner (Hyperband)  
- **Hyperparameter Search Space**:  
  - Filter: [16, 32, 64, 128]  
  - Kernel size: [3, 5]  
  - Dropout: [0.3, 0.4]  
  - Learning rate: [1e-2, 1e-3]  
- **Callbacks**:  
  - EarlyStopping (monitor `val_loss`, patience=5, restore best weights)  
  - ReduceLROnPlateau (monitor `val_loss`, factor=0.5, patience=2, min_lr=1e-5)  

---

## Training & Evaluation  
- **No NAS**:  
  - Training langsung dengan data augmentation  
  - Model terbaik disimpan dalam `.h5`  

- **NAS**:  
  - Tuning dengan `max_epochs=20`, `factor=3`  
  - Top 3 model terbaik dievaluasi dengan `(X_test, y_test)`  
  - Model terbaik disimpan dalam `.h5`  

---


## Notes  
- Repo ini memberikan perbandingan antara **manual design CNN** dan **automated design (NAS)**.  
- Cocok sebagai eksperimen untuk memahami kapan arsitektur fixed cukup baik dan kapan NAS bisa memberikan keuntungan.  
- Dataset bisa diganti dataset citra lain dengan preprocessing yang sama.  
