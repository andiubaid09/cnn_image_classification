# Image Classification with Neural Architecture Search (NAS)  

Repository ini berisi eksperimen klasifikasi gambar menggunakan dataset **MNIST (digit 0–9)** dengan pendekatan **Neural Architecture Search (NAS)** menggunakan **Keras Tuner (Hyperband)** untuk menemukan arsitektur CNN terbaik secara otomatis.  
Berbeda dengan repo *No NAS* yang arsitekturnya tetap, di repo ini CNN dirancang fleksibel dan parameter-parameter penting (filter, kernel, dropout, learning rate, dll) dicari otomatis oleh tuner.  

_Image Classification project using **Neural Architecture Search (NAS)** to automatically optimize CNN architectures for MNIST digits recognition (0–9)._  

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

Catatan: Sama dengan repo *No NAS*, preprocessing dilakukan di awal.  

---

## Model & Neural Architecture Search (NAS)  
- **Model**: Convolutional Neural Network (CNN)  
- **Framework**: TensorFlow / Keras  
- **NAS Framework**: keras-tuner (Hyperband)  
- **Hyperparameter Search Space**:  
  - Filter per layer: [16, 32, 64, 128]  
  - Kernel size: [3, 5]  
  - Dropout rate: [0.3, 0.4]  
  - Learning rate: [1e-2, 1e-3]  
- **Callbacks**:  
  - EarlyStopping (monitor `val_loss`, patience=5, restore best weights)  
  - ReduceLROnPlateau (monitor `val_loss`, factor=0.5, patience=2, min_lr=1e-5)  

👉 NAS digunakan untuk mencari kombinasi hyperparameter terbaik secara otomatis tanpa perlu trial-and-error manual.  

---

## Training & Evaluation  
- **NAS Search**:  
  - `max_epochs = 20`  
  - `factor = 3` (pengurangan epoch per bracket)  
- **Metrics**: Accuracy, Loss  
- **Evaluation**:  
  - Top 3 model terbaik dipilih berdasarkan `val_accuracy`  
  - Model dievaluasi pada `(X_val, y_val)`  
  - Visualisasi loss & accuracy selama training  

---

## Data Augmentation  
- **Methods**:  
  - Rotasi: ±10°  
  - Zoom: ±10%  
  - Perpindahan width/height: ±10%  
- **Purpose**: memperkaya variasi data untuk generalisasi model dan mencegah overfitting.  

---

## Model Download  
- Model terbaik disimpan dalam format `.h5`  

---

## Notes  
- NAS mempercepat pencarian arsitektur CNN dibanding trial manual.  
- Hasil model dari repo ini bisa dibandingkan dengan **Fixed CNN (No NAS)** di repo satunya.  
- Dataset MNIST dapat diganti dataset citra lain dengan preprocessing serupa.  
