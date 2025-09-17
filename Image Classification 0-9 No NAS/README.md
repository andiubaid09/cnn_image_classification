# Image Classification with Convolutional Neural Networks (CNN) – No NAS  

Repository ini berisi eksperimen klasifikasi gambar menggunakan dataset **MNIST (digit 0–9)** dengan pendekatan **CNN standar tanpa Neural Architecture Search (NAS)**.  
Model CNN dibangun secara manual dengan arsitektur tetap (fixed architecture), dilatih untuk mengenali angka dari dataset MNIST dengan bantuan **data augmentation** agar model lebih robust.  

_Image Classification project using a manually designed **Convolutional Neural Network (CNN)** for MNIST digits recognition (0–9). Unlike the NAS version, this project does **not** use automatic hyperparameter search – the architecture is fixed and predefined._  

---

## Dataset  
- **Source**: Kaggle MNIST CSV dataset  
- **Size**: 42,000 rows (train), 42,000 rows (test)  
- **Features**: `pixel0 … pixel783`  
- **Target**: `label` (0–9)  
- **Format**: CSV, diubah menjadi array 28x28 untuk input CNN  
- **Preprocessing**:  
  - Normalisasi ke [0,1]  
  - Reshape ke `(28,28,1)`  
  - One-hot encoding target  

Catatan: Semua preprocessing dilakukan sebelum training agar data siap dipakai CNN.  

---

## Model – Fixed CNN (No NAS)  
- **Model Type**: Convolutional Neural Network (CNN)  
- **Framework**: TensorFlow / Keras  
- **Arsitektur**:  
  - `Conv2D(32, kernel=3)` → BatchNorm → `Conv2D(32, kernel=3)` → BatchNorm → `Conv2D(32, kernel=5, stride=2, padding=same)` → BatchNorm → Dropout(0.4)  
  - `Conv2D(64, kernel=3)` → BatchNorm → `Conv2D(64, kernel=3)` → BatchNorm → `Conv2D(64, kernel=5, stride=2, padding=same)` → BatchNorm → Dropout(0.4)  
  - `Conv2D(128, kernel=4)` → BatchNorm → Flatten → Dropout(0.4)  
  - `Dense(10, softmax)`  
- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Metrics**: Accuracy  

👉 Tidak ada proses pencarian otomatis hyperparameter (seperti NAS). Semua nilai filter, kernel, dropout, dll ditentukan secara manual.  

---

## Training & Evaluation  
- **Epochs**: 20–30 (disesuaikan dengan early stopping)  
- **Metrics**: Accuracy, Loss  
- **Evaluation**: Model dievaluasi pada data val setelah training selesai  
- **Visualization**: Grafik loss & accuracy selama training  

---

## Data Augmentation  
- **Methods**:  
  - Rotasi: ±10°  
  - Zoom: ±10%  
  - Perpindahan width/height: ±10%  
- **Purpose**: menambah variasi data agar model generalisasi lebih baik dan mengurangi overfitting.  

---

## Model Download  
- Model terbaik disimpan dalam format `.h5`  

---

## Notes  
- CNN manual ini dijadikan baseline sebelum eksperimen NAS.  
- Hasil model bisa dijadikan perbandingan: **fixed CNN vs NAS CNN**.  
- Dataset MNIST bisa diganti dataset citra lain dengan preprocessing serupa.  
