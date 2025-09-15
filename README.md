# CNN Image Classification with Neural Architecture Search (NAS)

## Overview
This project implements a **Convolutional Neural Network (CNN)** for image classification, optimized using **Neural Architecture Search (NAS)**. The main dataset used is **MNIST (digits 0–9)**, but the architecture is flexible and can be adapted to other image classification datasets. NAS is employed to automatically find the best network architecture, including the number of convolutional layers, filters, kernel sizes, and activation functions.

---

## Features
- Preprocessing of image data from CSV format
- Normalization and reshaping to `(28, 28, 1)` for CNN input
- Visualization of sample images and label distributions
- Automatic CNN architecture search using **Keras Tuner Hyperband**
- Evaluation using accuracy, loss curves, and classification reports
- Model saving in `.h5` format
- Plotting **training vs validation loss and accuracy**

---

## Repository Structure

