import pandas as pd
import numpy as np
from google.colab import drive

drive.mount('/content/drive')

train_data = '/content/drive/My Drive/Datasheet/Image Classification/1-9 Classification/train.csv'
test_data  = '/content/drive/My Drive/Datasheet/Image Classification/1-9 Classification/train.csv'
train = pd.read_csv(train_data)
test  = pd.read_csv(test_data)
print(train.shape)
print(test.shape)
print(train.columns[:10])
print(test.columns[:10])

from tensorflow.keras.utils import to_categorical

# Train
X_train = train.drop('label', axis=1)
y_train = train['label']

# Test
X_test = test.drop('label', axis=1)
y_test = test['label']

# Normalisasi
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.10,
    width_shift_range = 0.1,
    height_shift_range = 0.1
)
