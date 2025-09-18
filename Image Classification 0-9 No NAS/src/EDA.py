import pandas as pd
import numpy as np
from google.colab import drive

drive.mount('/content/drive')

train_data = '/content/drive/My Drive/Datasheet/Image Classification/1-9 Classification/train.csv'
test_data  = '/content/drive/My Drive/Datasheet/Image Classification/1-9 Classification/test.csv'
train = pd.read_csv(train_data)
test  = pd.read_csv(test_data)
train.info()
train.columns
test.info()
test.columns
print('Jumlah Missing Values :\n')
print(train.isnull().sum().sum())
print('Jumlah distribusi label pada data train :\n')
print(train['label'].value_counts())
train['label'].unique()
print(train.shape)
print(test.shape)
print(train.columns[:10])
print(test.columns[:10])

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
sns.countplot(y_train)
plt.title('Distribusi kelas di Train Dataset:')
plt.show()
import matplotlib.pyplot as plt

# PREVIEW IMAGES
plt.figure(figsize=(15,4.5))
for i in range(30):
    plt.subplot(3, 10, i+1)
    plt.imshow(X_train[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.axis('off')
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()
