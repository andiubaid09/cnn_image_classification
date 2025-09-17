X_train = train.drop('label', axis=1).to_numpy()
y_train = train['label'].to_numpy()
print(f'Shape train X: {X_train.shape}, y:{y_train.shape}')
#Preprocessing
from tensorflow.keras.utils import to_categorical

X_test = test.to_numpy()
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes = 10)

# Visualisasi beberapa contoh gambar
plt.figure(figsize=(15,4.5))
for i in range(30):
  plt.subplot(3, 10, i+1)
  plt.imshow(X_train[i].reshape((28,28)), cmap=plt.cm.binary)
  plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=-0.1)
plt.show()

from sklearn.model_selection import train_test_split

X_train_sp, X_val, y_train_sp, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)
print('Train:',X_train_sp.shape, y_train_sp.shape)
print('Val :',X_val.shape, y_val.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.10,
        width_shift_range=0.1,
        height_shift_range=0.1)
datagen.fit(X_train_sp)
