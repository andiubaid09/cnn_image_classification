# Preview Images
import matplotlib.pyplot as plt

plt.figure(figsize=(15,4.5))
for i in range(30):
  plt.subplot(3, 10, i+1)
  plt.imshow(X_train[i].reshape((28,28)), cmap=plt.cm.binary)
  plt.axis('off')
plt.subplots_adjust(wspace = -0.1, hspace=-0.1)
plt.show()

# Preview Augmented Images
X_train3 = X_train[9,].reshape((1,28,28,1))
y_train3 = y_train[9,].reshape((1,10))
plt.figure(figsize=(15,4.5))
for i in range(30):
  plt.subplot(3, 10, i+1)
  X_train2, y_train2 = next(datagen.flow(X_train3, y_train3))
  plt.imshow(X_train2[0].reshape((28,28)), cmap=plt.cm.binary)
  plt.axis('off')
  if i==9: X_train3 = X_train[11, ].reshape((1, 28, 28, 1))
  if i==19: X_train3 = X_train[18,].reshape((1, 28, 28, 1))
plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
plt.show()
