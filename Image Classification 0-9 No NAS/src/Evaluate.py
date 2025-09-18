prediction = model.predict(X_test)
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

print(f'Akurasi Training dari model ini adalah:{training_accuracy[-1]:.4f}')
print(f'Akurasi Validation dari model ini adalah:{validation_accuracy[-1]:.4f}')

plt.figure(figsize=(12,6))
for i in range(10):
  plt.subplot(2,5,i+1)
  plt.imshow(X_test[i].reshape(28,28),cmap='gray')
  plt.title(f'Pred:{prediction[i]}')
  plt.axis('off')
plt.show()

plt.figure(figsize=(10,6))

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy', color ='#8502d1')
plt.plot(history.history['val_accuracy'], label='validation Accuracy', color='darkorange')

plt.title('Accuracy Evolution')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from google.colab import files
model.save('image_classification_CNN.h5')
files.download('image_classification_CNN.h5')
