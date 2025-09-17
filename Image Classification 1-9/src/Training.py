import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import keras_tuner as kt

# Build Model dengan Neural Architecture Search
def build_model_cnn (hp):
  model = models.Sequential()

  # Blok 1
  model.add(layers.Conv2D(
      filters=hp.Choice('block1_conv1_filters', [16,32,64]),   # Pencarian kombinasi filters menggunakan NAS
      kernel_size=hp.Choice('block1_conv1_kernel', [3,5]),         # Pencarian kombinasi kernel_size menggunakan NAS
      activation='relu',
      padding='same',
      input_shape=(28,28,1)
  ))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(
      filters=hp.Choice('block1_conv2_filters', [16,32,64]),
      kernel_size=hp.Choice('block2_conv2_kernel', [3,5]),
      activation='relu',
      padding='same'
  ))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2,2)))
  model.add(layers.Dropout(hp.Choice('dropout1', [0.2,0.3])))

  # Blok 2
  model.add(layers.Conv2D(
      filters=hp.Choice('block2_conv1_filters', [32,64]),
      kernel_size=hp.Choice('block2_conv1_kernel', [3,5]),
      activation='relu',
      padding='same'
  ))
  model.add(layers.BatchNormalization())

  model.add(layers.Conv2D(
      filters=hp.Choice('block2_conv2_filters',[32,64]),
      kernel_size=hp.Choice('block2_conv2_kernel',[3,5]),
      activation='relu',
      padding='same'
  ))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPooling2D(pool_size=(2,2)))
  model.add(layers.Dropout(hp.Choice('dropout2',[0.3,0.4])))

  # Blok 3
  model.add(layers.Conv2D(
      filters=hp.Choice('block3_conv1_filters', [64,128]),
      kernel_size=hp.Choice('block3_conv1_kernel', [3,5]),
      activation='relu',
      padding='same'
  ))
  model.add(layers.BatchNormalization())
  model.add(layers.Flatten())
  model.add(layers.Dropout(hp.Choice('dropout3',[0.3,0.5])))

  #Output
  model.add(layers.Dense(10, activation='softmax'))

  #Optimizier
  model.compile(
      optimizer=tf.keras.optimizers.Adam(
          hp.Choice('lr', [1e-3, 5e-4])
      ),
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  return model

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

tuner = kt.Hyperband(
    build_model_cnn,
    objective='val_accuracy',
    executions_per_trial=1,         # 1x training tiap kombinasi
    max_epochs=20,
    factor=3,
    directory='mnist_tuner',
    project_name='cnn_imageclassification',
    overwrite =True
)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
]

tuner.search(
    datagen.flow(X_train_sp, y_train_sp),
    validation_data=(X_val, y_val),
    epochs = 20, # Epochs per trial awal
    #batch_size=128,
    verbose=1,
    callbacks=callbacks
)
