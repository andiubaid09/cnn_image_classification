import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import keras_tuner as kt

# Build Model dengan Neural Architecture Search
def model_cnn (hp):
  model = models.Sequential()

  # CNN Layer Pertama (Conv2D tetap)
  # Layer pertama menerima input dan layer fix tidak termasuk lagi ke dalam proses NAS
  model.add(layers.Conv2D(
      filters=hp.Int('conv_1_filters', 32, 64, step=32),
      kernel_size = hp.Choice('conv_1_kernel', values=[3,5]),
      activation='relu',
      input_shape=(28,28,1)
  ))
  model.add(layers.MaxPooling2D(pool_size=2))

  # ----- Layer Tambahan (1-3 layer conv opsional)
  for i in range(hp.Int('num_conv_layers', 1,2, step=1)): # NAS akan coba 1-3 layer tambahan
    model.add(layers.Conv2D(
        filters=hp.Int(f'conv_{i+2}_filters', 32, 64, step=32),
        kernel_size=hp.Choice(f'conv_{i+2}_kernel', [3,5]),
        activation=hp.Choice(f'conv_{i+2}_activation', ['relu','elu'])
    ))
    model.add(layers.MaxPooling2D(pool_size=2))

  # ==== Flaten + Dense ====
  model.add(layers.Flatten())
  model.add(layers.Dense(units=hp.Int('dense_units', 32,64, step=32), activation='relu'))
  model.add(layers.Dropout(rate=hp.Float('dropout', 0.2, 0.4, step=0.1)))

  # Output Layers
  model.add(layers.Dense(10,activation='softmax'))

  lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  return model

# NAS
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

tuner = kt.Hyperband(
    model_cnn,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='mnist_tuner',
    project_name='cnn_imageclassification',
    overwrite =True
)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
]
tuner.search(
    X_train, y_train,
    validation_split=0.2,
    epochs = 5, # Epochs per trial awal
    callbacks=callbacks
)
