import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense

# BUILD CONVOLUTIONAL NEURAL NETWORK
def build_cnn(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Block 2
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Block 3
    model.add(Conv2D(128, kernel_size=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# Contoh penggunaan
model = build_cnn()
model.summary()

history = model.fit(
    datagen.flow(X_train_sp, y_train_sp),
    epochs=20,
    validation_data =(X_val,y_val)
)
