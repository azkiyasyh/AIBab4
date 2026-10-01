import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Menghilangkan warning oneDNN

import keras
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential

# Create the AlexNet model
model = Sequential()

# 1st Convolutional Layer
model.add(
    Conv2D(
        filters=96,
        input_shape=(224, 224, 3),
        kernel_size=(11, 11),
        activation="relu",
        strides=(4, 4),
        padding="valid",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# 2nd Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(11, 11),
        activation="relu",
        strides=(1, 1),
        padding="valid",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# 3rd Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        activation="relu",
        strides=(1, 1),
        padding="valid",
    )
)

# 4th Convolutional Layer
model.add(
    Conv2D(
        filters=384,
        kernel_size=(3, 3),
        activation="relu",
        strides=(1, 1),
        padding="valid",
    )
)

# 5th Convolutional Layer
model.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation="relu",
        strides=(1, 1),
        padding="valid",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

# Flatten Output dari Conv Layers
model.add(Flatten())

# ==============================================================================
# MODIFIKASI EXERCISE 4.5: Mengurangi unit pada Dense Layers
# Model Asli : 4096 -> 4096 -> 1000 -> 17
# Modifikasi : 1024 -> 512  -> 256  -> 17
# ==============================================================================

# 1st Fully Connected Layer (Diubah dari 4096 ke 1024)
# Catatan: parameter input_shape pada Dense di sini opsional karena sudah ada Flatten()
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.4))

# 2nd Fully Connected Layer (Diubah dari 4096 ke 512)
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.4))

# 3rd Fully Connected Layer (Diubah dari 1000 ke 256)
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(17, activation="softmax"))

# Tampilkan Ringkasan Arsitektur
model.summary()

# Compile the model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer="adam",
    metrics=["accuracy"],
)