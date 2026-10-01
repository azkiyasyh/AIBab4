import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Menghilangkan warning oneDNN

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

# Create model
model = Sequential()

# Layer Set 1 (Sesuai kode awal)
model.add(
    Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu")
)  # Output shape: (24, 24, 32)
model.add(MaxPooling2D(pool_size=(2, 2)))  # Output shape: (12, 12, 32)
model.add(Dropout(0.2))

# ==============================================================================
# MODIFIKASI EXERCISE 4.6: Menambahkan Conv2D, MaxPooling2D, dan Dropout ke-2
# ==============================================================================
model.add(
    Conv2D(64, (3, 3), activation="relu")
)  # Output shape: (10, 10, 64)
model.add(MaxPooling2D(pool_size=(2, 2)))  # Output shape: (5, 5, 64)
model.add(Dropout(0.2))

# Flatten & Fully Connected Layers
model.add(Flatten())  # Output shape: 5 * 5 * 64 = 1.600
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

# Compile model
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Tampilkan ringkasan arsitektur
print(model.summary())