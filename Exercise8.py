import os

# Sembunyikan warning oneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

# 1. Load Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Reshape & Normalisasi
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype("float32") / 255.0

# 3. One-hot Encoding Label
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# 4. Build Model Exercise 4.8 (2 Set Layer Konvolusi)
model = Sequential()

# Layer Set 1
model.add(
    Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu")
)  # Output: (24, 24, 32)
model.add(MaxPooling2D(pool_size=(2, 2)))  # Output: (12, 12, 32)
model.add(Dropout(0.2))

# Layer Set 2 (Penambahan Baru)
model.add(Conv2D(64, (3, 3), activation="relu"))  # Output: (10, 10, 64)
model.add(MaxPooling2D(pool_size=(2, 2)))  # Output: (5, 5, 64)
model.add(Dropout(0.2))

# Flatten & Fully Connected
model.add(Flatten())  # Output Vector: 5 * 5 * 64 = 1,600 nodes
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

# 5. Compile Model
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Tampilkan ringkasan parameter arsitektur
print("\n--- RINGKASAN ARSITEKTUR MODEL EXERCISE 4.8 ---")
model.summary()

# 6. Fit / Train Model
print("\n--- PROSES PELATIHAN (TRAINING) ---")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=200,
    verbose=1,
)

# 7. Evaluasi Akhir
scores = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "=" * 45)
print("HASIL EVALUASI MODEL EXERCISE 4.8:")
print("=" * 45)
print(f"Test Loss     : {scores[0]:.4f}")
print(f"Test Error    : {(100 - scores[1] * 100):.2f}%")
print(f"Test Accuracy : {(scores[1] * 100):.2f}%")