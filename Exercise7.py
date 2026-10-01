import os

# Sembunyikan warning oneDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

# 1. Load & Preprocess Data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype("float32") / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype("float32") / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


# Fungsi pembuat model fleksibel untuk Exercise 4.7
def build_model(filters, kernel_size):
  model = Sequential()
  model.add(
      Conv2D(
          filters, kernel_size, input_shape=(28, 28, 1), activation="relu"
      )
  )
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation="relu"))
  model.add(Dense(num_classes, activation="softmax"))

  model.compile(
      loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
  )
  return model


# Running Model 1: Base Model (32 filters, 5x5)
print("\n=================== 1. Base Model (32 Filters, 5x5) ===================")
model1 = build_model(filters=32, kernel_size=(5, 5))
model1.summary()
model1.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=200,
    verbose=1,
)
score1 = model1.evaluate(X_test, y_test, verbose=0)

# Running Model 2: 64 Filters, 5x5
print("\n=================== 2. Variasi 1 (64 Filters, 5x5) ===================")
model2 = build_model(filters=64, kernel_size=(5, 5))
model2.summary()
model2.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=200,
    verbose=1,
)
score2 = model2.evaluate(X_test, y_test, verbose=0)

# Running Model 3: 32 Filters, 3x3
print("\n=================== 3. Variasi 2 (32 Filters, 3x3) ===================")
model3 = build_model(filters=32, kernel_size=(3, 3))
model3.summary()
model3.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=200,
    verbose=1,
)
score3 = model3.evaluate(X_test, y_test, verbose=0)

# Ringkasan Hasil
print("\n" + "=" * 50)
print("HASIL PERBANDINGAN EXERCISE 4.7:")
print("=" * 50)
print(f"1. Base Model (32 Filter, 5x5) -> Akurasi: {score1[1]*100:.2f}%")
print(f"2. Variasi 1  (64 Filter, 5x5) -> Akurasi: {score2[1]*100:.2f}%")
print(f"3. Variasi 2  (32 Filter, 3x3) -> Akurasi: {score3[1]*100:.2f}%")