import os

# Sembunyikan warning log TensorFlow & CPU feature guard
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input, InputLayer, Reshape
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import numpy as np

# 1. Load MNIST Dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# 2. Build Autoencoder
def build_autoencoder(img_shape, code_size):
  # The encoder
  encoder = Sequential()
  encoder.add(InputLayer(shape=img_shape))
  encoder.add(Flatten())
  encoder.add(Dense(code_size))

  # The decoder
  decoder = Sequential()
  decoder.add(InputLayer(shape=(code_size,)))

  # Perbaikan: Konversi ke Python int biasa
  decoder.add(Dense(int(np.prod(img_shape))))

  decoder.add(Reshape(img_shape))

  return encoder, decoder


IMG_SHAPE = x_train[0].shape  # (28, 28)
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)

inp = Input(shape=IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer="adamax", loss="mse")

print("\n--- RINGKASAN ARSITEKTUR AUTOENCODER ---")
print(autoencoder.summary())

# 3. Train Autoencoder
print("\n--- PROSES TRAINING ---")
history = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=20,
    batch_size=256,
    validation_data=(x_test, x_test),
    verbose=1,
)

# 4. Plot Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.title("Model Loss (Autoencoder Training)")
plt.ylabel("Loss (MSE)")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()


# 5. Visualisasi Hasil Rekonstruksi
def show_image(x):
  plt.imshow(x, cmap="gray")
  plt.axis("off")


def visualize(img, encoder, decoder):
  code = encoder.predict(img[None], verbose=0)[0]
  reco = decoder.predict(code[None], verbose=0)[0]

  plt.figure(figsize=(9, 3))

  plt.subplot(1, 3, 1)
  plt.title("Original")
  show_image(img)

  plt.subplot(1, 3, 2)
  plt.title("Code (Latent)")
  plt.imshow(code.reshape([code.shape[-1] // 4, -1]), cmap="gray")
  plt.axis("off")

  plt.subplot(1, 3, 3)
  plt.title("Reconstructed")
  show_image(reco)

  plt.tight_layout()
  plt.show()


for i in range(5):
  img = x_test[i]
  visualize(img, encoder, decoder)