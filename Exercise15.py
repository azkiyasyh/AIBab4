import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from matplotlib import pyplot
import numpy as np

# ==============================================================================
# EXERCISE 4.15: Memilih Layer Konvolusi Kedua (n = 2)
# ==============================================================================
# Catatan: Pastikan variabel 'model' merujuk ke model CNN yang memiliki
# lebih dari 1 layer Conv2D (misalnya model dari Exercise 4.8)
n = 2  # Mengubah dari n = 1 ke n = 2

filters, biases = model.layers[n].get_weights()
s = filters.shape

print("--- INFORMASI FILTER LAYER KE-2 ---")
print("Kernel height/width :", s[0], s[1])
print("Input Channels      :", s[2])
print("Total Filter Count  :", s[3])

# Normalisasi nilai filter ke rentang 0-1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Plot 4 filter pertama
n_filters, ix = 4, 1
pyplot.figure(figsize=(10, 10))

for i in range(n_filters):
  f = filters[:, :, :, i]
  # Loop untuk channel input (s[2] menggantikan s[0] untuk dimensi channel pada Keras/TF)
  for j in range(s[2]):
    ax = pyplot.subplot(n_filters, s[2], ix)
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.imshow(f[:, :, j], cmap="gray")
    ix += 1

pyplot.tight_layout()
pyplot.show()