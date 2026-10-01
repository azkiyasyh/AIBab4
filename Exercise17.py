import os

# Sembunyikan warning log TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
from matplotlib import pyplot
import numpy as np

# ==============================================================================
# PILIH URUTAN LAYER KONVOLUSI YANG INGIN DIVISUALISASI (1, 2, 3, DST.)
# ==============================================================================
target_conv_index = 2  # Ubah ke 2, 3, atau 4 sesuai kebutuhan

# Filter semua layer yang merupakan Conv2D dari model
conv_layers = [
    layer for layer in model.layers if "conv" in layer.name.lower()
]

if target_conv_index > len(conv_layers):
  print(
      f"Model hanya memiliki {len(conv_layers)} layer Conv2D. Menggunakan layer"
      " terakhir."
  )
  selected_layer = conv_layers[-1]
else:
  selected_layer = conv_layers[target_conv_index - 1]

# Ambil bobot (weights) dan bias dari layer terpilih
filters, biases = selected_layer.get_weights()
s = filters.shape  # Format Shape: (height, width, in_channels, num_filters)

print(
    f"--- VISUALISASI CONVOLUTIONAL LAYER KE-{target_conv_index} ({selected_layer.name}) ---"
)
print(f"Ukuran Kernel (Height x Width) : {s[0]} x {s[1]}")
print(f"Jumlah Input Channel (s[2])    : {s[2]}")
print(f"Jumlah Filter / Output (s[3])  : {s[3]}")

# Normalisasi nilai filter ke rentang 0 - 1 untuk visualisasi
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Tampilkan 4 filter pertama dari layer tersebut
n_filters = min(4, s[3])
n_channels_to_show = min(
    4, s[2]
)  # Batasi visualisasi max 4 channel pertama agar rapi

pyplot.figure(figsize=(10, 2.5 * n_filters))
ix = 1

for i in range(n_filters):
  for j in range(n_channels_to_show):
    ax = pyplot.subplot(n_filters, n_channels_to_show, ix)
    ax.set_xticks([])
    ax.set_yticks([])
    # Visualisasi matriks filter pada channel ke-j dan filter ke-i
    pyplot.imshow(filters[:, :, j, i], cmap="gray")
    if i == 0:
      pyplot.title(f"Channel {j+1}", fontsize=9)
    if j == 0:
      ax.set_ylabel(f"Filter {i+1}", fontsize=9)
    ix += 1

pyplot.tight_layout()
pyplot.show()