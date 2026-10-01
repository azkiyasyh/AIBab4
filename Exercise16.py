import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import math
from matplotlib import pyplot
import numpy as np

# ==============================================================================
# EXERCISE 4.16: Menampilkan Seluruh Filter (n_filters = s[3])
# ==============================================================================
n = 0  # Ambil layer Conv2D pertama (atau ganti sesuai layer yang ingin dilihat)

filters, biases = model.layers[n].get_weights()
s = filters.shape

# Normalisasi nilai filter ke rentang 0-1
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Mengubah jumlah filter menjadi seluruh filter yang ada di layer tersebut (s[3])
n_filters, ix = s[3], 1

# Menghitung tata letak grid subplot otomatis agar tampilan rapi
num_cols = 8  # Tampilkan 8 filter per baris
num_rows = math.ceil(n_filters / num_cols)

pyplot.figure(figsize=(12, 2 * num_rows))

for i in range(n_filters):
  f = filters[:, :, :, i]
  ax = pyplot.subplot(num_rows, num_cols, i + 1)
  ax.set_xticks([])
  ax.set_yticks([])
  # Mengambil channel pertama (index 0) dari filter
  pyplot.imshow(f[:, :, 0], cmap="gray")
  pyplot.title(f"F-{i+1}", fontsize=8)

pyplot.tight_layout()
pyplot.show()