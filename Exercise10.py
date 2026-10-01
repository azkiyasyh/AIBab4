import os
import time

# Sembunyikan warning log TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input as preprocess_vgg16,
)
from tensorflow.keras.applications.vgg19 import (
    VGG19,
    preprocess_input as preprocess_vgg19,
)
from tensorflow.keras.preprocessing import image

# Path Gambar
img_path = "Elephant.jpg"

# Preprocessing Gambar (224x224 piksel untuk keluarga VGG)
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# ==============================================================================
# 1. KLASIFIKASI DENGAN VGG16
# ==============================================================================
print("--- Memuat & Menguji Model VGG16 ---")
model_vgg16 = VGG16(weights="imagenet")
x_vgg16 = preprocess_vgg16(x.copy())

start_time = time.time()
pred_vgg16 = model_vgg16.predict(x_vgg16)
time_vgg16 = time.time() - start_time

results_vgg16 = decode_predictions(pred_vgg16, top=3)[0]

# ==============================================================================
# 2. KLASIFIKASI DENGAN VGG19 (MODIFIKASI EXERCISE 4.10)
# ==============================================================================
print("\n--- Memuat & Menguji Model VGG19 ---")
model_vgg19 = VGG19(weights="imagenet")
x_vgg19 = preprocess_vgg19(x.copy())

start_time = time.time()
pred_vgg19 = model_vgg19.predict(x_vgg19)
time_vgg19 = time.time() - start_time

results_vgg19 = decode_predictions(pred_vgg19, top=3)[0]

# ==============================================================================
# 3. PERBANDINGAN PERFORMA
# ==============================================================================
print("\n" + "=" * 65)
print("HASIL PERBANDINGAN KLASIFIKASI (VGG16 VS VGG19)")
print("=" * 65)

print(
    f"\n[VGG16] - Waktu Inferensi: {time_vgg16:.4f} detik | Total Parameter:"
    " ~138.3 Juta"
)
print("Top-3 Prediksi:")
for idx, (label_id, label, score) in enumerate(results_vgg16, 1):
  print(f"  {idx}. {label:<20} : {score * 100:.2f}%")

print(
    f"\n[VGG19] - Waktu Inferensi: {time_vgg19:.4f} detik | Total Parameter:"
    " ~143.6 Juta"
)
print("Top-3 Prediksi:")
for idx, (label_id, label, score) in enumerate(results_vgg19, 1):
  print(f"  {idx}. {label:<20} : {score * 100:.2f}%")