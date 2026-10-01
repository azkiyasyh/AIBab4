import os
import time

# Sembunyikan warning log TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from tensorflow.keras.applications import (
    inception_v3,
    mobilenet,
    resnet50,
    vgg16,
)
from tensorflow.keras.preprocessing.image import img_to_array

# ==============================================================================
# SELEKSI MODEL MENGGUNAKAN IF-ELIF-ELSE
# Pilihan opsi: "vgg16", "resnet50", "mobilenet", atau "inception_v3"
# ==============================================================================
selected_model_name = "mobilenet"

if selected_model_name == "vgg16":
  print("Memuat model VGG16...")
  model = vgg16.VGG16(weights="imagenet")
  preprocess_fn = vgg16.preprocess_input
  image_size = 224

elif selected_model_name == "resnet50":
  print("Memuat model ResNet50...")
  model = resnet50.ResNet50(weights="imagenet")
  preprocess_fn = resnet50.preprocess_input
  image_size = 224

elif selected_model_name == "mobilenet":
  print("Memuat model MobileNet...")
  model = mobilenet.MobileNet(weights="imagenet")
  preprocess_fn = mobilenet.preprocess_input
  image_size = 224

elif selected_model_name == "inception_v3":
  print("Memuat model InceptionV3...")
  model = inception_v3.InceptionV3(weights="imagenet")
  preprocess_fn = inception_v3.preprocess_input
  image_size = 299

else:
  raise ValueError(
      f"Model '{selected_model_name}' tidak valid! Pilih opsi yang tersedia."
  )

print(model.summary())

# Inisialisasi Kamera
camera = cv2.VideoCapture(0)

while camera.isOpened():
  ok, cam_frame = camera.read()
  if not ok:
    print("Gagal mengambil gambar dari kamera.")
    break

  # Preprocessing frame
  frame = cv2.resize(cam_frame, (image_size, image_size))
  numpy_image = img_to_array(frame)
  image_batch = np.expand_dims(numpy_image, axis=0)

  # Preprocessing otomatis sesuai fungsi variabel preprocess_fn
  processed_image = preprocess_fn(image_batch.copy())

  # Prediksi
  start_time = time.time()
  predictions = model.predict(processed_image, verbose=0)
  inference_time = (time.time() - start_time) * 1000  # ms

  label = decode_predictions(predictions)

  # Visualisasi Teks pada Frame
  text_model = f"Model: {selected_model_name.upper()}"
  text_result = (
      f"{label[0][0][1]}: {label[0][0][2]*100:.1f}% ({inference_time:.0f} ms)"
  )

  cv2.putText(
      cam_frame,
      text_model,
      (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (0, 255, 0),
      2,
  )
  cv2.putText(
      cam_frame,
      text_result,
      (10, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (255, 0, 0),
      2,
  )

  cv2.imshow("video image", cam_frame)

  key = cv2.waitKey(1) & 0xFF
  if key == 27 or key == ord("q"):
    break

camera.release()
cv2.destroyAllWindows()