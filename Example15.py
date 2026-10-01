import os

# Sembunyikan warning log TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing.image import img_to_array

# 1. Inisialisasi Model
image_size = 224
model = resnet50.ResNet50(weights="imagenet")

# 2. INISIALISASI KAMERA (Baris yang sebelumnya terlewat/hilang)
camera = cv2.VideoCapture(0)

# 3. Loop Pemrosesan Webcam
while camera.isOpened():
  ret, cam_frame = camera.read()
  if not ret:
    print("Gagal mengambil gambar dari kamera.")
    break

  # Preprocessing frame
  frame = cv2.resize(cam_frame, (image_size, image_size))
  numpy_image = img_to_array(frame)
  image_batch = np.expand_dims(numpy_image, axis=0)
  processed_image = resnet50.preprocess_input(image_batch.copy())

  # Prediksi
  predictions = model.predict(processed_image, verbose=0)
  label = decode_predictions(predictions)

  # Visualisasi Teks pada Frame
  text_result = f"{label[0][0][1]}: {label[0][0][2]*100:.1f}%"
  cv2.putText(
      cam_frame,
      text_result,
      (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 0, 0),
      2,
  )

  cv2.imshow("video image", cam_frame)

  # Tekan 'ESC' (27) atau 'q' untuk keluar
  key = cv2.waitKey(1) & 0xFF
  if key == 27 or key == ord("q"):
    break

# Release kamera & tutup semua window
camera.release()
cv2.destroyAllWindows()