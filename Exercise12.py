import os
import time

# Sembunyikan warning log TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from tensorflow.keras.applications import (
    densenet,  # Ditambahkan untuk Exercise 4.12
    inception_v3,
    mobilenet,
    resnet50,
    vgg16,
)
from tensorflow.keras.preprocessing.image import img_to_array

# ==============================================================================
# PILIH MODEL UNTUK DIGUNAKAN
# ==============================================================================
# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
# model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')

# Modifikasi Exercise 4.12: Menggunakan DenseNet121
model = densenet.DenseNet121(weights="imagenet")
print(model.summary())

# Ukuran input gambar standar (224x224 untuk VGG, ResNet, MobileNet, DenseNet; 299x299 untuk InceptionV3)
image_size = 224

camera = cv2.VideoCapture(0)

while camera.isOpened():
  ok, cam_frame = camera.read()
  if not ok:
    break

  # Preprocessing frame untuk model
  frame = cv2.resize(cam_frame, (image_size, image_size))
  numpy_image = img_to_array(frame)
  image_batch = np.expand_dims(numpy_image, axis=0)

  # Preprocessing sesuai model DenseNet
  # processed_image = vgg16.preprocess_input(image_batch.copy())
  # processed_image = resnet50.preprocess_input(image_batch.copy())
  # processed_image = mobilenet.preprocess_input(image_batch.copy())
  # processed_image = inception_v3.preprocess_input(image_batch.copy())
  processed_image = densenet.preprocess_input(image_batch.copy())

  # Hitung waktu inferensi
  start_time = time.time()
  predictions = model.predict(processed_image, verbose=0)
  inference_time = (time.time() - start_time) * 1000  # ms

  label = decode_predictions(predictions)

  # Visualisasi hasil pada webcam
  text_label = f"DenseNet121: {label[0][0][1]} ({label[0][0][2]*100:.1f}%)"
  text_speed = f"Inference: {inference_time:.0f} ms"

  cv2.putText(
      cam_frame,
      text_label,
      (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 0, 0),
      2,
  )
  cv2.putText(
      cam_frame,
      text_speed,
      (10, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 255, 0),
      2,
  )

  cv2.imshow("video image", cam_frame)

  key = cv2.waitKey(1) & 0xFF
  if key == 27 or key == ord("q"):  # Tekan 'ESC' atau 'q' untuk keluar
    break

camera.release()
cv2.destroyAllWindows()