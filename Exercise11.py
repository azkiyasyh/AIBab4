import os
import time

# Sembunyikan warning log TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import img_to_array

image_size = 224

# Modifikasi: Menggunakan VGG19
print("Memuat model VGG19...")
model = vgg19.VGG19(weights="imagenet")
print(model.summary())

camera = cv2.VideoCapture(0)

while camera.isOpened():
  ok, cam_frame = camera.read()
  if not ok:
    break

  # Preprocessing frame untuk model
  frame = cv2.resize(cam_frame, (image_size, image_size))
  numpy_image = img_to_array(frame)
  image_batch = np.expand_dims(numpy_image, axis=0)
  processed_image = vgg19.preprocess_input(image_batch.copy())

  # Hitung waktu inferensi
  start_time = time.time()
  predictions = model.predict(processed_image, verbose=0)
  inference_time = (time.time() - start_time) * 1000  # dalam ms

  label = decode_predictions(predictions)

  # Format visualisasi teks pada webcam
  text_label = f"VGG19: {label[0][0][1]} ({label[0][0][2]*100:.1f}%)"
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

  # Tekan 'ESC' (key 27) atau 'q' untuk keluar
  key = cv2.waitKey(1) & 0xFF
  if key == 27 or key == ord("q"):
    break

camera.release()
cv2.destroyAllWindows()