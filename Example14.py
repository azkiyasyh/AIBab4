import os

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

# Inisialisasi model (Pilih salah satu)
# model = vgg16.VGG16(weights='imagenet')
model = resnet50.ResNet50(weights="imagenet")
# model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')

print(model.summary())

# Inisialisasi kamera dan ukuran gambar
camera = cv2.VideoCapture(0)
image_size = 224  # Gunakan 299 jika memilih InceptionV3

while camera.isOpened():
  ok, cam_frame = camera.read()
  if not ok:
    print("Tidak dapat mengakses kamera.")
    break

  # Preprocessing gambar
  frame = cv2.resize(cam_frame, (image_size, image_size))
  numpy_image = img_to_array(frame)
  image_batch = np.expand_dims(numpy_image, axis=0)

  # Preprocess sesuai model yang dipilih
  # processed_image = vgg16.preprocess_input(image_batch.copy())
  processed_image = resnet50.preprocess_input(image_batch.copy())
  # processed_image = mobilenet.preprocess_input(image_batch.copy())
  # processed_image = inception_v3.preprocess_input(image_batch.copy())

  # Prediksi kelas
  predictions = model.predict(processed_image, verbose=0)
  label = decode_predictions(predictions)

  # Visualisasi hasil prediksi pada frame webcam
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

  # Tampilkan window
  cv2.imshow("video image", cam_frame)

  # Tekan 'ESC' (key 27) atau 'q' untuk keluar
  key = cv2.waitKey(1) & 0xFF
  if key == 27 or key == ord("q"):
    break

# Release kamera dan tutup semua window
camera.release()
cv2.destroyAllWindows()