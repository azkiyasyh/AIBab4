import os

# Sembunyikan warning log TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image

# Load model VGG19 dengan bobot prapelatihan ImageNet
model = VGG19(weights="imagenet")

# Tampilkan ringkasan arsitektur model
print(model.summary())