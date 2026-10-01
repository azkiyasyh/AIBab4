from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from keras.models import Sequential

# LeNet-5 Model dengan Jumlah Filter yang Dimodifikasi
model = Sequential()

# Layer Conv1: Ditingkatkan dari 6 menjadi 16 filters
model.add(
    Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(32, 32, 1),
    )
)
model.add(AveragePooling2D(pool_size=(2, 2)))

# Layer Conv2: Ditingkatkan dari 16 menjadi 32 filters
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(AveragePooling2D(pool_size=(2, 2)))

# Layer Flatten & Fully Connected (Dense)
model.add(Flatten())
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=84, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

# Tampilkan ringkasan arsitektur & jumlah parameter
model.summary()