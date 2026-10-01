from keras.layers import Activation, Dense
from keras.models import Sequential
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time

# Load Dataset
whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, random_state=7
)

# MLP 3-Layer Model (Diubah ke 5 Neurons) ========================
model = Sequential()
model.add(Dense(5, activation="relu", input_shape=[X_train.shape[1]]))
model.add(Dense(5, activation="relu"))
model.add(Dense(1, activation="sigmoid"))  # Output layer untuk klasifikasi biner

print(model.summary())

# Compile Model
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Benchmark Training Time
start_time = time.time()
history = model.fit(
    X_train,
    y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=1,
)
end_time = time.time()

training_time = end_time - start_time

# Evaluasi Model
results = model.evaluate(X_test, y_test, verbose=0)

print("\n================ Hasil Evaluasi (5 Neurons) ================")
print(f"Waktu Pelatihan (Training Time) : {training_time:.4f} detik")
print(f"Test Loss                       : {results[0]:.4f}")
print(f"Test Accuracy                   : {results[1] * 100:.2f}%")