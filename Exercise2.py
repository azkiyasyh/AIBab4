from sklearn.neural_network import MLPClassifier

# Set up data dengan 3 input (3 fitur per sampel) ==================
# Contoh kombinasi 3 input biner
X = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
]

# Output target biner (misal: logika OR untuk 3 input)
y = [0, 1, 1, 1, 1, 1, 1, 1]

# Inisialisasi & Pelatihan MLP =====================================
# hidden_layer_sizes=(5, 2) artinya 2 hidden layer dengan masing-masing 5 dan 2 neuron
clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)
clf.fit(X, y)

# Testing dengan data baru yang juga memiliki 3 input ==============
test_data = [[2.0, 2.0, 2.0], [-1.0, -2.0, -1.0]]
print("Hasil Prediksi:", clf.predict(test_data))

# Cek dimensi matriks bobot (weights)
print("Dimensi Matriks Bobot:", [coef.shape for coef in clf.coefs_])