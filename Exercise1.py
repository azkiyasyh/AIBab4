from numpy import array, dot, exp, random

# Set up parameters ===============================================
# Input gerbang logika OR
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])

# Output target untuk gerbang logika OR
outputs = array([[0, 1, 1, 1]]).T

# Inisialisasi bobot acak dengan seed agar hasil konsisten
random.seed(1)
weights = 2 * random.random((2, 1)) - 1


# Define a Single Neuron function =================================
def neuron(inputs, weights):
  output = 1 / (1 + exp(-(dot(inputs, weights))))
  return output


# Train the Neuron ================================================
for iteration in range(50000):
  output = 1 / (1 + exp(-(dot(inputs, weights))))
  # Update bobot menggunakan aturan delta (gradient descent dengan Sigmoid derivative)
  weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test the Neuron =================================================
print("Hasil Pengujian Gerbang Logika OR:")
print("---------------------------------")
for x in inputs:
  prediksi = neuron(x, weights)[0]
  # Menggunakan pembulatan (round) untuk mendapatkan output biner 0 atau 1
  print(f"Input: {x} -> Probabilitas: {prediksi:.4f} | Output Biner: {round(prediksi)}")