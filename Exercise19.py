import os

# Sembunyikan warning log TensorFlow / Transformers
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from transformers import pipeline

# Inisialisasi pipeline sentiment analysis
# Secara default menggunakan model pre-trained distilbert-base-uncased-finetuned-sst-2-english
classifier = pipeline("sentiment-analysis")

# Latihan 4.19: Kumpulan kalimat pengujian (positif & negatif)
test_sentences = [
    "This is a good movie.",  # Kalimat asli Example 4.23
    "I absolutely loved the plot and the acting was phenomenal!",  # Positif
    "The food at this restaurant was delicious and the service was super fast.",  # Positif
    "This was the worst experience I have ever had.",  # Negatif
    "The movie was incredibly boring and a complete waste of time.",  # Negatif
    "It's not bad, but I wouldn't watch it again.",  # Nuansa Campuran / Ambigu
]

print("\n--- HASIL ANALISIS SENTIMEN (EXERCISE 4.19) ---\n")

# Jalankan pengujian untuk setiap kalimat
results = classifier(test_sentences)

for sentence, result in zip(test_sentences, results):
  label = result["label"]
  score = result["score"] * 100  # Konversi ke persentase
  print(f"Kalimat : \"{sentence}\"")
  print(f"Hasil   : {label} (Tingkat Keyakinan: {score:.2f}%)\n")