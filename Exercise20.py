import os

# Sembunyikan warning log TensorFlow / Transformers
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from transformers import pipeline

# Inisialisasi pipeline Question Answering
# Secara default menggunakan model distilbert-base-cased-distilled-squad
question_answerer = pipeline("question-answering")

# Paragraf konteks baru
context_text = (
    "Politeknik Elektronika Negeri Surabaya (PENS) was established in 1988. "
    "It is a leading vocational tertiary institution in Indonesia, specializing "
    "in engineering, technology, and applied sciences. The campus is located in "
    "Surabaya, East Java."
)

# Daftar pertanyaan yang bervariasi
questions = [
    "When was PENS established?",
    "Where is the campus located?",
    "What does PENS specialize in?",
]

print("--- KONTEKS TEKS ---")
print(context_text)
print("\n--- HASIL QUESTION ANSWERING (EXERCISE 4.20) ---\n")

# Evaluasi setiap pertanyaan terhadap konteks
for q in questions:
  result = question_answerer(question=q, context=context_text)
  answer = result["answer"]
  score = result["score"] * 100  # Konversi ke persentase
  start = result["start"]
  end = result["end"]

  print(f"Pertanyaan : {q}")
  print(f"Jawaban    : {answer}")
  print(f"Keyakinan  : {score:.2f}% (Posisi Karakter: {start}-{end})\n")