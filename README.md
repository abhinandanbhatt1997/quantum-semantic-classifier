# 🧠⚛️ Quantum vs Classical Spam Classification

This repository implements a hybrid **Quantum Machine Learning (QML)** pipeline and benchmarks it against classical NLP methods to classify SMS messages as spam or ham. The quantum pipeline encodes semantic embeddings into quantum circuits, extracts bitstring measurements, and trains a classical classifier using quantum output distributions.

---

## 🧪 Project Overview

### 🔹 Classical Baseline
- Uses Bag-of-Words (BoW) + Logistic Regression
- Achieves >98% accuracy on standard SMS spam dataset

### 🔹 Quantum Semantic Pipeline
- Embeds text using `sentence-transformers` (MiniLM)
- Reduces to 4D via PCA for 4-qubit circuit encoding
- Applies `Ry` rotations in Qiskit
- Measures bitstring outcomes over 1024 shots
- Trains a Random Forest Classifier on the bitstring count vectors
- Achieves ~79% accuracy using quantum data

---

## 🗂️ Directory Structure

quantum-computation/
├── classical_structure.py # Classical BoW pipeline
├── semantic_encoder.py # Qiskit-based quantum encoding
├── semantic_classifier.py # Random Forest on quantum output
├── quantum_classifier.py # Extended multi-bitstring QML
├── benchmark.py # Classical vs Quantum benchmark
├── main.py, bow_pipeline.py # Minimal classical-to-quantum workflow
├── quantum_data_export.csv # Quantum bitstring features
├── data/
│ └── spam.csv # SMS spam dataset
├── figures/
│ └── bitstring_histogram_*.png # Quantum measurement plots
├── example.txt # Sample email for testing
├── template.tex # Research paper (Quantum Reports format)
├── requirements.txt
└── README.md

yaml
Copy code

---

## 📊 Results

| Model                      | Accuracy |
|---------------------------|----------|
| Classical (BoW + LR)      | 98.0%    |
| Quantum (Semantic + QML)  | 79.4%    |

See `figures/` for quantum bitstring histograms.

---

## 📦 Dependencies

Install via pip:

```bash
pip install -r requirements.txt
Main libraries:

qiskit

pandas

scikit-learn

sentence-transformers

matplotlib

🚀 Run Example
Classical
bash
Copy code
python3 classical_structure.py
Quantum Pipeline
bash
Copy code
python3 semantic_encoder.py
python3 semantic_classifier.py
📄 Research Paper
The full LaTeX paper is written using the Quantum Reports template and included in template.tex.

🧠 Authors
Abhinandan Bhatt — Project lead, quantum circuit design, NLP integration

Quantum Rick 🧪 — (Fictional) spirit guide for all things quantum

🛡 License
MIT License — free to use, fork, modify with credit.

✨ Acknowledgments
Qiskit

HuggingFace SentenceTransformers

UCI SMS Spam Dataset

yaml
Copy code
