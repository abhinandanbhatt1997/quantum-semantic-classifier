# semantic_encoder.py (FINAL WORKING VERSION)
import pandas as pd
import numpy as np
import os, json

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler  # ✅ Qiskit 1.x+ compatible

# === Config ===
NUM_QUBITS = 4
SHOTS = 1024
DATA_FILE = "data/semantic_dataset.csv"
EXPORT_FILE = "export/semantic_counts.csv"

# === Ensure folders exist ===
os.makedirs("export", exist_ok=True)

# === Load and reduce embeddings ===
print("📥 Loading sentence embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv(DATA_FILE)

embeddings = model.encode(df['text'].tolist())
pca = PCA(n_components=NUM_QUBITS)
reduced = pca.fit_transform(embeddings)

# === Quantum Encoding + Measurement using Sampler ===
sampler = Sampler()

def sentence_to_counts(semantic_vector):
    qc = QuantumCircuit(NUM_QUBITS)
    for i, val in enumerate(semantic_vector):
        angle = val * np.pi
        qc.ry(angle, i)
    qc.measure_all()
    result = sampler.run(qc, shots=SHOTS).result()
    counts = result.quasi_dists[0].binary_probabilities()
    return counts

# === Process sentences ===
print("⚛️ Encoding and measuring circuits...")
results = []

for i, vec in enumerate(reduced):
    counts = sentence_to_counts(vec)
    results.append({
        "text": df.loc[i, "text"],
        "label": df.loc[i, "label"],
        "top_counts": json.dumps(counts)
    })

# === Export ===
print("📤 Saving to CSV...")
pd.DataFrame(results).to_csv(EXPORT_FILE, index=False)
print(f"✅ Done: Results saved to {EXPORT_FILE}")
