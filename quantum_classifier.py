# quantum_classifier.py (Multi-bitstring Quantum Classifier)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import ast
from collections import Counter
from sklearn.feature_extraction import DictVectorizer

# === Load quantum export data ===
print("📥 Loading quantum export data...")
df = pd.read_csv("quantum_data_export.csv")

# === Parse bitstring counts safely ===
def safe_parse(d):
    try:
        return ast.literal_eval(d)
    except:
        return {}

df['top_counts'] = df['top_counts'].apply(safe_parse)

# === Drop rows with missing or empty bitstring data ===
df = df[df['top_counts'].apply(lambda x: bool(x))]

print(f"🔍 Data shape: {df.shape}")

# === Expand to multiple features ===
bitstring_features = df['top_counts'].tolist()
dv = DictVectorizer(sparse=False)
X = dv.fit_transform(bitstring_features)
y = df['label'].values

print(f"🧠 Unique bitstrings: {len(dv.feature_names_)}")

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === Results ===
print("\n📊 Quantum Bitstring Classifier Results:")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# === Visualization: Save Sample Bitstring Histograms ===
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import os

os.makedirs("figures", exist_ok=True)

for i, row in df.head(3).iterrows():
    counts = row['top_counts']
    fig = plot_histogram(counts, title=f"Bitstring Histogram #{i+1}")
    fig.savefig(f"figures/bitstring_histogram_{i+1}.png")
    print(f"📊 Saved bitstring_histogram_{i+1}.png to figures/")
