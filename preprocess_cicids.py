import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print("[INFO] Loading dataset...")
df = pd.read_csv("data/cicids2017_cleaned.csv")

# -------------------------
# Remove leakage columns
# -------------------------
leak_cols = [
    "Flow ID", "Timestamp",
    "Src IP", "Dst IP",
    "Source IP", "Destination IP"
]

df = df.drop(columns=[c for c in leak_cols if c in df.columns])

df.columns = df.columns.str.strip()

# -------------------------
# Encode label safely
# -------------------------
label_col = df.columns[-1]

# Clean text
df[label_col] = df[label_col].astype(str).str.strip().str.upper()

print("Before encoding:")
print(df[label_col].value_counts())

# NORMAL = 0 , all attacks = 1
df[label_col] = df[label_col].apply(
    lambda x: 0 if x == "NORMAL TRAFFIC" else 1
)

print("\nAfter encoding:")
print(df[label_col].value_counts())


# -------------------------
# Split BEFORE scaling
# -------------------------
X = df.drop(columns=[label_col]).values
y = df[label_col].values

print("Class distribution:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------
# Scale properly
# -------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------
# Create FL clients (non-IID realistic)
# -------------------------
NUM_CLIENTS = 5
client_splits = np.array_split(np.arange(len(X_train)), NUM_CLIENTS)

os.makedirs("data", exist_ok=True)

for i, idx in enumerate(client_splits):
    np.save(f"data/client_{i}_X.npy", X_train[idx])
    np.save(f"data/client_{i}_y.npy", y_train[idx])

np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print("[INFO] Preprocessing finished correctly")
