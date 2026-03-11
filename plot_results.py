import pandas as pd
import matplotlib.pyplot as plt

# Load results table
df = pd.read_csv("results.csv")

rounds = df["Round"]
accuracy = df["Accuracy"]
precision = df["Precision"]
recall = df["Recall"]
f1 = df["F1"]

# ---------------------------
# Accuracy plot
# ---------------------------
plt.figure(figsize=(6,4))
plt.plot(rounds, accuracy, marker='o')
plt.xlabel("Federated Rounds")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Rounds")
plt.grid(True)
plt.show()

# ---------------------------
# Precision plot
# ---------------------------
plt.figure(figsize=(6,4))
plt.plot(rounds, precision, marker='o')
plt.xlabel("Federated Rounds")
plt.ylabel("Precision")
plt.title("Precision vs Rounds")
plt.grid(True)
plt.show()

# ---------------------------
# Recall plot
# ---------------------------
plt.figure(figsize=(6,4))
plt.plot(rounds, recall, marker='o')
plt.xlabel("Federated Rounds")
plt.ylabel("Recall")
plt.title("Recall vs Rounds")
plt.grid(True)
plt.show()

# ---------------------------
# F1-score plot
# ---------------------------
plt.figure(figsize=(6,4))
plt.plot(rounds, f1, marker='o')
plt.xlabel("Federated Rounds")
plt.ylabel("F1-score")
plt.title("F1-score vs Rounds")
plt.grid(True)
plt.show()
