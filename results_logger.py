import csv
import os

FILE = "results.csv"

def init_results():
    if not os.path.exists(FILE):
        with open(FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Loss", "Accuracy", "Precision", "Recall", "F1"])

def log_round(rnd, loss, acc, prec, rec, f1):
    with open(FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([rnd, loss, acc, prec, rec, f1])
