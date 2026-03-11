import csv
import os

FILE = "accuracy_log.csv"

def log_accuracy(round_no, accuracy):
    write_header = not os.path.exists(FILE)
    with open(FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["round", "accuracy"])
        writer.writerow([round_no, accuracy])
