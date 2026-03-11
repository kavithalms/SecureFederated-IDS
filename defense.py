import numpy as np
from sklearn.ensemble import IsolationForest

def detect_malicious(updates):
    flat = [np.concatenate([p.flatten() for p in u]) for u in updates]
    clf = IsolationForest(contamination=0.2)
    preds = clf.fit_predict(flat)
    return [u for u, p in zip(updates, preds) if p == 1]

def trimmed_mean(updates, trim_ratio=0.2):
    stacked = np.stack(updates)
    lower = int(trim_ratio * len(updates))
    upper = len(updates) - lower
    sorted_vals = np.sort(stacked, axis=0)
    return np.mean(sorted_vals[lower:upper], axis=0)
