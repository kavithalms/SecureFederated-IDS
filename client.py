import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import IDSModelNew

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Load data for specific FL client
# -------------------------------------------------

def load_client_data(client_id):
    X_train = np.load(f"data/client_{client_id}_X.npy")
    y_train = np.load(f"data/client_{client_id}_y.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    return X_train, y_train, X_test, y_test


# -------------------------------------------------
# Flower weight handling (PyTorch)
# -------------------------------------------------

def get_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, parameters):
    params = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params}
    model.load_state_dict(state_dict, strict=True)


# -------------------------------------------------
# Flower Client
# -------------------------------------------------

class IDSClient(fl.client.NumPyClient):

    def __init__(self, client_id, input_size):
        self.model = IDSModelNew(input_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

        self.X_train, self.y_train, self.X_test, self.y_test = load_client_data(client_id)


    def get_parameters(self, config):
        return get_parameters(self.model)


    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        X = torch.tensor(self.X_train, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)

        for _ in range(1):  # one local epoch per round
            self.optimizer.zero_grad()
            out = self.model(X)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(config), len(X), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        X = torch.tensor(self.X_test, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            preds = self.model(X).cpu().numpy()

        y_pred = (preds > 0.5).astype(int)
        y_true = self.y_test

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return float(1 - accuracy), len(X), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }


# -------------------------------------------------
# Start client
# -------------------------------------------------

if __name__ == "__main__":
    CLIENT_ID = 0   # change 0 → 1 → 2 → 3 → 4 for other clients
    INPUT_SIZE = np.load("data/client_0_X.npy").shape[1]

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=IDSClient(CLIENT_ID, INPUT_SIZE),
    )
