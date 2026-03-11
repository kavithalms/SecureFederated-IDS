import flwr as fl
import numpy as np
from defense import detect_malicious, trimmed_mean
from results_logger import init_results, log_round

class SecureStrategy(fl.server.strategy.FedAvg):

    # ---------------------------------------------------
    # Secure aggregation of model updates
    # ---------------------------------------------------
    def aggregate_fit(self, rnd, results, failures):

        if not results:
            return None, {}

        updates = []

        for _, fit_res in results:
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            flat = np.concatenate([w.flatten() for w in weights])
            updates.append(flat)

        # Detect poisoned updates
        safe_updates = detect_malicious(updates)

        # Robust aggregation
        agg_update = trimmed_mean(safe_updates)

        # Rebuild weight shapes
        ref_weights = fl.common.parameters_to_ndarrays(results[0][1].parameters)
        new_weights = []

        idx = 0
        for w in ref_weights:
            size = w.size
            new_weights.append(agg_update[idx:idx + size].reshape(w.shape))
            idx += size

        return fl.common.ndarrays_to_parameters(new_weights), {}


    # ---------------------------------------------------
    # Correct evaluation aggregation (THIS WAS BROKEN)
    # ---------------------------------------------------
    def aggregate_evaluate(self, rnd, results, failures):

        if not results:
            print(f"Round {rnd} | No evaluation results")
            return None, {}

        losses = []
        accs, precs, recs, f1s = [], [], [], []

        for _, eval_res in results:
            losses.append(eval_res.loss)

            metrics = eval_res.metrics
            accs.append(metrics.get("accuracy", 0))
            precs.append(metrics.get("precision", 0))
            recs.append(metrics.get("recall", 0))
            f1s.append(metrics.get("f1", 0))

        # Average safely
        loss = sum(losses) / len(losses)

        avg_acc  = sum(accs) / len(accs)
        avg_prec = sum(precs) / len(precs)
        avg_rec  = sum(recs) / len(recs)
        avg_f1   = sum(f1s) / len(f1s)

        print(
            f"Round {rnd} | "
            f"Loss: {loss:.4f} | "
            f"Acc: {avg_acc:.4f} | "
            f"Prec: {avg_prec:.4f} | "
            f"Rec: {avg_rec:.4f} | "
            f"F1: {avg_f1:.4f}"
        )
        log_round(rnd, loss, avg_acc, avg_prec, avg_rec, avg_f1)
        return loss, {
            "accuracy": avg_acc,
            "precision": avg_prec,
            "recall": avg_rec,
            "f1": avg_f1,
        }
    


# ---------------------------------------------------
# Start secure FL server
# ---------------------------------------------------

init_results() 
strategy = SecureStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
)

fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
