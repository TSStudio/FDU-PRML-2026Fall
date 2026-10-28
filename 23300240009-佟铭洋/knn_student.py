import os
import numpy as np
from typing import List, Tuple
from data_generate import load_prepared_dataset
from viz_knn import plot_k_curve, plot_decision_boundary_multi

# 输出目录
OUT_DIR = "./output"
DATA_DIR = "./input_knn"

# ============ 距离计算 ============
def pairwise_dist(X_test, X_train, metric, mode):
    """
    Compute pairwise distances between X_test (Nte,D) and X_train (Ntr,D).

    Required:
      - L2 distance 'l2' with modes:
          * 'two_loops'  
          * 'no_loops' 
      - 'cosine' distance (distance = 1 - cosine_similarity)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_test.ndim != 2 or X_train.ndim != 2:
        raise ValueError("Inputs must be two-dimensional feature arrays.")
    Nte, D  = X_test.shape
    Ntr, D2 = X_train.shape
    if D != D2:
        raise ValueError("Dim mismatch between test and train.")

    if metric == "l2":
        if mode == "two_loops":
            dists = np.empty((Nte, Ntr), dtype=np.float64)
            for i in range(Nte):
                for j in range(Ntr):
                    diff = X_test[i] - X_train[j]
                    dists[i, j] = np.sqrt(np.dot(diff, diff))
            return dists

        elif mode == "no_loops":
            squared = (np.sum(X_test ** 2, axis=1)[:, None]
                       + np.sum(X_train ** 2, axis=1)[None, :]
                       - 2.0 * (X_test @ X_train.T))
            return np.sqrt(np.maximum(squared, 0.0))

        else:
            raise ValueError("Unknown mode for L2.")

    elif metric == "cosine":
        norm_test = np.linalg.norm(X_test, axis=1, keepdims=True)
        norm_train = np.linalg.norm(X_train, axis=1, keepdims=True)
        # Convention: a zero vector has zero similarity to every vector.
        unit_test = np.divide(X_test, norm_test, out=np.zeros_like(X_test),
                              where=norm_test != 0)
        unit_train = np.divide(X_train, norm_train, out=np.zeros_like(X_train),
                               where=norm_train != 0)
        similarity = np.clip(unit_test @ unit_train.T, -1.0, 1.0)
        return 1.0 - similarity
    else:
        raise ValueError("metric must be 'l2' or 'cosine'.")


# ============ kNN 预测（多数表决） ============
def knn_predict(X_test, X_train, y_train, k, metric, mode):
    """
    kNN prediction.
    Required: majority vote with L2 distance.

    Returns
    -------
    y_pred : (Nte,) int
    """
    dists = pairwise_dist(X_test, X_train, metric=metric, mode=mode)
    y_train = np.asarray(y_train).reshape(-1).astype(int)
    if y_train.size != dists.shape[1]:
        raise ValueError("Training features and labels must have equal lengths.")
    if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or not 1 <= k <= y_train.size):
        raise ValueError("k must be an integer between 1 and the training size.")
    Nte = dists.shape[0]
    y_pred = np.zeros(Nte, dtype=int)

    for i in range(Nte):
        # Equal-distance neighbors retain their original training order.
        idx = np.argsort(dists[i], kind="stable")[:k]
        neighbors = y_train[idx]

        labels, counts = np.unique(neighbors, return_counts=True)
        # np.unique sorts labels; argmax picks the first maximum on ties.
        y_pred[i] = labels[np.argmax(counts)]

    return y_pred


# ============ 验证集选参 ============
def select_k_by_validation(X_train, y_train, X_val, y_val, ks: List[int], metric, mode) -> Tuple[int, List[float]]:
    """
    Grid-search K on validation set.

    Returns
    -------
    best_k : int
    accs   : list of validation accuracies aligned with ks
    """
    ks = list(ks)
    y_val = np.asarray(y_val).reshape(-1)
    if not ks or y_val.size == 0:
        raise ValueError("Candidate ks and validation labels must be nonempty.")
    if len(X_val) != y_val.size:
        raise ValueError("Validation features and labels must have equal lengths.")
    accs = []
    for k in ks:
        y_pred = knn_predict(X_val, X_train, y_train, k, metric, mode)
        accs.append(float(np.mean(y_pred == y_val)))
    best_index = min(range(len(ks)), key=lambda i: (-accs[i], ks[i]))
    return int(ks[best_index]), accs


def run_with_visualization(metric="l2", mode="no_loops"):
    X_train, y_train, X_val, y_val, X_test, y_test = load_prepared_dataset(DATA_DIR)

    # Search every feasible integer, including even k under the tie rule.
    ks = list(range(1, len(X_train) + 1))

    best_k, accs = select_k_by_validation(X_train, y_train, X_val, y_val,
                                          ks, metric=metric, mode=mode)
    print(f"[ModelSelect] metric={metric} best k={best_k} (val acc={max(accs):.4f})")
    plot_k_curve(ks, accs, os.path.join(OUT_DIR, f"knn_k_curve_{metric}.png"))

    X_trv = np.vstack([X_train, X_val]); y_trv = np.hstack([y_train, y_val])
    def predict_fn_for_k(k):
        return lambda Xq: knn_predict(Xq, X_trv, y_trv, k, metric=metric, mode=mode)

    # The merged fitting set has more samples than the selection fitting set.
    # Its all-neighbor endpoint is diagnostic and does not enter selection.
    ks_panel = sorted({k for k in [1, best_k, 51, 151, 225, 250, len(X_trv)]
                       if k <= len(X_trv)})
    for k in ks_panel:
        prediction = predict_fn_for_k(k)(X_test)
        correct = int(np.sum(prediction == y_test))
        print(f"[TestDiagnostic] metric={metric} k={k} "
              f"correct={correct}/{len(y_test)} acc={correct / len(y_test):.4f}")
    plot_decision_boundary_multi(predict_fn_for_k, X_train, y_train, X_test, y_test,
                                 ks=ks_panel,
                                 out_path=os.path.join(OUT_DIR, f"knn_boundary_grid_{metric}.png"),
                                 grid_n=200, batch_size=4096)


if __name__ == "__main__":
    # metric: ["l2", "cosine"];  mode: ["two_loops", "no_loops"]
    for metric in ["l2", "cosine"]:
        run_with_visualization(metric=metric, mode="no_loops")
