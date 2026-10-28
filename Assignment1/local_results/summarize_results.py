"""Collect reproducible experiment measurements without choosing on test data."""
import csv
import json
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "decision_tree"), str(ROOT / "k_nearest_neighbors")]
from decision_tree import DecisionTreeClassifier
from data_generate import load_prepared_dataset
from knn_student import knn_predict, select_k_by_validation

result = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
          "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
          "scikit_learn": sklearn.__version__}, "trees": [], "knn": {}}
train = pd.read_csv(ROOT / "decision_tree/dataset/wine_train.csv")
test = pd.read_csv(ROOT / "decision_tree/dataset/wine_test.csv")
features = list(train.columns[:-1])
X, y = train[features].to_numpy(), train.label.to_numpy()
Xt, yt = test[features].to_numpy(), test.label.to_numpy()
for criterion in ("info_gain", "info_gain_ratio", "gini", "error_rate"):
    tree = DecisionTreeClassifier(criterion=criterion, random_state=0)
    tree.fit(X, y)
    result["trees"].append({
        "criterion": criterion, "depth": tree.tree_depth, "leaves": tree.tree_leaf_num,
        "train_correct": int(np.sum(tree.predict(X) == y)), "train_size": len(y),
        "test_correct": int(np.sum(tree.predict(Xt) == yt)), "test_size": len(yt),
        "root_feature": features[tree.root.feat_idx], "root_threshold": float(tree.root.threshold),
        "importances": dict(zip(features, map(float, tree.feature_importances_))),
    })

Xtr, ytr, Xv, yv, Xte, yte = load_prepared_dataset(str(ROOT / "k_nearest_neighbors/input_knn"))
result["knn_data"] = {"dimension": Xtr.shape[1], "splits": {}}
for name, xx, yy in (("train", Xtr, ytr), ("val", Xv, yv), ("test", Xte, yte)):
    labels, counts = np.unique(yy, return_counts=True)
    result["knn_data"]["splits"][name] = {
        "size": len(xx), "classes": dict(zip(map(str, labels), map(int, counts)))}
Xtv, ytv = np.vstack([Xtr, Xv]), np.hstack([ytr, yv])
ks = list(range(1, len(Xtr) + 1))
for metric in ("l2", "cosine"):
    best, accs = select_k_by_validation(Xtr, ytr, Xv, yv, ks, metric, "no_loops")
    diagnostic_ks = list(range(1, len(Xtv) + 1))
    diagnostic_tests = [int(np.sum(knn_predict(Xte, Xtv, ytv, k, metric, "no_loops") == yte))
                        for k in diagnostic_ks]
    tests = diagnostic_tests[:len(ks)]
    result["knn"][metric] = {"ks": ks, "val_accs": accs, "best_k": best,
        "test_correct_by_k": tests, "test_size": len(yte),
        "test_correct": tests[ks.index(best)],
        "validation_tied_best_ks": [k for k, acc in zip(ks, accs) if acc == max(accs)],
        "diagnostic_ks": diagnostic_ks, "diagnostic_test_correct": diagnostic_tests,
        "boundary_ks": sorted({1, best, 51, 151, 225, 250, len(Xtv)}),
        "selection_fit_size": len(Xtr), "final_fit_size": len(Xtv)}

out = ROOT / "local_results/summary.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
with (ROOT / "local_results/logs/knn_full_sweep.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["k", "l2_val_accuracy_train250", "cosine_val_accuracy_train250",
                     "l2_test_accuracy_trainval375", "cosine_test_accuracy_trainval375"])
    for k in range(1, len(Xtv) + 1):
        val_scores = [result["knn"][metric]["val_accs"][k - 1] if k <= len(Xtr) else ""
                      for metric in ("l2", "cosine")]
        test_scores = [result["knn"][metric]["diagnostic_test_correct"][k - 1] / len(yte)
                       for metric in ("l2", "cosine")]
        writer.writerow([k, *val_scores, *test_scores])
print(json.dumps(result, indent=2, ensure_ascii=False))
