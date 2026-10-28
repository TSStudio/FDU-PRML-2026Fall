"""
criterion
"""
import math

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    def entropy(counts):
        total = sum(counts.values())
        return -sum((n / total) * math.log2(n / total)
                    for n in counts.values()) if total else 0.0

    return __impurity_decrease(y, l_y, r_y, entropy)


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    n_left, n_right = l_y.size, r_y.size
    total = n_left + n_right
    if not n_left or not n_right:
        return 0.0
    split_info = -sum((n / total) * math.log2(n / total)
                      for n in (n_left, n_right))
    return info_gain / split_info


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    def gini(counts):
        total = sum(counts.values())
        return 1.0 - sum((n / total) ** 2 for n in counts.values()) if total else 0.0

    return __impurity_decrease(y, l_y, r_y, gini)


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    def error(counts):
        total = sum(counts.values())
        return 1.0 - max(counts.values()) / total if total else 0.0

    return __impurity_decrease(y, l_y, r_y, error)


def __impurity_decrease(y, l_y, r_y, impurity):
    """Parent impurity minus child impurities weighted by sample counts."""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    total = sum(all_labels.values())
    n_left, n_right = sum(left_labels.values()), sum(right_labels.values())
    # An empty child does not constitute a useful binary split.
    if total == 0 or n_left == 0 or n_right == 0:
        return 0.0
    return (impurity(all_labels)
            - n_left / total * impurity(left_labels)
            - n_right / total * impurity(right_labels))
