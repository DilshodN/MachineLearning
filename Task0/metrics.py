import numpy as np


def metrics(y_pred: np.ndarray, y_gt: np.array):
    return {'TP': TP(y_pred, y_gt),
            'FP': FP(y_pred, y_gt),
            'FN': FN(y_pred, y_gt),
            'TN': TN(y_pred, y_gt)}

def TP(y_pred: np.array, y_gt: np.array):
    return np.count_nonzero(np.logical_and(y_pred == 1, y_gt == 1))

def FP(y_pred: np.array, y_gt: np.array):
    return np.count_nonzero(np.logical_and(y_pred == 0, y_gt == 1))

def FN(y_pred: np.array, y_gt: np.array):
    return np.count_nonzero(np.logical_and(y_pred == 1, y_gt == 0))

def TN(y_pred: np.array, y_gt: np.array):
    return np.count_nonzero(np.logical_and(y_pred == 0, y_gt == 0))

def precision(y_pred: np.array, y_gt: np.array):
    TruePositive = TP(y_pred, y_gt)
    FalsePositive = FP(y_pred, y_gt)
    return TruePositive / (TruePositive + FalsePositive)

def recall(y_pred: np.array, y_gt: np.array):
    TruePositive = TP(y_pred, y_gt)
    FalseNegative = FN(y_pred, y_gt)
    return TruePositive / (TruePositive + FalseNegative + 1e-5)

def F1(y_pred: np.array, y_gt: np.array):
    PR = precision(y_pred, y_gt)
    REC = recall(y_pred, y_gt)
    return 2 * PR * REC / (PR + REC)

def TPR(y_pred: np.array, y_gt: np.array):
    return recall(y_pred, y_gt)

def FPR(y_pred: np.array, y_gt: np.array):
    FalsePositive = FP(y_pred, y_gt)
    TrueNegative = TN(y_pred, y_gt)
    return FalsePositive / (FalsePositive + TrueNegative + 1e-5)

def getPRCurve(y_prob: np.array, y_gt: np.array):
    thresholds = y_prob
    
    x = np.zeros_like(thresholds)
    y = np.zeros_like(thresholds)
    for i, threshold in enumerate(sorted(thresholds, reverse=True)):
        y_predict = y_prob >= threshold
        x[i] = recall(y_predict, y_gt)
        y[i] = precision(y_predict, y_gt)
    return np.concatenate([[1.],x]), np.concatenate([[0.],y])

def getROCCurve(y_prob: np.array, y_gt: np.array):
    thresholds = y_prob
    
    x = np.zeros_like(thresholds)
    y = np.zeros_like(thresholds)
    for i, threshold in enumerate(sorted(thresholds, reverse=True)):
        y_predict = y_prob >= threshold
        FalsePosRate = FPR(y_predict, y_gt)
        TruePosRate = TPR(y_predict, y_gt)
        x[i] = FalsePosRate
        y[i] = TruePosRate
    return np.concatenate([[1.],x,[0.]]), np.concatenate([[1.],y,[0.]])
        
def integrate(x_vals, y_vals):
    i = 1
    h = x_vals[1] - x_vals[0]
    total = y_vals[0] + y_vals[-1]
    for y in y_vals[1:-1]:
        if i % 2 == 0:
            total += 2 * y
        else:
            total += 4 * y
        i += 1
    return total * (h / 3.0)


def get_roc_curve(y_prob: np.array, y_ground_truth: np.array):
    thresholds = np.concatenate(([0.], y_prob, [1.]))
    x = np.zeros_like(thresholds)
    y = np.zeros_like(thresholds)
    for i, threshold in enumerate(sorted(thresholds, reverse=True)):
        y_predict = y_prob >= threshold
        x[i] = FPR(y_prob, y_ground_truth)
        y[i] = TPR(y_prob, y_ground_truth)
    return x, y


def get_pr_curve(y_prob: np.array, y_ground_truth: np.array):
    thresholds = y_prob
    x = np.zeros_like(thresholds)
    y = np.zeros_like(thresholds)
    for i, threshold in enumerate(sorted(thresholds, reverse=True)):
        y_predict = y_prob >= threshold
        x[i] = recall(y_prob, y_ground_truth)
        y[i] = precision(y_prob, y_ground_truth)
    return np.concatenate(([0.], x)), np.concatenate(([1.], y))