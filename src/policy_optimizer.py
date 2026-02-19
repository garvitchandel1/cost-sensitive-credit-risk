
import numpy as np
from sklearn.metrics import confusion_matrix

def optimize_profit_threshold(proba, y_true, thresholds, loan_profit, lgd):
    profits = []

    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        profit = tn * loan_profit - fn * lgd
        profits.append(profit)

    max_profit = max(profits)
    optimal_t = thresholds[np.argmax(profits)]

    return optimal_t, max_profit