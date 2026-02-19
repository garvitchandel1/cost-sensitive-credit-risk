# main.py

import numpy as np
from config import *
from src.model import train_xgb
from src.policy_optimizer import optimize_profit_threshold


thresholds = np.linspace(0.01, 0.99, 99)

model = train_xgb(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]

optimal_t, max_profit = optimize_profit_threshold(
    proba,
    y_test,
    thresholds,
    LOAN_PROFIT,
    LGD_NORMAL
)

print("Optimal Threshold:", optimal_t)
print("Max Profit:", max_profit)