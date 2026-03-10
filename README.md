# Cost-Sensitive and Profit-Optimized Credit Risk Modeling

## 1. Overview

This project develops a credit risk decision system that explicitly separates:

- Probability estimation (default prediction)
- Decision policy (approval threshold selection)
- Economic optimization (portfolio profit maximization)

Instead of optimizing traditional ML metrics such as accuracy or ROC-AUC alone, the objective is to derive an **economically optimal approval policy** based on expected portfolio profit under both normal and stressed economic conditions.

---

## 2. Problem Formulation

Let:

- X = borrower feature vector  
- Y ∈ {0,1} = loan outcome  

1 → Default  
0 → Non-default  

We estimate:

p(x) = P(Y = 1 | X = x)

using an **XGBoost classifier**.

A decision policy is defined via threshold **t**:

- Reject loan if p(x) ≥ t  
- Approve loan if p(x) < t  

The central problem becomes:

**What threshold maximizes portfolio-level economic value?**

---

## 3. Dataset

Source: **LendingClub accepted loans dataset (2007–2018)**

### Dataset Characteristics

- 200,000 sampled loans  
- ~20% default rate  

Binary target:

- Fully Paid → 0  
- Charged Off → 1  

Over **150+ original features**, including:

- Credit history metrics (FICO range, delinquencies, utilization)
- Loan attributes (term, interest rate, installment)
- Borrower attributes (income, employment length, home ownership)

### Preprocessing Pipeline

- Removal of post-outcome leakage variables using SHAP
- High-missing feature pruning
- Structured missing-value handling (indicator + imputation)
- Date feature engineering (credit age)
- Ordinal and one-hot encoding
- Stratified train-test split

Note: Raw dataset is not included in the repository due to size constraints.

---

## 4. Model Performance

Model: **XGBoost**

| Metric | Value |
|------|------|
| ROC-AUC | ~0.73 |
| PR-AUC | ~0.42 |

Probability calibration was evaluated via **isotonic regression**.

Brier score improvement was marginal, indicating that base probabilities were already reasonably calibrated.

---

## 5. Economic Framework

Let:

- π = profit per performing loan  
- L = Loss Given Default (LGD)  

- TN(t) = performing loans approved  
- FN(t) = defaulted loans approved  

Total portfolio profit under threshold **t**:

Profit(t) = TN(t) × π − FN(t) × L

The optimal policy is defined as:

t* = threshold that maximizes Profit(t)

This reframes classification as a **profit optimization problem rather than a prediction accuracy problem**.

---

## 6. Normal Economic Scenario

Assumptions:

π = 1200  
L = 7000  

Profit-optimized threshold:

t* ≈ 0.16

### Portfolio Outcomes

| Metric | Value |
|------|------|
| Approval Rate | 47% |
| Total Profit | $4.92M |
| Profit per Approved Loan | ~$440 |

---

## 7. Stress Scenario (Recession Simulation)

To simulate adverse macroeconomic conditions:

Loss Given Default increased to:

L = 9000

Re-optimized threshold:

t*_stress ≈ 0.11

### Portfolio Outcomes

| Metric | Value |
|------|------|
| Approval Rate | 30% |
| Total Profit | $3.47M |
| Profit per Approved Loan | ~$488 |

### Observations

- Higher LGD leads to lower optimal threshold
- Approval rate contracts under stress
- Portfolio becomes more conservative
- Profit per approved loan increases

---

## 8. Theoretical Insight

A loan should be approved when expected profit is positive:

Expected Profit = (1 − p) × π − p × L

Approval condition:

(1 − p) × π − p × L > 0

Solving for p:

p < π / (π + L)

Thus, the economically rational decision boundary depends directly on:

π / (π + L)

Higher default severity shifts the threshold downward, enforcing stricter credit policy.

---

## 9. Key Contributions

- Explicit separation of prediction and decision layers
- Cost-sensitive and profit-based threshold optimization
- SHAP-based leakage detection and feature validation
- Probability calibration assessment
- Macroeconomic stress testing via LGD adjustment
- Sensitivity analysis of decision policy

---

## 10. Conclusion

This project formalizes credit risk modeling as an **economic decision optimization problem**.

By integrating probabilistic machine learning with explicit portfolio-level profit maximization, the system derives **economically consistent and stress-adaptive approval policies**.

The result is a **decision-aware credit scoring framework rather than a pure classification model**.

---

## 👤 Author

**Garvit Chandel**
