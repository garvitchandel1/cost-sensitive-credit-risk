Cost-Sensitive and Profit-Optimized Credit Risk Modeling
1. Overview

This project develops a credit risk decision system that explicitly separates:

Probability estimation (default prediction)

Decision policy (approval threshold selection)

Economic optimization (portfolio profit maximization)

Instead of optimizing traditional ML metrics such as accuracy or ROC-AUC alone, the objective is to derive an economically optimal approval policy based on expected portfolio profit under both normal and stressed economic conditions.

2. Problem Formulation

Let:

X = borrower feature vector

Y ∈ {0,1} = loan outcome

1 → Default

0 → Non-default

We estimate:

p(x) = P(Y = 1 | X = x)

using an XGBoost classifier.

A decision policy is defined via threshold t:

Reject loan if p(x) ≥ t

Approve loan if p(x) < t

The central problem is:

What threshold maximizes portfolio-level economic value?

3. Dataset

Source: LendingClub accepted loans dataset (2007–2018)

Dataset Characteristics

200,000 sampled loans

~20% default rate

Binary target:

Fully Paid → 0

Charged Off → 1

150+ original features including:

Credit history metrics (FICO range, delinquencies, utilization)

Loan attributes (term, interest rate, installment)

Borrower attributes (income, employment length, home ownership)

Preprocessing Pipeline

Removal of post-outcome (leakage) variables identified via SHAP

High-missing feature pruning

Structured missing-value handling (indicator + imputation)

Date feature engineering (credit age)

Ordinal and one-hot encoding

Stratified train-test split

Raw dataset is not included in the repository due to size constraints.

4. Model Performance

Model: XGBoost

Test ROC-AUC ≈ 0.73

Test PR-AUC ≈ 0.42

Probability calibration was evaluated via isotonic regression.
Brier score improvement was marginal, indicating reasonably calibrated base probabilities.

5. Economic Framework

Let:

π = profit per performing loan

L = Loss Given Default (LGD)

TN(t) = number of performing loans approved

FN(t) = number of defaulted loans approved

Total portfolio profit under threshold t:

Profit(t) = TN(t) × π − FN(t) × L

The optimal policy is:

t* = argmax_t Profit(t)

This reframes classification as a profit optimization problem rather than a prediction accuracy problem.

6. Normal Economic Scenario

Assumptions:

π = 1200

L = 7000

Profit-optimized threshold:

t* ≈ 0.16

Portfolio outcomes:

Metric	Value
Approval Rate	47%
Total Profit	$4.92M
Profit per Approved Loan	~$440
7. Stress Scenario (Recession Simulation)

To simulate adverse macroeconomic conditions:

L = 9000

Re-optimizing threshold:

t*_stress ≈ 0.11

Portfolio outcomes:

Metric	Value
Approval Rate	30%
Total Profit	$3.47M
Profit per Approved Loan	~$488

Observation:

Higher LGD → lower optimal threshold

Approval rate contracts under stress

Portfolio becomes more conservative

Profit per approved loan increases

8. Theoretical Insight

A loan should be approved when expected profit is positive:

(1 − p) × π − p × L > 0

Solving for p:

p < π / (π + L)

Thus, the economically rational decision boundary depends directly on the ratio:

π / (π + L)

Higher default severity (L) shifts the threshold downward, enforcing stricter credit policy.

9. Key Contributions

Explicit separation of prediction and decision layers

Cost-sensitive and profit-based threshold optimization

SHAP-based leakage detection and feature validation

Probability calibration assessment

Macroeconomic stress testing via LGD adjustment

Sensitivity analysis of decision policy

10. Conclusion

This project formalizes credit risk modeling as an economic decision optimization problem. By integrating probabilistic machine learning with explicit portfolio-level profit maximization, it derives economically consistent, stress-adaptive approval policies.

The result is a decision-aware credit scoring framework rather than a pure classification model.

## 👤 Author

**Garvit Chandel**  