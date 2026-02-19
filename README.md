Cost-Sensitive and Profit-Optimized Credit Risk Modeling
1. Problem Definition

Let:

X = borrower features

Y ∈ {0,1} = default outcome

Y = 1 → Default

Y = 0 → Non-default

We estimate:

P(Y = 1 | X)

using XGBoost.

Decision rule with threshold t:

Reject if predicted probability ≥ t

Approve if predicted probability < t

2. Economic Objective

Let:

π = profit per performing loan

L = loss given default (LGD)

TN(t) = good loans approved

FN(t) = defaulted loans approved

Total portfolio profit under threshold t:

Profit(t) = TN(t) × π − FN(t) × L

Optimal threshold:

t* = argmaxₜ Profit(t)

3. Dataset

LendingClub (2007–2018)

200,000 loans

~20% default rate

Target: Charged Off vs Fully Paid

Leakage detection performed via SHAP to remove post-loan repayment variables.

4. Model

XGBoost

Test ROC-AUC ≈ 0.73

Test PR-AUC ≈ 0.42

Preprocessing included:

High-missing feature pruning

Missing indicator + imputation

Date feature engineering

Ordinal + one-hot encoding

Stratified train-test split

5. Normal Economic Scenario

Assumptions:

Profit per good loan = $1,200

LGD = $7,000

Optimal threshold ≈ 0.16

Metric	Value
Approval Rate	47%
Total Profit	$4.92M
Profit per Approved	~$440
6. Stress Scenario

LGD increased to $9,000.

Re-optimized threshold ≈ 0.11

Metric	Value
Approval Rate	30%
Total Profit	$3.47M
Profit per Approved	~$488
7. Key Insights

ROC optimization ≠ economic optimization

Threshold is a business parameter

Higher LGD → stricter policy

Portfolio risk-return tradeoff is explicit

8. Conclusion

This project reframes credit risk modeling as a decision optimization problem. It integrates probabilistic modeling with portfolio-level profit maximization to derive economically consistent, stress-aware approval policies.