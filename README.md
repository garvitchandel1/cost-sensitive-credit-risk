1. Problem Formulation

Let:

X∈Rd
X∈R
d
 denote borrower feature vectors

Y∈{0,1}
Y∈{0,1} denote default outcome

Y=1
Y=1 → Default

Y=0
Y=0 → Non-default

We estimate the conditional probability of default:

p^(x)=P(Y=1∣X=x)
p
^
	​

(x)=P(Y=1∣X=x)

using a gradient-boosted decision tree model (XGBoost).

A decision policy is defined via thresholding:

y^(x;t)={1	if p^(x)≥t(Reject)
0	if p^(x)<t(Approve)
y
^
	​

(x;t)={
1
0
	​

if 
p
^
	​

(x)≥t(Reject)
if 
p
^
	​

(x)<t(Approve)
	​


The core question is not prediction accuracy, but:

What threshold 
t
t maximizes economic value?

2. Dataset

This project uses the publicly available LendingClub dataset (2007–2018).

Dataset Characteristics

200,000 sampled loans

Default rate ≈ 20%

Binary target:

Fully Paid → 0

Charged Off → 1

150+ original features including:

Credit history metrics (FICO range, delinquencies, credit utilization)

Loan attributes (term, interest rate, installment)

Borrower attributes (income, employment length, home ownership)

Preprocessing Pipeline

Removal of post-outcome (leakage) variables identified via SHAP

High-missing feature pruning

Structured missing-value treatment (indicator + imputation)

Date feature engineering (credit age)

Ordinal and one-hot encoding

Stratified train-test split

Raw data is not included in the repository due to size constraints.

3. Economic Objective

Traditional ML optimizes metrics such as ROC-AUC:

max⁡ROC-AUC
maxROC-AUC

However, credit decision-making requires optimizing portfolio-level profit.

Let:

π
π = profit per performing loan

L
L = Loss Given Default (LGD)

TN(t)
TN(t) = number of performing loans approved

FN(t)
FN(t) = number of defaulted loans approved

Total portfolio profit under threshold 
t
t:

Π(t)=TN(t)π−FN(t)L
Π(t)=TN(t)π−FN(t)L

The optimal decision rule is:

t∗=arg⁡max⁡tΠ(t)
t
∗
=arg
t
max
	​

Π(t)

This reframes classification as an economic optimization problem.

4. Model Performance

Model: XGBoost

Test ROC-AUC ≈ 0.73

Test PR-AUC ≈ 0.42

Probability calibration was evaluated via isotonic regression.
Brier score improvement was marginal, indicating reasonably calibrated base probabilities.

5. Normal Economic Scenario

Assumptions:

π=1200
π=1200

L=7000
L=7000

Profit-optimized threshold:

t∗≈0.16
t
∗
≈0.16

Portfolio outcomes:

Metric	Value
Approval Rate	47%
Total Profit	$4.92M
Profit per Approved Loan	~$440
6. Stress Scenario

To simulate recessionary conditions:

L=9000
L=9000

Re-optimizing:

tstress∗≈0.11
t
stress
∗
	​

≈0.11

Portfolio outcomes:

Metric	Value
Approval Rate	30%
Total Profit	$3.47M
Profit per Approved Loan	~$488
7. Theoretical Implication

Under a simplified expected-value framework, approving a loan is rational when:

(1−p)π−pL>0
(1−p)π−pL>0

Solving:

p<ππ+L
p<
π+L
π
	​


Thus, the economically rational threshold depends on:

ππ+L
π+L
π
	​


Higher LGD → lower optimal threshold → stricter policy.

8. Key Insights

Classification-optimal thresholds do not align with profit-optimal thresholds.

The decision boundary is a function of economic parameters, not just model accuracy.

Portfolio policy must adapt to macroeconomic shifts.

Probability estimation and decision policy should be explicitly separated.

Economic assumptions dominate marginal ROC improvements.

9. Conclusion

This project formalizes credit risk modeling as a constrained economic optimization problem. By integrating probabilistic modeling with explicit portfolio-level profit maximization, it derives economically consistent and stress-adaptive approval policies.

The result is a decision-aware credit scoring framework rather than a pure classification model.
