# Measured findings

Selected model: **Gradient boosting**, using validation average precision.
Frozen validation-F1 threshold: **0.27**.

## Held-out customers

Test customers: 1,409.
- AP: 0.6227
- ROC_AUC: 0.8267
- Brier: 0.1432
- Accuracy: 0.7466
- Precision: 0.5150
- Recall: 0.7807
- F1: 0.6206

## Hypotheses

- Gradient boosting minus logistic AP: 0.0003; 95% paired bootstrap interval [-0.0223, 0.0245]. Inconclusive.
- Tuned minus default threshold F1: 0.0698; 95% paired bootstrap interval [0.0340, 0.1084]. Positive difference supported.

## What this supports

The comparison measures retrospective classification on a random
customer holdout. Model selection and thresholds used validation only.
The test table uses 0.50 for all models; the selected operating-point
table also reports the frozen validation threshold.

## Limits

No timestamp or future observation window is available. These results
do not establish future churn accuracy, causal retention effects,
individual certainty, or financial return. Cohort error bars and
bootstrap intervals do not include population or training uncertainty.
F1 is an educational choice; actual outreach requires cost and capacity
data and a separately evaluated policy. Subgroup metrics are descriptive.
