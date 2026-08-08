## Model Comparison Summary

Test-set performance across candidate models (ranked by ROC-AUC):

| rank | model | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | gradient_boosting | 0.9947 | 0.9931 | 0.9931 | 0.9931 | 1.0 |
| 2 | logistic_regression | 0.9907 | 1.0 | 0.9757 | 0.9877 | 0.9999 |
| 3 | random_forest | 0.9973 | 0.9965 | 0.9965 | 0.9965 | 0.9997 |
| 4 | decision_tree | 0.9947 | 0.9965 | 0.9896 | 0.993 | 0.9963 |

### Best Model

- **Model:** `gradient_boosting`
- **Selected by:** `roc_auc` = `1.0`

| Metric | Score |
|--------|-------|
| accuracy | 0.9947 |
| precision | 0.9931 |
| recall | 0.9931 |
| f1 | 0.9931 |
| roc_auc | 1.0 |
