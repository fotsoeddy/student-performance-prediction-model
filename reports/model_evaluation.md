# Model Evaluation Report

Generated on: 2026-04-28 12:12:27.668411

## Performance Metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.8835 |
| Precision | 0.9506 |
| Recall (Pass) | 0.8717 |
| Recall (Fail/Risk) | 0.9077 |
| F1-Score | 0.9094 |
| ROC-AUC | 0.9554 |

## Why Fail Recall Matters
In an Early Warning System, **Recall for the Fail class** is more important than overall accuracy. It measures our ability to correctly identify students who are actually at risk. Higher recall means fewer students slip through the cracks without interventions.

## Confusion Matrix
```
[[118  12]
 [ 34 231]]
```
