# Judge Robustness Results

Primary judge: `google/gemini-3.1-flash-lite`

## Run-level results

### anthropic/claude-sonnet-4#run1

- C0-C8 accuracy: 0.978
- C0-C8 macro-F1: 0.978
- Operational-tier accuracy: 1.000
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.0

### google/gemini-3.1-flash-lite#run1

- C0-C8 accuracy: 1.000
- C0-C8 macro-F1: 1.000
- Operational-tier accuracy: 1.000
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.0

### google/gemini-3.1-flash-lite#run2

- C0-C8 accuracy: 1.000
- C0-C8 macro-F1: 1.000
- Operational-tier accuracy: 1.000
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.0

### google/gemini-3.1-flash-lite#run3

- C0-C8 accuracy: 0.378
- C0-C8 macro-F1: 0.364
- Operational-tier accuracy: 0.856
- Leakage F1: 0.908
- Leakage FP/FN: 13/0
- Date MAE (days): 0.0

### moonshotai/kimi-k2.5#run1

- C0-C8 accuracy: 0.989
- C0-C8 macro-F1: 0.989
- Operational-tier accuracy: 1.000
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.03333333333333333

## Agreement

```json
{
  "repeatability": {
    "category": 0.5,
    "is_leaked": 0.8916666666666667,
    "effective_source_date": 1.0
  },
  "cross_judge_vs_primary": {
    "anthropic/claude-sonnet-4": {
      "n": 120,
      "category_agreement": 0.7833333333333333,
      "tier_agreement": 1.0,
      "leakage_agreement": 1.0
    },
    "moonshotai/kimi-k2.5": {
      "n": 120,
      "category_agreement": 0.825,
      "tier_agreement": 1.0,
      "leakage_agreement": 1.0
    }
  }
}
```
