# Judge Robustness Results

Primary judge: `google/gemini-3.1-flash-lite`

## Run-level results

### anthropic/claude-sonnet-4#run1

- C0-C8 accuracy: 0.989
- C0-C8 macro-F1: 0.989
- Operational-tier accuracy: 1.000
- Leakage F1: 0.992
- Leakage FP/FN: 1/0
- Date MAE (days): 0.0

### google/gemini-3.1-flash-lite#run1

- C0-C8 accuracy: 0.756
- C0-C8 macro-F1: 0.716
- Operational-tier accuracy: 0.878
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.0

### google/gemini-3.1-flash-lite#run2

- C0-C8 accuracy: 0.756
- C0-C8 macro-F1: 0.716
- Operational-tier accuracy: 0.878
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.0

### google/gemini-3.1-flash-lite#run3

- C0-C8 accuracy: 0.822
- C0-C8 macro-F1: 0.781
- Operational-tier accuracy: 0.944
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.0

### moonshotai/kimi-k2.5#run1

- C0-C8 accuracy: 0.967
- C0-C8 macro-F1: 0.966
- Operational-tier accuracy: 0.978
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.03333333333333333

## Agreement

```json
{
  "repeatability": {
    "category": 0.925,
    "is_leaked": 1.0,
    "effective_source_date": 0.8333333333333334
  },
  "cross_judge_vs_primary": {
    "anthropic/claude-sonnet-4": {
      "n": 120,
      "category_agreement": 0.6083333333333333,
      "tier_agreement": 0.9083333333333333,
      "leakage_agreement": 0.9916666666666667
    },
    "moonshotai/kimi-k2.5": {
      "n": 120,
      "category_agreement": 0.6333333333333333,
      "tier_agreement": 0.8916666666666667,
      "leakage_agreement": 1.0
    }
  }
}
```
