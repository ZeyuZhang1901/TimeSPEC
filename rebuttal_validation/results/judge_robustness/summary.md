# Judge Robustness Results

Primary judge: `google/gemini-3.1-flash-lite`

## Run-level results

### google/gemini-3.1-flash-lite#run1

- C0-C8 accuracy: 0.222
- C0-C8 macro-F1: 0.134
- Operational-tier accuracy: 0.433
- Leakage F1: 1.000
- Leakage FP/FN: 0/0
- Date MAE (days): 0.0

## Agreement

```json
{
  "repeatability": {
    "category": 1.0,
    "is_leaked": 1.0,
    "effective_source_date": 1.0
  },
  "cross_judge_vs_primary": {}
}
```
