# Compact Qwen Rebuttal Pilot

Targeted validation pilot on 12 public Autocast Treatment questions across three domains.

## Main results

| Method | Perf. | OLR | DCLR | Recon. gap | High/Med faithful | LOO-DCLR | Claims | Coalitions | Eval sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Temporal Hint | 0.738 | 0.219 | 0.210 | 0.165 | 75.0% | 0.250 | 48 | 306 | 64.7 |
| Self-Correction | 0.743 | 0.000 | 0.000 | 0.213 | 58.3% | 0.000 | 45 | 200 | 52.2 |
| Temporal RAG | 0.670 | 0.103 | 0.138 | 0.118 | 83.3% | 0.083 | 64 | 720 | 129.1 |
| TimeSPEC | 0.675 | 0.014 | 0.019 | 0.069 | 83.3% | 0.000 | 63 | 740 | 116.9 |

Generation wall time for all four methods (shared phases/retrieval): 184.9 seconds.

## Taxonomy-label sensitivity

```json
{
  "baseline_means": {
    "b1": 0.2099416012109311,
    "b2": 0.0,
    "b3": 0.1383994126284875,
    "b4": 0.018889698736637515
  },
  "baseline_order": [
    "b2",
    "b4",
    "b3",
    "b1"
  ],
  "cross_tier_label_flip": {
    "0.05": {
      "mean_absolute_dclr_change": 0.0400118652384162,
      "method_order_flip_rate": 0.195,
      "timespec_lower_than_temporal_rag_rate": 0.995,
      "self_correction_lower_than_temporal_hint_rate": 1.0,
      "repeats": 200
    },
    "0.1": {
      "mean_absolute_dclr_change": 0.07796042871710934,
      "method_order_flip_rate": 0.33,
      "timespec_lower_than_temporal_rag_rate": 0.995,
      "self_correction_lower_than_temporal_hint_rate": 0.995,
      "repeats": 200
    },
    "0.2": {
      "mean_absolute_dclr_change": 0.1657592648202355,
      "method_order_flip_rate": 0.645,
      "timespec_lower_than_temporal_rag_rate": 0.865,
      "self_correction_lower_than_temporal_hint_rate": 0.98,
      "repeats": 200
    }
  }
}
```

## DCLR bounds for flagged/low-confidence claims

```json
{
  "b1": {
    "unresolved_claims": 2,
    "lower_dclr": 0.19035826787759777,
    "upper_dclr": 0.22209182212403863
  },
  "b2": {
    "unresolved_claims": 1,
    "lower_dclr": 0.0,
    "upper_dclr": 0.01618730518560499
  },
  "b3": {
    "unresolved_claims": 3,
    "lower_dclr": 0.1383994126284875,
    "upper_dclr": 0.18419164859474724
  },
  "b4": {
    "unresolved_claims": 3,
    "lower_dclr": 0.018889698736637515,
    "upper_dclr": 0.06621112730806608
  }
}
```

## Scope

This is a small validation pilot, not a replacement for the submitted 2,769-instance experiment.