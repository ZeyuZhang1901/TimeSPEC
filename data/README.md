# Data

## task_sensitivity_full_metadata.json (1.2 MB, 337 instances)

The three-task diagnostic probe (Legal, NBA Salary, Stock ranking) used in
Section 4.2 of the paper. Consumed by `scripts/run_task_sensitivity_eval.sh`.

## binary.jsonl (8.7 MB, 2,769 rows)

The binary-question split of the benchmark used throughout the paper
(1,942 treatment / 827 control). The main 2x2 ablation runs on this split,
and the camera-ready experiments sample from it:

- E3 Snapshot-RAG: 350 seeded treatment rows
  (`scripts/cr_sample_treatment.py --n 350 --seed 42`, filter:
  `group == treatment` and non-empty `background_source_urls`)
- E6 cost run: 100 seeded treatment rows (`--n 100 --seed 123`)

Each row carries: `id`, `question`, `background`, `temporal_cutoff`,
`resolution_date`, `answer`, `answer_type`, `group`, `domain`, `source_url`,
`background_source_urls` (the crawlable article URLs used for Wayback
snapshots), and provenance metadata.

## Raw Autocast archive (excluded, ~181 MB)

The rebuttal pilot sets were derived from the Autocast dataset. The raw
archive is not shipped to keep the repository lean. To re-fetch:

```bash
curl -L -o autocast.tar.gz \
  "https://people.eecs.berkeley.edu/~hendrycks/autocast.tar.gz"
tar xzf autocast.tar.gz -C rebuttal_validation/data/autocast
```

The derived subsets that experiments actually consume are already included:

- `rebuttal_validation/data/autocast_rebuttal_20.jsonl` — 20-question pilot set
- `rebuttal_validation/data/autocast_provenance_60.jsonl` — provenance audit sample
- `rebuttal_validation/data/taxonomy_date_challenge.jsonl` — the 120-claim human annotation challenge
