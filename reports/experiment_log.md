# Experiment Log

| file | local validation | public score | note |
|---|---:|---:|---|
| `submissions/baseline_lgbm.csv` | AP 0.05833 / WLL 0.61761 | 0.340445203 | First raw LightGBM baseline, no seq. |
| `submissions/baseline_lgbm_prior_calibrated.csv` | n/a | 0.2045772218 | Prior correction hurt; do not use true-CTR calibration for this metric. |
| `submissions/baseline_lgbm_seq.csv` | AP 0.05890 / WLL 0.61836 | pending | Adds basic seq length/first/last/unique features. |
| `submissions/blend_base70_seq30.csv` | n/a | pending | 70% baseline + 30% seq. Conservative blend candidate. |
| `submissions/blend_base50_seq50.csv` | n/a | pending | 50% baseline + 50% seq. More seq influence. |

## Current Takeaways

- Class-balanced raw predictions are better aligned with the competition WLL than true CTR prior correction.
- Basic seq features slightly improved AP locally but slightly worsened WLL, so leaderboard validation is needed.
- Next high-value direction is count/target encoding on stable categorical groups.
