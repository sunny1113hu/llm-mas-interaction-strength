# Quantitative Summary

This directory contains compact aggregate results underlying quantitative
claims in the manuscript. Bulk per-step and per-trial logs are not included in
the Git repository because they exceed practical repository limits; they are
available from the corresponding author upon reasonable request.

| File | Contents | Generator |
| --- | --- | --- |
| `table_qwen_conditions.md` | Qwen main, no-receptivity, longer-horizon, and grid-size conditions | `scripts/make_paper_quantitative_summary.py` |
| `table_qwen_regimes.md` | Low, best, and high $\mu$ for the main $4\times4$, $T=10$ setting | `scripts/make_paper_quantitative_summary.py` |
| `table_model_best.md` | Best observed metric and $\mu$ for each model | `scripts/make_paper_quantitative_summary.py` |
| `table_pi_ablation.csv` | No-receptivity baseline and receptivity prompting under oracle and leave-one-seed-out selection | `scripts/make_pi_ablation_table.py` |
| `table_pi_ablation.md` | Human-readable version of the Table 1 aggregate | Formatted from `table_pi_ablation.csv` |
| `invalid_output_rates.csv` | Invalid-output counts and rates by model and task | `scripts/aggregate_invalid_outputs.py` |
| `rsp_scale_sensitivity_summary.csv` | Deterministic RSP results at scale factors 0.25, 0.5, and 1.0 | `scripts/make_rsp_scale_sensitivity_summary.py` |

The leave-one-seed-out procedure selects $\mu$ on nine seeds and evaluates it
on the held-out seed, so the evaluation seed does not influence parameter
selection. Across all reported LLM runs, 941 of 1,279,400 agent updates were
invalid (0.074%); every model-task rate is below 0.15%.
