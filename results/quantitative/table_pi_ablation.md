# Receptivity Ablation

Lower values are better for both metrics. Positive reductions indicate an
improvement over the no-receptivity baseline. The oracle result selects $\mu$
on the evaluation seeds and therefore characterizes the swept design space.
The held-out result uses leave-one-seed-out selection, so no evaluation seed
influences the choice of $\mu$.

| Task (metric) | Model | w/o $P_i$ | Oracle w/ $P_i$ ($\mu$) | Reduction | Held-out w/ $P_i$ | Reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Consensus (MAD) | Qwen3-8B | 2.03 | 1.18 (-0.50) | +41.6% | 1.53 | +24.4% |
| Consensus (MAD) | Llama-3.1-8B | 4.03 | 2.35 (-0.50) | +41.6% | 2.35 | +41.6% |
| Consensus (MAD) | Ministral-3-8B | 2.46 | 2.10 (0.50) | +14.5% | 2.49 | -1.2% |
| Consensus (MAD) | Ministral-3-14B | 1.92 | 1.98 (-0.50) | -3.3% | 2.13 | -11.0% |
| Consensus (MAD) | Phi-4-mini | 4.71 | 3.90 (4.50) | +17.2% | 5.65 | -20.0% |
| Anchored diffusion (roughness) | Qwen3-8B | 41.6 | 29.6 (-0.25) | +29.0% | 29.6 | +29.0% |
| Anchored diffusion (roughness) | Llama-3.1-8B | 35.5 | 45.9 (-1.25) | -29.2% | 51.2 | -44.3% |
| Anchored diffusion (roughness) | Ministral-3-8B | 167 | 41.2 (0.50) | +75.3% | 41.2 | +75.3% |
| Anchored diffusion (roughness) | Ministral-3-14B | 26.6 | 30.9 (0.25) | -16.2% | 33.2 | -25.1% |
| Anchored diffusion (roughness) | Phi-4-mini | 83.7 | 78.8 (-0.50) | +5.9% | 78.8 | +5.9% |
