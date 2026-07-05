# LLM-MAS Interaction Strength

Code for the paper:

> **Emergent Coordination via Tunable Local Interactions in Distributed LLM Agents**
> Taiyo Sato, Keisuke Maeda, Naoki Saito, Takahiro Ogawa, and Miki Haseyama
> (submitted to *Autonomous Agents and Multi-Agent Systems*)

This repository implements a minimal distributed LLM multi-agent system (LLM-MAS)
in which agents on a 2-D grid observe only **local comparison counts** relative to
their four neighbors and update integer states through prompt-conditioned LLM
inference. The population-level interaction strength is controlled by a single
continuous parameter $\mu$ through an agent-wise receptivity value
$P_i = \sigma(\tau_i)$, $\tau_i \sim \mathcal{N}(\mu, \sigma^2)$.
Sweeping $\mu$ reveals a non-monotonic three-regime landscape (stagnation /
coordinated intermediate regime / destabilization) across two tasks:

- **Numerical consensus** — toroidal grid, converge toward a uniform value (metric: MAD)
- **Anchored diffusion** — open grid with two fixed anchor cells, form a smooth gradient (metric: roughness)

A deterministic **receptivity-scaled pressure (RSP) rule** is included as an
analytical reference that reproduces a qualitatively similar regime structure
without language generation.

## Repository layout

```
src/                  simulation core (grid, agents, prompts, LLM backends, metrics)
scripts/              entry points (runs, sweeps, figure generation)
configs/              experiment configurations (single source of parameters)
results/quantitative/ aggregated result tables reported in the paper
```

## Installation

Python >= 3.10. Using [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

### LLM backend (vLLM)

Experiments use locally served open-weight models via an OpenAI-compatible
[vLLM](https://github.com/vllm-project/vllm) endpoint.

```bash
cp .env.example .env                 # edit MODEL_ID / MODEL_SUBDIR if needed
bash scripts/download_model.sh       # download weights into models/
docker compose up --build vllm       # serve the model at http://localhost:8000/v1
```

Models used in the paper: Qwen3-8B (main), Llama-3.1-8B-Instruct,
Ministral-3-8B-Instruct-2512, Ministral-3-14B-Instruct-2512, Phi-4-mini-instruct.

## Running experiments

```bash
# minimal end-to-end check (single trial)
python -m scripts.run_one --config configs/default.yaml

# full mu sweep (29 values, 10 seeds) for the LLM system
python -m scripts.run_sweep --config configs/default.yaml

# deterministic baselines (RSP rule = "mobility_scaled", pressure family)
bash scripts/run_mobility_main_sweeps.sh
bash scripts/run_baseline_main_sweeps.sh

# grid-size robustness runs for the diffusion task
bash scripts/run_diffusion_6x6_8x8.sh
```

Each run writes `outputs/runs/<run_id>/` containing a resolved config snapshot,
a machine metadata manifest, per-step agent logs (JSONL), and per-trial
summaries (JSONL).

Key parameters (all in `configs/*.yaml`): grid size, number of steps `T`,
`mu_list` (29-point sweep; step 0.25 in [-2, 2], 0.5 outside), `sigma` (0.5),
seeds (10 per condition), value range [0, 25], decoding settings
(temperature 0.2, max_tokens 256).

## Reproducing the paper figures

Figure scripts read run logs from `outputs/runs/<run_id>/`. The script
numbering is historical and does not match the paper numbering:

| Paper figure | Script |
| --- | --- |
| Fig. 2 (regime landscape) | `scripts/make_journal_fig1.py` |
| Fig. 3 (snapshots) | `scripts/make_journal_fig2.py` |
| Figs. 4-5 (receptivity ablation) | `scripts/make_journal_fig6_fig7.py` |
| Fig. 6 (LLM vs. RSP rule) | `scripts/make_journal_fig10.py` |
| Fig. 7 (step response) | `scripts/make_journal_fig12.py` |
| Fig. 8 (8B-class models) | `scripts/make_journal_fig4.py` |
| Fig. 9 (model scales) | `scripts/make_journal_fig5.py` |
| Fig. 10 (T=30 robustness) | `scripts/make_journal_fig8.py` |
| Fig. 11 (grid-size robustness) | `scripts/make_journal_fig9.py` |

Each script contains the run IDs used in the paper as defaults; regenerate a
figure with, e.g., `python -m scripts.make_journal_fig1`.

### Data availability

The aggregated tables underlying the quantitative claims in the paper are
included in `results/quantitative/`. The full raw simulation logs (~1.8 GB)
exceed practical repository limits; they are available from the corresponding
author upon reasonable request, and a DOI-archived release is planned upon
acceptance. Because all randomness is seed-controlled, the runs can also be
regenerated from this repository (LLM decoding is deterministic only up to
serving hardware/software).

## License

MIT (see `LICENSE`).

## Citation

Citation information will be added upon publication.
