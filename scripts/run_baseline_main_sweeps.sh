#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PREFIX="${1:-$(date +%Y%m%d_%H%M%S)}"

echo "Running consensus PressureScaled..."
python -m scripts.run_sweep \
  --config configs/consensus_pressure_scaled.yaml \
  --run-id "${PREFIX}_consensus_pressure_scaled"

echo "Running consensus PressureRaw..."
python -m scripts.run_sweep \
  --config configs/consensus_pressure_raw.yaml \
  --run-id "${PREFIX}_consensus_pressure_raw"

echo "Running diffusion PressureScaled..."
python -m scripts.run_sweep \
  --config configs/diffusion_pressure_scaled.yaml \
  --run-id "${PREFIX}_diffusion_pressure_scaled"

echo "Running diffusion PressureRaw..."
python -m scripts.run_sweep \
  --config configs/diffusion_pressure_raw.yaml \
  --run-id "${PREFIX}_diffusion_pressure_raw"

echo "Completed baseline sweeps."
echo "Run IDs:"
echo "  ${PREFIX}_consensus_pressure_scaled"
echo "  ${PREFIX}_consensus_pressure_raw"
echo "  ${PREFIX}_diffusion_pressure_scaled"
echo "  ${PREFIX}_diffusion_pressure_raw"
