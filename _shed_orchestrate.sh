#!/usr/bin/env bash
# Experiment: optimize SHED-Palma (PALMA_TARGET=shed) instead of served, T=8,
# integer + relaxed. Writes to results/.../palma[_relaxed]_trade_off_mn_shedobj/.
cd "$(dirname "$0")" || exit 1
L="results/t24_orch"; mkdir -p "$L"
echo "START shed_integer $(date)" >> "$L/shed_orch.log"
PALMA_TARGET=shed RELAXED=false julia --project=. script/single_level/palma_trade_off_mn.jl > "$L/shed_integer.log" 2>&1
echo "DONE shed_integer exit=$? $(date)" >> "$L/shed_orch.log"
echo "START shed_relaxed $(date)" >> "$L/shed_orch.log"
PALMA_TARGET=shed RELAXED=true  julia --project=. script/single_level/palma_trade_off_mn.jl > "$L/shed_relaxed.log" 2>&1
echo "DONE shed_relaxed exit=$? $(date)" >> "$L/shed_orch.log"
echo "ALL DONE $(date)" >> "$L/shed_orch.log"
