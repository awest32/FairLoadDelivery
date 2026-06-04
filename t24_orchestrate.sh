#!/usr/bin/env bash
# Sequential T=24 runs (no-bd case6): single-level integer, single-level relaxed,
# then bilevel (one run → both integer+relaxed). Untracked scratch script.
cd "$(dirname "$0")" || exit 1
L="results/t24_orch"
mkdir -p "$L"

echo "START sl_integer $(date)" >> "$L/orch.log"
RELAXED=false julia --project=. script/single_level/palma_trade_off_mn.jl > "$L/sl_integer.log" 2>&1
echo "DONE sl_integer exit=$? $(date)" >> "$L/orch.log"

echo "START sl_relaxed $(date)" >> "$L/orch.log"
RELAXED=true  julia --project=. script/single_level/palma_trade_off_mn.jl > "$L/sl_relaxed.log" 2>&1
echo "DONE sl_relaxed exit=$? $(date)" >> "$L/orch.log"

echo "START bilevel $(date)" >> "$L/orch.log"
julia --project=. script/bilevel_validation/run_validation_mn.jl > "$L/bilevel.log" 2>&1
echo "DONE bilevel exit=$? $(date)" >> "$L/orch.log"

echo "ALL DONE $(date)" >> "$L/orch.log"
