"""
    Weak-CC vs Formal-CC Palma parity test
    ======================================

    Smallest possible runner: single period (T=1), 6-bus more_meshed case.
    Drives one real lower-level DiffOpt solve to produce a Jacobian + pshed_prev,
    then calls both `palma_ratio_minimization` (weak CC, MIQCP NonConvex) and
    `palma_ratio_minimization_formal_cc` (formal CC, MILP) on identical inputs
    and reports the differences.

    Run:
        julia --project=. script/bilevel_validation/parity_palma_cc.jl
"""

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution
using Ipopt, Gurobi
using DiffOpt
using JuMP
import MathOptInterface as MOI
using LinearAlgebra
using Printf

PMD = PowerModelsDistribution

include("../../src/implementation/load_shed_as_parameter.jl")

# ============================================================
# CONFIGURATION — keep tiny for fast turnaround
# ============================================================
CASE_FILE  = joinpath(@__DIR__, "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss")
LS_PERCENT = 0.8
N_PERIODS  = 1
HOURS      = [12]              # noon
PEAK_COSTS = [1.0]             # uniform single-period

switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)]) * LS_PERCENT

# ============================================================
# STEP 1: NETWORK + JACOBIAN (one real lower-level solve)
# ============================================================
println("\n[1/3] Building network and computing Jacobian…")
eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating = switch_rating)
mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = HOURS, peak_stress = 1.0)

dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs =
    lower_level_soln_mn(mn_data, Float64[], 1)

n_loads = length(weight_ids)
m       = N_PERIODS * n_loads
pd_all  = Float64[sum(refs[nw][:load][lid]["pd"]) for (nw, lid) in pshed_nw_ids]

println("  T=$N_PERIODS, N=$n_loads, m=$m")
println("  pshed_prev = ", round.(pshed_val,   digits=3))
println("  weights    = ", round.(weight_vals, digits=3))
println("  pd         = ", round.(pd_all,      digits=3))

# ============================================================
# STEP 2: WEAK CC vs FORMAL CC on identical inputs
# ============================================================
println("\n[2/3] Solving weak-CC Palma (MIQCP, NonConvex=2)…")
result_weak = palma_ratio_minimization(
    dpshed, copy(pshed_val), copy(weight_vals), pd_all;
    trust_radius    = 0.5,
    w_bounds        = (1.0, 10.0),
    relax_binary    = false,
    critical_ids    = critical_id,
    weight_ids      = weight_ids,
    peak_time_costs = PEAK_COSTS,
    n_loads         = n_loads,
)

println("\n[2/3] Solving formal-CC Palma (MILP)…")
result_formal = palma_ratio_minimization_formal_cc(
    dpshed, copy(pshed_val), copy(weight_vals), pd_all;
    trust_radius    = 0.5,
    w_bounds        = (1.0, 10.0),
    critical_ids    = critical_id,
    weight_ids      = weight_ids,
    peak_time_costs = PEAK_COSTS,
    n_loads         = n_loads,
)

# ============================================================
# STEP 3: PARITY COMPARISON
# ============================================================
println("\n[3/3] Parity report")
println("="^60)

@printf "  status        weak=%s  formal=%s\n"  result_weak.status   result_formal.status
@printf "  solve_time    weak=%.3fs            formal=%.3fs\n"        result_weak.solve_time result_formal.solve_time
@printf("  palma_ratio   weak=%.6f             formal=%.6f             |diff|=%.2e\n",
        result_weak.palma_ratio, result_formal.palma_ratio,
        abs(result_weak.palma_ratio - result_formal.palma_ratio))

dw_diff    = maximum(abs.(result_weak.delta_w   .- result_formal.delta_w))
psh_diff   = maximum(abs.(result_weak.pshed_new .- result_formal.pshed_new))
w_diff     = maximum(abs.(result_weak.weights_new .- result_formal.weights_new))

@printf "  max |Δw|      diff = %.2e\n" dw_diff
@printf "  max |pshed|   diff = %.2e\n" psh_diff
@printf "  max |weights| diff = %.2e\n" w_diff

println("\n  Δw  weak    = ", round.(result_weak.delta_w,    digits=4))
println("  Δw  formal  = ", round.(result_formal.delta_w,  digits=4))
println()
println("  pshed weak  = ", round.(result_weak.pshed_new,   digits=4))
println("  pshed formal= ", round.(result_formal.pshed_new, digits=4))

# Acceptance gates
ratio_tol    = 1e-4
elementwise  = 1e-4
ok_status    = result_weak.status   in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL) &&
               result_formal.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
ok_ratio     = abs(result_weak.palma_ratio - result_formal.palma_ratio) < ratio_tol
ok_pshed     = psh_diff < elementwise

println()
println("="^60)
if ok_status && ok_ratio && ok_pshed
    println("  PARITY PASS")
else
    println("  PARITY FAIL")
    ok_status || println("    - status mismatch (weak=$(result_weak.status), formal=$(result_formal.status))")
    ok_ratio  || println("    - palma ratio diff $(abs(result_weak.palma_ratio - result_formal.palma_ratio)) exceeds $ratio_tol")
    ok_pshed  || println("    - max pshed diff $psh_diff exceeds $elementwise")
end
println("="^60)
