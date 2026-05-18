"""
    Weak-CC vs Formal-CC Palma parity + timing at T=8
    =================================================

    Multi-period version of parity_palma_cc.jl. Mirrors run_validation_mn.jl's
    T=8 setup (hours [4,6,8,12,15,18,20,22], TOU peak costs, profiled load
    schedules) on case6_more_meshed_good4integer. Drives ONE real lower-level
    DiffOpt solve, then calls weak CC (MIQCP NonConvex) and formal CC (MILP) on
    identical inputs and reports parity + runtime.

    This is the timing-relevant test: weak-CC's spatial branching scales with
    the number of σ_t bilinearities (one per period), so T=8 is where formal
    CC's linearization should start to pay off.

    Run:
        julia --project=. script/bilevel_validation/parity_palma_cc_mn.jl
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

const PMD = PowerModelsDistribution

include("../../src/implementation/load_shed_as_parameter.jl")

# ============================================================
# CONFIGURATION — matches run_validation_mn.jl T=8 setup
# ============================================================
# Switch between cases by changing CASE. case6 is fast (N=9 → quick runs);
# motivation_c is the dissertation case (N=16 → represents the real workload).
CASE = "case6_more_meshed"   # "case6_more_meshed" | "motivation_c"
const CASE_FILE  = CASE == "motivation_c" ?
    joinpath(@__DIR__, "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss") :
    joinpath(@__DIR__, "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss")
const LS_PERCENT = 0.8
# T=24 full diurnal cycle. T=8 default from run_validation_mn.jl was
# [4, 6, 8, 12, 15, 18, 20, 22]; switch back if needed.
SELECTED_HOURS   = collect(0:23)
N_PERIODS        = length(SELECTED_HOURS)
const PEAK_STRESS       = 1.0
const CENTER_AT_NOMINAL = true
 PEAK_TIME_COSTS   = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                          for h in SELECTED_HOURS]

# Need to update for the 13 bus version, current version is for the 6-bus case.
switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)]) * LS_PERCENT

# ============================================================
# STEP 1: NETWORK + JACOBIAN (one real lower-level solve)
# ============================================================
println("\n[1/3] Building network and computing T=$N_PERIODS Jacobian…")
eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating = switch_rating)

mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS,
    center_at_nominal = CENTER_AT_NOMINAL)

dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs =
    lower_level_soln_mn(mn_data, Float64[], 1)

# diff_forward_full_jacobian_mn now returns a ZERO Jacobian if the primal
# terminated at ITERATION_LIMIT — the parity comparison still exercises the
# permutation/McCormick/CC machinery on the incumbent pshed_prev (pshed_new
# is fixed at pshed_prev since J·Δw = 0). A warn line above will flag it.
if iszero(dpshed)
    println("\n  ⚠  Zero Jacobian — primal didn't reach KKT. Parity will only test ",
            "the sort/permutation half (Δw is in the trust-region null space).")
end

n_loads = length(weight_ids)
m       = N_PERIODS * n_loads
pd_all  = Float64[sum(refs[nw][:load][lid]["pd"]) for (nw, lid) in pshed_nw_ids]

println("  T=$N_PERIODS, N=$n_loads, m=$m, λ=$PEAK_TIME_COSTS")
println("  Jacobian size: $(size(dpshed)), max |J| = $(round(maximum(abs.(dpshed)), sigdigits=4))")
println("  Σ pshed_prev = $(round(sum(pshed_val), digits=3))")

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
    peak_time_costs = PEAK_TIME_COSTS,
    n_loads         = n_loads,
)

println("\n[2/3] Solving formal-CC Palma (MILP)…")
result_formal = palma_ratio_minimization_formal_cc(
    dpshed, copy(pshed_val), copy(weight_vals), pd_all;
    trust_radius    = 0.5,
    w_bounds        = (1.0, 10.0),
    critical_ids    = critical_id,
    weight_ids      = weight_ids,
    peak_time_costs = PEAK_TIME_COSTS,
    n_loads         = n_loads,
)

# ============================================================
# STEP 3: PARITY + TIMING REPORT
# ============================================================
println("\n[3/3] Parity + timing report  (T=$N_PERIODS, N=$n_loads)")
println("="^70)

@printf "  status        weak=%s   formal=%s\n" result_weak.status result_formal.status
@printf("  solve_time    weak=%.2fs   formal=%.2fs   speedup=%.1fx\n",
        result_weak.solve_time, result_formal.solve_time,
        result_weak.solve_time / max(result_formal.solve_time, 1e-9))
@printf("  palma_ratio   weak=%.6f   formal=%.6f   |diff|=%.2e\n",
        result_weak.palma_ratio, result_formal.palma_ratio,
        abs(result_weak.palma_ratio - result_formal.palma_ratio))

dw_diff   = maximum(abs.(result_weak.delta_w   .- result_formal.delta_w))
psh_diff  = maximum(abs.(result_weak.pshed_new .- result_formal.pshed_new))
w_diff    = maximum(abs.(result_weak.weights_new .- result_formal.weights_new))

@printf "  max |Δw|      diff = %.2e\n"  dw_diff
@printf "  max |pshed|   diff = %.2e\n"  psh_diff
@printf "  max |weights| diff = %.2e\n"  w_diff

# Per-period palma ratio comparison
top_10_idx, bottom_40_idx = compute_palma_indices(n_loads)
println("\n  Per-period palma ratios:")
println("    t    λ      weak      formal    |diff|")
for t in 1:N_PERIODS
    offset = (t - 1) * n_loads
    pserved_weak_t   = pd_all[offset+1:offset+n_loads] .- result_weak.pshed_new[offset+1:offset+n_loads]
    pserved_formal_t = pd_all[offset+1:offset+n_loads] .- result_formal.pshed_new[offset+1:offset+n_loads]
    pr_weak   = palma_ratio(pserved_weak_t)
    pr_formal = palma_ratio(pserved_formal_t)
    @printf "    %d    %5.2f  %8.4f  %8.4f  %.2e\n"  t PEAK_TIME_COSTS[t] pr_weak pr_formal abs(pr_weak - pr_formal)
end

# Acceptance gates — same as T=1 test
ratio_tol   = 1e-4
elementwise = 1e-4
ok_status   = result_weak.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.TIME_LIMIT, MOI.ITERATION_LIMIT) &&
              result_formal.status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.TIME_LIMIT, MOI.ITERATION_LIMIT)
ok_ratio    = abs(result_weak.palma_ratio - result_formal.palma_ratio) < ratio_tol
ok_pshed    = psh_diff < elementwise

println()
println("="^70)
if ok_status && ok_ratio && ok_pshed
    println("  PARITY PASS")
    if result_formal.solve_time < result_weak.solve_time
        @printf("  Formal CC is %.1fx faster at T=%d\n",
                result_weak.solve_time / max(result_formal.solve_time, 1e-9), N_PERIODS)
    else
        @printf("  Weak CC is %.1fx faster at T=%d (unexpected — investigate σ_max / Big-M)\n",
                result_formal.solve_time / max(result_weak.solve_time, 1e-9), N_PERIODS)
    end
else
    println("  PARITY FAIL")
    ok_status || println("    - status mismatch (weak=$(result_weak.status), formal=$(result_formal.status))")
    ok_ratio  || println("    - palma ratio diff $(abs(result_weak.palma_ratio - result_formal.palma_ratio)) exceeds $ratio_tol")
    ok_pshed  || println("    - max pshed diff $psh_diff exceeds $elementwise")
end
println("="^70)

if result_weak.status == MOI.TIME_LIMIT || result_formal.status == MOI.TIME_LIMIT
    println("\nNote: at least one solve hit TimeLimit (15 min default). Timing comparison reflects time-to-incumbent, not time-to-optimal. Re-run with longer limit or smaller case for a definitive comparison.")
end
