"""
    Frank-Wolfe upper-level smoke test (case6, T=5).

    End-to-end check that the inline FW driver runs on the real three-phase MLD:
      - builds the multinetwork implicit-diff model,
      - runs a few FW iterations (primal + adjoint per step),
      - prints the Palma trajectory, solve counts, and final weights/gap.

    This is a SMOKE test (does it run + descend?), not a correctness proof — the
    gradient pieces are proved separately (test_palma_grad.jl, spike_reverse_mode.jl).
    Keep MAX_ITERS small; each iter is 2 NLP solves on the LinDist3Flow model.

    Usage:
        julia --project=. script/bilevel_validation/test_fw_smoke.jl
"""

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi
using HSL_jll
using DiffOpt
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra
using Printf

const PMD = PowerModelsDistribution

include("../../src/implementation/load_shed_as_parameter.jl")  # compute_palma_indices, palma_ratio
include("../../src/implementation/frank_wolfe_palma.jl")
include("../../src/implementation/slp_cc_palma.jl")            # slp_cc_palma (reuses FW helpers)

# --- config mirrors run_validation_mn.jl Step 1 -------------------------------
CASE       = "case6_unbalanced_switch_more_meshed_bd_good4integer"
CASE_FILE  = joinpath(@__DIR__, "../../data/pmd_opendss/$CASE.dss")
LS_PERCENT = 0.8
SELECTED_HOURS    = [4, 12, 15, 18, 22]
N_PERIODS         = length(SELECTED_HOURS)
PEAK_STRESS       = 1.0
CENTER_AT_NOMINAL = true
PERIOD_HOURS      = SELECTED_HOURS
PEAK_TIME_COSTS   = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                     for h in PERIOD_HOURS]
switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT

MAX_ITERS = 8   # smoke: keep small

println("="^70)
println("Frank-Wolfe smoke test — $CASE, T=$N_PERIODS, max_iters=$MAX_ITERS")
println("λ = $PEAK_TIME_COSTS")
println("="^70)

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating = switch_rating)
mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS, center_at_nominal = CENTER_AT_NOMINAL)

res = frank_wolfe_palma(mn_data;
    critical_ids   = critical_id,
    peak_time_costs = PEAK_TIME_COSTS,
    w_bounds       = (1.0, 10.0),
    max_iters      = MAX_ITERS,
    tol            = 1e-4,
    verbose        = true)

println("\n" * "="^70)
println("RESULT")
@printf("  converged          : %s (%d FW iters)\n", res.converged, res.fw_iters)
@printf("  final FW gap       : %.4e\n", res.fw_gap)
@printf("  palma_value (obj)  : %.6f\n", res.palma_value)
@printf("  palma_ratio (rep)  : %.6f\n", res.palma_ratio_reported)
@printf("  primal solves      : %d\n", res.n_primal)
@printf("  adjoint solves     : %d\n", res.n_adjoint)
println("  objective trajectory (min_bot = smallest per-period bot-40% served):")
for h in res.history
    @printf("    iter %3d  obj=%.6f  gap=%.4e  min_bot=%.3e  ‖g‖=%.3e\n",
            h.iter, h.obj, h.gap, h.min_bot, h.gnorm)
end
# Descent sanity: last obj should not exceed first by more than rounding.
if length(res.history) ≥ 2
    descended = res.history[end].obj ≤ res.history[1].obj + 1e-6
    println("\n  monotone-ish descent: $(descended ? "OK" : "WARN (obj rose)")")
end
println("="^70)

# =============================================================================
# Sequential Charnes–Cooper LP upper level (variant b) — reuses mn_data above.
# Same objective as FW (Σ_t λ_t·top_t/bot_t) but exact fraction via CC + fixed
# sort, fed by 4 reverse adjoints/iter. Compare its trajectory to FW's.
# =============================================================================
println("\n" * "="^70)
println("SLP-CC (variant b) — same case/mn_data, max_iters=12")
println("="^70)
slp = slp_cc_palma(mn_data;
    lp_optimizer    = Gurobi.Optimizer,
    critical_ids    = critical_id,
    peak_time_costs = PEAK_TIME_COSTS,
    w_bounds        = (1.0, 10.0),
    trust_radius    = 0.5,
    max_iters       = 12,
    tol             = 1e-4,
    verbose         = true)

println("\n" * "="^70)
println("SLP-CC RESULT")
@printf("  converged          : %s (%d iters)\n", slp.converged, slp.slp_iters)
@printf("  palma_value (obj)  : %.6f\n", slp.palma_value)
@printf("  equality floor     : %.6f  (ceil(0.1N)/floor(0.4N), N=%d)\n",
        slp.equality_floor, length(slp.weight_ids))
@printf("  primal / adjoint   : %d / %d  (4 adjoints per iter, T-independent)\n",
        slp.n_primal, slp.n_adjoint)
println("  per-period Palma vs floor:")
for (t, r) in enumerate(slp.period_ratios)
    @printf("    period %d (λ=%5.2f):  Palma=%.4f  (floor %.4f, gap %+.4f)\n",
            t, PEAK_TIME_COSTS[t], r, slp.equality_floor, r - slp.equality_floor)
end
println("  trajectory:")
for h in slp.history
    @printf("    iter %3d  obj=%.6f  step=%.3e  min_bot=%.3e\n",
            h.iter, h.obj, h.step, h.min_bot)
end
if length(slp.history) ≥ 2
    descended = slp.history[end].obj ≤ slp.history[1].obj + 1e-6
    println("\n  monotone descent: $(descended ? "OK" : "WARN (obj rose)")")
end
println("="^70)
