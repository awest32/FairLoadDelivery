"""
    SLP-CC on motivation_c at the do-nothing-trap scale.

    The formal-CC MILP bilevel stalled at the trivial do-nothing incumbent on
    motivation_c (T=8, N≈16; palma_ratio≈3.75 = palma at uniform weights, Δw=0);
    weak-CC TimeLimited at 2.52 (see project_palma_frank_wolfe_followup). This
    runs the reverse-mode SLP-CC upper level on the same case to test the two
    goals directly:
      (1) DIFFOPT CALLS — SLP-CC uses 4 reverse adjoints/iter (T-independent) vs
          the MILP path's T·N forward Jacobian columns (128 at T=8,N=16).
      (2) BETTER SOLUTIONS — does it DESCEND from the do-nothing start (iter-0
          obj = uniform-weight Palma ≈ the MILP's stuck incumbent)?

    T defaults to 8 (trap scale). Override with e.g. SLP_HOURS="4,18,8".

    Usage:
        julia --project=. script/bilevel_validation/test_slp_cc_motivation_c.jl
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

include("../../src/implementation/load_shed_as_parameter.jl")
include("../../src/implementation/frank_wolfe_palma.jl")
include("../../src/implementation/slp_cc_palma.jl")

# --- motivation_c config (mirrors run_validation_mn_motivation_c.jl) ----------
CASE       = get(ENV, "SLP_CASE", "motivation_c_good4integer")  # or "motivation_c"
CASE_FILE  = joinpath(@__DIR__, "../../data/ieee_13_aw_edit/$CASE.dss")
LS_PERCENT = 0.8
SELECTED_HOURS = haskey(ENV, "SLP_HOURS") ?
    parse.(Int, split(ENV["SLP_HOURS"], ",")) : [2, 4, 8, 12, 15, 18, 20, 22]  # T=8, peak @18
N_PERIODS         = length(SELECTED_HOURS)
PEAK_STRESS       = 1.0
CENTER_AT_NOMINAL = true
PERIOD_HOURS      = SELECTED_HOURS
PEAK_TIME_COSTS   = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                     for h in PERIOD_HOURS]
switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT

MAX_ITERS = 20

println("="^70)
println("SLP-CC on $CASE — T=$N_PERIODS (do-nothing-trap scale), max_iters=$MAX_ITERS")
println("hours=$PERIOD_HOURS  λ=$PEAK_TIME_COSTS")
println("="^70)

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating = switch_rating)
mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS, center_at_nominal = CENTER_AT_NOMINAL)

res = slp_cc_palma(mn_data;
    lp_optimizer    = Gurobi.Optimizer,
    critical_ids    = critical_id,
    peak_time_costs = PEAK_TIME_COSTS,
    w_bounds        = (1.0, 10.0),
    trust_radius    = 0.5,
    max_iters       = MAX_ITERS,
    tol             = 1e-4,
    verbose         = true)

N = length(res.weight_ids)
do_nothing_obj   = res.history[1].obj          # iter-0 = uniform weights = do-nothing
do_nothing_ratio = res.history[1].obj          # (λ-weighted; per-period below)

println("\n" * "="^70)
println("RESULT — SLP-CC on motivation_c, T=$N_PERIODS, N=$N")
@printf("  do-nothing obj (iter 0)  : %.6f\n", do_nothing_obj)
@printf("  final obj (best)         : %.6f\n", res.palma_value)
@printf("  improvement              : %.1f%%  (%s)\n",
        100 * (do_nothing_obj - res.palma_value) / abs(do_nothing_obj),
        res.palma_value < do_nothing_obj ? "ESCAPED do-nothing" : "stuck at do-nothing")
@printf("  equality floor           : %.4f  (N=%d)\n", res.equality_floor, N)
println("  --- goal 1: DiffOpt cost ---")
@printf("  adjoint (DiffOpt) solves : %d   = 4 × %d iters  (T-independent)\n",
        res.n_adjoint, res.slp_iters)
@printf("  MILP path would need     : %d   = T·N × %d iters forward Jacobian cols\n",
        N_PERIODS * N * res.slp_iters, res.slp_iters)
@printf("  primal (no-sens) solves  : %d\n", res.n_primal)
println("  --- per-period Palma vs floor ---")
for (t, r) in enumerate(res.period_ratios)
    @printf("    period %d (h=%2d, λ=%5.2f):  Palma=%.4f  (floor %.4f)\n",
            t, PERIOD_HOURS[t], PEAK_TIME_COSTS[t], r, res.equality_floor)
end
println("  trajectory:")
for h in res.history
    @printf("    iter %3d  obj=%.6f  step=%.3e  min_bot=%.3e\n",
            h.iter, h.obj, h.step, h.min_bot)
end
println("="^70)
