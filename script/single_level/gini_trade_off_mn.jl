#=
Multi-period, single-level shed-Gini vs efficiency trade-off
================================================================

Gini analogue of `palma_trade_off_mn.jl`. SAME structure: builds a multinetwork
MLD problem (per-period constraints from `build_mn_mc_mld_min_max[_integer]`,
instantiated purely for the constraint set), then attaches **cost-weighted
AGGREGATE shed-Gini** machinery — ONE sort over a single per-load aggregate,
ONE Gini coefficient:

    cost-weighted aggregate SHED per load:  Υ^agg_j = Σ_t λ_t · pshed_{t,j}
    sort the length-n vector Υ^agg ONCE  →  sorted[1..n] (ascending)
    cumsum     = Σ_{i=1}^{n-1}(sum of i LARGEST) = Σ_k (k−1)·sorted[k]   ← top cumulative
    denom      = n · Σ_j Υ^agg_j                      ← Gini DENOMINATOR (n·Σshed)
    fairness   = Gini = −(n−1)/n + 2·cumsum / denom   ← Lorenz/cumulative form
    objective  = α · (−(n−1)/n + 2·σ·cumsum)  +  (1 − α) · eff_total

Like the Palma trade-off, this is a SINGLE sort over the per-load cost-weighted
aggregate Υ^agg (NOT per period), structurally mirroring `palma_trade_off_mn.jl`.
The fairness functional is the Lorenz/cumulative Gini (martin_using_2025). Written
with the TOP (descending) cumulative `cumsum = Σ_{i=1}^{n-1}(sum of the i largest
sorted) = Σ_k (k−1)·sorted[k]` (coefficients 0,1,…,n−1), it is exactly
    Gini = −(n−1)/n + 2·cumsum / (n·Σ sorted) = FairLoadDelivery.gini_index,
algebraically equal to the small-first paper form
`1 − 1/n − 2·(Σ_i Σ_{j≤i} sorted[j])/(n·Σ)`. We use the TOP cumulative because
MINIMIZING Gini ⇔ MINIMIZING σ·cumsum — a nonnegative product driven toward 0, so
its McCormick relaxation is bounded below WITHOUT a σ upper bound, exactly like
Palma's `min σ·top_sum`. (The small-first cumulative is MAXIMIZED when Gini is
minimized → unbounded relaxation → integer sweep frozen at the warm start.)

Charnes-Cooper "weak" form, ONE σ over the aggregate:

    σ free, lower-bounded (NO upper bound needed)
    σ · denom = 1                                ← bilinear constraint
    min σ · cumsum in the objective              ← bilinear; nonneg & MINIMIZED ⇒ bounded

both via Gurobi `NonConvex=2`. As in the Palma trade-off, pshed is a VARIABLE
here (not a fixed `pshed_prev`), so σ·pshed is genuinely bilinear and the single-
level Gini is a NonConvex MIQCP — NOT the pure formal-CC MILP used in the bilevel
upper level (`gini_index_minimization_formal_cc`, where `pshed_prev` is constant).
The bilevel sorts PER PERIOD; this aggregate single-sort mirrors the Palma
trade-off's structure by deliberate choice (see the project discussion), so it is
NOT a per-period parity match to the gini bilevel — it is the Palma-trade-off
structure applied to the Gini functional.

ONE σ over the cost-weighted aggregate is also feasible for integer shedding:
σ·denom=1 only needs `Σ_j Υ^agg_j > 0` (SOME aggregate shed across the horizon) —
even weaker than Palma's `bot40(Υ^agg) > 0`, because Gini's denominator is the
full-vector sum, not the bottom-40%.

The Gini *permutation* `a` stays BINARY even when the MLD is LP-relaxed
(`relaxed=true`): relaxing it collapses the McCormick `u` and breaks the sort.

The default sort target is SHED (`GINI_TARGET` defaults to "shed",
sort_target=:pshed). Setting `GINI_TARGET=served` sorts the cost-weighted
aggregate SERVED instead — a supported non-default variant.

An UNCOSTED aggregate served-Gini (λ_t=1, served TOTALS) is also computed as a
SECONDARY diagnostic; saved/plotted alongside but NOT the optimized metric.

This script is a sandbox: Gini machinery is defined locally and not pushed into
`src/`, exactly like `palma_trade_off_mn.jl`.

Only the absolute-pshed variant is implemented.

Smoke knobs (env, do not change the default full run):
    SMOKE=true            → SELECTED_HOURS=[4,18] (T=2) for a fast check
    ALPHA_POINTS=2        → fewer α sweep points
    TIME_LIMIT_ALPHA=30   → shorter per-α Gurobi TimeLimit (seconds)
    GINI_TARGET=served    → sort served instead of shed
    RELAXED=true          → LP-relaxed MLD lower level
=#

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS, Juniper
using HSL_jll
using Plots, StatsPlots
using Random
using Distributions
using DiffOpt
using JuMP
using LinearAlgebra, SparseArrays
using Statistics
using PowerPlots
using DataFrames
using CSV
using Dates
import MathOptInterface as MOI

const PMD  = PowerModelsDistribution
const _PMD = PowerModelsDistribution   # matches the alias used inside FairLoadDelivery so the local Gini machinery below can call _PMD.var/_PMD.ref/_PMD.ids the same way the module-internal code does.

include("../../src/implementation/visualization.jl")

# Unified 9pt font defaults for every figure in this script.
include(joinpath(@__DIR__, "../figure_defaults.jl"))

# ============================================================
# CONFIGURATION
# ============================================================
case_name = "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss"  # no-bd; matches the Palma trade-off + bilevel finals
case = "more_meshed_6_bus"   # baseline label; matches the pinned finals JLD2 key.

dir = @__DIR__
case_path = joinpath(dir, case_name)
date = Dates.format(now(), "yyyy-mm-dd")
LS_PERCENT = 0.8
pshed_type = "absolute"  # only absolute supported in this script
# When true, the lower-level MLD is genuinely LP-relaxed; pshed is continuous and
# the per-load aggregate shed distribution is non-degenerate. The Gini permutation
# `a` stays BINARY regardless. When false the MLD switch/block vars are binary.
relaxed = get(ENV, "RELAXED", "false") == "true"   # RELAXED=true → relaxed; unset/false → integer
# Which quantity the Gini objective sorts: "shed" (default) or "served".
GINI_SORT = get(ENV, "GINI_TARGET", "shed") == "served" ? :pd : :pshed
# Multi-period setup mirrors palma_trade_off_mn.jl so results are directly comparable.
SELECTED_HOURS    = [4, 6, 8, 12, 15, 18, 20, 22]   # T=8: trough, ramp, midday, pre-peak, evening peak, descent
if get(ENV, "SMOKE", "false") == "true"
    SELECTED_HOURS = [4, 18]   # T=2 fast smoke
elseif haskey(ENV, "SELECTED_HOURS")
    SELECTED_HOURS = parse.(Int, split(ENV["SELECTED_HOURS"], ","))
end

N_PERIODS      = length(SELECTED_HOURS)
# Peak-stress multiplier: scales every schedule value uniformly.
PEAK_STRESS = 1.0
# When true, each schedule is first divided by its own daily mean so the daily-
# average per-load scale equals PEAK_STRESS exactly. Matches the bilevel /
# Palma-trade-off convention so trade-off vs bilevel results share a demand axis.
CENTER_AT_NOMINAL = true
PEAK_TIME_COSTS = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                        for h in SELECTED_HOURS]
# Representative periods for the 3D Pareto; clamp to available periods (so SMOKE T=2 is fine).
REP_PERIODS_FULL = [1, 4, 6]   # T=8 indices → hours 4, 12, 18 (trough, midday, evening peak)
REP_PERIODS = filter(t -> t <= N_PERIODS, REP_PERIODS_FULL)
isempty(REP_PERIODS) && (REP_PERIODS = collect(1:N_PERIODS))

# Gini sweep: each solve is a multi-period bilinear MIP (one σ·denom = 1 + bilinear
# σ·cumsum objective), like the Palma trade-off.
alpha_points = parse(Int, get(ENV, "ALPHA_POINTS", "12"))
alphas = collect(LinRange(0.0, 1.0, alpha_points))

# Per-α Gurobi TimeLimit (seconds). Env-overridable for smoke runs.
TIME_LIMIT_ALPHA = parse(Int, get(ENV, "TIME_LIMIT_ALPHA", "180"))

# ============================================================
# NETWORK SETUP + MULTINETWORK DATA
# ============================================================
eng, math, lbs, critical_id = setup_network(case_path, LS_PERCENT;
    switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT)

mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS,
    center_at_nominal = CENTER_AT_NOMINAL)

# Pure-diagnostic per-period aggregate demand fraction (not consumed by the MIP).
LOAD_SCALE_FACTORS = FairLoadDelivery.aggregate_demand_fraction(math, N_PERIODS;
    hours = SELECTED_HOURS, center_at_nominal = CENTER_AT_NOMINAL) .* PEAK_STRESS
@info "LOAD_SCALE_FACTORS (agg_scales per period): $(round.(LOAD_SCALE_FACTORS, digits=3))"

println("Load profile assignments for $case:")
for row in FairLoadDelivery.profile_assignment_table(math)
    println("  ", row)
end
nw_ids_sorted     = sort(collect(keys(mn_data["nw"])), by = x -> parse(Int, x))
nw_ids_int_sorted = parse.(Int, nw_ids_sorted)        # ints, matches PMD nw_ids
n_loads           = length(mn_data["nw"][nw_ids_sorted[1]]["load"])

if relaxed
    rel = "_relaxed"
    kind = "relaxed"
else
    rel = ""
    kind = "integer"
end
# Shed-Gini is the canonical run (→ main folder). The served-Gini experiment
# writes to a separate folder so it doesn't overwrite it.
obj_suffix = GINI_SORT === :pd ? "_servedobj" : ""
output_dir = joinpath(@__DIR__, "../../results/$date/gini$(rel)_trade_off_mn$(obj_suffix)")
isdir(output_dir) || mkpath(output_dir)

# ============================================================
# Gini helpers (local — sandbox, mirrors palma_trade_off_mn.jl)
# ============================================================

"Gini of a numeric vector (= FairLoadDelivery.gini_index). Returns NaN if Σ < eps_sum."
function gini_value(vals::AbstractVector; eps_sum::Float64 = 1e-6)
    v = collect(float.(vals))
    s = sum(v)
    return s < eps_sum ? NaN : FairLoadDelivery.gini_index(v)
end

"""
Attach COST-WEIGHTED AGGREGATE shed-Gini machinery to a multinetwork JuMP model:
ONE sort over a single per-load aggregate → ONE Gini coefficient. Mirrors
`add_palma_machinery_cw_aggregate!`, swapping the top10/bot40 ratio for the full
Gini sorted-weighting:

    cost-weighted aggregate per load:  Υ^agg_j = Σ_t λ_t · (·)_{t,j}
    sort the length-n vector Υ^agg ONCE  →  sorted[1..n] (ascending)
    cumsum   = Σ_k (k−1)·sorted[k]               ← TOP cumulative (≥ 0)
    denom    = n · Σ_j Υ^agg_j                    ← Gini DENOMINATOR
    fairness = Gini = −(n−1)/n + 2·cumsum/denom,  one σ,   σ·denom = 1 (min σ·cumsum)

`u` is the McCormick BILINEAR term `u[i,j] = a[i,j]·Υ^agg_j` that linearizes the
SINGLE n×n sort. ONE σ for the whole horizon (Charnes-Cooper weak form):
σ·denom=1 (constraint) and σ·cumsum (objective) are bilinear → Gurobi
`NonConvex=2`.

The Gini permutation `a` stays BINARY even when the MLD is LP-relaxed
(`relax_binary` governs only the permutation).

`λ` must align with `nw_ids_int` (λ[ti] is the cost of period nw_ids_int[ti]).
"""
function add_gini_machinery_cw_aggregate!(pm; nw_ids_int::Vector{Int},
                                          λ::Vector{Float64},
                                          relax_binary::Bool = false,
                                          sort_target::Symbol = :pshed)  # :pshed = shed-Gini (default); :pd = served-Gini
    model = pm.model

    nw0      = nw_ids_int[1]
    load_ids = sort(collect(_PMD.ids(pm, nw0, :load)))
    n        = length(load_ids)
    T        = length(nw_ids_int)
    @assert length(λ) == T "λ length must match number of periods"

    # Per-period nameplate demand P_{t,i}.
    P_period = Dict{Int,Vector{Float64}}()
    for nw in nw_ids_int
        P_period[nw] = [sum(_PMD.ref(pm, nw, :load, i)["pd"]) for i in load_ids]
    end
    total_demand_all = sum(sum(P_period[nw]) for nw in nw_ids_int)
    @assert total_demand_all > 0 "Aggregate demand is zero; Gini is undefined."

    # Cost-weighted aggregate shed/served per load: Υ^agg_j = Σ_t λ_t·(·)_{t,j}.
    pshed_agg = JuMP.@expression(model, [k = 1:n],
        sum(λ[nw+1] * sum(_PMD.var(pm, nw, :pshed, load_ids[k])) for nw in nw_ids_int))
    pserved_agg = JuMP.@expression(model, [k = 1:n],
        sum(λ[nw+1] * sum(_PMD.var(pm, nw, :pd, load_ids[k])) for nw in nw_ids_int))

    # The aggregate vector actually sorted (shed by default; served if sort_target=:pd).
    Υ_agg = sort_target === :pshed ? pshed_agg : pserved_agg
    # McCormick upper bound: cost-weighted aggregate demand Σ_t λ_t·pd_{t,j} ≥ Υ^agg_j.
    P_agg = [sum(λ[nw+1] * P_period[nw][j] for nw in nw_ids_int) for j in 1:n]

    # ONE permutation + McCormick aux for the single aggregate sort.
    a = relax_binary ?
        JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, upper_bound = 1, base_name = "gini_a") :
        JuMP.@variable(model, [1:n, 1:n], Bin, base_name = "gini_a")
    u = JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, base_name = "gini_u")

    # McCormick bilinear term: u[i,j] = a[i,j] · Υ^agg_j, 0 ≤ Υ^agg_j ≤ P_agg_j.
    for i in 1:n, j in 1:n
        Pj = P_agg[j]
        JuMP.@constraint(model, u[i, j] >= Υ_agg[j] + a[i, j] * Pj - Pj)
        JuMP.@constraint(model, u[i, j] <= a[i, j] * Pj)
        JuMP.@constraint(model, u[i, j] <= Υ_agg[j])
    end
    for i in 1:n
        JuMP.@constraint(model, sum(a[i, j] for j in 1:n) == 1)
    end
    for j in 1:n
        JuMP.@constraint(model, sum(a[i, j] for i in 1:n) == 1)
    end

    sorted = JuMP.@expression(model, [i = 1:n], sum(u[i, j] for j in 1:n))
    for k in 1:(n - 1)
        JuMP.@constraint(model, sorted[k] <= sorted[k + 1])
    end

    # Lorenz/cumulative Gini, TOP-cumulative decomposition. With sorted ascending,
    # the top cumulative cumsum = Σ_{i=1}^{n-1}(sum of the i LARGEST sorted values)
    # = Σ_k (k−1)·sorted[k] (coefficients 0,1,…,n−1, weighting LARGE values). Then
    #     Gini = −(n−1)/n + 2·cumsum / denom,   denom = n·Σ sorted.
    # MINIMIZING Gini ⇔ MINIMIZING σ·cumsum, a nonnegative product driven toward 0,
    # so its McCormick relaxation is bounded below WITHOUT a σ upper bound — exactly
    # like Palma's min σ·top_sum. (The small-first cumulative Σ_i (n−i)·sorted[i] is
    # the algebraically-equal paper form, but minimizing Gini MAXIMIZES it →
    # unbounded relaxation → integer sweep frozen at the warm start.)
    cumsum   = JuMP.@expression(model, sum((k - 1) * sorted[k] for k in 1:n))
    denom    = JuMP.@expression(model, n * sum(Υ_agg[j] for j in 1:n))

    # Weak Charnes-Cooper: σ free, σ·denom=1 (NonConvex=2).
    σ = JuMP.@variable(model, base_name = "gini_sigma", lower_bound = 1e-8)
    JuMP.@constraint(model, σ * denom == 1.0)

    # UNCOSTED efficiency: total fraction of horizon demand shed (NO λ), so α=1 is
    # exactly the cost-weighted aggregate Gini and (1−α) trades it against plain
    # efficiency.
    eff_total = JuMP.@expression(model,
        sum(sum(sum(_PMD.var(pm, nw, :pshed, d)) for d in _PMD.ids(pm, nw, :load))
            for nw in nw_ids_int) / total_demand_all)

    return (
        n = n, T = T, load_ids = load_ids, nw_ids_int = nw_ids_int,
        P_period = P_period, P_agg = P_agg, total_demand_all = total_demand_all,
        pshed_agg = pshed_agg, pserved_agg = pserved_agg,
        Υ_agg = Υ_agg,
        a = a, u = u, sorted = sorted,
        cumsum = cumsum, denom = denom, σ = σ,
        eff_total = eff_total,
    )
end

"""
Set the multinetwork objective:

    min  α · (−(n−1)/n + 2·σ·cumsum)  +  (1 − α) · eff_total

The fairness part is the Lorenz/cumulative Gini `−(n−1)/n + 2·cumsum/denom`, with
the TOP cumulative `cumsum = Σ_k (k−1)·sorted[k]` and `denom = n·Σ_j Υ^agg_j` over
the single per-load cost-weighted aggregate `Υ^agg_j = Σ_t λ_t·pshed_{t,j}`. At the
CC optimum `σ·denom = 1`, so the fairness part equals `Gini(Υ^agg)`. Because `cumsum`
is nonnegative and MINIMIZED, `σ·cumsum` is bounded below (McCormick under-estimator
≥ 0) — no σ upper bound needed, exactly like Palma's `min σ·top_sum`. The efficiency
term is the UNCOSTED total-shed fraction.
"""
function set_gini_alpha_objective_cw!(pm, gini::NamedTuple; alpha::Float64)
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1]"
    fairness_part = alpha * (-(gini.n - 1.0) / gini.n + 2.0 * gini.σ * gini.cumsum)
    eff_part      = (1.0 - alpha) * gini.eff_total
    JuMP.@objective(pm.model, Min, fairness_part + eff_part)
end

# ============================================================
# INSTANTIATE MULTINETWORK MODEL + ATTACH GINI MACHINERY
# ============================================================
build_fn = relaxed ?
    (pm -> FairLoadDelivery.build_mn_mc_mld_min_max(pm; alpha = 1.0)) :
    (pm -> FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm; alpha = 1.0))
mld_mn = _PMD.instantiate_mc_model(mn_data, _PMD.LinDist3FlowPowerModel, build_fn;
    multinetwork  = true,
    ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])

# Weak CC: ONE σ·denom=1 (constraint) and σ·cumsum (objective) are bilinear.
# Gurobi NonConvex=2 spatially branches. A single aggregate σ is tractable and
# feasible for integer shedding (only needs Σ Υ^agg > 0). Expect possible
# TIME_LIMIT returns at non-zero gap; the script accepts feasible incumbents.
JuMP.set_optimizer(mld_mn.model, Gurobi.Optimizer)
JuMP.set_optimizer_attribute(mld_mn.model, "NonConvex",    2)
JuMP.set_optimizer_attribute(mld_mn.model, "MIPGap",       1e-2)        # 1% — control, not tight
JuMP.set_optimizer_attribute(mld_mn.model, "TimeLimit",    TIME_LIMIT_ALPHA)
JuMP.set_optimizer_attribute(mld_mn.model, "MIPFocus",     1)
JuMP.set_optimizer_attribute(mld_mn.model, "NumericFocus", 2)

gini = add_gini_machinery_cw_aggregate!(mld_mn;
    nw_ids_int = nw_ids_int_sorted, λ = PEAK_TIME_COSTS, relax_binary = false,
    sort_target = GINI_SORT)

# ============================================================
# WARMSTART: pure-efficiency multi-period MLD (no Gini machinery)
# ============================================================
println("Warmstart: solving pure-efficiency multi-period MLD (no Gini machinery)…")
build_fn_eff = relaxed ?
    (pm -> FairLoadDelivery.build_mn_mc_mld_min_max(pm;
        peak_time_costs = PEAK_TIME_COSTS, alpha = 0.0)) :
    (pm -> FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm;
        peak_time_costs = PEAK_TIME_COSTS, alpha = 0.0))
mld_eff = _PMD.instantiate_mc_model(mn_data, _PMD.LinDist3FlowPowerModel, build_fn_eff;
    multinetwork = true, ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])
JuMP.set_optimizer(mld_eff.model, Gurobi.Optimizer)
JuMP.set_optimizer_attribute(mld_eff.model, "MIPGap",    1e-2)
JuMP.set_optimizer_attribute(mld_eff.model, "TimeLimit", 60 * 5)
JuMP.set_optimizer_attribute(mld_eff.model, "MIPFocus",  1)
JuMP.optimize!(mld_eff.model)
@assert JuMP.primal_status(mld_eff.model) == MOI.FEASIBLE_POINT  "warmstart efficiency MLD produced no incumbent — increase TimeLimit"
println("  efficiency-warmstart status=$(JuMP.termination_status(mld_eff.model))")

# Copy MLD decisions (pshed continuous + switch_state / z_block / z_demand binaries) onto mld_mn.
for nw in nw_ids_int_sorted
    for lid in gini.load_ids
        eff_var   = _PMD.var(mld_eff, nw, :pshed, lid)
        gini_var  = _PMD.var(mld_mn,  nw, :pshed, lid)
        JuMP.set_start_value.(gini_var, JuMP.value.(eff_var))
    end
    eff_v  = _PMD.var(mld_eff, nw)
    gini_v = _PMD.var(mld_mn,  nw)
    for sym in (:switch_state, :z_block, :z_demand)
        haskey(eff_v, sym) || continue
        JuMP.set_start_value.(gini_v[sym], JuMP.value.(eff_v[sym]))
    end
end

# Gini-specific starts: ONE permutation a/u from the cost-weighted AGGREGATE Υ^agg
# (the same single vector the objective sorts), plus ONE σ from its n·Σ.
shed_warm_agg = zeros(gini.n)   # Σ_t λ_t·pshed warm per load
pd_warm_agg   = zeros(gini.n)   # Σ_t λ_t·pd   warm per load
for (ti, nw) in enumerate(nw_ids_int_sorted)
    shed_t = [sum(JuMP.value.(_PMD.var(mld_eff, nw, :pshed, lid))) for lid in gini.load_ids]
    pd_t   = [sum(_PMD.ref(mld_eff, nw, :load, lid)["pd"])          for lid in gini.load_ids]
    shed_warm_agg .+= PEAK_TIME_COSTS[ti] .* shed_t
    pd_warm_agg   .+= PEAK_TIME_COSTS[ti] .* pd_t
end
# Warm-sort the SAME aggregate the objective sorts (shed by default).
Υ_warm  = GINI_SORT === :pshed ? shed_warm_agg : (pd_warm_agg .- shed_warm_agg)
perm    = sortperm(Υ_warm)            # ascending positions
a_start = zeros(gini.n, gini.n)
for (i, j) in enumerate(perm); a_start[i, j] = 1.0; end
u_start = a_start .* reshape(Υ_warm, 1, :)
for i in 1:gini.n, j in 1:gini.n
    JuMP.set_start_value(gini.a[i, j], a_start[i, j])
    JuMP.set_start_value(gini.u[i, j], u_start[i, j])
end
denom_warm = gini.n * sum(Υ_warm)
σ_start = 1.0 / max(denom_warm, 1e-8)
JuMP.set_start_value(gini.σ, σ_start)
println("  warm cost-wtd agg total=$(round(sum(shed_warm_agg), digits=2))   σ_start=$(round(σ_start, digits=6))")

# ============================================================
# ALPHA SWEEP
# ============================================================
# NaN-init so a TimeLimit-with-no-incumbent α leaves explicit NaN in the CSV/plots.
total_shed       = fill(NaN, alpha_points, N_PERIODS)     # per-period totals (for plots)
max_shed         = fill(NaN, alpha_points, N_PERIODS)
# PRIMARY fairness metric (the optimized one): cost-weighted aggregate shed-Gini =
#   gini_cost_weighted_log[α] = Gini(Υ^agg),  Υ^agg_j = Σ_t λ_t·pshed_{t,j}.
gini_cost_weighted_log = fill(NaN, alpha_points)
# SECONDARY (diagnostic only): UNCOSTED aggregate served-Gini (λ_t = 1).
gini_served_log  = fill(NaN, alpha_points)
# PLOTTED fairness metric: UNWEIGHTED aggregate shed-Gini = Gini(Σ_t pshed). The
# same neutral post-hoc metric the comparison plotter uses, so the standalone
# trade-off Pareto and the bilevel-vs-trade-off Pareto share one (raw-kW,
# unweighted) x-axis and Gini definition.
gini_unweighted_log = fill(NaN, alpha_points)
per_load_dist_a0 = fill(NaN, n_loads, N_PERIODS)
per_load_dist_a1 = fill(NaN, n_loads, N_PERIODS)
# Per-α, per-load aggregate shed (sum across periods).
per_load_agg     = fill(NaN, alpha_points, n_loads)
# Full (α × load × period) tensor — needed by replot scripts.
per_load_period_shed = fill(NaN, alpha_points, n_loads, N_PERIODS)
# Raw per-period solution dict per α; `nothing` at non-feasible α points.
solutions_per_alpha = Vector{Any}(nothing, alpha_points)

for (idx, alpha) in enumerate(alphas)
    set_gini_alpha_objective_cw!(mld_mn, gini; alpha = alpha)

    JuMP.optimize!(mld_mn.model)
    status = JuMP.termination_status(mld_mn.model)
    flush(stdout)
    println("alpha=$alpha  status=$status")
    flush(stdout)
    # Accept any feasible incumbent (OPTIMAL, TIME_LIMIT, or LOCALLY_SOLVED with a
    # solution) — `has_values` mirrors the bilevel's acceptance logic, so a
    # time-limited α with a valid incumbent is recorded instead of dropped to NaN.
    if !JuMP.has_values(mld_mn.model)
        @warn "no incumbent at alpha=$alpha (status=$status) — skipping; metrics left as NaN"
        continue
    end

    # Per-period totals/distributions (for the per-period plots + the JLD2 tensor).
    for (t, nw) in enumerate(nw_ids_int_sorted)
        per_load_shed_t = [sum(JuMP.value.(_PMD.var(mld_mn, nw, :pshed, lid)))
                           for lid in gini.load_ids]
        total_shed[idx, t] = sum(per_load_shed_t)
        max_shed[idx, t]   = maximum(per_load_shed_t)
        per_load_period_shed[idx, :, t] .= per_load_shed_t
        if idx == 1
            per_load_dist_a0[:, t] .= per_load_shed_t
        elseif idx == alpha_points
            per_load_dist_a1[:, t] .= per_load_shed_t
        end
    end

    # PRIMARY (optimized) fairness: cost-weighted AGGREGATE shed-Gini = Gini(Υ^agg).
    agg_vals = [JuMP.value(gini.Υ_agg[k]) for k in 1:gini.n]
    gini_cost_weighted_log[idx] = gini_value(agg_vals)

    # SECONDARY: uncosted served-Gini diagnostic.
    pserved_agg_vals = [JuMP.value(gini.pserved_agg[k]) for k in 1:gini.n]
    gini_served_log[idx] = gini_value(pserved_agg_vals)

    # UNWEIGHTED per-load aggregate shed (Σ_t pshed, NO λ) — drives ALL standalone
    # figures and the plotted Gini, matching the post-hoc comparison plots.
    pshed_agg_unw = [sum(per_load_period_shed[idx, j, t] for t in 1:N_PERIODS)
                     for j in 1:n_loads]
    per_load_agg[idx, :] .= pshed_agg_unw
    gini_unweighted_log[idx] = gini_value(pshed_agg_unw)

    model_fair = -(gini.n - 1.0) / gini.n + 2.0 * JuMP.value(gini.σ) * JuMP.value(gini.cumsum)
    flush(stdout)
    println("  agg total_shed = $(round(sum(agg_vals), digits=3))   ",
            "cost-wtd Gini(Υ^agg) (post-hoc) = $(round(gini_cost_weighted_log[idx], digits=4))   ",
            "model(−(n−1)/n+2σ·cumsum) = $(round(model_fair, digits=4))   ",
            "uncosted-agg-served-Gini (diag) = $(round(gini_served_log[idx], digits=4))")
    flush(stdout)

    # Capture full per-period solution dict for JLD2 persistence.
    try
        solutions_per_alpha[idx] = FairLoadDelivery._IM.build_solution(mld_mn)
    catch err
        @warn "build_solution failed at alpha=$alpha — JLD2 will have nothing for this α ($err)"
    end
end

# ============================================================
# 3D PARETO PLOT (rep periods only — total shed vs max shed along z = period)
# ============================================================
period_markers = [:circle, :diamond, :utriangle, :rect, :star5, :pentagon, :hexagon]
p3d = plot3d(xlabel = "total load shed (kW)",
             ylabel = "max load shed (kW)",
             zlabel = "period",
             title  = "Multi-period Pareto (Gini, $kind) — rep. periods",
             legend = :topright)
for (k, t) in enumerate(REP_PERIODS)
    plot3d!(p3d, total_shed[:, t], max_shed[:, t], fill(t, alpha_points),
            label  = "t=$t (λ=$(PEAK_TIME_COSTS[t]))",
            marker = period_markers[mod1(k, length(period_markers))], lw = 2,
            line_z = alphas)
end
savefig(p3d, joinpath(output_dir, "pareto3d_$(kind)_$(pshed_type).svg"))
display(p3d)

# ============================================================
# PER-PERIOD 2D PARETO PANEL (total shed vs max shed, color = alpha)
# ============================================================
panel_cols = N_PERIODS <= 6 ? N_PERIODS : 6
panel_rows = ceil(Int, N_PERIODS / panel_cols)
panel = plot(layout = (panel_rows, panel_cols),
             size = (220 * panel_cols, 180 * panel_rows),
             plot_title = "Per-period Pareto ($(pshed_type), Gini) — color = alpha")
for t in 1:N_PERIODS
    row = ceil(Int, t / panel_cols)
    col = ((t - 1) % panel_cols) + 1
    plot!(panel[t], total_shed[:, t], max_shed[:, t],
          marker = :circle, lc = :grey, marker_z = alphas, color = :cividis,
          xlabel = row == panel_rows ? "total shed (kW)" : "",
          ylabel = col == 1            ? "max shed (kW)"   : "",
          title  = "t=$t  λ=$(PEAK_TIME_COSTS[t])",
          colorbar = false, legend = false)
end
savefig(panel, joinpath(output_dir, "pareto_per_period_$(kind)_$(pshed_type).svg"))
display(panel)

# Cost-weighted aggregates retained for CSV export only (no plot).
weighted_total = [sum(PEAK_TIME_COSTS[t] * total_shed[i, t] for t in 1:N_PERIODS) for i in 1:alpha_points]
weighted_max   = [sum(PEAK_TIME_COSTS[t] * max_shed[i, t]   for t in 1:N_PERIODS) for i in 1:alpha_points]

# ============================================================
# Per-α aggregates and per-load-shed-vector norms.
# ============================================================
agg_total_shed = [all(isfinite, per_load_agg[i, :]) ? sum(per_load_agg[i, :]) : NaN
                  for i in 1:alpha_points]
agg_max_shed   = [all(isfinite, per_load_agg[i, :]) ? maximum(per_load_agg[i, :]) : NaN
                  for i in 1:alpha_points]

function shed_norms(shed_vec::AbstractVector{<:Real})
    any(!isfinite, shed_vec) && return (l1 = NaN, l2 = NaN, linf = NaN, cov = NaN)
    m = mean(shed_vec)
    s = std(shed_vec)
    return (
        l1   = norm(shed_vec, 1),
        l2   = norm(shed_vec, 2),
        linf = norm(shed_vec, Inf),
        cov  = m > 1e-9 ? s / m : NaN,
    )
end

norms_per_alpha = [shed_norms(per_load_agg[i, :]) for i in 1:alpha_points]
l1_vec   = [nm.l1   for nm in norms_per_alpha]
l2_vec   = [nm.l2   for nm in norms_per_alpha]
linf_vec = [nm.linf for nm in norms_per_alpha]
cov_vec  = [nm.cov  for nm in norms_per_alpha]

# ============================================================
# FIGURE 1: per-load aggregate shed distribution at α=0 and α=1, plus the Pareto
# of unweighted aggregate shed-Gini vs aggregate total shed.
# ============================================================
ref_nw0 = mn_data["nw"][nw_ids_sorted[1]]
load_labels = [ref_nw0["load"][lid]["name"]
               for lid in sort(collect(keys(ref_nw0["load"])), by = x -> parse(Int, x))]

FONT_KW = (tickfontsize = 22, guidefontsize = 22,
           titlefontsize = 26, legendfontsize = 14,
           fontfamily = "Computer Modern")
const ANNOT_PT = 14  # bar-top / pareto-endpoint annotation size

function build_dist_plot_agg(per_load_agg_vec::AbstractVector{<:Real}; ylim_max::Real)
    p = bar(load_labels, per_load_agg_vec,
        xlabel = "load",
        ylabel = "aggregate load shed (kW)",
        legend = false,
        color  = :steelblue,
        linecolor = :black,
        ylims = (0.0, ylim_max * 1.10);
        FONT_KW...)
    for (i, v) in enumerate(per_load_agg_vec)
        isfinite(v) || continue
        annotate!(p, i, v + ylim_max * 0.02,
            text("$(round(v, digits = 1))", ANNOT_PT, :center))
    end
    return p
end

ymax_shared = max(
    maximum(filter(isfinite, per_load_agg[1, :]);   init = 0.0),
    maximum(filter(isfinite, per_load_agg[end, :]); init = 0.0))

p_dist_a0 = build_dist_plot_agg(per_load_agg[1, :];   ylim_max = ymax_shared)
p_dist_a1 = build_dist_plot_agg(per_load_agg[end, :]; ylim_max = ymax_shared)

function build_pareto_curve(xvec, yvec, ylab)
    fin = findall(isfinite, yvec)
    p = plot(xvec[fin], yvec[fin],
        seriestype = :line, lc = :grey,
        marker = :circle, markersize = 14, color = :steelblue,
        markerstrokecolor = :steelblue,
        xlabel = "total load shed (kW)",
        ylabel = ylab,
        legend = false;
        FONT_KW...)
    if !isempty(fin)
        _ys = yvec[fin]; _xs = xvec[fin]; _αs = alphas[fin]
        yrange = maximum(_ys) - minimum(_ys)
        yoff = 0.05 * (yrange == 0 ? 1.0 : yrange)
        annotate!(p, _xs[1],   _ys[1]   + yoff, text("ν=$(round(_αs[1],   digits=2))", ANNOT_PT, :center))
        annotate!(p, _xs[end], _ys[end] + yoff, text("ν=$(round(_αs[end], digits=2))", ANNOT_PT, :center))
    end
    return p
end

# PRIMARY Pareto: UNWEIGHTED aggregate shed-Gini vs unweighted total shed.
p_pareto = build_pareto_curve(agg_total_shed, gini_unweighted_log,
    "Gini (unitless)")
# SECONDARY (diagnostic): UNCOSTED aggregate served-Gini (λ_t=1).
p_pareto_aggdiag = build_pareto_curve(agg_total_shed, gini_served_log,
    "Gini (unitless, uncosted served diagnostic)")

for (_name, _p) in (("alpha0", p_dist_a0), ("alpha1", p_dist_a1),
                    ("pareto", p_pareto), ("pareto_aggdiag", p_pareto_aggdiag))
    _fig = plot(_p; size = (900, 760),
        left_margin = 7Plots.mm, right_margin = 6Plots.mm,
        top_margin = 8Plots.mm, bottom_margin = 7Plots.mm)
    savefig(_fig, joinpath(output_dir,
        "summary_single_$(_name)_$(kind)_$(pshed_type).svg"))
    _name == "pareto" && display(_fig)
end

# ============================================================
# FIGURE 2: Pareto fronts (aggregate total shed vs L1 / L2 / L∞ / CoV of the
# per-load aggregate-shed vector).
# ============================================================
p_l1   = build_pareto_curve(agg_total_shed, l1_vec,   "L1 norm of shed (kW)")
p_l2   = build_pareto_curve(agg_total_shed, l2_vec,   "L2 norm of shed (kW)")
p_linf = build_pareto_curve(agg_total_shed, linf_vec, "L∞ norm of shed (kW)")
p_cov  = build_pareto_curve(agg_total_shed, cov_vec,  "CoV (stdev/mean)")

fig2 = plot(p_l1, p_l2, p_linf, p_cov,
    layout = (1, 4),
    size = (2200, 600),
    left_margin = 14Plots.mm, right_margin = 6Plots.mm,
    top_margin = 8Plots.mm, bottom_margin = 14Plots.mm)
savefig(fig2, joinpath(output_dir, "pareto_norms_$(kind)_$(pshed_type).svg"))
display(fig2)

# ============================================================
# CSV: per-period shed; aggregate fairness metrics per α
# ============================================================
period_rows = DataFrame(alpha = Float64[], period = Int[], lambda = Float64[],
    total_shed = Float64[], max_shed = Float64[])
for (i, a) in enumerate(alphas), t in 1:N_PERIODS
    push!(period_rows, (a, t, PEAK_TIME_COSTS[t],
                        total_shed[i, t], max_shed[i, t]))
end
CSV.write(joinpath(output_dir, "gini_sweep_mn_per_period_$(pshed_type).csv"), period_rows)

agg_rows = DataFrame(alpha = alphas,
    agg_total_shed      = [sum(total_shed[i, :]) for i in 1:alpha_points],
    cost_weighted_shed  = weighted_total,
    cost_weighted_max   = weighted_max,
    gini_cost_weighted  = gini_cost_weighted_log,   # PRIMARY: Gini(Υ^agg), Υ^agg=Σ_t λ_t·pshed
    gini_unweighted     = gini_unweighted_log,      # PLOTTED: Gini(Σ_t pshed)
    gini_served_uncosted = gini_served_log)         # SECONDARY: uncosted served-Gini diagnostic
CSV.write(joinpath(output_dir, "gini_sweep_mn_aggregate_$(pshed_type).csv"), agg_rows)

# ============================================================
# PERSIST SWEEP DATA FOR STANDALONE VISUALIZATION SCRIPTS
# Same schema as palma_trade_off_mn.jl (gini-named metric keys, fair_func="gini").
# ============================================================
using JLD2
math_ref = mn_data["nw"][nw_ids_sorted[1]]
bus_name_map = FairLoadDelivery.build_bus_name_maps(math_ref)
ref_load_ids = sort(collect(keys(math_ref["load"])), by = x -> parse(Int, x))
load_bus_ids   = [math_ref["load"][lid]["load_bus"] for lid in ref_load_ids]
load_bus_names = [get(bus_name_map, bid, "bus_$bid") for bid in load_bus_ids]

per_load_period_pd = zeros(n_loads, N_PERIODS)
for (t, nw_id) in enumerate(nw_ids_sorted)
    nw_data = mn_data["nw"][nw_id]
    for (j, lid) in enumerate(ref_load_ids)
        per_load_period_pd[j, t] = sum(nw_data["load"][lid]["pd"])
    end
end

jld_path = joinpath(output_dir, "gini_sweep_mn_$(case)_$(pshed_type).jld2")
JLD2.jldsave(jld_path;
    alphas               = alphas,
    per_load_period_shed = per_load_period_shed,  # alpha × load × period
    per_load_period_pd   = per_load_period_pd,    # load × period
    per_load_agg         = per_load_agg,          # alpha × load (sum over periods)
    total_shed           = total_shed,            # alpha × period
    max_shed             = max_shed,              # alpha × period
    # PRIMARY fairness metric (optimized): Gini(Υ^agg), Υ^agg_j = Σ_t λ_t·pshed_{t,j}
    gini_cost_weighted_log = gini_cost_weighted_log,   # alpha
    # PLOTTED: unweighted aggregate shed-Gini = Gini(Σ_t pshed)
    gini_unweighted_log  = gini_unweighted_log,
    # SECONDARY (diagnostic): uncosted aggregate served-Gini (λ_t=1)
    gini_served_log      = gini_served_log,
    load_labels          = load_labels,
    load_bus_ids         = load_bus_ids,
    load_bus_names       = load_bus_names,
    LOAD_SCALE_FACTORS   = LOAD_SCALE_FACTORS,
    PEAK_TIME_COSTS      = PEAK_TIME_COSTS,
    N_PERIODS            = N_PERIODS,
    PEAK_STRESS          = PEAK_STRESS,
    CENTER_AT_NOMINAL    = CENTER_AT_NOMINAL,
    case                 = case,
    pshed_type           = pshed_type,
    fair_func            = "gini",
    solutions_per_alpha  = solutions_per_alpha,
    math_switch          = math["switch"],
    math_branch          = math["branch"],
    math_bus             = math["bus"],
    math_load            = math["load"],
    load_block_sets      = lbs,
)
println("Saved trade-off sweep data → $jld_path")

println("Done. Results written to: ", output_dir)
