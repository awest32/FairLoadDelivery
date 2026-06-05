#=
Multi-period, single-level shed-Palma vs efficiency trade-off
================================================================

Multi-period analogue of `palma_trade_off.jl`. Builds a multinetwork MLD
problem (per-period constraints from `build_mn_mc_mld_min_max[_integer]`,
which we instantiate purely for the constraint set), then attaches
**cost-weighted AGGREGATE shed-Palma** machinery — ONE sort over a single
per-load aggregate, ONE ratio — whose fairness objective is matched to the
bilevel upper level (`palma_ratio_minimization` in `load_shed_as_parameter.jl`):

    cost-weighted aggregate SHED per load:  Υ^agg_j = Σ_t λ_t · pshed_{t,j}
    sort the length-n vector Υ^agg ONCE  →  top10%(Υ^agg),  bot40%(Υ^agg)
    top_sum    = top10%(Υ^agg)                      ← Palma NUMERATOR
    bot_sum    = bot40%(Υ^agg)                      ← Palma DENOMINATOR
    fairness   = Palma = top_sum / bot_sum
    objective  = α · (σ · top_sum)  +  (1 − α) · eff_total

The Palma sort runs ONCE over the per-load cost-weighted aggregate shed Υ^agg
(λ = peak_time_costs / ρ_t folds the period costs into each load's aggregate),
NOT per period — a SINGLE n×n permutation, NOT a per-period `a[t]` loop. This is
structurally IDENTICAL to the bilevel's `palma_ratio_minimization`, which sorts
the same cost-weighted aggregate once. `eff_total` is the UNCOSTED total-shed
fraction (no λ), so α=1 recovers exactly the bilevel fairness objective and
(1−α) trades it against plain efficiency. Single-level and bilevel thus share
ONE fairness objective and ONE sort structure.

Charnes-Cooper "weak" form, ONE σ over the aggregate top/bot:

    σ free, lower-bounded
    σ · bot_sum = 1                              ← bilinear constraint
    σ · top_sum in the objective                 ← bilinear

both via Gurobi `NonConvex=2`. ONE sort + ONE σ over the cost-weighted aggregate
is what makes this both tractable (one bilinear σ, not T) AND feasible for
integer shedding: σ·bot_sum=1 only needs `bot40(Υ^agg) > 0` (the smallest-shed
loads have SOME aggregate shed across the horizon), whereas a per-period σ_t form
was infeasible because whole-block integer shedding zeroes a period's bottom-40%
(see `script/diagnostics/probe_per_period_palma_feasibility.jl`).

The Palma *permutation* `a` stays BINARY even when the MLD is LP-relaxed
(`relaxed=true`): relaxing it collapses the McCormick `u` and breaks the sort.

The default sort target is SHED (`PALMA_SORT`/`PALMA_TARGET` defaults to "shed",
sort_target=:pshed). Setting `PALMA_TARGET=served` (sort_target=:pd) sorts the
cost-weighted aggregate SERVED instead — a supported non-default variant.

An UNCOSTED aggregate served-Palma (λ_t=1, served TOTALS) is also computed as a
SECONDARY diagnostic; saved/plotted alongside but NOT the optimized metric.

This script is a sandbox: Palma machinery is defined locally and not
pushed into `src/`. The existing `objective_mn_palma_mld` in
`src/core/objective.jl` is the per-period shed-Palma variant — we
deliberately do NOT use it here.

Only the absolute-pshed variant is implemented (proportional Palma is
skipped per project memory: with binary load-shed the sort key collapses).
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
const _PMD = PowerModelsDistribution   # matches the alias used inside FairLoadDelivery (src/FairLoadDelivery.jl:41) so the local Palma machinery below (lines 225+) can call _PMD.var/_PMD.ref/_PMD.ids the same way the module-internal code does.

include("../../src/implementation/visualization.jl")

# Unified 9pt font defaults for every figure in this script.
include(joinpath(@__DIR__, "../figure_defaults.jl"))

# ============================================================
# CONFIGURATION
# ============================================================
case_name = "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss"  # no-bd; matches pinned T=24 finals + run_validation_mn_slp.jl
#case_name = "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_bd_good4integer.dss"  # BD variant (T=5 HEAD)
#case_name = "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss"
case = "more_meshed_6_bus"   # baseline label; matches the pinned finals JLD2 key. BD variant was "more_meshed_bd_6_bus".

dir = @__DIR__
case_path = joinpath(dir, case_name)
date = Dates.format(now(), "yyyy-mm-dd")
LS_PERCENT = 0.8
pshed_type = "absolute"  # only absolute supported in this script
# When true, the lower-level MLD is genuinely LP-relaxed (continuous
# switch_state / z_block / z_demand via `build_mn_mc_mld_min_max`, i.e.
# `_build_mn_period_fair!(…; relax=true)`), so pshed is continuous and the
# per-load aggregate shed distribution is non-degenerate — the right "relaxed"
# analogue for the aggregate Palma sort. The Palma *permutation* `a` stays
# BINARY regardless (relaxing it collapses the McCormick `u` to zero and
# breaks the sort — see legacy/palma_reformulation/README.md). When false the
# MLD switch/block vars are binary (`build_mn_mc_mld_min_max_integer`).
relaxed = get(ENV, "RELAXED", "false") == "true"   # RELAXED=true → relaxed; unset/false → integer (env-overridable for orchestration)
# Which quantity the Palma objective sorts: "shed" (default — matches the bilevel
# upper level, which optimizes the cost-weighted aggregate SHED Palma) or "served".
PALMA_SORT = get(ENV, "PALMA_TARGET", "shed") == "served" ? :pd : :pshed
# Multi-period setup mirrors min_max_trade_off_mn.jl so results are directly comparable.
# Per-load profiles follow Hamilton & Aliprantis (PECI 2023): each load name is
# deterministically mapped to (schedule, ±1h shift). The per-period demand level
# is implicit in the reported load-shed values, so no aggregate-scale label is
# carried in plots or CSVs.
# Downsampled hours-of-day (0-indexed); mirrors min_max_trade_off_mn.jl so
# results stay comparable across fair-funcs.
#SELECTED_HOURS = [4, 18, 8]   # 13-bus motivation_c: T=3, peak in middle position so plots show off-peak → peak → off-peak. λ=[5.0, 30.0, 5.01].
#SELECTED_HOURS    = [4, 12, 15, 18, 22]   # T=5 (BD HEAD): trough, midday, pre-peak, evening peak, descent
#SELECTED_HOURS    = collect(0:23)   # T=24 full diurnal cycle
SELECTED_HOURS    = [4, 6, 8, 12, 15, 18, 20, 22]   # T=8: trough, ramp, midday, pre-peak, evening peak, descent (defense; original MILP method tractable here)

N_PERIODS      = length(SELECTED_HOURS)
# Peak-stress multiplier: scales every schedule value uniformly so peak-hour
# demand pushes past nameplate. Bump up for more shedding, down for less.
PEAK_STRESS = 1.0
# When true, each schedule is first divided by its own daily mean so the
# daily-average per-load scale equals PEAK_STRESS exactly and the nameplate pd
# is the daily mean (peaks reach ~1.15× nominal at the daily max). Matches the
# bilevel run_efficiency_mn.jl / run_validation_mn.jl convention so trade-off
# vs bilevel results are on the same demand axis.
CENTER_AT_NOMINAL = true
# OLD uniform-scalar profile (commented for reference / quick A/B):
# const LOAD_SCALE_FACTORS = [round(s, digits=3) for s in LinRange(0.7, 1.0, N_PERIODS)]
PEAK_TIME_COSTS = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                        for h in SELECTED_HOURS]
#REP_PERIODS = [1, 3, 5]   # T=5 indices → hours 4, 15, 22
REP_PERIODS = [1, 4, 6]   # T=8 indices → hours 4, 12, 18 (trough, midday, evening peak)

# Palma sweep: kept smaller than min-max because each solve is a multi-period
# bilinear MIP (one σ · bot_sum = 1 + bilinear σ · top_sum objective).
alpha_points = 12
alphas = collect(LinRange(0.0, 1.0, alpha_points))

# ============================================================
# NETWORK SETUP + MULTINETWORK DATA
# ============================================================
eng, math, lbs, critical_id = setup_network(case_path, LS_PERCENT;
    switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT)

# OLD uniform-scalar multinetwork builder. Kept commented for reference;
# replaced by `create_multinetwork_data_profiled` (per-load, per-phase schedules
# from Hamilton & Aliprantis 2023).
#
# function create_multinetwork_data(base_math::Dict{String,Any}, n_periods::Int, load_scales::Vector{Float64})
#     @assert length(load_scales) == n_periods
#     mn_data = Dict{String,Any}(
#         "multinetwork" => true,
#         "per_unit"     => true,
#         "data_model"   => PMD.MATHEMATICAL,
#         "nw"           => Dict{String,Any}()
#     )
#     for key in ["baseMVA", "basekv", "bus_lookup", "settings"]
#         haskey(base_math, key) && (mn_data[key] = deepcopy(base_math[key]))
#     end
#     for t in 1:n_periods
#         nw_id = string(t - 1)
#         nw_data = deepcopy(base_math)
#         delete!(nw_data, "multinetwork")
#         scale = load_scales[t]
#         for (_, load) in nw_data["load"]
#             load["pd"] = load["pd"] .* scale
#             load["qd"] = load["qd"] .* scale
#         end
#         nw_data["time_period"] = t
#         nw_data["load_scale"]  = scale
#         mn_data["nw"][nw_id] = nw_data
#     end
#     return mn_data
# end
# mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)

mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS,
    center_at_nominal = CENTER_AT_NOMINAL)

# Pure-diagnostic per-period aggregate demand fraction (not consumed by the
# MIP). Saved to the JLD2 below so plot/log annotations can label periods
# with their effective aggregate scale, matching the bilevel scripts.
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
# Shed-Palma is the canonical run (→ main folder, matches the bilevel). The
# served-Palma experiment writes to a separate folder so it doesn't overwrite it.
obj_suffix = PALMA_SORT === :pd ? "_servedobj" : ""
output_dir = joinpath(@__DIR__, "../../results/$date/palma$(rel)_trade_off_mn$(obj_suffix)")
isdir(output_dir) || mkpath(output_dir)

# ============================================================
# Palma helpers (local — copied from palma_trade_off.jl sandbox)
# ============================================================

"Indices in ASCENDING-sorted order. top10 = ceil(0.1n) largest, bot40 = floor(0.4n) smallest."
function compute_palma_indices(n::Int)
    n_bot = max(1, floor(Int, 0.4 * n))
    n_top = max(1, ceil(Int, 0.1 * n))
    bottom_40_idx = collect(1:n_bot)
    top_10_idx    = collect((n - n_top + 1):n)
    return top_10_idx, bottom_40_idx
end

"Palma ratio of a numeric vector (top10% / bot40% of sorted values). Returns Inf if denom < eps_denom."
function palma_ratio_value(vals::AbstractVector; eps_denom::Float64 = 1e-6)
    n = length(vals)
    s = sort(collect(vals))
    top_10_idx, bottom_40_idx = compute_palma_indices(n)
    num = sum(s[i] for i in top_10_idx)
    den = sum(s[i] for i in bottom_40_idx)
    return den < eps_denom ? Inf : num / den
end

"""
Attach COST-WEIGHTED AGGREGATE shed-Palma machinery to a multinetwork JuMP
model: ONE sort over a single per-load aggregate → ONE Palma ratio. This is the
single-level control whose **fairness objective matches the bilevel upper
level**:

    cost-weighted aggregate per load:  Υ^agg_j = Σ_t λ_t · (·)_{t,j}
    sort the length-n vector Υ^agg ONCE  →  top10%(Υ^agg),  bot40%(Υ^agg)
    top_sum = top10%(Υ^agg)        ← Palma NUMERATOR
    bot_sum = bot40%(Υ^agg)        ← Palma DENOMINATOR
    fairness = Palma = top_sum / bot_sum,   one σ,   σ·bot_sum = 1

The sorted quantity is the per-load cost-weighted aggregate (default SHED,
`sort_target=:pshed`; served if `:pd`). λ_t folds the period costs INTO each
load's aggregate Υ^agg_j = Σ_t λ_t·(·)_{t,j} BEFORE the single sort, so the
objective is a pure ratio σ·top_sum over one length-n vector. `u` is the
McCormick BILINEAR term `u[i,j] = a[i,j]·Υ^agg_j` that linearizes the SINGLE
n×n sort (NOT a per-period `u[t]` / `a[t]`).

ONE σ for the whole horizon (Charnes-Cooper weak form): σ·bot_sum=1 (constraint)
and σ·top_sum (objective) are bilinear → Gurobi `NonConvex=2`. ONE sort + ONE σ
over the cost-weighted AGGREGATE is what makes this both tractable and feasible
for integer shedding: σ·bot_sum=1 only needs `bot40(Υ^agg) > 0` (the
smallest-shed loads have SOME aggregate shed across the horizon), whereas a
per-period σ_t form was infeasible because whole-block integer shedding zeroes a
period's bottom-40% (see `script/diagnostics/probe_per_period_palma_feasibility.jl`).

The Palma permutation `a` stays BINARY even when the MLD itself is LP-relaxed
(`relax_binary` governs only the permutation): relaxing it collapses the
McCormick `u` and breaks the sort (legacy/palma_reformulation/README.md).

`λ` must align with `nw_ids_int` (λ[ti] is the cost of period nw_ids_int[ti]).
"""
function add_palma_machinery_cw_aggregate!(pm; nw_ids_int::Vector{Int},
                                           λ::Vector{Float64},
                                           relax_binary::Bool = false,
                                           sort_target::Symbol = :pshed)  # :pshed = shed-Palma (matches the bilevel; default); :pd = served-Palma
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
    @assert total_demand_all > 0 "Aggregate demand is zero; Palma is undefined."

    top_10_idx, bottom_40_idx = compute_palma_indices(n)

    # Cost-weighted aggregate shed/served per load: Υ^agg_j = Σ_t λ_t·(·)_{t,j}.
    # This IS the quantity the Palma sort operates on — structurally identical to
    # the bilevel upper level (`palma_ratio_minimization[_formal_cc]`): ONE sort
    # over the per-load cost-weighted aggregate, NOT a per-period sort. pshed_agg
    # also doubles as the x-axis total-shed diagnostic.
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
        JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, upper_bound = 1, base_name = "palma_a") :
        JuMP.@variable(model, [1:n, 1:n], Bin, base_name = "palma_a")
    u = JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, base_name = "palma_u")

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

    # Aggregate Palma numerator/denominator → ONE ratio: top10(Υ^agg)/bot40(Υ^agg).
    top_sum = JuMP.@expression(model, sum(sorted[i] for i in top_10_idx))
    bot_sum = JuMP.@expression(model, sum(sorted[i] for i in bottom_40_idx))

    # Weak Charnes-Cooper: σ free, σ·bot_sum=1 (NonConvex=2).
    σ = JuMP.@variable(model, base_name = "palma_sigma", lower_bound = 1e-8)
    JuMP.@constraint(model, σ * bot_sum == 1.0)

    # UNCOSTED efficiency: total fraction of horizon demand shed (NO λ — period
    # costs live only in the fairness aggregate, so α=1 is exactly the bilevel
    # fairness objective).
    eff_total = JuMP.@expression(model,
        sum(sum(sum(_PMD.var(pm, nw, :pshed, d)) for d in _PMD.ids(pm, nw, :load))
            for nw in nw_ids_int) / total_demand_all)

    return (
        n = n, T = T, load_ids = load_ids, nw_ids_int = nw_ids_int,
        P_period = P_period, P_agg = P_agg, total_demand_all = total_demand_all,
        top_10_idx = top_10_idx, bottom_40_idx = bottom_40_idx,
        pshed_agg = pshed_agg, pserved_agg = pserved_agg,
        Υ_agg = Υ_agg,
        a = a, u = u,
        top_sum = top_sum, bot_sum = bot_sum, σ = σ,
        eff_total = eff_total,
    )
end

"""
Set the multinetwork objective — fairness term matched EXACTLY to the bilevel
upper level:

    min  α · (σ · top_sum)  +  (1 − α) · eff_total

The fairness part `σ · top_sum` is the cost-weighted aggregate Palma ratio
`top_sum/bot_sum`, with `top_sum = top10%(Υ^agg)` and `bot_sum = bot40%(Υ^agg)`
over the single per-load cost-weighted aggregate `Υ^agg_j = Σ_t λ_t·pshed_{t,j}`
set up in the machinery — the SAME pure-ratio fairness the bilevel upper level
minimizes (one sort of the cost-weighted aggregate). At the CC optimum
`σ · top_sum = top_sum / bot_sum`. The efficiency term is the UNCOSTED
total-shed fraction (no λ — period costs live only in the top/bot sums), so
α=1 recovers exactly the bilevel fairness objective and (1−α) trades it against
plain efficiency. This is what makes single-level and bilevel share one
fairness objective.
"""
function set_palma_alpha_objective_cw!(pm, palma::NamedTuple; alpha::Float64)
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1]"
    fairness_part = alpha * (palma.σ * palma.top_sum)
    eff_part      = (1.0 - alpha) * palma.eff_total
    JuMP.@objective(pm.model, Min, fairness_part + eff_part)
end

# ============================================================
# INSTANTIATE MULTINETWORK MODEL + ATTACH PALMA MACHINERY
# ============================================================
# Build the per-period MLD constraint set; the objective is overwritten with
# the per-α matched (cost-weighted aggregate shed-Palma) + efficiency one.
# `relaxed` selects whether the MLD switch/block vars are continuous
# (build_mn_mc_mld_min_max → _build_mn_period_fair!(…; relax=true)) or binary
# (build_mn_mc_mld_min_max_integer). The Palma permutation stays binary either way.
build_fn = relaxed ?
    (pm -> FairLoadDelivery.build_mn_mc_mld_min_max(pm; alpha = 1.0)) :
    (pm -> FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm; alpha = 1.0))
mld_mn = _PMD.instantiate_mc_model(mn_data, _PMD.LinDist3FlowPowerModel, build_fn;
    multinetwork  = true,
    ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])

# Weak CC: ONE σ·bot_sum=1 (constraint) and σ·top_sum (objective) are bilinear.
# Gurobi NonConvex=2 spatially branches. A single aggregate σ (over the
# cost-weighted aggregate shed top/bot) is far more tractable than a per-period
# σ_t form — and feasible for integer shedding (only needs the bottom-40% loads
# to shed something across the horizon). Expect possible TIME_LIMIT returns at
# non-zero gap; the script accepts feasible incumbents.
JuMP.set_optimizer(mld_mn.model, Gurobi.Optimizer)
JuMP.set_optimizer_attribute(mld_mn.model, "NonConvex",    2)
JuMP.set_optimizer_attribute(mld_mn.model, "MIPGap",       1e-2)        # 1% — control, not tight
JuMP.set_optimizer_attribute(mld_mn.model, "TimeLimit",    180)         # 3 min per α (gaps don't close past this; bounds the NonConvex-MIQCP overrun)
JuMP.set_optimizer_attribute(mld_mn.model, "MIPFocus",     1)
JuMP.set_optimizer_attribute(mld_mn.model, "NumericFocus", 2)

palma = add_palma_machinery_cw_aggregate!(mld_mn;
    nw_ids_int = nw_ids_int_sorted, λ = PEAK_TIME_COSTS, relax_binary = false,
    sort_target = PALMA_SORT)

# ============================================================
# WARMSTART: pure-efficiency multi-period MLD (no Palma machinery)
# ============================================================
# At low α the σ·bot_sum=1 + binary-permutation MIP can leave Gurobi
# without a feasible incumbent in TimeLimit (root LP relaxation is loose
# for the McCormick u and the 24-period MLD stack). We pre-solve the same
# constraint set with a pure-efficiency objective — no σ, no permutation —
# then push its (pshed, switch, block, a, u, σ) values onto `mld_mn` as
# JuMP start values so Gurobi has an incumbent at root for every α.
println("Warmstart: solving pure-efficiency multi-period MLD (no Palma machinery)…")
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
# Use broadcast so this works whether PMD returns a scalar VariableRef (single-phase)
# or a JuMP container (per-phase vector / DenseAxisArray) for each variable family.
for nw in nw_ids_int_sorted
    for lid in palma.load_ids
        eff_var   = _PMD.var(mld_eff, nw, :pshed, lid)
        palma_var = _PMD.var(mld_mn,  nw, :pshed, lid)
        JuMP.set_start_value.(palma_var, JuMP.value.(eff_var))
    end
    eff_v   = _PMD.var(mld_eff, nw)
    palma_v = _PMD.var(mld_mn,  nw)
    for sym in (:switch_state, :z_block, :z_demand)
        haskey(eff_v, sym) || continue
        JuMP.set_start_value.(palma_v[sym], JuMP.value.(eff_v[sym]))
    end
end

# Palma-specific starts: ONE permutation a/u from the cost-weighted AGGREGATE
# Υ^agg (the same single vector the objective sorts), plus ONE σ from its bot40.
n_bot_palma = palma.bottom_40_idx[end]   # = floor(0.4n) (bot40 count)
# Build the warm cost-weighted aggregate Υ^agg_j = Σ_t λ_t·(·)_{t,j} from the
# pure-efficiency solution (mutate in place so this works at top-level scope).
shed_warm_agg = zeros(palma.n)   # Σ_t λ_t·pshed warm per load
pd_warm_agg   = zeros(palma.n)   # Σ_t λ_t·pd   warm per load
for (ti, nw) in enumerate(nw_ids_int_sorted)
    shed_t = [sum(JuMP.value.(_PMD.var(mld_eff, nw, :pshed, lid))) for lid in palma.load_ids]
    pd_t   = [sum(_PMD.ref(mld_eff, nw, :load, lid)["pd"])          for lid in palma.load_ids]
    shed_warm_agg .+= PEAK_TIME_COSTS[ti] .* shed_t
    pd_warm_agg   .+= PEAK_TIME_COSTS[ti] .* pd_t
end
# Warm-sort the SAME aggregate the objective sorts (shed by default).
Υ_warm  = PALMA_SORT === :pshed ? shed_warm_agg : (pd_warm_agg .- shed_warm_agg)
perm    = sortperm(Υ_warm)            # ascending positions
a_start = zeros(palma.n, palma.n)
for (i, j) in enumerate(perm); a_start[i, j] = 1.0; end
u_start = a_start .* reshape(Υ_warm, 1, :)
for i in 1:palma.n, j in 1:palma.n
    JuMP.set_start_value(palma.a[i, j], a_start[i, j])
    JuMP.set_start_value(palma.u[i, j], u_start[i, j])
end
bot_sum_warm = sum(Υ_warm[perm[k]] for k in 1:n_bot_palma)
σ_start = 1.0 / max(bot_sum_warm, 1e-8)
JuMP.set_start_value(palma.σ, σ_start)
println("  warm cost-wtd agg total=$(round(sum(shed_warm_agg), digits=2))   σ_start=$(round(σ_start, digits=6))")

# ============================================================
# ALPHA SWEEP
# ============================================================
# NaN-init so a TimeLimit-with-no-incumbent α leaves explicit NaN in the
# CSV and plots, rather than masquerading as 0 shed.
total_shed       = fill(NaN, alpha_points, N_PERIODS)     # per-period totals (for plots)
max_shed         = fill(NaN, alpha_points, N_PERIODS)
# PRIMARY fairness metric — matches the bilevel upper level:
#   palma_cost_weighted_log[α] = top10(Υ^agg) / bot40(Υ^agg),  Υ^agg_j = Σ_t λ_t·pshed_{t,j}
# (cost-weighted aggregate shed-Palma — the quantity σ·top_sum minimizes).
palma_cost_weighted_log = fill(NaN, alpha_points)
# SECONDARY (diagnostic only): the UNCOSTED horizon-aggregate served-Palma
# (λ_t = 1). Kept for comparison; NOT the optimized metric.
palma_ratio_log  = fill(NaN, alpha_points)
# PLOTTED fairness metric for the standalone figures: UNWEIGHTED aggregate
# shed-Palma = top10(Σ_t pshed) / bot40(Σ_t pshed). This is the SAME neutral
# post-hoc metric the comparison plotter uses (post_hoc_fairness_pareto.jl), so
# the standalone trade-off Pareto and the bilevel-vs-trade-off Pareto share one
# (raw-kW, unweighted) x-axis and Palma definition. palma_cost_weighted_log
# above stays the saved objective value, but is no longer the plotted y.
palma_unweighted_log = fill(NaN, alpha_points)
per_load_dist_a0 = fill(NaN, n_loads, N_PERIODS)
per_load_dist_a1 = fill(NaN, n_loads, N_PERIODS)
# Per-α, per-load aggregate shed (sum across periods) — used for Figure 2 norms.
per_load_agg     = fill(NaN, alpha_points, n_loads)
# Full (α × load × period) tensor — needed by the trade-off heatmap /
# grouped-bar replot scripts. Matches min_max_trade_off_mn.jl's JLD2 schema.
per_load_period_shed = fill(NaN, alpha_points, n_loads, N_PERIODS)
# Raw per-period solution dict per α — Dict("nw_id" => solution_nw) keyed by string.
# `nothing` at non-feasible α points.
solutions_per_alpha = Vector{Any}(nothing, alpha_points)

for (idx, alpha) in enumerate(alphas)
    set_palma_alpha_objective_cw!(mld_mn, palma; alpha = alpha)

    JuMP.optimize!(mld_mn.model)
    status = JuMP.termination_status(mld_mn.model)
    flush(stdout)
    println("alpha=$alpha  status=$status")
    flush(stdout)
    if JuMP.primal_status(mld_mn.model) != MOI.FEASIBLE_POINT
        @warn "non-feasible at alpha=$alpha — skipping; total_shed/max_shed/palma left as NaN"
        continue   # NaN-init means CSV + plots will show this α as missing, not as 0 shed.
    end

    # Per-period totals/distributions (for the per-period plots + the JLD2 tensor).
    for (t, nw) in enumerate(nw_ids_int_sorted)
        per_load_shed_t = [sum(JuMP.value.(_PMD.var(mld_mn, nw, :pshed, lid)))
                           for lid in palma.load_ids]
        total_shed[idx, t] = sum(per_load_shed_t)
        max_shed[idx, t]   = maximum(per_load_shed_t)
        per_load_period_shed[idx, :, t] .= per_load_shed_t
        if idx == 1
            per_load_dist_a0[:, t] .= per_load_shed_t
        elseif idx == alpha_points
            per_load_dist_a1[:, t] .= per_load_shed_t
        end
    end

    # PRIMARY (matched) fairness: cost-weighted AGGREGATE shed-Palma —
    # top10(Υ^agg)/bot40(Υ^agg) with Υ^agg_j = Σ_t λ_t·pshed_{t,j}. This is the
    # IDENTICAL functional the bilevel optimizes (one sort of the cost-weighted
    # aggregate); cross-checked against the model's σ·top_sum below.
    agg_sorted = sort([JuMP.value(palma.Υ_agg[k]) for k in 1:palma.n])
    cw_top = sum(agg_sorted[i] for i in palma.top_10_idx)
    cw_bot = sum(agg_sorted[i] for i in palma.bottom_40_idx)
    palma_cost_weighted_log[idx] = cw_bot < 1e-6 ? Inf : cw_top / cw_bot

    # Cost-weighted aggregate shed/served (Σ_t λ_t··) — kept for the diagnostic
    # println + saved objective metric. NOTE: palma.pshed_agg is COST-WEIGHTED.
    pshed_agg_vals   = [JuMP.value(palma.pshed_agg[k])   for k in 1:palma.n]
    pserved_agg_vals = [JuMP.value(palma.pserved_agg[k]) for k in 1:palma.n]
    palma_ratio_log[idx] = palma_ratio_value(pserved_agg_vals)
    # UNWEIGHTED per-load aggregate shed (Σ_t pshed, NO λ) — drives ALL standalone
    # figures (x-axis total shed, Fig-2 norms, Fig-1 bars) and the plotted Palma,
    # matching the post-hoc comparison plots.
    pshed_agg_unw = [sum(per_load_period_shed[idx, j, t] for t in 1:N_PERIODS)
                     for j in 1:n_loads]
    per_load_agg[idx, :] .= pshed_agg_unw
    palma_unweighted_log[idx] = palma_ratio_value(pshed_agg_unw)

    model_fair = JuMP.value(palma.σ) * JuMP.value(palma.top_sum)
    flush(stdout)
    println("  agg total_shed = $(round(sum(pshed_agg_vals), digits=3))   ",
            "cost-wtd Palma Σλ·top10/Σλ·bot40 (post-hoc) = $(round(palma_cost_weighted_log[idx], digits=4))   ",
            "model(σ·top) = $(round(model_fair, digits=4))   ",
            "uncosted-agg-served-Palma (diag) = $(round(palma_ratio_log[idx], digits=4))")
    flush(stdout)

    # Capture full per-period solution dict (loads/blocks/switches/buses/branches
    # with their PMD-standard fields) for JLD2 persistence. build_solution reads
    # JuMP values off the already-optimized mld_mn.model — uses the sol_component_value
    # hooks wired into the variable_mc_* constructors (see src/core/variable.jl).
    try
        solutions_per_alpha[idx] = FairLoadDelivery._IM.build_solution(mld_mn)
    catch err
        @warn "build_solution failed at alpha=$alpha — JLD2 will have nothing for this α ($err)"
    end
end

# ============================================================
# 3D PARETO PLOT (rep periods only — total shed vs max shed along z = period)
# Same structure as min_max_trade_off_mn.jl, just driven by the Palma objective.
# ============================================================
period_markers = [:circle, :diamond, :utriangle, :rect, :star5, :pentagon, :hexagon]
p3d = plot3d(xlabel = "total load shed (kW)",
             ylabel = "max load shed (kW)",
             zlabel = "period",
             title  = "Multi-period Pareto (Palma, $kind) — rep. periods",
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
             plot_title = "Per-period Pareto ($(pshed_type), Palma) — color = alpha")
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
# Norms are computed on the per-load aggregate-shed vector (sum across
# periods); NaN-α rows (TimeLimit-with-no-incumbent) propagate as NaN.
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
# FIGURE 1: per-load aggregate shed distribution at α=0 and α=1, plus the
# Pareto of unweighted aggregate shed-Palma vs aggregate total shed.
# ============================================================
ref_nw0 = mn_data["nw"][nw_ids_sorted[1]]
load_labels = [ref_nw0["load"][lid]["name"]
               for lid in sort(collect(keys(ref_nw0["load"])), by = x -> parse(Int, x))]

# FONT_KW now mirrors post_hoc_fairness_pareto.jl's _PARETO_FONT so the
# single-panel summary figures share their typographic scale with the
# paper-ready Pareto plots.
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

# Share the y-axis between α=0 and α=1 bars so the visual comparison reflects
# absolute magnitudes; pick the larger of the two so neither chart is clipped.
ymax_shared = max(
    maximum(filter(isfinite, per_load_agg[1, :]);   init = 0.0),
    maximum(filter(isfinite, per_load_agg[end, :]); init = 0.0))

p_dist_a0 = build_dist_plot_agg(per_load_agg[1, :];   ylim_max = ymax_shared)
p_dist_a1 = build_dist_plot_agg(per_load_agg[end, :]; ylim_max = ymax_shared)

# Helper: build a Pareto curve (aggregate total shed x vs a fairness metric y),
# dropping NaN-y entries (TimeLimit-with-no-incumbent or bot40=0 — same guard as
# post_hoc_fairness_pareto.jl). Dots match the steelblue bars at 14pt; α (=ν)
# endpoints annotated 5% of the y-range above the marker.
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

# PRIMARY Pareto: UNWEIGHTED aggregate shed-Palma vs unweighted total shed —
# the same neutral metric/axes as the post-hoc comparison plots. (The
# cost-weighted per-period Palma `palma_cost_weighted_log` is still the solved
# objective and is saved to CSV/JLD2, just not the plotted y here.)
p_pareto = build_pareto_curve(agg_total_shed, palma_unweighted_log,
    "Palma (unitless)")
# SECONDARY (diagnostic): UNCOSTED aggregate served-Palma (λ_t=1). Saved for
# side-by-side comparison; NOT the optimized quantity.
p_pareto_aggdiag = build_pareto_curve(agg_total_shed, palma_ratio_log,
    "Palma (unitless, uncosted diagnostic)")

# Save each panel as its own figure (was a single 3-panel fig1).
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
# per-load aggregate-shed vector). Uses the SAME post-hoc style as Figure 1 /
# post_hoc_palma_pareto_finals.jl (steelblue dots + grey line + ν endpoint
# annotations) — no cividis colorbar.
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
CSV.write(joinpath(output_dir, "palma_sweep_mn_per_period_$(pshed_type).csv"), period_rows)

agg_rows = DataFrame(alpha = alphas,
    agg_total_shed       = [sum(total_shed[i, :]) for i in 1:alpha_points],
    cost_weighted_shed   = weighted_total,
    cost_weighted_max    = weighted_max,
    palma_cost_weighted  = palma_cost_weighted_log,  # PRIMARY: top10(Υ^agg)/bot40(Υ^agg), Υ^agg=Σ_t λ_t·pshed (matched to bilevel)
    palma_ratio_uncosted = palma_ratio_log)          # SECONDARY: uncosted aggregate served-Palma (diagnostic)
CSV.write(joinpath(output_dir, "palma_sweep_mn_aggregate_$(pshed_type).csv"), agg_rows)

# ============================================================
# PERSIST SWEEP DATA FOR STANDALONE VISUALIZATION SCRIPTS
# Same schema as min_max_trade_off_mn.jl's trade_off_mn_*.jld2 so
# trade_off_heatmap_mn.jl and trade_off_grouped_mn.jl can target either
# fair_func with one loader. Saved as palma_sweep_mn_<case>_<pshed_type>.jld2.
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

jld_path = joinpath(output_dir, "palma_sweep_mn_$(case)_$(pshed_type).jld2")
JLD2.jldsave(jld_path;
    alphas               = alphas,
    per_load_period_shed = per_load_period_shed,  # alpha × load × period
    per_load_period_pd   = per_load_period_pd,    # load × period
    per_load_agg         = per_load_agg,          # alpha × load (sum over periods)
    total_shed           = total_shed,            # alpha × period
    max_shed             = max_shed,              # alpha × period
    # PRIMARY fairness metric — matches the bilevel upper level:
    #   palma_cost_weighted_log[α] = top10(Υ^agg) / bot40(Υ^agg),  Υ^agg_j = Σ_t λ_t·pshed_{t,j}
    palma_cost_weighted_log = palma_cost_weighted_log,   # alpha
    # SECONDARY (diagnostic): UNCOSTED aggregate served-Palma (λ_t=1).
    palma_ratio_log      = palma_ratio_log,
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
    fair_func            = "palma",
    # Per-α raw per-period solution dicts (loads/blocks/switches/buses/branches).
    # Built via _IM.build_solution after each α's JuMP.optimize!; nothing where
    # the α point was infeasible. See run_validation_mn.jl Step 6 for the
    # downstream schema.
    solutions_per_alpha  = solutions_per_alpha,
    math_switch          = math["switch"],
    math_branch          = math["branch"],
    math_bus             = math["bus"],
    math_load            = math["load"],
    load_block_sets      = lbs,
)
println("Saved trade-off sweep data → $jld_path")

println("Done. Results written to: ", output_dir)
