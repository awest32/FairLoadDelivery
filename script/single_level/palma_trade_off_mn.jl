#=
Multi-period, single-level served-Palma vs efficiency trade-off
================================================================

Multi-period analogue of `palma_trade_off.jl`. Builds a multinetwork MLD
problem (per-period constraints from `build_mn_mc_mld_min_max[_integer]`,
which we instantiate purely for the constraint set), then attaches
**cost-weighted, per-period-sorted served-Palma** machinery, aggregated into
ONE ratio, whose fairness objective is matched to the bilevel upper level
(`palma_ratio_minimization` in `load_shed_as_parameter.jl`):

    per period t:  sort pserved_t  →  top10%(pserved_t),  bot40%(pserved_t)
    top_sum    = Σ_t λ_t · top10%(pserved_t)        ← cost-weighted Palma NUMERATOR
    bot_sum    = Σ_t λ_t · bot40%(pserved_t)
    fairness   = Palma = top_sum / bot_sum
    objective  = α · (σ · top_sum)  +  (1 − α) · eff_total

The Palma numerator is the **top-10% of served IN EACH period** (a true decile,
not the sum of all served), cost-weighted by λ_t. Period costs enter only as
multipliers on the per-period top/bot SUMS, so the fairness term is still a pure
ratio σ·top_sum — identical to the bilevel's. `eff_total` is the UNCOSTED
total-shed fraction (no λ), so α=1 recovers exactly the bilevel fairness
objective and (1−α) trades it against plain efficiency. Single-level and bilevel
thus share ONE fairness objective.

Charnes-Cooper "weak" form, ONE σ over the cost-weighted AGGREGATE top/bot:

    σ free, lower-bounded
    σ · bot_sum = 1                              ← bilinear constraint
    σ · top_sum in the objective                 ← bilinear

both via Gurobi `NonConvex=2`. Per-period SORTING (deciles within each period)
but a SINGLE σ over the aggregated cost-weighted top/bot is what makes this both
tractable (one bilinear σ, not T) AND feasible for integer shedding: σ·bot_sum=1
only needs `Σ_t λ_t·bot40_t > 0` (SOME period has a nonzero bottom-40%), whereas
the per-σ_t form was infeasible because whole-block integer shedding zeroes a
period's bottom-40% (see `script/diagnostics/probe_per_period_palma_feasibility.jl`).

The Palma *permutations* `a[t]` stay BINARY even when the MLD is LP-relaxed
(`relaxed=true`): relaxing them collapses the McCormick `u` and breaks the sort.

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
# per-period served distribution is non-degenerate — the right "relaxed"
# analogue for the served-Palma sort. The Palma *permutation* `a[t]` stays
# BINARY regardless (relaxing it collapses the McCormick `u` to zero and
# breaks the sort — see legacy/palma_reformulation/README.md). When false the
# MLD switch/block vars are binary (`build_mn_mc_mld_min_max_integer`).
relaxed = get(ENV, "RELAXED", "true") == "true"   # env-overridable so one orchestration can run integer + relaxed
# Which quantity the Palma objective sorts: "served" (matches the income-Palma
# formulation; default) or "shed" (experiment — optimize fairness of the shed burden).
PALMA_SORT = get(ENV, "PALMA_TARGET", "served") == "shed" ? :pshed : :pd
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

# Palma sweep: kept smaller than min-max because each solve is a 24-period
# bilinear MIP (per-period σ_t · bot_sum_t = 1 + bilinear objective).
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
# Shed-objective experiment writes to a separate folder so it doesn't overwrite
# the served-objective (default) results.
obj_suffix = PALMA_SORT === :pshed ? "_shedobj" : ""
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
Attach COST-WEIGHTED PER-PERIOD-SORTED served-Palma machinery to a multinetwork
JuMP model, aggregated into ONE Palma ratio. This is the single-level control
whose **fairness objective matches the bilevel upper level**:

    per period t:  sort pserved_t   →   top10%(pserved_t),  bot40%(pserved_t)
    top_sum = Σ_t λ_t · top10%(pserved_t)        ← cost-weighted Palma NUMERATOR
    bot_sum = Σ_t λ_t · bot40%(pserved_t)         ← cost-weighted Palma DENOMINATOR
    fairness = Palma = top_sum / bot_sum,   one σ,   σ·bot_sum = 1

The Palma numerator is the **top-10% of served IN EACH period** (a true Palma
decile), cost-weighted by λ_t and summed — NOT the sum of all served. Period
costs λ_t enter only as multipliers on the per-period top/bot SUMS (so the
objective is still a pure ratio σ·top_sum). `u[t]` is the McCormick BILINEAR
term `u[t][i,j] = a[t][i,j]·pserved_{t,j}` that linearizes the per-period sort.

ONE σ for the whole horizon (Charnes-Cooper weak form): σ·bot_sum=1 (constraint)
and σ·top_sum (objective) are bilinear → Gurobi `NonConvex=2`. The single σ over
the cost-weighted AGGREGATE top/bot is what makes this both tractable and
feasible for integer shedding: σ·bot_sum=1 only needs `Σ_t λ_t·bot40_t > 0`
(SOME period has a nonzero bottom-40%), not every period — the per-σ_t form was
infeasible because whole-block integer shedding zeroes a period's bottom-40%
(see `script/diagnostics/probe_per_period_palma_feasibility.jl`).

The Palma permutations `a[t]` stay BINARY even when the MLD itself is LP-relaxed
(`relax_binary` governs only the permutations): relaxing them collapses the
McCormick `u` and breaks the sort (legacy/palma_reformulation/README.md).

`λ` must align with `nw_ids_int` (λ[ti] is the cost of period nw_ids_int[ti]).
"""
function add_palma_machinery_cw_aggregate!(pm; nw_ids_int::Vector{Int},
                                           λ::Vector{Float64},
                                           relax_binary::Bool = false,
                                           sort_target::Symbol = :pd)  # :pd = served-Palma (default); :pshed = shed-Palma (experiment)
    model = pm.model

    nw0      = nw_ids_int[1]
    load_ids = sort(collect(_PMD.ids(pm, nw0, :load)))
    n        = length(load_ids)
    T        = length(nw_ids_int)
    @assert length(λ) == T "λ length must match number of periods"

    # Per-period nameplate demand P_{t,i} (= McCormick upper bound on served).
    P_period = Dict{Int,Vector{Float64}}()
    for nw in nw_ids_int
        P_period[nw] = [sum(_PMD.ref(pm, nw, :load, i)["pd"]) for i in load_ids]
    end
    total_demand_all = sum(sum(P_period[nw]) for nw in nw_ids_int)
    @assert total_demand_all > 0 "Aggregate demand is zero; Palma is undefined."

    top_10_idx, bottom_40_idx = compute_palma_indices(n)

    # Uncosted aggregate shed/served per load — for x-axis total shed + diagnostics.
    pshed_agg = JuMP.@expression(model, [k = 1:n],
        sum(sum(_PMD.var(pm, nw, :pshed, load_ids[k])) for nw in nw_ids_int))
    pserved_agg = JuMP.@expression(model, [k = 1:n],
        sum(sum(_PMD.var(pm, nw, :pd, load_ids[k])) for nw in nw_ids_int))

    # Per-period sort machinery; accumulate cost-weighted top/bot contributions.
    a        = Vector{Any}(undef, T)
    u        = Vector{Any}(undef, T)
    pserved_period = Dict{Int,Any}()
    top10_terms = Any[]   # λ_t · top10%(pserved_t)
    bot40_terms = Any[]   # λ_t · bot40%(pserved_t)

    for (ti, nw) in enumerate(nw_ids_int)
        Pt = P_period[nw]
        # sort_target = :pd → served-Palma; :pshed → shed-Palma. Both ∈ [0, Pt].
        pserved_t = JuMP.@expression(model, [k = 1:n],
            sum(_PMD.var(pm, nw, sort_target, load_ids[k])))
        pserved_period[nw] = pserved_t

        a[ti] = relax_binary ?
            JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, upper_bound = 1, base_name = "palma_a_$ti") :
            JuMP.@variable(model, [1:n, 1:n], Bin, base_name = "palma_a_$ti")
        u[ti] = JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, base_name = "palma_u_$ti")

        # McCormick bilinear term: u[t][i,j] = a[t][i,j] · pserved_{t,j}, 0 ≤ pserved ≤ Pt.
        for i in 1:n, j in 1:n
            Pj = Pt[j]
            JuMP.@constraint(model, u[ti][i, j] >= pserved_t[j] + a[ti][i, j] * Pj - Pj)
            JuMP.@constraint(model, u[ti][i, j] <= a[ti][i, j] * Pj)
            JuMP.@constraint(model, u[ti][i, j] <= pserved_t[j])
        end
        for i in 1:n
            JuMP.@constraint(model, sum(a[ti][i, j] for j in 1:n) == 1)
        end
        for j in 1:n
            JuMP.@constraint(model, sum(a[ti][i, j] for i in 1:n) == 1)
        end

        sorted_t = JuMP.@expression(model, [i = 1:n], sum(u[ti][i, j] for j in 1:n))
        for k in 1:(n - 1)
            JuMP.@constraint(model, sorted_t[k] <= sorted_t[k + 1])
        end

        push!(top10_terms, λ[ti] * sum(sorted_t[i] for i in top_10_idx))
        push!(bot40_terms, λ[ti] * sum(sorted_t[i] for i in bottom_40_idx))
    end

    # Cost-weighted aggregate Palma numerator/denominator → ONE ratio.
    top_sum = JuMP.@expression(model, sum(top10_terms))   # Σ_t λ_t · top10%(pserved_t)
    bot_sum = JuMP.@expression(model, sum(bot40_terms))   # Σ_t λ_t · bot40%(pserved_t)

    # Weak Charnes-Cooper: σ free, σ·bot_sum=1 (NonConvex=2).
    σ = JuMP.@variable(model, base_name = "palma_sigma", lower_bound = 1e-8)
    JuMP.@constraint(model, σ * bot_sum == 1.0)

    # UNCOSTED efficiency: total fraction of horizon demand shed (NO λ — period
    # costs live only in the fairness top/bot sums, so α=1 is exactly the bilevel
    # fairness objective).
    eff_total = JuMP.@expression(model,
        sum(sum(sum(_PMD.var(pm, nw, :pshed, d)) for d in _PMD.ids(pm, nw, :load))
            for nw in nw_ids_int) / total_demand_all)

    return (
        n = n, T = T, load_ids = load_ids, nw_ids_int = nw_ids_int,
        P_period = P_period, total_demand_all = total_demand_all,
        top_10_idx = top_10_idx, bottom_40_idx = bottom_40_idx,
        pshed_agg = pshed_agg, pserved_agg = pserved_agg,
        pserved_period = pserved_period,
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
`top_sum/bot_sum`, with `top_sum = Σ_t λ_t·top10%(pserved_t)` and
`bot_sum = Σ_t λ_t·bot40%(pserved_t)` set up in the machinery — the SAME
pure-ratio fairness the bilevel upper level minimizes. At the CC optimum
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
# the per-α matched (per-period cost-weighted served-Palma) + efficiency one.
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
# Gurobi NonConvex=2 spatially branches. A single aggregate σ (over cost-weighted
# served totals) is far more tractable than the per-period form — and feasible
# for integer shedding (only needs each load served sometime). Expect possible
# TIME_LIMIT returns at non-zero gap; the script accepts feasible incumbents.
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

# Palma-specific starts: per-period permutation a[t]/u[t] from each period's
# SORT-TARGET vector (served or shed, matching the objective), and ONE σ from the
# cost-weighted aggregate bot_sum.
n_bot_palma = palma.bottom_40_idx[end]   # = floor(0.4n) (bot40 count)
bot_sum_warm   = 0.0
pshed_warm_tot = 0.0
for (ti, nw) in enumerate(nw_ids_int_sorted)
    global pshed_warm_tot, bot_sum_warm   # accumulators live in the script's global scope
    shed_warm_t = [sum(JuMP.value.(_PMD.var(mld_eff, nw, :pshed, lid))) for lid in palma.load_ids]
    pshed_warm_tot += sum(shed_warm_t)
    # the warm permutation must sort the SAME quantity the objective sorts
    pserved_warm_t = PALMA_SORT === :pshed ? shed_warm_t :
        [sum(_PMD.ref(mld_eff, nw, :load, lid)["pd"]) for lid in palma.load_ids] .- shed_warm_t

    perm    = sortperm(pserved_warm_t)            # ascending positions
    a_start = zeros(palma.n, palma.n)
    for (i, j) in enumerate(perm); a_start[i, j] = 1.0; end
    u_start = a_start .* reshape(pserved_warm_t, 1, :)
    for i in 1:palma.n, j in 1:palma.n
        JuMP.set_start_value(palma.a[ti][i, j], a_start[i, j])
        JuMP.set_start_value(palma.u[ti][i, j], u_start[i, j])
    end
    # contribution to the cost-weighted aggregate denominator: λ_t · bot40(served_t)
    bot_sum_warm += PEAK_TIME_COSTS[ti] * sum(pserved_warm_t[perm[k]] for k in 1:n_bot_palma)
end
σ_start = 1.0 / max(bot_sum_warm, 1e-8)
JuMP.set_start_value(palma.σ, σ_start)
println("  warm pshed_total=$(round(pshed_warm_tot, digits=2))   σ_start=$(round(σ_start, digits=6))")

# ============================================================
# ALPHA SWEEP
# ============================================================
# NaN-init so a TimeLimit-with-no-incumbent α leaves explicit NaN in the
# CSV and plots, rather than masquerading as 0 shed.
total_shed       = fill(NaN, alpha_points, N_PERIODS)     # per-period totals (for plots)
max_shed         = fill(NaN, alpha_points, N_PERIODS)
# PRIMARY fairness metric — matches the bilevel upper level:
#   palma_cost_weighted_log[α] = Σ_t λ_t·top10(srv_t) / Σ_t λ_t·bot40(srv_t)
# (cost-weighted per-period served-Palma — the quantity σ·top_sum minimizes).
palma_cost_weighted_log = fill(NaN, alpha_points)
# SECONDARY (diagnostic only): the UNCOSTED horizon-aggregate served-Palma
# (λ_t = 1). Kept for comparison; NOT the optimized metric.
palma_ratio_log  = fill(NaN, alpha_points)
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

    # PRIMARY (matched) fairness: cost-weighted aggregate Palma over per-period
    # deciles — top_sum/bot_sum with top_sum=Σ_t λ_t·top10%(pserved_t),
    # bot_sum=Σ_t λ_t·bot40%(pserved_t). Computed post-hoc by sorting each
    # period's served values, and cross-checked against the model's σ·top_sum.
    cw_top = 0.0; cw_bot = 0.0
    for (t, nw) in enumerate(nw_ids_int_sorted)
        s = sort([JuMP.value(palma.pserved_period[nw][k]) for k in 1:palma.n])
        cw_top += PEAK_TIME_COSTS[t] * sum(s[i] for i in palma.top_10_idx)
        cw_bot += PEAK_TIME_COSTS[t] * sum(s[i] for i in palma.bottom_40_idx)
    end
    palma_cost_weighted_log[idx] = cw_bot < 1e-6 ? Inf : cw_top / cw_bot

    # Uncosted aggregate shed/served (x-axis total shed + diagnostic Palma).
    pshed_agg_vals   = [JuMP.value(palma.pshed_agg[k])   for k in 1:palma.n]
    pserved_agg_vals = [JuMP.value(palma.pserved_agg[k]) for k in 1:palma.n]
    palma_ratio_log[idx] = palma_ratio_value(pserved_agg_vals)
    per_load_agg[idx, :] .= pshed_agg_vals

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
# FIGURE 1: per-load aggregate shed distribution at α=0 and α=1, plus
# aggregate total shed + served-Palma (twin axis) vs α.
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

# PRIMARY Pareto: matches the bilevel upper level — total shed vs the
# cost-weighted per-period served-Palma: Σ_t λ_t·top10(srv_t) / Σ_t λ_t·bot40(srv_t).
p_pareto = build_pareto_curve(agg_total_shed, palma_cost_weighted_log,
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
    palma_cost_weighted  = palma_cost_weighted_log,  # PRIMARY: Σ_t λ_t·top10(srv_t)/Σ_t λ_t·bot40(srv_t) (matched to bilevel)
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
    #   palma_cost_weighted_log[α] = Σ_t λ_t·top10(srv_t) / Σ_t λ_t·bot40(srv_t)
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
