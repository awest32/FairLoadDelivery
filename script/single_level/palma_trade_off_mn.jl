#=
Multi-period, single-level served-Palma vs efficiency trade-off
================================================================

Multi-period analogue of `palma_trade_off.jl`. Builds a multinetwork MLD
problem (per-period constraints from `build_mn_mc_mld_min_max_integer`,
which we instantiate purely for the constraint set), then attaches a
**single aggregate served-Palma** machinery — Palma is computed across
all periods on per-load aggregate served:

    served_i = Σ_t ( pd_{t,i} − Σ_phase pshed_{t,i,phase} )    ∈ [0, P_i]
    P_i      = Σ_t pd_{t,i}

This is the **single-level control** for the bilevel Palma comparison.
Uses the same "weak" Charnes-Cooper form as `load_shed_as_parameter.jl`
and the single-period `palma_trade_off.jl`:

    σ free, lower-bounded
    σ · bot_sum = 1                              ← bilinear constraint
    objective:  α · (σ · top_sum) + (1 − α) · Σ_t λ_t · eff_term_t

with σ·top_sum and σ·bot_sum=1 enforced by Gurobi `NonConvex=2` spatial
branching.

Why "control": the bilevel Palma upper level (`load_shed_as_parameter.jl`)
uses this same Palma machinery on top of a Jacobian-derived linear
expression for pshed — pshed is a *parameter* there, not a decision
variable, so σ × pshed bilinearity sits over a small, tractable
expression. In the single-level version pshed is live (subject to the
full multi-period MLD), so σ × top_sum couples through every period's
constraints. The result is a hard MIP: expect `TIME_LIMIT` returns at
non-zero gap on most α points. The script accepts feasible incumbents
and writes whatever Pareto curve Gurobi can produce in `TimeLimit`.

If you want the script to converge tightly, that's exactly the
motivation for going bilevel.

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
case_name = "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss"
#case_name = "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss"
case = "more_meshed_6_bus"#"13_bus" #"more_meshed_6_bus"   # 13-bus motivation_c run (T=3, [4,8,18]); flip back to "more_meshed_6_bus" + 6-bus dss for case6 runs.

dir = @__DIR__
case_path = joinpath(dir, case_name)
date = Dates.format(now(), "yyyy-mm-dd")
LS_PERCENT = 0.8
pshed_type = "absolute"  # only absolute supported in this script
relaxed = false
# Multi-period setup mirrors min_max_trade_off_mn.jl so results are directly comparable.
# Per-load profiles follow Hamilton & Aliprantis (PECI 2023): each load name is
# deterministically mapped to (schedule, ±1h shift). The per-period demand level
# is implicit in the reported load-shed values, so no aggregate-scale label is
# carried in plots or CSVs.
# Downsampled hours-of-day (0-indexed); mirrors min_max_trade_off_mn.jl so
# results stay comparable across fair-funcs.
#SELECTED_HOURS = [4, 18, 8]   # 13-bus motivation_c: T=3, peak in middle position so plots show off-peak → peak → off-peak. λ=[5.0, 30.0, 5.01]. Was collect(0:23) for case6 T=24.
SELECTED_HOURS    = collect(0:23)   # T=24 full diurnal cycle (was [4,6,8,12,15,18,20,22] for T=8)

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
REP_PERIODS = [6, 11, 20]   # T=3 with [4, 8, 18] — plot all periods. Was [6, 11, 20] for T=24.

# Palma sweep: kept smaller than min-max because each solve is a 24-period
# bilinear MIP (per-period σ_t · bot_sum_t = 1 + bilinear objective).
alpha_points = 20
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
output_dir = joinpath(@__DIR__, "../../results/$date/palma$(rel)_trade_off_mn")
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
Attach AGGREGATE served-Palma machinery to a multinetwork JuMP model.
Palma is computed once over loads, where each load's served value is the
sum across all periods:

    served_i = Σ_t ( pd_{t,i} − Σ_phase pshed_{t,i,phase} )    ∈ [0, P_i]
    P_i      = Σ_t pd_{t,i}

Charnes-Cooper "weak" form, matching `palma_trade_off.jl` and
`load_shed_as_parameter.jl`: σ enters as a free auxiliary variable and
the bilinearities `σ·bot_sum=1` (constraint) and `σ·top_sum` (objective)
are handled by Gurobi `NonConvex=2` spatial branching. This is the
intentionally "less-tractable" single-level control — it exists to be
compared against the bilevel formulation, where the Jacobian severs the
σ × pshed coupling and the upper-level Palma becomes tractable.

The single-level multi-period MIP is HARD: all MLD constraints are
live, σ multiplies a sort-derived expression that depends on every
period's pshed, and Gurobi spatial branching has to subdivide the σ
domain inside the full B&B tree. Expect TIME_LIMIT returns at non-zero
gap — the script accepts feasible incumbents.
"""
function add_palma_machinery_aggregate!(pm; nw_ids_int::Vector{Int},
                                         relax_binary::Bool = false)
    model = pm.model

    nw0      = nw_ids_int[1]
    load_ids = sort(collect(_PMD.ids(pm, nw0, :load)))
    n        = length(load_ids)

    P_total = [sum(sum(_PMD.ref(pm, nw, :load, i)["pd"]) for nw in nw_ids_int) for i in load_ids]
    total_demand_all = sum(P_total)
    @assert total_demand_all > 0 "Aggregate demand is zero; Palma is undefined."

    # Aggregate per-load shed / served (sum over periods + phases).
    # pserved_agg reads from the PMD-native :pd dispatch variable directly
    # (equivalent to P_total[k] − pshed_agg[k] via constraint_load_shed_definition,
    # but reads as "served" without the subtraction). pshed_agg is kept for
    # post-hoc reporting in the sweep loop below.
    pshed_agg = JuMP.@expression(model, [k = 1:n],
        sum(sum(_PMD.var(pm, nw, :pshed, load_ids[k])) for nw in nw_ids_int))
    pserved_agg = JuMP.@expression(model, [k = 1:n],
        sum(sum(_PMD.var(pm, nw, :pd, load_ids[k])) for nw in nw_ids_int))

    # Permutation matrix
    a = if relax_binary
        JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, upper_bound = 1, base_name = "palma_a")
    else
        JuMP.@variable(model, [1:n, 1:n], Bin, base_name = "palma_a")
    end
    u = JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, base_name = "palma_u")

    # McCormick: u[i,j] = a[i,j] * pserved_agg[j], with 0 ≤ pserved_agg[j] ≤ P_total[j]
    for i in 1:n, j in 1:n
        Pj = P_total[j]
        JuMP.@constraint(model, u[i, j] >= pserved_agg[j] + a[i, j] * Pj - Pj)
        JuMP.@constraint(model, u[i, j] <= a[i, j] * Pj)
        JuMP.@constraint(model, u[i, j] <= pserved_agg[j])
    end

    for i in 1:n
        JuMP.@constraint(model, sum(a[i, j] for j in 1:n) == 1)
    end
    for j in 1:n
        JuMP.@constraint(model, sum(a[i, j] for i in 1:n) == 1)
    end

    sorted_v = JuMP.@expression(model, [i = 1:n], sum(u[i, j] for j in 1:n))
    for k in 1:(n - 1)
        JuMP.@constraint(model, sorted_v[k] <= sorted_v[k + 1])
    end

    top_10_idx, bottom_40_idx = compute_palma_indices(n)
    top_sum = JuMP.@expression(model, sum(sorted_v[i] for i in top_10_idx))
    bot_sum = JuMP.@expression(model, sum(sorted_v[i] for i in bottom_40_idx))

    # Weak Charnes-Cooper: σ free, σ·bot_sum=1 enforced by NonConvex=2.
    σ = JuMP.@variable(model, base_name = "palma_sigma", lower_bound = 1e-8)
    JuMP.@constraint(model, σ * bot_sum == 1.0)

    # Per-period efficiency expressions (in x-space)
    eff_per_nw = Dict{Int,Any}()
    for nw in nw_ids_int
        td_nw = sum(sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
        eff_per_nw[nw] = JuMP.@expression(model,
            sum(sum(_PMD.var(pm, nw, :pshed, d)) for d in _PMD.ids(pm, nw, :load)) / td_nw)
    end

    return (
        n = n, load_ids = load_ids,
        P_total = P_total, total_demand_all = total_demand_all,
        pshed_agg = pshed_agg, pserved_agg = pserved_agg,
        a = a, u = u, sorted_v = sorted_v,
        top_sum = top_sum, bot_sum = bot_sum, σ = σ,
        eff_per_nw = eff_per_nw,
    )
end

"""
Set the multinetwork objective:

    min  α · (σ · top_sum) + (1 − α) · Σ_t λ_t · eff_term_t

Both σ·top_sum and the σ·bot_sum=1 constraint are bilinear; Gurobi
NonConvex=2 handles them. At the optimum (CC normalization active),
σ·top_sum = top_sum/bot_sum = the Palma ratio of x-space served values.
"""
function set_palma_alpha_objective_agg!(pm, palma::NamedTuple,
                                        nw_ids_int::Vector{Int},
                                        λ::Vector{Float64}; alpha::Float64)
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1]"
    @assert length(λ) == length(nw_ids_int) "λ length must match number of periods"
    fairness_part = alpha * (palma.σ * palma.top_sum)
    eff_part = (1.0 - alpha) * sum(λ[idx] * palma.eff_per_nw[nw]
                                   for (idx, nw) in enumerate(nw_ids_int))
    JuMP.@objective(pm.model, Min, fairness_part + eff_part)
end

# ============================================================
# INSTANTIATE MULTINETWORK MODEL + ATTACH PALMA MACHINERY
# ============================================================
# Use the existing multi-period min-max integer build for the constraint set;
# we'll overwrite its objective with the per-α Palma+efficiency one.
build_fn = pm -> FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm; alpha = 1.0)
mld_mn = _PMD.instantiate_mc_model(mn_data, _PMD.LinDist3FlowPowerModel, build_fn;
    multinetwork  = true,
    ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])

# Weak CC: σ·bot_sum=1 (constraint) and σ·top_sum (objective) are bilinear.
# Gurobi NonConvex=2 spatially branches. Expect TIME_LIMIT returns at non-zero gap;
# script accepts feasible incumbents.
JuMP.set_optimizer(mld_mn.model, Gurobi.Optimizer)
JuMP.set_optimizer_attribute(mld_mn.model, "NonConvex",    2)
JuMP.set_optimizer_attribute(mld_mn.model, "MIPGap",       1e-2)        # 1% — control, not tight
JuMP.set_optimizer_attribute(mld_mn.model, "TimeLimit",    60 * 5)      # 5 min per α
JuMP.set_optimizer_attribute(mld_mn.model, "MIPFocus",     1)
JuMP.set_optimizer_attribute(mld_mn.model, "NumericFocus", 2)

palma = add_palma_machinery_aggregate!(mld_mn;
    nw_ids_int = nw_ids_int_sorted, relax_binary = false)

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
build_fn_eff = pm -> FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm;
    peak_time_costs = PEAK_TIME_COSTS, alpha = 0.0)
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

# Palma-specific (a, u, σ) from per-load aggregate served.
pshed_warm_agg = [sum(sum(JuMP.value.(_PMD.var(mld_eff, nw, :pshed, lid)))
                      for nw in nw_ids_int_sorted)
                  for lid in palma.load_ids]
pserved_warm = palma.P_total .- pshed_warm_agg
perm         = sortperm(pserved_warm)              # ascending positions
a_start      = zeros(palma.n, palma.n)
for (i, j) in enumerate(perm); a_start[i, j] = 1.0; end
u_start      = a_start .* reshape(pserved_warm, 1, :)
n_bot_palma  = max(1, floor(Int, 0.4 * palma.n))
bot_sum_val  = sum(pserved_warm[perm[k]] for k in 1:n_bot_palma)
σ_start      = 1.0 / max(bot_sum_val, 1e-8)
for i in 1:palma.n, j in 1:palma.n
    JuMP.set_start_value(palma.a[i, j], a_start[i, j])
    JuMP.set_start_value(palma.u[i, j], u_start[i, j])
end
JuMP.set_start_value(palma.σ, σ_start)
println("  warm pshed_total=$(round(sum(pshed_warm_agg), digits=2))   σ_start=$(round(σ_start, digits=6))")

# ============================================================
# ALPHA SWEEP
# ============================================================
# NaN-init so a TimeLimit-with-no-incumbent α leaves explicit NaN in the
# CSV and plots, rather than masquerading as 0 shed.
total_shed       = fill(NaN, alpha_points, N_PERIODS)     # per-period totals (for plots)
max_shed         = fill(NaN, alpha_points, N_PERIODS)
palma_ratio_log  = fill(NaN, alpha_points)                # one aggregate Palma per α
per_load_dist_a0 = fill(NaN, n_loads, N_PERIODS)
per_load_dist_a1 = fill(NaN, n_loads, N_PERIODS)
# Per-α, per-load aggregate shed (sum across periods) — used for Figure 2 norms.
per_load_agg     = fill(NaN, alpha_points, n_loads)
# Full (α × load × period) tensor — needed by the trade-off heatmap /
# grouped-bar replot scripts. Matches min_max_trade_off_mn.jl's JLD2 schema.
per_load_period_shed = fill(NaN, alpha_points, n_loads, N_PERIODS)

for (idx, alpha) in enumerate(alphas)
    set_palma_alpha_objective_agg!(mld_mn, palma, nw_ids_int_sorted,
                                   PEAK_TIME_COSTS; alpha = alpha)

    JuMP.optimize!(mld_mn.model)
    status = JuMP.termination_status(mld_mn.model)
    flush(stdout)
    println("alpha=$alpha  status=$status")
    flush(stdout)
    if JuMP.primal_status(mld_mn.model) != MOI.FEASIBLE_POINT
        @warn "non-feasible at alpha=$alpha — skipping; total_shed/max_shed/palma left as NaN"
        continue   # NaN-init means CSV + plots will show this α as missing, not as 0 shed.
    end

    # Per-period totals/distributions (still useful for visualizing how each
    # period absorbs the day's shed)
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

    # Aggregate post-hoc served-Palma (computed from x-space pserved) +
    # model sanity check (y-space y_top_sum should equal the same Palma value
    # because y_bot_sum = 1 forces y = (top_x/bot_x) at equality).
    pshed_agg_vals   = [JuMP.value(palma.pshed_agg[k])   for k in 1:palma.n]
    pserved_agg_vals = [JuMP.value(palma.pserved_agg[k]) for k in 1:palma.n]
    palma_ratio_log[idx] = palma_ratio_value(pserved_agg_vals)
    per_load_agg[idx, :] .= pshed_agg_vals

    σ_val   = JuMP.value(palma.σ)
    top_val = JuMP.value(palma.top_sum)
    bot_val = JuMP.value(palma.bot_sum)
    flush(stdout)
    println("  agg total_shed = $(round(sum(pshed_agg_vals), digits=3))   ",
            "palma_served (x-space post-hoc) = $(round(palma_ratio_log[idx], digits=4))   ",
            "model(σ·top) = $(round(σ_val * top_val, digits=4))   ",
            "[σ=$(round(σ_val, digits=6)), top_sum=$(round(top_val, digits=3)), bot_sum=$(round(bot_val, digits=3))]")
    flush(stdout)
end

# ============================================================
# 3D PARETO PLOT (rep periods only — total shed vs max shed along z = period)
# Same structure as min_max_trade_off_mn.jl, just driven by the Palma objective.
# ============================================================
period_markers = [:circle, :diamond, :utriangle, :rect, :star5, :pentagon, :hexagon]
p3d = plot3d(xlabel = "total load shed (kW)",
             ylabel = "max load shed (kW)",
             zlabel = "period",
             title  = "Multi-period Pareto (Palma, integer) — rep. periods",
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

# Pareto curve: aggregate total shed (x) vs Palma ratio of shed (y).
# Dots match the steelblue of the bar charts and are enlarged to 14pt
# (3.5× the default); α encoding lives on the top axis below. Drops NaN
# Palma entries (low-α points where bot40 = 0 — same guard as
# post_hoc_fairness_pareto.jl).
finite_palma = findall(isfinite, palma_ratio_log)
p_pareto = plot(agg_total_shed[finite_palma], palma_ratio_log[finite_palma],
    seriestype = :line, lc = :grey,
    marker = :circle, markersize = 14, color = :steelblue,
    markerstrokecolor = :steelblue,
    xlabel = "total load shed (kW)",
    ylabel = "Palma ratio of shed (unitless)",
    legend = false;
    FONT_KW...)
# Annotate the α=0 / α=1 endpoints (5% of the y-range above the marker)
# using the filtered finite-Palma α values. Same style as the bar-chart
# value labels.
let _ts_e = agg_total_shed[finite_palma],
    _ys_e = palma_ratio_log[finite_palma],
    _αs_e = alphas[finite_palma]
    yrange = maximum(_ys_e) - minimum(_ys_e)
    yoff = 0.05 * (yrange == 0 ? 1.0 : yrange)
    annotate!(p_pareto, _ts_e[1],   _ys_e[1]   + yoff,
        text("ν=$(round(_αs_e[1],   digits=2))", ANNOT_PT, :center))
    annotate!(p_pareto, _ts_e[end], _ys_e[end] + yoff,
        text("ν=$(round(_αs_e[end], digits=2))", ANNOT_PT, :center))
end

# Save each panel as its own figure (was a single 3-panel fig1).
for (_name, _p) in (("alpha0", p_dist_a0), ("alpha1", p_dist_a1), ("pareto", p_pareto))
    _fig = plot(_p; size = (900, 760),
        left_margin = 7Plots.mm, right_margin = 6Plots.mm,
        top_margin = 8Plots.mm, bottom_margin = 7Plots.mm)
    savefig(_fig, joinpath(output_dir,
        "summary_single_$(_name)_$(kind)_$(pshed_type).svg"))
    _name == "pareto" && display(_fig)
end

# ============================================================
# FIGURE 2: Pareto fronts (aggregate total shed vs L1 / L2 / L∞ / CoV of the
# per-load aggregate-shed vector), α encoded by marker color. Colorbar lives
# in a dedicated narrow subplot so the four data panels stay equally sized.
# ============================================================
function pareto_norm_plot(total_shed_vec, norm_vec, alphas_vec, ylab)
    plot(total_shed_vec, norm_vec,
        seriestype = :line, lc = :grey,
        marker = :circle, marker_z = alphas_vec, color = :cividis,
        clims = (0.0, 1.0), colorbar = false,
        xlabel = "total load shed (kW)", ylabel = ylab,
        legend = false; FONT_KW...)
end

p_l1   = pareto_norm_plot(agg_total_shed, l1_vec,   alphas, "L1 norm of shed (kW)")
p_l2   = pareto_norm_plot(agg_total_shed, l2_vec,   alphas, "L2 norm of shed (kW)")
p_linf = pareto_norm_plot(agg_total_shed, linf_vec, alphas, "L∞ norm of shed (kW)")
p_cov  = pareto_norm_plot(agg_total_shed, cov_vec,  alphas, "CoV (stdev/mean)")

p_cbar = heatmap(reshape(collect(LinRange(0.0, 1.0, 256)), :, 1);
    color = :cividis, colorbar = false,
    xticks = false, yticks = ([1, 128, 256], ["0", "0.5", "1"]),
    ylabel = "ν", title = "", framestyle = :box)

fig2 = plot(p_l1, p_l2, p_linf, p_cov, p_cbar,
    layout = @layout([a b c d e{0.02w}]),
    size = (2200, 600),
    left_margin = 14Plots.mm, right_margin = 6Plots.mm,
    top_margin = 8Plots.mm, bottom_margin = 14Plots.mm)
savefig(fig2, joinpath(output_dir, "pareto_norms_$(kind)_$(pshed_type).svg"))
display(fig2)

# ============================================================
# CSV: per-period shed + one aggregate Palma per α
# ============================================================
period_rows = DataFrame(alpha = Float64[], period = Int[], lambda = Float64[],
    total_shed = Float64[], max_shed = Float64[])
for (i, a) in enumerate(alphas), t in 1:N_PERIODS
    push!(period_rows, (a, t, PEAK_TIME_COSTS[t],
                        total_shed[i, t], max_shed[i, t]))
end
CSV.write(joinpath(output_dir, "palma_sweep_mn_per_period_$(pshed_type).csv"), period_rows)

agg_rows = DataFrame(alpha = alphas,
    agg_total_shed     = [sum(total_shed[i, :]) for i in 1:alpha_points],
    cost_weighted_shed = weighted_total,
    cost_weighted_max  = weighted_max,
    palma_ratio        = palma_ratio_log)
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
)
println("Saved trade-off sweep data → $jld_path")

println("Done. Results written to: ", output_dir)
