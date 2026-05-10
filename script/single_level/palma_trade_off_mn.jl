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
using PowerPlots
using DataFrames
using CSV
using Dates
import MathOptInterface as MOI

const _PMD = PowerModelsDistribution
const PMD  = PowerModelsDistribution

include("../../src/implementation/visualization.jl")

# ============================================================
# CONFIGURATION
# ============================================================
case_name = "../../data/pmd_opendss/case6_unbalanced_switch_meshed_good4integer.dss"
dir = @__DIR__
case_path = joinpath(dir, case_name)
date = Dates.format(now(), "yyyy-mm-dd")
LS_PERCENT = 0.8
pshed_type = "absolute"  # only absolute supported in this script

# Multi-period setup mirrors min_max_trade_off_mn.jl so results are directly comparable.
const N_PERIODS = 24
const LOAD_SCALE_FACTORS = [round(s, digits=3) for s in LinRange(0.65, 1.4, N_PERIODS)]
const PEAK_TIME_COSTS    = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                            for h in 0:N_PERIODS-1]
const REP_PERIODS = [6, 11, 20]   # off-peak, mid-day, evening peak

# Palma sweep: kept smaller than min-max because each solve is a 24-period
# bilinear MIP (per-period σ_t · bot_sum_t = 1 + bilinear objective).
alpha_points = 6
alphas = collect(LinRange(0.0, 1.0, alpha_points))

# ============================================================
# NETWORK SETUP + MULTINETWORK DATA
# ============================================================
eng, math, lbs, critical_id = setup_network(case_path, LS_PERCENT;
    switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT)

"Replicate single-period math dict into a multinetwork dict with per-period load scaling."
function create_multinetwork_data(base_math::Dict{String,Any}, n_periods::Int, load_scales::Vector{Float64})
    @assert length(load_scales) == n_periods
    mn_data = Dict{String,Any}(
        "multinetwork" => true,
        "per_unit"     => true,
        "data_model"   => PMD.MATHEMATICAL,
        "nw"           => Dict{String,Any}()
    )
    for key in ["baseMVA", "basekv", "bus_lookup", "settings"]
        haskey(base_math, key) && (mn_data[key] = deepcopy(base_math[key]))
    end
    for t in 1:n_periods
        nw_id = string(t - 1)
        nw_data = deepcopy(base_math)
        delete!(nw_data, "multinetwork")
        scale = load_scales[t]
        for (_, load) in nw_data["load"]
            load["pd"] = load["pd"] .* scale
            load["qd"] = load["qd"] .* scale
        end
        nw_data["time_period"] = t
        nw_data["load_scale"]  = scale
        mn_data["nw"][nw_id] = nw_data
    end
    return mn_data
end

mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)
nw_ids_sorted     = sort(collect(keys(mn_data["nw"])), by = x -> parse(Int, x))
nw_ids_int_sorted = parse.(Int, nw_ids_sorted)        # ints, matches PMD nw_ids
n_loads           = length(mn_data["nw"][nw_ids_sorted[1]]["load"])

output_dir = joinpath(@__DIR__, "../../results/$date/palma_trade_off_mn")
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

    # Aggregate per-load shed / served (sum over periods + phases)
    pshed_agg = JuMP.@expression(model, [k = 1:n],
        sum(sum(_PMD.var(pm, nw, :pshed, load_ids[k])) for nw in nw_ids_int))
    pserved_agg = JuMP.@expression(model, [k = 1:n], P_total[k] - pshed_agg[k])

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
# ALPHA SWEEP
# ============================================================
total_shed       = zeros(alpha_points, N_PERIODS)         # per-period totals (for plots)
max_shed         = zeros(alpha_points, N_PERIODS)
palma_ratio_log  = fill(NaN, alpha_points)                # one aggregate Palma per α
per_load_dist_a0 = zeros(n_loads, N_PERIODS)
per_load_dist_a1 = zeros(n_loads, N_PERIODS)

for (idx, alpha) in enumerate(alphas)
    set_palma_alpha_objective_agg!(mld_mn, palma, nw_ids_int_sorted,
                                   PEAK_TIME_COSTS; alpha = alpha)

    JuMP.optimize!(mld_mn.model)
    status = JuMP.termination_status(mld_mn.model)
    flush(stdout)
    println("alpha=$alpha  status=$status")
    flush(stdout)
    if JuMP.primal_status(mld_mn.model) != MOI.FEASIBLE_POINT
        @warn "non-feasible at alpha=$alpha — skipping"
        continue
    end

    # Per-period totals/distributions (still useful for visualizing how each
    # period absorbs the day's shed)
    for (t, nw) in enumerate(nw_ids_int_sorted)
        per_load_shed_t = [sum(JuMP.value.(_PMD.var(mld_mn, nw, :pshed, lid)))
                           for lid in palma.load_ids]
        total_shed[idx, t] = sum(per_load_shed_t)
        max_shed[idx, t]   = maximum(per_load_shed_t)
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
    pserved_agg_vals = palma.P_total .- pshed_agg_vals
    palma_ratio_log[idx] = palma_ratio_value(pserved_agg_vals)

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
            label  = "t=$t (λ=$(PEAK_TIME_COSTS[t]), s=$(LOAD_SCALE_FACTORS[t]))",
            marker = period_markers[mod1(k, length(period_markers))], lw = 2,
            line_z = alphas)
end
savefig(p3d, joinpath(output_dir, "pareto3d_integer_$(pshed_type).svg"))
display(p3d)

# ============================================================
# PER-PERIOD 2D PARETO PANEL (total shed vs max shed, color = alpha)
# ============================================================
panel_cols = N_PERIODS <= 6 ? N_PERIODS : 6
panel_rows = ceil(Int, N_PERIODS / panel_cols)
panel = plot(layout = (panel_rows, panel_cols),
             size = (220 * panel_cols, 180 * panel_rows),
             plot_title = "Per-period Pareto ($(pshed_type), Palma) — color = alpha",
             plot_titlefontsize = 11)
for t in 1:N_PERIODS
    row = ceil(Int, t / panel_cols)
    col = ((t - 1) % panel_cols) + 1
    plot!(panel[t], total_shed[:, t], max_shed[:, t],
          marker = :circle, lc = :grey, marker_z = alphas, color = :cividis,
          xlabel = row == panel_rows ? "total shed (kW)" : "",
          ylabel = col == 1            ? "max shed (kW)"   : "",
          title  = "t=$t  s=$(LOAD_SCALE_FACTORS[t])  λ=$(PEAK_TIME_COSTS[t])",
          colorbar = false, legend = false,
          titlefontsize = 8, guidefontsize = 7, tickfontsize = 6)
end
savefig(panel, joinpath(output_dir, "pareto_per_period_integer_$(pshed_type).svg"))
display(panel)

# ============================================================
# COST-WEIGHTED METRICS VS ALPHA (+ aggregate Palma on a twin axis)
# ============================================================
weighted_total = [sum(PEAK_TIME_COSTS[t] * total_shed[i, t] for t in 1:N_PERIODS) for i in 1:alpha_points]
weighted_max   = [sum(PEAK_TIME_COSTS[t] * max_shed[i, t]   for t in 1:N_PERIODS) for i in 1:alpha_points]

p_metrics = plot(alphas, weighted_total, label = "Σ_t λ_t · total shed_t",
    lw = 2, marker = :circle, xlabel = "alpha", ylabel = "kW (cost-weighted)")
plot!(p_metrics, alphas, weighted_max, label = "Σ_t λ_t · max shed_t",
    lw = 2, marker = :square)
# Aggregate Palma on a twin y-axis. Plot only finite entries to keep the axis
# readable when the α=1 corner returns Palma = Inf.
finite_idx = findall(isfinite, palma_ratio_log)
if !isempty(finite_idx)
    plot!(twinx(p_metrics), alphas[finite_idx], palma_ratio_log[finite_idx],
        label = "aggregate Palma", lw = 2, marker = :diamond, ls = :dash,
        color = :purple, ylabel = "aggregate Palma (top10/bot40, served-day)",
        legend = :topleft)
end
savefig(p_metrics, joinpath(output_dir, "metrics_vs_alpha_integer_$(pshed_type).svg"))
display(p_metrics)

# ============================================================
# SUMMARY (mirrors min_max_trade_off_mn.jl's 2×2 layout)
# ============================================================
ref_nw0 = mn_data["nw"][nw_ids_sorted[1]]
load_labels = [ref_nw0["load"][lid]["name"]
               for lid in sort(collect(keys(ref_nw0["load"])), by = x -> parse(Int, x))]
rep_period_labels = reshape(["t=$t" for t in REP_PERIODS], 1, length(REP_PERIODS))

function build_dist_plot_mn(per_load_per_period::Matrix{Float64}, title_str::String)
    groupedbar(load_labels, per_load_per_period[:, REP_PERIODS],
        bar_position = :dodge,
        labels    = rep_period_labels,
        xlabel    = "load",
        ylabel    = "load shed (kW)",
        title     = title_str,
        legend    = :topright,
        linecolor = :black)
end

p_dist_a0 = build_dist_plot_mn(per_load_dist_a0, "alpha = 0 (efficiency) — rep. periods")
p_dist_a1 = build_dist_plot_mn(per_load_dist_a1, "alpha = 1 (Palma) — rep. periods")

# Combined Pareto overlay (rep periods, total shed vs max shed) — same as min_max
p_pareto_combined = plot(xlabel = "total shed (kW)", ylabel = "max shed (kW)",
                         title = "Pareto by period (rep.)", legend = :topright)
for (k, t) in enumerate(REP_PERIODS)
    plot!(p_pareto_combined, total_shed[:, t], max_shed[:, t],
          marker = period_markers[mod1(k, length(period_markers))],
          label = "t=$t (s=$(LOAD_SCALE_FACTORS[t]), λ=$(PEAK_TIME_COSTS[t]))",
          line_z = alphas, color = :cividis)
end

combined = plot(p_dist_a0, p_dist_a1, p_metrics, p_pareto_combined,
    layout = (2, 2), size = (1400, 900),
    left_margin = 10Plots.mm, right_margin = 5Plots.mm,
    top_margin  = 5Plots.mm,  bottom_margin = 10Plots.mm)
savefig(combined, joinpath(output_dir, "summary_integer_all_$(pshed_type).svg"))
display(combined)

# ============================================================
# CSV: per-period shed + one aggregate Palma per α
# ============================================================
period_rows = DataFrame(alpha = Float64[], period = Int[], lambda = Float64[],
    load_scale = Float64[], total_shed = Float64[], max_shed = Float64[])
for (i, a) in enumerate(alphas), t in 1:N_PERIODS
    push!(period_rows, (a, t, PEAK_TIME_COSTS[t], LOAD_SCALE_FACTORS[t],
                        total_shed[i, t], max_shed[i, t]))
end
CSV.write(joinpath(output_dir, "palma_sweep_mn_per_period_$(pshed_type).csv"), period_rows)

agg_rows = DataFrame(alpha = alphas,
    agg_total_shed     = [sum(total_shed[i, :]) for i in 1:alpha_points],
    cost_weighted_shed = weighted_total,
    cost_weighted_max  = weighted_max,
    palma_ratio        = palma_ratio_log)
CSV.write(joinpath(output_dir, "palma_sweep_mn_aggregate_$(pshed_type).csv"), agg_rows)

println("Done. Results written to: ", output_dir)
