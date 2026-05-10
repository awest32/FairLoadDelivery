#=
Single-period, single-level Palma-ratio vs efficiency trade-off
================================================================

Mirrors `min_max_trade_off.jl` but replaces the min-max fairness objective
with the Palma-ratio fairness objective (top 10% / bottom 40% of sorted
per-load shed). Builds on the same MLD network model, sweeps an alpha
trade-off knob, and records / plots the Pareto front.

This script is a sandbox: the Palma machinery is defined locally and not
pushed into `src/`. The `load_shed_as_parameter.jl` source file is the
formulation reference and is intentionally NOT modified here.

Only the absolute-pshed variant is implemented for now (proportional left
for a follow-up).
=#

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS, Juniper
using HSL_jll
using Plots
using Random
using Distributions
using DiffOpt
using JuMP
using LinearAlgebra, SparseArrays
using PowerPlots
using DataFrames
using CSV
using Plots
using Dates
import MathOptInterface as MOI

const _PMD = PowerModelsDistribution

include("../../src/implementation/visualization.jl")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
case_name = "../../data/pmd_opendss/case6_unbalanced_switch_meshed_good4integer.dss"
dir = @__DIR__
case_path = joinpath(dir, case_name)
date = Dates.format(now(), "yyyy-mm-dd")
LS_PERCENT = 0.8
pshed_type = "absolute"  # only absolute supported in this script

eng, math, lbs, critical_id = setup_network(case_path, LS_PERCENT;
    switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT)

# Build with the min-max MLD problem so the constraint set matches
# `min_max_trade_off.jl`; we'll overwrite the objective with the Palma+α one.
mld_model_int = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_min_max_integer;
    ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])
ref = mld_model_int.ref[:it][:pmd][:nw][0]

alpha_points = 10
loadshed = zeros(alpha_points, length(ref[:load]) + 2)
palma_ratio_log = fill(NaN, alpha_points)

output_dir = joinpath(@__DIR__, "../../results/$date/palma_trade_off")
isdir(output_dir) || mkpath(output_dir)

# ----------------------------------------------------------------------------
# Palma helpers (local — formulation mirrors load_shed_as_parameter.jl)
# ----------------------------------------------------------------------------

"""
    compute_palma_indices(n) -> (top_10_idx, bottom_40_idx)

Indices in ASCENDING-sorted order. Top 10% = ceil(0.1n) largest (last
positions), bottom 40% = floor(0.4n) smallest (first positions).
"""
function compute_palma_indices(n::Int)
    n_bot = max(1, floor(Int, 0.4 * n))
    n_top = max(1, ceil(Int, 0.1 * n))
    bottom_40_idx = collect(1:n_bot)
    top_10_idx = collect((n - n_top + 1):n)
    return top_10_idx, bottom_40_idx
end

"""
    palma_ratio_value(vals) -> Float64

Compute the Palma ratio of a numeric vector (top 10% / bottom 40% of
sorted values). Caller decides whether to pass shed or served.
Returns Inf if denominator is below `eps_denom`.
"""
function palma_ratio_value(vals::AbstractVector; eps_denom::Float64 = 1e-6)
    n = length(vals)
    s = sort(collect(vals))
    top_10_idx, bottom_40_idx = compute_palma_indices(n)
    num = sum(s[i] for i in top_10_idx)
    den = sum(s[i] for i in bottom_40_idx)
    return den < eps_denom ? Inf : num / den
end

"""
    add_palma_machinery!(pm; relax_binary=false)

Attach Palma sorting + Charnes-Cooper auxiliary variables and constraints
to a single-period MLD JuMP model (nw = 0). Returns a NamedTuple of handles
the caller uses to set the objective.

Palma is computed over **load served** (not shed). This avoids the
denominator-inflation perverse incentive of shed-Palma (where shedding
*more* from already-shed loads can lower the ratio). Served-Palma rewards
keeping the worst-served loads as close to fully served as possible.

Formulation:

    pserved_scalar[k] = pd[k] − Σ_phase pshed[load_ids[k]]   ∈ [0, pd[k]]
    a[i,j] ∈ {0,1}                   permutation matrix
    Σ_j a[i,j] = 1, Σ_i a[i,j] = 1   doubly stochastic
    u[i,j] = a[i,j] * pserved_scalar[j]  (McCormick)
    sorted[i] = Σ_j u[i,j], ascending
    σ * Σ_{bot40} sorted[i] = 1      Charnes-Cooper
    Palma = σ * Σ_{top10} sorted[i]  = top10_served / bot40_served

Efficiency term still tracks total shed (Σ pshed / total_demand), so the
α trade-off is "served-Palma vs shed-efficiency".
"""
function add_palma_machinery!(pm; relax_binary::Bool = false, nw::Int = 0)
    model = pm.model
    pshed = _PMD.var(pm, nw, :pshed)
    load_ids = sort(collect(_PMD.ids(pm, nw, :load)))
    n = length(load_ids)

    pd_per_load = [sum(_PMD.ref(pm, nw, :load, i)["pd"]) for i in load_ids]
    total_demand = sum(pd_per_load)
    @assert total_demand > 0 "Network has zero total demand; Palma is undefined."

    # Per-load served = pd − shed   (range [0, pd[k]])
    pserved_scalar = JuMP.@expression(model, [k = 1:n],
        pd_per_load[k] - sum(pshed[load_ids[k]]))
    # Per-load shed (kept around so callers can still report the shed CSV)
    pshed_scalar = JuMP.@expression(model, [k = 1:n], sum(pshed[load_ids[k]]))

    # Permutation matrix
    a = if relax_binary
        JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, upper_bound = 1, base_name = "palma_a")
    else
        JuMP.@variable(model, [1:n, 1:n], Bin, base_name = "palma_a")
    end
    u = JuMP.@variable(model, [1:n, 1:n], lower_bound = 0, base_name = "palma_u")

    # McCormick envelopes for u[i,j] = a[i,j] * pserved_scalar[j],
    # with bounds 0 ≤ pserved_scalar[j] ≤ pd[j].
    for i in 1:n, j in 1:n
        Pj = pd_per_load[j]
        JuMP.@constraint(model, u[i, j] >= pserved_scalar[j] + a[i, j] * Pj - Pj)
        JuMP.@constraint(model, u[i, j] <= a[i, j] * Pj)
        JuMP.@constraint(model, u[i, j] <= pserved_scalar[j])
    end

    # Doubly stochastic
    for i in 1:n
        JuMP.@constraint(model, sum(a[i, j] for j in 1:n) == 1)
    end
    for j in 1:n
        JuMP.@constraint(model, sum(a[i, j] for i in 1:n) == 1)
    end

    # Sorted served, ascending (so bot40 = 3 LEAST served, top10 = 1 MOST served)
    sorted_t = JuMP.@expression(model, [i = 1:n], sum(u[i, j] for j in 1:n))
    for k in 1:(n - 1)
        JuMP.@constraint(model, sorted_t[k] <= sorted_t[k + 1])
    end

    top_10_idx, bottom_40_idx = compute_palma_indices(n)
    top_sum = JuMP.@expression(model, sum(sorted_t[i] for i in top_10_idx))
    bot_sum = JuMP.@expression(model, sum(sorted_t[i] for i in bottom_40_idx))

    # Charnes-Cooper: σ * bot_sum = 1
    σ = JuMP.@variable(model, base_name = "palma_sigma", lower_bound = 1e-8)
    JuMP.@constraint(model, σ * bot_sum == 1.0)

    # Efficiency term = total shed / total demand (minimize → less shed)
    eff_term = JuMP.@expression(model, sum(pshed_scalar[k] for k in 1:n) / total_demand)

    return (
        n = n,
        load_ids = load_ids,
        pd_per_load = pd_per_load,
        total_demand = total_demand,
        pshed_scalar = pshed_scalar,
        pserved_scalar = pserved_scalar,
        a = a,
        u = u,
        sorted_t = sorted_t,
        top_sum = top_sum,
        bot_sum = bot_sum,
        σ = σ,
        eff_term = eff_term,
    )
end

"""
    set_palma_alpha_objective!(pm, palma; alpha=1.0)

Set the JuMP objective to a pure convex combination of the Palma ratio
(`σ * top_sum`, via Charnes-Cooper) and the efficiency term
(total shed / total demand):

    min α · (σ · top_sum) + (1 − α) · eff_term

Note: the σ · bot_sum = 1 constraint is always active, so even at α = 0
the solution must have a strictly positive bottom-40% sum. For a clean
pure-efficiency solve at α = 0, use the standalone min_max model.
"""
function set_palma_alpha_objective!(pm, palma; alpha::Float64 = 1.0)
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1]"
    fairness_part = alpha * (palma.σ * palma.top_sum)
    eff_part = (1.0 - alpha) * palma.eff_term
    JuMP.@objective(pm.model, Min, fairness_part + eff_part)
end

# ----------------------------------------------------------------------------
# Build Palma machinery and configure Gurobi for nonconvex QP
# ----------------------------------------------------------------------------

JuMP.set_optimizer(mld_model_int.model, Gurobi.Optimizer)
JuMP.set_optimizer_attribute(mld_model_int.model, "NonConvex", 2)
JuMP.set_optimizer_attribute(mld_model_int.model, "MIPGap", 1e-4)
JuMP.set_optimizer_attribute(mld_model_int.model, "TimeLimit", 60 * 20)
JuMP.set_optimizer_attribute(mld_model_int.model, "MIPFocus", 1)
JuMP.set_optimizer_attribute(mld_model_int.model, "NumericFocus", 2)

palma_int = add_palma_machinery!(mld_model_int; relax_binary = false)

# ----------------------------------------------------------------------------
# Alpha sweep (integer)
# ----------------------------------------------------------------------------

loadshed_keys = []

for (index, alpha) in enumerate(LinRange(0, 1, alpha_points))
    set_palma_alpha_objective!(mld_model_int, palma_int; alpha = alpha)

    JuMP.optimize!(mld_model_int.model)
    status = JuMP.termination_status(mld_model_int.model)
    println("alpha=$alpha  status=$status")
    if JuMP.primal_status(mld_model_int.model) != MOI.FEASIBLE_POINT
        println("Termination status is $status — stopping integer sweep at alpha=$alpha")
        break
    end

    loads = mld_model_int.sol[:it][:pmd][:nw][0][:load]
    push!(loadshed_keys, keys(loads))
    for (load_id, load_data) in loads
        loadshed[index, load_id] = sum(value.(load_data[:pshed]))
    end
    loadshed[index, length(loads) + 1] = sum(loadshed[index, 1:length(loads)])
    loadshed[index, end] = alpha

    # Palma is computed over LOAD SERVED (= pd − shed), in load_ids order.
    shed_in_order   = [loadshed[index, palma_int.load_ids[k]] for k in 1:palma_int.n]
    served_in_order = palma_int.pd_per_load .- shed_in_order
    palma_ratio_log[index] = palma_ratio_value(served_in_order)

    # Sanity check: model's σ * top_sum should equal palma_ratio_log[index]
    σ_val   = JuMP.value(palma_int.σ)
    top_val = JuMP.value(palma_int.top_sum)
    bot_val = JuMP.value(palma_int.bot_sum)
    model_palma = σ_val * top_val
    println("  total_shed = $(round(loadshed[index, length(loads) + 1], digits=3))   ",
            "palma_served = $(round(palma_ratio_log[index], digits=4))   ",
            "model(σ·top) = $(round(model_palma, digits=4))   ",
            "[σ=$(round(σ_val, digits=4)), top_sum=$(round(top_val, digits=3)), bot_sum=$(round(bot_val, digits=3))]")
end

# ----------------------------------------------------------------------------
# Plot Pareto curve + Palma vs alpha
# ----------------------------------------------------------------------------

n = length(ref[:load])
total_shed = loadshed[:, n + 1]
max_shed   = [maximum(loadshed[i, 1:n]) for i in 1:alpha_points]
alphas     = loadshed[:, end]

# Pareto curve: total shed (efficiency) vs max load shed (fairness proxy)
p3 = plot(total_shed, max_shed, label = "solution (kW)",
    seriestype = :line,
    lc = :grey,
    marker = :circle,
    marker_z = alphas, colorbar_title = "alpha", color = :cividis,
    ylabel = "max load shed (kW)",
    xlabel = "total load shed (kW)",
    title  = "Pareto: integer Palma vs efficiency",
    legend = true)

# Metrics vs alpha (total + max shed)
p4 = plot(alphas, total_shed, label = "total shed (kW)", lw = 2, marker = :circle,
    xlabel = "alpha", ylabel = "load shed (kW)")
plot!(p4, alphas, max_shed, label = "max load shed (kW)", lw = 2, marker = :square)

# Palma (over served) vs alpha
p5 = plot(alphas, palma_ratio_log, label = "Palma (served)", lw = 2, marker = :diamond,
    xlabel = "alpha", ylabel = "Palma ratio  (top10% served / bot40% served)",
    title  = "Served-Palma vs alpha")

savefig(plot(p3, p4, p5, layout = (1, 3), size = (1500, 400)),
    joinpath(output_dir, "pareto_summary_integer_$(pshed_type).svg"))

# ----------------------------------------------------------------------------
# Distribution plots at alpha = 0 (efficiency end) and alpha = 1 (Palma end)
# ----------------------------------------------------------------------------

load_labels = [load_data["name"] for (id, load_data) in sort(ref[:load])]

function build_dist_plot(pshed_per_load, title_str)
    p = bar(load_labels, pshed_per_load,
        xlabel = "load ID",
        ylabel = "load shed (kW)",
        title  = title_str,
        legend = false,
        color  = :steelblue,
        linecolor = :black,
    )
    ymax = maximum(pshed_per_load)
    for (i, v) in enumerate(pshed_per_load)
        annotate!(p, i, v + (ymax > 0 ? ymax : 1.0) * 0.02,
            text("$(round(v, digits = 1))", 8, :center))
    end
    return p
end

p_dist_a0 = build_dist_plot(loadshed[1, 1:n],   "alpha = 0 (efficiency end)")
p_dist_a1 = build_dist_plot(loadshed[end, 1:n], "alpha = 1 (Palma end)")

savefig(p_dist_a0, joinpath(output_dir, "loadshed_distribution_integer_alpha0.svg"))
savefig(p_dist_a1, joinpath(output_dir, "loadshed_distribution_integer_alpha1.svg"))

combined = plot(p_dist_a0, p_dist_a1, p4, p3, layout = (2, 2), size = (1400, 900),
    left_margin = 10Plots.mm, right_margin = 5Plots.mm,
    top_margin = 5Plots.mm, bottom_margin = 10Plots.mm)
savefig(combined, joinpath(output_dir, "summary_integer_all_$(pshed_type).svg"))
display(combined)

# ----------------------------------------------------------------------------
# Save raw sweep results to CSV for downstream inspection
# ----------------------------------------------------------------------------

sweep_df = DataFrame(
    alpha       = alphas,
    total_shed  = total_shed,
    max_shed    = max_shed,
    palma_ratio = palma_ratio_log,
)
for k in 1:n
    sweep_df[!, Symbol("load_$(k)_shed")] = loadshed[:, k]
end
CSV.write(joinpath(output_dir, "palma_sweep_integer_$(pshed_type).csv"), sweep_df)

println("Done. Results written to: ", output_dir)
