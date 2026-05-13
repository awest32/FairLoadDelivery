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

const PMD = PowerModelsDistribution

include("../../src/implementation/visualization.jl")

# ============================================================
# CONFIGURATION
# ============================================================
#case_name = "../../data/pmd_opendss/case6_unbalanced_switch_meshed_good4integer.dss"
case_name = "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss"
case = "13_bus"
dir = @__DIR__
case_path = joinpath(dir, case_name)
date = Dates.format(now(), "yyyy-mm-dd")
LS_PERCENT = 0.8

# Multi-period setup: 24 hourly periods with linear-ramp load + TOU peak-cost profiles
const N_PERIODS = 24
# Linear ramp from 0.7 (period 1) to 1.0 (period 24): every period is a distinct
# load level, monotonically increasing across the day.
const LOAD_SCALE_FACTORS = [round(s, digits=3) for s in LinRange(0.75, 1.1, N_PERIODS)]
# TOU pricing: low overnight, peak in evening (h≈18)
const PEAK_TIME_COSTS    = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                            for h in 0:N_PERIODS-1]

# Representative subset (1-indexed period indices) for the busy plots:
# overnight off-peak (h=3), morning ramp (h=8), evening peak (h=19)
 REP_PERIODS = [6, 11, 20]

pshed_type = "absolute"  # "absolute" or "proportional"
solve_min_max = pshed_type == "proportional" ?
    FairLoadDelivery.solve_mn_mc_mld_min_max_proportional_integer :
    FairLoadDelivery.solve_mn_mc_mld_min_max_integer

# ============================================================
# NETWORK SETUP
# ============================================================
eng, math, lbs, critical_id = setup_network(case_path, LS_PERCENT;
    switch_rating = [Inf,Inf,Inf])#sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT)

"""
Replicate a single-period math dict into a multinetwork dict with per-period
load scaling. Same structure as legacy/brute_force/multi_period_trade_off_comparison.jl.
"""
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
        nw_data["load_scale"] = scale
        mn_data["nw"][nw_id] = nw_data
    end
    return mn_data
end

mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)
nw_ids_sorted = sort(collect(keys(mn_data["nw"])), by=x->parse(Int, x))
n_loads = length(mn_data["nw"][nw_ids_sorted[1]]["load"])

output_dir = joinpath(@__DIR__, "../../results/$date/trade_off_mn")
isdir(output_dir) || mkpath(output_dir)

# ============================================================
# ALPHA SWEEP
# ============================================================
alpha_points = 20
alphas = collect(LinRange(0, 1, alpha_points))

# Per-(alpha, period) totals for the 3D Pareto
total_shed = zeros(alpha_points, N_PERIODS)
max_shed   = zeros(alpha_points, N_PERIODS)
# Per-load × period distribution captured at alpha=0 and alpha=1 for the summary
per_load_dist_a0 = zeros(n_loads, N_PERIODS)
per_load_dist_a1 = zeros(n_loads, N_PERIODS)

for (idx, alpha) in enumerate(alphas)
    soln = solve_min_max(mn_data, Gurobi.Optimizer;
        peak_time_costs=PEAK_TIME_COSTS, alpha=alpha)
    status = soln["termination_status"]
    println("alpha=$alpha  status=$status")
    if !(status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.ALMOST_LOCALLY_SOLVED)
        @warn "non-optimal at alpha=$alpha"
        continue
    end
    for (t, nw_id) in enumerate(nw_ids_sorted)
        loads_t = soln["solution"]["nw"][nw_id]["load"]
        sorted_load_ids = sort(collect(keys(loads_t)), by=x->parse(Int, x))
        per_load_shed = [sum(loads_t[lid]["pshed"]) for lid in sorted_load_ids]
        total_shed[idx, t] = sum(per_load_shed)
        max_shed[idx, t]   = maximum(per_load_shed)
        if idx == 1
            per_load_dist_a0[:, t] .= per_load_shed
        elseif idx == alpha_points
            per_load_dist_a1[:, t] .= per_load_shed
        end
    end
end

# ============================================================
# 3D PARETO PLOT (one curve per period along z = period index)
# ============================================================
period_markers = [:circle, :diamond, :utriangle, :rect, :star5, :pentagon, :hexagon]
p3d = plot3d(xlabel = "total load shed (kW)",
             ylabel = "max load shed (kW)",
             zlabel = "period",
             title  = "Multi-period Pareto ($(pshed_type), integer) — rep. periods",
             legend = :topright)
for (k, t) in enumerate(REP_PERIODS)
    plot3d!(p3d, total_shed[:, t], max_shed[:, t], fill(t, alpha_points),
            label = "t=$t (λ=$(PEAK_TIME_COSTS[t]), scale=$(LOAD_SCALE_FACTORS[t]))",
            marker = period_markers[mod1(k, length(period_markers))], lw = 2,
            line_z = alphas)
end
savefig(p3d, joinpath(output_dir, "pareto3d_integer_$(pshed_type)_$case.svg"))
display(p3d)

# Per-period 2D Pareto panel (grid layout — readable for many periods)
panel_cols = N_PERIODS <= 6 ? N_PERIODS : 6
panel_rows = ceil(Int, N_PERIODS / panel_cols)
panel = plot(layout = (panel_rows, panel_cols),
             size = (220 * panel_cols, 180 * panel_rows),
             plot_title = "Per-period Pareto ($(pshed_type), integer) — color = alpha",
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
savefig(panel, joinpath(output_dir, "pareto_per_period_integer_$(pshed_type)_$case.svg"))
display(panel)

# Global metrics across alpha (cost-weighted aggregates)
weighted_total = [sum(PEAK_TIME_COSTS[t] * total_shed[i, t] for t in 1:N_PERIODS) for i in 1:alpha_points]
weighted_max   = [sum(PEAK_TIME_COSTS[t] * max_shed[i, t]   for t in 1:N_PERIODS) for i in 1:alpha_points]
p_metrics = plot(alphas, weighted_total, label = "Σ_t λ_t · total shed_t", lw = 2, marker = :circle,
                 xlabel = "alpha", ylabel = "kW (cost-weighted)")
plot!(p_metrics, alphas, weighted_max, label = "Σ_t λ_t · max shed_t", lw = 2, marker = :square)
savefig(p_metrics, joinpath(output_dir, "metrics_vs_alpha_integer_$(pshed_type)_$case.svg"))
display(p_metrics)

# ============================================================
# SUMMARY PLOT (mirrors single-period summary_integer_all_*.svg)
# ============================================================
ref_nw0 = mn_data["nw"][nw_ids_sorted[1]]
load_labels = [ref_nw0["load"][lid]["name"]
               for lid in sort(collect(keys(ref_nw0["load"])), by=x->parse(Int, x))]
# Distributions and overlay Pareto use only the representative subset for readability
rep_period_labels = reshape(["t=$t" for t in REP_PERIODS], 1, length(REP_PERIODS))

function build_dist_plot_mn(per_load_per_period::Matrix{Float64}, title_str::String)
    groupedbar(load_labels, per_load_per_period[:, REP_PERIODS],
        bar_position = :dodge,
        labels = rep_period_labels,
        xlabel = "load",
        ylabel = "load shed (kW)",
        title  = title_str,
        legend = :topright,
        linecolor = :black)
end

p_dist_a0 = build_dist_plot_mn(per_load_dist_a0, "alpha = 0 (efficiency) — rep. periods")
p_dist_a1 = build_dist_plot_mn(per_load_dist_a1, "alpha = 1 (fairness) — rep. periods")

# Combined Pareto overlay limited to representative periods (full set is in the panel grid)
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
    top_margin = 5Plots.mm, bottom_margin = 10Plots.mm)
savefig(combined, joinpath(output_dir, "summary_integer_all_$(pshed_type)_$case.svg"))
display(combined)
