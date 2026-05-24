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

const PMD = PowerModelsDistribution

include("../../src/implementation/visualization.jl")

# Unified 9pt font defaults for every figure in this script.
include(joinpath(@__DIR__, "../figure_defaults.jl"))

# ============================================================
# CONFIGURATION
# ============================================================
case_name = "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss"
#case_name = "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss"
case = "more_meshed_6_bus"#"13_bus"   # spelling normalized to match palma_trade_off_mn.jl (was "more_meshed_6bus")
dir = @__DIR__
case_path = joinpath(dir, case_name)
date = Dates.format(now(), "yyyy-mm-dd")
LS_PERCENT = 0.8
fair_func = "efficiency"  # "efficiency" or "min_max"
alpha_end = 1
if fair_func == "efficiency"
    alpha_end = 0
elseif fair_func == "min_max"
    alpha_end = 1
else
    error("Unsupported fair_func=$fair_func (expected \"efficiency\" or \"min_max\")")
end
# Multi-period setup: 24 hourly periods. New profile-driven path follows the
# Hamilton & Aliprantis (PECI 2023) strategy — each load gets a deterministic
# (schedule, ±1h shift) assignment from FairLoadDelivery.assign_load_profile.
# Phase-level variation at unbalanced 3-phase buses (e.g. 634a/b/c, L1/L2/L3)
# arises from independent per-phase-load schedules; multi-phase loads whose pd
# is balanced share one schedule across phases.
# Downsampled hours-of-day (0-indexed) covering trough → peak → descent. Cuts
# the single-level multi-period MILP from T=24 to T=8 to keep solve times in
# range comparable to the bilevel scripts.
SELECTED_HOURS    = collect(0:23)   # T=24 full diurnal cycle (was [4,6,8,12,15,18,20,22] for T=8)
N_PERIODS      = length(SELECTED_HOURS)
# Peak-stress multiplier: scales every schedule value uniformly so peak-hour
# demand pushes past nameplate and the network is forced to shed. Paper-faithful
# schedules cap at ~1.10; bump this to drive more shedding, dial it down for
# less stress.
PEAK_STRESS = 1.0
# When true, each schedule is first divided by its own daily mean so the
# daily-average per-load scale equals PEAK_STRESS exactly and the nameplate pd
# is the daily mean (peaks reach ~1.15× nominal at the daily max). Matches the
# bilevel run_efficiency_mn.jl / run_validation_mn.jl convention so trade-off
# vs bilevel results are on the same demand axis. With center_at_nominal=false
# (the create_multinetwork_data_profiled default), raw paper schedules cap
# at ~1.0× nominal and the network is barely stressed.
CENTER_AT_NOMINAL = true

# OLD: uniform linear-ramp scalar applied to every load/phase identically.
# Kept (commented) for reference / quick A/B against the per-load profiles.
# const LOAD_SCALE_FACTORS = [round(s, digits=3) for s in LinRange(0.75, 1.1, N_PERIODS)]

# TOU pricing: low overnight, peak in evening (h≈18)
PEAK_TIME_COSTS = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                        for h in SELECTED_HOURS]

# Representative subset (1-indexed period indices into SELECTED_HOURS) for the
# busy 3-period plots: trough (h=2), midday plateau (h=12), evening peak (h=18).
REP_PERIODS = [6, 11, 20]

pshed_type = "absolute"  # "absolute" or "proportional"
# Solver selection.
#   * fair_func == "efficiency": use the same switch-integer formulation as the
#     bilevel efficiency runner (run_efficiency_mn.jl). Its objective is
#     `min Σ_t λ_t · Σ_i w_i · pshed_{t,i}` — absolute kW weighted shed, no
#     per-period normalization. Wrapped in a closure that silently drops the
#     `alpha` kwarg passed by the sweep loop below.
#   * fair_func == "min_max": route through the min-max integer formulation
#     (proportional variant when pshed_type == "proportional"). Its α=0 endpoint
#     is NOT the bilevel efficiency objective — it minimizes the per-period
#     shed *fraction* (Σ pshed_t / total_demand_t), which compresses peak-vs-
#     off-peak cost differences and yields different optima even when
#     PEAK_TIME_COSTS are identical.
solve_min_max = if fair_func == "efficiency"
    (data, solver; alpha=0.0, kwargs...) ->
        FairLoadDelivery.solve_mn_mc_mld_switch_integer(data, solver; kwargs...)
else
    (data, solver; alpha=1.0, kwargs...) ->
        FairLoadDelivery.solve_mn_mc_mld_min_max_integer(data, solver; alpha=alpha, kwargs...)
end

# ============================================================
# NETWORK SETUP
# ============================================================
eng, math, lbs, critical_id = setup_network(case_path, LS_PERCENT;
    switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT)

# OLD: uniform-scalar multinetwork builder. Kept commented for reference;
# replaced by `create_multinetwork_data_profiled` (per-load, per-phase
# schedules from Hamilton & Aliprantis 2023).
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
#         nw_data["load_scale"] = scale
#         mn_data["nw"][nw_id] = nw_data
#     end
#     return mn_data
# end
# mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)

# Per-load, per-phase schedules (Hamilton & Aliprantis 2023). Each load name is
# deterministically mapped to (schedule_idx ∈ 1:3, shift ∈ {-1,0,+1}); balanced
# multi-phase loads share one schedule across phases, unbalanced ones rotate.
mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS,
    center_at_nominal = CENTER_AT_NOMINAL)

# Pure-diagnostic per-period aggregate demand fraction (not consumed by the
# MIP — mn_data above already encodes the per-phase scales). Saved to the
# JLD2 below so plot/log annotations can label periods with their effective
# aggregate scale, matching the bilevel run_efficiency_mn.jl / run_validation_mn.jl
# convention.
LOAD_SCALE_FACTORS = FairLoadDelivery.aggregate_demand_fraction(math, N_PERIODS;
    hours = SELECTED_HOURS, center_at_nominal = CENTER_AT_NOMINAL) .* PEAK_STRESS
@info "LOAD_SCALE_FACTORS (agg_scales per period): $(round.(LOAD_SCALE_FACTORS, digits=3))"

# Quick sanity dump of the assignment (handy when comparing across cases).
println("Load profile assignments for $case:")
for row in FairLoadDelivery.profile_assignment_table(math)
    println("  ", row)
end
nw_ids_sorted = sort(collect(keys(mn_data["nw"])), by=x->parse(Int, x))
n_loads = length(mn_data["nw"][nw_ids_sorted[1]]["load"])

output_dir = joinpath(@__DIR__, "../../results/$date/trade_off_mn")
isdir(output_dir) || mkpath(output_dir)

# ============================================================
# ALPHA SWEEP
# ============================================================
alpha_points = 20
alphas = collect(LinRange(0, alpha_end, alpha_points))

# Per-(alpha, period) totals for the 3D Pareto
total_shed = zeros(alpha_points, N_PERIODS)
max_shed   = zeros(alpha_points, N_PERIODS)
# Per-load × period distribution captured at alpha=0 and alpha=1 for the summary
per_load_dist_a0 = zeros(n_loads, N_PERIODS)
per_load_dist_a1 = zeros(n_loads, N_PERIODS)
# Per-α, per-load aggregate shed (sum across periods) — used for Figure 2 norms.
per_load_agg = zeros(alpha_points, n_loads)
# Full per-(α, load, period) tensor — used for per-α distribution heatmaps.
per_load_period_shed = zeros(alpha_points, n_loads, N_PERIODS)

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
        per_load_agg[idx, :] .+= per_load_shed
        per_load_period_shed[idx, :, t] .= per_load_shed
        if idx == 1
            per_load_dist_a0[:, t] .= per_load_shed
        elseif idx == alpha_points
            per_load_dist_a1[:, t] .= per_load_shed
        end
    end
end

# ============================================================
# Per-α aggregates and per-load-shed-vector norms
# ============================================================
agg_total_shed = [sum(total_shed[i, :]) for i in 1:alpha_points]
agg_max_shed   = [maximum(per_load_agg[i, :]) for i in 1:alpha_points]

function shed_norms(shed_vec::AbstractVector{<:Real})
    m = Statistics.mean(shed_vec)
    s = Statistics.std(shed_vec)
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
            label = "t=$t (λ=$(PEAK_TIME_COSTS[t]))",
            marker = period_markers[mod1(k, length(period_markers))], lw = 2,
            line_z = alphas)
end
savefig(p3d, joinpath(output_dir, "pareto3d_integer_$(pshed_type)_$(case)_$(fair_func).svg"))
display(p3d)

# Per-period 2D Pareto panel (grid layout — readable for many periods)
panel_cols = N_PERIODS <= 6 ? N_PERIODS : 6
panel_rows = ceil(Int, N_PERIODS / panel_cols)
panel = plot(layout = (panel_rows, panel_cols),
             size = (220 * panel_cols, 180 * panel_rows),
             plot_title = "Per-period Pareto ($(pshed_type), integer) — color = alpha")
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
savefig(panel, joinpath(output_dir, "pareto_per_period_integer_$(pshed_type)_$(case)_$(fair_func).svg"))
display(panel)

# ============================================================
# FIGURE 1: per-load aggregate shed distribution at α=0 and α=1, plus
# aggregate total + max per-load shed vs α (raw kW, summed across periods).
# Distributions show per-load shed summed across all periods (one bar per load).
# ============================================================
ref_nw0 = mn_data["nw"][nw_ids_sorted[1]]
load_labels = [ref_nw0["load"][lid]["name"]
               for lid in sort(collect(keys(ref_nw0["load"])), by=x->parse(Int, x))]

# FONT_KW kept for backwards compat with existing call sites, but now matches
# the 9pt defaults set in figure_defaults.jl so nothing in this script
# overrides the unified font sizes.
FONT_KW = (tickfontsize = 9, guidefontsize = 9,
                 titlefontsize = 9, legendfontsize = 9)

function build_dist_plot_agg(per_load_agg_vec::AbstractVector{<:Real}, title_str::String)
    p = bar(load_labels, per_load_agg_vec,
        xlabel = "load",
        ylabel = "aggregate load shed (kW)",
        title  = title_str,
        legend = false,
        color  = :steelblue,
        linecolor = :black;
        FONT_KW...)
    ymax = maximum(per_load_agg_vec)
    for (i, v) in enumerate(per_load_agg_vec)
        annotate!(p, i, v + (ymax > 0 ? ymax : 1.0) * 0.02,
            text("$(round(v, digits = 1))", 9, :center))
    end
    return p
end

p_dist_a0 = build_dist_plot_agg(per_load_agg[1, :],
    "alpha = 0 (efficiency) — aggregate over periods")
p_dist_a1 = build_dist_plot_agg(per_load_agg[end, :],
    "alpha = 1 (fairness) — aggregate over periods")

p_metrics = plot(alphas, agg_total_shed, label = "total shed (kW)",
    lw = 2, marker = :circle,
    xlabel = "alpha", ylabel = "aggregate load shed (kW)",
    title  = "Aggregate total + max per-load shed vs alpha";
    FONT_KW...)
plot!(p_metrics, alphas, agg_max_shed, label = "max per-load shed (kW)",
    lw = 2, marker = :square)

fig1 = plot(p_dist_a0, p_dist_a1, p_metrics,
    layout = (1, 3), size = (1900, 600),
    left_margin = 14Plots.mm, right_margin = 6Plots.mm,
    top_margin = 8Plots.mm, bottom_margin = 14Plots.mm)
savefig(fig1, joinpath(output_dir, "summary_integer_$(pshed_type)_$(case)_$(fair_func).svg"))
display(fig1)

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
    ylabel = "alpha", title = "", framestyle = :box)

fig2 = plot(p_l1, p_l2, p_linf, p_cov, p_cbar,
    layout = @layout([a b c d e{0.02w}]),
    size = (2200, 600),
    left_margin = 14Plots.mm, right_margin = 6Plots.mm,
    top_margin = 8Plots.mm, bottom_margin = 14Plots.mm)
savefig(fig2, joinpath(output_dir, "pareto_norms_integer_$(pshed_type)_$(case)_$(fair_func).svg"))
display(fig2)

# ============================================================
# PERSIST SWEEP DATA FOR STANDALONE VISUALIZATION SCRIPTS
# Filename pins (case, pshed_type) so each sweep lands in its own JLD2 and
# downstream visualization scripts (e.g. per_alpha_heatmaps_mn.jl) can target
# them by key. Includes everything the heatmap renderer needs without
# re-running the α-sweep.
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

jld_path = joinpath(output_dir, "$(fair_func)_trade_off_mn_$(case)_$(pshed_type).jld2")
JLD2.jldsave(jld_path;
    alphas               = alphas,
    per_load_period_shed = per_load_period_shed,  # alpha × load × period
    per_load_period_pd   = per_load_period_pd,    # load × period
    per_load_agg         = per_load_agg,          # alpha × load (sum over periods)
    total_shed           = total_shed,            # alpha × period
    max_shed             = max_shed,              # alpha × period
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
    fair_func            = fair_func,
)
println("Saved trade-off sweep data → $jld_path")
