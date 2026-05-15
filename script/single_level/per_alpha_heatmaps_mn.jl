"""
    Standalone per-α load-shed-distribution heatmap renderer.

    Loads a saved JLD2 (written by min_max_trade_off_mn.jl) and renders one
    heatmap per α point in the trade-off sweep: per-bus served fraction
    (1 = on, 0 = shed) across all time periods, grayscale, no per-panel title
    besides the α value.

    Usage:
        julia --project=. script/single_level/per_alpha_heatmaps_mn.jl <path-to-jld2>
"""

using JLD2
using Plots
using Statistics
using LinearAlgebra
using Dates

# Unified 10pt Arial font defaults.
include(joinpath(@__DIR__, "../figure_defaults.jl"))

# ============================================================
# RESOLVE INPUT JLD2 PATH
# ============================================================
function _find_latest_trade_off_jld2(case::String, pshed_type::String)
    base = joinpath(@__DIR__, "../../results")
    isdir(base) || error("results dir not found: $base")
    dates = sort(filter(d -> isdir(joinpath(base, d, "trade_off_mn")), readdir(base)); rev = true)
    for date in dates
        cand = joinpath(base, date, "trade_off_mn", "trade_off_mn_$(case)_$(pshed_type).jld2")
        isfile(cand) && return cand
    end
    error("No saved sweep found for case=$case, pshed_type=$pshed_type")
end

jld_path = if @isdefined(TRADE_OFF_MN_JLD2)
    TRADE_OFF_MN_JLD2
elseif length(ARGS) == 1
    ARGS[1]
elseif length(ARGS) == 2
    _find_latest_trade_off_jld2(ARGS[1], ARGS[2])
else
    error("Provide a JLD2 path or (case pshed_type), or set TRADE_OFF_MN_JLD2 before include.")
end
isfile(jld_path) || error("JLD2 file not found: $jld_path")

println("Loading trade-off sweep data → $jld_path")
saved = JLD2.load(jld_path)
alphas               = saved["alphas"]
per_load_period_shed = saved["per_load_period_shed"]   # alpha × load × period
per_load_period_pd   = saved["per_load_period_pd"]     # load × period
load_bus_names       = saved["load_bus_names"]         # one bus name per load
N_PERIODS            = saved["N_PERIODS"]
case                 = saved["case"]
pshed_type           = saved["pshed_type"]
save_dir             = dirname(jld_path)

n_alphas, n_loads, _ = size(per_load_period_shed)

# ============================================================
# AGGREGATE PER-LOAD VALUES TO PER-BUS (merges split phase buses)
# ============================================================
unique_buses = sort(unique(load_bus_names))
bus_col      = Dict(b => k for (k, b) in enumerate(unique_buses))
load_to_bus  = [bus_col[load_bus_names[j]] for j in 1:n_loads]
n_buses      = length(unique_buses)

bus_pd_matrix = zeros(N_PERIODS, n_buses)
for t in 1:N_PERIODS, j in 1:n_loads
    bus_pd_matrix[t, load_to_bus[j]] += per_load_period_pd[j, t]
end

bus_pshed_alpha = zeros(n_alphas, N_PERIODS, n_buses)
for a in 1:n_alphas, t in 1:N_PERIODS, j in 1:n_loads
    bus_pshed_alpha[a, t, load_to_bus[j]] += per_load_period_shed[a, j, t]
end

# Served fraction in {0, 1} per (α, period, bus); NaN where bus has no load
bus_status_alpha = fill(NaN, n_alphas, N_PERIODS, n_buses)
for a in 1:n_alphas, t in 1:N_PERIODS, b in 1:n_buses
    if bus_pd_matrix[t, b] > 1e-9
        bus_status_alpha[a, t, b] = 1.0 - bus_pshed_alpha[a, t, b] / bus_pd_matrix[t, b]
    end
end

period_labels = ["t=$t" for t in 1:N_PERIODS]

# ============================================================
# GRID OF HEATMAPS — one panel per α
# ============================================================
heatmap_panes = Plots.Plot[]
for a in 1:n_alphas
    p = heatmap(unique_buses, period_labels, bus_status_alpha[a, :, :],
        color  = :grays,
        clims  = (0.0, 1.0),
        title  = "α = $(round(alphas[a], digits=3))",
        xlabel = "Bus",
        ylabel = "Period",
        xrotation = 45,
        yticks = (1:N_PERIODS, period_labels),
        colorbar = false,
    )
    push!(heatmap_panes, p)
end

n_cols = 5
n_rows = ceil(Int, n_alphas / n_cols)
fig = plot(heatmap_panes...,
    layout = (n_rows, n_cols),
    size = (300 * n_cols, 230 * n_rows),
    left_margin = 4Plots.mm, right_margin = 4Plots.mm,
    top_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
)
display(fig)
out_path = joinpath(save_dir, "shed_heatmaps_per_alpha_$(case)_$(pshed_type).svg")
savefig(fig, out_path)
println("Per-α shed-distribution heatmaps → $out_path")
