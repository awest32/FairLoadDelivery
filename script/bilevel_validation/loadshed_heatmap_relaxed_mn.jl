"""
    Standalone per-block relaxed-MLD (pre-rounding) heatmap replot.

    The bilevel pipeline's Step 3 solves the relaxed multi-period MLD with
    DiffOpt on the converged fair weights; Step 4 then per-period rounds
    the relaxed switch/block states to integer-feasible topologies and
    re-solves the integer single-period MLD. The standard
    `loadshed_heatmap_mn.jl` renders the *rounded* result. This script
    renders the *relaxed* result — same block ordering, same color ramp,
    but continuous served-fraction in [0, 1] (relaxed pshed can be
    fractional, so we do NOT binarize).

    Reads `relaxed_bus_pshed_matrix` + `relaxed_bus_status_matrix` from the
    JLD2 written by run_validation_mn.jl (Step 6) and renders the per-block
    heatmap. JLD2s that predate the relaxed-capture instrumentation will
    error on the missing-key check below — re-run run_validation_mn.jl to
    refresh them.

    Usage:
        # CLI
        julia --project=. script/bilevel_validation/loadshed_heatmap_relaxed_mn.jl
        # env override of CASE / FAIR_FUNC, pshed_type is always "absolute"
        POSTHOC_CASE=case6_unbalanced_switch_more_meshed_good4integer \\
            POSTHOC_FAIR_FUNC=palma julia --project=. \\
            script/bilevel_validation/loadshed_heatmap_relaxed_mn.jl
"""

using JLD2
using Plots
using Dates

include(joinpath(@__DIR__, "../figure_defaults.jl"))
include(joinpath(@__DIR__, "../block_display.jl"))

CASE       = get(ENV, "POSTHOC_CASE",      "case6_unbalanced_switch_more_meshed_good4integer")
FAIR_FUNC  = get(ENV, "POSTHOC_FAIR_FUNC", "palma")
pshed_type = "absolute"

function _find_latest_jld2(case::String, fair_func::String, pshed_type::String)
    base = joinpath(@__DIR__, "../../results")
    isdir(base) || error("results dir not found: $base")
    dates = sort(filter(d -> isdir(joinpath(base, d, "bilevel_validation_mn", case,
                                            "$(fair_func)_$(pshed_type)")),
                        readdir(base)); rev=true)
    isempty(dates) && error("No saved run found for case=$case, fair_func=$fair_func, pshed_type=$pshed_type")
    return joinpath(base, dates[1], "bilevel_validation_mn", case,
                    "$(fair_func)_$(pshed_type)",
                    "bilevel_mn_$(case)_$(fair_func)_$(pshed_type).jld2")
end

jld_path = _find_latest_jld2(CASE, FAIR_FUNC, pshed_type)
isfile(jld_path) || error("JLD2 file not found: $jld_path")

println("Loading bilevel run data → $jld_path")
saved = JLD2.load(jld_path)

CASE       = saved["CASE"]
FAIR_FUNC  = saved["FAIR_FUNC"]
pshed_type = saved["pshed_type"]
N_PERIODS  = saved["N_PERIODS"]
save_dir   = dirname(jld_path)

bus_labels    = saved["bus_labels"]
bus_pd_matrix = saved["bus_pd_matrix"]

haskey(saved, "relaxed_bus_pshed_matrix") && haskey(saved, "relaxed_bus_status_matrix") ||
    error("JLD2 has no relaxed_bus_* matrices — re-run run_validation_mn.jl with the relaxed-capture instrumentation. Path: $jld_path")
relaxed_bus_pshed_matrix  = saved["relaxed_bus_pshed_matrix"]
relaxed_bus_status_matrix = saved["relaxed_bus_status_matrix"]

period_labels = [string(t) for t in 1:N_PERIODS]

display_info = resolve_block_display_from_buses(CASE, bus_labels)

if display_info !== nothing
    display_blocks, bus2block = display_info
    @assert all(>(0), bus2block) "case $CASE block_display mapping does not cover every bus in bus_labels " *
        "(uncovered: $(bus_labels[bus2block .== 0])). Fix BLOCK_DISPLAY in block_display.jl."
    n_blocks = length(display_blocks)
    block_tick_labels = [string(num) for (num, _) in display_blocks]
    block_index_names = [label    for (_, label) in display_blocks]
    block_order_descr = "paper-aligned (script/block_display.jl)"

    relaxed_block_pd    = zeros(N_PERIODS, n_blocks)
    relaxed_block_pshed = zeros(N_PERIODS, n_blocks)
    for t in 1:N_PERIODS, b in 1:length(bus_labels)
        col = bus2block[b]
        col == 0 && continue
        relaxed_block_pd[t, col]    += bus_pd_matrix[t, b]
        relaxed_block_pshed[t, col] += relaxed_bus_pshed_matrix[t, b]
    end
    status_matrix = fill(NaN, N_PERIODS, n_blocks)
    for t in 1:N_PERIODS, b in 1:n_blocks
        relaxed_block_pd[t, b] > 1e-9 || continue
        status_matrix[t, b] = 1.0 - relaxed_block_pshed[t, b] / relaxed_block_pd[t, b]
    end
else
    n_blocks = length(bus_labels)
    block_tick_labels = string.(1:n_blocks)
    block_index_names = bus_labels
    block_order_descr = "bus-level fallback (no block_display mapping for case $CASE)"
    status_matrix = relaxed_bus_status_matrix
end

println("\nBlock index → name mapping:")
for (i, name) in enumerate(block_index_names)
    println("  $(block_tick_labels[i])\t→\t$name")
end
map_path = joinpath(save_dir, "block_index_map_relaxed_$(pshed_type)_$(CASE).txt")
open(map_path, "w") do io
    println(io, "# Relaxed block index → name mapping for $(CASE) / $(FAIR_FUNC) / $(pshed_type)")
    println(io, "# Block order: $(block_order_descr)")
    println(io, "# index\tname")
    for (i, name) in enumerate(block_index_names)
        println(io, "$(block_tick_labels[i])\t$name")
    end
end
println("Block index map → $map_path")

p_heat = heatmap(block_tick_labels, period_labels, status_matrix,
    xlabel = "Load Block",
    ylabel = "Time Period",
    color  = cgrad(["#E5EFEA", "#2A6F6B"]),
    clims  = (0.0, 1.0),
    xrotation = 0,
    yticks = (1:N_PERIODS, period_labels),
    colorbar = true,
    colorbar_title = "served fraction",
    size = (760, 600),
    left_margin = 5Plots.mm,
    right_margin = 8Plots.mm,
    bottom_margin = 3Plots.mm,
    top_margin = 6Plots.mm,
    tickfontsize = 12,
    guidefontsize = 12,
    titlefontsize = 12,
    legendfontsize = 12,
)
display(p_heat)
out_path = joinpath(save_dir, "loadshed_heatmap_relaxed_replot_$(pshed_type)_$(CASE)_$(FAIR_FUNC).svg")
savefig(p_heat, out_path)
println("Relaxed per-block heatmap → $out_path")
