"""
    Standalone period × block served-fraction heatmap from a saved
    single-level multi-period trade-off JLD2.

    Sibling of trade_off_grouped_mn.jl — same loader (handles both min_max
    and palma trade-off JLD2s via FAIR_FUNC switch), same α-nearest-to-target
    behavior. The only difference is the figure: binarized per-block
    served-fraction heatmap with the muted teal palette used by
    loadshed_heatmap_mn.jl so this trade-off view lives on the same visual
    scale as the bilevel heatmap.

    Block aggregation has two paths:

      1. Paper-aligned (preferred): script/block_display.jl declares a
         per-case (paper_block_number, label, bus_names) mapping. Blocks
         with no loads in this run are dropped, and the x-axis tick labels
         are the paper block numbers (with gaps if a paper block is
         skipped — e.g. case6 "primary" leaves a 1, 2, 4, 5, 6 axis).

      2. Fallback (case missing from block_display.jl): aggregate to bus
         level using `load_bus_ids` (sort by integer bus_id ascending,
         matching loadshed_heatmap_mn.jl) or first-occurrence by name if
         `load_bus_ids` is absent.

    The block_index_map_*.txt file pins the displayed mapping in either
    path.

      * min_max_trade_off_mn.jl →
            results/<date>/trade_off_mn/trade_off_mn_<case>_<pshed_type>.jld2
      * palma_trade_off_mn.jl   →
            results/<date>/palma_trade_off_mn/palma_sweep_mn_<case>_<pshed_type>.jld2

    Both share the same on-disk schema (alphas, per_load_period_shed,
    per_load_period_pd, load_labels, load_bus_names, N_PERIODS, case,
    pshed_type, fair_func), so this loader handles either by branching on
    FAIR_FUNC.

    Override CASE / FAIR_FUNC / pshed_type / ALPHA_TARGET at the top before
    include.
"""

using JLD2
using Plots
using Dates

include(joinpath(@__DIR__, "../figure_defaults.jl"))
include(joinpath(@__DIR__, "../block_display.jl"))

# CASE here is the short tag used as the JLD2 filename suffix by the trade-off
# scripts (their local `case` variable) — NOT the full opendss case path. The
# two trade-off scripts currently disagree:
#   min_max_trade_off_mn.jl → case = "more_meshed_6bus"   (no underscore)
#   palma_trade_off_mn.jl   → case = "more_meshed_6_bus"  (with underscore)
# Set CASE to whichever matches the JLD2 you want to plot.
CASE         = "more_meshed_6_bus"
#CASE         = "13_bus"
FAIR_FUNC    = "palma"     # "min_max" or "palma"
pshed_type   = "absolute"
ALPHA_TARGET = 0.9

function _find_latest_trade_off_jld2(case::String, fair_func::String,
                                     pshed_type::String)
    base = joinpath(@__DIR__, "../../results")
    isdir(base) || error("results dir not found: $base")

    # min_max_trade_off_mn.jl now writes to "$(fair_func)_trade_off_mn_*.jld2"
    # (both efficiency and min_max modes). palma_trade_off_mn.jl still uses
    # the older "palma_sweep_mn_*" convention.
    subdir, fname = if fair_func == "min_max"
        ("trade_off_mn",
         "min_max_trade_off_mn_$(case)_$(pshed_type).jld2")
    elseif fair_func == "palma"
        ("palma_trade_off_mn",
         "palma_sweep_mn_$(case)_$(pshed_type).jld2")
    elseif fair_func == "efficiency"
        ("trade_off_mn",
         "efficiency_trade_off_mn_$(case)_$(pshed_type).jld2")
    else
        error("Unsupported FAIR_FUNC=$fair_func (expected \"min_max\", \"palma\", or \"efficiency\")")
    end

    dates = sort(filter(d -> isfile(joinpath(base, d, subdir, fname)),
                        readdir(base)); rev=true)
    isempty(dates) && error("No trade-off JLD2 found for case=$case, " *
                            "fair_func=$fair_func, pshed_type=$pshed_type " *
                            "(looked for $subdir/$fname under $base)")
    return joinpath(base, dates[1], subdir, fname)
end

jld_path = _find_latest_trade_off_jld2(CASE, FAIR_FUNC, pshed_type)
isfile(jld_path) || error("JLD2 file not found: $jld_path")

println("Loading trade-off sweep data → $jld_path")
saved = JLD2.load(jld_path)

CASE                 = get(saved, "case", CASE)
FAIR_FUNC            = get(saved, "fair_func", FAIR_FUNC)   # older JLD2s pre-date this field
pshed_type           = get(saved, "pshed_type", pshed_type)
N_PERIODS            = saved["N_PERIODS"]
alphas               = saved["alphas"]
per_load_period_shed = saved["per_load_period_shed"]   # α × load × period
per_load_period_pd   = saved["per_load_period_pd"]     # load × period
load_labels          = saved["load_labels"]
haskey(saved, "load_bus_names") ||
    error("JLD2 missing 'load_bus_names' — re-run the trade-off sweep with the " *
          "current min_max_trade_off_mn.jl / palma_trade_off_mn.jl scripts " *
          "(both save load_bus_names alongside load_labels).")
load_bus_names       = saved["load_bus_names"]
load_bus_ids         = get(saved, "load_bus_ids", nothing)   # nothing on older JLD2s
save_dir             = dirname(jld_path)

# Pick the α nearest to ALPHA_TARGET. Sweeps run on a coarse grid, so the
# actual α may differ slightly — log both so the figure caption can quote
# the realized α.
idx_alpha    = argmin(abs.(alphas .- ALPHA_TARGET))
alpha_actual = alphas[idx_alpha]
println("α target = $(ALPHA_TARGET); nearest saved α = $(alpha_actual) " *
        "(index $(idx_alpha) of $(length(alphas)))")

# Slice the (α × load × period) cube down to (load × period) at the chosen α.
shed_matrix = per_load_period_shed[idx_alpha, :, :]   # n_loads × N_PERIODS
n_loads = size(shed_matrix, 1)
@assert length(load_labels) == n_loads "load_labels length ($(length(load_labels))) ≠ shed_matrix rows ($n_loads)"
@assert length(load_bus_names) == n_loads "load_bus_names length ($(length(load_bus_names))) ≠ n_loads ($n_loads)"
@assert size(per_load_period_pd) == (n_loads, N_PERIODS) "per_load_period_pd has wrong shape"

# Aggregate (load × period) → (block × period). Prefer paper-aligned block
# ordering from script/block_display.jl; if the case has no entry there,
# fall back to bus-level aggregation by sorted integer bus_id (or
# first-occurrence by name if load_bus_ids is also missing).
display_info = resolve_block_display(CASE, load_bus_names)

block_tick_labels  = String[]   # x-axis tick text (paper block numbers, or bus indices)
block_index_names  = String[]   # written into the block_index_map file
load2block         = Vector{Int}(undef, length(load_bus_names))
block_order_descr  = ""

if display_info !== nothing
    display_blocks, load2block = display_info
    @assert all(>(0), load2block) "case $CASE block_display mapping does not cover every load_bus_name " *
        "(uncovered: $(unique(load_bus_names[load2block .== 0]))). Fix BLOCK_DISPLAY in block_display.jl."
    block_tick_labels = [string(num) for (num, _) in display_blocks]
    block_index_names = [label    for (_, label) in display_blocks]
    block_order_descr = "paper-aligned (script/block_display.jl)"
elseif load_bus_ids !== nothing
    unique_ids = sort(unique(load_bus_ids))
    id_to_col  = Dict(bid => k for (k, bid) in enumerate(unique_ids))
    id_to_name = Dict{eltype(load_bus_ids),String}()
    for (j, bid) in enumerate(load_bus_ids)
        haskey(id_to_name, bid) || (id_to_name[bid] = load_bus_names[j])
    end
    block_index_names = [id_to_name[bid] for bid in unique_ids]
    block_tick_labels = string.(1:length(unique_ids))
    for (j, bid) in enumerate(load_bus_ids)
        load2block[j] = id_to_col[bid]
    end
    block_order_descr = "sort by integer bus_id ascending (matches loadshed_heatmap_mn.jl) — no block_display mapping for case $CASE"
else
    bus_col = Dict{String,Int}()
    for (j, name) in enumerate(load_bus_names)
        if !haskey(bus_col, name)
            push!(block_index_names, name)
            bus_col[name] = length(block_index_names)
        end
        load2block[j] = bus_col[name]
    end
    block_tick_labels = string.(1:length(block_index_names))
    block_order_descr = "first-occurrence in load-id-sorted load list (fallback — no block_display mapping, JLD2 had no load_bus_ids)"
end
n_blocks = length(block_index_names)

block_pshed = zeros(n_blocks, N_PERIODS)
block_pd    = zeros(n_blocks, N_PERIODS)
for j in 1:n_loads, t in 1:N_PERIODS
    col = load2block[j]
    col == 0 && continue
    s  = shed_matrix[j, t]
    pd = per_load_period_pd[j, t]
    block_pshed[col, t] += isnan(s) ? 0.0 : s
    block_pd[col, t]    += pd
end

# Served fraction per (block, period). NaN where the block has no demand.
# Then binarize at 1-1e-9 to match loadshed_heatmap_mn.jl: served=1.0,
# any shed = 0.0. The integer MLD returns {0, pd} per load; with the block
# constraint, all loads in a block share shed status — the binarizer just
# guards against fp rounding noise.
served_fraction = fill(NaN, n_blocks, N_PERIODS)
for b in 1:n_blocks, t in 1:N_PERIODS
    block_pd[b, t] > 1e-9 || continue
    served_fraction[b, t] = 1.0 - block_pshed[b, t] / block_pd[b, t]
end
served_binary = map(served_fraction) do v
    isnan(v) && return NaN
    v >= 1.0 - 1e-9 ? 1.0 : 0.0
end
# Heatmap wants (period × block) — rows = y (period), cols = x (block).
served_binary_pt = permutedims(served_binary)   # N_PERIODS × n_blocks

println("\nBlock index → name mapping:")
for (i, name) in enumerate(block_index_names)
    println("  $(block_tick_labels[i])\t→\t$name")
end
map_path = joinpath(save_dir, "block_index_map_$(pshed_type)_$(CASE).txt")
open(map_path, "w") do io
    println(io, "# Block index → name mapping for $(CASE) / $(FAIR_FUNC) / $(pshed_type)")
    println(io, "# α slice: target=$(ALPHA_TARGET), realized=$(alpha_actual)")
    println(io, "# Block order: $(block_order_descr)")
    println(io, "# index\tname")
    for (i, name) in enumerate(block_index_names)
        println(io, "$(block_tick_labels[i])\t$name")
    end
end
println("Block index map → $map_path")

period_labels = [string(t) for t in 1:N_PERIODS]

p_heat = heatmap(block_tick_labels, period_labels, served_binary_pt;
    xlabel = "Load Block",
    ylabel = "Time Period",
    color  = cgrad(["#E5EFEA", "#2A6F6B"]),   # pale sage (shed) → muted teal (served)
    clims  = (0.0, 1.0),
    xrotation = 0,
    yticks = (1:N_PERIODS, period_labels),
    colorbar = false,
    size = (700, 500),
    left_margin = 5Plots.mm,
    right_margin = 5Plots.mm,
    bottom_margin = 3Plots.mm,
    top_margin = 6Plots.mm,
    tickfontsize = 12,
    guidefontsize = 12,
    titlefontsize = 12,
    legendfontsize = 12,
)
display(p_heat)

alpha_tag = replace(string(round(alpha_actual; digits = 3)), "." => "p")
out_path = joinpath(save_dir,
    "trade_off_heatmap_$(pshed_type)_$(CASE)_$(FAIR_FUNC)_alpha$(alpha_tag).svg")
savefig(p_heat, out_path)
println("Per-block heatmap (α=$(alpha_actual)) → $out_path")
