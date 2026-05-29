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
FAIR_FUNC  = get(ENV, "POSTHOC_FAIR_FUNC", "efficiency")
pshed_type = "absolute"

# Bilevel CASE → trade-off case tag mapping (mirrors post_hoc_fairness_pareto.jl
# CASE_TAGS). Trade-off JLD2 filenames use the shorter tag.
const _BILEVEL_TO_TRADEOFF_CASE = Dict(
    "case6_unbalanced_switch_more_meshed_good4integer" => "more_meshed_6_bus",
    "motivation_c_good4integer"                        => "13_bus",
)

"""
Locate the freshest JLD2 carrying a relaxed multi-period MLD solution for
(case, fair_func, pshed). Sources, in preference order:

  1. Single-level *relaxed trade-off* sweep — the truly fractional solve.
     The bilevel Step 3 with integer warm-start usually collapses to an
     integer optimum, so its `relaxed_bus_*` fields look identical to the
     rounded heatmap. The trade-off sweep skips the warm-start, so its
     α-slice is genuinely fractional (verified empirically for case6 T=24
     efficient: 166/216 (load × period) cells in (0.01, 0.99)).
     Path:  results/<date>/trade_off_mn/<fair_func>_relaxed_trade_off_mn_<tag>_<pshed>.jld2
            results/<date>/palma_relaxed_trade_off_mn/palma_sweep_mn_<tag>_<pshed>.jld2
  2. Relaxed-only bilevel pipeline (run_validation_mn_relaxed.jl).
  3. Full bilevel pipeline (run_validation_mn.jl) — its `relaxed_bus_*`
     fields are the integer-warm-started Step 3 solution.
"""
function _find_latest_jld2(case::String, fair_func::String, pshed_type::String)
    base = joinpath(@__DIR__, "../../results")
    isdir(base) || error("results dir not found: $base")

    to_case = get(_BILEVEL_TO_TRADEOFF_CASE, case, case)

    # source tag: :tradeoff > :bilevel_relaxed > :bilevel_integer.
    candidates = Tuple{String,Symbol}[]
    for d in readdir(base)
        run_dir = joinpath(base, d)
        isdir(run_dir) || continue

        # 1a. Trade-off sweep for min_max / efficiency (relaxed variant).
        if fair_func in ("efficiency", "min_max")
            tp = joinpath(run_dir, "trade_off_mn",
                          "$(fair_func)_relaxed_trade_off_mn_$(to_case)_$(pshed_type).jld2")
            isfile(tp) && push!(candidates, (tp, :tradeoff))
        end
        # 1b. Trade-off sweep for palma (folder name encodes "relaxed").
        if fair_func == "palma"
            tp = joinpath(run_dir, "palma_relaxed_trade_off_mn",
                          "palma_sweep_mn_$(to_case)_$(pshed_type).jld2")
            isfile(tp) && push!(candidates, (tp, :tradeoff))
        end
        # 2. Relaxed-only bilevel pipeline.
        rlx_path = joinpath(run_dir, "bilevel_validation_mn_relaxed", case,
                            "$(fair_func)_$(pshed_type)",
                            "bilevel_mn_relaxed_$(case)_$(fair_func)_$(pshed_type).jld2")
        isfile(rlx_path) && push!(candidates, (rlx_path, :bilevel_relaxed))
        # 3. Full bilevel pipeline (its relaxed_bus_* fields).
        int_path = joinpath(run_dir, "bilevel_validation_mn", case,
                            "$(fair_func)_$(pshed_type)",
                            "bilevel_mn_$(case)_$(fair_func)_$(pshed_type).jld2")
        isfile(int_path) && push!(candidates, (int_path, :bilevel_integer))
    end
    isempty(candidates) && error("No saved run found for case=$case, fair_func=$fair_func, pshed_type=$pshed_type")

    # Sort: lower rank wins. Within rank, freshest mtime wins.
    rank(src) = src == :tradeoff ? 0 : src == :bilevel_relaxed ? 1 : 2
    sort!(candidates; by = c -> (rank(c[2]), -mtime(c[1])))
    return candidates[1]
end

jld_path, source_kind = _find_latest_jld2(CASE, FAIR_FUNC, pshed_type)
isfile(jld_path) || error("JLD2 file not found: $jld_path")
println("Source: $source_kind  ($jld_path)")

println("Loading run data → $jld_path")
saved = JLD2.load(jld_path)

# Bilevel JLD2s expose CASE / FAIR_FUNC at top level; trade-off JLD2s use
# lowercase `case` / `fair_func` and don't carry bus_pd_matrix or
# relaxed_bus_*. Build a uniform view first, then dispatch.
N_PERIODS  = saved["N_PERIODS"]
pshed_type = saved["pshed_type"]
save_dir   = dirname(jld_path)

if source_kind === :tradeoff
    # Trade-off JLD2 layout: per_load_period_shed (α, load, period),
    # per_load_period_pd (load, period), load_bus_ids (load → bus id).
    # Pick the α slice — default 1 (efficient endpoint, α=0) for efficiency
    # and `end` (fair endpoint, α≈1) for palma / min_max so the heatmap shows
    # the formulation's natural target solution. Overridable via env.
    fair_func_raw = saved["fair_func"]
    default_α_idx = startswith(fair_func_raw, "efficiency") ? 1 :
                    length(saved["alphas"])
    α_idx = parse(Int, get(ENV, "POSTHOC_ALPHA_IDX", string(default_α_idx)))
    @assert 1 <= α_idx <= length(saved["alphas"]) "POSTHOC_ALPHA_IDX out of range"

    per_load_period_shed = saved["per_load_period_shed"]   # (α, load, period)
    per_load_period_pd   = saved["per_load_period_pd"]     # (load, period)
    load_bus_ids         = saved["load_bus_ids"]
    load_bus_names       = saved["load_bus_names"]

    α_val = saved["alphas"][α_idx]
    println("Trade-off α slice: idx=$α_idx (α=$(round(α_val, digits=3)))")

    # Aggregate per-load → per-bus. bus_labels = unique bus_ids in sorted
    # order (matches the bilevel pipeline's convention).
    unique_bus_ids = sort(unique(load_bus_ids))
    bus_labels = String[]
    bus_id_to_name = Dict(zip(load_bus_ids, load_bus_names))
    for bid in unique_bus_ids
        push!(bus_labels, get(bus_id_to_name, bid, "bus_$bid"))
    end
    bus_col = Dict(bid => k for (k, bid) in enumerate(unique_bus_ids))

    bus_pd_matrix             = zeros(N_PERIODS, length(unique_bus_ids))
    relaxed_bus_pshed_matrix  = zeros(N_PERIODS, length(unique_bus_ids))
    for (l_idx, bid) in enumerate(load_bus_ids), t in 1:N_PERIODS
        col = bus_col[bid]
        bus_pd_matrix[t, col]            += per_load_period_pd[l_idx, t]
        relaxed_bus_pshed_matrix[t, col] += per_load_period_shed[α_idx, l_idx, t]
    end
    relaxed_bus_status_matrix = fill(NaN, N_PERIODS, length(unique_bus_ids))
    for t in 1:N_PERIODS, b in 1:length(unique_bus_ids)
        if bus_pd_matrix[t, b] > 1e-9
            relaxed_bus_status_matrix[t, b] =
                1.0 - relaxed_bus_pshed_matrix[t, b] / bus_pd_matrix[t, b]
        end
    end
else
    # Bilevel JLD2 — fields are already in the right shape.
    CASE       = saved["CASE"]
    FAIR_FUNC  = saved["FAIR_FUNC"]
    bus_labels    = saved["bus_labels"]
    bus_pd_matrix = saved["bus_pd_matrix"]
    haskey(saved, "relaxed_bus_pshed_matrix") && haskey(saved, "relaxed_bus_status_matrix") ||
        error("JLD2 has no relaxed_bus_* matrices — re-run run_validation_mn.jl with the relaxed-capture instrumentation. Path: $jld_path")
    relaxed_bus_pshed_matrix  = saved["relaxed_bus_pshed_matrix"]
    relaxed_bus_status_matrix = saved["relaxed_bus_status_matrix"]
end

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
