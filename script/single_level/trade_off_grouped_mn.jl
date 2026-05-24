"""
    Standalone per-load grouped-bar replot from a saved single-level
    multi-period trade-off JLD2.

    Sibling of loadshed_grouped_mn.jl, but instead of reading a bilevel-mn
    run keyed by (CASE, FAIR_FUNC, pshed_type), this script reads an α-sweep
    saved by one of the single-level trade-off scripts:

      * min_max_trade_off_mn.jl →
            results/<date>/trade_off_mn/trade_off_mn_<case>_<pshed_type>.jld2
      * palma_trade_off_mn.jl   →
            results/<date>/palma_trade_off_mn/palma_sweep_mn_<case>_<pshed_type>.jld2

    Both share the same on-disk schema (alphas, per_load_period_shed,
    load_labels, N_PERIODS, case, pshed_type, fair_func), so this loader
    handles either by branching on FAIR_FUNC.

    The α-sweep stores shed for every (α, load, period) triple. This script
    picks the single α-slice nearest ALPHA_TARGET (configurable), then renders
    a grouped bar of pshed (kW) per load over REP_PERIODS — matching the
    bilevel grouped-bar's font/margins/save-path conventions so the trade-off
    and bilevel views live on a single visual scale.

    Override CASE / FAIR_FUNC / pshed_type / ALPHA_TARGET / REP_PERIODS at the
    top before include.
"""

using JLD2
using Plots
using StatsPlots
using Dates

include(joinpath(@__DIR__, "../figure_defaults.jl"))

# CASE here is the short tag used as the JLD2 filename suffix by the trade-off
# scripts (their local `case` variable) — NOT the full opendss case path. The
# two trade-off scripts currently disagree:
#   min_max_trade_off_mn.jl → case = "more_meshed_6bus"   (no underscore)
#   palma_trade_off_mn.jl   → case = "more_meshed_6_bus"  (with underscore)
# Set CASE to whichever matches the JLD2 you want to plot.
CASE         = "more_meshed_6_bus"
FAIR_FUNC    = "efficiency"     # "min_max" or "palma"
pshed_type   = "absolute"
ALPHA_TARGET = 0.75

# Representative periods (1-indexed into the saved sweep's N_PERIODS). Pick
# trough / plateau / peak indices to span the day.
# `nothing` → auto-pick based on N_PERIODS after the JLD2 is loaded:
#   T=8  → [2, 4, 6]   (matches palma_trade_off_mn.jl / min_max_trade_off_mn.jl)
#   T=24 → [6, 11, 20] (matches run_validation_mn.jl bilevel default)
# Override here to pin a specific subset.
REP_PERIODS = nothing

function _find_latest_trade_off_jld2(case::String, fair_func::String,
                                     pshed_type::String)
    base = joinpath(@__DIR__, "../../results")
    isdir(base) || error("results dir not found: $base")

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
per_load_period_shed = saved["per_load_period_shed"]  # α × load × period
load_labels          = saved["load_labels"]
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

# Resolve REP_PERIODS — either honor an explicit override or pick a sensible
# default that's in-range for this sweep's N_PERIODS.
rep_periods_resolved = if REP_PERIODS === nothing
    if N_PERIODS == 24
        [6, 11, 20]
    elseif N_PERIODS == 8
        [2, 4, 6]
    elseif N_PERIODS == 3
        collect(1:3)
    else
        unique([1, max(1, N_PERIODS ÷ 2), N_PERIODS])
    end
else
    REP_PERIODS
end
println("REP_PERIODS = $rep_periods_resolved (N_PERIODS=$N_PERIODS)")

rep_valid = filter(t -> 1 <= t <= N_PERIODS, rep_periods_resolved)
if length(rep_valid) != length(rep_periods_resolved)
    @warn "Dropped out-of-range REP_PERIODS entries; using $rep_valid (of $rep_periods_resolved) for N_PERIODS=$N_PERIODS"
end
isempty(rep_valid) && error("REP_PERIODS=$rep_periods_resolved has no valid entries for N_PERIODS=$N_PERIODS")

# Build (n_loads × |rep|) matrix of pshed (kW), NaN → 0 so missing periods
# show as empty bars rather than gaps.
rep_matrix = zeros(n_loads, length(rep_valid))
for (k, t) in enumerate(rep_valid)
    for j in 1:n_loads
        v = shed_matrix[j, t]
        rep_matrix[j, k] = isnan(v) ? 0.0 : v
    end
end

# Match loadshed_grouped_mn.jl: number loads 1..n and persist the original
# names so the bar chart can be cross-referenced. Avoids cramming long load
# strings (e.g. `L9`, `loadbusC.1.2.3.0`) into the x-axis.
load_index_labels = string.(1:n_loads)
println("\nLoad index → name mapping:")
for (i, name) in enumerate(load_labels)
    println("  $i\t→\t$name")
end
map_path = joinpath(save_dir, "load_index_map_$(pshed_type)_$(CASE).txt")
open(map_path, "w") do io
    println(io, "# Load index → original load name mapping for $(CASE) / $(FAIR_FUNC) / $(pshed_type)")
    println(io, "# index\tname")
    for (i, name) in enumerate(load_labels)
        println(io, "$i\t$name")
    end
end
println("Load index map → $map_path")

rep_labels = reshape(["t=$t" for t in rep_valid], 1, length(rep_valid))

p_grouped = groupedbar(load_index_labels, rep_matrix;
    bar_position = :dodge,
    labels = rep_labels,
    xlabel = "Load",
    ylabel = "Load shed (kW)",
    legend = :topright,
    linecolor = :match,   # outline matches fill — no heavy black border
    xrotation = 0,
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
display(p_grouped)

alpha_tag = replace(string(round(alpha_actual; digits = 3)), "." => "p")
out_path = joinpath(save_dir,
    "trade_off_grouped_$(pshed_type)_$(CASE)_$(FAIR_FUNC)_alpha$(alpha_tag).svg")
savefig(p_grouped, out_path)
println("Per-load grouped bar (α=$(alpha_actual)) → $out_path")
