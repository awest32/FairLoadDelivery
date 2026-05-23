"""
    Standalone per-load grouped-bar replot from a saved bilevel-mn JLD2.

    Renders pshed (kW) per load, grouped over user-specified REP_PERIODS.
    Sibling of loadshed_heatmap_mn.jl — same JLD2 loader, font defaults, and
    save-path convention so the heatmap and grouped bar live on a single
    visual scale.

    Reads pshed_matrix + load_labels directly from the JLD2 (no backfill
    needed — both have been in the saved schema since 2026-05-08). Override
    CASE / FAIR_FUNC / pshed_type / REP_PERIODS at the top before include.

    Each saved run is keyed by (CASE, FAIR_FUNC, pshed_type) — JLD2 lives at
    results/<date>/bilevel_validation_mn/<CASE>/<FAIR_FUNC>_<pshed_type>/
        bilevel_mn_<CASE>_<FAIR_FUNC>_<pshed_type>.jld2.
"""

using JLD2
using Plots
using StatsPlots
using Dates

include(joinpath(@__DIR__, "../figure_defaults.jl"))

CASE       = "case6_unbalanced_switch_more_meshed_good4integer"
FAIR_FUNC  = "min_max"
pshed_type = "absolute"

# Representative periods (1-indexed into the saved run's N_PERIODS). Pick
# trough / plateau / peak indices to span the day. For case6 T=24 with
# SELECTED_HOURS = 0:23, [6, 11, 20] → hours 5, 10, 19.
REP_PERIODS = [6, 11, 20]

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

CASE         = saved["CASE"]
FAIR_FUNC    = saved["FAIR_FUNC"]
pshed_type   = saved["pshed_type"]
N_PERIODS    = saved["N_PERIODS"]
pshed_matrix = saved["pshed_matrix"]
load_labels  = saved["load_labels"]
save_dir     = dirname(jld_path)

n_loads = size(pshed_matrix, 2)
@assert length(load_labels) == n_loads "load_labels length ($(length(load_labels))) ≠ pshed_matrix cols ($n_loads)"

rep_valid = filter(t -> 1 <= t <= N_PERIODS, REP_PERIODS)
if length(rep_valid) != length(REP_PERIODS)
    @warn "Dropped out-of-range REP_PERIODS entries; using $rep_valid (of $REP_PERIODS) for N_PERIODS=$N_PERIODS"
end
isempty(rep_valid) && error("REP_PERIODS=$REP_PERIODS has no valid entries for N_PERIODS=$N_PERIODS")

# Build (n_loads × |rep|) matrix of pshed (kW), NaN → 0 so missing periods
# show as empty bars rather than gaps.
rep_matrix = zeros(n_loads, length(rep_valid))
for (k, t) in enumerate(rep_valid)
    for j in 1:n_loads
        v = pshed_matrix[t, j]
        rep_matrix[j, k] = isnan(v) ? 0.0 : v
    end
end

# Match loadshed_heatmap_mn.jl: number loads 1..n and persist the original
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
out_path = joinpath(save_dir, "loadshed_grouped_replot_$(pshed_type)_$(CASE)_$(FAIR_FUNC).svg")
savefig(p_grouped, out_path)
println("Per-load grouped bar → $out_path")
