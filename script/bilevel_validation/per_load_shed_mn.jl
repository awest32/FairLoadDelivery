"""
    Standalone per-load aggregate-over-periods load-shed bar replot from a
    saved bilevel-mn JLD2.

    Mirrors per_block_fairness_mn.jl's single-panel shed-bar style, but at the
    per-LOAD granularity and in absolute kW (Σ_t pshed[t, load]) rather than
    %. No backfill needed — pshed_matrix + load_labels have been in the JLD2
    schema since 2026-05-08.

    Each saved run is keyed by (CASE, FAIR_FUNC, pshed_type) — JLD2 lives at
    results/<date>/bilevel_validation_mn/<CASE>/<FAIR_FUNC>_<pshed_type>/
        bilevel_mn_<CASE>_<FAIR_FUNC>_<pshed_type>.jld2.
"""

using FairLoadDelivery
using JLD2
using Plots
using Dates
using DataFrames
using CSV
using Printf

include("../../src/implementation/visualization.jl")   # FAIR_FUNC_COLORS etc.
include(joinpath(@__DIR__, "../figure_defaults.jl"))

# ============================================================
# CONFIGURATION — set these to target a saved run
# ============================================================
CASE       = "case6_unbalanced_switch_more_meshed_good4integer"
FAIR_FUNC  = "palma"      # "min_max", "palma", or "efficiency"
pshed_type = "absolute"

# Font-size bump only — matches per_block_fairness_mn.jl so this figure
# lives on the same visual scale as the other bilevel-validation aggregates.
default(
    guidefontsize  = 20,
    tickfontsize   = 18,
    titlefontsize  = 22,
    legendfontsize = 16,
)

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

# Aggregate Σ_t pshed[t, load] (NaN → 0 so missing periods don't poison the sum).
load_agg_pshed = [sum(pshed_matrix[t, j] for t in 1:N_PERIODS if !isnan(pshed_matrix[t, j]); init = 0.0)
                  for j in 1:n_loads]

# Number loads 1..N for the x-axis and persist the original names. Same
# convention as loadshed_heatmap_mn.jl / loadshed_grouped_mn.jl, so all three
# figures cross-reference via one mapping file per case.
load_index_labels = string.(1:n_loads)
println("\nLoad index → name → aggregate shed (kW):")
for (i, name) in enumerate(load_labels)
    @printf "  %3d  %-30s  %10.3f kW\n" i name load_agg_pshed[i]
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

# CSV: one row per load
per_load_df = DataFrame(
    case       = fill(CASE, n_loads),
    fair_func  = fill(FAIR_FUNC, n_loads),
    pshed_type = fill(pshed_type, n_loads),
    load_idx   = collect(1:n_loads),
    load_name  = load_labels,
    pshed_kw   = load_agg_pshed,
)
per_load_csv = joinpath(save_dir, "per_load_shed_$(CASE)_$(FAIR_FUNC)_$(pshed_type).csv")
CSV.write(per_load_csv, per_load_df)
println("Per-load shed table → $per_load_csv")

# Bar chart — mirrors per_block_fairness_mn.jl's single-panel shed-% layout.
ff_color = get(FAIR_FUNC_COLORS, FAIR_FUNC, :steelblue)
ymax     = maximum(load_agg_pshed; init = 0.0)
p_load = bar(load_index_labels, load_agg_pshed;
    xlabel = "Load",
    ylabel = "Active Power Load Shed (kW)",
    color     = ff_color,
    linecolor = :match,        # no heavy black border (matches grouped-bar fix)
    linewidth = 0.6,
    legend    = false,
    ylims     = (0.0, ymax > 0 ? ymax * 1.15 : 1.0),
)
for (i, v) in enumerate(load_agg_pshed)
    annotate!(p_load, i, v + (ymax > 0 ? ymax : 1.0) * 0.03,
        text(@sprintf("%.1f", v), 16, :center))
end
plot!(p_load, size = (1100, 650),
    left_margin = 18Plots.mm, right_margin = 8Plots.mm,
    top_margin  = 6Plots.mm,  bottom_margin = 16Plots.mm)
display(p_load)
out_path = joinpath(save_dir, "per_load_shed_$(CASE)_$(FAIR_FUNC)_$(pshed_type).svg")
savefig(p_load, out_path)
println("Per-load shed figure → $out_path")
