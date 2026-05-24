"""
    Standalone period × load served-fraction heatmap from a saved single-level
    multi-period trade-off JLD2.

    Sibling of trade_off_grouped_mn.jl — same loader (handles both min_max
    and palma trade-off JLD2s via FAIR_FUNC switch), same α-nearest-to-target
    behavior, same load-index labeling. The only difference is the figure:
    binarized served-fraction heatmap with the muted teal palette used by
    loadshed_heatmap_mn.jl so this trade-off view lives on the same visual
    scale as the bilevel heatmap.

      * min_max_trade_off_mn.jl →
            results/<date>/trade_off_mn/trade_off_mn_<case>_<pshed_type>.jld2
      * palma_trade_off_mn.jl   →
            results/<date>/palma_trade_off_mn/palma_sweep_mn_<case>_<pshed_type>.jld2

    Both share the same on-disk schema (alphas, per_load_period_shed,
    per_load_period_pd, load_labels, N_PERIODS, case, pshed_type, fair_func),
    so this loader handles either by branching on FAIR_FUNC.

    Override CASE / FAIR_FUNC / pshed_type / ALPHA_TARGET at the top before
    include.
"""

using JLD2
using Plots
using Dates

include(joinpath(@__DIR__, "../figure_defaults.jl"))

# CASE here is the short tag used as the JLD2 filename suffix by the trade-off
# scripts (their local `case` variable) — NOT the full opendss case path. The
# two trade-off scripts currently disagree:
#   min_max_trade_off_mn.jl → case = "more_meshed_6bus"   (no underscore)
#   palma_trade_off_mn.jl   → case = "more_meshed_6_bus"  (with underscore)
# Set CASE to whichever matches the JLD2 you want to plot.
CASE         = "more_meshed_6_bus"
FAIR_FUNC    = "palma"     # "min_max" or "palma"
pshed_type   = "absolute"
ALPHA_TARGET = 0.75

function _find_latest_trade_off_jld2(case::String, fair_func::String,
                                     pshed_type::String)
    base = joinpath(@__DIR__, "../../results")
    isdir(base) || error("results dir not found: $base")

    subdir, fname = if fair_func == "min_max"
        ("trade_off_mn",
         "trade_off_mn_$(case)_$(pshed_type).jld2")
    elseif fair_func == "palma"
        ("palma_trade_off_mn",
         "palma_sweep_mn_$(case)_$(pshed_type).jld2")
    else
        error("Unsupported FAIR_FUNC=$fair_func (expected \"min_max\" or \"palma\")")
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
@assert size(per_load_period_pd) == (n_loads, N_PERIODS) "per_load_period_pd has wrong shape"

# Served fraction: 1 - pshed/pd per (load, period). NaN where pd≈0.
# Then binarize at 1-1e-9 to match loadshed_heatmap_mn.jl: served=1.0,
# any shed=0.0. The integer trade-off MILP returns {0, pd} per load so
# binarization is exact; the threshold guards against fp rounding noise.
served_fraction = fill(NaN, n_loads, N_PERIODS)
for j in 1:n_loads, t in 1:N_PERIODS
    pd = per_load_period_pd[j, t]
    pd > 1e-9 || continue
    s  = shed_matrix[j, t]
    served_fraction[j, t] = isnan(s) ? NaN : 1.0 - s / pd
end
served_binary = map(served_fraction) do v
    isnan(v) && return NaN
    v >= 1.0 - 1e-9 ? 1.0 : 0.0
end
# Heatmap wants (period × load) — rows = y (period), cols = x (load).
served_binary_pt = permutedims(served_binary)   # N_PERIODS × n_loads

# Number loads 1..n and persist the mapping. Same convention as
# loadshed_heatmap_mn.jl and trade_off_grouped_mn.jl so all three replots
# share one load-index map per case.
load_index_labels = string.(1:n_loads)
println("\nLoad index → name mapping:")
for (i, name) in enumerate(load_labels)
    println("  $i\t→\t$name")
end
map_path = joinpath(save_dir, "load_index_map_$(pshed_type)_$(CASE).txt")
open(map_path, "w") do io
    println(io, "# Load index → original load name mapping for $(CASE) / $(FAIR_FUNC) / $(pshed_type)")
    println(io, "# α slice: target=$(ALPHA_TARGET), realized=$(alpha_actual)")
    println(io, "# index\tname")
    for (i, name) in enumerate(load_labels)
        println(io, "$i\t$name")
    end
end
println("Load index map → $map_path")

period_labels = ["t=$t" for t in 1:N_PERIODS]

p_heat = heatmap(load_index_labels, period_labels, served_binary_pt;
    xlabel = "Load",
    ylabel = "Period",
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
println("Per-load heatmap (α=$(alpha_actual)) → $out_path")
