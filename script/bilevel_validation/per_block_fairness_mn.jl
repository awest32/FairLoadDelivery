"""
    Post-hoc per-BLOCK fairness function values for a bilevel multi-period run.

    Loads a saved bilevel-mn JLD2 (per-load `pshed_matrix`, written by
    run_validation_mn.jl Step 6) and computes the same fairness norms the
    single-level trade-off analysis uses
    (`min_max_trade_off_mn.jl::shed_norms`) — L1, L2, L∞, CoV — but evaluated
    on the **per-block aggregate-over-periods** shed vector.

    Why per-block (not per-bus): the block constraint forces every load in a
    load block to share its on/off (z_demand) status. In the 6-bus case each
    block contains a single bus, so per-block and per-bus aggregates coincide;
    in the 13-bus case multiple buses share a block and the right granularity
    is the block. Matches the bilevel single-period per-bus aggregation in
    `compare_fair_funcs_and_networks.jl::extract_per_load_data`, generalized
    to multi-bus blocks.

    Block mapping AND per-period per-load pd are reconstructed by re-parsing
    the case file and rebuilding the profiled multinetwork — so existing JLD2s
    (which may predate `pd_ref_matrix` being saved) work without re-running
    the bilevel. SELECTED_HOURS / PEAK_STRESS / CENTER_AT_NOMINAL must match
    the run-time config (defaults below mirror run_validation_mn.jl).
"""

using FairLoadDelivery
using PowerModelsDistribution
using JLD2
using Plots
using Statistics
using LinearAlgebra
using Dates
using DataFrames
using CSV
using Printf

_PMD = PowerModelsDistribution

include("../../src/implementation/visualization.jl")
include(joinpath(@__DIR__, "../figure_defaults.jl"))

# ============================================================
# CONFIGURATION — set these to target a saved run
# ============================================================
CASE       = "case6_unbalanced_switch_more_meshed_good4integer"
FAIR_FUNC  = "palma"      # "min_max", "palma", or "efficiency"
pshed_type = "absolute"

# Multinetwork profile config — must match what run_validation_mn.jl used at
# the time the JLD2 was written, so pd_ref_matrix reconstructs identically.
SELECTED_HOURS     = collect(0:23)
PEAK_STRESS        = 1.0
CENTER_AT_NOMINAL  = true

# Explicit block ordering. Setup_network enumerates blocks starting at 1; the
# substation block (id 1 or 2) carries no load and gets dropped automatically,
# but among load-bearing blocks the table/plot/metrics use this order. Setting
# to 3:7 for the 6-bus case skips the (empty) source block and gives a fixed
# left-to-right layout that matches the case's bus numbering.
BLOCK_ORDER = collect(3:7)

# Resolve the .dss file the same way run_validation_mn.jl does (6-bus vs
# 13-bus motivation_c live in different subdirs). LS_PERCENT + switch_rating
# don't affect block identification (topology-only) or the per-load pd values
# the multinetwork builder produces, but setup_network requires them.
LS_PERCENT    = 0.8
switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT



# Font-size bump only — overrides the 9pt baseline from figure_defaults.jl so
# the single-panel shed-% chart is legible when scaled down in the paper.
# Margins / canvas size stay on each `plot` call; bumping them globally
# distorts the layout of every figure produced here.
default(
    guidefontsize   = 20,
    tickfontsize    = 18,
    titlefontsize   = 22,
    legendfontsize  = 16,
)


function _resolve_case_file(case::String)
    candidates = [
        joinpath(@__DIR__, "../../data/pmd_opendss/$(case).dss"),
        joinpath(@__DIR__, "../../data/ieee_13_aw_edit/$(case).dss"),
    ]
    for c in candidates
        isfile(c) && return c
    end
    error("Could not find $(case).dss in data/pmd_opendss or data/ieee_13_aw_edit")
end

CASE_FILE = _resolve_case_file(CASE)

# ============================================================
# RESOLVE INPUT JLD2 PATH (latest date for this case/fair_func/pshed_type)
# ============================================================
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
pshed_matrix = saved["pshed_matrix"]      # N_PERIODS × N_LOADS
load_labels  = saved["load_labels"]
CASE         = saved["CASE"]
FAIR_FUNC    = saved["FAIR_FUNC"]
pshed_type   = saved["pshed_type"]
N_PERIODS    = saved["N_PERIODS"]
save_dir     = dirname(jld_path)
@assert size(pshed_matrix, 1) == N_PERIODS
n_loads = size(pshed_matrix, 2)
@assert length(SELECTED_HOURS) == N_PERIODS "SELECTED_HOURS (length $(length(SELECTED_HOURS))) must match N_PERIODS=$N_PERIODS — update top-of-script config to mirror the run's hours"

# ============================================================
# REBUILD LOAD → BLOCK MAP + PER-LOAD PD MATRIX FROM THE CASE FILE
# Re-parses the case so existing JLD2s (without pd_ref_matrix) can still
# produce shed-% values. setup_network installs `math["block"][bid]["loads"]`
# and uses the same load-id parse/sort order results_block_mn.jl used to
# build pshed_matrix, so columns line up.
# ============================================================
println("Re-parsing $CASE_FILE to recover block structure …")
_, math, _, _ = setup_network(CASE_FILE, LS_PERCENT; switch_rating = switch_rating)

ref_load_ids = sort(collect(keys(math["load"])), by = x -> parse(Int, x))
@assert length(ref_load_ids) == n_loads "load count mismatch: JLD2=$n_loads, math=$(length(ref_load_ids))"
for (j, lid) in enumerate(ref_load_ids)
    expected = math["load"][lid]["name"]
    @assert string(load_labels[j]) == expected "load label mismatch at col $j: JLD2=$(load_labels[j]) math=$expected"
end

block_ids = sort(parse.(Int, collect(keys(math["block"]))))
block_to_cols = Dict(b => Int[] for b in block_ids)
load_to_col   = Dict(ref_load_ids[j] => j for j in 1:n_loads)
for b in block_ids
    for lid_int in math["block"][string(b)]["loads"]
        haskey(load_to_col, string(lid_int)) || continue
        push!(block_to_cols[b], load_to_col[string(lid_int)])
    end
end
load_bearing = filter(b -> !isempty(block_to_cols[b]), block_ids)
println("Identified $(length(load_bearing)) load-bearing blocks (of $(length(block_ids)) total): $load_bearing")

# Honor BLOCK_ORDER: keep only listed blocks (in that order). Any requested
# block missing from the network is dropped with a warning so the script
# still produces an output.
nonempty_blocks = Int[]
for b in BLOCK_ORDER
    if b in load_bearing
        push!(nonempty_blocks, b)
    else
        @warn "BLOCK_ORDER includes block $b but it is not a load-bearing block in this case — skipping"
    end
end
isempty(nonempty_blocks) && error("None of BLOCK_ORDER=$BLOCK_ORDER overlap with load-bearing blocks $load_bearing")
n_blocks_eff = length(nonempty_blocks)
println("Using block order: $nonempty_blocks")

# Per-period per-load pd via the same profiled multinetwork builder the run used.
println("Reconstructing per-period per-load pd via create_multinetwork_data_profiled …")
mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS,
    center_at_nominal = CENTER_AT_NOMINAL)
nw_ids_sorted = sort(collect(keys(mn_data["nw"])), by = x -> parse(Int, x))
pd_ref_matrix = zeros(N_PERIODS, n_loads)
for (t, nw_id) in enumerate(nw_ids_sorted)
    for (j, lid) in enumerate(ref_load_ids)
        pd_ref_matrix[t, j] = sum(mn_data["nw"][nw_id]["load"][lid]["pd"])
    end
end

# ============================================================
# AGGREGATE: per-period per-block, then sum over periods
# ============================================================
block_pshed = zeros(N_PERIODS, n_blocks_eff)
block_pd    = zeros(N_PERIODS, n_blocks_eff)
for (bi, b) in enumerate(nonempty_blocks)
    cols = block_to_cols[b]
    for t in 1:N_PERIODS
        block_pshed[t, bi] = sum(pshed_matrix[t, j]  for j in cols if !isnan(pshed_matrix[t, j]); init = 0.0)
        block_pd[t, bi]    = sum(pd_ref_matrix[t, j] for j in cols; init = 0.0)
    end
end
block_agg_pshed = vec(sum(block_pshed, dims = 1))
block_agg_pd    = vec(sum(block_pd,    dims = 1))
block_shed_pct  = [block_agg_pd[bi] > 1e-9 ? 100 * block_agg_pshed[bi] / block_agg_pd[bi] : 0.0
                   for bi in 1:n_blocks_eff]
block_labels    = [string(b) for b in nonempty_blocks]

# ============================================================
# FAIRNESS NORMS (identical to min_max_trade_off_mn.jl::shed_norms)
# ============================================================
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
nm = shed_norms(block_agg_pshed)

# ============================================================
# CONSOLE OUTPUT — per-block detail + post-hoc metrics table
# ============================================================
println("\n  Per-block aggregate-over-periods shed:")
println("    " * rpad("Block", 12) * rpad("# loads", 9) *
                rpad("pshed (kW)", 14) * rpad("pd (kW)", 14) * "shed %")
for (bi, b) in enumerate(nonempty_blocks)
    println("    " * rpad(block_labels[bi], 12) *
                    rpad(string(length(block_to_cols[b])), 9) *
                    rpad(string(round(block_agg_pshed[bi], digits = 3)), 14) *
                    rpad(string(round(block_agg_pd[bi],    digits = 3)), 14) *
                    string(round(block_shed_pct[bi], digits = 2)))
end

println("\n  Post-hoc fairness metrics (per-block aggregate shed vector):")
println("    " * rpad("metric", 8) * "value")
println("    " * rpad("L1",   8) * @sprintf("%.4f", nm.l1))
println("    " * rpad("L2",   8) * @sprintf("%.4f", nm.l2))
println("    " * rpad("L∞",   8) * @sprintf("%.4f", nm.linf))
println("    " * rpad("CoV",  8) * @sprintf("%.4f", nm.cov))

# ============================================================
# CSV OUTPUTS
# ============================================================
per_block_df = DataFrame(
    case       = fill(CASE, n_blocks_eff),
    fair_func  = fill(FAIR_FUNC, n_blocks_eff),
    pshed_type = fill(pshed_type, n_blocks_eff),
    block      = block_labels,
    n_loads    = [length(block_to_cols[b]) for b in nonempty_blocks],
    pshed_kw   = block_agg_pshed,
    pd_kw      = block_agg_pd,
    shed_pct   = block_shed_pct,
)
per_block_csv = joinpath(save_dir, "per_block_shed_$(CASE)_$(FAIR_FUNC)_$(pshed_type).csv")
CSV.write(per_block_csv, per_block_df)
println("\nPer-block shed table → $per_block_csv")

metrics_df = DataFrame(
    case       = fill(CASE, 4),
    fair_func  = fill(FAIR_FUNC, 4),
    pshed_type = fill(pshed_type, 4),
    metric     = ["L1", "L2", "Linf", "CoV"],
    value      = [nm.l1, nm.l2, nm.linf, nm.cov],
)
metrics_csv = joinpath(save_dir, "per_block_fairness_$(CASE)_$(FAIR_FUNC)_$(pshed_type).csv")
CSV.write(metrics_csv, metrics_df)
println("Post-hoc fairness metrics table → $metrics_csv")

# ============================================================
# FIGURE — single panel: shed % per block
# ============================================================
ff_color = get(FAIR_FUNC_COLORS, FAIR_FUNC, :steelblue)
p_block = bar(block_labels, block_shed_pct,
    xlabel = "Load Block", ylabel = "Active Power Load Shed %",# (Σ_t pshed / Σ_t pd × 100)",
    #title  = "Per-block shed % — $(FAIR_FUNC) / $(pshed_type)",
    color  = ff_color, linecolor = :black, linewidth = 0.6, legend = false,
    ylims  = (0, max(100.0, maximum(block_shed_pct; init = 0.0) * 1.15)))
ymax = maximum(block_shed_pct; init = 0.0)
for (i, v) in enumerate(block_shed_pct)
    annotate!(p_block, i, v + (ymax > 0 ? ymax : 1.0) * 0.03,
        text("$(round(v, digits = 1))%", 16, :center))
end
plot!(p_block, size = (1100, 650),
    left_margin = 18Plots.mm, right_margin = 8Plots.mm,
    top_margin = 6Plots.mm, bottom_margin = 16Plots.mm)
display(p_block)
out_path = joinpath(save_dir, "per_block_shed_pct_$(CASE)_$(FAIR_FUNC)_$(pshed_type).svg")
savefig(p_block, out_path)
println("Per-block shed-% figure → $out_path")
