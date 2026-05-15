"""
    Standalone per-period norm + CoV plotting script.

    Loads a saved bilevel-mn JLD2 (written by run_validation_mn.jl Step 6) and
    renders one 4-panel figure of the per-load raw pshed vector at each time step:
        L1   = norm(pshed_t, 1)
        L2   = norm(pshed_t, 2)
        L∞   = norm(pshed_t, Inf)
        CoV  = std(pshed_t) / mean(pshed_t)

    Each saved run is keyed by (CASE, FAIR_FUNC, pshed_type) — the JLD2 filename
    is bilevel_mn_<CASE>_<FAIR_FUNC>_<pshed_type>.jld2 inside
    results/<date>/bilevel_validation_mn/<CASE>/<FAIR_FUNC>_<pshed_type>/.
"""

using JLD2
using Plots
using Statistics
using LinearAlgebra
using Dates

# Shared fairness-function styling (FAIR_FUNC_COLORS / LABELS / MARKERS / LINESTYLES)
include("../../src/implementation/visualization.jl")

# Unified 10pt Arial font defaults.
include(joinpath(@__DIR__, "../figure_defaults.jl"))

# ============================================================
# RESOLVE INPUT JLD2 PATH
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

jld_path = if @isdefined(BILEVEL_MN_JLD2)
    BILEVEL_MN_JLD2
elseif length(ARGS) == 1
    ARGS[1]
elseif length(ARGS) == 3
    _find_latest_jld2(ARGS[1], ARGS[2], ARGS[3])
else
    error("Provide a JLD2 path, or (case fair_func pshed_type), or set BILEVEL_MN_JLD2 before include.")
end
isfile(jld_path) || error("JLD2 file not found: $jld_path")

println("Loading bilevel run data → $jld_path")
saved = JLD2.load(jld_path)
pshed_matrix       = saved["pshed_matrix"]
load_labels        = saved["load_labels"]
LOAD_SCALE_FACTORS = saved["LOAD_SCALE_FACTORS"]
PEAK_TIME_COSTS    = saved["PEAK_TIME_COSTS"]
CASE               = saved["CASE"]
FAIR_FUNC          = saved["FAIR_FUNC"]
pshed_type         = saved["pshed_type"]
N_PERIODS          = saved["N_PERIODS"]
save_dir           = dirname(jld_path)

# ============================================================
# PER-PERIOD NORMS + COV (mirrors min_max_trade_off_mn.jl::shed_norms)
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

norms_per_t = [begin
    row = pshed_matrix[t, :]
    any(isnan, row) ? (l1=NaN, l2=NaN, linf=NaN, cov=NaN) : shed_norms(row)
end for t in 1:N_PERIODS]

l1_per_t   = [nm.l1   for nm in norms_per_t]
l2_per_t   = [nm.l2   for nm in norms_per_t]
linf_per_t = [nm.linf for nm in norms_per_t]
cov_per_t  = [nm.cov  for nm in norms_per_t]

# ============================================================
# 4-PANEL FIGURE
# ============================================================
periods_axis = collect(1:N_PERIODS)

# Pull shared styling for this fair_func from visualization.jl
ff_color  = get(FAIR_FUNC_COLORS,  FAIR_FUNC, :steelblue)
ff_marker = get(FAIR_FUNC_MARKERS, FAIR_FUNC, :circle)
ff_label  = get(FAIR_FUNC_LABELS,  FAIR_FUNC, FAIR_FUNC)
ff_ls     = get(FAIR_FUNC_LINESTYLES, FAIR_FUNC, :solid)

p_l1 = plot(periods_axis, l1_per_t,
    marker = ff_marker, lw = 2, color = ff_color, linestyle = ff_ls, legend = false,
    xlabel = "period", ylabel = "L1 norm of pshed (kW)",
    title  = "L1 — total shed")
p_l2 = plot(periods_axis, l2_per_t,
    marker = ff_marker, lw = 2, color = ff_color, linestyle = ff_ls, legend = false,
    xlabel = "period", ylabel = "L2 norm of pshed (kW)",
    title  = "L2")
p_linf = plot(periods_axis, linf_per_t,
    marker = ff_marker, lw = 2, color = ff_color, linestyle = ff_ls, legend = false,
    xlabel = "period", ylabel = "L∞ norm of pshed (kW)",
    title  = "L∞ — max shed")
p_cov = plot(periods_axis, cov_per_t,
    marker = ff_marker, lw = 2, color = ff_color, linestyle = ff_ls, legend = false,
    xlabel = "period", ylabel = "CoV (stdev/mean)",
    title  = "Coefficient of variation")

fig = plot(p_l1, p_l2, p_linf, p_cov,
    layout = (2, 2), size = (1400, 1000),
    plot_title = "Per-period pshed norms + CoV — $ff_label  ($CASE / $pshed_type)",
    left_margin = 12Plots.mm, right_margin = 6Plots.mm,
    top_margin = 8Plots.mm, bottom_margin = 12Plots.mm)
display(fig)
out_path = joinpath(save_dir, "per_period_norms_cov_$(CASE)_$(FAIR_FUNC)_$(pshed_type).svg")
savefig(fig, out_path)
println("Per-period norms+CoV figure → $out_path")

println("\n  Per-period norm/CoV summary:")
println("    " * rpad("t", 4) * rpad("L1", 14) * rpad("L2", 14) *
                  rpad("L∞", 14) * "CoV")
for t in 1:N_PERIODS
    println("    " * rpad(string(t), 4) *
            rpad(string(round(l1_per_t[t],   digits=4)), 14) *
            rpad(string(round(l2_per_t[t],   digits=4)), 14) *
            rpad(string(round(linf_per_t[t], digits=4)), 14) *
            string(round(cov_per_t[t], digits=4)))
end
