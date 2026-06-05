"""
    Post-hoc fairness-vs-efficiency Pareto plots.

    Two figure sets per (CASE, PSHED_TYPE) — one for integer sweeps,
    one for relaxed sweeps. Each set shows palma and min_max as
    different line styles, overlaid with three bilevel points —
    Palma (★), efficient (♦), min-max (✱).

    Outputs (under results/<today>/post_hoc_fairness/), one per
    variant ∈ {integer, relaxed}:
      * pareto_<case>_<pshed>_<variant>_palma.svg  — y = Palma post-hoc
      * pareto_<case>_<pshed>_<variant>_cov.svg       — y = CoV
      * pareto_<case>_<pshed>_<variant>.svg           — 4-up [L1, L∞, Palma, CoV]

    Usage:
        julia --project=. script/post_hoc_fairness_pareto.jl
"""

using JLD2
using LinearAlgebra
using Statistics
using Plots
using Dates
using FairLoadDelivery: gini_index

include(joinpath(@__DIR__, "figure_defaults.jl"))

const CASE_TAGS = Dict(
    "case6" => (bilevel  = "case6_unbalanced_switch_more_meshed_good4integer",
                trade_off = "more_meshed_6_bus"),
    "motivation_c" => (bilevel  = "motivation_c_good4integer",
                       trade_off = "13_bus"),
)

# ============================================================
# CONFIGURATION
# ============================================================
CASE_KEY   = "case6"
PSHED_TYPE = "absolute"
SHOW_BILEVEL = true
PLOT_ZOOM    = false   # skip the bilevel-star-framed zoom fronts
PALMA_ONLY   = true    # restrict to the Palma front + Palma bilevel marker; output to *_palmaonly

# Manual bilevel-JLD2 overrides — keyed by BILEVEL_STYLES.key. Use when the
# latest run on disk is wrong (e.g. T mismatch with the single-level sweeps,
# bad convergence) and you want to pin a specific historical result instead.
# Set to nothing / drop the key to fall back to `_bilevel_jld2`'s mtime pick.
const BILEVEL_OVERRIDES = Dict{String, String}()

@assert haskey(CASE_TAGS, CASE_KEY) "Unknown CASE_KEY=$CASE_KEY"
tags = CASE_TAGS[CASE_KEY]
RESULTS_ROOT = joinpath(@__DIR__, "../results")

# Sweeps to overlay as single-level lines and bilevel objectives to
# overlay as scatter markers. Linestyle distinguishes fair-func (palma
# vs min-max). Integer and relaxed panels are rendered separately, so
# each fair-func uses the same full-intensity color across both variants.
const SWEEP_STYLES = [
    # 2026-06-04 T=8 comparison: the Palma trade-off was rerun at T=8 (results
    # 2026-06-03, N_PERIODS=8) so the Palma sweep is re-enabled. The min-max
    # trade-off only exists at T=24 on disk, which would contaminate a T=8
    # plot — disabled here until a T=8 min-max trade-off is run.
    (key = "palma_integer",   label = "Single-level Palma (integer)",
     fair_func = "palma",   relaxed = false,
     linestyle = :solid, color = RGB(0.20, 0.40, 0.85)),
    (key = "palma_relaxed",   label = "Single-level Palma (relaxed)",
     fair_func = "palma",   relaxed = true,
     linestyle = :solid, color = RGB(0.20, 0.40, 0.85)),
    # (key = "min_max_integer", label = "Single-level min-max (integer)",
    #  fair_func = "min_max", relaxed = false,
    #  linestyle = :dash,  color = RGB(0.85, 0.30, 0.20)),
    # (key = "min_max_relaxed", label = "Single-level min-max (relaxed)",
    #  fair_func = "min_max", relaxed = true,
    #  linestyle = :dash,  color = RGB(0.85, 0.30, 0.20)),
]

const BILEVEL_STYLES = [
    (key = "palma",      label = "Bi-level Palma",      marker = :star5,
     color = :black),
    (key = "gini",       label = "Bi-level Gini",       marker = :diamond,
     color = :black),
    (key = "min_max",    label = "Bi-level min-max",    marker = :star8,
     color = :black),
]

# ============================================================
# JLD2 LOCATORS
# ============================================================
function _latest_matching(rel_path_predicate)
    candidates = String[]
    for d in readdir(RESULTS_ROOT)
        dir = joinpath(RESULTS_ROOT, d)
        isdir(dir) || continue
        for path in rel_path_predicate(d)
            isfile(path) && push!(candidates, path)
        end
    end
    isempty(candidates) && return nothing
    # Break ties by filesystem mtime first, lexicographic path second.
    # Sorting by date-folder name alone is fragile: two same-day runs in
    # differently named subfolders (e.g. palma_trade_off_mn vs
    # palma_trade_off_mn_8_periods) would otherwise pick the alphabetically-
    # later subfolder regardless of which was actually written last.
    return sort(candidates; by = p -> (mtime(p), p), rev = true)[1]
end

function _trade_off_jld2(fair_func::String, relaxed::Bool)
    if fair_func == "min_max"
        rel = relaxed ? "_relaxed" : ""
        return _latest_matching(d -> [joinpath(RESULTS_ROOT, d, "trade_off_mn",
            "min_max$(rel)_trade_off_mn_$(tags.trade_off)_$(PSHED_TYPE).jld2")])
    elseif fair_func == "palma"
        # Palma writes the relaxed sweep into a sibling folder
        # (palma_relaxed_trade_off_mn) with the same inner filename.
        return _latest_matching(d -> begin
            date_dir = joinpath(RESULTS_ROOT, d)
            isdir(date_dir) || return String[]
            [joinpath(date_dir, sub,
                      "palma_sweep_mn_$(tags.trade_off)_$(PSHED_TYPE).jld2")
             for sub in readdir(date_dir)
             if isdir(joinpath(date_dir, sub)) &&
                (relaxed ? startswith(sub, "palma_relaxed_trade_off_mn") :
                           startswith(sub, "palma_trade_off_mn"))]
        end)
    end
    return nothing
end

function _bilevel_jld2(fair_func::String)
    return _latest_matching(d -> [joinpath(RESULTS_ROOT, d,
        "bilevel_validation_mn", tags.bilevel,
        "$(fair_func)_$(PSHED_TYPE)",
        "bilevel_mn_$(tags.bilevel)_$(fair_func)_$(PSHED_TYPE).jld2")])
end

# ============================================================
# NORMS
# ============================================================
function _palma_ratio_safe(x::AbstractVector{<:Real})
    n = length(x)
    n < 3 && return NaN
    sorted_x = sort(collect(x))
    top10 = sum(sorted_x[ceil(Int, 0.9n):end])
    bot40 = sum(sorted_x[1:floor(Int, 0.4n)])
    total = sum(sorted_x)
    # Relative threshold: bot40 must hold ≥ 0.01% of total shed.
    # The previous 1e-9 absolute floor admitted solver-tolerance noise at
    # low α (no loads truly shed in bot40), producing Palma ≈ 10⁹ that
    # crushed the y-axis when both sweeps share a panel.
    return (bot40 > 1e-4 * total) ? top10 / bot40 : NaN
end

function shed_norms(v::AbstractVector{<:Real})
    finite = filter(isfinite, v)
    isempty(finite) && return (L1 = NaN, L2 = NaN, Linf = NaN,
                               CoV = NaN, Palma = NaN, Gini = NaN)
    m = mean(finite)
    s = length(finite) > 1 ? std(finite) : 0.0
    return (
        L1    = norm(finite, 1),
        L2    = norm(finite, 2),
        Linf  = norm(finite, Inf),
        CoV   = m > 1e-9 ? s / m : NaN,
        Palma = _palma_ratio_safe(finite),
        Gini  = gini_index(Float64.(finite)),
    )
end

# ============================================================
# DATA LOADERS
# ============================================================
function load_trade_off_curve(path::String)
    saved        = JLD2.load(path)
    per_load_agg = saved["per_load_agg"]
    alphas       = saved["alphas"]
    n_α          = length(alphas)
    L1    = zeros(n_α); L2 = zeros(n_α); Linf = zeros(n_α)
    CoV   = zeros(n_α); Palma = zeros(n_α); Gini = zeros(n_α)
    for i in 1:n_α
        nm = shed_norms(collect(per_load_agg[i, :]))
        L1[i] = nm.L1; L2[i] = nm.L2; Linf[i] = nm.Linf
        CoV[i] = nm.CoV; Palma[i] = nm.Palma; Gini[i] = nm.Gini
    end
    total_shed = L1
    return (alphas = alphas, total_shed = total_shed,
            L1 = L1, L2 = L2, Linf = Linf, CoV = CoV, Palma = Palma, Gini = Gini,
            n_periods = saved["N_PERIODS"], source = path)
end

"Aggregate a (T × n_loads) shed matrix into per-load totals, ignoring NaNs."
function _aggregate_norms(matrix::AbstractMatrix)
    n_loads = size(matrix, 2)
    v = zeros(n_loads)
    for t in axes(matrix, 1), j in 1:n_loads
        x = matrix[t, j]; isnan(x) || (v[j] += x)
    end
    nm = shed_norms(v)
    return (total_shed = nm.L1, L1 = nm.L1, L2 = nm.L2,
            Linf = nm.Linf, CoV = nm.CoV, Palma = nm.Palma, Gini = nm.Gini)
end

function load_bilevel_point(path::String)
    saved = JLD2.load(path)
    int_norms = _aggregate_norms(saved["pshed_matrix"])
    # The bilevel pipeline saves the final relaxed MLD (Step 3, pre-rounding)
    # alongside the rounded integer solution. Older JLD2s predate this
    # instrumentation or may have an all-NaN matrix on non-convergence —
    # treat both cases as "no relaxed marker available".
    rlx_norms = nothing
    if haskey(saved, "relaxed_pshed_matrix")
        rlx_mat = saved["relaxed_pshed_matrix"]
        if any(!isnan, rlx_mat)
            rlx_norms = _aggregate_norms(rlx_mat)
        end
    end
    return (int = int_norms, rlx = rlx_norms,
            n_periods = saved["N_PERIODS"], source = path)
end

# ============================================================
# PLOT BUILDERS
# ============================================================
const _PARETO_FONT = (tickfontsize = 22, guidefontsize = 22,
                      titlefontsize = 26, legendfontsize = 14,
                      fontfamily = "Computer Modern")

"""
Filter NaN entries out of a sweep's (total_shed, norm_vec, alphas).
Used to drop α points where the post-hoc Palma is NaN (bot40 = 0).
"""
function _drop_nan(total_shed, norm_vec, alphas)
    mask = .!isnan.(norm_vec)
    return collect(total_shed)[mask], collect(norm_vec)[mask],
           collect(alphas)[mask]
end

"""
Build one Pareto panel containing both sweeps as lines (different
linestyles) and the three bilevel objectives as scatter markers.

`sweeps`    : Dict("palma"=>NamedTuple, "min_max"=>NamedTuple)
`bilevels`  : Dict("palma"=>NamedTuple, "efficiency"=>NamedTuple,
                   "min_max"=>NamedTuple)
`norm_field`: Symbol of the y-axis field, e.g. :Palma, :CoV, :L1, :Linf
"""
function _pareto_panel(sweeps::Dict, bilevels::Dict, norm_field::Symbol,
                       ylab::AbstractString;
                       show_legend::Bool = false,
                       legend_position::Symbol = :topleft,
                       zoom::Bool = false,
                       bilevel_alpha::Real = 1.0,
                       bilevel_label_suffix::AbstractString = "")
    # Collect xs/ys across both sweeps and bilevel points for axis limits.
    xs_all = Float64[]; ys_all = Float64[]
    sweep_data = Dict{String,Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}()
    for st in SWEEP_STYLES
        haskey(sweeps, st.key) || continue
        sw = sweeps[st.key]
        xs, ys, αs = _drop_nan(sw.total_shed, getfield(sw, norm_field), sw.alphas)
        sweep_data[st.key] = (xs, ys, αs)
        append!(xs_all, xs); append!(ys_all, ys)
    end
    # Bilevel marker positions — kept separately so zoom can be framed off
    # just the stars instead of the full sweep range.
    xs_bi = Float64[]; ys_bi = Float64[]
    for bs in BILEVEL_STYLES
        haskey(bilevels, bs.key) || continue
        bi = bilevels[bs.key]
        bx = bi.total_shed; by = getfield(bi, norm_field)
        if isfinite(bx) && isfinite(by)
            push!(xs_bi, bx); push!(ys_bi, by)
        end
    end
    append!(xs_all, xs_bi); append!(ys_all, ys_bi)
    isempty(xs_all) && error("Nothing to plot for y=$ylab.")

    if zoom && !isempty(xs_bi)
        # Frame around the bilevel markers with 50% padding so the stars sit
        # well inside the panel and any nearby sweep segment stays visible.
        xpad = 0.5 * (maximum(xs_bi) - minimum(xs_bi) + eps())
        ypad = 0.5 * (maximum(ys_bi) - minimum(ys_bi) + eps())
        xlim = (minimum(xs_bi) - xpad, maximum(xs_bi) + xpad)
        ylim = (minimum(ys_bi) - ypad, maximum(ys_bi) + ypad)
    else
        xpad = 0.20 * (maximum(xs_all) - minimum(xs_all) + eps())
        ypad = 0.28 * (maximum(ys_all) - minimum(ys_all) + eps())
        xlim = (minimum(xs_all) - xpad, maximum(xs_all) + xpad)
        ylim = (minimum(ys_all) - ypad, maximum(ys_all) + ypad)
    end
    xticks_vec = collect(range(xlim[1], xlim[2]; length = 4))

    p = plot(;
        xlabel = "total load shed (kW)", ylabel = ylab,
        xlims = xlim, ylims = ylim,
        xticks = (xticks_vec, [string(round(Int, x)) for x in xticks_vec]),
        xrotation = 30,
        grid = true, gridalpha = 0.5, gridstyle = :dot, gridlinewidth = 0.5,
        framestyle = :box,
        background_color = :white, foreground_color = :black,
        legend = show_legend ? legend_position : false,
        _PARETO_FONT...)

    for st in SWEEP_STYLES
        haskey(sweep_data, st.key) || continue
        xs, ys, _ = sweep_data[st.key]
        isempty(xs) && continue
        plot!(p, xs, ys;
            seriestype = :line,
            linestyle = st.linestyle, lw = 3.5, color = st.color,
            marker = :circle, markersize = 10,
            markerstrokecolor = st.color, markerstrokewidth = 1.0,
            label = st.label)
    end

    # α value annotations on each sweep's endpoints (first and last surviving
    # α point after NaN filtering). When a sweep has only two finite α points
    # — e.g. integer Palma where bus 6 stays energized for most α and most
    # post-hoc Palma values are NaN — both points are the endpoints and both
    # get labeled. Offset 4% of the y-range above the marker so the label
    # sits clear of the line.
    let yrange = ylim[2] - ylim[1]
        yoff = 0.04 * (yrange == 0 ? 1.0 : yrange)
        for st in SWEEP_STYLES
            haskey(sweep_data, st.key) || continue
            xs, ys, αs = sweep_data[st.key]
            isempty(xs) && continue
            idx = length(xs) == 1 ? [1] : [1, length(xs)]
            for i in idx
                annotate!(p, xs[i], ys[i] + yoff,
                    text("ν=$(round(αs[i], digits=2))", 14, :center, st.color))
            end
        end
    end

    for bs in BILEVEL_STYLES
        haskey(bilevels, bs.key) || continue
        bi = bilevels[bs.key]
        bx = bi.total_shed; by = getfield(bi, norm_field)
        (isfinite(bx) && isfinite(by)) || continue
        scatter!(p, [bx], [by];
            marker = bs.marker, markersize = 26, color = bs.color,
            markerstrokecolor = :black, markerstrokewidth = 1.5,
            alpha = bilevel_alpha,
            label = bs.label * bilevel_label_suffix)
    end

    return p
end

function build_summary_figure(sweeps::Dict, bilevels::Dict, out_path::String;
                              zoom::Bool = false,
                              bilevel_alpha::Real = 1.0,
                              bilevel_label_suffix::AbstractString = "")
    p_l1    = _pareto_panel(sweeps, bilevels, :L1,
        raw"$\ell_1$ norm of load shed (kW)"; show_legend = true, zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    p_linf  = _pareto_panel(sweeps, bilevels, :Linf,
        raw"$\ell_\infty$ norm of load shed (kW)"; zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    p_palma = _pareto_panel(sweeps, bilevels, :Palma,
        "Palma ratio of load shed (unitless)"; zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    p_cov   = _pareto_panel(sweeps, bilevels, :CoV,
        "CoV of load shed (unitless)"; zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    fig = plot(p_l1, p_linf, p_palma, p_cov;
        layout = (1, 4),
        size = (2800, 720),
        plot_titlefontsize = 26,
        left_margin = 24Plots.mm, right_margin = 14Plots.mm,
        top_margin = 12Plots.mm, bottom_margin = 32Plots.mm)
    savefig(fig, out_path)
    savefig(fig, replace(out_path, r"\.svg$"i => ".pdf"))
    return fig
end

function build_palma_figure(sweeps::Dict, bilevels::Dict, out_path::String;
                               zoom::Bool = false,
                               legend_position::Symbol = :topright,
                               bilevel_alpha::Real = 1.0,
                               bilevel_label_suffix::AbstractString = "")
    panel = _pareto_panel(sweeps, bilevels, :Palma,
        "Palma ratio of load shed (unitless)";
        show_legend = true, legend_position = legend_position, zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    fig = plot(panel;
        size = (900, 760),
        left_margin = 13Plots.mm, right_margin = 16Plots.mm,
        top_margin = 12Plots.mm, bottom_margin = 10Plots.mm)
    savefig(fig, out_path)
    savefig(fig, replace(out_path, r"\.svg$"i => ".pdf"))
    return fig
end

function build_cov_figure(sweeps::Dict, bilevels::Dict, out_path::String;
                          zoom::Bool = false,
                          legend_position::Symbol = :topright,
                          bilevel_alpha::Real = 1.0,
                          bilevel_label_suffix::AbstractString = "")
    panel = _pareto_panel(sweeps, bilevels, :CoV,
        "CoV of load shed (unitless)";
        show_legend = true, legend_position = legend_position, zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    fig = plot(panel;
        size = (900, 760),
        left_margin = 13Plots.mm, right_margin = 16Plots.mm,
        top_margin = 12Plots.mm, bottom_margin = 10Plots.mm)
    savefig(fig, out_path)
    savefig(fig, replace(out_path, r"\.svg$"i => ".pdf"))
    return fig
end

function build_gini_figure(sweeps::Dict, bilevels::Dict, out_path::String;
                           zoom::Bool = false,
                           legend_position::Symbol = :topright,
                           bilevel_alpha::Real = 1.0,
                           bilevel_label_suffix::AbstractString = "")
    panel = _pareto_panel(sweeps, bilevels, :Gini,
        "Gini index of load shed (unitless)";
        show_legend = true, legend_position = legend_position, zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    fig = plot(panel;
        size = (900, 760),
        left_margin = 13Plots.mm, right_margin = 16Plots.mm,
        top_margin = 12Plots.mm, bottom_margin = 10Plots.mm)
    savefig(fig, out_path)
    savefig(fig, replace(out_path, r"\.svg$"i => ".pdf"))
    return fig
end

"Three-panel summary: Palma | Gini | CoV (the fairness metrics requested for the T=3 paper figure)."
function build_palma_gini_cov_figure(sweeps::Dict, bilevels::Dict, out_path::String;
                                      zoom::Bool = false,
                                      bilevel_alpha::Real = 1.0,
                                      bilevel_label_suffix::AbstractString = "")
    p_palma = _pareto_panel(sweeps, bilevels, :Palma,
        "Palma ratio of load shed (unitless)"; show_legend = true, zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    p_gini  = _pareto_panel(sweeps, bilevels, :Gini,
        "Gini index of load shed (unitless)"; zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    p_cov   = _pareto_panel(sweeps, bilevels, :CoV,
        "CoV of load shed (unitless)"; zoom = zoom,
        bilevel_alpha = bilevel_alpha, bilevel_label_suffix = bilevel_label_suffix)
    fig = plot(p_palma, p_gini, p_cov;
        layout = (1, 3),
        size = (2100, 720),
        plot_titlefontsize = 26,
        left_margin = 24Plots.mm, right_margin = 14Plots.mm,
        top_margin = 12Plots.mm, bottom_margin = 32Plots.mm)
    savefig(fig, out_path)
    savefig(fig, replace(out_path, r"\.svg$"i => ".pdf"))
    return fig
end

# ============================================================
# RUN
# ============================================================
out_dir = joinpath(RESULTS_ROOT, Dates.format(now(), "yyyy-mm-dd"),
                   "post_hoc_fairness")
mkpath(out_dir)

sweeps   = Dict{String,NamedTuple}()
bilevels = Dict{String,NamedTuple}()

_fmt_mtime(p) = Dates.format(unix2datetime(mtime(p)), "yyyy-mm-dd HH:MM")

for st in SWEEP_STYLES
    path = _trade_off_jld2(st.fair_func, st.relaxed)
    if path === nothing
        @warn "[$(st.key)] no single-level trade-off JLD2 found — skipping sweep"
        continue
    end
    println("  sweep $(st.key) [$(_fmt_mtime(path))]: $(relpath(path, RESULTS_ROOT))")
    sweeps[st.key] = load_trade_off_curve(path)
end

# Trim the α=1 tail of the relaxed Palma sweep — the endpoint collapses
# (no bot40 mass / huge top10), pulling the y-axis to where the rest of the
# sweep and the bilevel stars become unreadable.
function _trim_last(sw::NamedTuple)
    n = length(sw.alphas)
    n < 2 && return sw
    keep = 1:(n-1)
    return (alphas = sw.alphas[keep], total_shed = sw.total_shed[keep],
            L1 = sw.L1[keep], L2 = sw.L2[keep], Linf = sw.Linf[keep],
            CoV = sw.CoV[keep], Palma = sw.Palma[keep], Gini = sw.Gini[keep],
            n_periods = sw.n_periods, source = sw.source)
end
if haskey(sweeps, "palma_relaxed")
    n_before = length(sweeps["palma_relaxed"].alphas)
    sweeps["palma_relaxed"] = _trim_last(sweeps["palma_relaxed"])
    println("  trimmed palma_relaxed: $n_before → $(length(sweeps["palma_relaxed"].alphas)) α points (dropped α=1 endpoint)")
end

if SHOW_BILEVEL
    for bs in BILEVEL_STYLES
        PALMA_ONLY && bs.key != "palma" && continue   # palma-only: drop gini/min_max markers
        override = get(BILEVEL_OVERRIDES, bs.key, nothing)
        path = if override !== nothing && isfile(override)
            println("  bilevel $(bs.key): OVERRIDE active → $(relpath(override, RESULTS_ROOT))")
            override
        else
            _bilevel_jld2(bs.key)
        end
        if path === nothing
            @warn "[$(bs.key)] no bilevel JLD2 found — skipping marker"
            continue
        end
        println("  bilevel $(bs.key) [$(_fmt_mtime(path))]: $(relpath(path, RESULTS_ROOT))")
        bilevels[bs.key] = load_bilevel_point(path)
    end
end

isempty(sweeps) && error("No single-level sweeps loaded — nothing to plot.")

# Sanity check: warn if T mismatches across sweeps or bilevels.
periods = Set{Int}()
for sw in values(sweeps); push!(periods, sw.n_periods); end
for bi in values(bilevels); push!(periods, bi.n_periods); end
if length(periods) > 1
    @warn "T mismatch across loaded results: $(collect(periods)). " *
          "Pareto comparison is across different demand profiles — " *
          "interpret with care."
end

suffix = SHOW_BILEVEL ? "" : "_no_bilevel"

"Flatten loaded bilevel points to just the (int|rlx) sub-tuple, dropping
 any fair_funcs that have no marker for this variant. Returns a Dict
 shaped the same as the old `bilevels` so `_pareto_panel` works unchanged."
function _bilevels_for_variant(want_relaxed::Bool)
    field = want_relaxed ? :rlx : :int
    out = Dict{String,NamedTuple}()
    for (k, bi) in bilevels
        sub = getfield(bi, field)
        sub === nothing && continue
        out[k] = merge(sub, (n_periods = bi.n_periods,))
    end
    return out
end

function _render_variant(variant_tag::String, want_relaxed::Bool)
    keep = Set(st.key for st in SWEEP_STYLES if st.relaxed == want_relaxed)
    sweeps_subset = Dict(k => v for (k, v) in sweeps if k in keep)
    if isempty(sweeps_subset)
        @warn "[$variant_tag] no sweeps loaded — skipping figure set"
        return nothing
    end
    bilevels_subset = _bilevels_for_variant(want_relaxed)
    if SHOW_BILEVEL && isempty(bilevels_subset)
        @warn "[$variant_tag] no bilevel markers available — figures will show sweeps only"
    end

    base = "pareto_$(CASE_KEY)_$(PSHED_TYPE)_$(variant_tag)$(suffix)$(PALMA_ONLY ? "_palmaonly" : "")"
    summary_path  = joinpath(out_dir, "$(base).svg")
    palma_path = joinpath(out_dir, "$(base)_palma.svg")
    gini_path  = joinpath(out_dir, "$(base)_gini.svg")
    cov_path      = joinpath(out_dir, "$(base)_cov.svg")
    pgc_path      = joinpath(out_dir, "$(base)_palma_gini_cov.svg")
    summary_zoom_path  = joinpath(out_dir, "$(base)_zoom.svg")
    palma_zoom_path = joinpath(out_dir, "$(base)_palma_zoom.svg")
    gini_zoom_path = joinpath(out_dir, "$(base)_gini_zoom.svg")
    cov_zoom_path      = joinpath(out_dir, "$(base)_cov_zoom.svg")
    pgc_zoom_path      = joinpath(out_dir, "$(base)_palma_gini_cov_zoom.svg")

    # Relaxed sweeps + markers sit near the low end of total_shed, leaving
    # the right side of the panel free; integer sits near the high end with
    # the left free. Anchor the legend opposite the data cluster.
    # Legend entries get an explicit "(integer)" / "(relaxed)" suffix so a
    # reader looking at both figure sets can tell them apart.
    legpos = want_relaxed ? :topleft : :topright
    bi_α   = 1.0
    bi_lbl = want_relaxed ? " (relaxed)" : " (integer)"
    fig_summary  = build_summary_figure(sweeps_subset, bilevels_subset, summary_path;
        bilevel_alpha = bi_α, bilevel_label_suffix = bi_lbl)
    build_palma_figure(sweeps_subset, bilevels_subset, palma_path;
        legend_position = legpos, bilevel_alpha = bi_α,
        bilevel_label_suffix = bi_lbl)
    build_gini_figure(sweeps_subset, bilevels_subset, gini_path;
        legend_position = legpos, bilevel_alpha = bi_α,
        bilevel_label_suffix = bi_lbl)
    build_cov_figure(sweeps_subset, bilevels_subset, cov_path;
        legend_position = legpos, bilevel_alpha = bi_α,
        bilevel_label_suffix = bi_lbl)
    build_palma_gini_cov_figure(sweeps_subset, bilevels_subset, pgc_path;
        bilevel_alpha = bi_α, bilevel_label_suffix = bi_lbl)
    # Zoomed variants — same data, axes framed around the bilevel stars.
    # Skip if no bilevel markers (no anchor to zoom to), or if PLOT_ZOOM is off.
    if PLOT_ZOOM && !isempty(bilevels_subset)
        build_summary_figure(sweeps_subset, bilevels_subset, summary_zoom_path;
            zoom = true, bilevel_alpha = bi_α, bilevel_label_suffix = bi_lbl)
        build_palma_figure(sweeps_subset, bilevels_subset, palma_zoom_path;
            zoom = true, legend_position = legpos, bilevel_alpha = bi_α,
            bilevel_label_suffix = bi_lbl)
        build_gini_figure(sweeps_subset, bilevels_subset, gini_zoom_path;
            zoom = true, legend_position = legpos, bilevel_alpha = bi_α,
            bilevel_label_suffix = bi_lbl)
        build_cov_figure(sweeps_subset, bilevels_subset, cov_zoom_path;
            zoom = true, legend_position = legpos, bilevel_alpha = bi_α,
            bilevel_label_suffix = bi_lbl)
        build_palma_gini_cov_figure(sweeps_subset, bilevels_subset, pgc_zoom_path;
            zoom = true, bilevel_alpha = bi_α, bilevel_label_suffix = bi_lbl)
    end

    println("\n[$variant_tag] figures written to:")
    println("  → $summary_path")
    println("  → $palma_path")
    println("  → $gini_path")
    println("  → $cov_path")
    println("  → $pgc_path  (3-panel: Palma | Gini | CoV)")
    if PLOT_ZOOM && !isempty(bilevels_subset)
        println("  → $summary_zoom_path")
        println("  → $palma_zoom_path")
        println("  → $gini_zoom_path")
        println("  → $cov_zoom_path")
        println("  → $pgc_zoom_path")
    end
    if SHOW_BILEVEL
        for bs in BILEVEL_STYLES
            haskey(bilevels_subset, bs.key) || continue
            bi = bilevels_subset[bs.key]
            println("    bilevel $(bs.key): total=$(round(bi.total_shed, digits=2))  " *
                    "L∞=$(round(bi.Linf, digits=2))  " *
                    "Palma=$(isnan(bi.Palma) ? "NaN" : string(round(bi.Palma, digits=3)))  " *
                    "Gini=$(round(bi.Gini, digits=3))  " *
                    "CoV=$(round(bi.CoV, digits=3))")
        end
    end
    return fig_summary
end

fig_integer = _render_variant("integer", false)
fig_relaxed = _render_variant("relaxed", true)
fig_integer === nothing || display(fig_integer)
fig_relaxed === nothing || display(fig_relaxed)
