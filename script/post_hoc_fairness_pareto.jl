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

@assert haskey(CASE_TAGS, CASE_KEY) "Unknown CASE_KEY=$CASE_KEY"
tags = CASE_TAGS[CASE_KEY]
RESULTS_ROOT = joinpath(@__DIR__, "../results")

# Sweeps to overlay as single-level lines and bilevel objectives to
# overlay as scatter markers. Linestyle distinguishes fair-func
# (palma vs min-max); color shade distinguishes integer (dark) vs
# relaxed (light) so the four lines stay legible on one panel.
const SWEEP_STYLES = [
    (key = "palma_integer",   label = "Single-level Palma (integer)",
     fair_func = "palma",   relaxed = false,
     linestyle = :solid, color = RGB(0.20, 0.40, 0.85)),
    (key = "palma_relaxed",   label = "Single-level Palma (relaxed)",
     fair_func = "palma",   relaxed = true,
     linestyle = :solid, color = RGB(0.55, 0.70, 0.95)),
    (key = "min_max_integer", label = "Single-level min-max (integer)",
     fair_func = "min_max", relaxed = false,
     linestyle = :dash,  color = RGB(0.85, 0.30, 0.20)),
    (key = "min_max_relaxed", label = "Single-level min-max (relaxed)",
     fair_func = "min_max", relaxed = true,
     linestyle = :dash,  color = RGB(0.95, 0.65, 0.55)),
]

const BILEVEL_STYLES = [
    (key = "palma",      label = "Bi-level Palma",      marker = :star5,
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
                               CoV = NaN, Palma = NaN)
    m = mean(finite)
    s = length(finite) > 1 ? std(finite) : 0.0
    return (
        L1    = norm(finite, 1),
        L2    = norm(finite, 2),
        Linf  = norm(finite, Inf),
        CoV   = m > 1e-9 ? s / m : NaN,
        Palma = _palma_ratio_safe(finite),
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
    CoV   = zeros(n_α); Palma = zeros(n_α)
    for i in 1:n_α
        nm = shed_norms(collect(per_load_agg[i, :]))
        L1[i] = nm.L1; L2[i] = nm.L2; Linf[i] = nm.Linf
        CoV[i] = nm.CoV; Palma[i] = nm.Palma
    end
    total_shed = L1
    return (alphas = alphas, total_shed = total_shed,
            L1 = L1, L2 = L2, Linf = Linf, CoV = CoV, Palma = Palma,
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
            Linf = nm.Linf, CoV = nm.CoV, Palma = nm.Palma)
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
const _PARETO_FONT = (tickfontsize = 25, guidefontsize = 29,
                      titlefontsize = 32, legendfontsize = 20,
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
                       zoom::Bool = false)
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

    for bs in BILEVEL_STYLES
        haskey(bilevels, bs.key) || continue
        bi = bilevels[bs.key]
        bx = bi.total_shed; by = getfield(bi, norm_field)
        (isfinite(bx) && isfinite(by)) || continue
        scatter!(p, [bx], [by];
            marker = bs.marker, markersize = 26, color = bs.color,
            markerstrokecolor = :black, markerstrokewidth = 1.5,
            label = bs.label)
    end

    return p
end

function build_summary_figure(sweeps::Dict, bilevels::Dict, out_path::String;
                              zoom::Bool = false)
    p_l1    = _pareto_panel(sweeps, bilevels, :L1,
        raw"$\ell_1$ norm of load shed (kW)"; show_legend = true, zoom = zoom)
    p_linf  = _pareto_panel(sweeps, bilevels, :Linf,
        raw"$\ell_\infty$ norm of load shed (kW)"; zoom = zoom)
    p_palma = _pareto_panel(sweeps, bilevels, :Palma,
        "Palma ratio of load shed (unitless)"; zoom = zoom)
    p_cov   = _pareto_panel(sweeps, bilevels, :CoV,
        "CoV of load shed (unitless)"; zoom = zoom)
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
                               zoom::Bool = false)
    panel = _pareto_panel(sweeps, bilevels, :Palma,
        "Palma ratio of load shed (unitless)";
        show_legend = true, legend_position = :topright, zoom = zoom)
    fig = plot(panel;
        size = (900, 760),
        left_margin = 26Plots.mm, right_margin = 16Plots.mm,
        top_margin = 12Plots.mm, bottom_margin = 32Plots.mm)
    savefig(fig, out_path)
    savefig(fig, replace(out_path, r"\.svg$"i => ".pdf"))
    return fig
end

function build_cov_figure(sweeps::Dict, bilevels::Dict, out_path::String;
                          zoom::Bool = false)
    panel = _pareto_panel(sweeps, bilevels, :CoV,
        "CoV of load shed (unitless)";
        show_legend = true, legend_position = :topright, zoom = zoom)
    fig = plot(panel;
        size = (900, 760),
        left_margin = 26Plots.mm, right_margin = 16Plots.mm,
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
            CoV = sw.CoV[keep], Palma = sw.Palma[keep],
            n_periods = sw.n_periods, source = sw.source)
end
if haskey(sweeps, "palma_relaxed")
    n_before = length(sweeps["palma_relaxed"].alphas)
    sweeps["palma_relaxed"] = _trim_last(sweeps["palma_relaxed"])
    println("  trimmed palma_relaxed: $n_before → $(length(sweeps["palma_relaxed"].alphas)) α points (dropped α=1 endpoint)")
end

if SHOW_BILEVEL
    for bs in BILEVEL_STYLES
        path = _bilevel_jld2(bs.key)
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

    base = "pareto_$(CASE_KEY)_$(PSHED_TYPE)_$(variant_tag)$(suffix)"
    summary_path  = joinpath(out_dir, "$(base).svg")
    palma_path = joinpath(out_dir, "$(base)_palma.svg")
    cov_path      = joinpath(out_dir, "$(base)_cov.svg")
    summary_zoom_path  = joinpath(out_dir, "$(base)_zoom.svg")
    palma_zoom_path = joinpath(out_dir, "$(base)_palma_zoom.svg")
    cov_zoom_path      = joinpath(out_dir, "$(base)_cov_zoom.svg")

    fig_summary  = build_summary_figure(sweeps_subset, bilevels_subset, summary_path)
    build_palma_figure(sweeps_subset, bilevels_subset, palma_path)
    build_cov_figure(sweeps_subset, bilevels_subset, cov_path)
    # Zoomed variants — same data, axes framed around the bilevel stars.
    # Skip if no bilevel markers (no anchor to zoom to).
    if !isempty(bilevels_subset)
        build_summary_figure(sweeps_subset, bilevels_subset, summary_zoom_path; zoom = true)
        build_palma_figure(sweeps_subset, bilevels_subset, palma_zoom_path; zoom = true)
        build_cov_figure(sweeps_subset, bilevels_subset, cov_zoom_path; zoom = true)
    end

    println("\n[$variant_tag] figures written to:")
    println("  → $summary_path")
    println("  → $palma_path")
    println("  → $cov_path")
    if !isempty(bilevels_subset)
        println("  → $summary_zoom_path")
        println("  → $palma_zoom_path")
        println("  → $cov_zoom_path")
    end
    if SHOW_BILEVEL
        for bs in BILEVEL_STYLES
            haskey(bilevels_subset, bs.key) || continue
            bi = bilevels_subset[bs.key]
            println("    bilevel $(bs.key): total=$(round(bi.total_shed, digits=2))  " *
                    "L∞=$(round(bi.Linf, digits=2))  " *
                    "Palma=$(isnan(bi.Palma) ? "NaN" : string(round(bi.Palma, digits=3)))  " *
                    "CoV=$(round(bi.CoV, digits=3))")
        end
    end
    return fig_summary
end

fig_integer = _render_variant("integer", false)
fig_relaxed = _render_variant("relaxed", true)
fig_integer === nothing || display(fig_integer)
fig_relaxed === nothing || display(fig_relaxed)
