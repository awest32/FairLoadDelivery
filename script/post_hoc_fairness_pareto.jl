"""
    Post-hoc fairness-vs-efficiency Pareto plots, single-level vs bilevel.

    Intent
    ------
    The bilevel framework was proposed as a scalable alternative to the
    single-level multi-period MIP for the fairness objectives (min-max,
    Palma): T·N small forward solves + Jacobian-driven upper-level steps,
    instead of one large T-period MIP. The expectation going in was that
    this scalability would buy us at least *parity* on solution quality —
    the bilevel descent converging to a competitive point on the
    single-level Pareto curve, while remaining tractable as T or N grow.

    On case6 (T=24, 9 loads), the single-level α-sweep is still tractable
    and provides a clean Pareto frontier in (total_shed, fairness-norm)
    space. This script overlays the bilevel result as a big ★ on each
    panel, so the position of the bilevel point relative to the
    single-level curve is visible directly. If the ★ sits below/left of
    the curve, the bilevel is finding something the single-level can't —
    a genuine win. If the ★ sits above/right, the bilevel is Pareto-
    dominated, and its only remaining selling point is scalability.

    Caveat on scalability: with DiffOpt-based gradients, per-iteration
    time is dominated by T·N forward solves with Jacobian extraction at
    each, and the upper-level gradients are prone to corner-collapse
    near {0, pd} integer solutions (project memory:
    `project_bilevel_bottleneck_diffopt_gradients.md`). So even the
    "scales better" claim needs measured runtime evidence, not asymptotic
    counting.

    What this script produces
    -------------------------
      * Three figures per fair_func ∈ {min_max, palma}, all with
        total_shed on the x-axis:
          1. Headline single-panel: y = post-hoc palma ratio (palma) or
             y = L∞ (min_max). Filename suffix `_headline`.
          2. CoV single-panel: y = CoV. Filename suffix `_cov`.
          3. 4-up summary: [L1, L∞, palma, CoV]. L2 dropped, post-hoc
             palma added. No suffix.
      * Single-level α-sweep is a line + markers; bi-level (when
        SHOW_BILEVEL=true) is a single ★ overlay.
      * SVG + PDF outputs at
        results/<today>/post_hoc_fairness/pareto_<fair_func>_<case>_<pshed>[_suffix].svg

    Usage
    -----
        julia --project=. script/post_hoc_fairness_pareto.jl
        # or in the REPL:
        include("script/post_hoc_fairness_pareto.jl")

    Override CASE_KEY / PSHED_TYPE at the top before include.
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
# fair_funcs to plot. "efficiency" is excluded — the trade-off pipeline only
# produces a single α=0 point there (no Pareto curve), and the bilevel
# efficiency result matches it exactly.
FAIR_FUNCS = ["min_max", "palma"]
# When false, skip loading the bilevel JLD2 and omit the ★ overlay so the
# output is the trade-off Pareto front alone. Filename gets a `_no_bilevel`
# suffix so it doesn't overwrite the bilevel-comparison version.
SHOW_BILEVEL = true

@assert haskey(CASE_TAGS, CASE_KEY) "Unknown CASE_KEY=$CASE_KEY"
tags = CASE_TAGS[CASE_KEY]
RESULTS_ROOT = joinpath(@__DIR__, "../results")

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
    return sort(candidates; rev = true)[1]
end

function _trade_off_jld2(fair_func::String)
    if fair_func == "min_max"
        return _latest_matching(d -> [joinpath(RESULTS_ROOT, d, "trade_off_mn",
            "min_max_trade_off_mn_$(tags.trade_off)_$(PSHED_TYPE).jld2")])
    elseif fair_func == "palma"
        # palma sweeps have historically lived in palma_trade_off_mn/ AND in
        # variant folders like palma_trade_off_mn_8_periods/ — match any
        # subdir whose name starts with `palma_trade_off_mn`.
        return _latest_matching(d -> begin
            date_dir = joinpath(RESULTS_ROOT, d)
            isdir(date_dir) || return String[]
            [joinpath(date_dir, sub,
                      "palma_sweep_mn_$(tags.trade_off)_$(PSHED_TYPE).jld2")
             for sub in readdir(date_dir)
             if isdir(joinpath(date_dir, sub)) &&
                startswith(sub, "palma_trade_off_mn")]
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
"""
Palma ratio on a non-negative per-load vector. Mirrors
`src/implementation/other_fair_funcs.jl::palma_ratio` (top-10% / bottom-40%
of sorted entries), but guards against an all-zero bottom 40% — returns NaN
instead of Inf so the curve has a gap rather than an off-scale spike.
"""
function _palma_ratio_safe(x::AbstractVector{<:Real})
    n = length(x)
    n < 3 && return NaN              # need ≥1 in bottom 40% and top 10%
    sorted_x = sort(collect(x))
    top10 = sum(sorted_x[ceil(Int, 0.9n):end])
    bot40 = sum(sorted_x[1:floor(Int, 0.4n)])
    return bot40 > 1e-9 ? top10 / bot40 : NaN
end

"""
Norms on per-load shed (L1, L2, L∞, CoV) plus the post-hoc Palma ratio on
**shed**. Note: the single-level palma problem and bilevel palma upper
level both *optimize* served-Palma (top10/bot40 of pd-pshed). We instead
report shed-Palma post-hoc because:
  * served-Palma becomes NaN as soon as ≥3 of 9 loads are fully shed
    (bot40 of served = 0), which happens both at single-level α=1 and at
    the bilevel's concentrated-shed operating point — flattening the very
    contrast we want to show.
  * shed-Palma directly answers "how unevenly was the shedding distributed
    across loads?" — which is the fairness question the paper asks.
The tradeoff is that shed-Palma is NaN at low single-level α (almost no
loads shed → bot40 of shed = 0); those α points drop from the curve.
"""
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
    per_load_agg = saved["per_load_agg"]   # alpha × load
    alphas       = saved["alphas"]
    n_α          = length(alphas)
    L1    = zeros(n_α); L2 = zeros(n_α); Linf = zeros(n_α)
    CoV   = zeros(n_α); Palma = zeros(n_α)
    for i in 1:n_α
        nm = shed_norms(collect(per_load_agg[i, :]))
        L1[i] = nm.L1; L2[i] = nm.L2; Linf[i] = nm.Linf
        CoV[i] = nm.CoV; Palma[i] = nm.Palma
    end
    # x-axis is total shed (= L1 for non-negative vectors). Identical to L1
    # here, but kept as a separate vector to mirror the existing convention
    # in min_max_trade_off_mn.jl / palma_trade_off_mn.jl.
    total_shed = L1
    return (alphas = alphas, total_shed = total_shed,
            L1 = L1, L2 = L2, Linf = Linf, CoV = CoV, Palma = Palma,
            n_periods = saved["N_PERIODS"], source = path)
end

function load_bilevel_point(path::String)
    saved        = JLD2.load(path)
    pshed_matrix = saved["pshed_matrix"]   # period × load (NaN where unsolved)
    n_loads = size(pshed_matrix, 2)
    v = zeros(n_loads)
    for t in axes(pshed_matrix, 1), j in 1:n_loads
        x = pshed_matrix[t, j]; isnan(x) || (v[j] += x)
    end
    nm = shed_norms(v)
    return (total_shed = nm.L1, L1 = nm.L1, L2 = nm.L2,
            Linf = nm.Linf, CoV = nm.CoV, Palma = nm.Palma,
            n_periods = saved["N_PERIODS"], source = path)
end

# ============================================================
# PLOT BUILDERS
# ============================================================
const _PARETO_FONT = (tickfontsize = 25, guidefontsize = 29,
                      titlefontsize = 32, legendfontsize = 23,
                      fontfamily = "Computer Modern")

const _SWEEP_COLOR = RGB(0.20, 0.40, 0.85)   # blue

function _pareto_panel(total_shed::AbstractVector, norm_vec::AbstractVector,
                       alphas::AbstractVector, ylab::AbstractString,
                       bilevel_x::Union{Real,Nothing},
                       bilevel_y::Union{Real,Nothing};
                       show_legend::Bool = false,
                       legend_position::Symbol = :topleft)
    # Drop α points where the y-value is NaN (e.g. post-hoc Palma at α=0
    # when most loads aren't shed → bottom-40% sum is 0).
    mask        = .!isnan.(norm_vec)
    total_shed  = collect(total_shed)[mask]
    norm_vec    = collect(norm_vec)[mask]
    alphas      = collect(alphas)[mask]
    isempty(total_shed) && error("All trade-off points dropped for y=$ylab — nothing to plot.")
    bx_finite   = bilevel_x !== nothing && isfinite(bilevel_x)
    by_finite   = bilevel_y !== nothing && isfinite(bilevel_y)
    has_bilevel = bx_finite && by_finite

    xs_all = has_bilevel ? vcat(total_shed, bilevel_x) : total_shed
    ys_all = has_bilevel ? vcat(norm_vec,  bilevel_y) : norm_vec
    xpad   = 0.20 * (maximum(xs_all) - minimum(xs_all) + eps())
    ypad   = 0.28 * (maximum(ys_all) - minimum(ys_all) + eps())
    xlim   = (minimum(xs_all) - xpad, maximum(xs_all) + xpad)
    ylim   = (minimum(ys_all) - ypad, maximum(ys_all) + ypad)

    xticks_vec = collect(range(xlim[1], xlim[2]; length = 4))

    p = plot(total_shed, norm_vec;
        seriestype = :line,
        marker = :rect, markersize = 12,
        color = _SWEEP_COLOR, lc = _SWEEP_COLOR, lw = 3.5,
        markerstrokecolor = _SWEEP_COLOR, markerstrokewidth = 1.0,
        label = "Single-level",
        xlabel = "total load shed (kW)", ylabel = ylab,
        xlims = xlim, ylims = ylim,
        xticks = (xticks_vec, [string(round(Int, x)) for x in xticks_vec]),
        xrotation = 30,
        grid = true, gridalpha = 0.5, gridstyle = :dot, gridlinewidth = 0.5,
        framestyle = :box,
        background_color = :white, foreground_color = :black,
        legend = show_legend ? legend_position : false,
        _PARETO_FONT...)

    if bilevel_x !== nothing && bilevel_y !== nothing
        scatter!(p, [bilevel_x], [bilevel_y];
            marker = :star5, markersize = 28, color = :black,
            markerstrokecolor = :black, markerstrokewidth = 1.0,
            label = "Bi-level")
    end

    # β-style endpoint labels (α minimum and maximum), placed where
    # they're least likely to collide with the curve.
    if length(alphas) ≥ 2
        i_lo = argmin(alphas)
        i_hi = argmax(alphas)
        # If α=0 sits above α=1 on the y-axis (e.g. CoV), put the α=0 label
        # above its point and the α=1 label below; otherwise vice-versa.
        inverted = norm_vec[i_lo] > norm_vec[i_hi]
        hi_valign = inverted ? :top    : :bottom
        # α=0 label: shifted right of the marker so it clears any cluster.
        dx = 0.07 * (maximum(xs_all) - minimum(xs_all))
        dy = 0.04 * (maximum(ys_all) - minimum(ys_all))
        hi_dy = hi_valign == :top ? -dy : dy
        annotate!(p, total_shed[i_lo] + dx, norm_vec[i_lo],
            Plots.text("α=$(round(alphas[i_lo]; digits=2))",
                       20, :left, :vcenter, _SWEEP_COLOR))
        annotate!(p, total_shed[i_hi], norm_vec[i_hi] + hi_dy,
            Plots.text("α=$(round(alphas[i_hi]; digits=2))",
                       20, :hcenter, hi_valign, _SWEEP_COLOR))
    end

    return p
end

function build_figure(fair_func::String,
                      trade_off::NamedTuple,
                      bilevel::Union{NamedTuple,Nothing},
                      out_path::String)
    bx(field) = bilevel === nothing ? nothing : getfield(bilevel, field)
    p_l1    = _pareto_panel(trade_off.total_shed, trade_off.L1,
        trade_off.alphas, raw"$\ell_1$ norm of load shed (kW)",
        bx(:total_shed), bx(:L1); show_legend = true)
    p_linf  = _pareto_panel(trade_off.total_shed, trade_off.Linf,
        trade_off.alphas, raw"$\ell_\infty$ norm of load shed (kW)",
        bx(:total_shed), bx(:Linf))
    p_palma = _pareto_panel(trade_off.total_shed, trade_off.Palma,
        trade_off.alphas, "Palma ratio of load shed (unitless)",
        bx(:total_shed), bx(:Palma))
    p_cov   = _pareto_panel(trade_off.total_shed, trade_off.CoV,
        trade_off.alphas, "CoV of load shed (unitless)",
        bx(:total_shed), bx(:CoV))

    pt = bilevel === nothing ?
        " (T=$(trade_off.n_periods))" :
        (trade_off.n_periods == bilevel.n_periods ?
            " (T=$(trade_off.n_periods))" :
            " (single-level T=$(trade_off.n_periods), bilevel T=$(bilevel.n_periods))")

    fig = plot(p_l1, p_linf, p_palma, p_cov;
        layout = (1, 4),
        size = (2800, 720),
        #plot_title = "Pareto: single-level α-sweep vs bilevel — $fair_func / $(CASE_KEY)$pt",
        plot_titlefontsize = 26,
        left_margin = 24Plots.mm, right_margin = 14Plots.mm,
        top_margin = 12Plots.mm, bottom_margin = 32Plots.mm)

    savefig(fig, out_path)
    pdf_path = replace(out_path, r"\.svg$"i => ".pdf")
    savefig(fig, pdf_path)
    return fig
end

"""
Headline single-panel: post-hoc Palma ratio (for `palma`) or L∞ (for `min_max`)
vs total load shed — the fair-func-specific metric shown on its own.
"""
function build_headline_figure(fair_func::String,
                               trade_off::NamedTuple,
                               bilevel::Union{NamedTuple,Nothing},
                               out_path::String)
    bx(field) = bilevel === nothing ? nothing : getfield(bilevel, field)
    if fair_func == "palma"
        # Palma curve descends left-to-right (high at low α, ≈1.5 at α=1) and
        # the bilevel ★ sits in the upper-left quadrant — put the legend
        # top-right so it doesn't overlap the data.
        panel = _pareto_panel(trade_off.total_shed, trade_off.Palma,
            trade_off.alphas, "Palma ratio of load shed (unitless)",
            bx(:total_shed), bx(:Palma);
            show_legend = true, legend_position = :topright)
    elseif fair_func == "min_max"
        panel = _pareto_panel(trade_off.total_shed, trade_off.Linf,
            trade_off.alphas, "min-max of load shed (kW)",
            bx(:total_shed), bx(:Linf); show_legend = true)
    else
        error("build_headline_figure: unsupported fair_func=$fair_func")
    end
    fig = plot(panel;
        size = (900, 760),
        left_margin = 26Plots.mm, right_margin = 16Plots.mm,
        top_margin = 12Plots.mm, bottom_margin = 32Plots.mm)
    savefig(fig, out_path)
    savefig(fig, replace(out_path, r"\.svg$"i => ".pdf"))
    return fig
end

"""
Standalone CoV-vs-total-shed panel, separate from the 4-up summary. CoV
descends left-to-right for both fair_funcs (more shed → more evenly spread),
so the legend goes top-right for either sweep.
"""
function build_cov_figure(fair_func::String,
                          trade_off::NamedTuple,
                          bilevel::Union{NamedTuple,Nothing},
                          out_path::String)
    bx(field) = bilevel === nothing ? nothing : getfield(bilevel, field)
    panel = _pareto_panel(trade_off.total_shed, trade_off.CoV,
        trade_off.alphas, "CoV of load shed (unitless)",
        bx(:total_shed), bx(:CoV);
        show_legend = true, legend_position = :topright)
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

for fair_func in FAIR_FUNCS
    println("\n=== $fair_func ===")
    to_path = _trade_off_jld2(fair_func)
    bi_path = SHOW_BILEVEL ? _bilevel_jld2(fair_func) : nothing

    if to_path === nothing
        @warn "[$fair_func] no single-level trade-off JLD2 found — skipping Pareto plot"
        continue
    end
    if SHOW_BILEVEL && bi_path === nothing
        @warn "[$fair_func] no bilevel JLD2 found — skipping Pareto plot"
        continue
    end

    println("  single-level: $(relpath(to_path, RESULTS_ROOT))")
    SHOW_BILEVEL && println("  bilevel:      $(relpath(bi_path, RESULTS_ROOT))")

    to = load_trade_off_curve(to_path)
    bi = SHOW_BILEVEL ? load_bilevel_point(bi_path) : nothing

    if SHOW_BILEVEL && to.n_periods != bi.n_periods
        @warn "T mismatch for $fair_func: single-level=$(to.n_periods) " *
              "vs bilevel=$(bi.n_periods). Pareto comparison is across " *
              "different demand profiles — interpret with care."
    end

    suffix = SHOW_BILEVEL ? "" : "_no_bilevel"
    base = "pareto_$(fair_func)_$(CASE_KEY)_$(PSHED_TYPE)$(suffix)"
    summary_path  = joinpath(out_dir, "$(base).svg")
    headline_path = joinpath(out_dir, "$(base)_headline.svg")
    cov_path      = joinpath(out_dir, "$(base)_cov.svg")

    fig_summary  = build_figure(fair_func, to, bi, summary_path)
    fig_headline = build_headline_figure(fair_func, to, bi, headline_path)
    fig_cov      = build_cov_figure(fair_func, to, bi, cov_path)
    display(fig_summary)
    println("  → $summary_path")
    println("  → $headline_path")
    println("  → $cov_path")

    if SHOW_BILEVEL
        dominates = bi.Palma < minimum(filter(isfinite, to.Palma)) &&
                    bi.Linf  < minimum(to.Linf) &&
                    bi.CoV   < minimum(to.CoV)  &&
                    bi.L1    < minimum(to.L1)
        dominated = bi.Linf > maximum(to.Linf) ||
                    bi.CoV  > maximum(to.CoV) ||
                    (isfinite(bi.Palma) &&
                     bi.Palma > maximum(filter(isfinite, to.Palma)))
        println("  bilevel coords: total=$(round(bi.total_shed, digits=2))  " *
                "L∞=$(round(bi.Linf, digits=2))  " *
                "Palma=$(isnan(bi.Palma) ? "NaN" : string(round(bi.Palma, digits=3)))  " *
                "CoV=$(round(bi.CoV, digits=3))")
        if dominates
            println("  ★ bilevel sits below the single-level frontier on every metric — genuine win.")
        elseif dominated
            println("  ★ bilevel is Pareto-dominated on at least one metric — scalability is the only remaining argument.")
        else
            println("  ★ bilevel sits within the single-level α-range — partial trade-off, no clean dominance.")
        end
    end
end

println("\nFigures written to: $out_dir")
