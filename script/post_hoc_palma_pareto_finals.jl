"""
    Post-hoc Palma Pareto plots — DEFENSE FINALS.

    A defense-only variant of post_hoc_fairness_pareto.jl, pared down to a
    single fairness function: Palma. It overlays the single-level Palma
    trade-off sweeps (integer + relaxed) with the single bilevel-Palma point.
    No min-max line, no min-max marker.

    Input data is PINNED (not mtime-picked) so the defense figures are
    reproducible regardless of what else lands on disk:
      * trade-off sweeps  — 2026-05-28 (T=24)
      * bilevel Palma     — 2026-05-27 (T=24, previous solution strategy via
                            run_validation_mn.jl — NOT the 2026-06-01 SLP
                            reverse-mode run).

    Outputs (under results/finals/), one per variant ∈ {integer, relaxed}:
      * pareto_<case>_<pshed>_<variant>_palma.svg  — y = Palma post-hoc
      * pareto_<case>_<pshed>_<variant>_cov.svg       — y = CoV
      * pareto_<case>_<pshed>_<variant>.svg           — 4-up [L1, L∞, Palma, CoV]

    Usage:
        julia --project=. script/post_hoc_palma_pareto_finals.jl
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

# ============================================================
# PINNED INPUT DATA (defense finals — do NOT mtime-pick)
# ============================================================
# Trade-off Palma sweeps: 2026-06-03 (T=8, no-bd, matched-objective + warm-start).
const PIN_DATE = get(ENV, "FINALS_DATE", "2026-06-03")           # trade-off sweeps date
# The bilevel point can come from a different run/date than the sweeps, e.g. the
# 2026-06-04 SLP reverse-mode shed run overlaid on the 2026-06-03 shed sweeps.
const BILEVEL_DATE = get(ENV, "FINALS_BILEVEL_DATE", PIN_DATE)
# Which quantity the Palma objective optimized: "shed" reads the *_shedobj output
# dirs, "served" the plain dirs. BOTH axes (sweeps + bilevel) must match — the
# plotted metric (per-load aggregate shed-Palma) is comparable only within a target.
const OBJ_SUFFIX = get(ENV, "PALMA_TARGET", "served") == "shed" ? "_shedobj" : ""
const PINNED_TRADE_OFF = Dict(
    (fair_func = "palma", relaxed = false) =>
        joinpath(RESULTS_ROOT, PIN_DATE, "palma_trade_off_mn$(OBJ_SUFFIX)",
                 "palma_sweep_mn_$(tags.trade_off)_$(PSHED_TYPE).jld2"),
    (fair_func = "palma", relaxed = true) =>
        joinpath(RESULTS_ROOT, PIN_DATE, "palma_relaxed_trade_off_mn$(OBJ_SUFFIX)",
                 "palma_sweep_mn_$(tags.trade_off)_$(PSHED_TYPE).jld2"),
)

# Bilevel Palma: served default = 2026-06-03 formal-CC MILP; shed = 2026-06-04 SLP
# reverse-mode run (set FINALS_BILEVEL_DATE=2026-06-04 PALMA_TARGET=shed).
const PINNED_BILEVEL = Dict(
    "palma" => joinpath(RESULTS_ROOT, BILEVEL_DATE, "bilevel_validation_mn",
        tags.bilevel, "palma_$(PSHED_TYPE)$(OBJ_SUFFIX)",
        "bilevel_mn_$(tags.bilevel)_palma_$(PSHED_TYPE).jld2"),
)

# Sweeps to overlay as single-level lines and bilevel objectives to
# overlay as scatter markers. Linestyle distinguishes fair-func (palma
# vs min-max). Integer and relaxed panels are rendered separately, so
# each fair-func uses the same full-intensity color across both variants.
const SWEEP_STYLES = [
    (key = "palma_integer",   label = "Single-level Palma (integer)",
     fair_func = "palma",   relaxed = false,
     linestyle = :solid, color = RGB(0.20, 0.40, 0.85)),
    (key = "palma_relaxed",   label = "Single-level Palma (relaxed)",
     fair_func = "palma",   relaxed = true,
     linestyle = :solid, color = RGB(0.20, 0.40, 0.85)),
]

const BILEVEL_STYLES = [
    (key = "palma",      label = "Bi-level Palma",      marker = :star5,
     color = :black),
]

# ============================================================
# JLD2 LOCATORS (pinned — see PINNED_* dicts above)
# ============================================================
function _trade_off_jld2(fair_func::String, relaxed::Bool)
    return get(PINNED_TRADE_OFF, (fair_func = fair_func, relaxed = relaxed), nothing)
end

function _bilevel_jld2(fair_func::String)
    return get(PINNED_BILEVEL, fair_func, nothing)
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

"""
Matched fairness metric — per-period cost-weighted SERVED-Palma:

    Σ_t λ_t · ( top10%(pserved_t) / bot40%(pserved_t) ),   pserved_{t,j} = pd_{t,j} − shed_{t,j}

This is exactly what the bilevel upper level (`palma_ratio_minimization`) and the
refactored single-level trade-off (`palma_trade_off_mn.jl`) optimize — sorting
each period independently, NOT the horizon-aggregate shed. `shed_pl` and `pd_pl`
are [period × load]; `λ` has length T. Returns NaN if any period's served-Palma
is undefined (bot40 below threshold), mirroring `palma_cost_weighted_log`.
"""
function _cost_weighted_served_palma(shed_pl::AbstractMatrix, pd_pl::AbstractMatrix,
                                     λ::AbstractVector)
    T = size(shed_pl, 1)
    @assert size(pd_pl, 1) == T "pd_pl period axis ($(size(pd_pl,1))) ≠ shed_pl ($T)"
    @assert length(λ) == T "λ length ($(length(λ))) ≠ T ($T)"
    acc = 0.0
    for t in 1:T
        served_t = [max(0.0, pd_pl[t, j] - shed_pl[t, j]) for j in axes(shed_pl, 2)]
        pt = _palma_ratio_safe(served_t)
        isfinite(pt) || return NaN
        acc += λ[t] * pt
    end
    return acc
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
    # All metrics on the per-load AGGREGATE-shed vector, UNCOSTED (λ=1): Palma is
    # the classic top10%/bot40% Palma ratio of total load shed (cost-weighting
    # distorted the bilevel comparison; absolute shed is the sensible measure).
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

"""
Norms for a [period × load] shed matrix. L1/L2/L∞/CoV are computed on the
per-load aggregate-shed vector (sum over periods, NaNs ignored). `Palma` is the
MATCHED per-period cost-weighted served-Palma — `Σ_t λ_t·Palma_t^served` — which
needs the per-period demand `pd_pl` [period × load] and costs `λ`. When those
are not supplied, `Palma` falls back to NaN (the aggregate-shed Palma is no
longer what we plot).
"""
function _aggregate_norms(matrix::AbstractMatrix)
    # matrix is [period × load]; sum over periods → per-load total shed, then the
    # UNCOSTED aggregate norms (Palma = top10%/bot40% of total shed).
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
    if isempty(xs_all)
        # No finite points for this y-axis. For the matched served-Palma this
        # happens in the INTEGER/rounded domain: whole-block integer shedding
        # zeroes a period's bot40-served, so the per-period Palma is undefined
        # (the σ_t·bot_sum_t=1 protection only holds in the continuous problem).
        # Degrade to an annotated empty panel instead of crashing the whole run,
        # so the other (L1/L∞/CoV) panels and the relaxed figures still render.
        @warn "No finite points for y=$ylab — rendering empty panel " *
              "(matched per-period served-Palma is undefined for integer/rounded shedding)."
        p = plot(; xlabel = "total load shed (kW)", ylabel = ylab,
            xlims = (0, 1), ylims = (0, 1), xticks = false, yticks = false,
            framestyle = :box, legend = false,
            background_color = :white, foreground_color = :black, _PARETO_FONT...)
        annotate!(p, 0.5, 0.5,
            text("undefined for integer shedding\n(per-period bot40 served = 0)",
                 18, :center, :gray40))
        return p
    end

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
        "Palma (unitless)"; zoom = zoom,
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
        "Palma (unitless)";
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

# ============================================================
# RUN
# ============================================================
out_dir = joinpath(RESULTS_ROOT, "finals")
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

# NOTE: the α=1 endpoint of the relaxed sweep used to be trimmed because the
# aggregate-SHED Palma collapsed there (no bot40 mass / huge top10), wrecking the
# y-axis. The MATCHED served-Palma we now plot is well-behaved at α=1 (pure
# fairness maximizes the bot40 served mass → small, finite ratio), so the trim is
# no longer needed. `_drop_nan` in `_pareto_panel` still removes any genuinely
# undefined low-α points (a period's bot40 served fully shed under pure efficiency).
function _trim_last(sw::NamedTuple)
    n = length(sw.alphas)
    n < 2 && return sw
    keep = 1:(n-1)
    return (alphas = sw.alphas[keep], total_shed = sw.total_shed[keep],
            L1 = sw.L1[keep], L2 = sw.L2[keep], Linf = sw.Linf[keep],
            CoV = sw.CoV[keep], Palma = sw.Palma[keep],
            n_periods = sw.n_periods, source = sw.source)
end

if SHOW_BILEVEL
    for bs in BILEVEL_STYLES
        path = _bilevel_jld2(bs.key)
        if path === nothing || !isfile(path)
            @warn "[$(bs.key)] no pinned bilevel JLD2 found — skipping marker" path
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
    build_cov_figure(sweeps_subset, bilevels_subset, cov_path;
        legend_position = legpos, bilevel_alpha = bi_α,
        bilevel_label_suffix = bi_lbl)
    # Zoomed variants — same data, axes framed around the bilevel stars.
    # Skip if no bilevel markers (no anchor to zoom to).
    if !isempty(bilevels_subset)
        build_summary_figure(sweeps_subset, bilevels_subset, summary_zoom_path;
            zoom = true, bilevel_alpha = bi_α, bilevel_label_suffix = bi_lbl)
        build_palma_figure(sweeps_subset, bilevels_subset, palma_zoom_path;
            zoom = true, legend_position = legpos, bilevel_alpha = bi_α,
            bilevel_label_suffix = bi_lbl)
        build_cov_figure(sweeps_subset, bilevels_subset, cov_zoom_path;
            zoom = true, legend_position = legpos, bilevel_alpha = bi_α,
            bilevel_label_suffix = bi_lbl)
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
