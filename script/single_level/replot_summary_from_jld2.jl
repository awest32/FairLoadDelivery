"""
    Replot the trade-off summary figure (fig1) from a saved sweep JLD2.

    Use when only the summary-figure code changed (titles, shared y-axis,
    Pareto-curve panel, etc.) and re-sweeping the trade-off MIPs would be
    a waste of solver time — every variable fig1 reads is already in the
    JLD2 (per_load_agg, max_shed, palma_ratio_log, alphas, load_labels).

    Overwrites summary_integer_*.svg in the same directory as each input.

    Usage:
        julia --project=. script/single_level/replot_summary_from_jld2.jl
"""

using JLD2
using Plots
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "..", "figure_defaults.jl"))

font_kw = (tickfontsize = 22, guidefontsize = 22,
           titlefontsize = 26, legendfontsize = 14,
           fontfamily = "Computer Modern")
const ANNOT_PT = 14

# Sweep files to refresh. Override with REPLOT_TARGETS (";"-separated paths) to
# re-plot specific JLD2s without editing this default list.
results_root = joinpath(@__DIR__, "..", "..", "results")
const TARGETS = haskey(ENV, "REPLOT_TARGETS") ?
    String.(split(ENV["REPLOT_TARGETS"], ";")) :
    [
    joinpath(results_root, "2026-05-28", "palma_trade_off_mn",
             "palma_sweep_mn_more_meshed_6_bus_absolute.jld2"),
    joinpath(results_root, "2026-05-28", "palma_relaxed_trade_off_mn",
             "palma_sweep_mn_more_meshed_6_bus_absolute.jld2"),
    joinpath(results_root, "2026-05-28", "trade_off_mn",
             "min_max_trade_off_mn_more_meshed_6_bus_absolute.jld2"),
    joinpath(results_root, "2026-05-28", "trade_off_mn",
             "min_max_relaxed_trade_off_mn_more_meshed_6_bus_absolute.jld2"),
    joinpath(results_root, "2026-05-28", "trade_off_mn",
             "efficiency_relaxed_trade_off_mn_more_meshed_6_bus_absolute.jld2"),
]

# Classify by fair_func — drives which Pareto y-axis goes in the third panel.
# Palma sweeps render Palma-ratio-vs-total-shed (matches the post_hoc panel
# but with only the trade-off curve, no bilevel markers). Min-max and
# efficiency sweeps render max-per-load-shed-vs-total-shed.
_is_palma(ff::AbstractString) = startswith(ff, "palma")

# Cost-weighted per-period Palma (the matched-objective form) over a [period × load]
# value matrix: Σ_t λ_t·top10(v_t) / Σ_t λ_t·bot40(v_t). Returns NaN if the
# denominator collapses. Used for both served (v = pserved) and shed (v = pshed).
_palma_idx(n) = (collect((n - max(1, ceil(Int, 0.1n)) + 1):n), collect(1:max(1, floor(Int, 0.4n))))
function _cw_palma(mat_pl::AbstractMatrix, λ::AbstractVector; eps = 1e-6)
    ti, bi = _palma_idx(size(mat_pl, 2))
    top = 0.0; bot = 0.0
    for t in axes(mat_pl, 1)
        s = sort(mat_pl[t, :])
        top += λ[t] * sum(s[i] for i in ti)
        bot += λ[t] * sum(s[i] for i in bi)
    end
    bot < eps ? NaN : top / bot
end

# UNCOSTED (λ=1) Palma of a per-load vector v: top10%/bot40% of the sorted totals.
# Relative denominator guard mirrors post_hoc_palma_pareto_finals.jl.
function _palma_uncosted(v::AbstractVector)
    s = sort(collect(v)); ti, bi = _palma_idx(length(v))
    num = sum(s[i] for i in ti); den = sum(s[i] for i in bi); tot = sum(s)
    (tot > 0 && den > 1e-4 * tot) ? num / den : NaN
end

function build_dist_plot_agg(load_labels, per_load_agg_vec; ylim_max)
    p = bar(load_labels, per_load_agg_vec,
        xlabel = "load",
        ylabel = "aggregate load shed (kW)",
        legend = false,
        color  = :steelblue,
        linecolor = :black,
        ylims = (0.0, ylim_max * 1.10);
        font_kw...)
    for (i, v) in enumerate(per_load_agg_vec)
        isfinite(v) || continue
        annotate!(p, i, v + ylim_max * 0.02,
            text("$(round(v, digits = 1))", ANNOT_PT, :center))
    end
    return p
end

function replot_one(path::String)
    isfile(path) || (@warn "missing $path"; return)
    d = JLD2.load(path)
    # UNWEIGHTED per-load aggregate (Σ_t pshed, NO cost weighting), rebuilt from
    # the per-period tensor so these standalone figures match the post-hoc
    # comparison plots and the "uncosted" Palma labels below are actually uncosted.
    # The saved `per_load_agg` is COST-WEIGHTED (Σ_t ρ_t·pshed); using it would put
    # every figure in ρ-weighted units. Older JLD2s without the tensor fall back.
    per_load_agg = if haskey(d, "per_load_period_shed")
        plps = d["per_load_period_shed"]            # alpha × load × period (unweighted)
        [sum(plps[i, j, t] for t in axes(plps, 3)) for i in axes(plps, 1), j in axes(plps, 2)]
    else
        d["per_load_agg"]
    end
    alphas       = d["alphas"]
    load_labels  = d["load_labels"]
    fair_func    = d["fair_func"]
    case         = d["case"]
    pshed_type   = d["pshed_type"]

    agg_total_shed = [sum(per_load_agg[i, :]) for i in 1:size(per_load_agg, 1)]

    ymax_shared = max(
        maximum(filter(isfinite, per_load_agg[1, :]);   init = 0.0),
        maximum(filter(isfinite, per_load_agg[end, :]); init = 0.0))

    p_dist_a0 = build_dist_plot_agg(load_labels, per_load_agg[1, :];   ylim_max = ymax_shared)
    p_dist_a1 = build_dist_plot_agg(load_labels, per_load_agg[end, :]; ylim_max = ymax_shared)

    # Pareto curve: steelblue dots (matching the bar charts), 14pt markers,
    # α=0 / α=1 endpoints annotated in the same style as the bar-chart value
    # labels. Branches by fair_func: palma uses Palma ratio on y;
    # min_max/efficiency uses max per-load shed.
    function _annotate_alpha_endpoints!(p, xs, ys, αs)
        yrange = maximum(ys) - minimum(ys)
        yoff = 0.05 * (yrange == 0 ? 1.0 : yrange)
        annotate!(p, xs[1],   ys[1]   + yoff,
            text("ν=$(round(αs[1],   digits=2))", ANNOT_PT, :center))
        annotate!(p, xs[end], ys[end] + yoff,
            text("ν=$(round(αs[end], digits=2))", ANNOT_PT, :center))
    end

    function _pareto_panel(yvec, ylab)
        fin = findall(isfinite, yvec)
        p = plot(agg_total_shed[fin], yvec[fin],
            seriestype = :line, lc = :grey,
            marker = :circle, markersize = 14, color = :steelblue,
            markerstrokecolor = :steelblue,
            xlabel = "total load shed (kW)", ylabel = ylab,
            legend = false; font_kw...)
        isempty(fin) || _annotate_alpha_endpoints!(p, agg_total_shed[fin], yvec[fin], alphas[fin])
        p
    end

    # Palma sweeps: PRIMARY plot is SHED-Palma (clean/monotone on integer +
    # relaxed); served-Palma kept as a secondary panel (it degenerates on the
    # relaxed sweep at high α). Both are cost-weighted per-period
    # (Σ_t λ_t·top10 / Σ_t λ_t·bot40), computed from the per-period tensors.
    # min_max/efficiency: max per-load shed.
    p_pareto_served = nothing
    p_pareto = if _is_palma(fair_func)
        # UNCOSTED aggregate Palma (λ=1) of the per-load totals — the classic
        # "Palma ratio of load shed". Cost-weighting distorted the bilevel
        # comparison; absolute shed is the reported metric. PRIMARY = shed,
        # secondary = served.
        pdp    = d["per_load_period_pd"]                       # load × period
        pd_tot = [sum(pdp[j, :]) for j in 1:size(pdp, 1)]      # per-load total demand
        nα = length(alphas)
        shed_p   = [_palma_uncosted(per_load_agg[i, :]) for i in 1:nα]
        served_p = [_palma_uncosted(pd_tot .- per_load_agg[i, :]) for i in 1:nα]
        p_pareto_served = _pareto_panel(served_p, "Palma served (unitless, uncosted)")
        _pareto_panel(shed_p, "Palma shed (unitless, uncosted)")
    else
        agg_max_shed = [maximum(per_load_agg[i, :]) for i in 1:size(per_load_agg, 1)]
        _pareto_panel(agg_max_shed, "max per-load shed (kW)")
    end

    # Per-panel filename suffix matches the trade-off scripts' conventions:
    #   * Palma: folder name encodes relaxed/integer; the suffix is just
    #     $(kind)_$(pshed_type) with kind detected from the folder.
    #   * Min-max / efficiency: folder is shared, so the suffix carries
    #     fair_func (which itself includes "_relaxed" when applicable).
    suffix = if _is_palma(fair_func)
        kind = occursin("palma_relaxed_trade_off_mn", dirname(path)) ?
            "relaxed" : "integer"
        "$(kind)_$(pshed_type)"
    else
        "$(pshed_type)_$(case)_$(fair_func)"
    end

    panels = Any[("alpha0", p_dist_a0), ("alpha1", p_dist_a1), ("pareto", p_pareto)]
    p_pareto_served === nothing || push!(panels, ("pareto_served", p_pareto_served))
    for (name, p) in panels
        fig = plot(p; size = (900, 760),
            left_margin = 7Plots.mm, right_margin = 6Plots.mm,
            top_margin = 8Plots.mm, bottom_margin = 7Plots.mm)
        out_path = joinpath(dirname(path), "summary_single_$(name)_$(suffix).svg")
        savefig(fig, out_path)
        println("  → $out_path")
    end

    # ------------------------------------------------------------------
    # pareto_norms_*.svg — 4-up L1/L2/L∞/CoV vs aggregate total shed,
    # α encoded as cividis marker color with a colorbar in a thin 5th
    # panel. Rebuilt from per_load_agg so a font/style change here
    # doesn't require a full sweep re-run.
    # Filename matches the trade-off scripts:
    #   * Palma: pareto_norms_$(kind)_$(pshed_type).svg
    #   * Min-max / efficiency: pareto_norms_integer_$(pshed_type)_$(case)_$(fair_func).svg
    # ------------------------------------------------------------------
    l1_vec   = zeros(size(per_load_agg, 1))
    l2_vec   = zeros(size(per_load_agg, 1))
    linf_vec = zeros(size(per_load_agg, 1))
    cov_vec  = zeros(size(per_load_agg, 1))
    for i in 1:size(per_load_agg, 1)
        v = filter(isfinite, per_load_agg[i, :])
        l1_vec[i]   = norm(v, 1)
        l2_vec[i]   = norm(v, 2)
        linf_vec[i] = norm(v, Inf)
        m = isempty(v) ? 0.0 : mean(v)
        s = length(v) > 1 ? std(v) : 0.0
        cov_vec[i]  = m > 1e-9 ? s / m : NaN
    end

    # Post-hoc style (steelblue dots + grey line + ν endpoint annotations), matching
    # the summary pareto above and post_hoc_palma_pareto_finals.jl — no cividis colorbar.
    function pareto_norm_panel(y, ylab)
        fin = findall(isfinite, y)
        p = plot(agg_total_shed[fin], y[fin],
            seriestype = :line, lc = :grey,
            marker = :circle, markersize = 14, color = :steelblue,
            markerstrokecolor = :steelblue,
            xlabel = "total load shed (kW)", ylabel = ylab,
            legend = false; font_kw...)
        _annotate_alpha_endpoints!(p, agg_total_shed[fin], y[fin], alphas[fin])
        p
    end
    p_l1   = pareto_norm_panel(l1_vec,   "L1 norm of shed (kW)")
    p_l2   = pareto_norm_panel(l2_vec,   "L2 norm of shed (kW)")
    p_linf = pareto_norm_panel(linf_vec, "L∞ norm of shed (kW)")
    p_cov  = pareto_norm_panel(cov_vec,  "CoV (stdev/mean)")
    fig_norms = plot(p_l1, p_l2, p_linf, p_cov,
        layout = (1, 4),
        size = (2200, 600),
        left_margin = 7Plots.mm, right_margin = 6Plots.mm,
        top_margin = 8Plots.mm, bottom_margin = 7Plots.mm)
    norms_name = _is_palma(fair_func) ?
        "pareto_norms_$(suffix).svg" :
        "pareto_norms_integer_$(suffix).svg"
    norms_path = joinpath(dirname(path), norms_name)
    savefig(fig_norms, norms_path)
    println("  → $norms_path")
end

for t in TARGETS
    replot_one(t)
end
