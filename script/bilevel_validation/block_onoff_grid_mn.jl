#=
    Per-block served-status comparison grid — trade-off vs bilevel Palma, T=8.

    For each load block (only blocks that host loads), one subplot with
        x-axis = time period (hour),
        y-axis = block served status,
    overlaying two solutions:
        * single-level Palma trade-off at the matched-efficiency α
          (α whose total shed is closest to the bilevel's — same selection
          rule as eval_t8_palma_tradeoff_vs_bilevel.jl), and
        * bilevel Palma.

    Two variants are produced:
      * integer — rounded MLD; served status ∈ {0,1} (on/off). Whole blocks
        toggle, so the two series are nudged ±0.045 to stay visible where
        they coincide.
      * relaxed — pre-rounding LP relaxation; served status ∈ [0,1]
        (continuous served fraction), no nudge.

    Usage: julia --project=. script/bilevel_validation/block_onoff_grid_mn.jl
=#
using JLD2, Plots
include(joinpath(@__DIR__, "../figure_defaults.jl"))
include(joinpath(@__DIR__, "../block_display.jl"))

const DATE      = "2026-06-05"
const CASE      = "case6_unbalanced_switch_more_meshed_good4integer"
const CASE_TAG  = "more_meshed_6_bus"
R(p) = joinpath(@__DIR__, "../../", p)

bl = JLD2.load(R("results/$DATE/bilevel_validation_mn/$CASE/palma_absolute/bilevel_mn_$(CASE)_palma_absolute.jld2"))

# Both files list loads in the same order (verified in eval_t8_…); bus name
# per load comes from the trade-off file.
load_labels = bl["load_labels"]
hours       = bl["SELECTED_HOURS"]
T           = bl["N_PERIODS"]
bl_pd       = bl["pd_ref_matrix"]               # T × N nameplate demand

col_to = "#C2702A"   # warm orange — trade-off
col_bl = "#2A6F6B"   # muted teal  — bilevel

# Aggregate per-load shed/pd to blocks → served indicator = 1 - shed/pd.
function block_status(shed, pd, load2block, nB)
    bshed = zeros(T, nB); bpd = zeros(T, nB)
    for j in eachindex(load2block)
        c = load2block[j]; c == 0 && continue
        for t in 1:T
            bshed[t, c] += isnan(shed[t, j]) ? 0.0 : shed[t, j]
            bpd[t, c]   += pd[t, j]
        end
    end
    st = fill(NaN, T, nB)
    for t in 1:T, b in 1:nB
        bpd[t, b] > 1e-9 && (st[t, b] = 1.0 - bshed[t, b] / bpd[t, b])
    end
    st
end

function build_variant(variant, bl_shed_key, to_subdir)
    to = JLD2.load(R("results/$DATE/$to_subdir/palma_sweep_mn_$(CASE_TAG)_absolute.jld2"))
    @assert load_labels == to["load_labels"] "load order mismatch ($variant)"
    load_bus_names = to["load_bus_names"]

    bl_shed = bl[bl_shed_key]                    # T × N

    # matched-efficiency α: closest total shed to bilevel, skip failed (NaN) α.
    to_tot   = vec(sum(to["per_load_agg"], dims = 2))
    bl_total = sum(bl_shed)
    cand     = findall(isfinite.(to_tot))
    a        = cand[argmin(abs.(to_tot[cand] .- bl_total))]
    αstar    = to["alphas"][a]
    to_shed  = permutedims(to["per_load_period_shed"][a, :, :])  # T × N
    to_pd    = permutedims(to["per_load_period_pd"])             # T × N

    disp = resolve_block_display(CASE_TAG, load_bus_names)
    @assert disp !== nothing "no block_display mapping for $CASE_TAG"
    display_blocks, load2block = disp
    nB       = length(display_blocks)
    blk_nums = [num   for (num, _) in display_blocks]
    blk_lbls = [label for (_, label) in display_blocks]

    st_to = block_status(to_shed, to_pd, load2block, nB)
    st_bl = block_status(bl_shed, bl_pd, load2block, nB)

    integer = variant == "integer"
    xs = collect(1:T)
    off_to, off_bl = integer ? (+0.045, -0.045) : (0.0, 0.0)
    yticks = integer ? ([0, 1], ["off", "on"]) : ([0, 0.5, 1], ["0", "0.5", "1"])
    ylims  = integer ? (-0.35, 1.35) : (-0.05, 1.08)
    ylab   = integer ? "" : "served frac."

    subplots = Plots.Plot[]
    for b in 1:nB
        p = plot(; ylims, yticks,
                 xticks = (xs, string.(hours)),
                 title  = "Block $(blk_nums[b]) ($(blk_lbls[b]))",
                 legend = false, grid = true, gridalpha = 0.25,
                 xlabel = b > nB - 2 ? "hour" : "",
                 ylabel = b % 2 == 1 ? ylab : "",
                 framestyle = :box)
        plot!(p, xs, st_to[:, b] .+ off_to; seriestype = :steppost,
              lc = col_to, lw = 2, marker = :circle, ms = 4, mc = col_to, msc = col_to)
        plot!(p, xs, st_bl[:, b] .+ off_bl; seriestype = :steppost,
              lc = col_bl, lw = 2, marker = :diamond, ms = 4, mc = col_bl, msc = col_bl)
        push!(subplots, p)
    end

    leg = plot(; framestyle = :none, legend = :left, legendfontsize = 10)
    plot!(leg, [NaN], [NaN]; seriestype = :steppost, lc = col_to, lw = 2,
          marker = :circle, ms = 4, mc = col_to, label = "trade-off (α=$(round(αstar, digits=2)))")
    plot!(leg, [NaN], [NaN]; seriestype = :steppost, lc = col_bl, lw = 2,
          marker = :diamond, ms = 4, mc = col_bl, label = "bilevel Palma")
    push!(subplots, leg)

    ncol = 2
    nrow = ceil(Int, length(subplots) / ncol)
    statuslabel = integer ? "on/off" : "served fraction"
    P = plot(subplots...; layout = (nrow, ncol), size = (820, 230 * nrow),
             plot_title = "Per-block $statuslabel — $variant Palma (T=$T)  |  trade-off=$(round(to_tot[a], digits=0)) kW, bilevel=$(round(bl_total, digits=0)) kW",
             left_margin = 5Plots.mm, bottom_margin = 4Plots.mm)

    out_dir = R("results/$DATE/block_onoff_grid")
    mkpath(out_dir)
    svg = joinpath(out_dir, "block_onoff_grid_tradeoff_vs_bilevel_$(variant)_absolute.svg")
    png = joinpath(out_dir, "block_onoff_grid_tradeoff_vs_bilevel_$(variant)_absolute.png")
    savefig(P, svg); savefig(P, png)

    println("\n==================== $(uppercase(variant)) ====================")
    println("matched-efficiency α index=$a  α=$(round(αstar, digits=4))")
    println("trade-off total=$(round(to_tot[a], digits=1)) kW   bilevel total=$(round(bl_total, digits=1)) kW")
    for b in 1:nB
        println("  Block $(blk_nums[b]) ($(blk_lbls[b])):")
        println("    trade-off: ", round.(st_to[:, b], digits = 2))
        println("    bilevel  : ", round.(st_bl[:, b], digits = 2))
    end
    println("Saved → $svg")
    println("        $png")
end

build_variant("integer", "pshed_matrix",         "palma_trade_off_mn")
build_variant("relaxed", "relaxed_pshed_matrix",  "palma_relaxed_trade_off_mn")
