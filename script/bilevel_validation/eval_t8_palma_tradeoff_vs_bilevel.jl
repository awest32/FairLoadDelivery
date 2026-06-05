#=
    Palma trade-off (single-level Pareto sweep) vs Palma bilevel — T=8.

    Same fairness function (shed-Palma), two methods. Compares the bilevel
    solution against the trade-off α-point at the SAME total shed (matched
    efficiency), so the question is: for equal kW shed, which method spreads
    it more fairly (lower post-hoc Palma)? Also reports the trade-off's
    pure-fairness end (α=1) to show the shed-everything degeneracy the bilevel
    avoids. Per-load / per-period / per-bus, for integer and relaxed.

    Post-hoc Palma = _palma_ratio_safe over per-load aggregate shed — the SAME
    neutral metric the pareto plot uses for both methods.

    Usage: julia --project=. script/bilevel_validation/eval_t8_palma_tradeoff_vs_bilevel.jl
=#
using JLD2, DataFrames, CSV, Printf, Dates

const CASE = "case6_unbalanced_switch_more_meshed_good4integer"
const OUT  = joinpath(@__DIR__, "../../results", Dates.format(now(), "yyyy-mm-dd"), "t8_palma_tradeoff_vs_bilevel")
mkpath(OUT)

R(p) = joinpath(@__DIR__, "../../", p)
bl = JLD2.load(R("results/2026-06-04/bilevel_validation_mn/$CASE/palma_absolute/bilevel_mn_$(CASE)_palma_absolute.jld2"))
to_int = JLD2.load(R("results/2026-06-03/palma_trade_off_mn_shedobj/palma_sweep_mn_more_meshed_6_bus_absolute.jld2"))
to_rlx = JLD2.load(R("results/2026-06-03/palma_relaxed_trade_off_mn_shedobj/palma_sweep_mn_more_meshed_6_bus_absolute.jld2"))

# Post-hoc Palma — identical to post_hoc_fairness_pareto.jl::_palma_ratio_safe.
function palma_safe(x::AbstractVector{<:Real})
    n = length(x); n < 3 && return NaN
    s = sort(collect(x))
    top10 = sum(s[ceil(Int, 0.9n):end]); bot40 = sum(s[1:floor(Int, 0.4n)]); tot = sum(s)
    return (bot40 > 1e-4 * tot) ? top10 / bot40 : NaN
end

load_labels = bl["load_labels"]
hours       = bl["SELECTED_HOURS"]
bus_names   = to_int["load_bus_names"]            # aligned to load_labels (verified identical order)
uniq_bus    = unique(bus_names)
nP          = length(hours)

function per_bus(per_load)
    Dict(b => sum(per_load[j] for j in eachindex(per_load) if bus_names[j] == b) for b in uniq_bus)
end

for (vtag, blkey, to) in [("integer", "pshed_matrix", to_int),
                          ("relaxed", "relaxed_pshed_matrix", to_rlx)]
    M_bl        = bl[blkey]                         # nP × 9
    perload_bl  = vec(sum(M_bl, dims = 1))
    total_bl    = sum(M_bl)
    palma_bl    = palma_safe(perload_bl)

    PLA         = to["per_load_agg"]                # 12 × 9
    totals_to   = vec(sum(PLA, dims = 2))
    a           = argmin(abs.(totals_to .- total_bl))   # matched-efficiency α index
    αstar       = to["alphas"][a]
    perload_to  = PLA[a, :]
    total_to    = totals_to[a]
    palma_to    = palma_safe(perload_to)
    # pure-fairness end (α=1) — the degenerate reference
    perload_a1  = PLA[end, :]; total_a1 = totals_to[end]; palma_a1 = palma_safe(perload_a1)

    perper_bl   = vec(sum(M_bl, dims = 2))
    perper_to   = vec(sum(to["per_load_period_shed"][a, :, :], dims = 1))

    pl = DataFrame(load = load_labels,
                   tradeoff = round.(perload_to, digits = 2),
                   bilevel  = round.(perload_bl, digits = 2),
                   diff     = round.(perload_bl .- perload_to, digits = 2))
    pp = DataFrame(period = 1:nP, hour = hours,
                   tradeoff = round.(perper_to, digits = 2),
                   bilevel  = round.(perper_bl, digits = 2))
    pbT = per_bus(perload_to); pbB = per_bus(perload_bl)
    pb = DataFrame(bus = uniq_bus,
                   tradeoff = [round(pbT[b], digits = 2) for b in uniq_bus],
                   bilevel  = [round(pbB[b], digits = 2) for b in uniq_bus])

    CSV.write(joinpath(OUT, "t8_palma_per_load_$(vtag).csv"),   pl)
    CSV.write(joinpath(OUT, "t8_palma_per_period_$(vtag).csv"), pp)
    CSV.write(joinpath(OUT, "t8_palma_per_bus_$(vtag).csv"),    pb)

    println("\n==================== $(uppercase(vtag)) ====================")
    @printf("matched at α=%.3f:  trade-off total=%.1f kW (Palma=%.3f)   |   bilevel total=%.1f kW (Palma=%.3f)\n",
            αstar, total_to, palma_to, total_bl, palma_bl)
    @printf("trade-off α=1.0 (pure fairness): total=%.1f kW  Palma=%s   <- degeneracy the bilevel avoids\n",
            total_a1, isnan(palma_a1) ? "NaN/shed-all" : @sprintf("%.3f", palma_a1))
    println("\nPER-LOAD (Σ_t kW):");   show(pl, allrows = true, allcols = true); println()
    println("\nPER-PERIOD (Σ_load kW):"); show(pp, allrows = true, allcols = true); println()
    println("\nPER-BUS (Σ_t kW):");    show(pb, allrows = true, allcols = true); println()
end
println("\nCSVs written to: ", OUT)
