"""
Diagnostic: compute the PER-PERIOD cost-weighted SERVED-Palma on the relaxed
sweep (and the pinned bilevel point), the form that mirrors the bilevel
upper-level objective:

    S(ν) = Σ_t λ_t · Palma_t^served ,
    Palma_t^served = top10%(served_{t,·}) / bot40%(served_{t,·}),
    served_{t,i} = pd_{t,i} − shed_{t,i},   λ_t = PEAK_TIME_COSTS.

Reported both as the raw weighted sum S and the cost-weighted AVERAGE
S / Σλ_t (on the same ~1–4 scale as a single Palma ratio, so it is
comparable to the raw-aggregate-shed Palma the finals plot uses).

For contrast it also prints the raw aggregate SHED-Palma (what the finals
plot currently shows) and the raw aggregate SERVED-Palma (the trade-off's
own internal metric / the CSV palma_ratio column).
"""

using JLD2, LinearAlgebra, Statistics, Printf

const RELAXED = joinpath(@__DIR__, "..", "..", "results", "2026-05-28",
    "palma_relaxed_trade_off_mn",
    "palma_sweep_mn_more_meshed_6_bus_absolute.jld2")
const BILEVEL = joinpath(@__DIR__, "..", "..", "results", "2026-05-27",
    "bilevel_validation_mn", "case6_unbalanced_switch_more_meshed_good4integer",
    "palma_absolute",
    "bilevel_mn_case6_unbalanced_switch_more_meshed_good4integer_palma_absolute.jld2")

function palma_indices(n)
    nb = max(1, floor(Int, 0.4n)); nt = max(1, ceil(Int, 0.1n))
    return (n - nt + 1):n, 1:nb
end
function palma_of(v; eps_denom = 1e-6)
    s = sort(collect(v)); top, bot = palma_indices(length(s))
    num = sum(max(0.0, s[i]) for i in top); den = sum(max(0.0, s[i]) for i in bot)
    return den < eps_denom ? Inf : num / den
end

# ---- relaxed sweep ----------------------------------------------------------
sw    = JLD2.load(RELAXED)
alphas = sw["alphas"]
shed  = sw["per_load_period_shed"]   # α × load × period
pd    = sw["per_load_period_pd"]     # load × period
costs = sw["PEAK_TIME_COSTS"]        # T
T     = sw["N_PERIODS"]
n_α, n_loads, n_per = size(shed)
Σλ = sum(costs)

@printf("relaxed sweep: %d α × %d loads × %d periods  |  Σλ=%.1f  peak λ=%.0f (period %d)\n\n",
        n_α, n_loads, n_per, Σλ, maximum(costs), argmax(costs))

function metrics(α_idx)
    # per-period served-Palma, cost-weighted
    S = 0.0; per_t = Float64[]
    for t in 1:n_per
        served_t = [pd[i, t] - shed[α_idx, i, t] for i in 1:n_loads]
        pt = palma_of(served_t)
        push!(per_t, pt)
        S += costs[t] * pt
    end
    # raw aggregate shed-Palma (finals plot) and served-Palma (trade-off internal)
    agg_shed   = [sum(shed[α_idx, i, :]) for i in 1:n_loads]
    agg_served = [sum(pd[i, :] - shed[α_idx, i, :]) for i in 1:n_loads]
    return (S = S, Savg = S / Σλ,
            shedP = palma_of(agg_shed), servedP = palma_of(agg_served),
            per_t = per_t)
end

@printf("%-6s | %12s %12s | %12s %12s\n",
        "ν", "cwPP-served", "(=S/Σλ)", "aggShed-P", "aggServed-P")
@printf("%-6s | %12s %12s | %12s %12s\n",
        "", "Σλ·Palma_t", "avg", "(finals)", "(tradeoff)")
println("-"^60)
for i in 1:n_α
    m = metrics(i)
    @printf("%-6.3f | %12.1f %12.3f | %12.3f %12.3f\n",
            alphas[i], m.S, m.Savg, m.shedP, m.servedP)
end

# Show which periods drive the cost-weighted served-Palma at ν≈0.79 vs 0.89.
println("\nPer-period served-Palma (cost-weighted contribution λ_t·Palma_t):")
for target in (0.79, 0.89)
    i = argmin(abs.(alphas .- target))
    m = metrics(i)
    contrib = costs .* m.per_t
    order = sortperm(contrib; rev = true)[1:5]
    @printf("ν=%.3f  S=%.1f  Savg=%.3f\n", alphas[i], m.S, m.Savg)
    for t in order
        @printf("    period %2d (λ=%5.2f): Palma_t=%.3f  contrib=%.1f\n",
                t, costs[t], m.per_t[t], contrib[t])
    end
end

# ---- bilevel point (to place the star on the same axis) ---------------------
println()
if isfile(BILEVEL)
    bl = JLD2.load(BILEVEL)
    ks = collect(keys(bl))
    println("bilevel JLD2 keys: ", ks)
    # Need per-period per-load shed + pd. pshed_matrix is T×n_loads.
    for (mat_key, lbl_) in (("relaxed_pshed_matrix", "relaxed"),
                            ("pshed_matrix", "integer (rounded)"))
        haskey(bl, mat_key) || continue
        M = bl[mat_key]                      # T × n_loads (per the finals loader)
        any(!isnan, M) || (println("  $lbl_: all-NaN, skip"); continue)
        bcosts = get(bl, "PEAK_TIME_COSTS", costs)
        # per-period pd for the bilevel: use its OWN pd_ref_matrix (same T×loads
        # orientation as pshed_matrix). Reusing the sweep pd would be wrong if the
        # demand profile / load ordering differs.
        @assert haskey(bl, "pd_ref_matrix") "bilevel JLD2 missing pd_ref_matrix"
        bpd = bl["pd_ref_matrix"]
        @printf("  [shapes] %s=%s  pd_ref_matrix=%s\n", mat_key, size(M), size(bpd))
        Tb, nl = size(M)
        Sb = 0.0; per_tb = Float64[]
        for t in 1:Tb
            served_t = [bpd[t, i] - M[t, i] for i in 1:nl]
            pt = palma_of(served_t); push!(per_tb, pt); Sb += bcosts[t] * pt
        end
        aggshed   = [sum(M[:, i]) for i in 1:nl]
        aggserved = [sum(bpd[:, i] - M[:, i]) for i in 1:nl]
        @printf("  bilevel %-18s  cwPP-served S=%.1f  Savg=%.3f | aggShed-P=%.3f  aggServed-P=%.3f\n",
                lbl_, Sb, Sb / sum(bcosts), palma_of(aggshed), palma_of(aggserved))
    end
else
    println("bilevel JLD2 not found at $BILEVEL — skipping star")
end
