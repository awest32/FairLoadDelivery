#=
Feasibility / definability probe for the PER-PERIOD served-Palma trade-off
==========================================================================

Purpose: the pinned sweep JLD2s were solved under the OLD aggregate objective,
so re-scoring them under the new per-period served-Palma tells us nothing about
what the NEW objective actually produces. This probe runs the refactored
machinery DIRECTLY at small T (fast), for both the relaxed and integer MLD, at a
few α, and reports for each solve:
  • Gurobi termination + primal status (→ is the integer per-period MIP feasible?)
  • per-period served-Palma definedness (how many of T periods have bot40>0)
  • the matched cost-weighted value Σ_t λ_t·Palma_t^served (NaN if any period undefined)

Hypotheses under test:
  (H1) RELAXED per-period trade-off solves with all-periods-defined served-Palma.
  (H2) INTEGER per-period trade-off is infeasible (or leaves periods undefined),
       because keeping bot40(served_t)>0 needs ≤2 of 9 loads fully shed per period,
       which whole-block integer shedding can't guarantee at high-stress periods.

This is a throwaway diagnostic; the machinery below is copied from
script/single_level/palma_trade_off_mn.jl so the probe exercises the SAME
formulation. Keep in sync if that script's machinery changes.
=#

using FairLoadDelivery
using PowerModelsDistribution
using Gurobi, Ipopt
using JuMP
using Statistics
import MathOptInterface as MOI

const _PMD = PowerModelsDistribution

# ============================================================
# CONFIG — small T for speed
# ============================================================
case_name      = "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss"
LS_PERCENT     = 0.8
SELECTED_HOURS = [4, 12, 18]          # T=3: trough / demand-peak / cost-peak
PEAK_STRESS    = 1.0
CENTER_AT_NOMINAL = true
PROBE_ALPHAS   = [0.0, 0.5, 1.0]
TIME_LIMIT     = 120                   # s per solve

dir       = @__DIR__
case_path = joinpath(dir, case_name)
N_PERIODS = length(SELECTED_HOURS)
PEAK_TIME_COSTS = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                   for h in SELECTED_HOURS]

# ============================================================
# Palma helpers (copied from palma_trade_off_mn.jl)
# ============================================================
function compute_palma_indices(n::Int)
    n_bot = max(1, floor(Int, 0.4 * n))
    n_top = max(1, ceil(Int, 0.1 * n))
    return collect((n - n_top + 1):n), collect(1:n_bot)
end

function palma_ratio_value(vals::AbstractVector; eps_denom::Float64 = 1e-6)
    n = length(vals); s = sort(collect(vals))
    top_10_idx, bottom_40_idx = compute_palma_indices(n)
    num = sum(s[i] for i in top_10_idx); den = sum(s[i] for i in bottom_40_idx)
    return den < eps_denom ? Inf : num / den
end

function add_palma_machinery_per_period!(pm; nw_ids_int::Vector{Int}, relax_binary::Bool=false)
    model = pm.model
    nw0 = nw_ids_int[1]
    load_ids = sort(collect(_PMD.ids(pm, nw0, :load)))
    n = length(load_ids); T = length(nw_ids_int)
    P_period = Dict{Int,Vector{Float64}}()
    for nw in nw_ids_int
        P_period[nw] = [sum(_PMD.ref(pm, nw, :load, i)["pd"]) for i in load_ids]
    end
    top_10_idx, bottom_40_idx = compute_palma_indices(n)
    a = Vector{Any}(undef, T); u = Vector{Any}(undef, T); σ = Vector{Any}(undef, T)
    top_sum = Vector{Any}(undef, T); bot_sum = Vector{Any}(undef, T)
    pserved_period = Dict{Int,Any}()
    for (ti, nw) in enumerate(nw_ids_int)
        Pt = P_period[nw]
        pserved_t = JuMP.@expression(model, [k = 1:n], sum(_PMD.var(pm, nw, :pd, load_ids[k])))
        pserved_period[nw] = pserved_t
        a[ti] = relax_binary ?
            JuMP.@variable(model, [1:n, 1:n], lower_bound=0, upper_bound=1, base_name="a_$ti") :
            JuMP.@variable(model, [1:n, 1:n], Bin, base_name="a_$ti")
        u[ti] = JuMP.@variable(model, [1:n, 1:n], lower_bound=0, base_name="u_$ti")
        for i in 1:n, j in 1:n
            Pj = Pt[j]
            JuMP.@constraint(model, u[ti][i, j] >= pserved_t[j] + a[ti][i, j] * Pj - Pj)
            JuMP.@constraint(model, u[ti][i, j] <= a[ti][i, j] * Pj)
            JuMP.@constraint(model, u[ti][i, j] <= pserved_t[j])
        end
        for i in 1:n; JuMP.@constraint(model, sum(a[ti][i, j] for j in 1:n) == 1); end
        for j in 1:n; JuMP.@constraint(model, sum(a[ti][i, j] for i in 1:n) == 1); end
        sorted_t = JuMP.@expression(model, [i = 1:n], sum(u[ti][i, j] for j in 1:n))
        for k in 1:(n-1); JuMP.@constraint(model, sorted_t[k] <= sorted_t[k+1]); end
        top_sum[ti] = JuMP.@expression(model, sum(sorted_t[i] for i in top_10_idx))
        bot_sum[ti] = JuMP.@expression(model, sum(sorted_t[i] for i in bottom_40_idx))
        σ[ti] = JuMP.@variable(model, base_name="sigma_$ti", lower_bound=1e-8)
        JuMP.@constraint(model, σ[ti] * bot_sum[ti] == 1.0)
    end
    eff_per_nw = Dict{Int,Any}()
    for nw in nw_ids_int
        td = sum(sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
        eff_per_nw[nw] = JuMP.@expression(model,
            sum(sum(_PMD.var(pm, nw, :pshed, d)) for d in _PMD.ids(pm, nw, :load)) / td)
    end
    return (n=n, T=T, load_ids=load_ids, nw_ids_int=nw_ids_int, P_period=P_period,
            pserved_period=pserved_period, a=a, u=u, σ=σ,
            top_sum=top_sum, bot_sum=bot_sum, eff_per_nw=eff_per_nw)
end

function set_obj!(pm, palma, nw_ids_int, λ; alpha)
    T = length(nw_ids_int)
    fair = alpha * sum(λ[ti] * (palma.σ[ti] * palma.top_sum[ti]) for ti in 1:T)
    eff  = (1.0 - alpha) * sum(λ[idx] * palma.eff_per_nw[nw] for (idx, nw) in enumerate(nw_ids_int))
    JuMP.@objective(pm.model, Min, fair + eff)
end

# ============================================================
# SETUP
# ============================================================
eng, math, lbs, critical_id = setup_network(case_path, LS_PERCENT;
    switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT)
mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS, center_at_nominal = CENTER_AT_NOMINAL)
nw_ids_sorted     = sort(collect(keys(mn_data["nw"])), by = x -> parse(Int, x))
nw_ids_int_sorted = parse.(Int, nw_ids_sorted)

println("\n", "="^70)
println("PROBE: per-period served-Palma trade-off — T=$N_PERIODS, λ=$PEAK_TIME_COSTS")
println("="^70)

for (variant, build_fn) in (
        ("RELAXED", pm -> FairLoadDelivery.build_mn_mc_mld_min_max(pm; alpha = 1.0)),
        ("INTEGER", pm -> FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm; alpha = 1.0)))
    println("\n", "-"^70, "\n### $variant MLD\n", "-"^70)
    mld = _PMD.instantiate_mc_model(mn_data, _PMD.LinDist3FlowPowerModel, build_fn;
        multinetwork = true, ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])
    JuMP.set_optimizer(mld.model, Gurobi.Optimizer)
    JuMP.set_optimizer_attribute(mld.model, "NonConvex", 2)
    JuMP.set_optimizer_attribute(mld.model, "MIPGap", 2e-2)
    JuMP.set_optimizer_attribute(mld.model, "TimeLimit", TIME_LIMIT)
    JuMP.set_optimizer_attribute(mld.model, "DualReductions", 0)  # distinguish INFEASIBLE from INF_OR_UNBD
    JuMP.set_optimizer_attribute(mld.model, "OutputFlag", 0)
    palma = add_palma_machinery_per_period!(mld; nw_ids_int = nw_ids_int_sorted, relax_binary = false)

    for alpha in PROBE_ALPHAS
        set_obj!(mld, palma, nw_ids_int_sorted, PEAK_TIME_COSTS; alpha = alpha)
        JuMP.optimize!(mld.model)
        term = JuMP.termination_status(mld.model)
        prim = JuMP.primal_status(mld.model)
        if prim != MOI.FEASIBLE_POINT
            println("  α=$alpha  term=$term  primal=$prim   → NO feasible point")
            continue
        end
        # per-period served-Palma definedness
        ndef = 0; cw = 0.0; pvals = Float64[]
        for (ti, nw) in enumerate(nw_ids_int_sorted)
            served = [JuMP.value(palma.pserved_period[nw][k]) for k in 1:palma.n]
            pt = palma_ratio_value(served)
            push!(pvals, pt)
            isfinite(pt) && (ndef += 1; cw += PEAK_TIME_COSTS[ti] * pt)
        end
        cw_str = (ndef == palma.T) ? string(round(cw, digits=3)) : "NaN (only $ndef/$(palma.T) defined)"
        total_shed = sum(sum(JuMP.value.(_PMD.var(mld, nw, :pshed, lid)))
                         for nw in nw_ids_int_sorted for lid in palma.load_ids)
        println("  α=$alpha  term=$term  defined=$ndef/$(palma.T)  ",
                "Σλ·Palma_served=$cw_str  total_shed=$(round(total_shed, digits=2))  ",
                "Palma_t=$(round.(pvals, digits=2))")
    end
end
println("\nProbe done.")
