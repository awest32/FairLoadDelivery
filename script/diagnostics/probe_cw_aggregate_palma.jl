#=
Feasibility / definability probe for the COST-WEIGHTED AGGREGATE served-Palma
trade-off (the formulation matched to the bilevel: Palma over cost-weighted
served totals u_i = Σ_t λ_t·pserved_{t,i}, one σ·bot=1).

Tests, at small T, both relaxed and integer MLD at a few α:
  • Gurobi termination + primal status (is integer feasible now?)
  • the cost-weighted aggregate served-Palma value (defined as long as each
    load is served sometime → bot40 of the weighted totals > 0)
  • uncosted aggregate served-Palma + total shed, as cross-checks

Machinery mirrors script/single_level/palma_trade_off_mn.jl's
add_palma_machinery_cw_aggregate! / set_palma_alpha_objective_cw!.
=#

using FairLoadDelivery
using PowerModelsDistribution
using Gurobi
using JuMP
import MathOptInterface as MOI

const _PMD = PowerModelsDistribution

case_name      = "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss"
LS_PERCENT     = 0.8
SELECTED_HOURS = [4, 12, 18]            # T=3
PEAK_STRESS    = 1.0
CENTER_AT_NOMINAL = true
PROBE_ALPHAS   = [0.0, 0.5, 1.0]
TIME_LIMIT     = 120

dir       = @__DIR__
case_path = joinpath(dir, case_name)
N_PERIODS = length(SELECTED_HOURS)
PEAK_TIME_COSTS = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                   for h in SELECTED_HOURS]

function compute_palma_indices(n::Int)
    n_bot = max(1, floor(Int, 0.4 * n)); n_top = max(1, ceil(Int, 0.1 * n))
    return collect((n - n_top + 1):n), collect(1:n_bot)
end
function palma_ratio_value(vals; eps_denom=1e-6)
    n=length(vals); s=sort(collect(vals)); ti,bi=compute_palma_indices(n)
    num=sum(s[i] for i in ti); den=sum(s[i] for i in bi)
    den < eps_denom ? Inf : num/den
end

function add_cw!(pm; nw_ids_int, λ, relax_binary=false)
    model = pm.model
    nw0 = nw_ids_int[1]; load_ids = sort(collect(_PMD.ids(pm, nw0, :load))); n = length(load_ids)
    T = length(nw_ids_int)
    P_period = Dict(nw => [sum(_PMD.ref(pm,nw,:load,i)["pd"]) for i in load_ids] for nw in nw_ids_int)
    total_demand = sum(sum(P_period[nw]) for nw in nw_ids_int)
    tix,bix = compute_palma_indices(n)
    pserved_unc = JuMP.@expression(model,[k=1:n], sum(sum(_PMD.var(pm,nw,:pd,load_ids[k])) for nw in nw_ids_int))
    a=Vector{Any}(undef,T); u=Vector{Any}(undef,T); pserved_period=Dict{Int,Any}()
    top_terms=Any[]; bot_terms=Any[]
    for (ti,nw) in enumerate(nw_ids_int)
        Pt=P_period[nw]
        ps = JuMP.@expression(model,[k=1:n], sum(_PMD.var(pm,nw,:pd,load_ids[k]))); pserved_period[nw]=ps
        a[ti] = relax_binary ? JuMP.@variable(model,[1:n,1:n],lower_bound=0,upper_bound=1,base_name="a_$ti") :
                               JuMP.@variable(model,[1:n,1:n],Bin,base_name="a_$ti")
        u[ti] = JuMP.@variable(model,[1:n,1:n],lower_bound=0,base_name="u_$ti")
        for i in 1:n, j in 1:n
            Pj=Pt[j]
            JuMP.@constraint(model, u[ti][i,j] >= ps[j] + a[ti][i,j]*Pj - Pj)
            JuMP.@constraint(model, u[ti][i,j] <= a[ti][i,j]*Pj)
            JuMP.@constraint(model, u[ti][i,j] <= ps[j])
        end
        for i in 1:n; JuMP.@constraint(model, sum(a[ti][i,j] for j in 1:n)==1); end
        for j in 1:n; JuMP.@constraint(model, sum(a[ti][i,j] for i in 1:n)==1); end
        sv = JuMP.@expression(model,[i=1:n], sum(u[ti][i,j] for j in 1:n))
        for k in 1:(n-1); JuMP.@constraint(model, sv[k] <= sv[k+1]); end
        push!(top_terms, λ[ti]*sum(sv[i] for i in tix))
        push!(bot_terms, λ[ti]*sum(sv[i] for i in bix))
    end
    top_sum = JuMP.@expression(model, sum(top_terms))   # Σ_t λ_t top10(srv_t)
    bot_sum = JuMP.@expression(model, sum(bot_terms))   # Σ_t λ_t bot40(srv_t)
    σ = JuMP.@variable(model, base_name="sigma", lower_bound=1e-8)
    JuMP.@constraint(model, σ*bot_sum == 1.0)
    eff = JuMP.@expression(model, sum(sum(sum(_PMD.var(pm,nw,:pshed,d)) for d in _PMD.ids(pm,nw,:load)) for nw in nw_ids_int)/total_demand)
    return (n=n, load_ids=load_ids, nw_ids_int=nw_ids_int, P_period=P_period,
            pserved_period=pserved_period, pserved_unc=pserved_unc, tix=tix, bix=bix,
            a=a, u=u, top_sum=top_sum, bot_sum=bot_sum, σ=σ, eff=eff)
end
set_obj!(pm, P; alpha) = JuMP.@objective(pm.model, Min, alpha*(P.σ*P.top_sum) + (1-alpha)*P.eff)

eng, math, lbs, cid = setup_network(case_path, LS_PERCENT;
    switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT)
mn = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours=SELECTED_HOURS, peak_stress=PEAK_STRESS, center_at_nominal=CENTER_AT_NOMINAL)
nw_ids = parse.(Int, sort(collect(keys(mn["nw"])), by=x->parse(Int,x)))

println("\n", "="^70, "\nPROBE: cost-weighted AGGREGATE served-Palma — T=$N_PERIODS, λ=$PEAK_TIME_COSTS\n", "="^70)
for (variant, bf) in (
        ("RELAXED", pm->FairLoadDelivery.build_mn_mc_mld_min_max(pm; alpha=1.0)),
        ("INTEGER", pm->FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm; alpha=1.0)))
    println("\n", "-"^70, "\n### $variant MLD\n", "-"^70)
    mld = _PMD.instantiate_mc_model(mn, _PMD.LinDist3FlowPowerModel, bf;
        multinetwork=true, ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
    JuMP.set_optimizer(mld.model, Gurobi.Optimizer)
    for (k,v) in (("NonConvex",2),("MIPGap",1e-2),("TimeLimit",TIME_LIMIT),("DualReductions",0),("OutputFlag",0))
        JuMP.set_optimizer_attribute(mld.model, k, v)
    end
    P = add_cw!(mld; nw_ids_int=nw_ids, λ=PEAK_TIME_COSTS, relax_binary=false)

    # Efficiency warm-start (mirror production): solve the same MLD with a pure
    # efficiency objective (NO Palma machinery → plain MIP), push pshed/switch/
    # block + the Palma (a,u,σ) starts.
    eff_bf = (variant == "RELAXED") ?
        (pm -> FairLoadDelivery.build_mn_mc_mld_min_max(pm; alpha=0.0)) :
        (pm -> FairLoadDelivery.build_mn_mc_mld_min_max_integer(pm; alpha=0.0))
    mld_e = _PMD.instantiate_mc_model(mn, _PMD.LinDist3FlowPowerModel, eff_bf;
        multinetwork=true, ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
    JuMP.set_optimizer(mld_e.model, Gurobi.Optimizer)
    for (k,v) in (("MIPGap",1e-2),("TimeLimit",120),("OutputFlag",0)); JuMP.set_optimizer_attribute(mld_e.model,k,v); end
    JuMP.optimize!(mld_e.model)
    if JuMP.primal_status(mld_e.model) == MOI.FEASIBLE_POINT
        λmap = Dict(nw => PEAK_TIME_COSTS[ti] for (ti,nw) in enumerate(nw_ids))
        for nw in nw_ids, lid in P.load_ids
            JuMP.set_start_value.(_PMD.var(mld,nw,:pshed,lid), JuMP.value.(_PMD.var(mld_e,nw,:pshed,lid)))
        end
        for nw in nw_ids
            ev=_PMD.var(mld_e,nw); pv=_PMD.var(mld,nw)
            for s in (:switch_state,:z_block,:z_demand); haskey(ev,s) && JuMP.set_start_value.(pv[s], JuMP.value.(ev[s])); end
        end
        nb=max(1,floor(Int,0.4*P.n)); bot_warm=0.0
        for (ti,nw) in enumerate(nw_ids)
            ps_t=[sum(_PMD.ref(mld_e,nw,:load,lid)["pd"]) - sum(JuMP.value.(_PMD.var(mld_e,nw,:pshed,lid))) for lid in P.load_ids]
            perm=sortperm(ps_t); as=zeros(P.n,P.n); for (i,j) in enumerate(perm); as[i,j]=1.0; end; us=as.*reshape(ps_t,1,:)
            for i in 1:P.n, j in 1:P.n; JuMP.set_start_value(P.a[ti][i,j],as[i,j]); JuMP.set_start_value(P.u[ti][i,j],us[i,j]); end
            bot_warm += PEAK_TIME_COSTS[ti]*sum(ps_t[perm[k]] for k in 1:nb)
        end
        JuMP.set_start_value(P.σ, 1.0/max(bot_warm,1e-8))
        println("  [warm-start applied: eff status=$(JuMP.termination_status(mld_e.model))]")
    else
        println("  [warm-start FAILED: $(JuMP.termination_status(mld_e.model))]")
    end

    for alpha in PROBE_ALPHAS
        set_obj!(mld, P; alpha=alpha); JuMP.optimize!(mld.model)
        term=JuMP.termination_status(mld.model); prim=JuMP.primal_status(mld.model)
        if prim != MOI.FEASIBLE_POINT
            println("  α=$alpha  term=$term  primal=$prim  → NO feasible point"); continue
        end
        cwt=0.0; cwb=0.0
        for (ti,nw) in enumerate(nw_ids)
            s=sort([JuMP.value(P.pserved_period[nw][k]) for k in 1:P.n])
            cwt += PEAK_TIME_COSTS[ti]*sum(s[i] for i in P.tix); cwb += PEAK_TIME_COSTS[ti]*sum(s[i] for i in P.bix)
        end
        cw = cwb < 1e-6 ? Inf : cwt/cwb
        unc = palma_ratio_value([JuMP.value(P.pserved_unc[k]) for k in 1:P.n])
        shed = sum(sum(JuMP.value.(_PMD.var(mld,nw,:pshed,lid))) for nw in nw_ids for lid in P.load_ids)
        model_fair = JuMP.value(P.σ)*JuMP.value(P.top_sum)
        println("  α=$alpha  term=$term  cost-wtd Palma(Σλtop/Σλbot)=$(round(cw,digits=3))  ",
                "model(σ·top)=$(round(model_fair,digits=3))  uncosted-Palma=$(round(unc,digits=3))  shed=$(round(shed,digits=2))")
    end
end
println("\nProbe done.")
