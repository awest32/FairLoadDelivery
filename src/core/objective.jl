

"""
    objective_mc_min_fuel_cost_pwl_voll(pm::AbstractUnbalancedPowerModel)

Fuel cost minimization objective with piecewise linear terms including VOLL for lloadshed
"""
function objective_mc_min_fuel_cost_pwl_voll(pm::_PMD.AbstractUnbalancedPowerModel; report::Bool=true)
    model = _PMD.check_gen_cost_models(pm)

    if model == 1
        return objective_mc_min_fuel_cost_pwl(pm; report=report)
    elseif model == 2
        return objective_mc_min_fuel_cost_polynomial(pm; report=report)
    else
        error("Only cost models of types 1 and 2 are supported at this time, given cost model type of $(model)")
    end
end


"""
    objective_mc_min_fuel_cost_pwl(pm::AbstractUnbalancedPowerModel)

Fuel cost minimization objective with piecewise linear terms
"""
function objective_mc_min_fuel_cost_pwl(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    _PMD.objective_mc_variable_pg_cost(pm; report=report)
    
    return JuMP.@objective(pm.model, Min,
        sum(
            sum( gen_cost[(nw,i)] for i in _PMD.ids(pm, nw, :gen)))
            +
        sum(
            sum( _PMD.ref(pm, nw, :load, i)["pd"].*(1- _PMD.var(pm, nw, :z_demand, i)) for i in _PMD.ids(pm, nw, :load)))
   )
    # return JuMP.@objective(pm.model, Min,
    #     sum(
    #         sum( _PMD.var(pm, n, :pg_cost, i) for (i,gen) in nw_ref[:gen])
    #     for (n, nw_ref) in _PMD.nws(pm))
    #         +
    #     voll * sum(
    #         sum( _PMD.ref(pm, nw, :load, i)["pd"].*(1- _PMD.var(pm, n, :z_demand, i)) for (i,load) in nw_ref[:load])
    #     for (n, nw_ref) in _PMD.nws(pm)) 
    # )
end


"""
    objective_mc_min_fuel_cost_polynomial(pm::AbstractUnbalancedPowerModel)

Fuel cost minimization objective for polynomial terms
"""
function objective_mc_min_fuel_cost_polynomial(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    order = _PMD.calc_max_cost_index(pm.data)-1

    if order <= 2
        return _objective_mc_min_fuel_cost_polynomial_linquad_fair(pm; report=report)
    else
        return _objective_mc_min_fuel_cost_polynomial_nl(pm; report=report)
    end
end

"gen connections adaptation of min fuel cost polynomial linquad objective"
function _objective_mc_min_fuel_cost_polynomial_linquad(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    gen_cost = Dict()

    for (n, nw_ref) in _PMD.nws(pm)
        for (i,gen) in nw_ref[:gen]
            pg = sum(_PMD.var(pm, n, :pg, i))

            if length(gen["cost"]) == 1
                gen_cost[(n,i)] = gen["cost"][1]
            elseif length(gen["cost"]) == 2
                gen_cost[(n,i)] = gen["cost"][1]*pg + gen["cost"][2]
            elseif length(gen["cost"]) == 3
                gen_cost[(n,i)] = gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
            else
                gen_cost[(n,i)] = 0.0
            end
        end
    end

    load_weights = Dict(
                l =>  1.0 for l in _PMD.ids(pm, nw, :load)
            )

    voll = Dict(
        l => 1.0 for l in _PMD.ids(pm, nw, :load)
    )

    #println("VOLL = $voll")
    return JuMP.@objective(pm.model, Min,
        sum(
            sum( gen_cost[(nw,i)] for i in _PMD.ids(pm, nw, :gen)))
            +
            sum( voll[i] * load_weights[i] * (1-_PMD.var(pm, nw, :z_demand, i)) * sum( _PMD.ref(pm, nw, :load, i)["pd"]) for i in _PMD.ids(pm, nw, :load))
    )
end

"gen connections adaptation of min fuel cost polynomial linquad objective"
function _objective_mc_min_fuel_cost_polynomial_linquad_fair(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    gen_cost = Dict()

    for (n, nw_ref) in _PMD.nws(pm)
        for (i,gen) in nw_ref[:gen]
            pg = sum(_PMD.var(pm, n, :pg, i))

            if length(gen["cost"]) == 1
                gen_cost[(n,i)] = gen["cost"][1]
            elseif length(gen["cost"]) == 2
                gen_cost[(n,i)] = gen["cost"][1]*pg + gen["cost"][2]
            elseif length(gen["cost"]) == 3
                gen_cost[(n,i)] = gen["cost"][1]*pg^2 + gen["cost"][2]*pg + gen["cost"][3]
            else
                gen_cost[(n,i)] = 0.0
            end
        end
    end

    load_weights = Dict(
                l =>  1.0 for l in _PMD.ids(pm, nw, :load)
            )

    # voll = Dict(
    #     l => 0.001 for l in _PMD.ids(pm, nw, :load)
    # )
    voll = Dict(
        l => 1 for l in _PMD.ids(pm, nw, :load)
    )
    alpha_val = 10
    #println("VOLL = $voll")
    "pshed[i] = (1-_PMD.var(pm, nw, :z_demand, i)) * sum( _PMD.ref(pm, nw, :load, i)[pd])"
    return JuMP.@objective(pm.model, Min,
        # sum(sum( gen_cost[(nw,i)] for i in _PMD.ids(pm, nw, :gen)))
        #    +
        sum(voll[i] * load_weights[i] * (1-_PMD.var(pm, nw, :z_demand, i)) * sum( _PMD.ref(pm, nw, :load, i)["pd"]) for i in _PMD.ids(pm, nw, :load))
        #       -
        #    sum((1-_PMD.var(pm, nw, :z_demand, i)) * sum(_PMD.ref(pm, nw, :load, i)["pd"])^2 for i in _PMD.ids(pm, nw, :load)) / 
        #    (length(_PMD.ids(pm, nw, :load)) * sum((1-_PMD.var(pm, nw, :z_demand, i)) * sum( _PMD.ref(pm, nw, :load, i)["pd"]) for i in _PMD.ids(pm, nw, :load))^2)
            
         #  - sum((_PMD.var(pm, nw, :z_demand,i)*_PMD.ref(pm, nw, :load, i)["pd"].^(1-alpha_val)) for i in _PMD.ids(pm, nw, :load))./(1-alpha_val) 
		#- sum(sum(log.(_PMD.var(pm, nw, :z_demand,i)*sum(_PMD.ref(pm, nw, :load, i)["pd"])) for i in _PMD.ids(pm, nw, :load)))
     #sum(sum( gen_cost[(nw,i)] for i in _PMD.ids(pm, nw, :gen)))
      #      +
        #sum(load_weights[i] * (1-_PMD.var(pm, nw, :z_demand, i)) * sum( _PMD.ref(pm, nw, :load, i)["pd"]) for i in _PMD.ids(pm, nw, :load))

        )

    #     for (n, nw_ref) in _PMD.nws(pm)) 
    #     sum(
    #         sum( sum(_PMD.ref(pm, n, :load, i)["pd"]) - sum(_PMD.var(pm, n, :pd, i)) for (i,load) in nw_ref[:load])
    #     for (n, nw_ref) in _PMD.nws(pm)) 
    
end

""
function _objective_mc_min_fuel_cost_polynomial_nl(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    gen_cost = Dict()
    for (n, nw_ref) in _PMD.nws(pm)
        for (i,gen) in _PMD.nw_ref[:gen]
            pg = sum( _PMD.var(pm, n, :pg, i)[c] for c in gen["connections"] )

            cost_rev = _PMD.reverse(gen["cost"])
            if length(cost_rev) == 1
                gen_cost[(n,i)] = JuMP.@expression(pm.model, cost_rev[1])
            elseif length(cost_rev) == 2
                gen_cost[(n,i)] = JuMP.@expression(pm.model, cost_rev[1] + cost_rev[2]*pg)
            elseif length(cost_rev) == 3
                gen_cost[(n,i)] = JuMP.@expression(pm.model, cost_rev[1] + cost_rev[2]*pg + cost_rev[3]*pg^2)
            elseif length(cost_rev) >= 4
                cost_rev_nl = cost_rev[4:end]
                gen_cost[(n,i)] = JuMP.@expression(pm.model, cost_rev[1] + cost_rev[2]*pg + cost_rev[3]*pg^2 + sum( v*pg^(d+2) for (d,v) in enumerate(_PMD.cost_rev_nl)) )
            else
                gen_cost[(n,i)] = JuMP.@expression(pm.model, 0.0)
            end
        end
    end

    return JuMP.@objective(pm.model, Min,
        sum(
            sum( gen_cost[(n,i)] for i in _PMD.ids(pm, n, :gen)))
            +
        sum(
            sum( _PMD.ref(pm, n, :load, i)["pd"].*(1- _PMD.var(pm, n, :z_demand, i)) for i in _PMD.ids(pm, n, :load)))
   )
end


function objective_min_dist_rounded(pm::JuMP.Model; z_bern_switch::Dict{Int,Int64})#, z_bern_block::Dict{Int,Int64})
    switch_dist = sum( (z_bern_switch[i] - _PMD.var(pm, :switch_state, i))^2 for i in keys(z_bern_switch))
    println("Switch distance: $switch_dist")
    return JuMP.@NLobjective(pm.model, Min, switch_dist)# + block_dist)
end


function objective_fairly_weighted_max_load_served_regd(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true, regularization::Float64=0.0)
    fair_load_weights = _PMD.var(pm, nw, :fair_load_weights)
    weighted_load_served = []
    regularization_term = []
    for d in _PMD.ids(pm, nw, :load)
        pd_var = _PMD.var(pm, nw, :pd)[d]
        push!(weighted_load_served, sum(fair_load_weights[d] .* pd_var))
        # Quadratic regularization to keep pd interior (fixes DiffOpt sensitivity computation)
        if regularization > 0.0
            push!(regularization_term, sum(pd_var .^ 2))
        end
    end
    #@info fair_load_weights
    #@info _PMD.var(pm, nw, :pd)
    if isempty(_PMD.var(pm, nw, :pd))
         return JuMP.@objective(pm.model, Max, 0.0)
    else
        if regularization > 0.0
            return JuMP.@objective(pm.model, Max,
                sum(weighted_load_served) - regularization * sum(regularization_term))
        else
            return JuMP.@objective(pm.model, Max,
                sum(weighted_load_served))
        end
    end
end

function objective_fairly_weighted_max_load_served(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    fair_load_weights = _PMD.var(pm, nw, :fair_load_weights)
    weighted_load_served = []
    for d in _PMD.ids(pm, nw, :load)
        pd_var = _PMD.var(pm, nw, :pd)[d]
        push!(weighted_load_served, sum(pd_var))
    end
    #@info fair_load_weights
    #@info _PMD.var(pm, nw, :pd)
    # if isempty(_PMD.var(pm, nw, :pd))
    #     return JuMP.@objective(pm.model, Max, 0.0)
    # else
    #     return 
    JuMP.@objective(pm.model, Max,
            sum(weighted_load_served))
    #end
end

function objective_fairly_weighted_min_load_shed(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    fair_load_weights = _PMD.var(pm, nw, :fair_load_weights)
    weighted_load_shed = []
    for d in _PMD.ids(pm, nw, :load)
        push!(weighted_load_shed,  sum(_PMD.var(pm,nw, :fair_load_weights,i) * _PMD.var(pm, nw, :pshed, i) for i in _PMD.ids(pm, nw, :load)) 
    )#sum(fair_load_weights[d].*(_PMD.ref(pm, nw, :load, d)["pd"] - _PMD.var(pm, nw, :pd)[d])))
    end
    return JuMP.@objective(pm.model, Min,
    sum(weighted_load_shed))
end

function objective_fair_max_load_served(pm::_PMD.AbstractUnbalancedPowerModel, fair::String; nw::Int=_IM.nw_id_default, report::Bool=true)
    weighted_load_served = []
    load_served = []
    load_ref = []
    for d in _PMD.ids(pm, nw, :load)
        push!(weighted_load_served, sum((1-_PMD.var(pm, nw, :z_demand, d)) * _PMD.var(pm, nw, :pd)[d]))
        push!(load_served, sum(((_PMD.var(pm, nw, :z_demand, d)) * _PMD.var(pm, nw, :pd)[d])))
        push!(load_ref, sum(_PMD.ref(pm, nw, :load, d)["pd"]))
    end

    served_array = collect(load_served./load_ref)    
    jain = jains_index(served_array)    
    proportional = alpha_fairness(served_array, 1)
    gini = gini_index(served_array) # Placeholder for Gini index calculation
    efficiency = alpha_fairness(served_array, 0.5)

    if fair == "jain"
        return JuMP.@objective(pm.model, Max,
        sum(weighted_load_served) + jain)
    else 
        if fair == "proportional"
            return JuMP.@objective(pm.model, Max,
      		  proportional + sum(weighted_load_served))
        else 
            if fair == "gini"
            return JuMP.@objective(pm.model, Max,
      		  gini + sum(weighted_load_served))
            else
                error("Fairness criterion $(fair) not recognized.")
            end
        end
    end
end

function objective_fairly_weighted_max_load_served_with_penalty(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
   fair_load_weights = _PMD.var(pm, nw, :fair_load_weights)
   z_demand = _PMD.var(pm, nw, :z_demand)
    weighted_load_served = []
    for d in _PMD.ids(pm, nw, :load)
        push!(weighted_load_served, sum(fair_load_weights[d].*_PMD.var(pm, nw, :pd)[d]))
    end
       # Add this term to your objective function:
    penalty_weight = 1000.0  # Tune this
    binary_penalty = sum(z_demand[i] * (1 - z_demand[i]) for (i, load) in _PMD.ref(pm, nw, :load))

    return JuMP.@objective(pm.model, Min,
    sum(weighted_load_served) + penalty_weight * binary_penalty)
end


"""
    objective_equality_min(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Equality min (min-max fairness) objective for load shedding.

This promotes fair distribution of load shedding across all loads.
"""
function objective_equality_min(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true, reg::Float64=1e-4)
    # Create auxiliary variable for the common shed amount
    α = JuMP.@variable(pm.model, base_name="common_shed_amount", lower_bound=0)

    # Get load shed variable (pshed) and load data
    pshed = _PMD.var(pm, nw, :pshed)
    load_prioritization_weights = _PMD.var(pm, nw, :fair_load_weights)
    # Constrain each load's shed amount to equal α
    for (i, load) in _PMD.ref(pm, nw, :load)
        JuMP.@constraint(pm.model, sum(load_prioritization_weights[i] * pshed[i]) == α)
    end

    total_demand = sum(sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
    reg_term = reg * sum(pshed[d] for d in _PMD.ids(pm, nw, :load)) / total_demand
    return JuMP.@objective(pm.model, Min, α + reg_term)
end

"""
    objective_min_max(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
Min-max fairness objective for load shedding.
Minimizes the maximum load shed across all loads, promoting fairness by ensuring no single load is disproportionately shed.

"""

function objective_min_max(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true, reg::Float64=1e-4)
    # Create auxiliary variable for the maximum shed amount
    max_shed = JuMP.@variable(pm.model, base_name="max_shed", lower_bound=0)

    # Get load shed variable (pshed)
    pshed = _PMD.var(pm, nw, :pshed)

    # Get the load prioritization weights
    load_prioritization_weights = _PMD.var(pm, nw, :fair_load_weights)
    # Constrain max_shed to be greater than or equal to each load's shed amount
    for (i, load) in _PMD.ref(pm, nw, :load)
        JuMP.@constraint(pm.model, max_shed >= sum(load_prioritization_weights[i] * pshed[i]))
    end

    total_demand = sum(sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
    reg_term = reg * sum(pshed[d] for d in _PMD.ids(pm, nw, :load)) / total_demand
    return JuMP.@objective(pm.model, Min, max_shed + reg_term)
end

"""
    objective_proportional_fairness_mld(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Proportional fairness (Nash bargaining) objective for load shedding.
Maximizes the sum of log(load_served) across all loads, which corresponds to
the Nash bargaining solution and alpha-fairness with alpha=1.

The formulation maximizes: sum_d log(pd[d] + epsilon)
where pd is load served and epsilon is a small constant to avoid log(0).

This objective balances efficiency with fairness - it tends to equalize
the percentage of load served across loads rather than absolute amounts.
"""
function objective_proportional_fairness_mld(pm::_PMD.AbstractUnbalancedPowerModel;
                                             nw::Int=_IM.nw_id_default,
                                             epsilon::Float64=1e-6,
                                             reg::Float64=1e-4)

    # Get load served variable (pd) - NOT pshed!
    pd = _PMD.var(pm, nw, :pd)
    load_prioritization_weights = _PMD.var(pm, nw, :fair_load_weights)
    # Build the log-sum objective
    log_terms = []

    for (i, load) in _PMD.ref(pm, nw, :load)
        # Total load served across phases
        # Add epsilon to avoid log(0) when load is fully shed
        served = sum(load_prioritization_weights[i] * pd[i]) + epsilon

        # Proportional fairness maximizes log(served)
        push!(log_terms, log(served))
    end

    total_demand = sum(sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
    reg_term = reg * sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load)) / total_demand
    return JuMP.@objective(pm.model, Max, sum(log_terms) + reg_term)
end

"""
    objective_jain_mld(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Jain's fairness index promoting objective for load shedding.

Jain's index = (sum x)^2 / (n * sum x^2) where x_i is the served fraction for load i.
Maximizing Jain's index with fixed total served is equivalent to minimizing sum(x^2).

This objective maximizes total load served while penalizing inequality in served fractions:
    Max sum(pd) - lambda * sum((pd[d] / pd_ref[d])^2)

The lambda parameter controls the efficiency-fairness tradeoff:
- lambda = 0: pure efficiency (max total served)
- lambda > 0: promotes equality in served fractions (higher Jain's index)
"""
function objective_jain_mld(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true, reg::Float64=1e-4)
    pd = _PMD.var(pm, nw, :pd)
    load_prioritization_weights = _PMD.var(pm, nw, :fair_load_weights)
    # Jain's index on absolute pd: (Σ pd_i)² / (n * Σ pd_i²)
    n_loads = length(collect(_PMD.ids(pm, nw, :load)))
    pd_sum = sum(sum(load_prioritization_weights[d] * pd[d]) for d in _PMD.ids(pm, nw, :load))
    pd_sum_sq = sum(sum(load_prioritization_weights[d] * pd[d])^2 for d in _PMD.ids(pm, nw, :load))

    jain = (pd_sum^2) / (n_loads * pd_sum_sq)

    total_demand = sum(sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
    reg_term = reg * sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load)) / total_demand
    return JuMP.@objective(pm.model, Max, jain + reg_term)
end

"""
    objective_palma_mld(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Palma ratio promoting objective for load shedding.

The Palma ratio = (top 10% served) / (bottom 40% served).
Since percentile selection requires sorting (non-smooth), we approximate
by minimizing the range of served fractions: max(fraction) - min(fraction).

This promotes compression of the served fraction distribution,
which tends to reduce the Palma ratio.

Objective: Min (t_max - t_min) - epsilon * sum(pd)
where t_max >= all served fractions and t_min <= all served fractions.
"""
function objective_palma_mld(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true, reg::Float64=1e-4)
    pshed = _PMD.var(pm, nw, :pshed)
    load_prioritization_weights = _PMD.var(pm, nw, :fair_load_weights)
    # Collect load ids and demands for upper bounds
    load_ids = sort(collect(_PMD.ids(pm, nw, :load)))
    n = length(load_ids)

    # Upper bound on each load's total pshed (= total demand)
    P = Dict(d => sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in load_ids)

    # pshed_total[j] = sum(pshed[load_ids[j]]) for indexing 1:n
    pshed_total = [sum(pshed[load_ids[j]]) for j in 1:n]

    # Binary permutation matrix a[i,j]: sorted[i] = Σ_j a[i,j] * pshed_total[j]
    a = JuMP.@variable(pm.model, [1:n, 1:n], Bin, base_name="palma_perm")
    u = JuMP.@variable(pm.model, [1:n, 1:n], lower_bound=0, base_name="palma_u")

    # Doubly stochastic constraints
    for i in 1:n
        JuMP.@constraint(pm.model, sum(a[i, j] for j in 1:n) == 1)
    end
    for j in 1:n
        JuMP.@constraint(pm.model, sum(a[i, j] for i in 1:n) == 1)
    end

    # McCormick envelopes: u[i,j] = a[i,j] * pshed_total[j]
    for i in 1:n, j in 1:n
        Pj = P[load_ids[j]]
        JuMP.@constraint(pm.model, u[i, j] >= pshed_total[j] + a[i, j] * Pj - Pj)
        JuMP.@constraint(pm.model, u[i, j] <= a[i, j] * Pj)
        JuMP.@constraint(pm.model, u[i, j] <= pshed_total[j])
    end

    # Sorted values (ascending)
    sorted = [sum(u[i, j] for j in 1:n) for i in 1:n]
    for k in 1:n-1
        JuMP.@constraint(pm.model, sorted[k] <= sorted[k+1])
    end

    # Palma indices: top 10% and bottom 40%
    n_top = max(1, ceil(Int, 0.1 * n))
    n_bot = max(1, floor(Int, 0.4 * n))
    top_sum = sum(sorted[i] for i in (n - n_top + 1):n)
    bot_sum = sum(sorted[i] for i in 1:n_bot)

    # Charnes-Cooper: min σ * top_sum  s.t. σ * bot_sum = 1
    σ = JuMP.@variable(pm.model, base_name="palma_sigma", lower_bound=1e-8)
    JuMP.@constraint(pm.model, σ * bot_sum == 1.0)

    total_demand = sum(sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
    reg_term = reg * sum(pshed[d] for d in _PMD.ids(pm, nw, :load)) / total_demand
    return JuMP.@objective(pm.model, Min, σ * top_sum + reg_term)
end

"""
Gini coefficient promoting objective for load shedding.

The Gini coefficient measures inequality via mean absolute difference:
Gini = Σᵢ Σⱼ |fᵢ - fⱼ| / (2n Σ fᵢ)

Since the denominator involves decision variables, we minimize the numerator
(sum of pairwise absolute differences of served fractions) as a linearizable proxy.

Objective: Min Σᵢ<ⱼ uᵢⱼ - epsilon * sum(pd)
where uᵢⱼ ≥ |fᵢ - fⱼ| for served fractions fᵢ = pd[i]/pd_ref[i].
"""
function objective_gini_mld(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    pshed = _PMD.var(pm, nw, :pshed)

    # Collect valid loads
    valid_loads = collect(_PMD.ids(pm, nw, :load))
    n = length(valid_loads)

    # Create pairs (i,j) with i < j for pairwise absolute differences
    pairs = [(valid_loads[i], valid_loads[j]) for i in 1:n for j in (i+1):n]

    # Auxiliary variables for |pshed_i - pshed_j|
    u = JuMP.@variable(pm.model, [1:length(pairs)], lower_bound=0, base_name="gini_abs_diff")

    for (k, (i, j)) in enumerate(pairs)
        si = sum(pshed[i])
        sj = sum(pshed[j])
        JuMP.@constraint(pm.model, u[k] >= si - sj)
        JuMP.@constraint(pm.model, u[k] >= sj - si)
    end

    # Minimize sum of pairwise absolute differences of absolute shed
    return JuMP.@objective(pm.model, Min, sum(u))
end

"""
Objective function: maximize weighted load served across all time periods.
Matches FairLoadDelivery's objective_fairly_weighted_max_load_served.
"""
function objective_mn_max_load_served(pm::_PMD.AbstractUnbalancedPowerModel)
    nw_ids = _PMD.nw_ids(pm)

    obj_expr = JuMP.AffExpr(0.0)

    for n in nw_ids
        for (i, load) in _PMD.ref(pm, n, :load)

            # Weight by load magnitude (serve more load = better)
            weight = get(load, "weight", 10.0)

            #for (idx, p) in enumerate(pd)
                # Maximize weighted load served
                JuMP.add_to_expression!(obj_expr, weight * _PMD.var(pm, n, :pd, i))
            #end
        end
    end

    JuMP.@objective(pm.model, Max, obj_expr)
end

"""
Multiperiod objective: maximize weighted load served across all time periods
using per-period fair_load_weights with regularization.
"""
function objective_mn_fairly_weighted_max_load_served_regd(pm::_PMD.AbstractUnbalancedPowerModel; regularization::Float64=0.05)
    nw_ids = sort(collect(_PMD.nw_ids(pm)))

    weighted_load_served = []
    regularization_terms = []

    for n in nw_ids
        # Per-period weights
        fair_load_weights_nw = _PMD.var(pm, n, :fair_load_weights)
        for d in _PMD.ids(pm, n, :load)
            pd_var = _PMD.var(pm, n, :pd)[d]
            push!(weighted_load_served, sum(fair_load_weights_nw[d] .* pd_var))
            if regularization > 0.0
                push!(regularization_terms, sum(pd_var .^ 2))
            end
        end
    end

    if isempty(weighted_load_served)
        return JuMP.@objective(pm.model, Max, 0.0)
    else
        if regularization > 0.0
            return JuMP.@objective(pm.model, Max,
                sum(weighted_load_served) - regularization * sum(regularization_terms))
        else
            return JuMP.@objective(pm.model, Max,
                sum(weighted_load_served))
        end
    end
end

"""
Multiperiod efficiency objective: minimize Σ_t λ_t * Σ_i w_{t,i} * pshed_{t,i}.
Mirrors objective_fairly_weighted_min_load_shed across all time periods.
peak_time_costs is a per-period weight; pass empty to weight all periods equally.
"""
function objective_mn_fairly_weighted_min_load_shed(pm::_PMD.AbstractUnbalancedPowerModel;
                                                    peak_time_costs::Vector{<:Real}=Float64[])
    nw_ids = sort(collect(_PMD.nw_ids(pm)))
    T = length(nw_ids)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"

    # fair_load_weights is a JuMP Parameter — product with pshed is a QuadExpr at
    # assembly time, so use the same sum-of-generators pattern as the single-period
    # objective_fairly_weighted_min_load_shed rather than add_to_expression! on an AffExpr.
    weighted_shed_terms = []
    for (idx, n) in enumerate(nw_ids)
        push!(weighted_shed_terms,
            λ[idx] * sum(_PMD.var(pm, n, :fair_load_weights, i) * _PMD.var(pm, n, :pshed, i)
                         for i in _PMD.ids(pm, n, :load)))
    end
    return JuMP.@objective(pm.model, Min, sum(weighted_shed_terms))
end

"""
Multiperiod min-max: sum of per-period max weighted shed,
optionally weighted by peak_time_costs.

With `alpha ∈ [0, 1]`: convex combination of efficiency and fairness terms.
- alpha=0: pure efficiency (min Σ λ_t · Σ_i pshed_{t,i} / total_demand_t)
- alpha=1: pure min-max fairness
- `reg` is an orthogonal small efficiency regularizer kept for back-compat.
"""
function objective_mn_min_max(pm::_PMD.AbstractUnbalancedPowerModel;
                              peak_time_costs::Vector{<:Real}=Float64[],
                              reg::Float64=1e-4, alpha::Float64=1.0)
    nw_ids = sort(collect(_PMD.nw_ids(pm)))
    T = length(nw_ids)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1], got $alpha"

    obj = JuMP.AffExpr(0.0)
    for (idx, n) in enumerate(nw_ids)
        max_shed_n = JuMP.@variable(pm.model, base_name="max_shed_nw_$(n)", lower_bound=0)
        pshed = _PMD.var(pm, n, :pshed)
        w = _PMD.var(pm, n, :fair_load_weights)
        for (i, load) in _PMD.ref(pm, n, :load)
            JuMP.@constraint(pm.model, max_shed_n >= sum(w[i] * pshed[i]))
        end
        total_demand_n = sum(sum(_PMD.ref(pm, n, :load, d)["pd"]) for d in _PMD.ids(pm, n, :load))
        eff_term = sum(pshed[d] for d in _PMD.ids(pm, n, :load)) / total_demand_n
        JuMP.add_to_expression!(obj, λ[idx] * (alpha * max_shed_n + (1.0 - alpha) * eff_term + reg * eff_term))
    end
    return JuMP.@objective(pm.model, Min, obj)
end

"""
Multiperiod proportional fairness: Σ_t λ_t * Σ_i log(w_{t,i} * pd_{t,i} + ε).

With `alpha ∈ [0, 1]`: Max α·Σ λ_t·Σ_i log(served) + (1-α)·Σ λ_t·served/total_demand.
- alpha=0: pure efficiency (Max served = Min shed)
- alpha=1: pure proportional fairness
"""
function objective_mn_proportional_fairness_mld(pm::_PMD.AbstractUnbalancedPowerModel;
                                                peak_time_costs::Vector{<:Real}=Float64[],
                                                epsilon::Float64=1e-6,
                                                reg::Float64=1e-4, alpha::Float64=1.0)
    nw_ids = sort(collect(_PMD.nw_ids(pm)))
    T = length(nw_ids)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1], got $alpha"

    log_terms = []
    eff_terms = []
    for (idx, n) in enumerate(nw_ids)
        pd = _PMD.var(pm, n, :pd)
        w = _PMD.var(pm, n, :fair_load_weights)
        for i in _PMD.ids(pm, n, :load)
            served = sum(w[i] * pd[i]) + epsilon
            push!(log_terms, λ[idx] * log(served))
        end
        total_demand_n = sum(sum(_PMD.ref(pm, n, :load, d)["pd"]) for d in _PMD.ids(pm, n, :load))
        push!(eff_terms, λ[idx] * sum(sum(pd[d]) for d in _PMD.ids(pm, n, :load)) / total_demand_n)
    end
    return JuMP.@objective(pm.model, Max,
        alpha * sum(log_terms) + (1.0 - alpha) * sum(eff_terms) + reg * sum(eff_terms))
end

"""
Multiperiod Jain: Σ_t λ_t * Jain_t where Jain_t is computed from weighted served in period t.

With `alpha ∈ [0, 1]`: Max α·Σ λ_t·Jain_t + (1-α)·Σ λ_t·served/total_demand.
- alpha=0: pure efficiency
- alpha=1: pure Jain fairness
"""
function objective_mn_jain_mld(pm::_PMD.AbstractUnbalancedPowerModel;
                               peak_time_costs::Vector{<:Real}=Float64[],
                               reg::Float64=1e-4, alpha::Float64=1.0)
    nw_ids = sort(collect(_PMD.nw_ids(pm)))
    T = length(nw_ids)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1], got $alpha"

    jain_terms = []
    eff_terms = []
    for (idx, n) in enumerate(nw_ids)
        pd = _PMD.var(pm, n, :pd)
        w = _PMD.var(pm, n, :fair_load_weights)
        n_loads = length(collect(_PMD.ids(pm, n, :load)))
        pd_sum = sum(sum(w[d] * pd[d]) for d in _PMD.ids(pm, n, :load))
        pd_sum_sq = sum(sum(w[d] * pd[d])^2 for d in _PMD.ids(pm, n, :load))
        push!(jain_terms, λ[idx] * (pd_sum^2) / (n_loads * pd_sum_sq))

        total_demand_n = sum(sum(_PMD.ref(pm, n, :load, d)["pd"]) for d in _PMD.ids(pm, n, :load))
        push!(eff_terms, λ[idx] * sum(sum(pd[d]) for d in _PMD.ids(pm, n, :load)) / total_demand_n)
    end
    return JuMP.@objective(pm.model, Max,
        alpha * sum(jain_terms) + (1.0 - alpha) * sum(eff_terms) + reg * sum(eff_terms))
end

"""
Multiperiod Palma: sum of per-period σ_t * top_sum_t (Charnes-Cooper reformulation
of the Palma ratio), optionally weighted by peak_time_costs. Builds a binary
permutation matrix per period to produce the sorted pshed vector.
"""
function objective_mn_palma_mld(pm::_PMD.AbstractUnbalancedPowerModel;
                                peak_time_costs::Vector{<:Real}=Float64[],
                                reg::Float64=1e-4, alpha::Float64=1.0)
    nw_ids = sort(collect(_PMD.nw_ids(pm)))
    T = length(nw_ids)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1], got $alpha"

    # σ * top_sum is bilinear, so accumulate the per-period palma terms as a Julia array
    # and let JuMP build the final QuadExpr in the @objective call.
    palma_terms = []
    eff_terms = []

    for (idx, nw) in enumerate(nw_ids)
        pshed = _PMD.var(pm, nw, :pshed)
        load_ids = sort(collect(_PMD.ids(pm, nw, :load)))
        n = length(load_ids)

        P = Dict(d => sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in load_ids)
        pshed_total = [sum(pshed[load_ids[j]]) for j in 1:n]

        a = JuMP.@variable(pm.model, [1:n, 1:n], Bin, base_name="palma_perm_nw_$(nw)")
        u = JuMP.@variable(pm.model, [1:n, 1:n], lower_bound=0, base_name="palma_u_nw_$(nw)")

        for i in 1:n
            JuMP.@constraint(pm.model, sum(a[i, j] for j in 1:n) == 1)
        end
        for j in 1:n
            JuMP.@constraint(pm.model, sum(a[i, j] for i in 1:n) == 1)
        end
        for i in 1:n, j in 1:n
            Pj = P[load_ids[j]]
            JuMP.@constraint(pm.model, u[i, j] >= pshed_total[j] + a[i, j] * Pj - Pj)
            JuMP.@constraint(pm.model, u[i, j] <= a[i, j] * Pj)
            JuMP.@constraint(pm.model, u[i, j] <= pshed_total[j])
        end

        sorted = [sum(u[i, j] for j in 1:n) for i in 1:n]
        for k in 1:n-1
            JuMP.@constraint(pm.model, sorted[k] <= sorted[k+1])
        end

        n_top = max(1, ceil(Int, 0.1 * n))
        n_bot = max(1, floor(Int, 0.4 * n))
        top_sum = sum(sorted[i] for i in (n - n_top + 1):n)
        bot_sum = sum(sorted[i] for i in 1:n_bot)

        σ = JuMP.@variable(pm.model, base_name="palma_sigma_nw_$(nw)", lower_bound=1e-8)
        JuMP.@constraint(pm.model, σ * bot_sum == 1.0)

        push!(palma_terms, λ[idx] * σ * top_sum)

        total_demand_n = sum(sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
        push!(eff_terms, λ[idx] * sum(pshed[d] for d in _PMD.ids(pm, nw, :load)) / total_demand_n)
    end
    return JuMP.@objective(pm.model, Min,
        alpha * sum(palma_terms) + (1.0 - alpha) * sum(eff_terms) + reg * sum(eff_terms))
end
