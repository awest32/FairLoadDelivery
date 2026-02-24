

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

function objective_weighted_max_load_served(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    load_weights = _PMD.ref(pm, nw, :load_weights)
    weighted_load_served = []
    for d in _PMD.ids(pm, nw, :load)
        push!(weighted_load_served, sum(load_weights[d].*_PMD.var(pm, nw, :pd)[d]))
    end
    return JuMP.@objective(pm.model, Max,
    sum(weighted_load_served))
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
    if regularization > 0.0
        return JuMP.@objective(pm.model, Max,
            sum(weighted_load_served) - regularization * sum(regularization_term))
    else
        return JuMP.@objective(pm.model, Max,
            sum(weighted_load_served))
    end
end

function objective_fairly_weighted_max_load_served(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    fair_load_weights = _PMD.var(pm, nw, :fair_load_weights)
    weighted_load_served = []
    for d in _PMD.ids(pm, nw, :load)
        pd_var = _PMD.var(pm, nw, :pd)[d]
        push!(weighted_load_served, sum(fair_load_weights[d] .* pd_var))
    end
    #@info fair_load_weights
    @info _PMD.var(pm, nw, :pd)
    if isempty(_PMD.var(pm, nw, :pd))
        return JuMP.@objective(pm.model, Max, 0.0)
    else
        return JuMP.@objective(pm.model, Max,
            sum(weighted_load_served))
    end
end

function objective_fairly_weighted_min_load_shed(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    fair_load_weights = _PMD.var(pm, nw, :fair_load_weights)
    weighted_load_shed = []
    for d in _PMD.ids(pm, nw, :load)
        push!(weighted_load_shed, sum(fair_load_weights[d].*(_PMD.ref(pm, nw, :load, d)["pd"] - _PMD.var(pm, nw, :pd)[d])))
    end
    return JuMP.@objective(pm.model, Min,
    sum(weighted_load_shed))
end

function objective_proportional_fairness(pm::JuMP.Model; report::Bool=true)
   
    return JuMP.@objective((model), Max,
  		 sum(sum(log.(_PMD.var(pm, nw, :z_demand,i)*sum(_PMD.ref(pm, nw, :load, i)["pd"])) for i in _PMD.ids(pm, nw, :load)))
    )
end

function objective_equal_fairness(pm::JuMP.Model; report::Bool=true)
   alpha_val = 10.0
    return JuMP.@objective(pm, Max,
    #     +
        # sum((1-_PMD.var(pm, nw, :z_demand, i)) * sum(_PMD.ref(pm, nw, :load, i)["pd"])^2 for i in _PMD.ids(pm, nw, :load)) / 
        # (length(_PMD.ids(pm, nw, :load)) * sum((1-_PMD.var(pm, nw, :z_demand, i)) * sum( _PMD.ref(pm, nw, :load, i)["pd"]) for i in _PMD.ids(pm, nw, :load))^2)
            
         sum(sum((_PMD.var(pm, nw, :z_demand,i)*_PMD.ref(pm, nw, :load, i)["pd"]).^(1-alpha_val) for i in _PMD.ids(pm, nw, :load))./(1-alpha_val) )
    )
end

function objective_jain_fairness(pm::JuMP.Model; report::Bool=true)
    return JuMP.@objective(pm, Max,
        sum((1-_PMD.var(pm, nw, :z_demand, i)) * sum(_PMD.ref(pm, nw, :load, i)["pd"])^2 for i in _PMD.ids(pm, nw, :load)) / 
        (length(_PMD.ids(pm, nw, :load)) * sum((1-_PMD.var(pm, nw, :z_demand, i)) * sum( _PMD.ref(pm, nw, :load, i)["pd"]) for i in _PMD.ids(pm, nw, :load))^2)
    )
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
function objective_equality_min(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
     # Create auxiliary variable for the common shed percentage [0, 1]
    α = JuMP.@variable(pm.model, base_name="common_shed_fraction", lower_bound=0, upper_bound=1)

    # Get load shed variable (pshed) and load data
    pshed = _PMD.var(pm, nw, :pshed)

    # Constrain each load's shed amount to equal α × its demand
    for (i, load) in _PMD.ref(pm, nw, :load)
        load_demand = sum(load["pd"])  # Total demand across all phases
        
        # pshed[i] = α × demand[i]
        # This enforces equal percentage shedding
        JuMP.@constraint(pm.model, sum(pshed[i]) == α * load_demand)
    end

    # Minimize the common shed percentage
    return JuMP.@objective(pm.model, Min, α)
end

"""
    objective_min_max(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
Min-max fairness objective for load shedding.
Minimizes the maximum load shed across all loads, promoting fairness by ensuring no single load is disproportionately shed.

"""

function objective_min_max(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    # Create auxiliary variable for the maximum shed amount
    max_shed = JuMP.@variable(pm.model, base_name="max_shed", lower_bound=0)

    # Get load shed variable (pshed)
    pshed = _PMD.var(pm, nw, :pshed)

    # Constrain max_shed to be greater than or equal to each load's shed amount
    for (i, load) in _PMD.ref(pm, nw, :load)
        JuMP.@constraint(pm.model, max_shed >= sum(pshed[i]))
    end

    # Minimize the maximum shed amount
    return JuMP.@objective(pm.model, Min, max_shed)
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
                                             epsilon::Float64=1e-6)
    
    # Get load served variable (pd) - NOT pshed!
    pd = _PMD.var(pm, nw, :pd)
    
    # Build the log-sum objective
    log_terms = []
    
    for (i, load) in _PMD.ref(pm, nw, :load)
        # Total load served across phases
        # Add epsilon to avoid log(0) when load is fully shed
        served = sum(pd[i]) + epsilon
        
        # Proportional fairness maximizes log(served)
        push!(log_terms, log(served))
    end

    # Maximize sum of log(load_served)
    return JuMP.@objective(pm.model, Max, sum(log_terms))
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
function objective_jain_mld(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, lambda::Float64=10.0, report::Bool=true)
    pd = _PMD.var(pm, nw, :pd)
    pshed = _PMD.var(pm, nw, :pshed)
    # Total load served (efficiency term)
    total_served = sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load))

    # Sum of squared served fractions (inequality penalty)
    # Lower sum of squares = higher Jain's index
    squared_fractions = []
    for d in _PMD.ids(pm, nw, :load)
        pd_ref = sum(_PMD.ref(pm, nw, :load, d)["pd"])
        if pd_ref > 1e-6
            push!(squared_fractions, sum(pshed[d]/pd_ref)^2)
        end
    end

    pshed_sum = sum(sum(pshed[d])/sum(_PMD.ref(pm, nw, :load, d)["pd"]) for d in _PMD.ids(pm, nw, :load))
    jain = (pshed_sum^2) / (length(squared_fractions) * sum(squared_fractions))

    # Maximize total served minus penalty for inequality
    return JuMP.@objective(pm.model, Max, jain)
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
function objective_palma_mld(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, epsilon::Float64=0.001, report::Bool=true)
    pd = _PMD.var(pm, nw, :pd)

    # Auxiliary variables for max and min served fractions
    t_max = JuMP.@variable(pm.model, base_name="max_served_fraction")
    t_min = JuMP.@variable(pm.model, base_name="min_served_fraction", lower_bound=0)

    # Constrain t_max >= all served fractions, t_min <= all served fractions
    for d in _PMD.ids(pm, nw, :load)
        pd_ref = sum(_PMD.ref(pm, nw, :load, d)["pd"])
        if pd_ref > 1e-6
            served_frac = sum(pd[d]) / pd_ref
            JuMP.@constraint(pm.model, t_max >= served_frac)
            JuMP.@constraint(pm.model, t_min <= served_frac)
        end
    end

    # Total served for efficiency tie-breaking
    total_served = sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load))

    # Minimize range of served fractions, with small incentive for efficiency
    return JuMP.@objective(pm.model, Min, t_max - t_min - epsilon * total_served)
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
function objective_gini_mld(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, epsilon::Float64=0.001, report::Bool=true)
    pd = _PMD.var(pm, nw, :pd)

    # Filter loads with non-negligible demand (same threshold as Palma)
    valid_loads = []
    pd_refs = Dict()
    for d in _PMD.ids(pm, nw, :load)
        pd_ref = sum(_PMD.ref(pm, nw, :load, d)["pd"])
        if pd_ref > 1e-6
            push!(valid_loads, d)
            pd_refs[d] = pd_ref
        end
    end

    # Create pairs (i,j) with i < j for pairwise absolute differences
    n = length(valid_loads)
    pairs = [(valid_loads[i], valid_loads[j]) for i in 1:n for j in (i+1):n]

    # Auxiliary variables for |fi - fj|
    u = JuMP.@variable(pm.model, [1:length(pairs)], lower_bound=0, base_name="gini_abs_diff")

    for (k, (i, j)) in enumerate(pairs)
        fi = sum(pd[i]) / pd_refs[i]
        fj = sum(pd[j]) / pd_refs[j]
        JuMP.@constraint(pm.model, u[k] >= fi - fj)
        JuMP.@constraint(pm.model, u[k] >= fj - fi)
    end

    # Total served for efficiency tie-breaking
    total_served = sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load))

    # Minimize sum of pairwise absolute differences, with small incentive for efficiency
    return JuMP.@objective(pm.model, Min, sum(u) - epsilon * total_served)
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
                JuMP.add_to_expression!(obj_expr, weight * _PMD.var(pm, n, :pshed, i))
            #end
        end
    end

    JuMP.@objective(pm.model, Min, obj_expr)
end