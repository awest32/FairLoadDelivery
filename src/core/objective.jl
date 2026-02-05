

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
#     block_dist = sum((z_bern_block[i] - _PMD.var(pm, :z_block, i))^2 for i in keys(z_bern_block))
#    println("Block distance: $block_dist")
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

function objective_fairly_weighted_max_load_served(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    fair_load_weights = _PMD.var(pm, nw, :fair_load_weights)
    weighted_load_served = []
    for d in _PMD.ids(pm, nw, :load)
        push!(weighted_load_served, sum(fair_load_weights[d].*_PMD.var(pm, nw, :pd)[d]))
    end
    #@info fair_load_weights
    #@info _PMD.var(pm, nw, :pd)
    return JuMP.@objective(pm.model, Max,
    sum(weighted_load_served))
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

function gini_index(x)
    n = length(x)
    x = sort(x)
    n = length(x)
    gini = (2 * sum(i * x[i] for i in 1:n) / (n * sum(x))) - ((n + 1)/n)
    return gini
end

#Jain's index
function jains_index(x)
    n = length(x)
    sum_x = sum(x)
    sum_x2 = sum(xi^2 for xi in x)
    return (sum_x^2) / (n * sum_x2)
end

#Palma Ratio
function palma_ratio(x)
    x = load_served ./ load_ref
    sorted_x = sort(x)
    n = length(x)
    top_10_percent = sum(sorted_x[ceil(Int, 0.9n):end])
    bottom_40_percent = sum(sorted_x[1:floor(Int, 0.4n)])
    return top_10_percent / bottom_40_percent
end
#Alpha fairness for alpha=1
function alpha_fairness(x, alpha)
if alpha == 1
    return sum(log(xi) for xi in x)
    else
        return sum((xi^(1 - alpha)) / (1 - alpha) for xi in x)
    end
end

"""
    objective_equality_min(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Equality min (min-max fairness) objective for load shedding.
Minimizes the maximum load shed across all loads, ensuring no single load
bears a disproportionate burden.

The formulation adds an auxiliary variable t and constrains:
    pshed[d] <= t for all loads d
Then minimizes t.

This promotes fair distribution of load shedding across all loads.
"""
function objective_equality_min(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true)
    # Create auxiliary variable for max load shed
    t = JuMP.@variable(pm.model, base_name="max_shed", lower_bound=0)

    # Get load shed variable (pshed)
    pshed = _PMD.var(pm, nw, :pshed)

    # Constrain all load shed values to be <= t
    for d in _PMD.ids(pm, nw, :load)
        # Sum across phases for multi-phase loads
        JuMP.@constraint(pm.model, sum(pshed[d]) <= t)
    end

    # Minimize the maximum load shed
    return JuMP.@objective(pm.model, Min, t)
end

"""
    objective_equality_min_weighted(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Weighted equality min objective that also considers total load served.
Combines min-max fairness with efficiency by adding a small weight on total load served.

Objective: Min (t - epsilon * sum(pd))
where t is the max load shed and pd is load served.
"""
function objective_equality_min_weighted(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, epsilon::Float64=0.01, report::Bool=true)
    # Create auxiliary variable for max load shed
    t = JuMP.@variable(pm.model, base_name="max_shed", lower_bound=0)

    # Get load shed and served variables
    pshed = _PMD.var(pm, nw, :pshed)
    pd = _PMD.var(pm, nw, :pd)

    # Constrain all load shed values to be <= t
    for d in _PMD.ids(pm, nw, :load)
        JuMP.@constraint(pm.model, sum(pshed[d]) <= t)
    end

    # Total load served (for tie-breaking)
    total_served = sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load))

    # Minimize max shed, with small incentive to maximize total served
    return JuMP.@objective(pm.model, Min, t - epsilon * total_served)
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
function objective_proportional_fairness_mld(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, epsilon::Float64=1e-6, report::Bool=true)
    # Get load served variable (pd)
    pd = _PMD.var(pm, nw, :pd)

    # Build the log-sum objective
    # We add epsilon to avoid log(0) when a load is fully shed
    log_terms = []
    for d in _PMD.ids(pm, nw, :load)
        # Sum load served across phases, add epsilon for numerical stability
        push!(log_terms, log(sum(pd[d]) + epsilon))
    end

    # Maximize sum of log(load_served)
    return JuMP.@objective(pm.model, Max, sum(log_terms))
end

"""
    objective_proportional_fairness_mld_weighted(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Weighted proportional fairness objective that adds a small efficiency term.
Combines proportional fairness with total load served for tie-breaking.

Objective: Max (sum(log(pd + epsilon)) + delta * sum(pd))
"""
function objective_proportional_fairness_mld_weighted(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, epsilon::Float64=1e-6, delta::Float64=0.001, report::Bool=true)
    pd = _PMD.var(pm, nw, :pd)

    log_terms = []
    for d in _PMD.ids(pm, nw, :load)
        push!(log_terms, log(sum(pd[d]) + epsilon))
    end

    total_served = sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load))

    return JuMP.@objective(pm.model, Max, sum(log_terms) + delta * total_served)
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

    # Total load served (efficiency term)
    total_served = sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load))

    # Sum of squared served fractions (inequality penalty)
    # Lower sum of squares = higher Jain's index
    squared_fractions = []
    for d in _PMD.ids(pm, nw, :load)
        pd_ref = sum(_PMD.ref(pm, nw, :load, d)["pd"])
        if pd_ref > 1e-6
            # (pd / pd_ref)^2 = pd^2 / pd_ref^2
            push!(squared_fractions, sum(pd[d])^2 / (pd_ref^2))
        end
    end

    # Maximize total served minus penalty for inequality
    return JuMP.@objective(pm.model, Max, total_served - lambda * sum(squared_fractions))
end

"""
    objective_jain_mld_minvar(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Alternative Jain-promoting objective that minimizes variance of served fractions.

Uses auxiliary variable for mean served fraction and minimizes sum of squared deviations.
Includes efficiency term to break ties.
"""
function objective_jain_mld_minvar(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, epsilon::Float64=0.01, report::Bool=true)
    pd = _PMD.var(pm, nw, :pd)
    n_loads = length(_PMD.ids(pm, nw, :load))

    # Compute served fractions
    served_fractions = []
    for d in _PMD.ids(pm, nw, :load)
        pd_ref = sum(_PMD.ref(pm, nw, :load, d)["pd"])
        if pd_ref > 1e-6
            push!(served_fractions, sum(pd[d]) / pd_ref)
        end
    end

    # Auxiliary variable for mean served fraction
    mean_frac = JuMP.@variable(pm.model, base_name="mean_served_fraction")
    JuMP.@constraint(pm.model, mean_frac * length(served_fractions) == sum(served_fractions))

    # Sum of squared deviations from mean (variance * n)
    variance_term = sum((f - mean_frac)^2 for f in served_fractions)

    # Total served for efficiency
    total_served = sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load))

    # Minimize variance, with small incentive for efficiency
    return JuMP.@objective(pm.model, Min, variance_term - epsilon * total_served)
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
    objective_palma_mld_maxmin(pm::AbstractUnbalancedPowerModel; nw::Int=nw_id_default)

Alternative Palma-promoting objective: Maximize minimum served fraction.

This is the "max-min fairness on fractions" objective.
It ensures no load gets a disproportionately small fraction of its demand.

Different from equality_min which minimizes max absolute shed.
"""
function objective_palma_mld_maxmin(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, epsilon::Float64=0.001, report::Bool=true)
    pd = _PMD.var(pm, nw, :pd)

    # Auxiliary variable for minimum served fraction
    t_min = JuMP.@variable(pm.model, base_name="min_served_fraction", lower_bound=0, upper_bound=1)

    # Constrain t_min <= all served fractions
    for d in _PMD.ids(pm, nw, :load)
        pd_ref = sum(_PMD.ref(pm, nw, :load, d)["pd"])
        if pd_ref > 1e-6
            served_frac = sum(pd[d]) / pd_ref
            JuMP.@constraint(pm.model, t_min <= served_frac)
        end
    end

    # Total served for tie-breaking
    total_served = sum(sum(pd[d]) for d in _PMD.ids(pm, nw, :load))

    # Maximize minimum served fraction, with small incentive for total efficiency
    return JuMP.@objective(pm.model, Max, t_min + epsilon * total_served)
end


"""
Objective function: maximize weighted load served across all time periods.
Matches FairLoadDelivery's objective_fairly_weighted_max_load_served.
"""
function objective_mn_max_load_served(pm::_PMD.AbstractUnbalancedPowerModel)
    nw_ids = _PMD.nw_ids(pm)

    obj_expr = JuMP.AffExpr(0.0)

    for n in nw_ids
        for (i, load) in PMD.ref(pm, n, :load)
            z_demand = PMD.var(pm, n, :z_demand, i)
            pd = load["pd"]

            # Weight by load magnitude (serve more load = better)
            weight = get(load, "weight", 10.0)

            for (idx, p) in enumerate(pd)
                # Maximize weighted load served
                JuMP.add_to_expression!(obj_expr, weight * p * z_demand)
            end
        end
    end

    JuMP.@objective(pm.model, Max, obj_expr)
end
