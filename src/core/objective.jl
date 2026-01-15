
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

function objective_fairly_weighted_max_load_served(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, report::Bool=true, regularization::Float64=0.0)
    fair_load_weights = _PMD.var(pm, nw, :fair_load_weights)
    weighted_load_served = []
    regularization_term = []
    for d in _PMD.ids(pm, nw, :load)
        pd_var = _PMD.var(pm, nw, :pd)[d]
        push!(weighted_load_served, sum(fair_load_weights[d] .* pd_var))
        # Quadratic regularization to keep pd interior (fixes DiffOpt sensitivity computation)
        if regularization > 0
            push!(regularization_term, sum(pd_var .^ 2))
        end
    end
    #@info fair_load_weights
    #@info _PMD.var(pm, nw, :pd)
    if regularization > 0
        return JuMP.@objective(pm.model, Max,
            sum(weighted_load_served) - regularization * sum(regularization_term))
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

# Note: gini_index, jains_index, palma_ratio, and alpha_fairness
# are defined in src/implementation/other_fair_funcs.jl
# Removed duplicates from here to avoid method overwriting errors