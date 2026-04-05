
""
function solve_mc_mld_switch_relaxed(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_switchable_relaxed; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

function solve_mc_mld_switch_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_switchable_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve the multiperiod MLD problem using FairLoadDelivery formulation.
"""
function solve_mn_mc_mld_switch_relaxed(data::Dict{String,Any}, solver)
    return _PMD.solve_mc_model(
        data,
        _PMD.LinDist3FlowPowerModel,
        solver,
        build_mn_mc_mld_switch_relaxed;
        multinetwork=true,
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!]
    )
end

"""
Solve MLD with equality_min (equality min fairness) objective (relaxed).
Minimizes the maximum percent of load shed while setting
that percentage to be euqal for all loads shed across all loads.
"""
function solve_mc_mld_equality_min(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_equality_min; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with equality_min (equality min fairness) objective (integer).
Minimizes the maximum percent of load shed while setting
that percentage to be euqal for all loads shed across all loads with binary switch/block variables.
"""
function solve_mc_mld_equality_min_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_equality_min_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end


"""
Solve MLD with equality_min (equality min fairness) objective (relaxed).
Minimizes the maximum percent of load shed while setting
that percentage to be euqal for all loads shed across all loads.
"""
function solve_mc_mld_min_max(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_min_max; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with equality_min (equality min fairness) objective (integer).
Minimizes the maximum percent of load shed while setting
that percentage to be euqal for all loads shed across all loads with binary switch/block variables.
"""
function solve_mc_mld_min_max_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_min_max_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with proportional fairness (Nash bargaining) objective (relaxed).
Maximizes sum of log(load_served) across all loads.
"""
function solve_mc_mld_proportional_fairness(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_proportional_fairness; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with proportional fairness (Nash bargaining) objective (integer).
Maximizes sum of log(load_served) with binary switch/block variables.
"""
function solve_mc_mld_proportional_fairness_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_proportional_fairness_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with Jain's index promoting objective (relaxed).
Maximizes total served while penalizing inequality in served fractions.
"""
function solve_mc_mld_jain(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_jain; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with Jain's index promoting objective (integer).
"""
function solve_mc_mld_jain_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_jain_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with Palma ratio promoting objective (relaxed).
Minimizes range of served fractions (max - min).
"""
function solve_mc_mld_palma(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_palma; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with Palma ratio promoting objective (integer).
"""
function solve_mc_mld_palma_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_palma_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with Gini coefficient promoting objective (relaxed).
Minimizes sum of pairwise absolute differences of served fractions.
"""
function solve_mc_mld_gini(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_gini; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

"""
Solve MLD with Gini coefficient promoting objective (integer).
"""
function solve_mc_mld_gini_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_gini_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

function solve_mc_mld_traditional(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_traditional; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

function solve_mc_mld_shed_random_round(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_shedding_random_rounding; ref_extensions=[ref_add_load_blocks!], kwargs...)
end
function solve_mc_mld_shed_random_round_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_shedding_random_rounding_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end
function solve_mc_mld_shed_implicit_diff(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_shedding_implicit_diff; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

function solve_mc_mld_weight_update(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_weight_update; ref_extensions=[ref_update_weights!], kwargs...)
end

"""
Build the multiperiod MLD problem using FairLoadDelivery constraints.
This creates variables and constraints for each time period with proper
radiality, switch ampacity, and load block constraints.
"""
function build_mn_mc_mld_switch_relaxed(pm::_PMD.AbstractUnbalancedPowerModel)
    # Get all network IDs and sort them
    nw_ids = sort(collect(_PMD.nw_ids(pm)))

    println("Building multiperiod FLDP with $(length(nw_ids)) periods...")

    for n in nw_ids
        println("  Building period $n...")

        # Variables (same as FairLoadDelivery's build_mc_mld_switchable_relaxed)
        _PMD.variable_mc_bus_voltage_indicator(pm; nw=n, relax=true)
        variable_mc_bus_voltage_magnitude_sqr_on_off(pm; nw=n)

        _PMD.variable_mc_branch_power(pm; nw=n)
        #_PMD.variable_mc_branch_current(pm; nw=n)
        _PMD.variable_mc_switch_power(pm; nw=n)
        _PMD.variable_mc_switch_state(pm; nw=n, relax=true)
        _PMD.variable_mc_shunt_indicator(pm; nw=n, relax=true)
        _PMD.variable_mc_transformer_power(pm; nw=n)

        _PMD.variable_mc_gen_indicator(pm; nw=n, relax=true)
        _PMD.variable_mc_generator_power_on_off(pm; nw=n)

        _PMD.variable_mc_load_indicator(pm; nw=n, relax=true)
        FairLoadDelivery.variable_mc_load_shed(pm; nw=n)

        FairLoadDelivery.variable_block_indicator(pm; nw=n, relax=true)
        FairLoadDelivery.variable_mc_fair_load_weights(pm; nw=n)

        # Constraints
        _PMD.constraint_mc_model_current(pm; nw=n)

        for i in _PMD.ids(pm, n, :ref_buses)
            _PMD.constraint_mc_theta_ref(pm, i; nw=n)
        end

        _PMD.constraint_mc_bus_voltage_on_off(pm; nw=n)

        for i in _PMD.ids(pm, n, :gen)
            _PMD.constraint_mc_generator_power(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :bus)
            FairLoadDelivery.constraint_mc_power_balance_shed(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :branch)
            _PMD.constraint_mc_power_losses(pm, i; nw=n)
            FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm, i; nw=n)
            _PMD.constraint_mc_voltage_angle_difference(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :switch)
            FairLoadDelivery.constraint_switch_state_on_off(pm, i; nw=n, relax=true)
            FairLoadDelivery.constraint_mc_switch_ampacity(pm, i; nw=n)
            constraint_model_switch_voltage_magnitude_difference_fld(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :transformer)
            _PMD.constraint_mc_transformer_power(pm, i; nw=n)
        end

        # FairLoadDelivery-specific constraints
        FairLoadDelivery.constraint_source_voltage_bounds(pm; nw=n)
        FairLoadDelivery.constraint_mc_isolate_block(pm; nw=n)
        FairLoadDelivery.constraint_radial_topology(pm; nw=n)

        FairLoadDelivery.constraint_block_budget(pm; nw=n)
        FairLoadDelivery.constraint_switch_budget(pm; nw=n)

        FairLoadDelivery.constraint_load_shed_definition(pm; nw=n)

        FairLoadDelivery.constraint_connect_block_load(pm; nw=n)
        FairLoadDelivery.constraint_connect_load_bus(pm; nw=n)
        FairLoadDelivery.constraint_connect_load_bus(pm; nw=n)
        FairLoadDelivery.constraint_connect_block_gen(pm; nw=n)
        FairLoadDelivery.constraint_connect_block_voltage(pm; nw=n)
        FairLoadDelivery.constraint_connect_block_shunt(pm; nw=n)
    end

    # Objective: maximize weighted load served across all periods
    objective_mn_max_load_served(pm)
end

"""
Solve the multiperiod MLD problem with implicit differentiation (DiffOpt).
One global set of fair_load_weights shared across all time periods.
"""
function solve_mn_mc_mld_shed_implicit_diff(data::Dict{String,Any}, solver; kwargs...)
    return _PMD.solve_mc_model(
        data,
        _PMD.LinDist3FlowPowerModel,
        solver,
        build_mn_mc_mld_shedding_implicit_diff;
        multinetwork=true,
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!],
        kwargs...
    )
end

"""
Build multiperiod MLD with implicit differentiation (DiffOpt).
Creates ONE global set of fair_load_weights shared across all T periods.
Per-period: all other variables (pshed, pd, switches, blocks, etc.)
Registers per-period pshed in model dictionary for Jacobian computation.
"""
function build_mn_mc_mld_shedding_implicit_diff(pm::_PMD.AbstractUBFModels)
    # Replace model with DiffOpt-wrapped optimizer for implicit differentiation
    pm.model = JuMP.Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))

    nw_ids = sort(collect(_PMD.nw_ids(pm)))
    first_nw = nw_ids[1]

    println("Building multiperiod implicit diff FLDP with $(length(nw_ids)) periods...")

    for (idx, n) in enumerate(nw_ids)
        println("  Building period $n...")

        # Variables (same as single-period build_mc_mld_shedding_implicit_diff)
        _PMD.variable_mc_bus_voltage_indicator(pm; nw=n, relax=true)
        variable_mc_bus_voltage_magnitude_sqr_on_off(pm; nw=n)

        _PMD.variable_mc_branch_power(pm; nw=n)
        _PMD.variable_mc_switch_power(pm; nw=n)
        _PMD.variable_mc_switch_state(pm; nw=n, relax=true)
        _PMD.variable_mc_shunt_indicator(pm; nw=n, relax=true)
        _PMD.variable_mc_transformer_power(pm; nw=n)

        _PMD.variable_mc_gen_indicator(pm; nw=n, relax=true)
        _PMD.variable_mc_generator_power_on_off(pm; nw=n)

        _PMD.variable_mc_storage_power_mi_on_off(pm; nw=n, relax=true, report=true)

        _PMD.variable_mc_load_indicator(pm; nw=n, relax=true)
        variable_mc_load_shed(pm; nw=n)

        variable_block_indicator(pm; nw=n, relax=true)

        # GLOBAL fair_load_weights: create only for the first period
        # All periods share the same weight parameters
        if idx == 1
            variable_mc_fair_load_weights(pm; nw=n)
        end

        # Constraints
        _PMD.constraint_mc_model_current(pm; nw=n)

        for i in _PMD.ids(pm, n, :ref_buses)
            _PMD.constraint_mc_theta_ref(pm, i; nw=n)
        end

        _PMD.constraint_mc_bus_voltage_on_off(pm; nw=n)

        for i in _PMD.ids(pm, n, :gen)
            _PMD.constraint_mc_generator_power(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :bus)
            constraint_mc_power_balance_shed(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :storage)
            _PMD.constraint_storage_state(pm, i; nw=n)
            _PMD.constraint_storage_complementarity_mi(pm, i; nw=n)
            _PMD.constraint_mc_storage_losses(pm, i; nw=n)
            _PMD.constraint_mc_storage_thermal_limit(pm, i; nw=n)
            constraint_mc_storage_on_off(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :branch)
            _PMD.constraint_mc_power_losses(pm, i; nw=n)
            constraint_model_voltage_magnitude_difference_fld(pm, i; nw=n)
            _PMD.constraint_mc_voltage_angle_difference(pm, i; nw=n)
            _PMD.constraint_mc_thermal_limit_from(pm, i; nw=n)
            _PMD.constraint_mc_thermal_limit_to(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :switch)
            constraint_switch_state_on_off(pm, i; nw=n, relax=true)
            _PMD.constraint_mc_switch_thermal_limit(pm, i; nw=n)
            constraint_mc_switch_ampacity(pm, i; nw=n)
            constraint_model_switch_voltage_magnitude_difference_fld(pm, i; nw=n)
        end

        for i in _PMD.ids(pm, n, :transformer)
            _PMD.constraint_mc_transformer_power(pm, i; nw=n)
        end

        # FairLoadDelivery-specific constraints
        constraint_source_voltage_bounds(pm; nw=n)
        constraint_mc_isolate_block(pm; nw=n)
        constraint_radial_topology(pm; nw=n)
        constraint_mc_block_energization_consistency_bigm(pm, n)

        constraint_block_budget(pm; nw=n)
        constraint_switch_budget(pm; nw=n)

        constraint_load_shed_definition(pm; nw=n)

        constraint_connect_block_load(pm; nw=n)
        constraint_connect_load_bus(pm; nw=n)
        constraint_connect_block_gen(pm; nw=n)
        constraint_connect_block_voltage(pm; nw=n)
        constraint_connect_block_shunt(pm; nw=n)
        constraint_connect_block_storage(pm; nw=n)
    end

    # Register per-period pshed in model dictionary for Jacobian computation
    for n in nw_ids
        pm.model[Symbol("pshed_nw_$(n)")] = _PMD.var(pm, n)[:pshed]
    end
    # Store nw_ids list in model for retrieval during Jacobian computation
    pm.model[:nw_ids] = nw_ids

    # Objective: maximize weighted load served across all periods with global weights
    # Regularization keeps pd interior for DiffOpt sensitivity computation
    objective_mn_fairly_weighted_max_load_served_regd(pm; regularization=0.05)
end

function build_mc_mld_shedding_implicit_diff(pm::_PMD.AbstractUBFModels)
    pm.model = JuMP.Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    # JuMP.set_attribute(pm.model, "hsllib", HSL_jll.libhsl_path)
    # JuMP.set_attribute(pm.model, "linear_solver", "ma27")
    #@info pm.model typeof(pm.model)
    
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	#_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

    # # The on-off variable is making the solution error at the report statement in the variable function
   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)
 

    _PMD.variable_mc_load_indicator(pm; relax=true)
    # #variable_mc_demand_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)



   	 _PMD.constraint_mc_model_current(pm)

     for i in _PMD.ids(pm, :ref_buses)
         _PMD.constraint_mc_theta_ref(pm, i)
     end

    _PMD.constraint_mc_bus_voltage_on_off(pm)    
     
    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
        #constraint_mc_gen_power_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        #_PMD.constraint_mc_model_voltage_magnitude_difference(pm,i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        #constraint_mc_model_voltage_magnitude_difference_block(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)

        _PMD.constraint_mc_thermal_limit_from(pm, i)
        _PMD.constraint_mc_thermal_limit_to(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
         # The switch thermal limit is not implemented in PowerModelsDistribution yet
         _PMD.constraint_mc_switch_thermal_limit(pm, i)
        # # The ampacity constraint is similar but instead of p^2 + q^2 <= w * s^2, it is p^2 + q^2 <= z * w * s^2
        constraint_mc_switch_ampacity(pm, i)
        constraint_model_switch_voltage_magnitude_difference_fld(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)
    #constraint_mc_radiality(pm)
    constraint_mc_block_energization_consistency_bigm(pm)

    # Must be disabled if there is no generation in the network
    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)
    #constraint_shed_single_load(pm)
   
    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Regularization keeps pd interior, fixing DiffOpt sensitivity computation
    # See script/reformulation/debug/ for analysis
    objective_fairly_weighted_max_load_served_regd(pm; regularization=0.05)
    #objective_fairly_weighted_max_load_served_with_penalty(pm)
    #objective_fairly_weighted_min_load_shed(pm)
end

"Multinetwork load shedding problem for Branch Flow model "
function build_mc_mld_shedding_random_rounding(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	#_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

    # # The on-off variable is making the solution error at the report statement in the variable function
   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)
 

    _PMD.variable_mc_load_indicator(pm; relax=true)
    # #variable_mc_demand_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)



   	 _PMD.constraint_mc_model_current(pm)

     for i in _PMD.ids(pm, :ref_buses)
         _PMD.constraint_mc_theta_ref(pm, i)
     end

    _PMD.constraint_mc_bus_voltage_on_off(pm)    
     
    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
        #constraint_mc_gen_power_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        #_PMD.constraint_mc_model_voltage_magnitude_difference(pm,i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        #constraint_mc_model_voltage_magnitude_difference_block(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)

        _PMD.constraint_mc_thermal_limit_from(pm, i)
        _PMD.constraint_mc_thermal_limit_to(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
         # The switch thermal limit is not implemented in PowerModelsDistribution yet
         _PMD.constraint_mc_switch_thermal_limit(pm, i)
        # # The ampacity constraint is similar but instead of p^2 + q^2 <= w * s^2, it is p^2 + q^2 <= z * w * s^2
        constraint_mc_switch_ampacity(pm, i)
        constraint_model_switch_voltage_magnitude_difference_fld(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end
    
    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block_ref(pm)
    constraint_radial_topology(pm)
    #constraint_mc_radiality(pm)
    constraint_mc_block_energization_consistency_bigm(pm)

    # Must be disabled if there is no generation in the network
    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)
    #constraint_shed_single_load(pm)
   
    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    #constraint_set_block_state_rounded(pm)
    constraint_set_switch_state_rounded(pm)
    
    #_PMD.objective_mc_min_load_setpoint_delta_simple(pm)
    #_PMD.objective_mc_min_fuel_cost(pm)
    #objective_mc_min_fuel_cost_pwl_voll(pm)
    objective_fairly_weighted_max_load_served(pm)
    #objective_fair_max_load_served(pm,"jain")
    #objective_fairly_weighted_max_load_served_with_penalty(pm)
    #objective_fairly_weighted_min_load_shed(pm)

end

function build_mc_mld_shedding_random_rounding_integer(pm::_PMD.AbstractUBFModels)
 _PMD.variable_mc_bus_voltage_indicator(pm; relax=false)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	#_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

    # # The on-off variable is making the solution error at the report statement in the variable function
   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true)
 

    _PMD.variable_mc_load_indicator(pm; relax=true)
    # #variable_mc_demand_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=false)
    variable_mc_fair_load_weights(pm)



   	 _PMD.constraint_mc_model_current(pm)

     for i in _PMD.ids(pm, :ref_buses)
         _PMD.constraint_mc_theta_ref(pm, i)
     end

    _PMD.constraint_mc_bus_voltage_on_off(pm)    
     
    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
        #constraint_mc_gen_power_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        #_PMD.constraint_mc_model_voltage_magnitude_difference(pm,i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        #constraint_mc_model_voltage_magnitude_difference_block(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)

        _PMD.constraint_mc_thermal_limit_from(pm, i)
        _PMD.constraint_mc_thermal_limit_to(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=false)
         # The switch thermal limit is not implemented in PowerModelsDistribution yet
         _PMD.constraint_mc_switch_thermal_limit(pm, i)
        # # The ampacity constraint is similar but instead of p^2 + q^2 <= w * s^2, it is p^2 + q^2 <= z * w * s^2
        constraint_mc_switch_ampacity(pm, i)
        constraint_model_switch_voltage_magnitude_difference_fld(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end
    
    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block_ref(pm)
    constraint_radial_topology(pm)
    # #constraint_mc_radiality(pm)
     constraint_mc_block_energization_consistency_bigm(pm)

    # Must be disabled if there is no generation in the network
    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)
    #constraint_shed_single_load(pm)
   
    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # #constraint_set_block_state_rounded(pm)
    constraint_set_switch_state_rounded(pm)
    
    # #_PMD.objective_mc_min_load_setpoint_delta_simple(pm)
    # #_PMD.objective_mc_min_fuel_cost(pm)
    # #objective_mc_min_fuel_cost_pwl_voll(pm)
    objective_fairly_weighted_max_load_served(pm)
    # #objective_fair_max_load_served(pm,"jain")
    # #objective_fairly_weighted_max_load_served_with_penalty(pm)
    # #objective_fairly_weighted_min_load_shed(pm)

end

"Load shedding problem for Branch Flow model with load blocks and radiality constraints and variables "
function build_mc_mld_switchable_integer(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=false)
 	#variable_mc_bus_voltage_magnitude_sqr_on_off(pm)
    variable_mc_bus_voltage_magnitude_sqr_on_off(pm)


    _PMD.variable_mc_branch_power(pm)
	#_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=false)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=false)
    _PMD.variable_mc_generator_power_on_off(pm)

    # # The on-off variable is making the solution error at the report statement in the variable function
   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=false, report=false)
 

    _PMD.variable_mc_load_indicator(pm; relax=false)
    # #variable_mc_demand_indicator(pm; relax=true)
    variable_mc_load_shed(pm)
    constraint_load_shed_definition(pm)

    variable_block_indicator(pm; relax=false)
    variable_mc_fair_load_weights(pm)



   	_PMD.constraint_mc_model_current(pm)

     for i in _PMD.ids(pm, :ref_buses)
         _PMD.constraint_mc_theta_ref(pm, i)
     end

    _PMD.constraint_mc_bus_voltage_on_off(pm)
    
     
    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end
    # for (i,bus) in _PMD.ref(pm, :bus)
    #     if bus["name"] == "646"
    #        println("Going to fix bus 646 mismatch")
    #         constraint_fix_bus_terminal_mismatch(pm,i)
    #     end
    # end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
         #_PMD.constraint_mc_model_voltage_magnitude_difference(pm,i)
         FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        #constraint_mc_model_voltage_magnitude_difference_block(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)

        _PMD.constraint_mc_thermal_limit_from(pm, i)
        _PMD.constraint_mc_thermal_limit_to(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=false)
         # The switch thermal limit is not implemented in PowerModelsDistribution yet
         _PMD.constraint_mc_switch_thermal_limit(pm, i)
        # # The ampacity constraint is similar but instead of p^2 + q^2 <= w * s^2, it is p^2 + q^2 <= z * w * s^2
        constraint_mc_switch_ampacity(pm, i)
        constraint_model_switch_voltage_magnitude_difference_fld(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)
    # #constraint_mc_radiality(pm)
    constraint_mc_block_energization_consistency_bigm(pm)

    # Must be disabled if there is no generation in the network
    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    # constraint_set_block_state_rounded(pm)
    # constraint_set_switch_state_rounded(pm)
   
   
    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)
    # #constraint_open_switches(pm)
    # #constraint_gen_event_simple(pm, ls_percent=0.5)
 

    
    # #_PMD.objective_mc_min_load_setpoint_delta_simple(pm)
    # #_PMD.objective_mc_min_fuel_cost(pm)
    # #objective_mc_min_fuel_cost_pwl_voll(pm)
    objective_fairly_weighted_max_load_served(pm)
    # #objective_fair_max_load_served(pm,"jain")
    # #objective_fairly_weighted_max_load_served_with_penalty(pm)
    # #objective_fairly_weighted_min_load_shed(pm)
    # JuMP._CONSTRAINT_LIMIT_FOR_PRINTING[] = 1E9
    # open("integer_mld_mdel.txt", "w") do io
    #     redirect_stdout(io) do
    #         print(pm)
    #     end
    # end

end

"Load shedding problem for Branch Flow model with load blocks and radiality constraints and variables "
function build_mc_mld_switchable_relaxed(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	#variable_mc_bus_voltage_magnitude_sqr_on_off(pm)
    variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	#_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

    # # The on-off variable is making the solution error at the report statement in the variable function
   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)
 

    _PMD.variable_mc_load_indicator(pm; relax=true)
    # #variable_mc_demand_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)



   	_PMD.constraint_mc_model_current(pm)


     for i in _PMD.ids(pm, :ref_buses)
         _PMD.constraint_mc_theta_ref(pm, i)
     end

    _PMD.constraint_mc_bus_voltage_on_off(pm)
    
     
    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
        #constraint_mc_gen_power_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end
    # for (i,bus) in _PMD.ref(pm, :bus)
    #     if bus["name"] == "646"
    #        println("Going to fix bus 646 mismatch")
    #         constraint_fix_bus_terminal_mismatch(pm,i)
    #     end
    # end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
                    #_PMD.constraint_mc_model_voltage_magnitude_difference(pm,i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
                    #constraint_mc_model_voltage_magnitude_difference_block(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)

        # The thermal limits are not activated
        #_PMD.constraint_mc_thermal_limit_from(pm, i)
        #_PMD.constraint_mc_thermal_limit_to(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
        # The switch thermal limit is not implemented in PowerModelsDistribution yet
        # The ampacity constraint is similar but instead of p^2 + q^2 <= w * s^2, it is p^2 + q^2 <= z * w * s^2
        constraint_mc_switch_ampacity(pm, i)
        
        # Not activated
        constraint_model_switch_voltage_magnitude_difference_fld(pm, i)
    end

    #Not activated
    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end
    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)
    # constraint_mc_radiality(pm)
    constraint_mc_block_energization_consistency_bigm(pm)

    # Must be disabled if there is no generation in the network
    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    # constraint_set_block_state_rounded(pm)
    # constraint_set_switch_state_rounded(pm)
   
    constraint_load_shed_definition(pm)
    #constraint_shed_single_load(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)
    #constraint_open_switches(pm)
    #constraint_gen_event_simple(pm, ls_percent=0.5)
 

    
    #_PMD.objective_mc_min_load_setpoint_delta_simple(pm)
    #_PMD.objective_mc_min_fuel_cost(pm)
    #objective_mc_min_fuel_cost_pwl_voll(pm)
    objective_fairly_weighted_max_load_served(pm)
    #objective_fair_max_load_served(pm,"jain")
    #objective_fairly_weighted_max_load_served_with_penalty(pm)
    #objective_fairly_weighted_min_load_shed(pm)
end

"""
MLD problem with equality_min (min-max fairness) objective.
Minimizes the maximum load shed across all loads.
Based on switchable_relaxed formulation but with equality_min objective.
"""
function build_mc_mld_equality_min(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    objective_equality_min(pm)
end

"""
MLD problem with equality_min (min-max fairness) objective (INTEGER version).
Minimizes the maximum load shed across all loads with binary switch/block variables.
"""
function build_mc_mld_equality_min_integer(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=false)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=false)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=false)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=false, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=false)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=false)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=false)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use equality_min (min-max fairness) objective
    objective_equality_min(pm)
end

"""
MLD problem with equality_min (min-max fairness) objective.
Minimizes the maximum load shed across all loads.
Based on switchable_relaxed formulation but with equality_min objective.
"""
function build_mc_mld_min_max(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    objective_min_max(pm)
end

"""
MLD problem with equality_min (min-max fairness) objective (INTEGER version).
Minimizes the maximum load shed across all loads with binary switch/block variables.
"""
function build_mc_mld_min_max_integer(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=false)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=false)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=false)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=false, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=false)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=false)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=false)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use equality_min (min-max fairness) objective
    objective_min_max(pm)
end


"""
MLD problem with proportional fairness (Nash bargaining) objective (RELAXED).
Maximizes sum of log(load_served) across all loads.
"""
function build_mc_mld_proportional_fairness(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use proportional fairness (log-sum) objective
    objective_proportional_fairness_mld(pm)
end

"""
MLD problem with proportional fairness (Nash bargaining) objective (INTEGER).
Maximizes sum of log(load_served) with binary switch/block variables.
"""
function build_mc_mld_proportional_fairness_integer(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=false)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=false)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=false)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=false, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=false)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=false)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=false)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use proportional fairness (log-sum) objective
    objective_proportional_fairness_mld(pm)
end

"""
MLD problem with Jain's index promoting objective (RELAXED).
Maximizes total served while penalizing inequality in served fractions.
"""
function build_mc_mld_jain(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use Jain's index promoting objective
    objective_jain_mld(pm)
end

"""
MLD problem with Jain's index promoting objective (INTEGER).
"""
function build_mc_mld_jain_integer(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=false)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=false)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=false)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=false, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=false)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=false)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=false)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use Jain's index promoting objective
    objective_jain_mld(pm)
end

"""
MLD problem with Palma ratio promoting objective (RELAXED).
Minimizes range of served fractions (max - min).
"""
function build_mc_mld_palma(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use Palma ratio promoting objective (minimize range)
    objective_palma_mld(pm)
end

"""
MLD problem with Palma ratio promoting objective (INTEGER).
"""
function build_mc_mld_palma_integer(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=false)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=false)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=false)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=false, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=false)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=false)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=false)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use Palma ratio promoting objective (minimize range)
    objective_palma_mld(pm)
end

"""
MLD problem with Gini coefficient promoting objective (RELAXED).
Minimizes sum of pairwise absolute differences of served fractions.
"""
function build_mc_mld_gini(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=true)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=true)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use Gini coefficient promoting objective (minimize pairwise differences)
    objective_gini_mld(pm)
end

"""
MLD problem with Gini coefficient promoting objective (INTEGER).
"""
function build_mc_mld_gini_integer(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)

   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=true, report=true)

    _PMD.variable_mc_load_indicator(pm; relax=true)
    variable_mc_load_shed(pm)

    variable_block_indicator(pm; relax=false)
    variable_mc_fair_load_weights(pm)

   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)

    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        FairLoadDelivery.constraint_model_voltage_magnitude_difference_fld(pm,i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        constraint_switch_state_on_off(pm,i; relax=false)
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_source_voltage_bounds(pm)
    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)

    constraint_connect_block_load(pm)
    constraint_connect_load_bus(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    # Use Gini coefficient promoting objective (minimize pairwise differences)
    objective_gini_mld(pm)
end

"MLD problem for Branch Flow model "
function build_mc_mld_traditional(pm::_PMD.AbstractUBFModels)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
   # variable_mc_switch_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_generator_power(pm)
    _PMD.variable_mc_bus_voltage(pm)

    # The on-off variable is making the solution error at the report statement in the variable function
    # _PMD.variable_mc_storage_indicator(pm, relax=true)
   	# _PMD.variable_mc_storage_power_mi_on_off(pm, relax=true)
    # Using the snapshot variable definition for now
    _PMD.variable_mc_storage_power_mi(pm, relax=false)

    #variable_mc_load_block_indicator(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=false)
    _PMD.variable_mc_load_indicator(pm; relax=false)
    variable_block_indicator(pm; relax=false)



   	_PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end


    for i in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        _PMD.constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
        #constraint_mc_storage_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        _PMD.constraint_mc_model_voltage_magnitude_difference(pm, i)

        _PMD.constraint_mc_voltage_angle_difference(pm, i)

        _PMD.constraint_mc_thermal_limit_from(pm, i)
        _PMD.constraint_mc_thermal_limit_to(pm, i)
        _PMD.constraint_mc_ampacity_from(pm, i)
        _PMD.constraint_mc_ampacity_to(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        _PMD.constraint_mc_switch_state(pm, i)
        _PMD.constraint_mc_switch_thermal_limit(pm, i)
        _PMD.constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_mc_isolate_block_ref(pm)
    constraint_radial_topology_gr(pm)
    constraint_connect_block_load(pm)
    #_PMD.objective_mc_min_load_setpoint_delta_simple(pm)
    #_PMD.objective_mc_min_fuel_cost(pm)
    objective_mc_min_fuel_cost_pwl_voll(pm)

end


function build_mc_mld_weight_update(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	variable_mc_bus_voltage_magnitude_sqr_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm)

     _PMD.variable_mc_gen_indicator(pm; relax=true)
     _PMD.variable_mc_generator_power_on_off(pm)

    # The on-off variable is making the solution error at the report statement in the variable function
    # _PMD.variable_mc_storage_indicator(pm, relax=true)
   	# _PMD.variable_mc_storage_power_mi_on_off(pm, relax=true)
    # Using the snapshot variable definition for now
     _PMD.variable_mc_storage_power_mi(pm, relax=true)

     _PMD.variable_mc_load_indicator(pm; relax=true)

    variable_block_indicator(pm; relax=true)

   	 _PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    _PMD.constraint_mc_bus_voltage_on_off(pm)
    
     
    for i in _PMD.ids(pm, :gen)
        constraint_mc_gen_power_on_off(pm, i)
    end

    for i in _PMD.ids(pm, :bus)
        _PMD.constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_mi(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        constraint_mc_model_voltage_magnitude_difference_block(pm,i)

         _PMD.constraint_mc_thermal_limit_from(pm, i)
        _PMD.constraint_mc_thermal_limit_to(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        _PMD.constraint_mc_switch_state_on_off(pm, i)
        # The switch thermal limit is not implemented in PowerModelsDistribution yet
        _PMD.constraint_mc_switch_thermal_limit(pm, i)
        # The ampacity constraint is imilar but instead of p^2 + q^2 <= s^2, it is p^2 + q^2 <= w * s^2
        constraint_mc_switch_ampacity(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)

    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
  
    objective_weighted_max_load_served(pm)
end
