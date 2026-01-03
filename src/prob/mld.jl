
""
function solve_mc_mld_switch_relaxed(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_switchable_relaxed; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

function solve_mc_mld_switch_integer(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_switchable_integer; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

function solve_mc_mld_traditional(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_traditional; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

function solve_mc_mld_shed_random_round(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_shedding_random_rounding; ref_extensions=[ref_add_load_blocks!], kwargs...)
end

function solve_mc_mld_shed_implicit_diff(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_shedding_implicit_diff; ref_extensions=[ref_add_rounded_load_blocks!], kwargs...)
end


function solve_mc_mld_weight_update(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_mld_weight_update; ref_extensions=[ref_update_weights!], kwargs...)
end

function build_mc_mld_shedding_implicit_diff(pm::_PMD.AbstractUBFModels)
    pm.model = JuMP.Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    # JuMP.set_attribute(pm.model, "hsllib", HSL_jll.libhsl_path)
    # JuMP.set_attribute(pm.model, "linear_solver", "ma27")
    @info pm.model typeof(pm.model)
    variable_mc_load_shed(pm)
    
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	_PMD.variable_mc_bus_voltage_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
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
        _PMD.constraint_mc_storage_on_off(pm, i)
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

    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)
    #constraint_mc_radiality(pm)
    #constraint_mc_block_energization_consistency_bigm(pm)

    # Must be disabled if there is no generation in the network
    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    constraint_load_shed_definition(pm)
   
    constraint_connect_block_load(pm)
    constraint_connect_block_gen(pm)
    constraint_connect_block_voltage(pm)
    constraint_connect_block_shunt(pm)
    constraint_connect_block_storage(pm)

    #constraint_set_block_state_rounded(pm)
    #constraint_set_switch_state_rounded(pm)
    
    #_PMD.objective_mc_min_load_setpoint_delta_simple(pm)
    #_PMD.objective_mc_min_fuel_cost(pm)
    #objective_mc_min_fuel_cost_pwl_voll(pm)
    objective_fairly_weighted_max_load_served(pm)
    #objective_fair_max_load_served(pm,"jain")
    #objective_fairly_weighted_max_load_served_with_penalty(pm)
    #objective_fairly_weighted_min_load_shed(pm)
end

"Multinetwork load shedding problem for Branch Flow model "
function build_mc_mld_shedding_random_rounding(pm::_PMD.AbstractUBFModels)
 _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	_PMD.variable_mc_bus_voltage_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
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
    constraint_load_shed_definition(pm)


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
        _PMD.constraint_mc_storage_on_off(pm, i)
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

    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)
    #constraint_mc_radiality(pm)
    constraint_mc_block_energization_consistency_bigm(pm)

    # Must be disabled if there is no generation in the network
    constraint_block_budget(pm)
    constraint_switch_budget(pm)

   
   
    constraint_connect_block_load(pm)
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

"Load shedding problem for Branch Flow model with load blocks and radiality constraints and variables "
function build_mc_mld_switchable_integer(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=false)
 	_PMD.variable_mc_bus_voltage_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_switch_state(pm; relax=false)
    _PMD.variable_mc_shunt_indicator(pm; relax=false)
    _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_gen_indicator(pm; relax=false)
    _PMD.variable_mc_generator_power_on_off(pm)

    # # The on-off variable is making the solution error at the report statement in the variable function
   	_PMD.variable_mc_storage_power_mi_on_off(pm, relax=false, report=true)
 

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
        _PMD.constraint_mc_storage_on_off(pm, i)
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

    constraint_mc_isolate_block(pm)
    constraint_radial_topology(pm)
    # #constraint_mc_radiality(pm)
    #constraint_mc_block_energization_consistency_bigm(pm)

    # Must be disabled if there is no generation in the network
    constraint_block_budget(pm)
    constraint_switch_budget(pm)

    # constraint_set_block_state_rounded(pm)
    # constraint_set_switch_state_rounded(pm)
   
   
    constraint_connect_block_load(pm)
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

    JuMP._CONSTRAINT_LIMIT_FOR_PRINTING[] = 1E9
    print(pm.model)

end

"Load shedding problem for Branch Flow model with load blocks and radiality constraints and variables "
function build_mc_mld_switchable_relaxed(pm::_PMD.AbstractUBFModels)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	_PMD.variable_mc_bus_voltage_on_off(pm)

    _PMD.variable_mc_branch_power(pm)
	_PMD.variable_mc_branch_current(pm)
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
    constraint_load_shed_definition(pm)

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
        _PMD.constraint_mc_storage_on_off(pm, i)
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
        # The ampacity constraint is similar but instead of p^2 + q^2 <= w * s^2, it is p^2 + q^2 <= z * w * s^2
        constraint_mc_switch_ampacity(pm, i)
        constraint_model_switch_voltage_magnitude_difference_fld(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
       _PMD.constraint_mc_transformer_power(pm, i)
    end

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
    JuMP._CONSTRAINT_LIMIT_FOR_PRINTING[] = 1E9
    open("relaxed_model_out.txt", "w") do io
            redirect_stdout(io) do
                print(pm.model)
            end
        end
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
        #_PMD.constraint_mc_storage_on_off(pm, i)
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
 	_PMD.variable_mc_bus_voltage_on_off(pm)

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
