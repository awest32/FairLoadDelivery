"Power Flow Problem"
function solve_mc_pf_aw(data::Union{Dict{String,<:Any},String}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_pf_switchable; ref_extensions=[ref_add_load_blocks!], kwargs...)
end


"Constructor for Branch Flow Power Flow"
function build_mc_pf_aw(pm::_PMD.AbstractUBFModels)
    # Variables
    _PMD.variable_mc_bus_voltage(pm; bounded=false)
    _PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_branch_power(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_transformer_power(pm; bounded=false)
    _PMD.variable_mc_generator_power(pm; bounded=false)
    _PMD.variable_mc_load_power(pm)
    _PMD.variable_mc_storage_power(pm; bounded=false)

    # Constraints
    _PMD.constraint_mc_model_current(pm)

    for (i,bus) in _PMD.ref(pm, :ref_buses)
        if !(typeof(pm)<:_PMD.LPUBFDiagPowerModel)
            _PMD.constraint_mc_theta_ref(pm, i)
        end
    
        @assert bus["bus_type"] == 3
        _PMD.constraint_mc_voltage_magnitude_only(pm, i)
    end

    for id in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, id)
    end

    for id in _PMD.ids(pm, :load)
        _PMD.constraint_mc_load_power(pm, id)
    end

    for (i,bus) in _PMD.ref(pm, :bus)
        _PMD.constraint_mc_power_balance(pm, i)
         println("Bus: $(bus["name"]) and i $i and the type of i is $(typeof)")
        if bus["name"] == "646"
            println("Going to fix bus 646 mismatch")
            constraint_fix_bus_terminal_mismatch(pm,i)
        end
        # PV Bus Constraints
        if (length(_PMD.ref(pm, :bus_gens, i)) > 0 || length(_PMD.ref(pm, :bus_storages, i)) > 0) && !(i in _PMD.ids(pm,:ref_buses))
            # this assumes inactive generators are filtered out of bus_gens
            @assert bus["bus_type"] == 2

            _PMD.constraint_mc_voltage_magnitude_only(pm, i)
            
            for j in _PMD.ref(pm, :bus_gens, i)
                _PMD.constraint_mc_gen_power_setpoint_real(pm, j)
            end
            for j in _PMD.ref(pm, :bus_storages, i)
                _PMD.constraint_mc_storage_power_setpoint_real(pm, j)
            end
        end
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_nl(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        _PMD.constraint_mc_model_voltage_magnitude_difference(pm, i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        _PMD.constraint_mc_switch_state(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
        _PMD.constraint_mc_transformer_power(pm, i)
    end
end

"Constructor for Branch Flow Power Flow"
function build_mc_pf_switchable(pm::_PMD.AbstractUBFModels)
    # Variables
    #_PMD.variable_mc_bus_voltage(pm; bounded=false)
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	_PMD.variable_mc_bus_voltage_on_off(pm)
    _PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_branch_power(pm)
    _PMD.variable_mc_switch_power(pm; bounded=true)
    _PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_power(pm; bounded=true)
    _PMD.variable_mc_gen_indicator(pm; relax=true)
    _PMD.variable_mc_generator_power_on_off(pm)
    #_PMD.variable_mc_generator_power(pm; bounded=false)

    variable_mc_demand_indicator(pm; relax=true)
    #_PMD.variable_mc_load_power(pm)
    _PMD.variable_mc_storage_power(pm; bounded=true)
    variable_block_indicator(pm; relax=true)
    _PMD.variable_mc_switch_state(pm; relax=true)


    # Constraints
    _PMD.constraint_mc_model_current(pm)

    for (i,bus) in _PMD.ref(pm, :ref_buses)
        if !(typeof(pm)<:_PMD.LPUBFDiagPowerModel)
            _PMD.constraint_mc_theta_ref(pm, i)
        end
    
        @assert bus["bus_type"] == 3
            _PMD.constraint_mc_bus_voltage_on_off(pm)

        #_PMD.constraint_mc_voltage_magnitude_only(pm, i)
    end

    for id in _PMD.ids(pm, :gen)
        #_PMD.constraint_mc_generator_power_on_off(pm, id)
        _PMD.constraint_mc_gen_power_on_off(pm, id)
    end

    # for id in _PMD.ids(pm, :load)
    #     _PMD.constraint_mc_load_power(pm, id)
    # end

    for (i,bus) in _PMD.ref(pm, :bus)
       #FairDSR.constraint_mc_power_balance_shed(pm, i)
        _PMD.constraint_mc_power_balance_shed(pm,i)
        # PV Bus Constraints
        if (length(_PMD.ref(pm, :bus_gens, i)) > 0 || length(_PMD.ref(pm, :bus_storages, i)) > 0) && !(i in _PMD.ids(pm,:ref_buses))
            # this assumes inactive generators are filtered out of bus_gens
            @assert bus["bus_type"] == 2
                _PMD.constraint_mc_bus_voltage_on_off(pm)

            #_PMD.constraint_mc_voltage_magnitude_only(pm, i)
            for j in _PMD.ref(pm, :bus_gens, i)
                _PMD.constraint_mc_gen_power_setpoint_real(pm, j)
            end
            for j in _PMD.ref(pm, :bus_storages, i)
                _PMD.constraint_mc_storage_power_setpoint_real(pm, j)
            end
        end
         if bus["name"] == "646"
            println("Going to fix bus 646 mismatch")
            constraint_fix_bus_terminal_mismatch(pm,i)
        end
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state_on_off(pm, i)
        _PMD.constraint_storage_complementarity_nl(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_power_losses(pm, i)
        #_PMD.constraint_mc_model_voltage_magnitude_difference(pm, i)
        constraint_mc_model_voltage_magnitude_difference_block(pm,i)
        _PMD.constraint_mc_thermal_limit_from(pm, i)
        _PMD.constraint_mc_voltage_angle_difference(pm, i)
    end
    
    #constraint_open_switches(pm)

    for i in _PMD.ids(pm, :switch)
        FairLoadDelivery.constraint_mc_switch_state_on_off_top(pm, i)
        #_PMD.constraint_mc_switch_state(pm, i)
        _PMD.constraint_mc_switch_thermal_limit(pm, i)
        #_PMD.constraint_mc_switch_ampacity(pm, i)
    end

    constraint_mc_isolate_block(pm)
   # constraint_connect_block_load(pm)
     constraint_connect_block_voltage(pm)
     constraint_connect_block_gen(pm)
     constraint_connect_block_shunt(pm)
     constraint_connect_block_storage(pm)
    #constraint_mc_block_energization_consistency(pm)

    # for i in _PMD.ids(pm, :switch)
    #     _PMD.constraint_mc_switch_state(pm, i)
    # end

    for i in _PMD.ids(pm, :transformer)
        _PMD.constraint_mc_transformer_power(pm, i)
    end
    

    constraint_gen_event_simple(pm, ls_percent=0.9)

end