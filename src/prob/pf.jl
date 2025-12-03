"Power Flow Problem"
function solve_mc_pf_aw(data::Dict{String,<:Any}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.IVRUPowerModel, solver, build_mc_pf_switch;ref_extensions=[ref_add_load_blocks!], kwargs...)
end


"Constructor for Power Flow in current-voltage variable space"
function build_mc_pf_switch(pm::_PMD.AbstractUnbalancedIVRModel)
    # Variables
    _PMD.variable_mc_bus_voltage_indicator(pm; relax=true)
 	_PMD.variable_mc_bus_voltage_on_off(pm)

   # _PMD.variable_mc_bus_voltage(pm, bounded = false)
    _PMD.variable_mc_branch_current(pm, bounded = false)
    _PMD.variable_mc_switch_current(pm, bounded=false)
    _PMD.variable_mc_switch_state(pm; relax=true)
    #_PMD.variable_mc_shunt_indicator(pm; relax=true)
    _PMD.variable_mc_transformer_current(pm, bounded = false)
    _PMD.variable_mc_generator_current(pm, bounded = false)
    #_PMD.variable_mc_gen_indicator(pm; relax=true)
    #_PMD.variable_mc_generator_power_on_off(pm)
    _PMD.variable_mc_load_current(pm, bounded = false)
    _PMD.variable_mc_load_indicator(pm; relax=true)


    FairLoadDelivery.variable_block_indicator(pm; relax=true)
    FairLoadDelivery.variable_mc_fair_load_weights(pm)

    # Constraints
    for (i,bus) in _PMD.ref(pm, :ref_buses)
        @assert bus["bus_type"] == 3
        _PMD.constraint_mc_theta_ref(pm, i)
        _PMD.constraint_mc_voltage_magnitude_only(pm, i)
    end

    # gens should be constrained before KCL, or Pd/Qd undefined
    for id in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, id)
    end

    # loads should be constrained before KCL, or Pd/Qd undefined
    for id in _PMD.ids(pm, :load)
        _PMD.constraint_mc_load_power(pm, id)
    end

    for (i,bus) in _PMD.ref(pm, :bus)
        _PMD.constraint_mc_current_balance(pm, i)

        # PV Bus Constraints
        if length(_PMD.ref(pm, :bus_gens, i)) > 0 && !(i in _PMD.ids(pm,:ref_buses))
            # this assumes inactive generators are filtered out of bus_gens
            @assert bus["bus_type"] == 2
            _PMD.constraint_mc_voltage_magnitude_only(pm, i)
            for j in _PMD.ref(pm, :bus_gens, i)
                _PMD.constraint_mc_gen_power_setpoint_real(pm, j)
            end
        end
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_current_from(pm, i)
        _PMD.constraint_mc_current_to(pm, i)

        _PMD.constraint_mc_bus_voltage_drop(pm, i)
    end

    for i in _PMD.ids(pm, :switch)
        _PMD.constraint_mc_switch_state_on_off(pm, i; relax=true)
        _PMD.constraint_mc_switch_current_limit(pm, i)
        #_PMD.constraint_mc_switch_state(pm, i)
    end

    for i in _PMD.ids(pm, :transformer)
        _PMD.constraint_mc_transformer_power(pm, i)
    end

    FairLoadDelivery.constraint_mc_isolate_block(pm)
   # constraint_connect_block_load(pm)
     FairLoadDelivery.constraint_connect_block_voltage(pm)
#     FairLoadDelivery.constraint_connect_block_gen(pm)
   #  FairLoadDelivery.constraint_connect_block_shunt(pm)
   #  FairLoadDelivery.constraint_connect_block_storage(pm)
    #constraint_mc_block_energization_consistency(pm)

    # for i in _PMD.ids(pm, :switch)
    #     _PMD.constraint_mc_switch_state(pm, i)
    # end

    for i in _PMD.ids(pm, :transformer)
        _PMD.constraint_mc_transformer_power(pm, i)
    end
    


end