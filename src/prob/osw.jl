"Solve optimal switching problem"
function solve_mc_osw_aw(data::Union{Dict{String,<:Any}, String}, model_type::Type, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_osw_aw; kwargs...)
end


"Solve mixed-integer optimal switching problem"
function solve_mc_osw_mi(data::Union{Dict{String,<:Any}, String}, model_type::Type, solver; kwargs...)
    return _PMD.solve_mc_model(data, LPUBFDiagPowerModel, solver, _build_mc_osw_mi; kwargs...)
end


"Constructor for Optimal Switching"
function build_mc_osw_aw(pm::_PMD.AbstractUnbalancedPowerModel)
    _PMD.variable_mc_bus_voltage(pm)
    # _PMD.variable_mc_branch_power(pm)
    # _PMD.variable_mc_transformer_power(pm)

    _PMD.variable_mc_switch_power(pm)
    # _PMD.variable_mc_switch_state(pm; relax=false)

    variable_mc_generator_power(pm)
    #  _PMD.variable_mc_load_power(pm)
    # _PMD.variable_mc_storage_power_mi(pm)

    # _PMD.constraint_mc_model_voltage(pm)

    
    # for i in _PMD.ids(pm, :ref_buses)
    #    _PMD.constraint_mc_theta_ref(pm, i)
    # end

    # # generators should be constrained before KCL, or Pd/Qd undefined
    # for id in _PMD.ids(pm, :gen)
    #     _PMD.constraint_mc_generator_power(pm, id)
    # end

    # # loads should be constrained before KCL, or Pd/Qd undefined
    # for id in _PMD.ids(pm, :load)
    #     _PMD.constraint_mc_load_power(pm, id)
    # end
    # for i in _PMD.ids(pm, :bus)
    #     _PMD.constraint_mc_power_balance(pm, i)
    # end

    # for i in _PMD.ids(pm, :storage)
    #     _PMD.constraint_storage_state(pm, i)
    #     _PMD.constraint_storage_complementarity_mi(pm, i)
    #     _PMD.constraint_mc_storage_losses(pm, i)
    #     _PMD.constraint_mc_storage_thermal_limit(pm, i)
    # end

    # for i in _PMD.ids(pm, :branch)
    #     # _PMD.constraint_mc_ohms_yt_from(pm, i)
    #     # _PMD.constraint_mc_ohms_yt_to(pm, i)

    #     _PMD.constraint_mc_voltage_angle_difference(pm, i)

    #     # _PMD.constraint_mc_thermal_limit_from(pm, i)
    #     # _PMD.constraint_mc_thermal_limit_to(pm, i)
    #     _PMD.constraint_mc_ampacity_from(pm, i)
    #     _PMD.constraint_mc_ampacity_to(pm, i)
    # end

    # for i in _PMD.ids(pm, :switch)
    #     _PMD.constraint_mc_switch_state_on_off(pm, i; relax=true)
    #     _PMD.constraint_mc_switch_thermal_limit(pm, i)
    #     _PMD.constraint_mc_switch_ampacity(pm, i)
    # end

    # for i in _PMD.ids(pm, :transformer)
    #     _PMD.constraint_mc_transformer_power(pm, i)
    # end
   # _PMD.objective_mc_min_fuel_cost(pm)
   # _PMD.objective_mc_min_fuel_cost_switch(pm)
end

