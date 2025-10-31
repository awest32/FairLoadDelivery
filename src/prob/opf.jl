"""
	function solve_mc_opf(
		data::Union{Dict{String,<:Any},String},
		model_type::Type,
		solver;
		kwargs...
	)

Solve Optimal Power Flow
"""
function solve_mc_opf_ld3f(data::Union{Dict{String,<:Any},String}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.LinDist3FlowPowerModel, solver, build_mc_opf_ldf; kwargs...)
end

function solve_mc_opf_acp(data::Union{Dict{String,<:Any},String}, solver; kwargs...)
    return _PMD.solve_mc_model(data, _PMD.ACPUPowerModel, solver, build_mc_opf_ac; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!],kwargs...)
end

"""
	function build_mc_opf(
		pm::AbstractUnbalancedPowerModel
	)

Constructor for Optimal Power Flow
"""
function build_mc_opf_ac(pm::_PMD.AbstractUnbalancedPowerModel)
    _PMD.variable_mc_bus_voltage(pm)
    _PMD.variable_mc_branch_power(pm)
    _PMD.variable_mc_transformer_power(pm)
    _PMD.variable_mc_switch_power(pm)
    #_PMD.variable_mc_switch_state(pm; relax=true)
    _PMD.variable_mc_generator_power(pm)
    _PMD.variable_mc_load_power(pm)
    _PMD.variable_mc_load_indicator(pm; relax=true)
    _PMD.variable_mc_storage_power(pm)
    variable_block_indicator(pm; relax=true)

    _PMD.constraint_mc_model_voltage(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    # generators should be _PMD.constrained before KCL, or Pd/Qd undefined
    for id in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, id)
    end

    #loads should be _PMD.constrained before KCL, or Pd/Qd undefined
    for id in _PMD.ids(pm, :load)
        _PMD.constraint_mc_load_power(pm, id)
    end

    for i in _PMD.ids(pm, :bus)
        _PMD.constraint_mc_power_balance_shed(pm, i)
    end

    for i in _PMD.ids(pm, :storage)
        _PMD.constraint_storage_state(pm, i)
        _PMD.constraint_storage_complementarity_nl(pm, i)
        _PMD.constraint_mc_storage_losses(pm, i)
        _PMD.constraint_mc_storage_thermal_limit(pm, i)
    end

    for i in _PMD.ids(pm, :branch)
        _PMD.constraint_mc_ohms_yt_from(pm, i)
        _PMD.constraint_mc_ohms_yt_to(pm, i)

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
     constraint_connect_block_load(pm)
     constraint_mc_isolate_block_ref(pm)
    #_PMD.objective_mc_min_fuel_cost(pm)
    objective_weighted_max_load_served(pm)

end




"""
	function build_mc_opf(
		pm::AbstractUBFModels
	)

constructor for branch flow opf
"""
function build_mc_opf_ldf(pm::_PMD.AbstractUBFModels)
    # Variables
    _PMD.variable_mc_bus_voltage(pm)
    _PMD.variable_mc_branch_current(pm)
    _PMD.variable_mc_branch_power(pm)
    _PMD.variable_mc_switch_power(pm)
    _PMD.variable_mc_transformer_power(pm)
    _PMD.variable_mc_generator_power(pm)
    _PMD.variable_mc_load_power(pm)
    _PMD.variable_mc_storage_power(pm)

    # Constraints
    _PMD.constraint_mc_model_current(pm)

    for i in _PMD.ids(pm, :ref_buses)
        _PMD.constraint_mc_theta_ref(pm, i)
    end

    # gens should be constrained before KCL, or Pd/Qd undefined
    for id in _PMD.ids(pm, :gen)
        _PMD.constraint_mc_generator_power(pm, id)
    end

    # loads should be constrained before KCL, or Pd/Qd undefined
    for id in _PMD.ids(pm, :load)
        _PMD.constraint_mc_load_power(pm, id)
    end

    for i in _PMD.ids(pm, :bus)
        _PMD.constraint_mc_power_balance(pm, i)
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

    # Objective
    _PMD.objective_mc_min_fuel_cost(pm)
end

