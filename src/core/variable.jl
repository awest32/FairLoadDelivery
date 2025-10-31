
"""
    variable_block_indicator(
        pm::AbstractUnbalancedPowerModel;
        nw::Int=nw_id_default,
        relax::Bool=false,
        report::Bool=true
    )

Create variables for block status by load block, z^{bl}_i in{0,1} forall i in B, binary if `relax=false`.
Variables will appear in solution if `report=true`.
"""
function variable_block_indicator(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, relax::Bool=false, report::Bool=true)
    if relax
        z_block = _PMD.var(pm, nw)[:z_block] = JuMP.@variable(pm.model,
            [i in _PMD.ids(pm, nw, :blocks)], base_name="$(nw)_z_block",
            lower_bound = 0,
            upper_bound = 1,
            start = 0.0
        )
    else
        z_block = _PMD.var(pm, nw)[:z_block] = JuMP.@variable(pm.model,
            [i in _PMD.ids(pm, nw, :blocks)], base_name="$(nw)_z_block",
            binary = true,
            start = 0
        )
    end
    
    bus_block_map = _PMD.ref(pm, nw, :bus_block_map)
    bus_to_block = Dict(i => bus_block_map[i] for i in _PMD.ids(pm, nw, :bus))
    gen_block_map = _PMD.ref(pm, nw, :gen_block_map)
    gen_to_block = Dict(i => gen_block_map[i] for i in _PMD.ids(pm, nw, :gen))
    load_block_map = _PMD.ref(pm, nw, :load_block_map)
    load_to_block = Dict(i => load_block_map[i] for i in _PMD.ids(pm, nw, :load))

    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :bus, :block,  _PMD.ids(pm, nw, :bus), bus_to_block)
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :block, :status, _PMD.ids(pm, nw, :blocks), z_block)
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :gen,  :block,  _PMD.ids(pm, nw, :gen), gen_to_block)
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load,  :block,  _PMD.ids(pm, nw, :load), load_to_block)
end


"create an indicator variable for the switch status"
function variable_mc_switch_indicator(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, relax::Bool=false, report::Bool=true)
    if relax
        z_switch = _PMD.var(pm, nw)[:z_switch] = JuMP.@variable(pm.model,
            [i in _PMD.ids(pm, nw, :switch)], base_name="$(nw)_z_switch",
            lower_bound = 0,
            upper_bound = 1,
            start = 1.0
        )
    else
       z_switch = _PMD.var(pm, nw)[:z_switch] = JuMP.@variable(pm.model,
            [i in _PMD.ids(pm, nw, :switch)], base_name="$(nw)_z_switch",
            binary = true,
            start = 1
        )
    end

end

"create variables for demand status by load block"
function variable_mc_demand_indicator(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, relax::Bool=false, report::Bool=true)
     # force tehe demand indicator to match the load block
     #in JuMP.Parameter(fbw_val[i])
    if relax
        z_demand = _PMD.var(pm, nw)[:z_demand] = JuMP.@variable(pm.model,
            z_demand[i in _PMD.ids(pm, nw, :load)] , base_name="$(nw)_z_demand",
            lower_bound = 0,
            upper_bound = 1,
            start = 0.0
        )
    else
        z_demand = _PMD.var(pm, nw)[:z_demand] = JuMP.@variable(pm.model,
            z_demand[i in _PMD.ids(pm, nw, :load)] , base_name="$(nw)_z_demand",
            binary = true,
            start = 1.0
        )
    end

    #block_load_map = _PMD.ref(pm, nw, :block_load_map)
    load_block_map = _PMD.ref(pm, nw, :load_block_map)
    #println(block_load_map)
    _PMD.var(pm, nw)[:z_demand] = Dict(l => z_demand[load_block_map[l]] for l in _PMD.ids(pm, nw, :load))

    # expressions for pd and qd
    pd = _PMD.var(pm, nw)[:pd] = Dict(i => _PMD.var(pm, nw)[:z_demand][i].*_PMD.ref(pm, nw, :load, i)["pd"] for i in _PMD.ids(pm, nw, :load))
    qd = _PMD.var(pm, nw)[:qd] = Dict(i => _PMD.var(pm, nw)[:z_demand][i].*_PMD.ref(pm, nw, :load, i)["qd"] for i in _PMD.ids(pm, nw, :load))
    pd0 = _PMD.var(pm, nw)[:pd] = Dict(i => _PMD.ref(pm, nw, :load, i)["pd"] for i in _PMD.ids(pm, nw, :load))
    qd0 = _PMD.var(pm, nw)[:qd] = Dict(i => _PMD.ref(pm, nw, :load, i)["qd"] for i in _PMD.ids(pm, nw, :load))
  
    load_to_lb = Dict(i => load_block_map[i] for i in _PMD.ids(pm, nw, :load))
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load, :load_block,  _PMD.ids(pm, nw, :load), load_to_lb)

    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load, :status, _PMD.ids(pm, nw, :load), _PMD.var(pm, nw)[:z_demand])
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load, :pd, _PMD.ids(pm, nw, :load), pd)
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load, :qd, _PMD.ids(pm, nw, :load), qd)
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load, :pd0, _PMD.ids(pm, nw, :load), pd0)
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load, :qd0, _PMD.ids(pm, nw, :load), qd0)
end

## voltage on/off variables

"on/off voltage magnitude variable"
function variable_mc_bus_voltage_magnitude_on_off(pm::_PMD.AbstractUBFModels; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    terminals = Dict(i => bus["terminals"] for (i,bus) in ref(pm, nw, :bus))
    vm = var(pm, nw)[:vm] = Dict(i => JuMP.@variable(pm.model,
        [t in terminals[i]], base_name="vm_$(i)",
        start = comp_start_value(ref(pm, nw, :bus, i), ["vm_start", "vm", "vmin"], t, 1.0)
    ) for i in ids(pm, nw, :bus))

    if bounded
        for (i, bus) in ref(pm, nw, :bus)
            for (idx,t) in enumerate(terminals[i])
                set_lower_bound_v(vm[i][t], 0.0)

                if haskey(bus, "vmax")
                    set_upper_bound_v(vm[i][t], bus["vmax"][idx])
                end
            end
        end
    end

    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :bus, :vm, ids(pm, nw, :bus), vm)

end


"variable: `w[i] >= 0` for `i` in `buses"
function variable_mc_bus_voltage_magnitude_sqr(pm::_PMD.AbstractUBFModels; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    terminals = Dict(i => bus["terminals"] for (i,bus) in ref(pm, nw, :bus))
    w = var(pm, nw)[:w] = Dict(i => JuMP.@variable(pm.model,
            [t in terminals[i]], base_name="w_$(i)",
            lower_bound = 0.0,
            start = comp_start_value(ref(pm, nw, :bus, i), "w_start", t, comp_start_value(ref(pm, nw, :bus, i), ["vm_start", "vm"], t, 1.0)^2)
        ) for i in ids(pm, nw, :bus)
    )

    if bounded
        for i in ids(pm, nw, :bus)
            bus = ref(pm, nw, :bus, i)
            for (idx, t) in enumerate(terminals[i])
                set_upper_bound_v(w[i][t], max(bus["vmin"][idx]^2, bus["vmax"][idx]^2))
                if bus["vmin"][idx] > 0
                    set_lower_bound_v(w[i][t], bus["vmin"][idx]^2)
                end
            end
        end
    end

    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :bus, :w, ids(pm, nw, :bus), w)
end

function variable_mc_fair_load_weights(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    fbw_val = _PMD.ref(pm, nw, :load_weights)  
    # Create parameter variables for each block weight
    fair_load_weights = _PMD.var(pm, nw)[:fair_load_weights] = JuMP.@variable(
        pm.model, 
        fair_load_weights[j in keys(fbw_val)] in JuMP.Parameter(fbw_val[j]),
        base_name = "fair_load_weights"
    )

#    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :block, :fair_load_weights, _PMD.ids(pm, nw, :blocks), fair_load_weights)
end

function variable_mc_load_shed(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    loads = _PMD.ids(pm, nw, :load)  
    # Create parameter variables for each block weight
    pshed = _PMD.var(pm, nw)[:pshed] = JuMP.@variable(
        pm.model, 
        pshed[j in loads],
        base_name = "pshed"
    )
    qshed = _PMD.var(pm, nw)[:qshed] = JuMP.@variable(
        pm.model, 
        qshed[j in loads],
        base_name = "qshed"
    )
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load, :pshed, _PMD.ids(pm, nw, :load), pshed)
    report && _IM.sol_component_value(pm, pmd_it_sym, nw, :load, :qshed, _PMD.ids(pm, nw, :load), qshed)
end   
