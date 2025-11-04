
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
    @info z_demand
   
    _PMD.var(pm, nw)[:pd] = Dict(i => _PMD.var(pm, nw)[:z_demand][i].*_PMD.ref(pm, nw, :load, i)["pd"] for i in _PMD.ids(pm, nw, :load))
    pd = Dict(i => _PMD.var(pm, nw)[:z_demand][i].*_PMD.ref(pm, nw, :load, i)["pd"] for i in _PMD.ids(pm, nw, :load))
    qd = _PMD.var(pm, nw)[:qd] = Dict(i => _PMD.var(pm, nw)[:z_demand][i].*_PMD.ref(pm, nw, :load, i)["qd"] for i in _PMD.ids(pm, nw, :load))
    pd0 = _PMD.var(pm, nw)[:pd] = Dict(i => _PMD.ref(pm, nw, :load, i)["pd"] for i in _PMD.ids(pm, nw, :load))
    qd0 = _PMD.var(pm, nw)[:qd] = Dict(i => _PMD.ref(pm, nw, :load, i)["qd"] for i in _PMD.ids(pm, nw, :load))
    @info pd
    @info _PMD.var(pm, nw, :pd)
    @info _PMD.var(pm, nw)[:pd]

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
        start = _PMD.comp_start_value(ref(pm, nw, :bus, i), ["vm_start", "vm", "vmin"], t, 1.0)
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
            start = _PMD.comp_start_value(ref(pm, nw, :bus, i), "w_start", t, _PMD.comp_start_value(ref(pm, nw, :bus, i), ["vm_start", "vm"], t, 1.0)^2)
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

# function variable_mc_storage_power_mi_on_off(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, relax::Bool=false, bounded::Bool=true, report::Bool=true)
#     variable_mc_storage_power_real_on_off(pm; nw=nw, bounded=bounded, report=report)
#     variable_mc_storage_power_imaginary_on_off(pm; nw=nw, bounded=bounded, report=report)
#     variable_mc_storage_power_control_imaginary_on_off(pm; nw=nw, bounded=bounded, report=report)
#     variable_mc_storage_indicator(pm; nw=nw, report=report)
#     variable_storage_energy(pm; nw=nw, bounded=bounded, report=report)
#     variable_storage_charge(pm; nw=nw, bounded=bounded, report=report)
#     variable_storage_discharge(pm; nw=nw, bounded=bounded, report=report)
#     variable_storage_complementary_indicator(pm; nw=nw, relax=relax, report=report)
# end

# "Create variables for `active` and `reactive` storage injection"
# function variable_mc_storage_power_on_off(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
#     variable_mc_storage_power_real_on_off(pm; nw=nw, bounded=bounded, report=report)
#     variable_mc_storage_power_imaginary_on_off(pm; nw=nw, bounded=bounded, report=report)
# end


# "Create variables for `active` storage injection"
# function variable_mc_storage_power_real_on_off(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
#     connections = Dict(i => strg["connections"] for (i,strg) in _PMD.ref(pm, nw, :storage))
#     ps = _PMD.var(pm, nw)[:ps] = Dict(i => JuMP.@variable(pm.model,
#         [c in connections[i]], base_name="$(nw)_ps_$(i)",
#         start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), "ps_start", c, 0.0)
#     ) for i in _PMD.ids(pm, nw, :storage))

#     if bounded
#         inj_lb, inj_ub = _PMD.ref_calc_storage_injection_bounds(_PMD.ref(pm, nw, :storage), _PMD.ref(pm, nw, :bus))
#         for (i, strg) in _PMD.ref(pm, nw, :storage)
#             for (idx, c) in enumerate(connections[i])
#                 FairLoadDelivery.set_lower_bound(ps[i][c], min(inj_lb[i][idx], 0.0))
#                 FairLoadDelivery.set_upper_bound(ps[i][c], max(inj_ub[i][idx], 0.0))
#             end
#         end
#     end

#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :ps, _PMD.ids(pm, nw, :storage), ps)
# end

# """
# a reactive power slack variable that enables the storage device to inject or
# consume reactive power at its connecting bus, subject to the injection limits
# of the device.
# """
# function variable_mc_storage_power_control_imaginary_on_off(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
#     qsc = _PMD.var(pm, nw)[:qsc] = JuMP.@variable(pm.model,
#         [i in ids(pm, nw, :storage)], base_name="$(nw)_qsc_$(i)",
#         start = comp_start_value(ref(pm, nw, :storage, i), "qsc_start")
#     )

#     if bounded
#         inj_lb, inj_ub = _PMD.ref_calc_storage_injection_bounds(_PMD.ref(pm, nw, :storage), ref(pm, nw, :bus))
#         for (i,storage) in ref(pm, nw, :storage)
#             if !isinf(sum(inj_lb[i])) || haskey(storage, "qmin")
#                 lb = max(sum(inj_lb[i]), sum(get(storage, "qmin", -Inf)))
#                 _PMD.set_lower_bound(qsc[i], min(lb, 0.0))
#             end
#             if !isinf(sum(inj_ub[i])) || haskey(storage, "qmax")
#                 ub = min(sum(inj_ub[i]), sum(get(storage, "qmax", Inf)))
#                 _PMD.set_upper_bound(qsc[i], max(ub, 0.0))
#             end
#         end
#     end

#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :qsc, ids(pm, nw, :storage), qsc)
# end



# "Create variables for `reactive` storage injection"
# function variable_mc_storage_power_imaginary_on_off(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
#     connections = Dict(i => strg["connections"] for (i,strg) in _PMD.ref(pm, nw, :storage))
#     qs = _PMD.var(pm, nw)[:qs] = Dict(i => JuMP.@variable(pm.model,
#         [c in connections[i]], base_name="$(nw)_qs_$(i)",
#         start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), "qs_start", c, 0.0)
#     ) for i in _PMD.ids(pm, nw, :storage))

#     if bounded
#         for (i, strg) in _PMD.ref(pm, nw, :storage)
#             if haskey(strg, "qmin")
#                 for (idx, c) in enumerate(connections[i])
#                     FairLoadDelivery.set_lower_bound(qs[i][c], min(strg["qmin"], 0.0))
#                 end
#             end

#             if haskey(strg, "qmax")
#                 for (idx, c) in enumerate(connections[i])
#                     FairLoadDelivery.set_upper_bound(qs[i][c], max(strg["qmax"], 0.0))
#                 end
#             end
#         end
#     end

#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :qs, _PMD.ids(pm, nw, :storage), qs)
# end


# "Create variables for storage status"
# function variable_mc_storage_indicator(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, relax::Bool=false, report::Bool=true)
#     if !relax
#         z_storage = _PMD.var(pm, nw)[:z_storage] = JuMP.@variable(pm.model,
#             [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)-z_storage",
#             binary = true,
#             start = _PMD.comp_start_value(ref(pm, nw, :storage, i), "z_storage_start", 1.0)
#         )
#     else
#         z_storage = _PMD.var(pm, nw)[:z_storage] = JuMP.@variable(pm.model,
#             [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)_z_storage",
#             lower_bound = 0,
#             upper_bound = 1,
#             start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), "z_storage_start", 1.0)
#         )
#     end

#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :status, ids(pm, nw, :storage), z_storage)
# end

# ""
# function variable_storage_energy(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
#     se = _PMD.var(pm, nw)[:se] = JuMP.@variable(pm.model,
#         [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)_se",
#         lower_bound = 0.0,
#         start = _PMD.comp_start_value(ref(pm, nw, :storage, i), ["se_start", "se", "energy"], 0.0)
#     )

#     if bounded
#         for (i, storage) in _PMD.ref(pm, nw, :storage)
#             FairLoadDelivery.set_upper_bound(se[i], storage["energy_rating"])
#         end
#     end

#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :se, _PMD.ids(pm, nw, :storage), se)
# end


# ""
# function variable_storage_charge(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
#     sc = _PMD.var(pm, nw)[:sc] = JuMP.@variable(pm.model,
#         [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)_sc",
#         lower_bound = 0.0,
#         start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), ["sc_start", "sc"], 1)
#     )

#     if bounded
#         for (i, storage) in _PMD.ref(pm, nw, :storage)
#             FairLoadDelivery.set_upper_bound(sc[i], storage["charge_rating"])
#         end
#     end

#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :sc, _PMD.ids(pm, nw, :storage), sc)
# end


# ""
# function variable_storage_discharge(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
#     sd = var(pm, nw)[:sd] = JuMP.@variable(pm.model,
#         [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)_sd",
#         lower_bound = 0.0,
#         start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), ["sd_start", "sd"], 0.0)
#     )

#     if bounded
#         for (i, storage) in _PMD.ref(pm, nw, :storage)
#             FairLoadDelivery.set_upper_bound(sd[i], storage["discharge_rating"])
#         end
#     end

#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :sd, _PMD.ids(pm, nw, :storage), sd)
# end


# ""
# function variable_storage_complementary_indicator(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, relax::Bool=false, report::Bool=true)
#     if !relax
#         sc_on = _PMD.var(pm, nw)[:sc_on] = JuMP.@variable(pm.model,
#             [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)_sc_on",
#             binary = true,
#             start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), "sc_on_start", 0)
#         )
#         sd_on = _PMD.var(pm, nw)[:sd_on] = JuMP.@variable(pm.model,
#             [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)_sd_on",
#             binary = true,
#             start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), "sd_on_start", 0)
#         )
#     else
#         sc_on = _PMD.var(pm, nw)[:sc_on] = JuMP.@variable(pm.model,
#             [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)_sc_on",
#             lower_bound = 0,
#             upper_bound = 1,
#             start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), "sc_on_start", 0)
#         )
#         sd_on = _PMD.var(pm, nw)[:sd_on] = JuMP.@variable(pm.model,
#             [i in _PMD.ids(pm, nw, :storage)], base_name="$(nw)_sd_on",
#             lower_bound = 0,
#             upper_bound = 1,
#             start = _PMD.comp_start_value(_PMD.ref(pm, nw, :storage, i), "sd_on_start", 0)
#         )
#     end

#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :sc_on, _PMD.ids(pm, nw, :storage), sc_on)
#     report && _IM.sol_component_value(pm, pmd_it_sym, nw, :storage, :sd_on, _PMD.ids(pm, nw, :storage), sd_on)
# end
