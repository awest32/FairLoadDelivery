
function constraint_gen_event_simple(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, ls_percent::Float64)
    total_load = sum(load["pd"][idx] for (i, load) in _PMD.ref(pm, nw, :load) for (idx, c) in enumerate(load["connections"]))
    for (i, gen) in _PMD.ref(pm, nw, :gen)
        if gen["source_id"] == "voltage_source.source"
            pg = sum(_PMD.var(pm, nw, :pg, i))
            JuMP.@constraint(pm.model, pg <= ls_percent * sum(total_load))
            JuMP.@constraint(pm.model, pg>=0)
        end
    end
end

function constraint_fix_bus_terminal_mismatch(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=nw_id_default)
    println("Fixing bus 646 mismatch")
    JuMP.@constraint(pm.model, _PMD.var(pm, nw, :w)[i][3] ==_PMD.var(pm, nw, :w)[i][1])
end

function constraint_load_shed_def(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    JuMP.@constraint(pm.model, pshed_constraint[d in keys(_PMD.ref(pm, nw, :load))], 
        _PMD.var(pm, nw)[:pshed][d] == (1-_PMD.var(pm, nw)[:z_demand][d])*sum(_PMD.ref(pm, nw, :load, d)["pd"])
    )
    
    JuMP.@constraint(pm.model, qshed_constraint[d in keys(_PMD.ref(pm, nw, :load))], 
        _PMD.var(pm, nw)[:qshed][d] == (1-_PMD.var(pm, nw)[:z_demand][d])*sum(_PMD.ref(pm, nw, :load, d)["qd"])
    )
end

function constraint_mc_isolate_block(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    for (s, switch) in _PMD.ref(pm, nw, :switch)
            z_block_fr = _PMD.var(pm, nw, :z_block)[_PMD.ref(pm, nw, :bus_block_map)[switch["f_bus"]]]
            z_block_to = _PMD.var(pm, nw, :z_block)[_PMD.ref(pm, nw, :bus_block_map)[switch["t_bus"]]]

            γ = _PMD.var(pm, nw, :switch_state, s)
            JuMP.@constraint(pm.model,  (z_block_fr - z_block_to) <=  (1-γ))
            JuMP.@constraint(pm.model,  (z_block_fr - z_block_to) >= -(1-γ))
    end
        # Binary enforcement for blocks
     for b in _PMD.ids(pm, nw, :blocks)
         z_block = _PMD.var(pm, nw, :z_block, b)

         n_gen = length(_PMD.ref(pm, nw, :block_gens, b))
         n_strg = length(_PMD.ref(pm, nw, :block_storages, b))
         n_neg_loads = length([_b for (_b,ls) in _PMD.ref(pm, nw, :block_loads) if any(any(_PMD.ref(pm, nw, :load, l, "pd") .< 0) for l in ls)])

    #     # Sum of switch states connected to this block
	   # @info _PMD.var(pm, nw, :switch_state) _PMD.ids(pm, nw, :block_switches)  _PMD.ids(pm, nw, :switch_dispatchable)
         switch_sum = sum(_PMD.var(pm, nw, :switch_state, s) for s in _PMD.ids(pm, nw, :block_switches) if s in _PMD.ids(pm, nw, :switch_dispatchable))
        
    #     # Total resources available to the block
         total_resources = n_gen + n_strg + n_neg_loads + switch_sum

    #     # EXISTING: Upper bound constraint
         JuMP.@constraint(pm.model, z_block <= total_resources)
    end  
        # NEW: Binary enforcement constraint
        # If total_resources > 0, block can be 1, otherwise must be 0
        # Using Big M formulation: z_block <= M * (total_resources > 0)
        # Since total_resources is integer, (total_resources > 0) ≡ (total_resources ≥ 1)
        
        # SIMPLER METHOD: Direct constraint when switches are forced
        # If total_resources = 0, force z_block = 0
        # If total_resources > 0, allow z_block ∈ {0,1} but bias toward 1
        
        # Constraint: z_block can only be 1 if resources exist
        # This automatically makes z_block binary when combined with the upper bound
       # epsilon = 0.001  # Small value to handle numerical precision
        
        # Add constraint to force binary behavior
        # z_block * (1 - total_resources) <= epsilon
        # This forces z_block ≈ 0 when total_resources = 0
       # JuMP.@constraint(pm.model, z_block * (1.0 + epsilon - total_resources) <= epsilon)
    #end
         """
        Add if have more time, but not a part of the initial model and may resolve with feasiblity
        """

end

function constraint_mc_isolate_block_ref(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, bounded::Bool=true, report::Bool=true)
    for (s, switch) in _PMD.ref(pm, nw, :switch)
            z_block_fr = _PMD.var(pm, nw, :z_block)[_PMD.ref(pm, nw, :bus_block_map)[switch["f_bus"]]]
            z_block_to = _PMD.var(pm, nw, :z_block)[_PMD.ref(pm, nw, :bus_block_map)[switch["t_bus"]]]

            γ = switch["state"]
            JuMP.@constraint(pm.model,  (z_block_fr - z_block_to) <=  (1-γ))
            JuMP.@constraint(pm.model,  (z_block_fr - z_block_to) >= -(1-γ))
    end
    #     # Binary enforcement for blocks
     for b in _PMD.ids(pm, nw, :blocks)
         z_block = _PMD.var(pm, nw, :z_block, b)

         n_gen = length(_PMD.ref(pm, nw, :block_gens, b))
         n_strg = length(_PMD.ref(pm, nw, :block_storages, b))
         n_neg_loads = length([_b for (_b,ls) in _PMD.ref(pm, nw, :block_loads) if any(any(_PMD.ref(pm, nw, :load, l, "pd") .< 0) for l in ls)])

    #     # Sum of switch states connected to this block
	    #@info _PMD.ref(pm, nw, :switch) _PMD.ids(pm, nw, :block_switches)  _PMD.ids(pm, nw, :switch_dispatchable)
         switch_sum = sum(_PMD.ref(pm, nw, :switch, s)["state"] for s in _PMD.ids(pm, nw, :block_switches) if s in _PMD.ids(pm, nw, :switch_dispatchable))
        
    #     # Total resources available to the block
         total_resources = n_gen + n_strg + n_neg_loads + switch_sum
        JuMP.@constraint(pm.model, z_block <= total_resources)
                #     # EXISTING: Upper bound constraint
#         JuMP.@constraint(pm.model, z_block == _PMD.ref(pm, nw, :block, b)["state"] )
    end  
end

function constraint_mc_gen_power_on_off(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=nw_id_default)::Nothing
    gen = _PMD.ref(pm, nw, :gen, i)
    ncnds = length(gen["connections"])

    pmin = get(gen, "pmin", fill(0, ncnds))
    pmax = get(gen, "pmax", fill( 0, ncnds))
    qmin = get(gen, "qmin", fill(0, ncnds))
    qmax = get(gen, "qmax", fill( 0, ncnds))

    constraint_mc_gen_power_on_off(pm, nw, i, gen["connections"], pmin, pmax, qmin, qmax)
    #nothing
end

"on/off constraint for generators"
function constraint_mc_gen_power_on_off(pm::_PMD.AbstractUnbalancedPowerModel, nw::Int, i::Int, connections::Vector{<:Int}, pmin::Vector{<:Real}, pmax::Vector{<:Real}, qmin::Vector{<:Real}, qmax::Vector{<:Real})
    pg = _PMD.var(pm, nw, :pg, i)
    qg = _PMD.var(pm, nw, :qg, i)
    z = _PMD.var(pm, nw, :z_gen, i)

    for (idx, c) in enumerate(connections)
        #if isfinite(pmax[idx])
            JuMP.@constraint(pm.model, pg[c] .<= pmax[idx].*z)
        #end

        #if isfinite(pmin[idx])
            JuMP.@constraint(pm.model, pg[c] .>= pmin[idx].*z)
        #end

        #if isfinite(qmax[idx])
            JuMP.@constraint(pm.model, qg[c] .<= qmax[idx].*z)
        #end

        #if isfinite(qmin[idx])
            JuMP.@constraint(pm.model, qg[c] .>= qmin[idx].*z)
        #end
    end
end

"""
Constraint to ensure logical consistency between block status, switch states, and load serving.
A block can only be energized (z_block = 1) if:
1. It has local generation, OR
2. It is connected to an energized block via a closed switch

This prevents the model from having energized blocks that are isolated from all sources.
"""
function constraint_mc_block_energization_consistency(pm::_PMD.AbstractUnbalancedPowerModel, nw::Int=nw_id_default)
            # quick determination of blocks to shed:
        # if no generation resources (gen, storage, or negative loads (e.g., rooftop pv models))
        # and no switches connected to the block are closed, then the island must be shed,
        # otherwise, to shed or not will be determined by feasibility
        # Initialize constraint dictionary if it doesn't exist
    if !haskey(_PMD.con(pm, nw), :block_energization)
        _PMD.con(pm, nw)[:block_energization] = Dict{Int, JuMP.ConstraintRef}()
    end
        for b in _PMD.ids(pm, nw, :blocks)
            z_block = _PMD.var(pm, nw, :z_block, b)

            # # connect individual dispatchable loads to blocks
            # for i in _PMD.ref(pm, nw, :block_dispatchable_loads, b)
            #     JuMP.@constraint(pm.model, _PMD.var(pm, nw, :z_demand, i) <= z_block)
            # end

            """
            Changed switch dispatchable to switch for test
            """
            n_gen = length(_PMD.ref(pm, nw, :block_gens, b))
            n_strg = length(_PMD.ref(pm, nw, :block_storages, b))
            n_neg_loads = length([_b for (_b,ls) in _PMD.ref(pm, nw, :block_loads) if any(any(_PMD.ref(pm, nw, :load, l, "pd") .< 0) for l in ls)])

            #JuMP.@constraint(pm.model, z_block <= n_gen + n_strg + n_neg_loads + sum(_PMD.var(pm, nw, :switch_state, s) for s in _PMD.ids(pm, nw, :block_switches) if s in _PMD.ids(pm, nw, :switch)))
            
            """
                If the generation in the block is equal to the load, allow the block to be on. Else, ensure the switches are connected
            """
            total_gen = 0
            total_load = 0
            for g in _PMD.ref(pm, nw, :block_gens, b)
                println("block gen is $g")
                total_gen += sum(_PMD.var(pm, nw, :pg)[g])
            end
            for d in _PMD.ref(pm, nw, :block_loads, b)
                total_load += sum(_PMD.var(pm, nw, :pd)[d])
            end
            
            if total_gen >= total_load
                _PMD.con(pm, nw, :block_energization)[b] =  JuMP.@constraint(pm.model, z_block <= n_gen + sum(_PMD.var(pm, nw, :switch_state, s) for s in _PMD.ref(pm, nw, :block_switches)[b] ))
            else
                _PMD.con(pm, nw, :block_energization)[b] =  JuMP.@constraint(pm.model, z_block <=  sum(_PMD.var(pm, nw, :switch_state, s) for s in _PMD.ref(pm, nw, :block_switches)[b] ))
            end
        end
end

function constraint_set_block_state_rounded(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, report::Bool=true)
    z_block = _PMD.var(pm, nw)[:z_block]  
    if !haskey(_PMD.con(pm, nw), :block_rounded_states)
        _PMD.con(pm, nw)[:block_rounded_states] = Dict{Int, Vector{JuMP.ConstraintRef}}()
    end

    JuMP.@constraint(pm.model, block_rounded_states[b in keys(_PMD.ref(pm, nw, :block))], 
        _PMD.var(pm, nw)[:z_block][b] == _PMD.ref(pm, nw, :block, b)["state"]
    )
end

function constraint_set_switch_state_rounded(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, report::Bool=true)
    # Force correct type (overwrite if wrong type exists)
    _PMD.con(pm, nw)[:switch_rounded_states] = Dict{Int, JuMP.ConstraintRef}()
  
    for (s, switch) in _PMD.ref(pm, nw, :switch)
        _PMD.con(pm, nw)[:switch_rounded_states][s] = JuMP.@constraint(
            pm.model,
            _PMD.var(pm, nw, :switch_state)[s] == switch["state"]
        )
    end
    
    if report
        _PMD.sol(pm, nw)[:switch_rounded_states] = _PMD.con(pm, nw)[:switch_rounded_states]
    end
end

"""
Alternative formulation using big-M constraints for better numerical properties
"""
function constraint_mc_block_energization_consistency_bigm(pm::_PMD.AbstractUnbalancedPowerModel, n::Int=nw_id_default)
    
    if !haskey(_PMD.con(pm, n), :block_energization)
        _PMD.con(pm, n)[:block_energization] = Dict{Int, Vector{JuMP.ConstraintRef}}()
    end
    
    blocks = _PMD.ref(pm, n, :blocks)
    
    for (block_id, block_data) in blocks
        z_block = _PMD.var(pm, n, :z_block)[block_id]
        constraints_for_block = []
        
        # Check for local generation
        has_local_generation = false
        if haskey(_PMD.ref(pm, n), :gen)
            for (gen_id, gen_data) in _PMD.ref(pm, n, :gen)
                if _PMD.ref(pm, n, :bus_block_map)[gen_data["gen_bus"]] == block_id
                    has_local_generation = true
                    break
                end
            end
        end
        
        if has_local_generation
            continue  # Block with generation can always be energized
        end
        
        # Find connecting switches
        connecting_switches = []
        connected_blocks = []
        
        if haskey(_PMD.ref(pm, n), :switch)
            for (switch_id, switch_data) in _PMD.ref(pm, n, :switch)
                f_bus = switch_data["f_bus"]
                t_bus = switch_data["t_bus"]
                f_block = _PMD.ref(pm, n, :bus_block_map)[f_bus]
                t_block = _PMD.ref(pm, n, :bus_block_map)[t_bus]
                
                if f_block == block_id && t_block != block_id
                    push!(connecting_switches, switch_id)
                    push!(connected_blocks, t_block)
                elseif t_block == block_id && f_block != block_id
                    push!(connecting_switches, switch_id)
                    push!(connected_blocks, f_block)
                end
            end
        end
        
        if isempty(connecting_switches)
            # No connections possible, block must be off
            push!(constraints_for_block, JuMP.@constraint(pm.model, z_block == 0))
        else
            # For each potential connection, add constraint using auxiliary binary variables
            for (switch_id, connected_block_id) in zip(connecting_switches, connected_blocks)
                z_switch = _PMD.var(pm, n, :switch_state)[switch_id]
                z_connected_block = _PMD.var(pm, n, :z_block)[connected_block_id]
                
                # If this block is energized, then either this switch is open OR 
                # (this switch is closed AND connected block is energized)
                # Equivalently: z_block <= (1 - z_switch) + z_switch * z_connected_block
                # Which simplifies to: z_block <= 1 - z_switch + z_switch * z_connected_block
                
                push!(constraints_for_block, 
                    JuMP.@constraint(pm.model, z_block <= 1 - z_switch + z_switch * z_connected_block))
            end
            # # If the block contains the substation_blocks, ensure it is on
            # if haskey(_PMD.ref(pm, n), :substation_blocks)
            #     for (sb_id, sb_data) in _PMD.ref(pm, n, :substation_blocks)
            #         sb_bus = sb_data["bus"]
            #         sb_block = _PMD.ref(pm, n, :bus_block_map)[sb_bus]
            #         if sb_block == block_id
            #             push!(constraints_for_block, JuMP.@constraint(pm.model, z_block == 1))
            #             break
            #         end
            #     end
            # end
        end
        
        _PMD.con(pm, n, :block_energization)[block_id] = constraints_for_block
    end
end



function constraint_switch_budget(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    max_switch = length(_PMD.ids(pm, nw, :switch))
    JuMP.@constraint(pm.model, sum(_PMD.var(pm, nw, :switch_state, s) for s in _PMD.ids(pm, nw, :switch_dispatchable)) <= max_switch)
    JuMP.@constraint(pm.model, sum(_PMD.var(pm, nw, :switch_state, s) for s in _PMD.ids(pm, nw, :switch_dispatchable)) >= 1)
end

function constraint_block_budget(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    max_blocks = length(_PMD.ids(pm, nw, :blocks))
    max_block_con = JuMP.@constraint(pm.model, sum(_PMD.var(pm, nw, :z_block, b) for b in _PMD.ids(pm, nw, :blocks)) <= max_blocks)
    min_block_con =JuMP.@constraint(pm.model, sum(_PMD.var(pm, nw, :z_block, b) for b in _PMD.ids(pm, nw, :blocks)) >= 1)
end

function constraint_rounded_switch_states(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, z_bern::Dict{Int,Int})
    for s in _PMD.ids(pm, nw, :switch)
       # println("The switch id is $s")
        if s in collect(keys(z_bern))
            JuMP.@constraint(pm.model, _PMD.var(pm, nw, :switch_state, s) == z_bern[s])
        end
    end
end

function constraint_rounded_switch_states_f(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, z_bern::Dict{Int,Float64})
    for s in _PMD.ids(pm, nw, :switch)
        #println("The switch id is $s")
        if s in collect(keys(z_bern))
            JuMP.@constraint(pm.model, _PMD.var(pm, nw, :switch_state, s) == z_bern[s])
        end
    end
end

function constraint_rounded_block_states(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, z_bern::Dict{Int,Int})
    for (b, block) in _PMD.ref(pm, nw, :blocks)
        JuMP.@constraint(pm.model, _PMD.var(pm, nw, :z_block, b) == z_bern[b])
    end
end
function constraint_rounded_block_states_f(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, z_bern::Dict{Int,Float64})
    for (b, block) in _PMD.ref(pm, nw, :blocks)
        JuMP.@constraint(pm.model, _PMD.var(pm, nw, :z_block, b) == z_bern[b])
    end
end

function constraint_open_switches(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    for s in _PMD.ids(pm, nw, :switch)
        JuMP.@constraint(pm.model, _PMD.var(pm, nw, :switch_state, s) == 0 )
    end
end
function constraint_mc_switch_state_on_off_top(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=nw_id_default, relax::Bool=true)::Nothing
    switch = _PMD.ref(pm, nw, :switch, i)
    f_bus = switch["f_bus"]
    t_bus = switch["t_bus"]

    f_idx = (i, f_bus, t_bus)

    #if switch["dispatchable"] != 0
        FairLoadDelivery.constraint_mc_switch_state_on_off(pm, nw, i, f_bus, t_bus, switch["f_connections"], switch["t_connections"]; relax=relax)
        FairLoadDelivery.constraint_mc_switch_power_on_off(pm, nw, f_idx; relax=relax)
    #else
    #     if switch["state"] != 0
    #         _PMD.constraint_mc_switch_state_closed(pm, nw, f_bus, t_bus, switch["f_connections"], switch["t_connections"])
    #     else
    #         _PMD.constraint_mc_switch_state_open(pm, nw, f_idx)
    #     end
    # end
    nothing
end

function constraint_mc_switch_state_on_off(pm::_PMD.AbstractUnbalancedWModels, nw::Int, i::Int, f_bus::Int, t_bus::Int, f_connections::Vector{Int}, t_connections::Vector{Int}; relax::Bool=true)
    w_fr = _PMD.var(pm, nw, :w, f_bus)
    w_to = _PMD.var(pm, nw, :w, t_bus)

    z_switch = _PMD.var(pm, nw, :switch_state, i)
    M = 1*10^3

    for (fc, tc) in zip(f_connections, t_connections)
        if relax
            JuMP.@constraint(pm.model, w_fr[fc] - w_to[tc] <=  M * (1-z_switch))
            JuMP.@constraint(pm.model, w_fr[fc] - w_to[tc] >= -M * (1-z_switch))
        else
            JuMP.@constraint(pm.model, z => {w_fr[fc] == w_to[tc]})
        end
    end
end

function constraint_mc_switch_power_on_off(pm::_PMD.AbstractUnbalancedPowerModel, nw::Int, f_idx::Tuple{Int,Int,Int}; relax::Bool=true)::Nothing
    i, f_bus, t_bus = f_idx

    psw = _PMD.var(pm, nw, :psw, f_idx)
    qsw = _PMD.var(pm, nw, :qsw, f_idx)

    z = _PMD.var(pm, nw, :switch_state, i)
    M = 1*10^3
    connections = _PMD.ref(pm, nw, :switch, i)["f_connections"]
    # determine rating
       if haskey(_PMD.ref(pm, nw, :switch, i), "current_rating")
            bound = maximum(_PMD.ref(pm, nw, :switch, i)["current_rating"][1])
        else
            bound = M
        end
    
    #rating = get(_PMD.ref(pm, nw, :switch, i), "rate_a", fill(bound, length(connections)))

    rating = get(_PMD.ref(pm, nw, :switch, i), "current_rating", fill(bound, length(connections)))
    for (idx, c) in enumerate(connections)
        # if JuMP.has_upper_bound(psw[c]) && JuMP.has_lower_bound(psw[c])
        #     continue
        # else
        #     if JuMP.has_upper_bound(psw[c]) && !JuMP.has_lower_bound(psw[c])
        #         FairLoadDelivery.set_lower_bound(psw[c],-M)
        #         FairLoadDelivery.set_lower_bound(qsw[c],-M)
        #     else
        if !JuMP.has_upper_bound(psw[c]) 
            FairLoadDelivery.set_upper_bound(psw[c], M)
        end
        if !JuMP.has_lower_bound(psw[c]) 
            FairLoadDelivery.set_lower_bound(psw[c], -M)
        end
        if !JuMP.has_upper_bound(qsw[c]) 
            FairLoadDelivery.set_upper_bound(qsw[c], M)
        end 
        if !JuMP.has_lower_bound(qsw[c]) 
            FairLoadDelivery.set_lower_bound(qsw[c], -M)
        end              
        #     end
        # end
        if relax
            JuMP.@constraint(pm.model, psw[c] <=  rating[idx] * z)
            JuMP.@constraint(pm.model, psw[c] >= -rating[idx] * z)
            JuMP.@constraint(pm.model, qsw[c] <=  rating[idx] * z)
            JuMP.@constraint(pm.model, qsw[c] >= -rating[idx] * z)
        else
            JuMP.@constraint(pm.model, !z => {psw[c] == 0.0})
            JuMP.@constraint(pm.model, !z => {qsw[c] == 0.0})
        end
    end
    nothing
end

function constraint_mc_power_balance_shed(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=nw_id_default)::Nothing
    bus = _PMD.ref(pm, nw, :bus, i)
    bus_arcs = _PMD.ref(pm, nw, :bus_arcs_conns_branch, i)
    bus_arcs_sw = _PMD.ref(pm, nw, :bus_arcs_conns_switch, i)
    bus_arcs_trans = _PMD.ref(pm, nw, :bus_arcs_conns_transformer, i)
    bus_gens = _PMD.ref(pm, nw, :bus_conns_gen, i)
    bus_storage = _PMD.ref(pm, nw, :bus_conns_storage, i)
    bus_loads = _PMD.ref(pm, nw, :bus_conns_load, i)
    bus_shunts = _PMD.ref(pm, nw, :bus_conns_shunt, i)

    if !haskey(_PMD.con(pm, nw), :lam_kcl_r)
        _PMD.con(pm, nw)[:lam_kcl_r] = Dict{Int,Array{JuMP.ConstraintRef}}()
    end

    if !haskey(_PMD.con(pm, nw), :lam_kcl_i)
        _PMD.con(pm, nw)[:lam_kcl_i] = Dict{Int,Array{JuMP.ConstraintRef}}()
    end

    FairLoadDelivery.constraint_mc_power_balance_shed(pm, nw, i, bus["terminals"], bus["grounded"], bus_arcs, bus_arcs_sw, bus_arcs_trans, bus_gens, bus_storage, bus_loads, bus_shunts)
    nothing
end
"KCL for load shed problem with transformers (AbstractWForms)"
function constraint_mc_power_balance_shed(pm::_PMD.AbstractUnbalancedPowerModel, nw::Int, i::Int, terminals::Vector{Int}, grounded::Vector{Bool}, bus_arcs::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_sw::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_arcs_trans::Vector{Tuple{Tuple{Int,Int,Int},Vector{Int}}}, bus_gens::Vector{Tuple{Int,Vector{Int}}}, bus_storage::Vector{Tuple{Int,Vector{Int}}}, bus_loads::Vector{Tuple{Int,Vector{Int}}}, bus_shunts::Vector{Tuple{Int,Vector{Int}}})
    w        = _PMD.var(pm, nw, :w, i)
    p        = get(_PMD.var(pm, nw),    :p, Dict()); _PMD._check_var_keys(p, bus_arcs, "active power", "branch")
    q        = get(_PMD.var(pm, nw),    :q, Dict()); _PMD._check_var_keys(q, bus_arcs, "reactive power", "branch")
    pg       = get(_PMD.var(pm, nw),   :pg, Dict()); _PMD._check_var_keys(pg, bus_gens, "active power", "generator")
    qg       = get(_PMD.var(pm, nw),   :qg, Dict()); _PMD._check_var_keys(qg, bus_gens, "reactive power", "generator")
    ps       = get(_PMD.var(pm, nw),   :ps, Dict()); _PMD._check_var_keys(ps, bus_storage, "active power", "storage")
    qs       = get(_PMD.var(pm, nw),   :qs, Dict()); _PMD._check_var_keys(qs, bus_storage, "reactive power", "storage")
    psw      = get(_PMD.var(pm, nw),  :psw, Dict()); _PMD._check_var_keys(psw, bus_arcs_sw, "active power", "switch")
    qsw      = get(_PMD.var(pm, nw),  :qsw, Dict()); _PMD._check_var_keys(qsw, bus_arcs_sw, "reactive power", "switch")
    pt       = get(_PMD.var(pm, nw),   :pt, Dict()); _PMD._check_var_keys(pt, bus_arcs_trans, "active power", "transformer")
    qt       = get(_PMD.var(pm, nw),   :qt, Dict()); _PMD._check_var_keys(qt, bus_arcs_trans, "reactive power", "transformer")
    z_demand = _PMD.var(pm, nw, :z_demand)
    z_shunt  = _PMD.var(pm, nw, :z_shunt)

    Gt, Bt = _PMD._build_bus_shunt_matrices(pm, nw, terminals, bus_shunts)

    cstr_p = []
    cstr_q = []

    ungrounded_terminals = [(idx,t) for (idx,t) in enumerate(terminals) if !grounded[idx]]

    for (idx, t) in ungrounded_terminals
        cp = JuMP.@constraint(pm.model,
              sum(p[a][t] for (a, conns) in bus_arcs if t in conns)
            + sum(psw[a_sw][t] for (a_sw, conns) in bus_arcs_sw if t in conns)
            + sum(pt[a_trans][t] for (a_trans, conns) in bus_arcs_trans if t in conns)
            ==
            sum(pg[g][t] for (g, conns) in bus_gens if t in conns)
            - sum(ps[s][t] for (s, conns) in bus_storage if t in conns)
            - sum(_PMD.ref(pm, nw, :load, l, "pd")[findfirst(isequal(t), conns)] * z_demand[l] for (l, conns) in bus_loads if t in conns)
            - sum(z_shunt[sh] *(w[t] * LinearAlgebra.diag(Gt')[idx]) for (sh, conns) in bus_shunts if t in conns)
        )
        push!(cstr_p, cp)
        cq = JuMP.@constraint(pm.model,
              sum(q[a][t] for (a, conns) in bus_arcs if t in conns)
            + sum(qsw[a_sw][t] for (a_sw, conns) in bus_arcs_sw if t in conns)
            + sum(qt[a_trans][t] for (a_trans, conns) in bus_arcs_trans if t in conns)
            ==
            sum(qg[g][t] for (g, conns) in bus_gens if t in conns)
            - sum(qs[s][t] for (s, conns) in bus_storage if t in conns)
            - sum(_PMD.ref(pm, nw, :load, l, "qd")[findfirst(isequal(t), conns)]*z_demand[l] for (l, conns) in bus_loads if t in conns)
            - sum(z_shunt[sh] * (-w[t] * LinearAlgebra.diag(Bt')[idx]) for (sh, conns) in bus_shunts if t in conns)
        )
        push!(cstr_q, cq)
    end

    _PMD.con(pm, nw, :lam_kcl_r)[i] = cstr_p
    _PMD.con(pm, nw, :lam_kcl_i)[i] = cstr_q

    if _IM.report_duals(pm)
        _PMD.sol(pm, nw, :bus, i)[:lam_kcl_r] = cstr_p
        _PMD.sol(pm, nw, :bus, i)[:lam_kcl_i] = cstr_q
    end
end
"""
constraint_connect_block_load(pm::AbstractUnbalancedPowerModel, nw::Int)
"""
function constraint_connect_block_load(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    for (i, load) in _PMD.ref(pm, nw, :load)
        z_demand = _PMD.var(pm, nw, :z_demand, i)
        z_block = _PMD.var(pm, nw, :z_block, _PMD.ref(pm, nw, :load_block_map, i))

        JuMP.@constraint(pm.model,  z_demand == z_block)
    end
end

function constraint_connect_block_gen(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    M = 1*10^3
    for (i, gen) in _PMD.ref(pm, nw, :gen)
        z_gen = _PMD.var(pm, nw, :z_gen, i)
        z_block = _PMD.var(pm, nw, :z_block, _PMD.ref(pm, nw, :gen_block_map, i))

        JuMP.@constraint(pm.model, z_gen == z_block)
    end
end

function constraint_connect_block_storage(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    for (i, strg) in _PMD.ref(pm, nw, :storage)
        z_strg = _PMD.var(pm, nw, :z_storage, i)
        z_block = _PMD.var(pm, nw, :z_block, _PMD.ref(pm, nw, :storage_block_map, i))

        JuMP.@constraint(pm.model, z_strg == z_block)
    end
end

function constraint_connect_block_shunt(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    for (i, shunt) in _PMD.ref(pm, nw, :shunt)
        z_shunt = _PMD.var(pm, nw, :z_shunt, i)
        z_block = _PMD.var(pm, nw, :z_block, _PMD.ref(pm, nw, :shunt_block_map, i))

        JuMP.@constraint(pm.model, z_shunt == z_block)
    end
end

function constraint_connect_block_voltage(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default)
    M=1 
    for (i, bus) in _PMD.ref(pm, nw, :bus)
        z_block = _PMD.var(pm, nw, :z_block, _PMD.ref(pm, nw, :bus_block_map, i))
        z_voltage = _PMD.var(pm, nw, :z_voltage, i)
        for t in bus["terminals"]
            JuMP.@constraint(pm.model, z_voltage >= z_block)
        end
    end
end
"""
constraint_radial_topology(pm::AbstractUnbalancedPowerModel, nw::Int; relax::Bool=false)

Constraint to enforce a radial topology

See 10.1109/TSG.2020.2985087
"""
function constraint_radial_topology(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, relax::Bool=false)
    # doi: 10.1109/TSG.2020.2985087
    _PMD.var(pm, nw)[:f] = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
    _PMD.var(pm, nw)[:lambda] = Dict{Tuple{Int,Int},JuMP.VariableRef}()
    _PMD.var(pm, nw)[:beta] = Dict{Tuple{Int,Int},JuMP.VariableRef}()
    _PMD.var(pm, nw)[:alpha] = Dict{Tuple{Int,Int},Union{JuMP.VariableRef,JuMP.AffExpr,Int}}()

    # "real" node and branch sets
    N₀ = _PMD.ids(pm, nw, :blocks)
    L₀ = _PMD.ref(pm, nw, :block_pairs)

    # Add "virtual" iᵣ to N
    virtual_iᵣ = maximum(N₀)+1
    N = [N₀..., virtual_iᵣ]
    iᵣ = [virtual_iᵣ]

    # create a set L of all branches, including virtual branches between iᵣ and all other nodes in L₀
    L = [L₀..., [(virtual_iᵣ, n) for n in N₀]...]

    # create a set L′ that inlcudes the branch reverses
    L′ = union(L, Set([(j,i) for (i,j) in L]))

    # create variables fᵏ and λ over all L, including virtual branches connected to iᵣ
    for (i,j) in L′
        for k in filter(kk->kk∉iᵣ,N)
            _PMD.var(pm, nw, :f)[(k, i, j)] = JuMP.@variable(pm.model, base_name="f_$((k,i,j))", start=(k,i,j) == (k,virtual_iᵣ,k) ? 1 : 0)
        end
        #_PMD.var(pm, nw, :lambda)[(i,j)] = JuMP.@variable(pm.model, base_name="lambda_$((i,j))", binary=!relax, lower_bound=0, upper_bound=1, start=(i,j) == (virtual_iᵣ,j) ? 1 : 0)
        _PMD.var(pm, nw, :lambda)[(i,j)] = JuMP.@variable(pm.model, base_name="lambda_$((i,j))", lower_bound=0, upper_bound=1, start=(i,j) == (virtual_iᵣ,j) ? 1 : 0)

        # create variable β over only original set L₀
        if (i,j) ∈ L₀
            _PMD.var(pm, nw, :beta)[(i,j)] = JuMP.@variable(pm.model, base_name="beta_$((i,j))", lower_bound=0, upper_bound=1)
        end
    end

    # create an aux varible α that maps to the switch states
    switch_lookup = Dict{Tuple{Int,Int},Vector{Int}}((_PMD.ref(pm, nw, :bus_block_map, sw["f_bus"]), _PMD.ref(pm, nw, :bus_block_map, sw["t_bus"])) => Int[ss for (ss,ssw) in _PMD.ref(pm, nw, :switch) if (_PMD.ref(pm, nw, :bus_block_map, sw["f_bus"])==_PMD.ref(pm, nw, :bus_block_map, ssw["f_bus"]) && _PMD.ref(pm, nw, :bus_block_map, sw["t_bus"])==_PMD.ref(pm, nw, :bus_block_map, ssw["t_bus"])) || (_PMD.ref(pm, nw, :bus_block_map, sw["f_bus"])==_PMD.ref(pm, nw, :bus_block_map, ssw["t_bus"]) && _PMD.ref(pm, nw, :bus_block_map, sw["t_bus"])==_PMD.ref(pm, nw, :bus_block_map, ssw["f_bus"]))] for (s,sw) in _PMD.ref(pm, nw, :switch))
    for ((i,j), switches) in switch_lookup
        _PMD.var(pm, nw, :alpha)[(i,j)] = JuMP.@expression(pm.model, sum(_PMD.var(pm, nw, :switch_state, s) for s in switches))
        JuMP.@constraint(pm.model, _PMD.var(pm, nw, :alpha, (i,j)) <= 1)
    end

    f = _PMD.var(pm, nw, :f)
    λ = _PMD.var(pm, nw, :lambda)
    β = _PMD.var(pm, nw, :beta)
    α = _PMD.var(pm, nw, :alpha)

    # Eq. (1) -> Eqs. (3-8)
    for k in filter(kk->kk∉iᵣ,N)
        # Eq. (3)
        for _iᵣ in iᵣ
            jiᵣ = filter(((j,i),)->i==_iᵣ&&i!=j,L)
            iᵣj = filter(((i,j),)->i==_iᵣ&&i!=j,L)
            if !(isempty(jiᵣ) && isempty(iᵣj))
                c = JuMP.@constraint(
                    pm.model,
                    sum(f[(k,j,i)] for (j,i) in jiᵣ) -
                    sum(f[(k,i,j)] for (i,j) in iᵣj)
                    ==
                    -1.0
                )
            end
        end

        # Eq. (4)
        jk = filter(((j,i),)->i==k&&i!=j,L′)
        kj = filter(((i,j),)->i==k&&i!=j,L′)
        if !(isempty(jk) && isempty(kj))
            c = JuMP.@constraint(
                pm.model,
                sum(f[(k,j,k)] for (j,i) in jk) -
                sum(f[(k,k,j)] for (i,j) in kj)
                ==
                1.0
            )
        end

        # Eq. (5)
        for i in filter(kk->kk∉iᵣ&&kk!=k,N)
            ji = filter(((j,ii),)->ii==i&&ii!=j,L′)
            ij = filter(((ii,j),)->ii==i&&ii!=j,L′)
            if !(isempty(ji) && isempty(ij))
                c = JuMP.@constraint(
                    pm.model,
                    sum(f[(k,j,i)] for (j,ii) in ji) -
                    sum(f[(k,i,j)] for (ii,j) in ij)
                    ==
                    0.0
                )
            end
        end

        # Eq. (6)
        for (i,j) in L
            JuMP.@constraint(pm.model, f[(k,i,j)] >= 0)
            JuMP.@constraint(pm.model, f[(k,i,j)] <= λ[(i,j)])
            JuMP.@constraint(pm.model, f[(k,j,i)] >= 0)
            JuMP.@constraint(pm.model, f[(k,j,i)] <= λ[(j,i)])
        end
    end

    # Eq. (7)
    JuMP.@constraint(pm.model, sum((λ[(i,j)] + λ[(j,i)]) for (i,j) in L) == length(N) - 1)

    # Connect λ and β, map β back to α, over only real switches (L₀)
    for (i,j) in L₀
        # Eq. (8)
        JuMP.@constraint(pm.model, λ[(i,j)] + λ[(j,i)] == β[(i,j)])

        # Eq. (2)
        JuMP.@constraint(pm.model, α[(i,j)] <= β[(i,j)])
    end
end

function constraint_mc_radiality(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=nw_id_default, report::Bool=true)
    
    # Get references
    branches = _PMD.ref(pm, nw, :branch)
    switches = _PMD.ref(pm, nw, :switch)
    buses = _PMD.ref(pm, nw, :bus)
    
    # Identify virtual root and substation nodes
    root_nodes = get(_PMD.ref(pm, nw), :root_nodes, [])
    substation_nodes = get(_PMD.ref(pm, nw), :substation_nodes, [])
    
    # Create beta variables for spanning tree (β_lijt)
    # β[(e, i, j)] represents edge e from node i to node j
    if !haskey(_PMD.var(pm, nw), :beta)
        # Use existing arc definitions from PowerModelsDistribution
        arcs_from = _PMD.ref(pm, nw, :arcs_branch_from)
        arcs_to = _PMD.ref(pm, nw, :arcs_branch_to)
        
        # Combine both directions
        beta_indices = vcat(arcs_from, arcs_to)
        
        _PMD.var(pm, nw)[:beta] = JuMP.@variable(
            pm.model,
            [idx in beta_indices],
            lower_bound = 0.0,
            upper_bound = 1.0,
            base_name = "beta"
        )
    end

    beta = _PMD.var(pm, nw, :beta)
    z_switch = _PMD.var(pm, nw, :switch_state)
    
    # Initialize constraint dictionaries
    if !haskey(_PMD.con(pm, nw), :radiality_beta_switch)
        _PMD.con(pm, nw)[:radiality_beta_switch] = Dict{Int, JuMP.ConstraintRef}()
    end
    if !haskey(_PMD.con(pm, nw), :radiality_beta_nonswitchable)
        _PMD.con(pm, nw)[:radiality_beta_nonswitchable] = Dict{Int, JuMP.ConstraintRef}()
    end
    if !haskey(_PMD.con(pm, nw), :radiality_no_parent_for_root)
        _PMD.con(pm, nw)[:radiality_no_parent_for_root] = Dict{Tuple, JuMP.ConstraintRef}()
    end
    if !haskey(_PMD.con(pm, nw), :radiality_substation_parent)
        _PMD.con(pm, nw)[:radiality_substation_parent] = Dict{Tuple, JuMP.ConstraintRef}()
    end
    if !haskey(_PMD.con(pm, nw), :radiality_one_parent)
        _PMD.con(pm, nw)[:radiality_one_parent] = Dict{Int, JuMP.ConstraintRef}()
    end
    
    # Constraint: β_lijt + β_ljit = z^sw_lijt for switchable edges
    for (s, switch) in switches
        # Get branch arcs for this switch
        branch_id = switch["branch_id"]
        
        # Find the corresponding arcs
        arc_from = findfirst(arc -> arc[1] == branch_id, _PMD.ref(pm, nw, :arcs_branch_from))
        arc_to = findfirst(arc -> arc[1] == branch_id, _PMD.ref(pm, nw, :arcs_branch_to))
        
        if !isnothing(arc_from) && !isnothing(arc_to)
            arc_from_tuple = _PMD.ref(pm, nw, :arcs_branch_from)[arc_from]
            arc_to_tuple = _PMD.ref(pm, nw, :arcs_branch_to)[arc_to]
            
            _PMD.con(pm, nw)[:radiality_beta_switch][s] = JuMP.@constraint(
                pm.model,
                beta[arc_from_tuple] + beta[arc_to_tuple] == z_switch[s]
            )
        end
    end

    # Constraint: β_lijt + β_ljit = 1 for non-switchable edges
    switchable_branches = Set([sw["branch_id"] for (s, sw) in switches])
    for arc_from in _PMD.ref(pm, nw, :arcs_branch_from)
        branch_id = arc_from[1]
        
        if !(branch_id in switchable_branches)
            # Find corresponding reverse arc
            arc_to = findfirst(arc -> arc[1] == branch_id, _PMD.ref(pm, nw, :arcs_branch_to))
            
            if !isnothing(arc_to)
                arc_to_tuple = _PMD.ref(pm, nw, :arcs_branch_to)[arc_to]
                
                _PMD.con(pm, nw)[:radiality_beta_nonswitchable][branch_id] = JuMP.@constraint(
                    pm.model,
                    beta[arc_from] + beta[arc_to_tuple] == 1
                )
            end
        end
    end

    # Constraint: β_lijt = 0 for edges where j is a root node
    for arc in vcat(_PMD.ref(pm, nw, :arcs_branch_from), _PMD.ref(pm, nw, :arcs_branch_to))
        branch_id, f_bus, t_bus = arc
        
        if t_bus in root_nodes
            _PMD.con(pm, nw)[:radiality_no_parent_for_root][arc] = 
                JuMP.@constraint(pm.model, beta[arc] == 0)
        end
    end

    # Constraint: β_lijt = 1 where i is root and j is substation
    for arc in vcat(_PMD.ref(pm, nw, :arcs_branch_from), _PMD.ref(pm, nw, :arcs_branch_to))
        branch_id, f_bus, t_bus = arc
        
        if f_bus in root_nodes && t_bus in substation_nodes
            _PMD.con(pm, nw)[:radiality_substation_parent][arc] = 
                JuMP.@constraint(pm.model, beta[arc] == 1)
        end
    end

    # Constraint: Each node has at most one parent
    for (j, bus) in buses
        if !(j in root_nodes)
            parent_sum = JuMP.AffExpr(0.0)
            
            # Look through all arcs that end at node j (where j is the child/to_bus)
            for arc in vcat(_PMD.ref(pm, nw, :arcs_branch_from), _PMD.ref(pm, nw, :arcs_branch_to))
                branch_id, f_bus, t_bus = arc
                
                if t_bus == j  # This arc points TO node j, so f_bus could be j's parent
                    JuMP.add_to_expression!(parent_sum, beta[arc])
                end
            end
            
            _PMD.con(pm, nw)[:radiality_one_parent][j] = JuMP.@constraint(
                pm.model,
                parent_sum <= 1
            )
        end
    end
    
    # if report
    # _PMD.sol(pm, nw)[:radiality_beta_switch] = _PMD.con(pm, nw)[:radiality_beta_switch]
    # _PMD.sol(pm, nw)[:radiality_beta_nonswitchable] = _PMD.con(pm, nw)[:radiality_beta_nonswitchable]
    # _PMD.sol(pm, nw)[:radiality_no_parent_for_root] = _PMD.con(pm, nw)[:radiality_no_parent_for_root]
    # _PMD.sol(pm, nw)[:radiality_substation_parent] = _PMD.con(pm, nw)[:radiality_substation_parent]
    # _PMD.sol(pm, nw)[:radiality_one_parent] = _PMD.con(pm, nw)[:radiality_one_parent]
    # end
end

function constraint_radial_topology_gr(pm::_PMD.AbstractUnbalancedPowerModel; nw::Int=_IM.nw_id_default, relax::Bool=false)
    _PMD.var(pm, nw)[:f] = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
    _PMD.var(pm, nw)[:beta] = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
    _PMD.var(pm, nw)[:s] = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
    _PMD.var(pm, nw)[:z] = Dict{Int,JuMP.VariableRef}()

    # Get load blocks B and switched lines S from the network model
    B₀ = _PMD.ids(pm, nw, :blocks) # the indices for the blocks
    N₀ = _PMD.ids(pm, nw, :bus)
    
    # Extract keys from dictionaries - these are the actual indices
    B_keys = collect(keys(_PMD.ref(pm, nw, :blocks)))  # Block indices
    BS_keys = collect(keys(_PMD.ref(pm, nw, :substation_blocks)))  # Substation block indices  
    S_keys = collect(keys(_PMD.ref(pm, nw, :switch)))  # Switch indices
    
    # Use the key collections for iterations
    B = B_keys
    BS = BS_keys  
    S = S_keys
    
    # Add virtual source to ensure radiality for isolated blocks
    V = maximum(N₀) + 1
    N = [N₀..., V]
    Nᵣ = [V]
    
    # Create parent-child variables β over switched lines and load blocks
    for l in S
        # Virtual root to load block connections: βlVn
        for n in B
            _PMD.var(pm, nw, :beta)[(l, V, n)] = JuMP.@variable(pm.model, 
                base_name="beta_$((l,V,n))", 
                binary=!relax, 
                lower_bound=0, 
                upper_bound=1,
                start=n ∈ BS ? 1 : 0)  # Start at 1 for substation blocks
        end
        
        # Load block to load block connections: βlmn
        for m in B, n in B
            if m != n  # Avoid self-connections
                _PMD.var(pm, nw, :beta)[(l, m, n)] = JuMP.@variable(pm.model, 
                    base_name="beta_$((l,m,n))", 
                    binary=!relax, 
                    lower_bound=0, 
                    upper_bound=1,
                    start=0)
            end
        end
    end

    # Create line status variables s for switched lines
    for l in S
        for m in B, n in B
            if m != n
                _PMD.var(pm, nw, :s)[(l, m, n)] = JuMP.@variable(pm.model, 
                    base_name="s_$((l,m,n))", 
                    binary=!relax, 
                    lower_bound=0, 
                    upper_bound=1,
                    start=0)
            end
        end
    end

    # Create energization variables z for load blocks
    for n in B
        _PMD.var(pm, nw, :z)[n] = JuMP.@variable(pm.model, 
            base_name="z_$(n)", 
            binary=!relax, 
            lower_bound=0, 
            upper_bound=1,
            start=n ∈ BS ? 1 : 0)  # Start at 1 for substation blocks
    end

    # Create virtual flow variables f over switched lines and load blocks
    B_size = length(B)  # |B| for bounds

    for l in S
        # Virtual root flows: flVn
        for n in B
            _PMD.var(pm, nw, :f)[(l, V, n)] = JuMP.@variable(pm.model, 
                base_name="f_$((l,V,n))", 
                lower_bound=0,
                upper_bound=B_size,  # Will be bounded by |B|βlVn in constraints
                start=n ∈ BS ? 1 : 0)
        end
        
        # Load block to load block flows: flmn
        for m in B, n in B
            if m != n
                _PMD.var(pm, nw, :f)[(l, m, n)] = JuMP.@variable(pm.model, 
                    base_name="f_$((l,m,n))", 
                    lower_bound=-B_size,  # Will be bounded by ±|B|slmn in constraints
                    upper_bound=B_size,
                    start=0)
            end
        end
    end
end


function constraint_radial_topology_jump(model::JuMP.Model,reference::Dict{Symbol,Any}, switch_state)
    f = Dict{Tuple{Int,Int,Int},JuMP.VariableRef}()
    lambda = Dict{Tuple{Int,Int},JuMP.VariableRef}()
    beta = Dict{Tuple{Int,Int},JuMP.VariableRef}()
    alpha = Dict{Tuple{Int,Int},Union{JuMP.VariableRef,JuMP.AffExpr,Int}}()

    # "real" node and branch sets
    N₀ = keys(reference[:blocks])
    L₀ = reference[:block_pairs]

    # Add "virtual" iᵣ to N
    virtual_iᵣ = maximum(N₀)+1
    N = [N₀..., virtual_iᵣ]
    iᵣ = [virtual_iᵣ]

    # create a set L of all branches, including virtual branches between iᵣ and all other nodes in L₀
    L = [L₀..., [(virtual_iᵣ, n) for n in N₀]...]

    # create a set L′ that inlcudes the branch reverses
    L′ = union(L, Set([(j,i) for (i,j) in L]))

    # create variables fᵏ and λ over all L, including virtual branches connected to iᵣ
    for (i,j) in L′
        for k in filter(kk->kk∉iᵣ,N)
           f[(k, i, j)] = JuMP.@variable(model, base_name="f_$((k,i,j))", start=(k,i,j) == (k,virtual_iᵣ,k) ? 1 : 0)
        end
        #_PMD.var(pm, nw, :lambda)[(i,j)] = JuMP.@variable(model, base_name="lambda_$((i,j))", binary=!relax, lower_bound=0, upper_bound=1, start=(i,j) == (virtual_iᵣ,j) ? 1 : 0)
        lambda[(i,j)] = JuMP.@variable(model, base_name="lambda_$((i,j))", lower_bound=0, upper_bound=1, start=(i,j) == (virtual_iᵣ,j) ? 1 : 0)

        # create variable β over only original set L₀
        if (i,j) ∈ L₀
            beta[(i,j)] = JuMP.@variable(model, base_name="beta_$((i,j))", lower_bound=0, upper_bound=1)
        end
    end

    # create an aux varible α that maps to the switch states
	switch_lookup = Dict{Tuple{Int,Int},Vector{Int}}()

	for (s, sw) in reference[:switch]
		f_bus_block = reference[:bus_block_map][sw["f_bus"]]
		t_bus_block = reference[:bus_block_map][sw["t_bus"]]
		key = (f_bus_block, t_bus_block)
		
		switch_ids = Int[]
		for (ss, ssw) in reference[:switch]
			ssw_f_bus_block = reference[:bus_block_map][ssw["f_bus"]]
			ssw_t_bus_block = reference[:bus_block_map][ssw["t_bus"]]
			
			# Check if switches connect the same blocks (either direction)
			if (f_bus_block == ssw_f_bus_block && t_bus_block == ssw_t_bus_block) ||
			(f_bus_block == ssw_t_bus_block && t_bus_block == ssw_f_bus_block)
				push!(switch_ids, ss)
			end
		end
		
		switch_lookup[key] = switch_ids
	end
    println("The switch lookup is: $switch_lookup")
    for ((i,j), switches) in switch_lookup
        alpha[(i,j)] = JuMP.@expression(model, sum(switch_state[s] for s in switches))
        JuMP.@constraint(model, alpha[(i,j)] <= 1)
    end

    f = f#JuMP.@variable(model,)
    λ = lambda #JuMP.@variable(model)
    β = beta#JuMP.@variable(model)
    α = alpha#JuMP.@variable(model)

    # Eq. (1) -> Eqs. (3-8)
    for k in filter(kk->kk∉iᵣ,N)
        # Eq. (3)
        for _iᵣ in iᵣ
            jiᵣ = filter(((j,i),)->i==_iᵣ&&i!=j,L)
            iᵣj = filter(((i,j),)->i==_iᵣ&&i!=j,L)
            if !(isempty(jiᵣ) && isempty(iᵣj))
                c = JuMP.@constraint(
                    model,
                    sum(f[(k,j,i)] for (j,i) in jiᵣ) -
                    sum(f[(k,i,j)] for (i,j) in iᵣj)
                    ==
                    -1.0
                )
            end
        end

        # Eq. (4)
        jk = filter(((j,i),)->i==k&&i!=j,L′)
        kj = filter(((i,j),)->i==k&&i!=j,L′)
        if !(isempty(jk) && isempty(kj))
            c = JuMP.@constraint(
                model,
                sum(f[(k,j,k)] for (j,i) in jk) -
                sum(f[(k,k,j)] for (i,j) in kj)
                ==
                1.0
            )
        end

        # Eq. (5)
        for i in filter(kk->kk∉iᵣ&&kk!=k,N)
            ji = filter(((j,ii),)->ii==i&&ii!=j,L′)
            ij = filter(((ii,j),)->ii==i&&ii!=j,L′)
            if !(isempty(ji) && isempty(ij))
                c = JuMP.@constraint(
                    model,
                    sum(f[(k,j,i)] for (j,ii) in ji) -
                    sum(f[(k,i,j)] for (ii,j) in ij)
                    ==
                    0.0
                )
            end
        end

        # Eq. (6)
        for (i,j) in L
            JuMP.@constraint(model, f[(k,i,j)] >= 0)
            JuMP.@constraint(model, f[(k,i,j)] <= λ[(i,j)])
            JuMP.@constraint(model, f[(k,j,i)] >= 0)
            JuMP.@constraint(model, f[(k,j,i)] <= λ[(j,i)])
        end
    end

    # Eq. (7)
    JuMP.@constraint(model, sum((λ[(i,j)] + λ[(j,i)]) for (i,j) in L) == length(N) - 1)

    # Connect λ and β, map β back to α, over only real switches (L₀)
    for (i,j) in L₀
        # Eq. (8)
        JuMP.@constraint(model, λ[(i,j)] + λ[(j,i)] == β[(i,j)])

        # Eq. (2)
        JuMP.@constraint(model, α[(i,j)] <= β[(i,j)])
    end
end

"""
    constraint_mc_model_voltage_magnitude_difference(pm::AbstractUnbalancedPowerModel, i::Int; nw::Int=nw_id_default)::Nothing

Template function for constraints for modeling voltage magnitude difference across branches
"""
function constraint_mc_model_voltage_magnitude_difference_fld(pm::_PMD.LPUBFDiagModel, i::Int; nw::Int=nw_id_default)::Nothing
    n = nw
    branch = _PMD.ref(pm, nw, :branch, i)
    f_bus = branch["f_bus"]
    t_bus = branch["t_bus"]
    f_idx = (i, f_bus, t_bus)
    t_idx = (i, t_bus, f_bus)

    r = branch["br_r"]
    x = branch["br_x"]
    g_sh_fr = branch["g_fr"]
    b_sh_fr = branch["b_fr"]

    f_connections = _PMD.ref(pm, n, :branch, i)["f_connections"]
    t_connections = _PMD.ref(pm, n, :branch, i)["t_connections"]

    w_fr = _PMD.var(pm, n, :w)[f_bus]
    w_to = _PMD.var(pm, n, :w)[t_bus]

    p_fr = _PMD.var(pm, n, :p)[f_idx]
    q_fr = _PMD.var(pm, n, :q)[f_idx]

    p_s_fr = [p_fr[fc] - LinearAlgebra.diag(g_sh_fr)[idx] * w_fr[fc] for (idx, fc) in enumerate(f_connections)]
    q_s_fr = [q_fr[fc] + LinearAlgebra.diag(b_sh_fr)[idx] * w_fr[fc] for (idx, fc) in enumerate(f_connections)]

    alpha = exp(-im * 2 * pi / 3)
    Gamma = [1 alpha^2 alpha; alpha 1 alpha^2; alpha^2 alpha 1][f_connections, t_connections]

    MP = 2 * (real(Gamma) .* r + imag(Gamma) .* x)
    MQ = 2 * (real(Gamma) .* x - imag(Gamma) .* r)

    N = length(f_connections)
    M = 1*10^3
    
    # Get block indices for from and to buses
    block_fr = _PMD.ref(pm, n, :bus_block_map)[f_bus]
    block_to = _PMD.ref(pm, n, :bus_block_map)[t_bus]
    
    # Only apply constraint if buses are in the same block
    if block_fr == block_to
        z_block = _PMD.var(pm, n, :z_block)[block_to]
        
        for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
            # Voltage drop equation - only enforced when block is energized (z_block = 1)
            # When z_block = 0, Big-M makes constraints redundant
            JuMP.@constraint(pm.model, 
                w_to[tc] <= w_fr[fc] - sum(MP[idx, j] * p_s_fr[j] for j in 1:N) - sum(MQ[idx, j] * q_s_fr[j] for j in 1:N) 
                + M * (1 - z_block)
            )
            
            JuMP.@constraint(pm.model, 
                w_to[tc] >= w_fr[fc] - sum(MP[idx, j] * p_s_fr[j] for j in 1:N) - sum(MQ[idx, j] * q_s_fr[j] for j in 1:N) 
                - M * (1 - z_block)
            )
        end
    end
end

function constraint_mc_model_switch_voltage_magnitude_difference_fld(pm::_PMD.LPUBFDiagModel, i::Int; nw::Int=nw_id_default)::Nothing
    n = nw
    branch = _PMD.ref(pm, nw, :branch, i)
    f_bus = branch["f_bus"]
    t_bus = branch["t_bus"]
    f_idx = (i, f_bus, t_bus)

    r = branch["br_r"]
    x = branch["br_x"]
    g_sh_fr = branch["g_fr"]
    b_sh_fr = branch["b_fr"]

    f_connections = _PMD.ref(pm, n, :branch, i)["f_connections"]
    t_connections = _PMD.ref(pm, n, :branch, i)["t_connections"]

    w_fr = _PMD.var(pm, n, :w)[f_bus]
    w_to = _PMD.var(pm, n, :w)[t_bus]

    p_fr = _PMD.var(pm, n, :p)[f_idx]
    q_fr = _PMD.var(pm, n, :q)[f_idx]

    p_s_fr = [p_fr[fc] - LinearAlgebra.diag(g_sh_fr)[idx] * w_fr[fc] for (idx, fc) in enumerate(f_connections)]
    q_s_fr = [q_fr[fc] + LinearAlgebra.diag(b_sh_fr)[idx] * w_fr[fc] for (idx, fc) in enumerate(f_connections)]

    alpha = exp(-im * 2 * pi / 3)
    Gamma = [1 alpha^2 alpha; alpha 1 alpha^2; alpha^2 alpha 1][f_connections, t_connections]

    MP = 2 * (real(Gamma) .* r + imag(Gamma) .* x)
    MQ = 2 * (real(Gamma) .* x - imag(Gamma) .* r)

    N = length(f_connections)
    M = 1*10^3
    
    # Get block status variables
    block_fr = _PMD.ref(pm, n, :bus_block_map)[f_bus]
    block_to = _PMD.ref(pm, n, :bus_block_map)[t_bus]
    
    z_block_fr = _PMD.var(pm, n, :z_block)[block_fr]
    z_block_to = _PMD.var(pm, n, :z_block)[block_to]
    
    # Get switch state if this branch is a switch
    if haskey(branch, "switch") && branch["switch"]
        # Find corresponding switch
        switch_id = findfirst(s -> _PMD.ref(pm, n, :switch, s)["f_bus"] == f_bus && 
                                    _PMD.ref(pm, n, :switch, s)["t_bus"] == t_bus, 
                              keys(_PMD.ref(pm, n, :switch)))
        
        if !isnothing(switch_id)
            z_switch = _PMD.var(pm, n, :switch_state)[switch_id]
            
            # Constraint active when: both blocks energized AND switch closed
            for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
                JuMP.@constraint(pm.model, 
                    w_to[tc] <= w_fr[fc] - sum(MP[idx, j] * p_s_fr[j] for j in 1:N) - sum(MQ[idx, j] * q_s_fr[j] for j in 1:N) 
                    + M * (1 - z_switch )
                )
                
                JuMP.@constraint(pm.model, 
                    w_to[tc] >= w_fr[fc] - sum(MP[idx, j] * p_s_fr[j] for j in 1:N) - sum(MQ[idx, j] * q_s_fr[j] for j in 1:N) 
                    - M * (1 - z_switch )
                )
            end
        end
    else
        # Non-switchable line: active when both blocks energized
        for (idx, (fc, tc)) in enumerate(zip(f_connections, t_connections))
            JuMP.@constraint(pm.model, 
                w_to[tc] <= w_fr[fc] - sum(MP[idx, j] * p_s_fr[j] for j in 1:N) - sum(MQ[idx, j] * q_s_fr[j] for j in 1:N) 
                + M * (2 - z_block_fr - z_block_to)
            )
            
            JuMP.@constraint(pm.model, 
                w_to[tc] >= w_fr[fc] - sum(MP[idx, j] * p_s_fr[j] for j in 1:N) - sum(MQ[idx, j] * q_s_fr[j] for j in 1:N) 
                - M * (2 - z_block_fr - z_block_to)
            )
        end
    end
end

"""
The thermal limit constraints p^2 + q^2 ≤ S^2 for each branch, for both the "from" and "to" side of the branch.
The power models parser creates seperate branches for switches and the other branches in the branch dictionary are not switches.
"""
function constraint_mc_switch_ampacity(pm::_PMD.AbstractUnbalancedPowerModel, i::Int; nw::Int=nw_id_default)::Nothing
    switch = _PMD.ref(pm, nw, :switch, i)
    f_idx = (i, switch["f_bus"], switch["t_bus"])

    if !haskey(_PMD.con(pm, nw), :mu_cm_switch)
        _PMD.con(pm, nw)[:mu_cm_switch] = Dict{Tuple{Int,Int,Int}, Vector{JuMP.ConstraintRef}}()
    end

    if haskey(switch, "current_rating") && any(switch["current_rating"] .< Inf)
        constraint_mc_switch_ampacity(pm, nw, f_idx, switch["f_connections"], switch["current_rating"])
    end
    nothing
end

function constraint_mc_switch_ampacity(pm::_PMD.LPUBFDiagModel, nw::Int, f_idx::Tuple{Int,Int,Int}, f_connections::Vector{Int}, c_rating::Vector{<:Real})
    psw_fr = [_PMD.var(pm, nw, :psw, f_idx)[c] for c in f_connections]
    qsw_fr = [_PMD.var(pm, nw, :qsw, f_idx)[c] for c in f_connections]
    w_fr = [_PMD.var(pm, nw, :w, f_idx[2])[c] for c in f_connections]

    psw_sqr_fr = [JuMP.@variable(pm.model, base_name="psw_sqr_$(f_idx)[$(c)]") for c in f_connections]
    qsw_sqr_fr = [JuMP.@variable(pm.model, base_name="qsw_sqr_$(f_idx)[$(c)]") for c in f_connections]

    # get the active reactive power variables of the switch arcs 
    psw = _PMD.var(pm, nw, :psw, f_idx)
    qsw = _PMD.var(pm, nw, :qsw, f_idx)

    z_switch = _PMD.var(pm, nw, :switch_state, f_idx[1])
    # for (idx,c) in enumerate(f_connections)
    #     if isfinite(c_rating[idx])
    #         p_lb, p_ub = _IM.variable_domain(psw_fr[idx])
    #         q_lb, q_ub = _IM.variable_domain(qsw_fr[idx])
    #         w_ub = _IM.variable_domain(w_fr[idx])[2]

    #         if (!isfinite(p_lb) || !isfinite(p_ub)) && isfinite(w_ub)
    #             p_ub = sum(c_rating[isfinite.(c_rating)]) * w_ub
    #             p_lb = -p_ub
    #         end
    #         if (!isfinite(q_lb) || !isfinite(q_ub)) && isfinite(w_ub)
    #             q_ub = sum(c_rating[isfinite.(c_rating)]) * w_ub
    #             q_lb = -q_ub
    #         end

    #         #all(isfinite(b) for b in [p_lb, p_ub]) && _PMD.PolyhedralRelaxations.construct_univariate_relaxation!(pm.model, x->x^2, psw_fr[idx], psw_sqr_fr[idx], [p_lb, p_ub], false)
    #         #all(isfinite(b) for b in [q_lb, q_ub]) && _PMD.PolyhedralRelaxations.construct_univariate_relaxation!(pm.model, x->x^2, qsw_fr[idx], qsw_sqr_fr[idx], [q_lb, q_ub], false)
    #     end
    # end

#    _PMD.con(pm, nw, :mu_cm_switch)[f_idx] = mu_cm_fr = [JuMP.@constraint(pm.model, psw_sqr_fr[idx] + qsw_sqr_fr[idx] .<= z_switch[idx] * w_fr[idx] * c_rating[idx]^2) for idx in findall(c_rating .< Inf)]

    #_PMD.con(pm, nw, :mu_cm_switch)[f_idx] = mu_cm_fr = [JuMP.@constraint(pm.model, psw[idx]^2 + qsw[idx]^2 .<= z_switch * 1.1^2 * c_rating[idx]^2) for idx in findall(c_rating .< Inf)]

    #_PMD.con(pm, nw, :mu_cm_switch)[f_idx] = mu_cm_fr = [JuMP.@constraint(pm.model, psw_sqr_fr[idx] + qsw_sqr_fr[idx] .<= z_switch * 1.1 * c_rating[idx]^2) for idx in findall(c_rating .< Inf)]

    if _IM.report_duals(pm)
        _PMD.sol(pm, nw, :switch, f_idx[1])[:mu_cm_fr] = mu_cm_fr
    end
end
