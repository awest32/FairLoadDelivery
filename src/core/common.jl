
"""
Local wrapper method for JuMP.set_lower_bound, which skips NaN and infinite (-Inf only)
"""
function set_lower_bound_v(x::JuMP.VariableRef, bound::Real)
    if !(isnan(bound) || bound==-Inf)
        JuMP.set_lower_bound(x, bound)
	end
end

function set_lower_bound(x::JuMP.VariableRef, bound::Real)
    if !(isnan(bound) || bound==-Inf)
        JuMP.set_lower_bound(x, bound)
	else
		JuMP.set_lower_bound(x, 0.9)
	end
end

"""
	function set_lower_bound(
		xs::Vector{JuMP.VariableRef},
		bound::Real
	)

Local wrapper method for JuMP.set_lower_bound, which skips NaN and infinite (-Inf only).
Note that with this signature, the bound is applied to every variable in the vector.
"""

"""
	function set_upper_bound(
		x::JuMP.VariableRef,
		bound
	)

Local wrapper method for JuMP.set_upper_bound, which skips NaN and infinite (+Inf only)
"""
function set_upper_bound(x::JuMP.VariableRef, bound::Real)
    if !(isnan(bound) || bound==Inf)
        JuMP.set_upper_bound(x, bound)
	else
		JuMP.set_upper_bound(x, 1.1)
	end
end

function set_upper_bound_v(x::JuMP.VariableRef, bound::Real)
    if !(isnan(bound) || bound==Inf)
        JuMP.set_upper_bound(x, bound)
	end
end


"""
	function set_upper_bound(
		xs::Vector{JuMP.VariableRef},
		bound::Real
	)

Local wrapper method for JuMP.set_upper_bound, which skips NaN and infinite (+Inf only).
Note that with this signature, the bound is applied to every variable in the vector.
"""


function ref_add_load_blocks!(ref::Dict{Symbol,<:Any}, data::Dict{String,<:Any})
    _PMD.apply_pmd!(_ref_add_load_blocks!, ref, data; apply_to_subnetworks=true)
end

"create the load block structure and store in ref"
function _ref_add_load_blocks!(ref::Dict{Symbol,<:Any}, data::Dict{String,<:Any})
	ref[:blocks] = Dict{Int,Set}(i => block for (i,block) in enumerate(_PMD.identify_load_blocks(data)))
    ref[:block_status] = Dict{Int,Int}()
    for (b,block) in get(data, "block", Dict())
        ref[:block_status][parse(Int,b)] = block["state"]
	end
    ref[:bus_block_map] = Dict{Int,Int}(bus => b for (b,block) in ref[:blocks] for bus in block)
	ref[:substation_blocks] = Vector{Int}()
	ref[:root_nodes] = Vector{Int}()

	for (g,gen) in ref[:gen]
        startswith(gen["source_id"], "voltage_source") && push!(ref[:substation_blocks], ref[:bus_block_map][gen["gen_bus"]])
        startswith(gen["source_id"], "voltage_source") && push!(ref[:root_nodes], gen["gen_bus"])
    end	

	load_block_map = Dict{Int,Int}()
    block_load_map = Dict{Int,Int}()
	for (l,load) in get(data, "load", Dict())
		for (b,block) in ref[:blocks]
			if load["load_bus"] in block
				load_block_map[parse(Int,l)] = b
			end
		end
	end
    
	ref[:load_block_map] = load_block_map
	load_block_switches = Dict{Int,Vector{Int}}(b => Vector{Int}([]) for (b, block) in ref[:blocks])
	for (b,block) in ref[:blocks]
		for (s,switch) in get(data, "switch", Dict())
			if switch["f_bus"] in block || switch["t_bus"] in block
				if switch["dispatchable"] == 1 && switch["status"] == 1
					push!(load_block_switches[b], parse(Int,s))
				end
			end
		end
	end
	ref[:load_block_switches] = load_block_switches

	#Build block pairs for radiality constraints
    ref[:block_pairs] = filter(((x,y),)->x!=y, Set{Tuple{Int,Int}}(
            Set([(ref[:bus_block_map][sw["f_bus"]],ref[:bus_block_map][sw["t_bus"]]) for (_,sw) in ref[:switch]]),
    ))
    ref[:block_branches] = Dict{Int,Set}(b => Set{Int}() for (b,_) in ref[:blocks])
    ref[:block_loads] = Dict{Int,Set}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_weights] = Dict{Int,Real}(i => 1.0 for (i,_) in ref[:blocks])

    ref[:block_shunts] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_gens] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_storages] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:microgrid_blocks] = Dict{Int,String}()
    ref[:substation_blocks] = Vector{Int}()
    ref[:bus_inverters] = Dict{Int,Set{Tuple{Symbol,Int}}}(i => Set{Tuple{Symbol,Int}}() for (i,_) in ref[:bus])
    ref[:block_inverters] = Dict{Int,Set{Tuple{Symbol,Int}}}(b => Set{Tuple{Symbol,Int}}() for (b,_) in ref[:blocks])
    #ref[:dispatchable_loads] = Dict{Int, Dict}(i => load for (i,load) in ref[:load] if Int(load["dispatchable"]) == Int(_PMD.YES))
    #ref[:nondispatchable_loads] = Dict{Int, Dict}(i => load for (i,load) in ref[:load] if Int(load["dispatchable"]) == Int(_PMD.NO))
    #ref[:block_dispatchable_loads] = Dict{Int,Set}(i => Set{Int}() for (i,_) in ref[:blocks])

    load_weight = Dict{Int,Float64}()
    critical_loads = Dict{Int,String}()
    for (d, load) in data["load"]
        load_weight[parse(Int,d)] = load["weight"]
        if load["critical"] == 1
            critical_loads[parse(Int,d)] = load["name"]
        end
    end
    ref[:load_weights] = load_weight
    ref[:critical_loads] = critical_loads

    for (b,bus) in ref[:bus]
        if !isempty(get(bus, "microgrid_id", ""))
            ref[:block_weights][ref[:bus_block_map][b]] = 10.0
            ref[:microgrid_blocks][ref[:bus_block_map][b]] = bus["microgrid_id"]
        end
    end

    for (br,branch) in ref[:branch]
        push!(ref[:block_branches][ref[:bus_block_map][branch["f_bus"]]], br)
    end
    ref[:block_line_losses] = Dict{Int,Float64}(i => sum(Float64[_PMD.LinearAlgebra.norm(ref[:branch][br]["br_r"].+1im*ref[:branch][br]["br_x"]) for br in branches if ref[:branch][br][_PMD.pmd_math_component_status["branch"]] != _PMD.pmd_math_component_status_inactive["branch"]]) for (i,branches) in ref[:block_branches])

    for (l,load) in ref[:load]
        push!(ref[:block_loads][ref[:bus_block_map][load["load_bus"]]], l)
        ref[:block_weights][ref[:bus_block_map][load["load_bus"]]] += 1e-2 * get(load, "priority", 1)
        #Int(load["dispatchable"]) == Int(_PMD.YES) && push!(ref[:block_dispatchable_loads][ref[:bus_block_map][load["load_bus"]]], l)
    end
    ref[:load_block_map] = Dict{Int,Int}(load => b for (b,block_loads) in ref[:block_loads] for load in block_loads)

    for (s,shunt) in ref[:shunt]
        push!(ref[:block_shunts][ref[:bus_block_map][shunt["shunt_bus"]]], s)
    end
    ref[:shunt_block_map] = Dict{Int,Int}(shunt => b for (b,block_shunts) in ref[:block_shunts] for shunt in block_shunts)

    for (g,gen) in ref[:gen]
        push!(ref[:block_gens][ref[:bus_block_map][gen["gen_bus"]]], g)
        startswith(gen["source_id"], "voltage_source") && push!(ref[:substation_blocks], ref[:bus_block_map][gen["gen_bus"]])
        push!(ref[:bus_inverters][gen["gen_bus"]], (:gen, g))
        push!(ref[:block_inverters][ref[:bus_block_map][gen["gen_bus"]]], (:gen, g))
    end
    ref[:gen_block_map] = Dict{Int,Int}(gen => b for (b,block_gens) in ref[:block_gens] for gen in block_gens)

    for (s,strg) in ref[:storage]
        push!(ref[:block_storages][ref[:bus_block_map][strg["storage_bus"]]], s)
        push!(ref[:bus_inverters][strg["storage_bus"]], (:storage, s))
        push!(ref[:block_inverters][ref[:bus_block_map][strg["storage_bus"]]], (:storage, s))
    end
    ref[:storage_block_map] = Dict{Int,Int}(strg => b for (b,block_storages) in ref[:block_storages] for strg in block_storages)

    for (i,_) in ref[:blocks]
        if isempty(ref[:block_loads][i]) && isempty(ref[:block_shunts][i]) && isempty(ref[:block_gens][i]) && isempty(ref[:block_storages][i])
            ref[:block_weights][i] = 0.0
        end
    end

    ref[:block_graph] = Graphs.SimpleGraph(length(ref[:blocks]))
    ref[:block_graph_edge_map] = Dict{Graphs.Edge,Int}()
    ref[:block_switches] = Dict{Int,Set{Int}}(b => Set{Int}() for (b,_) in ref[:blocks])

    for (s,switch) in ref[:switch]
        f_block = ref[:bus_block_map][switch["f_bus"]]
        t_block = ref[:bus_block_map][switch["t_bus"]]
        Graphs.add_edge!(ref[:block_graph], f_block, t_block)
        ref[:block_graph_edge_map][Graphs.Edge(f_block, t_block)] = s
        ref[:block_graph_edge_map][Graphs.Edge(t_block, f_block)] = s

        if Int(switch["dispatchable"]) == Int(_PMD.YES) && Int(switch["status"]) == Int(_PMD.ENABLED)
            push!(ref[:block_switches][f_block], s)
            push!(ref[:block_switches][t_block], s)
        end
    end

    # Build block pairs for radiality constraints
    ref[:block_pairs] = filter(((x,y),)->x!=y, Set{Tuple{Int,Int}}(
            Set([(ref[:bus_block_map][sw["f_bus"]],ref[:bus_block_map][sw["t_bus"]]) for (_,sw) in ref[:switch]]),
    ))
end
function ref_update_weights!(ref::Dict{Symbol,<:Any}, data::Dict{String,<:Any})
    _PMD.apply_pmd!(_ref_update_weights!, ref, data; apply_to_subnetworks=true)
end
function ref_add_rounded_load_blocks!(ref::Dict{Symbol,<:Any}, data::Dict{String,<:Any})
    _PMD.apply_pmd!(_ref_add_rounded_load_blocks!, ref, data; apply_to_subnetworks=true)
end

"create the load block structure and store in ref"
function _ref_add_rounded_load_blocks!(ref::Dict{Symbol,<:Any}, data::Dict{String,<:Any})
	ref[:blocks] = Dict{Int,Set}(i => block for (i,block) in enumerate(_PMD.identify_load_blocks(data)))
    ref[:block_status] = Dict{Int,Int}()
    for (b,block) in get(data, "block", Dict())
        ref[:block_status][parse(Int,b)] = block["state"]
	end
    ref[:bus_block_map] = Dict{Int,Int}(bus => b for (b,block) in ref[:blocks] for bus in block)
	ref[:substation_blocks] = Vector{Int}()
	ref[:root_nodes] = Vector{Int}()
    #block_state = Dict{Int,Int}()
    # for (b, block) in data["block"]
    #     block_state[parse(Int,b)] = block["state"]
    # end
    # ref[:block_state] = block_state

	for (g,gen) in ref[:gen]
        startswith(gen["source_id"], "voltage_source") && push!(ref[:substation_blocks], ref[:bus_block_map][gen["gen_bus"]])
        startswith(gen["source_id"], "voltage_source") && push!(ref[:root_nodes], gen["gen_bus"])
    end	

	load_block_map = Dict{Int,Int}()
    block_load_map = Dict{Int,Int}()
	for (l,load) in get(data, "load", Dict())
		for (b,block) in ref[:blocks]
			if load["load_bus"] in block
				load_block_map[parse(Int,l)] = b
			end
		end
	end
    
	ref[:load_block_map] = load_block_map
	load_block_switches = Dict{Int,Vector{Int}}(b => Vector{Int}([]) for (b, block) in ref[:blocks])
	for (b,block) in ref[:blocks]
		for (s,switch) in get(data, "switch", Dict())
			if switch["f_bus"] in block || switch["t_bus"] in block
				if switch["dispatchable"] == 1 && switch["status"] == 1
					push!(load_block_switches[b], parse(Int,s))
				end
			end
		end
	end
	ref[:load_block_switches] = load_block_switches

	#Build block pairs for radiality constraints
    ref[:block_pairs] = filter(((x,y),)->x!=y, Set{Tuple{Int,Int}}(
            Set([(ref[:bus_block_map][sw["f_bus"]],ref[:bus_block_map][sw["t_bus"]]) for (_,sw) in ref[:switch]]),
    ))
    ref[:block_branches] = Dict{Int,Set}(b => Set{Int}() for (b,_) in ref[:blocks])
    ref[:block_loads] = Dict{Int,Set}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_weights] = Dict{Int,Real}(i => 1.0 for (i,_) in ref[:blocks])
    
   load_weight = Dict{Int,Float64}()
    critical_loads = Dict{Int,String}()
    for (d, load) in data["load"]
        load_weight[parse(Int,d)] = load["weight"]
        if load["critical"] == 1
            critical_loads[parse(Int,d)] = load["name"]
        end
    end
    ref[:load_weights] = load_weight
    ref[:critical_loads] = critical_loads

    ref[:block_shunts] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_gens] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_storages] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:microgrid_blocks] = Dict{Int,String}()
    ref[:substation_blocks] = Vector{Int}()
    ref[:bus_inverters] = Dict{Int,Set{Tuple{Symbol,Int}}}(i => Set{Tuple{Symbol,Int}}() for (i,_) in ref[:bus])
    ref[:block_inverters] = Dict{Int,Set{Tuple{Symbol,Int}}}(b => Set{Tuple{Symbol,Int}}() for (b,_) in ref[:blocks])
    #ref[:dispatchable_loads] = Dict{Int, Dict}(i => load for (i,load) in ref[:load] if Int(load["dispatchable"]) == Int(_PMD.YES))
    #ref[:nondispatchable_loads] = Dict{Int, Dict}(i => load for (i,load) in ref[:load] if Int(load["dispatchable"]) == Int(_PMD.NO))
    #ref[:block_dispatchable_loads] = Dict{Int,Set}(i => Set{Int}() for (i,_) in ref[:blocks])

    for (b,bus) in ref[:bus]
        if !isempty(get(bus, "microgrid_id", ""))
            ref[:block_weights][ref[:bus_block_map][b]] = 10.0
            ref[:microgrid_blocks][ref[:bus_block_map][b]] = bus["microgrid_id"]
        end
    end

    for (br,branch) in ref[:branch]
        push!(ref[:block_branches][ref[:bus_block_map][branch["f_bus"]]], br)
    end
    ref[:block_line_losses] = Dict{Int,Float64}(i => sum(Float64[_PMD.LinearAlgebra.norm(ref[:branch][br]["br_r"].+1im*ref[:branch][br]["br_x"]) for br in branches if ref[:branch][br][_PMD.pmd_math_component_status["branch"]] != _PMD.pmd_math_component_status_inactive["branch"]]) for (i,branches) in ref[:block_branches])

    for (l,load) in ref[:load]
        push!(ref[:block_loads][ref[:bus_block_map][load["load_bus"]]], l)
        ref[:block_weights][ref[:bus_block_map][load["load_bus"]]] += 1e-2 * get(load, "priority", 1)
        #Int(load["dispatchable"]) == Int(_PMD.YES) && push!(ref[:block_dispatchable_loads][ref[:bus_block_map][load["load_bus"]]], l)
    end
    ref[:load_block_map] = Dict{Int,Int}(load => b for (b,block_loads) in ref[:block_loads] for load in block_loads)

    for (s,shunt) in ref[:shunt]
        push!(ref[:block_shunts][ref[:bus_block_map][shunt["shunt_bus"]]], s)
    end
    ref[:shunt_block_map] = Dict{Int,Int}(shunt => b for (b,block_shunts) in ref[:block_shunts] for shunt in block_shunts)

    for (g,gen) in ref[:gen]
        push!(ref[:block_gens][ref[:bus_block_map][gen["gen_bus"]]], g)
        startswith(gen["source_id"], "voltage_source") && push!(ref[:substation_blocks], ref[:bus_block_map][gen["gen_bus"]])
        push!(ref[:bus_inverters][gen["gen_bus"]], (:gen, g))
        push!(ref[:block_inverters][ref[:bus_block_map][gen["gen_bus"]]], (:gen, g))
    end
    ref[:gen_block_map] = Dict{Int,Int}(gen => b for (b,block_gens) in ref[:block_gens] for gen in block_gens)

    for (s,strg) in ref[:storage]
        push!(ref[:block_storages][ref[:bus_block_map][strg["storage_bus"]]], s)
        push!(ref[:bus_inverters][strg["storage_bus"]], (:storage, s))
        push!(ref[:block_inverters][ref[:bus_block_map][strg["storage_bus"]]], (:storage, s))
    end
    ref[:storage_block_map] = Dict{Int,Int}(strg => b for (b,block_storages) in ref[:block_storages] for strg in block_storages)

    for (i,_) in ref[:blocks]
        if isempty(ref[:block_loads][i]) && isempty(ref[:block_shunts][i]) && isempty(ref[:block_gens][i]) && isempty(ref[:block_storages][i])
            ref[:block_weights][i] = 0.0
        end
    end

    ref[:block_graph] = Graphs.SimpleGraph(length(ref[:blocks]))
    ref[:block_graph_edge_map] = Dict{Graphs.Edge,Int}()
    ref[:block_switches] = Dict{Int,Set{Int}}(b => Set{Int}() for (b,_) in ref[:blocks])

    for (s,switch) in ref[:switch]
        f_block = ref[:bus_block_map][switch["f_bus"]]
        t_block = ref[:bus_block_map][switch["t_bus"]]
        Graphs.add_edge!(ref[:block_graph], f_block, t_block)
        ref[:block_graph_edge_map][Graphs.Edge(f_block, t_block)] = s
        ref[:block_graph_edge_map][Graphs.Edge(t_block, f_block)] = s

        if Int(switch["dispatchable"]) == Int(_PMD.YES) && Int(switch["status"]) == Int(_PMD.ENABLED)
            push!(ref[:block_switches][f_block], s)
            push!(ref[:block_switches][t_block], s)
        end
    end

    # Build block pairs for radiality constraints
    ref[:block_pairs] = filter(((x,y),)->x!=y, Set{Tuple{Int,Int}}(
            Set([(ref[:bus_block_map][sw["f_bus"]],ref[:bus_block_map][sw["t_bus"]]) for (_,sw) in ref[:switch]]),
    ))
end

"create the load block structure and store in ref"
function _ref_update_weights!(ref::Dict{Symbol,<:Any}, data::Dict{String,<:Any})
	ref[:blocks] = Dict{Int,Set}(i => block for (i,block) in enumerate(_PMD.identify_load_blocks(data)))
    ref[:bus_block_map] = Dict{Int,Int}(bus => b for (b,block) in ref[:blocks] for bus in block)
	ref[:substation_blocks] = Vector{Int}()
	ref[:root_nodes] = Vector{Int}()
    block_status = Dict{Int,Int}()
    for (b, block) in data["block"]
        block_status[parse(Int,b)] = block["state"]
    end
    ref[:block_status] = block_status

	for (g,gen) in ref[:gen]
        startswith(gen["source_id"], "voltage_source") && push!(ref[:substation_blocks], ref[:bus_block_map][gen["gen_bus"]])
        startswith(gen["source_id"], "voltage_source") && push!(ref[:root_nodes], gen["gen_bus"])
    end	

	load_block_map = Dict{Int,Int}()
    block_load_map = Dict{Int,Int}()
	for (l,load) in get(data, "load", Dict())
		for (b,block) in ref[:blocks]
			if load["load_bus"] in block
				load_block_map[parse(Int,l)] = b
			end
		end
	end
    
	ref[:load_block_map] = load_block_map
	load_block_switches = Dict{Int,Vector{Int}}(b => Vector{Int}([]) for (b, block) in ref[:blocks])
	for (b,block) in ref[:blocks]
		for (s,switch) in get(data, "switch", Dict())
			if switch["f_bus"] in block || switch["t_bus"] in block
				if switch["dispatchable"] == 1 && switch["status"] == 1
					push!(load_block_switches[b], parse(Int,s))
				end
			end
		end
	end
	ref[:load_block_switches] = load_block_switches

	#Build block pairs for radiality constraints
    ref[:block_pairs] = filter(((x,y),)->x!=y, Set{Tuple{Int,Int}}(
            Set([(ref[:bus_block_map][sw["f_bus"]],ref[:bus_block_map][sw["t_bus"]]) for (_,sw) in ref[:switch]]),
    ))
    ref[:block_branches] = Dict{Int,Set}(b => Set{Int}() for (b,_) in ref[:blocks])
    ref[:block_loads] = Dict{Int,Set}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_weights] = Dict{Int,Real}(i => 1.0 for (i,_) in ref[:blocks])
    
     load_weight = Dict{Int,Int}()
    critical_loads = Dict{Int,String}()
    for (d, load) in data["load"]
        load_weight[parse(Int,d)] = load["weight"]
        if load["critical"] == 1
            critical_loads[parse(Int,d)] = load["name"]
        end
    end
    ref[:load_weights] = load_weight
    ref[:critical_loads] = critical_loads

    #ref[:load_weights] = Dict{Int,Real}(l => 10.0 for (l,_) in ref[:load])
    ref[:block_shunts] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_gens] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:block_storages] = Dict{Int,Set{Int}}(i => Set{Int}() for (i,_) in ref[:blocks])
    ref[:microgrid_blocks] = Dict{Int,String}()
    ref[:substation_blocks] = Vector{Int}()
    ref[:bus_inverters] = Dict{Int,Set{Tuple{Symbol,Int}}}(i => Set{Tuple{Symbol,Int}}() for (i,_) in ref[:bus])
    ref[:block_inverters] = Dict{Int,Set{Tuple{Symbol,Int}}}(b => Set{Tuple{Symbol,Int}}() for (b,_) in ref[:blocks])

    for (b,bus) in ref[:bus]
        if !isempty(get(bus, "microgrid_id", ""))
            ref[:block_weights][ref[:bus_block_map][b]] = 10.0
            ref[:microgrid_blocks][ref[:bus_block_map][b]] = bus["microgrid_id"]
        end
    end

    for (br,branch) in ref[:branch]
        push!(ref[:block_branches][ref[:bus_block_map][branch["f_bus"]]], br)
    end
    ref[:block_line_losses] = Dict{Int,Float64}(i => sum(Float64[_PMD.LinearAlgebra.norm(ref[:branch][br]["br_r"].+1im*ref[:branch][br]["br_x"]) for br in branches if ref[:branch][br][_PMD.pmd_math_component_status["branch"]] != _PMD.pmd_math_component_status_inactive["branch"]]) for (i,branches) in ref[:block_branches])

    for (l,load) in ref[:load]
        push!(ref[:block_loads][ref[:bus_block_map][load["load_bus"]]], l)
        ref[:block_weights][ref[:bus_block_map][load["load_bus"]]] += 1e-2 * get(load, "priority", 1)
        #Int(load["dispatchable"]) == Int(_PMD.YES) && push!(ref[:block_dispatchable_loads][ref[:bus_block_map][load["load_bus"]]], l)
    end
    ref[:load_block_map] = Dict{Int,Int}(load => b for (b,block_loads) in ref[:block_loads] for load in block_loads)

    for (s,shunt) in ref[:shunt]
        push!(ref[:block_shunts][ref[:bus_block_map][shunt["shunt_bus"]]], s)
    end
    ref[:shunt_block_map] = Dict{Int,Int}(shunt => b for (b,block_shunts) in ref[:block_shunts] for shunt in block_shunts)

    for (g,gen) in ref[:gen]
        push!(ref[:block_gens][ref[:bus_block_map][gen["gen_bus"]]], g)
        startswith(gen["source_id"], "voltage_source") && push!(ref[:substation_blocks], ref[:bus_block_map][gen["gen_bus"]])
        push!(ref[:bus_inverters][gen["gen_bus"]], (:gen, g))
        push!(ref[:block_inverters][ref[:bus_block_map][gen["gen_bus"]]], (:gen, g))
    end
    ref[:gen_block_map] = Dict{Int,Int}(gen => b for (b,block_gens) in ref[:block_gens] for gen in block_gens)

    for (s,strg) in ref[:storage]
        push!(ref[:block_storages][ref[:bus_block_map][strg["storage_bus"]]], s)
        push!(ref[:bus_inverters][strg["storage_bus"]], (:storage, s))
        push!(ref[:block_inverters][ref[:bus_block_map][strg["storage_bus"]]], (:storage, s))
    end
    ref[:storage_block_map] = Dict{Int,Int}(strg => b for (b,block_storages) in ref[:block_storages] for strg in block_storages)

    for (i,_) in ref[:blocks]
        if isempty(ref[:block_loads][i]) && isempty(ref[:block_shunts][i]) && isempty(ref[:block_gens][i]) && isempty(ref[:block_storages][i])
            ref[:block_weights][i] = 0.0
        end
    end

    ref[:block_graph] = Graphs.SimpleGraph(length(ref[:blocks]))
    ref[:block_graph_edge_map] = Dict{Graphs.Edge,Int}()
    ref[:block_switches] = Dict{Int,Set{Int}}(b => Set{Int}() for (b,_) in ref[:blocks])

    for (s,switch) in ref[:switch]
        f_block = ref[:bus_block_map][switch["f_bus"]]
        t_block = ref[:bus_block_map][switch["t_bus"]]
        Graphs.add_edge!(ref[:block_graph], f_block, t_block)
        ref[:block_graph_edge_map][Graphs.Edge(f_block, t_block)] = s
        ref[:block_graph_edge_map][Graphs.Edge(t_block, f_block)] = s

        if Int(switch["dispatchable"]) == Int(_PMD.YES) && Int(switch["status"]) == Int(_PMD.ENABLED)
            push!(ref[:block_switches][f_block], s)
            push!(ref[:block_switches][t_block], s)
        end
    end

    # Build block pairs for radiality constraints
    ref[:block_pairs] = filter(((x,y),)->x!=y, Set{Tuple{Int,Int}}(
            Set([(ref[:bus_block_map][sw["f_bus"]],ref[:bus_block_map][sw["t_bus"]]) for (_,sw) in ref[:switch]]),
    ))
end

"""
    identify_load_blocks(data::Dict{String,<:Any})

computes load blocks based on switch locations
"""
identify_load_blocks(data::Dict{String,<:Any})::Set{Set} = calc_connected_components(data; type="load_blocks")


"""
    calc_connected_components(data::Dict{String,<:Any}; edges::Union{Missing, Vector{<:String}}=missing, type::Union{Missing,String}=missing, check_enabled::Bool=true)::Set

computes the connected components of the network graph
returns a set of sets of bus ids, each set is a connected component
"""
function calc_connected_components(data::Dict{String,<:Any}; edges::Union{Missing, Vector{String}}=missing, type::Union{Missing,String}=missing, check_enabled::Bool=true)::Set{Set}
    pmd_data = get_pmd_data(data)

    if ismultinetwork(pmd_data)
        error("multinetwork data is not yet supported, recommend to use on each subnetwork independently")
    end

    if get(pmd_data, "data_model", MATHEMATICAL) == ENGINEERING
        return _calc_connected_components_eng(pmd_data; edges=ismissing(edges) ? _eng_edge_elements : edges, type=type, check_enabled=check_enabled)
    elseif get(pmd_data, "data_model", MATHEMATICAL) == MATHEMATICAL
        return _calc_connected_components_math(pmd_data; edges=ismissing(edges) ? _math_edge_elements : edges, type=type, check_enabled=check_enabled)
    else
        error("data_model `$(get(pmd_data, "data_model", MATHEMATICAL))` is unrecongized")
    end
end

function _calc_connected_components_math(data::Dict{String,<:Any}; edges::Vector{<:String}=_math_edge_elements, type::Union{Missing,String}=missing, check_enabled::Bool=true)::Set{Set{Int}}
    @assert get(data, "data_model", MATHEMATICAL) == MATHEMATICAL

    active_bus = Dict{String,Dict{String,Any}}(x for x in data["bus"] if x.second[pmd_math_component_status["bus"]] != pmd_math_component_status_inactive["bus"] || !check_enabled)
    active_bus_ids = Set{Int}([parse(Int,i) for (i,bus) in active_bus])

    neighbors = Dict{Int,Vector{Int}}(i => [] for i in active_bus_ids)
    for edge_type in edges
        for (id, edge_obj) in get(data, edge_type, Dict{Any,Dict{String,Any}}())
            if edge_obj[pmd_math_component_status[edge_type]] != pmd_math_component_status_inactive[edge_type] || !check_enabled
                if edge_type == "switch" && !ismissing(type)
                    if type == "load_blocks"
                        if edge_obj["state"] == 1
                            push!(neighbors[edge_obj["f_bus"]], edge_obj["t_bus"])
                            push!(neighbors[edge_obj["t_bus"]], edge_obj["f_bus"])
                        end
                    elseif type == "blocks"
                        if edge_obj["state"] != 0
                            push!(neighbors[edge_obj["f_bus"]], edge_obj["t_bus"])
                            push!(neighbors[edge_obj["t_bus"]], edge_obj["f_bus"])
                        end
                    end
                else
                    push!(neighbors[edge_obj["f_bus"]], edge_obj["t_bus"])
                    push!(neighbors[edge_obj["t_bus"]], edge_obj["f_bus"])
                end
            end
        end
    end

    component_lookup = Dict(i => Set{Int}([i]) for i in active_bus_ids)
    touched = Set{Int}()

    for i in active_bus_ids
        if !(i in touched)
            _cc_dfs(i, neighbors, component_lookup, touched)
        end
    end

    return Set{Set{Int}}(values(component_lookup))
end


"DFS on a graph"
function _cc_dfs(i::T, neighbors::Dict{T,Vector{T}}, component_lookup::Dict{T,Set{T}}, touched::Set{T})::Nothing where T <: Union{String,Int}
    push!(touched, i)
    for j in neighbors[i]
        if !(j in touched)
            for k in  component_lookup[j]
                push!(component_lookup[i], k)
            end
            for k in component_lookup[j]
                component_lookup[k] = component_lookup[i]
            end
            _cc_dfs(j, neighbors, component_lookup, touched)
        end
    end

    nothing
end
# function _ref_add_critical_block!(ref::Dict{Symbol,<:Any}, data::Dict{String,<:Any})#, critical_load_loc::Vector)
#     #ref[:critical_load_block_map] = Dict{Int,Int}(load => l for (l,block) in ref[:blocks] for load in block)
#   	load_block_map = Dict{Int,Int}()

#     for (l,load) in get(data, "load", Dict())
# 		for (b,block) in ref[:blocks]
# 			if load["load_bus"] in block
#                 if load["load_bus"] == critical_load_loc
# 				load_block_map[parse(Int,l)] = b
# 			end
# 		end
# 	end
#     #ref[:critical_load_block_map] = load_block_map
# end

"""
general relaxation of binlinear term (McCormick)

```
z >= JuMP.lower_bound(x)*y + JuMP.lower_bound(y)*x - JuMP.lower_bound(x)*JuMP.lower_bound(y)
z >= JuMP.upper_bound(x)*y + JuMP.upper_bound(y)*x - JuMP.upper_bound(x)*JuMP.upper_bound(y)
z <= JuMP.lower_bound(x)*y + JuMP.upper_bound(y)*x - JuMP.lower_bound(x)*JuMP.upper_bound(y)
z <= JuMP.upper_bound(x)*y + JuMP.lower_bound(y)*x - JuMP.upper_bound(x)*JuMP.lower_bound(y)
```
"""

"""
Computes the valid domain of a given JuMP variable taking into account bounds
and the varaible's implicit bounds (e.g. binary).
"""


"""
Taken from PolyhedralRelaxations.jl
    construct_univariate_relaxation!(m,f,x,y,x_partition;f_dash=x->ForwardDiff.derivative(f,x),error_tolerance=NaN64,length_tolerance=1e-6,derivative_tolerance=1e-6,num_additional_partitions=0)

Add MILP relaxation of `y=f(x)` to given JuMP model and return an object with
new variables and constraints.

# Mandatory Arguments
- `m::Jump.Model`: model to which relaxation is to be added.
- `f::Function`: function or oracle for which a polyhedral relaxation is
    required, usually non-linear.
- `x::Jump.VariableRef`: JuMP variable for domain of `f`.
- `y::JuMP.VariableRef`: JuMP variable for evaluation of `f`.
- `x_partition::Vector{<:Real}`: partition of the domain of `f`.
- `milp::Bool`: build MILP relaxation if true, LP relaxation otherwise. Note
    that the MILP relaxation uses the incremental formulation presented in the
    paper, but the LP relaxation uses a lambda form that builds a formulation
    as the convex combination of triangle vertices that are at the intersection
    of tangents to the curve.

# Optional Arguments
- `f_dash::Function`: function or oracle for derivative of `f`, defaults to 
    the `derivative` function from the `ForwardDiff` package.
- ` error_tolerance::Float64`: Maximum allowed vertical distance between over
    and under estimators of `f`, defaults to NaN64.
- `length_tolerance::Float64`: maximum length of a sub-interval in a partition,
    defaults to ``1 \\times 10^{-6}``.
- `derivative_tolerance::Float64`: minimum absolute difference between
    derivaties at successive elements of a partition for them to be considered
    different, defaults to ``1 \\times 10^{-6}``. If the difference of a partition sub-interval
    is smaller than this value, that sub-interval will be refined.
- `num_additional_partitions::Int64`: budget on number of sub-intervals in
    partition, defaults to 0. Note that if the number of partitions is `n` and
    the number of additional partitions is `m`, then the function will return a
    relaxation with at most `n+m` partitions.
- `variable_pre_base_name::AbstractString`: base_name that needs to be added to the auxiliary
    variables for meaningful LP files

Assume that:
- `f` is a bounded function of 1 variable.
- `x_partition` is a partition of the domain of `f` such that `f` is either
    convex or concave each sub-interval of the partition.
- `f_dash` is not equal at two consecutive elements of `x_partition`.

This function builds an incremental formulation, which is the formulation
presented in the paper.
"""