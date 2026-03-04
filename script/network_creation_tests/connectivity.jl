using Graphs
using Graphs

function is_connected_network(network)

    # Collect buses
    buses = collect(keys(network["bus"]))  # Strings
    bus_index = Dict(buses[i] => i for i in eachindex(buses))

    g = Graph(length(buses))

    # --- Add edges for branches (lines) ---
    for (id, branch) in network["branch"]
        if branch["br_status"] == 1
            f = string(branch["f_bus"])
            t = string(branch["t_bus"])
            if haskey(bus_index, f) && haskey(bus_index, t)
                add_edge!(g, bus_index[f], bus_index[t])
            else
                println("Warning: branch $id references missing bus")
            end
        end
    end

    # --- Add edges for transformers ---
    if haskey(network, "transformer")
        for (id, xfmr) in network["transformer"]
            if xfmr["status"] == 1
                f = string(xfmr["f_bus"])
                t = string(xfmr["t_bus"])
                if haskey(bus_index, f) && haskey(bus_index, t)
                    add_edge!(g, bus_index[f], bus_index[t])
                else
                    println("Warning: transformer $id references missing bus")
                end
            end
        end
    end

    # --- Add edges for closed switches ---
    if haskey(network, "switch")
        for (id, sw) in network["switch"]
            if sw["status"] == 1  # only closed switches
                f = string(sw["f_bus"])
                t = string(sw["t_bus"])
                if haskey(bus_index, f) && haskey(bus_index, t)
                    add_edge!(g, bus_index[f], bus_index[t])
                else
                    println("Warning: switch $id references missing bus")
                end
            end
        end
    end

    # Check connectivity
    return is_connected(g)
end
# Example usage
case = "motivation_a"
case_path = "ieee_13_aw_edit/$case.dss"
eng, math, lbs, critical_id  = setup_network(case_path,0.0,[])
is_connected_network(math)
