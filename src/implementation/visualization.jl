using Graphs
using GraphPlot
using Colors
using Statistics
import Compose
import Compose: compose, context, line, circle, text, stroke, fill, linewidth, fontsize, draw, SVG, inch, cm, mm, pt, font, hcenter, vcenter, hleft, vtop, vbottom, strokedash
using Cairo


"""
Map the bus names to their corresponding IDs from the math and solution dictionaries.
"""
function build_bus_name_maps(math::Dict{String,Any})
   # Build bus ID to name mapping
   # Strips trailing phase suffix (a, b, c) when the prefix is numeric,
   # so split buses like 670a, 670b, 670c all map to "670"
    bus_id_to_name = Dict{Int,String}()
    for (bus_id_str, bus) in math["bus"]
        bus_id = parse(Int, bus_id_str)
        bus_name = split(string(bus["source_id"]), ".")[end]
        if length(bus_name) > 1 && bus_name[end] in ('a', 'b', 'c') && all(isdigit, bus_name[1:end-1])
            bus_name = bus_name[1:end-1]
        end
        bus_id_to_name[bus_id] = bus_name
    end
    return bus_id_to_name
end
"""
A new function to extract the bus voltage per phase from the solution dictionary.
"""
function get_bus_voltage_per_phase(solution::Dict{String,Any}, math::Dict{String,Any})
    # Extract bus voltages for display
    bus_voltages = Dict{Int,Vector{Float64}}()
    for (bus_id_str, bus) in math["bus"]
        bus_id = parse(Int, bus_id_str)
        voltages = ones(3)  # Default to 1.0 per unit for 3 phases
        if haskey(solution, "bus") && haskey(solution["bus"], bus_id_str)
            bus_sol = solution["bus"][bus_id_str]
            if haskey(bus_sol, "w")
                w_vals = bus_sol["w"]
                for (id,c) in enumerate(bus["terminals"])
                    voltages[c] = sqrt(w_vals[id])
                end
            elseif haskey(bus_sol, "vm")
                vm_vals = bus_sol["vm"]
                 for (id,c) in enumerate(bus["terminals"])
                    voltages[c] = sqrt(vm_vals[id])
                end
            elseif haskey(bus_sol, "vr")
                vr_vals = bus_sol["vr"]
                vi_vals = bus_sol["vi"]
                vmag_vals = vr_vals .^ 2 .+ vi_vals .^ 2
                 for (id,c) in enumerate(bus["terminals"])
                    voltages[c] = sqrt(vmag_vals[id])
                end
            end
        end
        bus_voltages[bus_id] = voltages
    end

    return bus_voltages
end

"""
    aggregate_load_shed_by_bus(solution::Dict{String,Any}, math::Dict{String,Any})

Aggregate load shed values by bus from solution and math dictionaries.

Returns Dict{String => (bus_id, original_kw, served_kw, shed_kw)} where key is bus name.
"""
function aggregate_load_shed_by_bus_per_phase(solution::Dict{String,Any}, math::Dict{String,Any})
    # Build bus ID to name mapping
    bus_id_to_name = build_bus_name_maps(math)
    # Aggregate load shed by bus
    # Dict: bus_name => (bus_id, original_kw, served_kw, shed_kw)
    bus_load_phases = Dict{String,Dict{Int, Tuple{Float64,Float64}}}()
    bus_terminals = Dict{String,Vector{Int}}()
    for (load_id_str, load) in math["load"]
        load_bus = load["load_bus"]
        bus_name = bus_id_to_name[load_bus]
        original_demand = zeros(3)
        shed = zeros(3)
        for (id, c) in enumerate(load["connections"])
            # Original demand
            original_demand[c] = load["pd"][id]
            # Merge terminals rather than overwrite (handles split buses like 634a/b/c)
            if haskey(bus_terminals, bus_name)
                if !(c in bus_terminals[bus_name])
                    push!(bus_terminals[bus_name], c)
                end
            else
                bus_terminals[bus_name] = [c]
            end
            # Get per-phase served load from solution pd, shed = original - served
            if haskey(solution, "load") && haskey(solution["load"], load_id_str)
                load_soln = solution["load"][load_id_str]
                if haskey(load_soln, "pd")
                    pd_served = load_soln["pd"]
                    served_val = isa(pd_served, AbstractArray) ? pd_served[id] : pd_served
                    shed[c] = original_demand[c] - served_val
                end
            end
            served = original_demand[c] - shed[c]

            # Map to bus
            if haskey(bus_load_phases, bus_name)
                phase_data = bus_load_phases[bus_name]
                    if haskey(phase_data, c)
                        orig, srv = phase_data[c]
                        phase_data[c] = (orig + original_demand[c], srv + shed[c])
                    else
                        phase_data[c] = (original_demand[c], shed[c])
                    end
            else
                phase_data = Dict{Int,Tuple{Float64,Float64}}()
                    phase_data[c] = (original_demand[c], shed[c])
                bus_load_phases[bus_name] = phase_data
            end
        end
    end

    return bus_load_phases, bus_terminals
end

"""
    ieee13_bus_coordinates()

Return predefined coordinates for IEEE 13 bus feeder matching standard one-line diagram.
Returns Dict{String => (x, y)} mapping bus names to normalized coordinates.

Layout follows the standard IEEE 13 bus topology:
```
                650 (source)
                 |
                632
               / | \\
            645 633  671
             |   |  / | \\
           646  634 684 680 692
                    / \\      |
                  611 652   675
```
"""
function ieee13_bus_coordinates()
    # Coordinates designed to match standard IEEE 13 bus one-line diagram
    # Vertical layout: source on top, main trunk goes top-to-bottom
    # Rotated 90 degrees clockwise from horizontal layout
    #
    # Layout structure (top to bottom):
    #
    #                       650
    #                        |
    #                      RG60
    #                        |
    #       646 ── 645 ── 632 ── 633 ── 634
    #                        |
    #                       671 ── 692 ── 675
    #                      / |
    #                   684  680
    #                  /   \
    #                611   652
    #
    coords = Dict{String,Tuple{Float64,Float64}}(
        # Top: source bus
        "650"       => (0.50, 0.06),
        "sourcebus" => (0.50, 0.06),
        "rg60"      => (0.50, 0.06),

        # Upper horizontal line: 646 -- 645 -- 632 -- 633 -- 634
        "646"       => (0.08, 0.22),
        "645"       => (0.28, 0.22),
        "632"       => (0.50, 0.22),
        "670"       => (0.50, 0.38),  # Between 632 and 671
        "633"       => (0.72, 0.22),
        "634"       => (0.92, 0.22),

        # Lower horizontal line: 611 -- 684 -- 671 -- 692 -- 675
        "611"       => (0.08, 0.54),
        "684"       => (0.20, 0.54),
        "671"       => (0.50, 0.54),
        "692"       => (0.72, 0.54),
        "675"       => (0.92, 0.54),

        # Vertical branches down
        "652"       => (0.20, 0.72),  # Below 684
        "680"       => (0.62, 0.72),  # Below 671 (offset right)
    )
    return coords
end

"""
    extract_gen_utilization_per_phase(solution, math)

Extract per-phase generation capacity utilization for each generator, grouped by bus.
Phase identifiers use the gen["terminals"] convention (1=A, 2=B, 3=C).

Returns Dict{Int, Vector{Tuple{String, Vector{Int}, Vector{Float64}, Vector{Float64}}}}
    bus_id => [(gen_name, terminals, pmax, pg), ...]
"""
function extract_gen_utilization_per_phase(solution::Dict{String,Any}, math::Dict{String,Any})
    bus_gen_data = Dict{Int, Vector{Tuple{String, Vector{Int}, Vector{Float64}, Vector{Float64}}}}()

    if !haskey(math, "gen")
        return bus_gen_data
    end

    for (gen_id, gen) in math["gen"]
        gen_bus = gen["gen_bus"]
        gen_name = split(string(get(gen, "source_id", "gen.$gen_id")), ".")[end]
        terminals = get(gen, "terminals", Int[])
        pmax = get(gen, "pmax", Float64[])

        pg = zeros(length(pmax))
        if haskey(solution, "gen") && haskey(solution["gen"], gen_id)
            pg = get(solution["gen"][gen_id], "pg", zeros(length(pmax)))
        end

        if !haskey(bus_gen_data, gen_bus)
            bus_gen_data[gen_bus] = []
        end
        push!(bus_gen_data[gen_bus], (gen_name, terminals, pmax, pg))
    end

    return bus_gen_data
end

"""
    extract_switch_utilization_per_phase(solution, math, bus_voltages; ac_flag=false)

Extract per-phase current utilization percentage for each switch.
AC: (cr_fr² + ci_fr²) / current_rating² * 100
LD3F: S / (V * I_rating) * 100 where S = √(P² + Q²)

Phase identifiers use the switch f_terminals convention (1=A, 2=B, 3=C).

Returns Dict{Tuple{Int,Int}, Tuple{Bool, Dict{Int,Float64}}}
    (f_bus, t_bus) => (is_closed, phase => utilization_pct)
Both (f,t) and (t,f) directions are stored.
"""
function extract_switch_utilization_per_phase(solution::Dict{String,Any}, math::Dict{String,Any}, bus_voltages::Dict{Int,Vector{Float64}}; ac_flag=false)
    switch_info = Dict{Tuple{Int,Int}, Tuple{Bool, Dict{Int,Float64}}}()

    if !haskey(math, "switch")
        return switch_info
    end

    for (switch_id, sw) in math["switch"]
        f_bus = sw["f_bus"]
        t_bus = sw["t_bus"]
        is_closed = true
        phase_utils = Dict{Int,Float64}()

        if haskey(solution, "switch") && haskey(solution["switch"], switch_id)
            if !ac_flag
                state = solution["switch"][switch_id]["state"]
                is_closed = all(state .> 0.0)
            else
                state = math["switch"][switch_id]["state"]
                is_closed = all(state .> 0.0)
            end

            if is_closed
                sw_sol = solution["switch"][switch_id]
                current_rating = get(sw, "current_rating", [Inf])
                f_terminals = get(sw, "f_terminals", [1, 2, 3])

                if ac_flag
                    # AC: use cr_fr, ci_fr — util = (cr² + ci²) / current_rating² * 100
                    cr_fr = get(sw_sol, "cr_fr", Float64[])
                    ci_fr = get(sw_sol, "ci_fr", Float64[])
                    for (idx, (cr, ci)) in enumerate(zip(cr_fr, ci_fr))
                        conn = idx <= length(f_terminals) ? f_terminals[idx] : idx
                        rating = idx <= length(current_rating) ? current_rating[idx] : current_rating[1]
                        if rating > 0 && rating < Inf
                            util_pct = (cr^2 + ci^2) / rating^2 * 100
                            phase_utils[conn] = util_pct
                        end
                    end
                else
                    # LD3F: I² = (pf² + qf²) / w, util = I² / rating² * 100
                    pf = get(sw_sol, "pf", Float64[])
                    qf = get(sw_sol, "qf", Float64[])
                    f_voltages = get(bus_voltages, f_bus, ones(3))
                    for (idx, (p, q)) in enumerate(zip(pf, qf))
                        conn = idx <= length(f_terminals) ? f_terminals[idx] : idx
                        v_pu = conn <= length(f_voltages) ? f_voltages[conn] : 1.0
                        w = v_pu^2
                        rating = idx <= length(current_rating) ? current_rating[idx] : current_rating[1]
                        if rating > 0 && rating < Inf && w > 1e-6
                            i_sq = (p^2 + q^2) / w
                            util_pct = i_sq / rating^2 * 100
                            phase_utils[conn] = util_pct
                        end
                    end
                end
            end
        end

        switch_info[(f_bus, t_bus)] = (is_closed, phase_utils)
        switch_info[(t_bus, f_bus)] = (is_closed, phase_utils)
    end

    return switch_info
end

"""
Simple static SVG visualization of load shedding results
Saves directly as SVG for publication quality
Uses math dictionary directly instead of PowerModels reference.
"""
# function visualize_network_svg(solution::Dict{String,Any}, math::Dict{String,Any}; output_file="network_topology.svg", width=12, height=8)

#     println("Creating network visualization...")

#     # Build graph and extract data
#     g, bus_labels, bus_data = build_network_graph(solution, math)

#     # Get positions using spring layout
#     layout_func = spring_layout
#     locs_x, locs_y = layout_func(g)

#     # Normalize positions to [0, 1]
#     locs_x = (locs_x .- minimum(locs_x)) ./ (maximum(locs_x) - minimum(locs_x))
#     locs_y = (locs_y .- minimum(locs_y)) ./ (maximum(locs_y) - minimum(locs_y))

#     # Get colors and sizes
#     node_colors = get_node_colors(bus_data)
#     node_sizes = get_node_sizes(bus_data)
#     edge_colors, edge_widths = get_edge_properties(solution, math, g)

#     # Create plot
#     p = gplot(g, locs_x, locs_y,
#         nodelabel=bus_labels,
#         nodelabelc=colorant"black",
#         nodelabeldist=1.5,
#         nodelabelsize=3.0,
#         nodefillc=node_colors,
#         nodestrokec=colorant"black",
#         nodesize=node_sizes,
#         edgestrokec=edge_colors,
#         edgelinewidth=edge_widths,
#         EDGELINEWIDTH=0.5,
#         layout=nothing  # Use provided positions
#     )

#     # Add title and legend
#     title_text = "Load Shedding Network Topology"

#     # Draw to SVG
#     draw(SVG(output_file, width*inch, height*inch), p)

#     println("✓ Network visualization saved to $output_file")

#     # Print summary
#     print_summary(solution, math)

#     return output_file
# end

"""
Build graph structure from math dictionary
"""
# function build_network_graph(solution::Dict{String,Any}, math::Dict{String,Any})
#     # Get bus mapping
#     bus_ids = sort([parse(Int, k) for k in keys(math["bus"])])
#     bus_to_idx = Dict(bus_id => idx for (idx, bus_id) in enumerate(bus_ids))
#     n_buses = length(bus_ids)

#     # Create graph
#     g = SimpleGraph(n_buses)

#     # Add edges from branches
#     if haskey(math, "branch")
#         for (_, branch) in math["branch"]
#             f_bus = branch["f_bus"]
#             t_bus = branch["t_bus"]
#             if haskey(bus_to_idx, f_bus) && haskey(bus_to_idx, t_bus)
#                 add_edge!(g, bus_to_idx[f_bus], bus_to_idx[t_bus])
#             end
#         end
#     end

#     # Add edges from switches
#     if haskey(math, "switch")
#         for (_, switch) in math["switch"]
#             f_bus = switch["f_bus"]
#             t_bus = switch["t_bus"]
#             if haskey(bus_to_idx, f_bus) && haskey(bus_to_idx, t_bus)
#                 add_edge!(g, bus_to_idx[f_bus], bus_to_idx[t_bus])
#             end
#         end
#     end

#     # Create bus labels and data
#     bus_labels = []
#     bus_data = []

#     for bus_id in bus_ids
#         bus = math["bus"][string(bus_id)]

#         # Get voltage
#         voltage = 1.0
#         if haskey(solution, "bus") && haskey(solution["bus"], string(bus_id))
#             if haskey(solution["bus"][string(bus_id)], "vm")
#                 vm = solution["bus"][string(bus_id)]["vm"]
#                 voltage = mean(vm)
#             end
#         end

#         # Check if generator bus
#         is_generator = false
#         if haskey(math, "gen")
#             is_generator = any(gen["gen_bus"] == bus_id for (_, gen) in math["gen"])
#         end

#         # Get load info
#         load_demand = 0.0
#         load_served = 0.0
#         if haskey(math, "load")
#             for (load_id, load) in math["load"]
#                 if load["load_bus"] == bus_id
#                     load_demand += sum(load["pd"])
#                     if haskey(solution, "load") && haskey(solution["load"], load_id)
#                         load_soln = solution["load"][load_id]
#                         if haskey(load_soln, "pd_bus")
#                             load_served += sum(load_soln["pd_bus"])
#                         elseif haskey(load_soln, "pd")
#                             load_served += sum(load_soln["pd"])
#                         elseif haskey(load_soln, "pshed")
#                             # Calculate served from demand - shed
#                             load_served += sum(load["pd"]) - sum(load_soln["pshed"])
#                         end
#                     end
#                 end
#             end
#         end

#         # Create label
#         bus_name = split(string(bus["source_id"]), ".")[end]
#         if load_demand > 0
#             label = "$bus_name\n$(round(load_served, digits=0))/$(round(load_demand, digits=0))"
#         else
#             label = bus_name
#         end

#         push!(bus_labels, label)
#         push!(bus_data, Dict(
#             "voltage" => voltage,
#             "is_generator" => is_generator,
#             "load_demand" => load_demand,
#             "load_served" => load_served
#         ))
#     end

#     return g, bus_labels, bus_data
# end

"""
Get node colors based on voltage and load status
Uses shading (lightness) for load serving percentage - darker = more shed
"""
# function get_node_colors(bus_data)
#     colors = []

#     for data in bus_data
#         if data["is_generator"]
#             # Generator = yellow/gold
#             push!(colors, colorant"gold")
#         elseif data["load_demand"] > 0
#             # Load bus - use shading from dark red (0% served) to bright green (100% served)
#             served_pct = data["load_served"] / data["load_demand"]

#             # Option 1: Red shading (darker = more shed)
#             # lightness = 0.3 + 0.6 * served_pct  # 30% lightness (dark) to 90% lightness (bright)
#             # color = HSL(0, 0.8, lightness)  # Red hue with varying lightness

#             # Option 2: Single color with opacity/brightness
#             # Darker = more shed, Brighter = more served
#             lightness = 0.25 + 0.65 * served_pct  # 25% to 90% lightness
#             color = HSL(210, 0.7, lightness)  # Blue hue with varying lightness

#             push!(colors, color)
#         else
#             # Regular bus - light gray
#             push!(colors, colorant"lightgray")
#         end
#     end

#     return colors
# end

"""
Get node sizes
"""
# function get_node_sizes(bus_data)
#     sizes = []

#     for data in bus_data
#         if data["is_generator"]
#             push!(sizes, 0.08)  # Larger for generators
#         elseif data["load_demand"] > 0
#             push!(sizes, 0.06)  # Medium for loads
#         else
#             push!(sizes, 0.04)  # Smaller for regular buses
#         end
#     end

#     return sizes
# end

"""
Get edge colors and widths based on status and power flow
"""
# function get_edge_properties(solution::Dict{String,Any}, math::Dict{String,Any}, g)
#     edge_colors = []
#     edge_widths = []

#     # Create edge index mapping
#     bus_ids = sort([parse(Int, k) for k in keys(math["bus"])])
#     bus_to_idx = Dict(bus_id => idx for (idx, bus_id) in enumerate(bus_ids))

#     # Track which edges we've added
#     edge_data = Dict()

#     # Add branch data
#     if haskey(math, "branch")
#         for (branch_id, branch) in math["branch"]
#             f_bus = branch["f_bus"]
#             t_bus = branch["t_bus"]
#             if haskey(bus_to_idx, f_bus) && haskey(bus_to_idx, t_bus)
#                 f_idx = bus_to_idx[f_bus]
#                 t_idx = bus_to_idx[t_bus]
#                 edge = minmax(f_idx, t_idx)

#                 power = 0.0
#                 if haskey(solution, "branch") && haskey(solution["branch"], branch_id)
#                     if haskey(solution["branch"][branch_id], "pf")
#                         pf = solution["branch"][branch_id]["pf"]
#                         power = sum(abs.(pf))
#                     end
#                 end

#                 edge_data[edge] = Dict("power" => power, "status" => "closed", "type" => "branch")
#             end
#         end
#     end

#     # Add switch data
#     if haskey(math, "switch") && haskey(solution, "switch")
#         for (switch_id, switch) in math["switch"]
#             f_bus = switch["f_bus"]
#             t_bus = switch["t_bus"]
#             if haskey(bus_to_idx, f_bus) && haskey(bus_to_idx, t_bus)
#                 f_idx = bus_to_idx[f_bus]
#                 t_idx = bus_to_idx[t_bus]
#                 edge = minmax(f_idx, t_idx)

#                 status = "closed"
#                 power = 0.0

#                 if haskey(solution["switch"], switch_id)
#                     switch_state = solution["switch"][switch_id]["state"]
#                     status = all(switch_state .> 0.5) ? "closed" : "open"

#                     if status == "closed" && haskey(solution["switch"][switch_id], "pf")
#                         pf = solution["switch"][switch_id]["pf"]
#                         power = sum(abs.(pf))
#                     end
#                 end

#                 edge_data[edge] = Dict("power" => power, "status" => status, "type" => "switch")
#             end
#         end
#     end

#     # Assign colors and widths to edges in graph order
#     for edge in edges(g)
#         e = minmax(src(edge), dst(edge))

#         if haskey(edge_data, e)
#             data = edge_data[e]

#             # Color by status
#             if data["status"] == "open"
#                 push!(edge_colors, colorant"gray")
#                 push!(edge_widths, 1.0)
#             else
#                 push!(edge_colors, colorant"blue")
#                 # Width by power flow
#                 if data["power"] > 1000
#                     push!(edge_widths, 3.0)
#                 elseif data["power"] > 500
#                     push!(edge_widths, 2.0)
#                 else
#                     push!(edge_widths, 1.5)
#                 end
#             end
#         else
#             push!(edge_colors, colorant"blue")
#             push!(edge_widths, 1.5)
#         end
#     end

#     return edge_colors, edge_widths
# end

"""
Print summary statistics
"""
# function print_summary(solution::Dict{String,Any}, math::Dict{String,Any})
#     println("\n" * "="^60)
#     println("LOAD SHEDDING SUMMARY")
#     println("="^60)

#     # Generator info
#     total_gen_capacity = 0.0
#     total_gen_output = 0.0

#     if haskey(math, "gen")
#         for (gen_id, gen) in math["gen"]
#             total_gen_capacity += sum(gen["pmax"])
#             if haskey(solution, "gen") && haskey(solution["gen"], gen_id)
#                 if haskey(solution["gen"][gen_id], "pg")
#                     pg = solution["gen"][gen_id]["pg"]
#                     total_gen_output += sum(pg)
#                 end
#             end
#         end
#     end

#     println("Generation Capacity: $(round(total_gen_capacity, digits=1)) kW")
#     if total_gen_capacity > 0
#         println("Generation Output:   $(round(total_gen_output, digits=1)) kW ($(round(total_gen_output/total_gen_capacity*100, digits=1))%)")
#     end

#     # Load info
#     total_demand = 0.0
#     total_served = 0.0

#     if haskey(math, "load")
#         for (load_id, load) in math["load"]
#             total_demand += sum(load["pd"])
#             if haskey(solution, "load") && haskey(solution["load"], load_id)
#                 load_soln = solution["load"][load_id]
#                 if haskey(load_soln, "pd_bus")
#                     total_served += sum(load_soln["pd_bus"])
#                 elseif haskey(load_soln, "pd")
#                     total_served += sum(load_soln["pd"])
#                 elseif haskey(load_soln, "pshed")
#                     total_served += sum(load["pd"]) - sum(load_soln["pshed"])
#                 end
#             end
#         end
#     end

#     total_shed = total_demand - total_served

#     println("\nLoad Demand: $(round(total_demand, digits=1)) kW")
#     if total_demand > 0
#         println("Load Served: $(round(total_served, digits=1)) kW ($(round(total_served/total_demand*100, digits=1))%)")
#         println("Load Shed:   $(round(total_shed, digits=1)) kW ($(round(total_shed/total_demand*100, digits=1))%)")
#     end

#     # Switch info
#     if haskey(math, "switch")
#         n_switches = length(math["switch"])
#         n_open = 0

#         if haskey(solution, "switch")
#             for (_, switch_soln) in solution["switch"]
#                 if haskey(switch_soln, "state") && all(switch_soln["state"] .< 0.5)
#                     n_open += 1
#                 end
#             end
#         end

#         println("\nSwitches: $(n_switches - n_open) closed, $n_open open")
#     end

#     println("="^60)
# end



"""
    hierarchical_tree_layout(g::SimpleGraph, root::Int)

Compute hierarchical tree layout with root at top.
Returns (x_positions, y_positions) vectors.
"""
# function hierarchical_tree_layout(g::SimpleGraph, root::Int)
#     n = nv(g)
#     x_pos = zeros(Float64, n)
#     y_pos = zeros(Float64, n)

#     # BFS to determine level of each node
#     visited = falses(n)
#     level = zeros(Int, n)
#     parent = zeros(Int, n)

#     queue = [root]
#     visited[root] = true
#     level[root] = 0

#     while !isempty(queue)
#         node = popfirst!(queue)
#         for neighbor in neighbors(g, node)
#             if !visited[neighbor]
#                 visited[neighbor] = true
#                 level[neighbor] = level[node] + 1
#                 parent[neighbor] = node
#                 push!(queue, neighbor)
#             end
#         end
#     end

#     # Group nodes by level
#     max_level = maximum(level)
#     nodes_at_level = [Int[] for _ in 0:max_level]
#     for i in 1:n
#         push!(nodes_at_level[level[i]+1], i)
#     end

#     # Assign y positions (level determines y)
#     for i in 1:n
#         y_pos[i] = 1.0 - level[i] / max(max_level, 1)
#     end

#     # Assign x positions (spread nodes at each level)
#     for (lv, nodes) in enumerate(nodes_at_level)
#         n_nodes = length(nodes)
#         for (idx, node) in enumerate(nodes)
#             x_pos[node] = (idx - 0.5) / max(n_nodes, 1)
#         end
#     end

#     return x_pos, y_pos
# end

"""
    plot_network_load_shed(solution::Dict{String,Any}, math::Dict{String,Any};
        output_file::String="network_load_shed.svg",
        layout::Symbol=:tree,
        width::Int=14,
        height::Int=10)

Plot IEEE 13 bus one-line diagram with final load shed values per bus.

# Arguments
- `solution`: Solution dictionary from MLD solve (e.g., best_mld["solution"] or best_feasibility["solution"])
- `math`: Math dictionary containing network topology and bus mapping
- `output_file`: Path to save the SVG output
- `layout`: Layout algorithm - :ieee13 for standard IEEE 13 bus diagram, :tree for hierarchical, :spring for force-directed
- `width`, `height`: Dimensions in inches

# Example
```julia
plot_network_load_shed(best_mld["solution"], math_out[best_set];
    output_file="results/network_load_shed.svg")
```
"""
function plot_network_load_shed(
    solution::Dict{String,Any},
    math::Dict{String,Any};
    output_file::String="network_load_shed.svg",
    layout::Symbol=:ieee13,
    width::Int=18,
    height::Int=12,
    ac_flag::Bool=false
)
    println("Creating schematic network visualization...")

    # --- Data extraction via helpers ---
    bus_id_to_name = build_bus_name_maps(math)
    bus_voltages = get_bus_voltage_per_phase(solution, math)
    bus_load_phases, bus_terminals = aggregate_load_shed_by_bus_per_phase(solution, math)
    bus_gen_data = extract_gen_utilization_per_phase(solution, math)
    switch_info = extract_switch_utilization_per_phase(solution, math, bus_voltages; ac_flag=ac_flag)

    # Build bus name => load IDs mapping
    bus_name_to_load_ids = Dict{String,Vector{String}}()
    if haskey(math, "load")
        for (load_id_str, load) in math["load"]
            bname = get(bus_id_to_name, load["load_bus"], "")
            if !haskey(bus_name_to_load_ids, bname)
                bus_name_to_load_ids[bname] = String[]
            end
            if !(load_id_str in bus_name_to_load_ids[bname])
                push!(bus_name_to_load_ids[bname], load_id_str)
            end
        end
    end

    # Get coordinates and scale to left 70% of figure (leaving room for legend)
    ieee13_coords_raw = ieee13_bus_coordinates()
    ieee13_coords = Dict{String,Tuple{Float64,Float64}}()
    for (name, (x, y)) in ieee13_coords_raw
        ieee13_coords[name] = (0.02 + x * 0.70, y)
    end

    # Collect drawing elements in layers (back to front)
    bg_elements = []      # White backgrounds (behind everything)
    line_elements = []    # Lines drawn first (back)
    node_elements = []    # Nodes drawn on top of lines
    label_elements = []   # Labels drawn last (front)

    # Drawing constants
    line_color = colorant"darkcyan"
    switch_color = colorant"gray"
    node_radius = 0.012
    line_width = 1.425mm
    load_node_radius = 0.018
    phase_line_spacing = 0.020
    annotation_font = 6pt

    # Draw branches as straight lines
    if haskey(math, "branch")
        for (_, branch) in math["branch"]
            f_bus = branch["f_bus"]
            t_bus = branch["t_bus"]
            f_name = lowercase(get(bus_id_to_name, f_bus, ""))
            t_name = lowercase(get(bus_id_to_name, t_bus, ""))

            if haskey(ieee13_coords, f_name) && haskey(ieee13_coords, t_name)
                x1, y1 = ieee13_coords[f_name]
                x2, y2 = ieee13_coords[t_name]

                # Draw line
                push!(line_elements, compose(context(),
                    line([(x1, y1), (x2, y2)]),
                    stroke(line_color),
                    linewidth(line_width)
                ))
            end
        end
    end

    # --- Draw switches with per-phase utilization labels ---
    if haskey(math, "switch")
        for (switch_id, sw) in math["switch"]
            f_bus = sw["f_bus"]
            t_bus = sw["t_bus"]
            f_name = lowercase(get(bus_id_to_name, f_bus, ""))
            t_name = lowercase(get(bus_id_to_name, t_bus, ""))

            # Resolve virtual bus names using switch name (e.g. "632633" -> "632", "633")
            if !haskey(ieee13_coords, f_name) || !haskey(ieee13_coords, t_name)
                sw_name = get(sw, "name", "")
                if length(sw_name) == 6
                    f_name = lowercase(sw_name[1:3])
                    t_name = lowercase(sw_name[4:6])
                end
            end

            if !haskey(ieee13_coords, f_name) || !haskey(ieee13_coords, t_name)
                continue
            end

            x1, y1 = ieee13_coords[f_name]
            x2, y2 = ieee13_coords[t_name]

            is_closed, phase_utils = get(switch_info, (f_bus, t_bus), (true, Dict{Int,Float64}()))

            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            len = sqrt(dx^2 + dy^2)
            if len > 0
                dx, dy = dx / len, dy / len
            end
            perp_x, perp_y = -dy, dx
            label_offset = 0.025

            # Color switch line by max utilization percentage
            max_util = isempty(phase_utils) ? 0.0 : maximum(values(phase_utils))

            sw_color = switch_color
            if is_closed && max_util > 0
                if max_util > 100.000001
                    sw_color = colorant"red"          # Constraint violation
                elseif max_util >= 80
                    sw_color = colorant"purple"
                elseif max_util >= 50
                    sw_color = colorant"darkcyan"
                else
                    sw_color = colorant"steelblue"
                end
            elseif !is_closed
                sw_color = colorant"gray"
            end

            push!(line_elements, compose(context(),
                line([(x1, y1), (x2, y2)]),
                stroke(sw_color),
                linewidth(line_width),
                strokedash([3mm, 2mm])
            ))

            # Per-phase utilization labels (only for closed switches)
            if is_closed && !isempty(phase_utils)
                label_x_sw = mx + perp_x * label_offset
                label_y_sw = my + perp_y * label_offset
                for (i, phase) in enumerate(sort(collect(keys(phase_utils))))
                    util_pct = phase_utils[phase]
                    y_pos = label_y_sw + (i - 1) * phase_line_spacing
                    pc = if util_pct > 100.000001
                        colorant"red"
                    elseif util_pct >= 80
                        colorant"purple"
                    elseif util_pct >= 50
                        colorant"darkcyan"
                    else
                        colorant"steelblue"
                    end
                    push!(bg_elements, compose(context(),
                        Compose.rectangle(label_x_sw - 0.05, y_pos - 0.008, 0.10, 0.016),
                        fill("white"), stroke(nothing)
                    ))
                    push!(label_elements, compose(context(),
                        text(label_x_sw, y_pos, "φ$phase: $(round(util_pct, digits=1))%", hcenter, vcenter),
                        fontsize(annotation_font), fill(pc), font("sans-serif")
                    ))
                end
            end
        end
    end

    # --- Draw bus nodes with per-phase annotations ---
    # Buses that need labels shifted left
    left_shift_buses_3 = Set(["632", "670", "675", "680", "652", "611"])
    left_shift_buses_2 = Set(["671"])
    left_shift_buses_1 = Set(["646", "634"])
    label_shift_3 = 0.04
    label_shift_2 = 0.026
    label_shift_1 = 0.013

    for (bus_id, bus_name) in bus_id_to_name
        name_lower = lowercase(bus_name)

        # Skip virtual switch buses (names like 632645, 632633)
        if occursin("632", bus_name) && length(bus_name) > 3
            continue
        end

        if !haskey(ieee13_coords, name_lower)
            continue
        end

        x, y = ieee13_coords[name_lower]

        # Calculate label x position
        label_x = x
        if bus_name in left_shift_buses_3
            label_x = x - label_shift_3
        elseif bus_name in left_shift_buses_2
            label_x = x - label_shift_2
        elseif bus_name in left_shift_buses_1
            label_x = x - label_shift_1
        end

        is_load_bus = haskey(bus_load_phases, bus_name)
        is_gen_bus = haskey(bus_gen_data, bus_id)

        # Draw node circle
        if is_load_bus
            phases_data = bus_load_phases[bus_name]
            total_orig = sum(orig for (orig, _) in values(phases_data))
            total_shed = sum(shed for (_, shed) in values(phases_data))
            shed_frac = total_orig > 0 ? clamp(total_shed / total_orig, 0.0, 1.0) : 0.0
            lightness = 0.30 + 0.5 * shed_frac
            node_color = HSL(207, 0.6, lightness)
            push!(node_elements, compose(context(),
                circle(x, y, load_node_radius),
                fill(node_color), stroke("black"), linewidth(0.3mm)
            ))
        else
            push!(node_elements, compose(context(),
                circle(x, y, node_radius),
                fill("grey"), stroke("darkgrey")
            ))
        end

        # Bus label above node (show load IDs if any, otherwise just bus name)
        load_ids = get(bus_name_to_load_ids, bus_name, String[])
        bus_label = isempty(load_ids) ? bus_name : "$bus_name (L$(join(sort(load_ids), ",")))"
        push!(label_elements, compose(context(),
            text(label_x, y - 0.035, bus_label, hcenter, vbottom),
            fontsize(9pt), fill("black")
        ))

        # Running y position for stacked annotations below node
        current_y = y + 0.035

        # --- Per-phase load shed (only for load buses) ---
        if is_load_bus
            phases_data = bus_load_phases[bus_name]
            terminals = get(bus_terminals, bus_name, Int[])
            for phase in sort(terminals)
                if haskey(phases_data, phase)
                    orig, shed = phases_data[phase]
                    shed_pct = orig > 0 ? round(shed / orig * 100, digits=1) : 0.0
                    shed_label = "S$phase: $(shed_pct)%"
                else
                    shed_label = "S$phase: N/A"
                end
                push!(bg_elements, compose(context(),
                    Compose.rectangle(label_x - 0.05, current_y - 0.004, 0.10, 0.016),
                    fill("white"), stroke(nothing)
                ))
                push!(label_elements, compose(context(),
                    text(label_x, current_y, shed_label, hcenter, vtop),
                    fontsize(annotation_font), fill("black"), font("sans-serif")
                ))
                current_y += phase_line_spacing
            end
        end

        # --- Per-phase voltage ---
        voltages = get(bus_voltages, bus_id, ones(3))
        bus_terms = get(math["bus"][string(bus_id)], "terminals", [1, 2, 3])
        bus_terms_filtered = filter(t -> t <= 3, bus_terms)

        for phase in sort(bus_terms_filtered)
            v_pu = phase <= length(voltages) ? voltages[phase] : 1.0
            v_color = if v_pu < 0.95 || v_pu > 1.05
                colorant"darkmagenta"
            elseif v_pu < 0.97 || v_pu > 1.03
                colorant"goldenrod"
            else
                colorant"steelblue"
            end
            v_label = "V$phase: $(round(v_pu, digits=3)) pu"
            push!(bg_elements, compose(context(),
                Compose.rectangle(label_x - 0.06, current_y - 0.004, 0.12, 0.016),
                fill("white"), stroke(nothing)
            ))
            push!(label_elements, compose(context(),
                text(label_x, current_y, v_label, hcenter, vtop),
                fontsize(annotation_font), fill(v_color), font("sans-serif")
            ))
            current_y += phase_line_spacing
        end

        # --- Per-phase generator utilization (only for gen buses) ---
        if is_gen_bus
            for (gen_name, terminals, pmax, pg) in bus_gen_data[bus_id]
                push!(label_elements, compose(context(),
                    text(label_x, current_y, "Gen: $gen_name", hcenter, vtop),
                    fontsize(annotation_font), fill("forestgreen"), font("sans-serif")
                ))
                current_y += phase_line_spacing
                for (idx, phase) in enumerate(terminals)
                    p = idx <= length(pg) ? pg[idx] : 0.0
                    cap = idx <= length(pmax) ? pmax[idx] : 0.0
                    util_pct = cap > 0 ? round(p / cap * 100, digits=1) : 0.0
                    gen_label = "Gφ$phase: $(util_pct)%"
                    push!(bg_elements, compose(context(),
                        Compose.rectangle(label_x - 0.05, current_y - 0.004, 0.10, 0.016),
                        fill("white"), stroke(nothing)
                    ))
                    push!(label_elements, compose(context(),
                        text(label_x, current_y, gen_label, hcenter, vtop),
                        fontsize(annotation_font), fill("forestgreen"), font("sans-serif")
                    ))
                    current_y += phase_line_spacing
                end
            end
        end
    end

    # ============================================================
    # LEGEND (positioned on right side of figure)
    # ============================================================
    legend_elements = []

    # Legend position (right side, starting near top)
    leg_x = 0.75
    leg_y = 0.08
    leg_spacing = 0.032
    line_len = 0.04
    circle_r = 0.008

    # --- Switch Utilization Legend ---
    push!(legend_elements, compose(context(),
        text(leg_x, leg_y, "Switch Utilization:", hleft, vtop),
        fontsize(8pt),
        fill("black")
    ))

    util_colors = [
        (colorant"steelblue", "Under 50%"),
        (colorant"darkcyan", "50-80%"),
        (colorant"purple", "80-100%"),
        (colorant"red", ">100% (violation)"),
        (colorant"gray", "Open")
    ]

    for (i, (color, label)) in enumerate(util_colors)
        y_pos = leg_y + i * leg_spacing
        # Dashed line sample
        push!(legend_elements, compose(context(),
            line([(leg_x, y_pos), (leg_x + line_len, y_pos)]),
            stroke(color),
            linewidth(1.2mm),
            strokedash([2mm, 1mm])
        ))
        # Label
        push!(legend_elements, compose(context(),
            text(leg_x + line_len + 0.01, y_pos, label, hleft, vcenter),
            fontsize(7pt),
            fill("black")
        ))
    end

    # --- Voltage Legend ---
    volt_y = leg_y + (length(util_colors) + 1) * leg_spacing
    push!(legend_elements, compose(context(),
        text(leg_x, volt_y, "Bus Voltage:", hleft, vtop),
        fontsize(8pt),
        fill("black")
    ))

    volt_colors = [
        (colorant"steelblue", "0.97-1.03 pu (normal)"),
        (colorant"goldenrod", "0.95-0.97 or 1.03-1.05 pu"),
        (colorant"darkmagenta", "Outside 0.95-1.05 pu")
    ]

    for (i, (color, label)) in enumerate(volt_colors)
        y_pos = volt_y + i * leg_spacing
        # Colored "V:" sample
        push!(legend_elements, compose(context(),
            text(leg_x, y_pos, "V:", hleft, vcenter),
            fontsize(7pt),
            fill(color)
        ))
        # Label
        push!(legend_elements, compose(context(),
            text(leg_x + 0.025, y_pos, label, hleft, vcenter),
            fontsize(7pt),
            fill("black")
        ))
    end

    # --- Load Shed Legend ---
    shed_y = volt_y + (length(volt_colors) + 1) * leg_spacing
    push!(legend_elements, compose(context(),
        text(leg_x, shed_y, "Load Bus (% shed):", hleft, vtop),
        fontsize(8pt),
        fill("black")
    ))

    shed_levels = [
        (HSL(207, 0.6, 0.30), "0-25% shed"),
        (HSL(207, 0.6, 0.45), "25-50% shed"),
        (HSL(207, 0.6, 0.60), "50-75% shed"),
        (HSL(207, 0.6, 0.80), "75-100% shed")
    ]

    for (i, (color, label)) in enumerate(shed_levels)
        y_pos = shed_y + i * leg_spacing
        # Circle sample
        push!(legend_elements, compose(context(),
            circle(leg_x + circle_r, y_pos, circle_r),
            fill(color),
            stroke("black"),
            linewidth(0.2mm)
        ))
        # Label
        push!(legend_elements, compose(context(),
            text(leg_x + 2.5 * circle_r + 0.01, y_pos, label, hleft, vcenter),
            fontsize(7pt),
            fill("black")
        ))
    end

    # Legend background box
    leg_box_x = leg_x - 0.015
    leg_box_y = leg_y - 0.025
    leg_box_w = 0.26
    leg_box_h = shed_y + length(shed_levels) * leg_spacing - leg_y + 0.05

    legend_bg = compose(context(),
        Compose.rectangle(leg_box_x, leg_box_y, leg_box_w, leg_box_h),
        fill("white"),
        stroke("gray"),
        linewidth(0.3mm)
    )

    # Combine elements in layer order for Compose.jl (last = top/front)
    # So order is: legend_bg (back) -> bg_elements -> lines -> nodes -> legend -> labels (front)
    all_elements = []
    append!(all_elements, node_elements)
    append!(all_elements, legend_elements)
    append!(all_elements, label_elements)
    append!(all_elements, bg_elements)
    append!(all_elements, line_elements)
    push!(all_elements, legend_bg)

    # Compose all elements and draw
    drawing = compose(context(), all_elements...)

    # Save to SVG
    draw(SVG(output_file, width*cm, height*cm), drawing)

    println("Schematic network visualization saved to $output_file")

    # Print summary and save switch CSV alongside the SVG
    csv_file = replace(output_file, ".svg" => "_switch_flow.csv")
    print_load_shed_summary(solution, math; ac_flag=ac_flag, csv_file=csv_file)

    return output_file
end

"""
    print_load_shed_summary(solution, math; ac_flag=false, csv_file="")

Print summary of load shedding results and optionally save switch power flow data to CSV.
"""
function print_load_shed_summary(solution::Dict{String,Any}, math::Dict{String,Any}; ac_flag::Bool=false, csv_file::String="")
    println("\n" * "="^60)
    println("LOAD SHEDDING SUMMARY BY BUS")
    println("="^60)

    bus_load_phases, _ = aggregate_load_shed_by_bus_per_phase(solution, math)

    total_demand = 0.0
    total_served = 0.0

    for bus_name in sort(collect(keys(bus_load_phases)))
        phases_data = bus_load_phases[bus_name]
        bus_orig = sum(orig for (orig, _) in values(phases_data))
        bus_shed = sum(srv for (_, srv) in values(phases_data))
        total_demand += bus_orig
        total_served += (bus_orig - bus_shed)
        shed =  bus_shed
        shed_pct = bus_orig > 0 ? (shed / bus_orig * 100) : 0.0
        println("Bus $bus_name: Demand=$(round(bus_orig, digits=1)) kW, Served=$(round(bus_orig - bus_shed, digits=1)) kW, Shed=$(round(shed, digits=1)) kW ($(round(shed_pct, digits=1))%)")
    end

    total_shed = total_demand - total_served
    println("-"^60)
    println("TOTAL: Demand=$(round(total_demand, digits=1)) kW, Served=$(round(total_served, digits=1)) kW, Shed=$(round(total_shed, digits=1)) kW")
    println("Overall served: $(round(total_served/max(total_demand,1e-6)*100, digits=1))%")
    println("="^60)

    # --- Load shed summary per block ---
    if haskey(math, "block") && !isempty(math["block"])
        println("\n" * "="^60)
        println("LOAD SHEDDING SUMMARY BY BLOCK")
        println("="^60)
        bus_id_to_name = build_bus_name_maps(math)
        for block_id in sort(parse.(Int, collect(keys(math["block"]))))
            block = math["block"][string(block_id)]
            load_ids = get(block, "loads", Int[])
            block_demand = 0.0
            block_shed = 0.0
            block_buses = Set{String}()
            for lid in load_ids
                lid_str = string(lid)
                load = math["load"][lid_str]
                push!(block_buses, get(bus_id_to_name, load["load_bus"], string(load["load_bus"])))
                for (id, c) in enumerate(load["connections"])
                    orig = load["pd"][id]
                    served = 0.0
                    if haskey(solution, "load") && haskey(solution["load"], lid_str) && haskey(solution["load"][lid_str], "pd")
                        pd_served = solution["load"][lid_str]["pd"]
                        served = isa(pd_served, AbstractArray) ? pd_served[id] : pd_served
                    end
                    block_demand += orig
                    block_shed += (orig - served)
                end
            end
            shed_pct = block_demand > 0 ? round(block_shed / block_demand * 100, digits=1) : 0.0
            bus_list = join(sort(collect(block_buses)), ", ")
            println("Block $block_id (buses: $bus_list): Demand=$(round(block_demand, digits=1)) kW, Shed=$(round(block_shed, digits=1)) kW ($(shed_pct)%)")
        end
        println("="^60)
    end

    # Save switch power flow and current ratings to CSV
    if !isempty(csv_file) && haskey(math, "switch")
        open(csv_file, "w") do io
            println(io, "switch_name,phase,p,q,c_mag,current_rating,is_closed")
            for (switch_id, sw) in math["switch"]
                sw_name = get(sw, "name", switch_id)
                f_terminals = get(sw, "f_terminals", [1, 2, 3])
                current_rating = get(sw, "current_rating", [Inf])

                is_closed = true
                if haskey(solution, "switch") && haskey(solution["switch"], switch_id)
                    if !ac_flag
                        state = solution["switch"][switch_id]["state"]
                        is_closed = all(state .> 0.0)
                    else
                        state = math["switch"][switch_id]["state"]
                        is_closed = all(state .> 0.0)
                    end
                end

                if is_closed && haskey(solution, "switch") && haskey(solution["switch"], switch_id)
                    sw_sol = solution["switch"][switch_id]
                    pf = ac_flag ? get(sw_sol, "psw_fr", Float64[]) : get(sw_sol, "pf", Float64[])
                    qf = ac_flag ? get(sw_sol, "qsw_fr", Float64[]) : get(sw_sol, "qf", Float64[])
                    f_bus = sw["f_bus"]
                    w = get(solution["bus"][string(f_bus)], "w", ones(length(f_terminals)))
                    for (idx, (p, q)) in enumerate(zip(pf, qf))
                        conn = idx <= length(f_terminals) ? f_terminals[idx] : idx
                        rating = idx <= length(current_rating) ? current_rating[idx] : current_rating[1]
                        if ac_flag
                            cr_vals = get(sw_sol, "cr_fr", Float64[])
                            ci_vals = get(sw_sol, "ci_fr", Float64[])
                            cr_v = idx <= length(cr_vals) ? cr_vals[idx] : 0.0
                            ci_v = idx <= length(ci_vals) ? ci_vals[idx] : 0.0
                            i_mag = sqrt(cr_v^2 + ci_v^2)
                        else
                            w_val = idx <= length(w) ? w[idx] : 1.0
                            i_mag = w_val > 1e-6 ? sqrt((p^2 + q^2) / w_val) : 0.0
                        end
                        println(io, "$sw_name,$conn,$p,$q,$i_mag,$rating,$is_closed")
                    end
                else
                    for (idx, conn) in enumerate(f_terminals)
                        rating = idx <= length(current_rating) ? current_rating[idx] : current_rating[1]
                        println(io, "$sw_name,$conn,0.0,0.0,0.0,$rating,$is_closed")
                    end
                end
            end
        end
        println("Switch power flow data saved to $csv_file")
    end
end
