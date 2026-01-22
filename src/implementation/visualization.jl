using Graphs
using GraphPlot
using Colors
using Statistics
import Compose
import Compose: compose, context, line, circle, text, stroke, fill, linewidth, fontsize, draw, SVG, inch, cm, mm, pt, font, hcenter, vcenter, hleft, vtop, vbottom, strokedash
using Cairo

"""
Simple static SVG visualization of load shedding results
Saves directly as SVG for publication quality
Uses math dictionary directly instead of PowerModels reference.
"""
function visualize_network_svg(solution::Dict{String,Any}, math::Dict{String,Any}; output_file="network_topology.svg", width=12, height=8)

    println("Creating network visualization...")

    # Build graph and extract data
    g, bus_labels, bus_data = build_network_graph(solution, math)

    # Get positions using spring layout
    layout_func = spring_layout
    locs_x, locs_y = layout_func(g)

    # Normalize positions to [0, 1]
    locs_x = (locs_x .- minimum(locs_x)) ./ (maximum(locs_x) - minimum(locs_x))
    locs_y = (locs_y .- minimum(locs_y)) ./ (maximum(locs_y) - minimum(locs_y))

    # Get colors and sizes
    node_colors = get_node_colors(bus_data)
    node_sizes = get_node_sizes(bus_data)
    edge_colors, edge_widths = get_edge_properties(solution, math, g)

    # Create plot
    p = gplot(g, locs_x, locs_y,
        nodelabel=bus_labels,
        nodelabelc=colorant"black",
        nodelabeldist=1.5,
        nodelabelsize=3.0,
        nodefillc=node_colors,
        nodestrokec=colorant"black",
        nodesize=node_sizes,
        edgestrokec=edge_colors,
        edgelinewidth=edge_widths,
        EDGELINEWIDTH=0.5,
        layout=nothing  # Use provided positions
    )

    # Add title and legend
    title_text = "Load Shedding Network Topology"

    # Draw to SVG
    draw(SVG(output_file, width*inch, height*inch), p)

    println("✓ Network visualization saved to $output_file")

    # Print summary
    print_summary(solution, math)

    return output_file
end

"""
Build graph structure from math dictionary
"""
function build_network_graph(solution::Dict{String,Any}, math::Dict{String,Any})
    # Get bus mapping
    bus_ids = sort([parse(Int, k) for k in keys(math["bus"])])
    bus_to_idx = Dict(bus_id => idx for (idx, bus_id) in enumerate(bus_ids))
    n_buses = length(bus_ids)

    # Create graph
    g = SimpleGraph(n_buses)

    # Add edges from branches
    if haskey(math, "branch")
        for (_, branch) in math["branch"]
            f_bus = branch["f_bus"]
            t_bus = branch["t_bus"]
            if haskey(bus_to_idx, f_bus) && haskey(bus_to_idx, t_bus)
                add_edge!(g, bus_to_idx[f_bus], bus_to_idx[t_bus])
            end
        end
    end

    # Add edges from switches
    if haskey(math, "switch")
        for (_, switch) in math["switch"]
            f_bus = switch["f_bus"]
            t_bus = switch["t_bus"]
            if haskey(bus_to_idx, f_bus) && haskey(bus_to_idx, t_bus)
                add_edge!(g, bus_to_idx[f_bus], bus_to_idx[t_bus])
            end
        end
    end

    # Create bus labels and data
    bus_labels = []
    bus_data = []

    for bus_id in bus_ids
        bus = math["bus"][string(bus_id)]

        # Get voltage
        voltage = 1.0
        if haskey(solution, "bus") && haskey(solution["bus"], string(bus_id))
            if haskey(solution["bus"][string(bus_id)], "vm")
                vm = solution["bus"][string(bus_id)]["vm"]
                voltage = mean(vm)
            end
        end

        # Check if generator bus
        is_generator = false
        if haskey(math, "gen")
            is_generator = any(gen["gen_bus"] == bus_id for (_, gen) in math["gen"])
        end

        # Get load info
        load_demand = 0.0
        load_served = 0.0
        if haskey(math, "load")
            for (load_id, load) in math["load"]
                if load["load_bus"] == bus_id
                    load_demand += sum(load["pd"])
                    if haskey(solution, "load") && haskey(solution["load"], load_id)
                        load_soln = solution["load"][load_id]
                        if haskey(load_soln, "pd_bus")
                            load_served += sum(load_soln["pd_bus"])
                        elseif haskey(load_soln, "pd")
                            load_served += sum(load_soln["pd"])
                        elseif haskey(load_soln, "pshed")
                            # Calculate served from demand - shed
                            load_served += sum(load["pd"]) - sum(load_soln["pshed"])
                        end
                    end
                end
            end
        end

        # Create label
        bus_name = split(string(bus["source_id"]), ".")[end]
        if load_demand > 0
            label = "$bus_name\n$(round(load_served, digits=0))/$(round(load_demand, digits=0))"
        else
            label = bus_name
        end

        push!(bus_labels, label)
        push!(bus_data, Dict(
            "voltage" => voltage,
            "is_generator" => is_generator,
            "load_demand" => load_demand,
            "load_served" => load_served
        ))
    end

    return g, bus_labels, bus_data
end

"""
Get node colors based on voltage and load status
Uses shading (lightness) for load serving percentage - darker = more shed
"""
function get_node_colors(bus_data)
    colors = []

    for data in bus_data
        if data["is_generator"]
            # Generator = yellow/gold
            push!(colors, colorant"gold")
        elseif data["load_demand"] > 0
            # Load bus - use shading from dark red (0% served) to bright green (100% served)
            served_pct = data["load_served"] / data["load_demand"]

            # Option 1: Red shading (darker = more shed)
            # lightness = 0.3 + 0.6 * served_pct  # 30% lightness (dark) to 90% lightness (bright)
            # color = HSL(0, 0.8, lightness)  # Red hue with varying lightness

            # Option 2: Single color with opacity/brightness
            # Darker = more shed, Brighter = more served
            lightness = 0.25 + 0.65 * served_pct  # 25% to 90% lightness
            color = HSL(210, 0.7, lightness)  # Blue hue with varying lightness

            push!(colors, color)
        else
            # Regular bus - light gray
            push!(colors, colorant"lightgray")
        end
    end

    return colors
end

"""
Get node sizes
"""
function get_node_sizes(bus_data)
    sizes = []

    for data in bus_data
        if data["is_generator"]
            push!(sizes, 0.08)  # Larger for generators
        elseif data["load_demand"] > 0
            push!(sizes, 0.06)  # Medium for loads
        else
            push!(sizes, 0.04)  # Smaller for regular buses
        end
    end

    return sizes
end

"""
Get edge colors and widths based on status and power flow
"""
function get_edge_properties(solution::Dict{String,Any}, math::Dict{String,Any}, g)
    edge_colors = []
    edge_widths = []

    # Create edge index mapping
    bus_ids = sort([parse(Int, k) for k in keys(math["bus"])])
    bus_to_idx = Dict(bus_id => idx for (idx, bus_id) in enumerate(bus_ids))

    # Track which edges we've added
    edge_data = Dict()

    # Add branch data
    if haskey(math, "branch")
        for (branch_id, branch) in math["branch"]
            f_bus = branch["f_bus"]
            t_bus = branch["t_bus"]
            if haskey(bus_to_idx, f_bus) && haskey(bus_to_idx, t_bus)
                f_idx = bus_to_idx[f_bus]
                t_idx = bus_to_idx[t_bus]
                edge = minmax(f_idx, t_idx)

                power = 0.0
                if haskey(solution, "branch") && haskey(solution["branch"], branch_id)
                    if haskey(solution["branch"][branch_id], "pf")
                        pf = solution["branch"][branch_id]["pf"]
                        power = sum(abs.(pf))
                    end
                end

                edge_data[edge] = Dict("power" => power, "status" => "closed", "type" => "branch")
            end
        end
    end

    # Add switch data
    if haskey(math, "switch") && haskey(solution, "switch")
        for (switch_id, switch) in math["switch"]
            f_bus = switch["f_bus"]
            t_bus = switch["t_bus"]
            if haskey(bus_to_idx, f_bus) && haskey(bus_to_idx, t_bus)
                f_idx = bus_to_idx[f_bus]
                t_idx = bus_to_idx[t_bus]
                edge = minmax(f_idx, t_idx)

                status = "closed"
                power = 0.0

                if haskey(solution["switch"], switch_id)
                    switch_state = solution["switch"][switch_id]["state"]
                    status = all(switch_state .> 0.5) ? "closed" : "open"

                    if status == "closed" && haskey(solution["switch"][switch_id], "pf")
                        pf = solution["switch"][switch_id]["pf"]
                        power = sum(abs.(pf))
                    end
                end

                edge_data[edge] = Dict("power" => power, "status" => status, "type" => "switch")
            end
        end
    end

    # Assign colors and widths to edges in graph order
    for edge in edges(g)
        e = minmax(src(edge), dst(edge))

        if haskey(edge_data, e)
            data = edge_data[e]

            # Color by status
            if data["status"] == "open"
                push!(edge_colors, colorant"gray")
                push!(edge_widths, 1.0)
            else
                push!(edge_colors, colorant"blue")
                # Width by power flow
                if data["power"] > 1000
                    push!(edge_widths, 3.0)
                elseif data["power"] > 500
                    push!(edge_widths, 2.0)
                else
                    push!(edge_widths, 1.5)
                end
            end
        else
            push!(edge_colors, colorant"blue")
            push!(edge_widths, 1.5)
        end
    end

    return edge_colors, edge_widths
end

"""
Print summary statistics
"""
function print_summary(solution::Dict{String,Any}, math::Dict{String,Any})
    println("\n" * "="^60)
    println("LOAD SHEDDING SUMMARY")
    println("="^60)

    # Generator info
    total_gen_capacity = 0.0
    total_gen_output = 0.0

    if haskey(math, "gen")
        for (gen_id, gen) in math["gen"]
            total_gen_capacity += sum(gen["pmax"])
            if haskey(solution, "gen") && haskey(solution["gen"], gen_id)
                if haskey(solution["gen"][gen_id], "pg")
                    pg = solution["gen"][gen_id]["pg"]
                    total_gen_output += sum(pg)
                end
            end
        end
    end

    println("Generation Capacity: $(round(total_gen_capacity, digits=1)) kW")
    if total_gen_capacity > 0
        println("Generation Output:   $(round(total_gen_output, digits=1)) kW ($(round(total_gen_output/total_gen_capacity*100, digits=1))%)")
    end

    # Load info
    total_demand = 0.0
    total_served = 0.0

    if haskey(math, "load")
        for (load_id, load) in math["load"]
            total_demand += sum(load["pd"])
            if haskey(solution, "load") && haskey(solution["load"], load_id)
                load_soln = solution["load"][load_id]
                if haskey(load_soln, "pd_bus")
                    total_served += sum(load_soln["pd_bus"])
                elseif haskey(load_soln, "pd")
                    total_served += sum(load_soln["pd"])
                elseif haskey(load_soln, "pshed")
                    total_served += sum(load["pd"]) - sum(load_soln["pshed"])
                end
            end
        end
    end

    total_shed = total_demand - total_served

    println("\nLoad Demand: $(round(total_demand, digits=1)) kW")
    if total_demand > 0
        println("Load Served: $(round(total_served, digits=1)) kW ($(round(total_served/total_demand*100, digits=1))%)")
        println("Load Shed:   $(round(total_shed, digits=1)) kW ($(round(total_shed/total_demand*100, digits=1))%)")
    end

    # Switch info
    if haskey(math, "switch")
        n_switches = length(math["switch"])
        n_open = 0

        if haskey(solution, "switch")
            for (_, switch_soln) in solution["switch"]
                if haskey(switch_soln, "state") && all(switch_soln["state"] .< 0.5)
                    n_open += 1
                end
            end
        end

        println("\nSwitches: $(n_switches - n_open) closed, $n_open open")
    end

    println("="^60)
end

"""
    aggregate_load_shed_by_bus(solution::Dict{String,Any}, math::Dict{String,Any})

Aggregate load shed values by bus from solution and math dictionaries.

Returns Dict{String => (bus_id, original_kw, served_kw, shed_kw)} where key is bus name.
"""
function aggregate_load_shed_by_bus(solution::Dict{String,Any}, math::Dict{String,Any})
    # Build bus ID to name mapping
    bus_id_to_name = Dict{Int,String}()
    for (bus_id_str, bus) in math["bus"]
        bus_id = parse(Int, bus_id_str)
        bus_name = split(string(bus["source_id"]), ".")[end]
        bus_id_to_name[bus_id] = bus_name
    end

    # Aggregate load shed by bus
    # Dict: bus_name => (bus_id, original_kw, served_kw, shed_kw)
    bus_load_data = Dict{String,Tuple{Int,Float64,Float64,Float64}}()

    for (load_id_str, load) in math["load"]
        load_bus = load["load_bus"]
        bus_name = bus_id_to_name[load_bus]

        # Original demand
        original_demand = sum(load["pd"])

        # Get shed power from solution (pshed)
        shed = 0.0
        if haskey(solution, "load") && haskey(solution["load"], load_id_str)
            load_soln = solution["load"][load_id_str]
            if haskey(load_soln, "pshed")
                shed = sum(load_soln["pshed"])
            end
        end
        served = original_demand - shed

        # Aggregate to bus
        if haskey(bus_load_data, bus_name)
            _, orig, srv, shd = bus_load_data[bus_name]
            bus_load_data[bus_name] = (load_bus, orig + original_demand, srv + served, shd + shed)
        else
            bus_load_data[bus_name] = (load_bus, original_demand, served, shed)
        end
    end

    return bus_load_data
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
    hierarchical_tree_layout(g::SimpleGraph, root::Int)

Compute hierarchical tree layout with root at top.
Returns (x_positions, y_positions) vectors.
"""
function hierarchical_tree_layout(g::SimpleGraph, root::Int)
    n = nv(g)
    x_pos = zeros(Float64, n)
    y_pos = zeros(Float64, n)

    # BFS to determine level of each node
    visited = falses(n)
    level = zeros(Int, n)
    parent = zeros(Int, n)

    queue = [root]
    visited[root] = true
    level[root] = 0

    while !isempty(queue)
        node = popfirst!(queue)
        for neighbor in neighbors(g, node)
            if !visited[neighbor]
                visited[neighbor] = true
                level[neighbor] = level[node] + 1
                parent[neighbor] = node
                push!(queue, neighbor)
            end
        end
    end

    # Group nodes by level
    max_level = maximum(level)
    nodes_at_level = [Int[] for _ in 0:max_level]
    for i in 1:n
        push!(nodes_at_level[level[i]+1], i)
    end

    # Assign y positions (level determines y)
    for i in 1:n
        y_pos[i] = 1.0 - level[i] / max(max_level, 1)
    end

    # Assign x positions (spread nodes at each level)
    for (lv, nodes) in enumerate(nodes_at_level)
        n_nodes = length(nodes)
        for (idx, node) in enumerate(nodes)
            x_pos[node] = (idx - 0.5) / max(n_nodes, 1)
        end
    end

    return x_pos, y_pos
end

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
    height::Int=12
)
    println("Creating schematic network visualization...")

    # Build bus ID to name mapping
    bus_id_to_name = Dict{Int,String}()
    bus_name_to_id = Dict{String,Int}()
    for (bus_id_str, bus) in math["bus"]
        bus_id = parse(Int, bus_id_str)
        bus_name = split(string(bus["source_id"]), ".")[end]
        bus_id_to_name[bus_id] = bus_name
        bus_name_to_id[bus_name] = bus_id
    end

    # Get coordinates and scale to left 70% of figure (leaving room for legend)
    ieee13_coords_raw = ieee13_bus_coordinates()
    ieee13_coords = Dict{String,Tuple{Float64,Float64}}()
    for (name, (x, y)) in ieee13_coords_raw
        # Scale x from [0,1] to [0.02, 0.72] to leave room for legend on right
        ieee13_coords[name] = (0.02 + x * 0.70, y)
    end

    # Aggregate load shed by bus
    bus_load_data = aggregate_load_shed_by_bus(solution, math)

    # Track switch info: (f_bus_id, t_bus_id) => (is_closed, utilization_pct)
    switch_info = Dict{Tuple{Int,Int},Tuple{Bool,Float64}}()
    if haskey(math, "switch")
        for (switch_id, switch) in math["switch"]
            f_bus = switch["f_bus"]
            t_bus = switch["t_bus"]
            is_closed = true
            utilization_pct = 0.0

            if haskey(solution, "switch") && haskey(solution["switch"], switch_id)
                state = solution["switch"][switch_id]["state"]
                is_closed = all(state .> 0.0)

                # Calculate current utilization if switch is closed
                # Based on constraint: P² + Q² ≤ w * I_rating²
                # So utilization = √(P² + Q²) / (√w * I_rating)
                if is_closed
                    pf = get(solution["switch"][switch_id], "psw_fr", get(solution["switch"][switch_id], "pf", [0.0]))
                    qf = get(solution["switch"][switch_id], "qsw_fr", get(solution["switch"][switch_id], "qf", [0.0]))

                    # Get squared voltage (w) at from bus for each connection
                    w_vals = Dict{Int,Float64}()
                    if haskey(solution, "bus") && haskey(solution["bus"], string(f_bus))
                        bus_sol = solution["bus"][string(f_bus)]
                        if haskey(bus_sol, "w")
                            w_vals = bus_sol["w"]
                        elseif haskey(bus_sol, "vm")
                            # Convert vm to w (squared)
                            for (k, v) in bus_sol["vm"]
                                w_vals[k] = v^2
                            end
                        end
                    end

                    # Get current rating and connections
                    current_rating = get(switch, "current_rating", [Inf])
                    f_connections = get(switch, "f_connections", [1, 2, 3])

                    # Calculate utilization per phase using constraint formula
                    max_util = 0.0
                    for (idx, (p, q)) in enumerate(zip(pf, qf))
                        s_squared = p^2 + q^2  # S² in constraint units

                        # Get w for this connection
                        conn = idx <= length(f_connections) ? f_connections[idx] : idx
                        w = get(w_vals, conn, 1.0)

                        rating = idx <= length(current_rating) ? current_rating[idx] : current_rating[1]
                        if rating > 0 && rating < Inf && w > 1e-6
                            # From constraint: S² ≤ w * I² => util = S / (√w * I)
                            util = sqrt(s_squared) / (sqrt(w) * rating) * 100
                            max_util = max(max_util, util)
                        end
                    end
                    utilization_pct = max_util
                end
            end
            switch_info[(f_bus, t_bus)] = (is_closed, utilization_pct)
            switch_info[(t_bus, f_bus)] = (is_closed, utilization_pct)
        end
    end

    # Extract bus voltages for display
    bus_voltages = Dict{Int,Float64}()
    for (bus_id_str, bus) in math["bus"]
        bus_id = parse(Int, bus_id_str)
        v_pu = 1.0
        if haskey(solution, "bus") && haskey(solution["bus"], bus_id_str)
            bus_sol = solution["bus"][bus_id_str]
            if haskey(bus_sol, "w")
                w_vals = bus_sol["w"]
                v_pu = sqrt(Statistics.mean(values(w_vals)))
            elseif haskey(bus_sol, "vm")
                v_pu = Statistics.mean(values(bus_sol["vm"]))
            end
        end
        bus_voltages[bus_id] = v_pu
    end

    # Collect drawing elements in layers (back to front)
    line_elements = []    # Lines drawn first (back)
    node_elements = []    # Nodes drawn on top of lines
    label_elements = []   # Labels drawn last (front)

    # Line colors
    line_color = colorant"darkcyan"      # Regular branches
    switch_color = colorant"darkorange"  # Switches (different color to distinguish)
    node_radius = 0.012
    line_width = 1.425mm  # Reduced by 5% from 1.5mm

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

    # Draw switches with symbol (angled line for open, straight for closed)
    if haskey(math, "switch")
        for (switch_id, switch) in math["switch"]
            f_bus = switch["f_bus"]
            t_bus = switch["t_bus"]
            f_name = lowercase(get(bus_id_to_name, f_bus, ""))
            t_name = lowercase(get(bus_id_to_name, t_bus, ""))

            # Try to get coordinates - if not found, parse from switch name (e.g., "632633" -> "632", "633")
            x1, y1, x2, y2 = nothing, nothing, nothing, nothing

            if haskey(ieee13_coords, f_name)
                x1, y1 = ieee13_coords[f_name]
            end
            if haskey(ieee13_coords, t_name)
                x2, y2 = ieee13_coords[t_name]
            end

            # If coordinates not found, try parsing from switch name
            if (x1 === nothing || x2 === nothing) && haskey(switch, "name")
                sw_name = switch["name"]
                # Switch names like "632633" connect buses 632 and 633
                if length(sw_name) == 6 && all(isdigit, sw_name)
                    bus1 = lowercase(sw_name[1:3])
                    bus2 = lowercase(sw_name[4:6])
                    if haskey(ieee13_coords, bus1) && haskey(ieee13_coords, bus2)
                        x1, y1 = ieee13_coords[bus1]
                        x2, y2 = ieee13_coords[bus2]
                    end
                end
            end

            # Only draw if we have valid coordinates
            if x1 !== nothing && y1 !== nothing && x2 !== nothing && y2 !== nothing
                is_closed, utilization_pct = get(switch_info, (f_bus, t_bus), (true, 0.0))

                # Calculate midpoint and direction
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                len = sqrt(dx^2 + dy^2)
                if len > 0
                    dx, dy = dx / len, dy / len
                end

                # Perpendicular offset for label
                perp_x, perp_y = -dy, dx
                label_offset = 0.025

                # Switch symbol size
                sw_len = 0.04

                # Color switch by utilization (colorblind-friendly: blue < 50%, teal 50-80%, purple 80-100%)
                sw_color = switch_color
                if is_closed && utilization_pct > 0
                    if utilization_pct >= 80
                        sw_color = colorant"purple"
                    elseif utilization_pct >= 50
                        sw_color = colorant"darkcyan"
                    else
                        sw_color = colorant"steelblue"
                    end
                elseif !is_closed
                    sw_color = colorant"gray"
                end

                # Draw switch as dashed line with utilization-based color
                push!(line_elements, compose(context(),
                    line([(x1, y1), (x2, y2)]),
                    stroke(sw_color),
                    linewidth(line_width),
                    strokedash([3mm, 2mm])
                ))

                # Add utilization label near switch midpoint (offset perpendicular to line)
                if is_closed && utilization_pct > 0
                    util_label = "$(round(utilization_pct, digits=0))%"
                    label_x = mx + perp_x * label_offset
                    label_y = my + perp_y * label_offset
                    push!(label_elements, compose(context(),
                        text(label_x, label_y, util_label, hcenter, vcenter),
                        fontsize(7pt),
                        fill(sw_color),
                        font("sans-serif")
                    ))
                end
            end
        end
    end

    # Draw bus nodes as filled circles
    # Buses that need labels shifted left
    left_shift_buses_3 = Set(["632", "670", "675", "680", "652", "611"])  # 3 digit widths
    left_shift_buses_2 = Set(["671"])  # 2 digit widths
    left_shift_buses_1 = Set(["646", "634"])  # 1 digit width
    label_shift_3 = 0.04   # Approximately 3 digit widths
    label_shift_2 = 0.026  # Approximately 2 digit widths
    label_shift_1 = 0.013  # Approximately 1 digit width
    load_node_radius = 0.018  # Radius for load buses

    for (bus_id, bus_name) in bus_id_to_name
        name_lower = lowercase(bus_name)

        # Skip switch buses (names like 632645, 632633)
        if occursin("632", bus_name) && length(bus_name) > 3
            continue
        end

        if haskey(ieee13_coords, name_lower)
            x, y = ieee13_coords[name_lower]

            # Check if this bus has load data for coloring
            if haskey(bus_load_data, bus_name)
                _, original, served, shed = bus_load_data[bus_name]
                shed_pct = original > 0 ? (shed / original * 100) : 0.0

                # Blue shading based on shed percentage (colorblind-friendly)
                # Darker = less shed (more served), Lighter = more shed
                # Clamp shed_pct to [0, 100]
                shed_frac = clamp(shed_pct / 100.0, 0.0, 1.0)

                # Steel blue hue (~207), vary lightness: 0.30 (dark, 0% shed) to 0.80 (light, 100% shed)
                lightness = 0.30 + 0.5 * shed_frac
                node_color = HSL(207, 0.6, lightness)  # Steel blue with varying lightness

                # Draw larger colored circle for load bus
                push!(node_elements, compose(context(),
                    circle(x, y, load_node_radius),
                    fill(node_color),
                    stroke("black"),
                    linewidth(0.3mm)
                ))

                # Calculate label x position (shift left for specific buses)
                label_x = x
                if bus_name in left_shift_buses_3
                    label_x = x - label_shift_3
                elseif bus_name in left_shift_buses_2
                    label_x = x - label_shift_2
                elseif bus_name in left_shift_buses_1
                    label_x = x - label_shift_1
                end

                # Add bus label above node (official name + math ID)
                bus_label = "$bus_name ($bus_id)"
                push!(label_elements, compose(context(),
                    text(label_x, y - 0.035, bus_label, hcenter, vbottom),
                    fontsize(9pt),
                    fill("black")
                ))

                # Add load shed label below node with white background
                shed_label = "Shed: $(round(shed_pct, digits=1))%"
                push!(label_elements, compose(context(),
                    text(label_x, y + 0.04, shed_label, hcenter, vtop),
                    fontsize(7pt),
                    fill("black")
                ))
                # White background rectangle for shed label
                push!(label_elements, compose(context(),
                    Compose.rectangle(label_x - 0.055, y + 0.03, 0.11, 0.025),
                    fill("white"),
                    stroke(nothing)
                ))

                # Add voltage label below shed label
                v_pu = get(bus_voltages, bus_id, 1.0)
                # Color voltage by range (colorblind-friendly: magenta < 0.95 or > 1.05, gold warning, blue normal)
                v_color = if v_pu < 0.95 || v_pu > 1.05
                    colorant"darkmagenta"
                elseif v_pu < 0.97 || v_pu > 1.03
                    colorant"goldenrod"
                else
                    colorant"steelblue"
                end
                v_label = "V: $(round(v_pu, digits=3)) pu"
                push!(label_elements, compose(context(),
                    text(label_x, y + 0.07, v_label, hcenter, vtop),
                    fontsize(7pt),
                    fill(v_color)
                ))
                # White background rectangle for voltage label
                push!(label_elements, compose(context(),
                    Compose.rectangle(label_x - 0.05, y + 0.06, 0.10, 0.025),
                    fill("white"),
                    stroke(nothing)
                ))
            else
                # Non-load bus: grey circle
                push!(node_elements, compose(context(),
                    circle(x, y, node_radius),
                    fill("grey"),
                    stroke("darkgrey")
                ))

                # Calculate label x position (shift left for specific buses)
                label_x = x
                if bus_name in left_shift_buses_3
                    label_x = x - label_shift_3
                elseif bus_name in left_shift_buses_2
                    label_x = x - label_shift_2
                elseif bus_name in left_shift_buses_1
                    label_x = x - label_shift_1
                end

                # Add bus label above node (official name + math ID)
                bus_label = "$bus_name ($bus_id)"
                push!(label_elements, compose(context(),
                    text(label_x, y - 0.025, bus_label, hcenter, vbottom),
                    fontsize(9pt),
                    fill("black")
                ))

                # Add voltage label below node for non-load buses too
                v_pu = get(bus_voltages, bus_id, 1.0)
                # Color voltage by range (colorblind-friendly: magenta < 0.95 or > 1.05, gold warning, blue normal)
                v_color = if v_pu < 0.95 || v_pu > 1.05
                    colorant"darkmagenta"
                elseif v_pu < 0.97 || v_pu > 1.03
                    colorant"goldenrod"
                else
                    colorant"steelblue"
                end
                v_label = "V: $(round(v_pu, digits=3)) pu"
                push!(label_elements, compose(context(),
                    text(label_x, y + 0.025, v_label, hcenter, vtop),
                    fontsize(7pt),
                    fill(v_color)
                ))
                # White background rectangle for voltage label
                push!(label_elements, compose(context(),
                    Compose.rectangle(label_x - 0.05, y + 0.015, 0.10, 0.025),
                    fill("white"),
                    stroke(nothing)
                ))
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

    # Combine elements in layer order for Compose.jl (first = top, last = back)
    # So order is: labels (front/top) -> legend -> nodes -> lines -> legend_bg (back/bottom)
    all_elements = []
    append!(all_elements, label_elements)
    append!(all_elements, legend_elements)
    append!(all_elements, node_elements)
    append!(all_elements, line_elements)
    push!(all_elements, legend_bg)

    # Compose all elements and draw
    drawing = compose(context(), all_elements...)

    # Save to SVG
    draw(SVG(output_file, width*cm, height*cm), drawing)

    println("Schematic network visualization saved to $output_file")

    # Print summary
    print_load_shed_summary(solution, math)

    return output_file
end

"""
    print_load_shed_summary(solution::Dict{String,Any}, math::Dict{String,Any})

Print summary of load shedding results.
"""
function print_load_shed_summary(solution::Dict{String,Any}, math::Dict{String,Any})
    println("\n" * "="^60)
    println("LOAD SHEDDING SUMMARY BY BUS")
    println("="^60)

    bus_load_data = aggregate_load_shed_by_bus(solution, math)

    total_demand = 0.0
    total_served = 0.0
    total_shed = 0.0

    # Sort by bus name for consistent output
    sorted_buses = sort(collect(keys(bus_load_data)))

    for bus_name in sorted_buses
        bus_id, original, served, shed = bus_load_data[bus_name]
        total_demand += original
        total_served += served
        total_shed += shed

        shed_pct = original > 0 ? (shed / original * 100) : 0.0
        println("Bus $bus_name: Demand=$(round(original, digits=1)) kW, Served=$(round(served, digits=1)) kW, Shed=$(round(shed, digits=1)) kW ($(round(shed_pct, digits=1))%)")
    end

    println("-"^60)
    println("TOTAL: Demand=$(round(total_demand, digits=1)) kW, Served=$(round(total_served, digits=1)) kW, Shed=$(round(total_shed, digits=1)) kW")
    println("Overall served: $(round(total_served/max(total_demand,1e-6)*100, digits=1))%")
    println("="^60)
end
