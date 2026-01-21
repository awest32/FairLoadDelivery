using PowerModelsDistribution
using Graphs
using GraphPlot
using Colors
import Compose
import Compose: compose, context, line, circle, text, stroke, fill, linewidth, fontsize, draw, SVG, inch, cm, mm, pt, font, hcenter, vcenter, hleft, vtop, vbottom, strokedash
using Cairo

"""
Simple static SVG visualization of load shedding results
Saves directly as SVG for publication quality
"""
function visualize_network_svg(pm, result; output_file="network_topology.svg", width=12, height=8)
    
    println("Creating network visualization...")
    
    # Build graph and extract data
    g, bus_labels, bus_data = build_network_graph(pm, result)
    
    # Get positions using spring layout
    layout_func = spring_layout
    locs_x, locs_y = layout_func(g)
    
    # Normalize positions to [0, 1]
    locs_x = (locs_x .- minimum(locs_x)) ./ (maximum(locs_x) - minimum(locs_x))
    locs_y = (locs_y .- minimum(locs_y)) ./ (maximum(locs_y) - minimum(locs_y))
    
    # Get colors and sizes
    node_colors = get_node_colors(bus_data)
    node_sizes = get_node_sizes(bus_data)
    edge_colors, edge_widths = get_edge_properties(pm, result, g)
    
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
    print_summary(pm, result)
    
    return output_file
end

"""
Build graph structure from PowerModel
"""
function build_network_graph(pm, result)
    # Get bus mapping
    bus_ids = sort(collect(keys(_PMD.ref(pm, :bus))))
    bus_to_idx = Dict(bus_id => idx for (idx, bus_id) in enumerate(bus_ids))
    n_buses = length(bus_ids)
    
    # Create graph
    g = SimpleGraph(n_buses)
    
    # Add edges from branches
    for (i, branch) in _PMD.ref(pm, :branch)
        f_bus = branch["f_bus"]
        t_bus = branch["t_bus"]
        add_edge!(g, bus_to_idx[f_bus], bus_to_idx[t_bus])
    end
    
    # Add edges from switches
    if haskey(_PMD.ref(pm), :switch)
        for (i, switch) in _PMD.ref(pm, :switch)
            f_bus = switch["f_bus"]
            t_bus = switch["t_bus"]
            add_edge!(g, bus_to_idx[f_bus], bus_to_idx[t_bus])
        end
    end
    
    # Create bus labels and data
    bus_labels = []
    bus_data = []
    
    for bus_id in bus_ids
        bus = _PMD.ref(pm, :bus, bus_id)
        
        # Get voltage
        voltage = 1.0
        if haskey(result["solution"]["bus"], string(bus_id))
            vm = result["solution"]["bus"][string(bus_id)]["vm"]
            voltage = mean(vm)
        end
        
        # Check if generator bus
        is_generator = any(gen["gen_bus"] == bus_id for (i, gen) in _PMD.ref(pm, :gen))
        
        # Get load info
        load_demand = 0.0
        load_served = 0.0
        for (i, load) in _PMD.ref(pm, :load)
            if load["load_bus"] == bus_id
                load_demand += sum(load["pd"])
                if haskey(result["solution"]["load"], string(i))
                    load_served += sum(result["solution"]["load"][string(i)]["pd_bus"])
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
function get_edge_properties(pm, result, g)
    edge_colors = []
    edge_widths = []
    
    # Create edge index mapping
    bus_ids = sort(collect(keys(_PMD.ref(pm, :bus))))
    bus_to_idx = Dict(bus_id => idx for (idx, bus_id) in enumerate(bus_ids))
    
    # Track which edges we've added
    edge_data = Dict()
    
    # Add branch data
    for (i, branch) in _PMD.ref(pm, :branch)
        f_idx = bus_to_idx[branch["f_bus"]]
        t_idx = bus_to_idx[branch["t_bus"]]
        edge = minmax(f_idx, t_idx)
        
        power = 0.0
        if haskey(result["solution"]["branch"], string(i))
            pf = result["solution"]["branch"][string(i)]["pf"]
            power = sum(abs.(pf))
        end
        
        edge_data[edge] = Dict("power" => power, "status" => "closed", "type" => "branch")
    end
    
    # Add switch data
    if haskey(_PMD.ref(pm), :switch) && haskey(result["solution"], "switch")
        for (i, switch) in _PMD.ref(pm, :switch)
            f_idx = bus_to_idx[switch["f_bus"]]
            t_idx = bus_to_idx[switch["t_bus"]]
            edge = minmax(f_idx, t_idx)
            
            status = "closed"
            power = 0.0
            
            if haskey(result["solution"]["switch"], string(i))
                switch_state = result["solution"]["switch"][string(i)]["state"]
                status = all(switch_state .> 0.5) ? "closed" : "open"
                
                if status == "closed" && haskey(result["solution"]["switch"][string(i)], "pf")
                    pf = result["solution"]["switch"][string(i)]["pf"]
                    power = sum(abs.(pf))
                end
            end
            
            edge_data[edge] = Dict("power" => power, "status" => status, "type" => "switch")
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
function print_summary(pm, result)
    println("\n" * "="^60)
    println("LOAD SHEDDING SUMMARY")
    println("="^60)
    
    # Generator info
    total_gen_capacity = 0.0
    total_gen_output = 0.0
    
    for (i, gen) in _PMD.ref(pm, :gen)
        total_gen_capacity += sum(gen["pmax"])
        if haskey(result["solution"]["gen"], string(i))
            pg = result["solution"]["gen"][string(i)]["pg"]
            total_gen_output += sum(pg)
        end
    end
    
    println("Generation Capacity: $(round(total_gen_capacity, digits=1)) kW")
    println("Generation Output:   $(round(total_gen_output, digits=1)) kW ($(round(total_gen_output/total_gen_capacity*100, digits=1))%)")
    
    # Load info
    total_demand = 0.0
    total_served = 0.0
    
    for (i, load) in _PMD.ref(pm, :load)
        total_demand += sum(load["pd"])
        if haskey(result["solution"]["load"], string(i))
            total_served += sum(result["solution"]["load"][string(i)]["pd_bus"])
        end
    end
    
    total_shed = total_demand - total_served
    
    println("\nLoad Demand: $(round(total_demand, digits=1)) kW")
    println("Load Served: $(round(total_served, digits=1)) kW ($(round(total_served/total_demand*100, digits=1))%)")
    println("Load Shed:   $(round(total_shed, digits=1)) kW ($(round(total_shed/total_demand*100, digits=1))%)")
    
    # Switch info
    if haskey(_PMD.ref(pm), :switch)
        n_switches = length(_PMD.ref(pm, :switch))
        n_open = 0
        
        if haskey(result["solution"], "switch")
            for (i, switch) in result["solution"]["switch"]
                if all(switch["state"] .< 0.5)
                    n_open += 1
                end
            end
        end
        
        println("\nSwitches: $(n_switches - n_open) closed, $n_open open")
    end
    
    println("="^60)
end

"""
Example usage
"""
function example_usage()
    # Solve your MLD problem
    file = "ieee13_feeder.dss"
    eng = _PMD.parse_file(file)
    math = _PMD.transform_data_model(eng)
    
    # Set generation limits (80% for testing)
    total_load = sum(sum(load["pd"]) for (i, load) in math["load"])
    for (i, gen) in math["gen"]
        gen["pmax"] .= 0.8 * total_load / length(gen["pmax"])
    end
    
    # Solve
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    result = solve_mc_mld(math, ipopt)
    
    # Get PowerModel object
    pm = _PMD.instantiate_mc_model(math, _PMD.LPUBFDiagModel, build_mc_mld)
    
    # Create visualization
    visualize_network_svg(pm, result, output_file="mld_results.svg", width=14, height=10)
    
    return pm, result
end

# Run example
# pm, result = example_usage()

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

        # Served power from solution
        served = 0.0
        if haskey(solution, "load") && haskey(solution["load"], load_id_str)
            load_soln = solution["load"][load_id_str]
            # Check for pd or pd_bus (different solution formats)
            if haskey(load_soln, "pd")
                served = sum(load_soln["pd"])
            elseif haskey(load_soln, "pd_bus")
                served = sum(load_soln["pd_bus"])
            end
        end

        shed = original_demand - served

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
        "650"       => (0.50, 0.08),
        "sourcebus" => (0.50, 0.08),
        "rg60"      => (0.50, 0.08),

        # Upper horizontal line: 646 -- 645 -- 632 -- 633 -- 634
        "646"       => (0.10, 0.30),
        "645"       => (0.30, 0.30),
        "632"       => (0.50, 0.30),
        "670"       => (0.50, 0.50),  # Between 632 and 671
        "633"       => (0.70, 0.30),
        "634"       => (0.90, 0.30),

        # Lower horizontal line: 611 -- 684 -- 671 -- 692 -- 675
        "611"       => (0.10, 0.70),
        "684"       => (0.30, 0.70),
        "671"       => (0.50, 0.70),
        "692"       => (0.70, 0.70),
        "675"       => (0.90, 0.70),

        # Vertical branches down
        "652"       => (0.30, 0.90),  # Below 684
        "680"       => (0.50, 0.90),  # Below 671
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
    width::Int=14,
    height::Int=10
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

    # Get coordinates
    ieee13_coords = ieee13_bus_coordinates()

    # Aggregate load shed by bus
    bus_load_data = aggregate_load_shed_by_bus(solution, math)

    # Track switch info: (f_bus_id, t_bus_id) => is_closed
    switch_info = Dict{Tuple{Int,Int},Bool}()
    if haskey(math, "switch")
        for (switch_id, switch) in math["switch"]
            f_bus = switch["f_bus"]
            t_bus = switch["t_bus"]
            is_closed = true
            if haskey(solution, "switch") && haskey(solution["switch"], switch_id)
                state = solution["switch"][switch_id]["state"]
                is_closed = all(state .> 0.5)
            end
            switch_info[(f_bus, t_bus)] = is_closed
            switch_info[(t_bus, f_bus)] = is_closed
        end
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
                is_closed = get(switch_info, (f_bus, t_bus), true)

                # Calculate midpoint and direction
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                dx, dy = x2 - x1, y2 - y1
                len = sqrt(dx^2 + dy^2)
                if len > 0
                    dx, dy = dx / len, dy / len
                end

                # Switch symbol size
                sw_len = 0.04

                # Draw switch as dashed orange line (same for open or closed)
                push!(line_elements, compose(context(),
                    line([(x1, y1), (x2, y2)]),
                    stroke(switch_color),
                    linewidth(line_width),
                    strokedash([3mm, 2mm])
                ))
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
    load_node_radius = 0.025  # Larger radius for load buses

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

                # Forest green with shading based on shed percentage
                # Darker = less shed (more served), Lighter = more shed
                # Clamp shed_pct to [0, 100]
                shed_frac = clamp(shed_pct / 100.0, 0.0, 1.0)

                # Forest green hue (~120), vary lightness: 0.25 (dark, 0% shed) to 0.75 (light, 100% shed)
                lightness = 0.25 + 0.5 * shed_frac
                node_color = HSL(120, 0.6, lightness)  # Forest green with varying lightness

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

                # Add bus label above node
                push!(label_elements, compose(context(),
                    text(label_x, y - 0.05, bus_name, hcenter, vbottom),
                    fontsize(10pt),
                    fill("black")
                ))

                # Add load shed label below node with white background
                shed_label = "Shed: $(round(shed_pct, digits=1))%"
                # Text first (will be on top in Compose.jl's reverse ordering)
                push!(label_elements, compose(context(),
                    text(label_x, y + 0.06, shed_label, hcenter, vtop),
                    fontsize(8pt),
                    fill("black")
                ))
                # White background rectangle for label (drawn behind text)
                push!(label_elements, compose(context(),
                    Compose.rectangle(label_x - 0.06, y + 0.045, 0.12, 0.035),
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

                # Add bus label above node
                push!(label_elements, compose(context(),
                    text(label_x, y - 0.04, bus_name, hcenter, vbottom),
                    fontsize(10pt),
                    fill("black")
                ))
            end
        end
    end

    # Combine elements in layer order for Compose.jl (first = top, last = back)
    # So order is: labels (front/top) -> nodes -> lines (back/bottom)
    all_elements = []
    append!(all_elements, label_elements)
    append!(all_elements, node_elements)
    append!(all_elements, line_elements)

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