using Graphs
using GraphPlot
using Colors
using Statistics
import Compose
import Compose: compose, context, line, circle, text, stroke, fill, linewidth, fontsize, draw, SVG, inch, cm, mm, pt, font, hcenter, vcenter, hleft, vtop, vbottom, strokedash
using Cairo

# Fairness function styling constants (colorblind-friendly palette)
const FAIR_FUNC_COLORS = Dict(
    "proportional" => RGB(0.30, 0.65, 0.40),   # dark green
    "efficiency"   => RGB(0.35, 0.55, 0.80),   # dark blue
    "min_max"      => RGB(0.85, 0.35, 0.35),   # dark red/coral
    "equality_min" => RGB(0.90, 0.65, 0.25),   # dark amber
    "jain"         => RGB(0.55, 0.40, 0.75)    # dark purple
)
const FAIR_FUNC_LABELS = Dict("proportional"=>"Proportional", "efficiency"=>"Efficiency", "min_max"=>"Min-Max", "equality_min"=>"Equality Min", "jain"=>"Jain's Index")
const FAIR_FUNC_MARKERS = Dict("proportional"=>:square, "efficiency"=>:circle, "min_max"=>:star5, "equality_min"=>:diamond, "jain"=>:utriangle)
const FAIR_FUNC_LINESTYLES = Dict("proportional"=>:dash, "efficiency"=>:solid, "min_max"=>:dashdot, "equality_min"=>:dot, "jain"=>:dashdotdot)

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
Map the load IDs to their engineering names from the math dictionary.
"""
function build_load_name_map(math::Dict{String,Any})
    load_id_to_name = Dict{Int,String}()
    for (lid, load) in math["load"]
        load_id_to_name[parse(Int, lid)] = load["name"]
    end
    return load_id_to_name
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
                    if w_vals[id] < -1e-6
                        @warn "Negative w value at bus $bus_id_str phase $c: $(w_vals[id])"
                        voltages[c] = 0.0
                    elseif w_vals[id] < 0.0
                        voltages[c] = 0.0
                    else
                        voltages[c] = sqrt(w_vals[id])
                    end
                end
            elseif haskey(bus_sol, "vm")
                vm_vals = bus_sol["vm"]
                for (id,c) in enumerate(bus["terminals"])
                    if vm_vals[id] < -1e-6
                        @warn "Negative vm value at bus $bus_id_str phase $c: $(vm_vals[id])"
                        voltages[c] = 0.0
                    elseif vm_vals[id] < 0.0
                        voltages[c] = 0.0
                    else
                        voltages[c] = vm_vals[id]
                    end
                end
            elseif haskey(bus_sol, "vr")
                vr_vals = bus_sol["vr"]
                vi_vals = bus_sol["vi"]
                vmag_vals = vr_vals .^ 2 .+ vi_vals .^ 2
                for (id,c) in enumerate(bus["terminals"])
                    if vmag_vals[id] < -1e-6
                        @warn "Negative vmag² value at bus $bus_id_str phase $c: $(vmag_vals[id])"
                        voltages[c] = 0.0
                    elseif vmag_vals[id] < 0.0
                        voltages[c] = 0.0
                    else
                        voltages[c] = sqrt(vmag_vals[id])
                    end
                end
            end
        end
        bus_voltages[bus_id] = voltages
    end

    return bus_voltages
end

"""
    aggregate_load_shed_by_bus_per_phase(solution, math; original_math=nothing)

Aggregate load shed values by bus from solution and math dictionaries.

Original demand is taken from `original_math` when provided, since `math` may be
a post-rounding copy where de-energized loads have pd/qd zeroed out.
Loads missing from the solution are treated as fully shed.
`something(a, b)` is a Julia built-in that returns the first non-nothing argument.

Returns:
- bus_load_phases_p: Dict{String => Dict{Int => (original_p, shed_p)}} for active power
- bus_load_phases_q: Dict{String => Dict{Int => (original_q, shed_q)}} for reactive power
- bus_terminals: Dict{String => Vector{Int}} mapping bus name to phase terminals
"""
function aggregate_load_shed_by_bus_per_phase(solution::Dict{String,Any}, math::Dict{String,Any}; original_math::Union{Dict{String,Any},Nothing}=nothing)
    # something(a, b): Julia built-in — returns `a` if not nothing, else `b`
    orig_math = something(original_math, math)
    # Build bus ID to name mapping
    bus_id_to_name = build_bus_name_maps(math)
    # Aggregate load shed by bus for both P and Q
    # Dict: bus_name => phase => (original, shed)
    bus_load_phases_p = Dict{String,Dict{Int, Tuple{Float64,Float64}}}()
    bus_load_phases_q = Dict{String,Dict{Int, Tuple{Float64,Float64}}}()
    bus_terminals = Dict{String,Vector{Int}}()

    for (load_id_str, load) in math["load"]
        load_bus = load["load_bus"]
        bus_name = bus_id_to_name[load_bus]
        original_pd = zeros(3)
        original_qd = zeros(3)
        pshed = zeros(3)
        qshed = zeros(3)

        # Get original demand from original_math (not the rounded math)
        orig_load = orig_math["load"][load_id_str]

        for (id, c) in enumerate(load["connections"])
            # Original demand from the pre-rounding math
            original_pd[c] = orig_load["pd"][id]
            original_qd[c] = orig_load["qd"][id]

            # Merge terminals rather than overwrite (handles split buses like 634a/b/c)
            if haskey(bus_terminals, bus_name)
                if !(c in bus_terminals[bus_name])
                    push!(bus_terminals[bus_name], c)
                end
            else
                bus_terminals[bus_name] = [c]
            end

            # Get per-phase served load from solution, shed = original - served
            if haskey(solution, "load") && haskey(solution["load"], load_id_str)
                load_soln = solution["load"][load_id_str]
                if haskey(load_soln, "pd")
                    pd_served = load_soln["pd"]
                    served_val = isa(pd_served, AbstractArray) ? pd_served[id] : pd_served
                    pshed[c] = original_pd[c] - served_val
                end
                if haskey(load_soln, "qd")
                    qd_served = load_soln["qd"]
                    served_val = isa(qd_served, AbstractArray) ? qd_served[id] : qd_served
                    qshed[c] = original_qd[c] - served_val
                end
            else
                # Load not in solution (de-energized) => fully shed
                pshed[c] = original_pd[c]
                qshed[c] = original_qd[c]
            end

            # Map active power to bus
            if haskey(bus_load_phases_p, bus_name)
                phase_data = bus_load_phases_p[bus_name]
                if haskey(phase_data, c)
                    orig, shd = phase_data[c]
                    phase_data[c] = (orig + original_pd[c], shd + pshed[c])
                else
                    phase_data[c] = (original_pd[c], pshed[c])
                end
            else
                phase_data = Dict{Int,Tuple{Float64,Float64}}()
                phase_data[c] = (original_pd[c], pshed[c])
                bus_load_phases_p[bus_name] = phase_data
            end

            # Map reactive power to bus
            if haskey(bus_load_phases_q, bus_name)
                phase_data = bus_load_phases_q[bus_name]
                if haskey(phase_data, c)
                    orig, shd = phase_data[c]
                    phase_data[c] = (orig + original_qd[c], shd + qshed[c])
                else
                    phase_data[c] = (original_qd[c], qshed[c])
                end
            else
                phase_data = Dict{Int,Tuple{Float64,Float64}}()
                phase_data[c] = (original_qd[c], qshed[c])
                bus_load_phases_q[bus_name] = phase_data
            end
        end
    end

    return bus_load_phases_p, bus_load_phases_q, bus_terminals
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
    plot_network_load_shed(solution, math; output_file, layout, width, height, ac_flag, original_math)

Plot IEEE 13 bus one-line diagram with final load shed values per bus.

Original demand is taken from `original_math` when provided, since `math` may be
a post-rounding copy where de-energized loads have pd/qd zeroed out.
Total load on the network always reflects the true original demand, and
shed = original_demand - served.

# Arguments
- `solution`: Solution dictionary from MLD solve (e.g., best_mld["solution"])
- `math`: Math dictionary containing network topology and bus mapping
- `original_math`: (optional) Pre-rounding math dictionary with true original load demands
- `output_file`: Path to save the SVG output
- `layout`: Layout algorithm - :ieee13 for standard IEEE 13 bus diagram
- `width`, `height`: Dimensions in cm
- `ac_flag`: Whether solution is from an AC power flow solve

# Example
```julia
plot_network_load_shed(best_mld["solution"], math_out[best_set];
    output_file="results/network_load_shed.svg",
    original_math=math)
```
"""
function plot_network_load_shed(
    solution::Dict{String,Any},
    math::Dict{String,Any};
    output_file::String="network_load_shed.svg",
    layout::Symbol=:ieee13,
    width::Int=18,
    height::Int=12,
    ac_flag::Bool=false,
    original_math::Union{Dict{String,Any},Nothing}=nothing
)
    println("Creating schematic network visualization...")
    # something(a, b): Julia built-in — returns `a` if not nothing, else `b`
    orig_math = something(original_math, math)

    # --- Data extraction via helpers ---
    bus_id_to_name = build_bus_name_maps(math)
    bus_voltages = get_bus_voltage_per_phase(solution, math)
    bus_load_phases_p, bus_load_phases_q, bus_terminals = aggregate_load_shed_by_bus_per_phase(solution, math; original_math=orig_math)
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

        is_load_bus = haskey(bus_load_phases_p, bus_name)
        is_gen_bus = haskey(bus_gen_data, bus_id)

        # Draw node circle
        if is_load_bus
            phases_data_p = bus_load_phases_p[bus_name]
            total_orig = sum(orig for (orig, _) in values(phases_data_p))
            total_shed = sum(shed for (_, shed) in values(phases_data_p))
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
            phases_data_p = bus_load_phases_p[bus_name]
            phases_data_q = get(bus_load_phases_q, bus_name, Dict{Int,Tuple{Float64,Float64}}())
            terminals = get(bus_terminals, bus_name, Int[])
            for phase in sort(terminals)
                p_pct_str = "N/A"
                q_pct_str = "N/A"
                if haskey(phases_data_p, phase)
                    orig_p, shed_p = phases_data_p[phase]
                    p_pct = orig_p > 0 ? round(shed_p / orig_p * 100, digits=0) : 0.0
                    p_pct_str = "$(Int(p_pct))%"
                end
                if haskey(phases_data_q, phase)
                    orig_q, shed_q = phases_data_q[phase]
                    q_pct = orig_q > 0 ? round(shed_q / orig_q * 100, digits=0) : 0.0
                    q_pct_str = "$(Int(q_pct))%"
                end
                shed_label = "S$phase: $(p_pct_str)"
                push!(bg_elements, compose(context(),
                    Compose.rectangle(label_x - 0.07, current_y - 0.004, 0.14, 0.016),
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
        fontsize(9pt),
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
            fontsize(9pt),
            fill("black")
        ))
    end

    # --- Voltage Legend ---
    volt_y = leg_y + (length(util_colors) + 1) * leg_spacing
    push!(legend_elements, compose(context(),
        text(leg_x, volt_y, "Bus Voltage:", hleft, vtop),
        fontsize(9pt),
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
            fontsize(9pt),
            fill(color)
        ))
        # Label
        push!(legend_elements, compose(context(),
            text(leg_x + 0.025, y_pos, label, hleft, vcenter),
            fontsize(9pt),
            fill("black")
        ))
    end

    # --- Load Shed Legend ---
    shed_y = volt_y + (length(volt_colors) + 1) * leg_spacing
    push!(legend_elements, compose(context(),
        text(leg_x, shed_y, "Load Bus (% shed):", hleft, vtop),
        fontsize(9pt),
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
            fontsize(9pt),
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
    print_load_shed_summary(solution, math; ac_flag=ac_flag, csv_file=csv_file, original_math=orig_math)

    return output_file
end

"""
    print_load_shed_summary(solution, math; ac_flag=false, csv_file="", original_math=nothing)

Print summary of load shedding results and optionally save switch power flow data to CSV.
Uses `original_math` for true demand when provided (see `aggregate_load_shed_by_bus_per_phase`).
"""
function print_load_shed_summary(solution::Dict{String,Any}, math::Dict{String,Any}; ac_flag::Bool=false, csv_file::String="", original_math::Union{Dict{String,Any},Nothing}=nothing)
    # something(a, b): Julia built-in — returns `a` if not nothing, else `b`
    orig_math = something(original_math, math)
    println("\n" * "="^60)
    println("LOAD SHEDDING SUMMARY BY BUS")
    println("="^60)

    bus_load_phases_p, bus_load_phases_q, _ = aggregate_load_shed_by_bus_per_phase(solution, math; original_math=orig_math)

    total_demand = 0.0
    total_served = 0.0

    for bus_name in sort(collect(keys(bus_load_phases_p)))
        phases_data = bus_load_phases_p[bus_name]
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
                orig_load = orig_math["load"][lid_str]
                push!(block_buses, get(bus_id_to_name, load["load_bus"], string(load["load_bus"])))
                for (id, c) in enumerate(load["connections"])
                    orig = orig_load["pd"][id]
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

"""
    extract_voltage_by_bus_name(solution, math)

Extract per-phase voltage for each bus, grouped by engineering bus name.
Returns Dict{String, Dict{Int, Float64}} mapping bus_name => phase => voltage_pu.
"""
function extract_voltage_by_bus_name(solution::Dict, math::Dict)
    bus_voltages = get_bus_voltage_per_phase(solution, math)
    bus_id_to_name = build_bus_name_maps(math)

    voltage_by_name = Dict{String, Dict{Int, Float64}}()

    for (bus_id_str, bus) in math["bus"]
        bus_id = parse(Int, bus_id_str)
        bus_name = bus_id_to_name[bus_id]

        # Skip virtual switch buses (e.g. "632633", "671692") — they duplicate real bus voltages
        if length(bus_name) > 3 && all(isdigit, bus_name)
            continue
        end

        bus_terms = filter(t -> t <= 3, bus["terminals"])
        voltages = bus_voltages[bus_id]

        if !haskey(voltage_by_name, bus_name)
            voltage_by_name[bus_name] = Dict{Int, Float64}()
        end

        for phase in bus_terms
            voltage_by_name[bus_name][phase] = voltages[phase]
        end
    end

    return voltage_by_name
end

"""
    extract_per_bus_loadshed(solution, math; original_math=nothing)

Extract total load shed per bus (summed across phases) as percentages of original demand.
Original demand is taken from `original_math` when provided, since `math` may be
a post-rounding copy where de-energized loads have pd/qd zeroed out.
Loads missing from the solution are treated as fully shed (served = 0).

Returns `(bus_names::Vector{String}, pshed_pct::Vector{Float64}, qshed_pct::Vector{Float64})`.
"""
function extract_per_bus_loadshed(solution::Dict{String,Any}, math::Dict{String,Any}; original_math::Union{Dict{String,Any},Nothing}=nothing)
    # something(a, b): Julia built-in — returns `a` if not nothing, else `b`
    orig_math = something(original_math, math)
    bus_id_to_name = build_bus_name_maps(math)

    bus_pd_orig = Dict{String, Float64}()
    bus_qd_orig = Dict{String, Float64}()
    bus_pshed = Dict{String, Float64}()
    bus_qshed = Dict{String, Float64}()

    for (lid, load) in math["load"]
        bus_name = bus_id_to_name[load["load_bus"]]

        # Original demand from pre-rounding math
        orig_load = orig_math["load"][lid]
        pd_orig = sum(orig_load["pd"])
        qd_orig = sum(orig_load["qd"])

        # Served from solution; loads not in solution are fully shed (served = 0)
        pd_served = 0.0
        qd_served = 0.0
        if haskey(solution, "load") && haskey(solution["load"], lid)
            load_sol = solution["load"][lid]
            if haskey(load_sol, "pd")
                pd_s = load_sol["pd"]
                pd_served = isa(pd_s, AbstractArray) ? sum(pd_s) : pd_s
            end
            if haskey(load_sol, "qd")
                qd_s = load_sol["qd"]
                qd_served = isa(qd_s, AbstractArray) ? sum(qd_s) : qd_s
            end
        end

        pshed = pd_orig - pd_served
        qshed = qd_orig - qd_served

        bus_pd_orig[bus_name] = get(bus_pd_orig, bus_name, 0.0) + pd_orig
        bus_qd_orig[bus_name] = get(bus_qd_orig, bus_name, 0.0) + qd_orig
        bus_pshed[bus_name] = get(bus_pshed, bus_name, 0.0) + pshed
        bus_qshed[bus_name] = get(bus_qshed, bus_name, 0.0) + qshed
    end

    bus_names = sort(collect(keys(bus_pd_orig)))
    pshed_pct = Float64[]
    qshed_pct = Float64[]

    for bn in bus_names
        push!(pshed_pct, bus_pd_orig[bn] > 0 ? (bus_pshed[bn] / bus_pd_orig[bn]) * 100 : 0.0)
        push!(qshed_pct, bus_qd_orig[bn] > 0 ? (bus_qshed[bn] / bus_qd_orig[bn]) * 100 : 0.0)
    end

    return bus_names, pshed_pct, qshed_pct
end

"""
    plot_voltage_per_bus_comparison(voltage_data_per_func, save_path; title="", target_buses=String[])

Scatter plot comparing per-phase bus voltages across fairness functions.

# Arguments
- `voltage_data_per_func::Dict{String, Dict{String, Dict{Int, Float64}}}` — fair_func => bus_name => phase => voltage
- `save_path::String` — file path to save the SVG
- `title::String` — plot title (auto-generated if empty)
- `target_buses::Vector{String}` — subset of buses to plot (all load buses if empty)
"""
function plot_voltage_per_bus_comparison(
    voltage_data_per_func::Dict{String, Dict{String, Dict{Int, Float64}}},
    save_path::String;
    title::String="Bus Voltage Per Phase by Fairness Function",
    target_buses::Vector{String}=String[]
)
    phase_labels = Dict(1 => "A", 2 => "B", 3 => "C")
    phases = [1, 2, 3]

    # Determine target buses: use provided list or collect all buses with data
    if isempty(target_buses)
        all_buses = Set{String}()
        for (_, bus_volt) in voltage_data_per_func
            union!(all_buses, keys(bus_volt))
        end
        target_buses = sort(collect(all_buses))
    end

    # Build x-axis: bus groups with phase positions, gaps between buses
    x_labels = String[]
    x_positions = Float64[]
    pos = 1.0

    for bus in target_buses
        for phase in phases
            push!(x_labels, "$(bus)-$(phase_labels[phase])")
            push!(x_positions, pos)
            pos += 1.0
        end
        pos += 1.0  # gap between bus groups
    end

    active_funcs = [ff for ff in keys(voltage_data_per_func) if !isempty(voltage_data_per_func[ff])]
    sort!(active_funcs)
    n_funcs = length(active_funcs)

    if n_funcs == 0
        @error "No voltage data available"
    end

    offset_width = 0.25 / max(n_funcs, 1)
    offsets = n_funcs == 1 ? [0.0] : collect(range(-(n_funcs-1)/2 * offset_width, (n_funcs-1)/2 * offset_width, length=n_funcs))

    p = Plots.plot(
        xlabel = "Bus - Phase",
        ylabel = "Voltage (pu)",
        title = title,
        legend = :outertopright,
        xticks = (x_positions, x_labels),
        xrotation = 45,
        size = (1400, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 20Plots.mm,
        top_margin = 5Plots.mm,
        right_margin = 15Plots.mm,
        ylims = (0.80, 1.10),
        grid = :y
    )

    # Voltage limit lines
    Plots.hline!(p, [0.95], color=:red, linestyle=:dash, linewidth=2, label="V_min (0.95 pu)")
    Plots.hline!(p, [1.05], color=:red, linestyle=:dash, linewidth=2, label="V_max (1.05 pu)")

    for (fi, fair_func) in enumerate(active_funcs)
        bus_volt = voltage_data_per_func[fair_func]

        plot_x = Float64[]
        plot_y = Float64[]

        for (bi, bus) in enumerate(target_buses)
            bus_data = bus_volt[bus]
            for (pi, phase) in enumerate(phases)
                push!(plot_x, x_positions[(bi-1)*3 + pi] + offsets[fi])
                push!(plot_y, bus_data[phase])
            end
        end

        Plots.scatter!(p, plot_x, plot_y,
            label = FAIR_FUNC_LABELS[fair_func],
            marker = FAIR_FUNC_MARKERS[fair_func],
            markersize = 8,
            color = FAIR_FUNC_COLORS[fair_func]
        )
    end

    # Vertical separators between bus groups
    for i in 1:(length(target_buses)-1)
        sep_x = x_positions[i*3] + 0.5
        Plots.vline!(p, [sep_x], color=:lightgray, linestyle=:dot, linewidth=0.5, label=false)
    end

    Plots.savefig(p, save_path)
    println("  Saved voltage comparison: $save_path")
    return p
end

"""
    plot_loadshed_per_bus_comparison(loadshed_data_per_func, save_path; title="")

Grouped bar chart comparing per-bus load shed (%) across fairness functions.
Each fairness function group has two bars: P shed % (full opacity) and Q shed % (reduced opacity).

# Arguments
- `loadshed_data_per_func::Dict{String, Tuple{Vector{String}, Vector{Float64}, Vector{Float64}}}` — fair_func => (bus_names, pshed_pct, qshed_pct)
- `save_path::String` — file path to save the SVG
- `title::String` — plot title
"""
function plot_loadshed_per_bus_comparison(
    loadshed_data_per_func::Dict{String, Tuple{Vector{String}, Vector{Float64}, Vector{Float64}}},
    save_path::String;
    title::String="Load Shed Per Bus by Fairness Function"
)
    # Collect union of all bus names in order
    all_buses = String[]
    for (_, (bus_names, _, _)) in loadshed_data_per_func
        for bn in bus_names
            if !(bn in all_buses)
                push!(all_buses, bn)
            end
        end
    end
    sort!(all_buses)

    if isempty(all_buses)
        @error "No loadshed data available"
    end

    active_funcs = sort(collect(keys(loadshed_data_per_func)))
    n_funcs = length(active_funcs)
    n_buses = length(all_buses)

    group_width = 0.8
    bar_w = group_width / n_funcs
    x_positions = collect(1:n_buses)

    p = Plots.plot(
        xlabel = "Bus",
        ylabel = "Active Power Load Shed (%)",
        title = title,
        legend = :top,
        xticks = (x_positions, all_buses),
        xrotation = 45,
        size = (1400, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm,
        right_margin = 20Plots.mm,
        legendfontsize = 12,
        tickfontsize = 12,
        guidefontsize = 12,
        titlefontsize = 14
    )

    for (fi, fair_func) in enumerate(active_funcs)
        bus_names, pshed_pct, qshed_pct = loadshed_data_per_func[fair_func]

        # Build lookup
        name_to_idx = Dict(bn => i for (i, bn) in enumerate(bus_names))

        p_vals = [pshed_pct[name_to_idx[bn]] for bn in all_buses]

        # Offset: center the bars
        offset = (fi - (n_funcs + 1) / 2) * bar_w

        fc = FAIR_FUNC_COLORS[fair_func]
        fm = FAIR_FUNC_MARKERS[fair_func]
        fls = FAIR_FUNC_LINESTYLES[fair_func]

        bar_x = x_positions .+ offset
        Plots.bar!(p, bar_x, p_vals,
            bar_width = bar_w * 0.9,
            label = false,
            color = fc,
            linecolor = :gray30,
            linestyle = fls,
            linewidth = 1.5
        )

        # Overlay markers on bars for pattern differentiation
        for (xi, vi) in zip(bar_x, p_vals)
            if vi > 2.0
                n_markers = max(1, round(Int, vi / 25))
                for mi in 1:n_markers
                    y_pos = vi * mi / (n_markers + 1)
                    Plots.scatter!(p, [xi], [y_pos],
                        marker = fm, markersize = 5,
                        color = :gray40, markerstrokecolor = :gray40,
                        label = false
                    )
                end
            end
        end
    end

    # Add marker-based legend entries (symbols, not bar colors)
    for fair_func in active_funcs
        fl = FAIR_FUNC_LABELS[fair_func]
        fm = FAIR_FUNC_MARKERS[fair_func]
        fc = FAIR_FUNC_COLORS[fair_func]
        Plots.scatter!(p, [NaN], [NaN],
            marker = fm, markersize = 8,
            color = fc, markerstrokecolor = :gray30,
            label = fl
        )
    end

    Plots.savefig(p, save_path)
    println("  Saved loadshed comparison: $save_path")
    return p
end

"""
    plot_original_load_bars(case, math, save_dir)

Create bar plots for original pd and qd per load, plus a combined P+Q plot.
"""
function plot_original_load_bars(case::String, math::Dict, save_dir::String)
    load_ids, load_names, pd_original, qd_original = extract_original_load(math)
    x_positions = collect(1:length(load_ids))

    # Active power plot
    p_pd = Plots.bar(
        x_positions, pd_original,
        xlabel = "Load",
        ylabel = "Active Power (kW)",
        title = "Original Active Power Demand: $case",
        legend = false,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm,
        color = :blue
    )
    Plots.savefig(p_pd, joinpath(save_dir, "original_pd_$case.svg"))

    # Reactive power plot
    p_qd = Plots.bar(
        x_positions, qd_original,
        xlabel = "Load",
        ylabel = "Reactive Power (kVAR)",
        title = "Original Reactive Power Demand: $case",
        legend = false,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm,
        color = :red
    )
    Plots.savefig(p_qd, joinpath(save_dir, "original_qd_$case.svg"))

    # Combined P and Q plot using manual bar! with offsets (avoids StatsPlots dependency)
    n_groups = 2
    bar_width = 0.35
    offsets = [-bar_width/2, bar_width/2]

    p_combined = Plots.plot(
        xlabel = "Load",
        ylabel = "Power",
        title = "Original Demand (P and Q): $case",
        legend = :outertopright,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm
    )
    Plots.bar!(p_combined, x_positions .+ offsets[1], pd_original,
        bar_width = bar_width * 0.9,
        label = "Active Power (kW)",
        color = :blue
    )
    Plots.bar!(p_combined, x_positions .+ offsets[2], qd_original,
        bar_width = bar_width * 0.9,
        label = "Reactive Power (kVAR)",
        color = :red
    )
    Plots.savefig(p_combined, joinpath(save_dir, "original_demand_$case.svg"))

    return p_pd, p_qd, p_combined
end

"""
    plot_original_network_load(case, math, save_dir)

Create network visualization showing original load distribution (no shedding).
"""
function plot_original_network_load(case::String, math::Dict, save_dir::String)
    mock_solution = Dict{String, Any}()

    mock_solution["load"] = Dict{String, Any}()
    for (lid, load_data) in math["load"]
        mock_solution["load"][lid] = Dict(
            "pd" => load_data["pd"],
            "qd" => load_data["qd"],
            "pshed" => zeros(length(load_data["pd"])),
            "qshed" => zeros(length(load_data["qd"])),
            "status" => 1.0
        )
    end

    mock_solution["bus"] = Dict{String, Any}()
    for (bid, bus_data) in math["bus"]
        n_phases = length(bus_data["terminals"])
        mock_solution["bus"][bid] = Dict(
            "w" => ones(n_phases),
            "status" => 1.0
        )
    end

    mock_solution["switch"] = Dict{String, Any}()
    if haskey(math, "switch")
        for (sid, switch_data) in math["switch"]
            n_phases = length(switch_data["f_connections"])
            mock_solution["switch"][sid] = Dict(
                "state" => 1.0,
                "pf" => zeros(n_phases),
                "qf" => zeros(n_phases)
            )
        end
    end

    mock_solution["block"] = Dict{String, Any}()
    if haskey(math, "block")
        for (bid, _) in math["block"]
            mock_solution["block"][bid] = Dict("status" => 1.0)
        end
    end

    plot_filename = joinpath(save_dir, "network_original_$case.svg")
    plot_network_load_shed(mock_solution, math; output_file=plot_filename)
end

"""
    create_grouped_bar_chart(case, per_load_data, data_type, save_dir; fair_funcs=String[])

Create grouped bar chart for load shed or load served across fairness functions.
`data_type`: `:pshed`, `:pd_served`, `:qshed`, or `:qd_served`.
`fair_funcs`: ordered list of fairness function keys to plot.
"""
function create_grouped_bar_chart(case::String, per_load_data::Dict, data_type::Symbol, save_dir::String; fair_funcs::Vector{String}=String[])
    all_load_ids = Set{Int}()
    id_to_name = Dict{Int,String}()
    for (_, data) in per_load_data
        if !isempty(data[:load_ids])
            union!(all_load_ids, data[:load_ids])
            for (lid, lname) in zip(data[:load_ids], data[:load_names])
                id_to_name[lid] = lname
            end
        end
    end

    if isempty(all_load_ids)
        @warn "No load data available for $case"
        return nothing
    end

    load_ids = sort(collect(all_load_ids))
    load_names = [id_to_name[lid] for lid in load_ids]

    n_funcs_with_data = count(ff -> haskey(per_load_data, ff) && !isempty(per_load_data[ff][:load_ids]), fair_funcs)
    if n_funcs_with_data == 0
        @warn "No valid data for $case"
        return nothing
    end

    bar_width = 0.8 / n_funcs_with_data
    x_offset = bar_width

    if data_type == :pshed
        title = "Active Power Load Shed per Bus: $case"
        ylabel = "Load Shed (%)"
        filename = "pshed_per_bus_$case.svg"
    elseif data_type == :pd_served
        title = "Active Power Load Served per Bus: $case"
        ylabel = "Load Served (%)"
        filename = "pd_served_per_bus_$case.svg"
    elseif data_type == :qshed
        title = "Reactive Power Load Shed per Bus: $case"
        ylabel = "Load Shed (%)"
        filename = "qshed_per_bus_$case.svg"
    elseif data_type == :qd_served
        title = "Reactive Power Load Served per Bus: $case"
        ylabel = "Load Served (%)"
        filename = "qd_served_per_bus_$case.svg"
    else
        error("Unknown data_type: $data_type")
    end

    x_positions = collect(1:length(load_ids))

    p = Plots.plot(
        xlabel = "Bus",
        ylabel = ylabel,
        title = title,
        legend = :top,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm,
        legendfontsize = 12,
        tickfontsize = 12,
        guidefontsize = 12,
        titlefontsize = 16
    )

    offsets = collect(range(-(n_funcs_with_data-1)/2 * x_offset, (n_funcs_with_data-1)/2 * x_offset, length=n_funcs_with_data))

    func_idx = 0
    for fair_func in fair_funcs
        if !haskey(per_load_data, fair_func)
            continue
        end

        data = per_load_data[fair_func]
        func_load_ids = data[:load_ids]
        func_values = data[data_type]

        if isempty(func_load_ids)
            continue
        end

        func_idx += 1

        id_to_value = Dict(zip(func_load_ids, func_values))
        aligned_values = [get(id_to_value, lid, 0.0) for lid in load_ids]

        bar_x = x_positions .+ offsets[func_idx]
        Plots.bar!(p, bar_x, aligned_values,
            bar_width = bar_width * 0.9,
            label = false,
            color = FAIR_FUNC_COLORS[fair_func],
            linecolor = :gray30,
            linestyle = FAIR_FUNC_LINESTYLES[fair_func],
            linewidth = 1.5
        )

        # Overlay markers on bars for pattern differentiation
        fm = FAIR_FUNC_MARKERS[fair_func]
        for (xi, vi) in zip(bar_x, aligned_values)
            if vi > 2.0
                n_markers = max(1, round(Int, vi / 25))
                for mi in 1:n_markers
                    y_pos = vi * mi / (n_markers + 1)
                    Plots.scatter!(p, [xi], [y_pos],
                        marker = fm, markersize = 5,
                        color = :gray40, markerstrokecolor = :gray40,
                        label = false
                    )
                end
            end
        end
    end

    # Add marker-based legend entries (symbols, not bar colors)
    for fair_func in fair_funcs
        if haskey(per_load_data, fair_func) && !isempty(per_load_data[fair_func][:load_ids])
            fl = FAIR_FUNC_LABELS[fair_func]
            fm = FAIR_FUNC_MARKERS[fair_func]
            fc = FAIR_FUNC_COLORS[fair_func]
            Plots.scatter!(p, [NaN], [NaN],
                marker = fm, markersize = 8,
                color = fc, markerstrokecolor = :gray30,
                label = fl
            )
        end
    end

    Plots.savefig(p, joinpath(save_dir, filename))
    return p
end

"""
    create_shed_distribution_plot(case, per_load_data, power_type, save_dir; fair_funcs=String[])

Create scatter plot showing shed distribution by fairness function.
`power_type`: `:pshed` for active power, `:qshed` for reactive power.
`fair_funcs`: ordered list of fairness function keys to plot.
"""
function create_shed_distribution_plot(case::String, per_load_data::Dict, power_type::Symbol, save_dir::String; fair_funcs::Vector{String}=String[])
    all_load_ids = Set{Int}()
    id_to_name = Dict{Int,String}()
    for (_, data) in per_load_data
        if !isempty(data[:load_ids])
            union!(all_load_ids, data[:load_ids])
            for (lid, lname) in zip(data[:load_ids], data[:load_names])
                id_to_name[lid] = lname
            end
        end
    end

    if isempty(all_load_ids)
        return nothing
    end

    load_ids = sort(collect(all_load_ids))
    load_names = [id_to_name[lid] for lid in load_ids]
    x_positions = collect(1:length(load_ids))

    if power_type == :pshed
        ylabel = "Load Shed (%)"
        title = "Active Power Shed Distribution: $case"
        filename = "pshed_distribution_$case.svg"
    elseif power_type == :qshed
        ylabel = "Load Shed (%)"
        title = "Reactive Power Shed Distribution: $case"
        filename = "qshed_distribution_$case.svg"
    else
        error("Unknown power_type: $power_type")
    end

    p = Plots.plot(
        xlabel = "Load",
        ylabel = ylabel,
        title = title,
        legend = :outertopright,
        xticks = (x_positions, load_names),
        xrotation = 45,
        size = (1200, 600),
        left_margin = 10Plots.mm,
        bottom_margin = 15Plots.mm,
        top_margin = 5Plots.mm
    )

    for fair_func in fair_funcs
        if !haskey(per_load_data, fair_func)
            continue
        end

        data = per_load_data[fair_func]
        if isempty(data[:load_ids])
            continue
        end

        id_to_value = Dict(zip(data[:load_ids], data[power_type]))
        aligned_values = [haskey(id_to_value, lid) ? id_to_value[lid] : error("Load ID $lid not found in id_to_value") for lid in load_ids]

        Plots.scatter!(p, x_positions, aligned_values,
            label = FAIR_FUNC_LABELS[fair_func],
            marker = FAIR_FUNC_MARKERS[fair_func],
            markersize = 8,
            color = FAIR_FUNC_COLORS[fair_func]
        )
    end

    if haskey(per_load_data, "equality_min") && !isempty(per_load_data["equality_min"][power_type])
        max_eq_shed = maximum(per_load_data["equality_min"][power_type])
        Plots.hline!(p, [max_eq_shed], label="EqMin Max Shed", linestyle=:dash, color=:orange, linewidth=2)
    end

    Plots.savefig(p, joinpath(save_dir, filename))
    return p
end
