#######################################################################
#
# RESULTS
#
#######################################################################
test_condition = "case_13_bus_constrained_generation_switchable_mld"
# Find the load served and load shed per bus
load_shed = Dict()
generation = Dict()
load_served = Dict()
expected_load = Dict()
total_power_generated =  []
total_load = []
total_load_shed = []
total_power_gen_capacity = 0

total_ref_load = []
for i in collect(keys(res["gen"]))
	generation[i] = sum(res["gen"][i]["pg"])#+sum(res["gen"][i]["qg"].^2))
	push!(total_power_generated, generation[i])
end
for i in keys(res["load"])
	load_served[i] = sum(res["load"][i]["pd"])#+sum(res["load"][i]["qd"].^2))
	expected_load[i] = sum(ref[:load][parse(Int,i)]["pd"])#sum(res["load"][i]["pd0"])
    load_shed[i] =  expected_load[i]-load_served[i]
	push!(total_load, load_served[i])
	push!(total_load_shed, load_shed[i])
	push!(total_ref_load, expected_load[i])
end
println("Total power generated: ", sum(total_power_generated))
#println("Total load: ", sum(total_load))
println("Total ref load: ", sum(total_ref_load))
println("Total load shed: ", sum(total_ref_load)-sum(total_load))
#println("Total load served: ", sum(total_load))#- sum(total_load_shed))
#println("Total load shed: ", sum(total_ref_load) - sum(total_power_generated))

# Create a results folder
if !isdir("results")
	mkdir("results")
end

# Create a folder per date
using Dates
today = Dates.today()
date_folder = "results/$(today)"
if !isdir(date_folder)
	mkdir(date_folder)
end

# Create a folder for the voltage magnitudes
voltage_folder = "results/$(today)/voltage"
if !isdir(voltage_folder)
	mkdir(voltage_folder)
end

# Create a folder for the power delivered and generated
power_folder = "results/$(today)/power"
if !isdir(power_folder)
	mkdir(power_folder)
end

# Create a folder for the switch states
switch_folder = "results/$(today)/switch_states"
if !isdir(switch_folder)
	mkdir(switch_folder)
end



# Create a bar plot for the generation per bus
#bar(collect(keys(generation)), collect(values(generation)), label="Generation", legend=:topright)
lp = bar(collect(keys(expected_load)), collect(values(expected_load)), label="Expected Load", linestyle=:dash, color=:black, alpha=0.15)
bar!(collect(keys(load_served)), collect(values(load_served)), label="Load Served",linestyle=:solid, color=:green, alpha=0.25)
bar!(collect(keys(load_shed)), collect(values(load_shed)), label="Load Shed",linestyle=:solid, color=:red, alpha=0.5)
xlabel!("Bus")
ylabel!("Power (MW)")
title!("Active Power Delivered per Bus")			

# Create a grouped bar chart to show each load component per bus side by side
bar_width = 0.25
x_positions = collect(keys(expected_load))


# Save this figure in the power folder
savefig(lp, "$(power_folder)/power_delivered_per_bus_$test_condition.png")
#savefig(lp_grouped, "$(power_folder)/grouped_power_delivered_per_bus_$test_condition.svg")
# Create a separate plot for the generation per bus
gp = bar(collect(keys(generation)), collect(values(generation)), label="Generation", legend=:topright)
xlabel!("Generator")
ylabel!("Power (MW)")
title!("Active Power Generated per Generator")


# Save the figure
savefig(gp, "$(power_folder)/generation_per_bus_$test_condition.png")

# Create and save a figure with the power delivered and generated per bus as two subfigures
p = plot(lp, gp, layout=(2, 1))
savefig(p, "$(power_folder)/power_delivered_and_generated_per_bus_$test_condition.png")


# Simple approach: plot all voltage values for each bus
bus_data = res["bus"]
bus_ref_data = ref[:bus]
all_bus_ids = Float64[]
all_v_mags = Float64[]
phase_labels = String[]

# identify the sourcebus from the bus_data
sourcebus = findall(x -> x["name"] == "sourcebus", bus_ref_data)

# Sort bus data based upon electrical distance from source bus
#From Ryan: redefine determine_node_locs to be determine_node_locs
function determine_node_locs(ref, hops, sourcebus)
    # cont_batts --> sourcebus
    #determine the possible battery locations
    if isinf(hops)
        #if hops == Inf, then we can put batteries at any bus
        node_locs = sort(collect(keys(ref[:bus])))
    else #if hops < Inf, then determine possible battery locations
        node_locs = []
        for i in sourcebus
           push!(node_locs, i)
        end
        unique!(node_locs)
        node_search = 0
        while node_search < hops
            new_node_locs = []
            for (i,branch) in sort(ref[:branch])
                if ref[:branch][i]["f_bus"] in node_locs
                    push!(new_node_locs, ref[:branch][i]["t_bus"])
                elseif ref[:branch][i]["t_bus"] in node_locs
                    push!(new_node_locs, ref[:branch][i]["f_bus"])
                end
            end
            for (i,switch) in sort(ref[:switch])
                if ref[:switch][i]["f_bus"] in node_locs
                    push!(new_node_locs, ref[:switch][i]["t_bus"])
                elseif ref[:switch][i]["t_bus"] in node_locs
                    push!(new_node_locs, ref[:switch][i]["f_bus"])
                end
            end
            
            for j in new_node_locs
                push!(node_locs, j)
            end
            unique!(node_locs)
            node_search = node_search + 1
        end
       unique!(node_locs)
    end
    return node_locs
end
function determine_branch_locs(ref, hops, sourcebus)
    # Determine the reachable branches within a certain number of hops
    if isinf(hops)
        # If hops == Inf, return all branches and switches
        branch_locs = collect(keys(ref[:branch]))
        switch_locs = collect(keys(ref[:switch]))
        return vcat(branch_locs, switch_locs)
    else
        node_locs = []
        for i in sourcebus
           push!(node_locs, i)
           println("Starting at sourcebus node: $i")
        end
        unique!(node_locs)
        
        branch_locs = []
        node_search = 0
        
        while node_search < hops
            new_node_locs = []
            # Iterate through branches in their original order
            for (i,branch) in ref[:branch]
                if ref[:branch][i]["f_bus"] in node_locs
                    push!(new_node_locs, ref[:branch][i]["t_bus"])
                    push!(branch_locs, i)  # Add branch index
                    println("Branch locations so far: $branch_locs")
                    @info "Found branch $i from f_bus $(ref[:branch][i]["f_bus"]) to t_bus $(ref[:branch][i]["t_bus"])"
                elseif ref[:branch][i]["t_bus"] in node_locs
                    push!(new_node_locs, ref[:branch][i]["f_bus"])
                    push!(branch_locs, i)  # Add branch index
                    @info "Found branch $i from t_bus $(ref[:branch][i]["t_bus"]) to f_bus $(ref[:branch][i]["f_bus"])"
                end
            end
            # Iterate through switches in their original order
            for (i,switch) in ref[:switch]
                if ref[:switch][i]["f_bus"] in node_locs
                    push!(new_node_locs, ref[:switch][i]["t_bus"])
                    push!(branch_locs, i)  # Add switch index
                    println("Switch locations so far: $branch_locs")
                    @info "Found switch $i from f_bus $(ref[:switch][i]["f_bus"]) to t_bus $(ref[:switch][i]["t_bus"])"
                elseif ref[:switch][i]["t_bus"] in node_locs
                    push!(new_node_locs, ref[:switch][i]["f_bus"])
                    push!(branch_locs, i)  # Add switch index
                    @info "Found switch $i from t_bus $(ref[:switch][i]["t_bus"]) to f_bus $(ref[:switch][i]["f_bus"])"
                end
            end
            
            for j in new_node_locs
                push!(node_locs, j)
                @info "Found node $j into node_locs $node_locs"
            end
            unique!(node_locs)
            @info "After hop $node_search, node_locs: $node_locs"
            unique!(branch_locs)  # Remove duplicate branches
            @info "After hop $node_search, branch_locs: $branch_locs"
            node_search = node_search + 1
            @info "Hops: $node_search, Nodes found: $(length(node_locs)), Branches found: $(length(branch_locs))"
        end
        unique!(branch_locs)
    end
    return branch_locs
end
 
max_hops = 15
branch_locations = Dict{Int,Any}()
for i in 1:max_hops
    println("Determining branch locations for hop $i")
    branch_locations[i] = determine_branch_locs(ref, i, sourcebus)
end
println(branch_locations)

exclusive_branch_hops = Dict{Int,Any}()
for i in 1:max_hops
    if i == 1
        exclusive_branch_hops[i] = branch_locations[i]
    else
        exclusive_branch_hops[i] = filter(x -> !(x in branch_locations[i-1]), branch_locations[i])
    end
end
println(exclusive_branch_hops)
 
max_hops = 6
node_locations = Dict{Int,Any}()
for i in 1:max_hops
    node_locations[i] = determine_node_locs(ref, i, sourcebus)
end
println(node_locations)
exclusive_hops = Dict{Int,Any}()
for i in 1:max_hops
    if i == 1
        exclusive_hops[i]= node_locations[i]
    else
        exclusive_hops[i]= filter(x -> !(x in node_locations[i-1]), node_locations[i])
    end
end
println(exclusive_hops)

# Collect all voltage data
global x_ind = 0
#sorted_buses = Dict{Int,Any,Any}()
for buses in sort(exclusive_hops)
    hops = buses
    #println(hops[1])
    buses_in_hop = hops[2]
    #println(buses_in_hop)
    x_ind = hops[1]
    #println(x_ind)
    y_ind = 0
    for index in buses_in_hop
        #println(index)
        # y_ind += 1#0.10
        # println(x_ind+y_ind)
        bus = bus_data[string(index)]
        #v_values = bus["vm"][:]
        v_values = bus["w"][:]
        for (i, v_val) in enumerate(v_values)
            push!(all_bus_ids, index)
            push!(all_v_mags, v_val)
            # Label phases as a, b, c or use index if more than 3
            phase_label = i == 1 ? "a" : (i == 2 ? "b" : (i == 3 ? "c" : "phase_$i"))
            push!(phase_labels, phase_label)
        end
    end
end
# Use the same bus ordering as the scatter plot
# unique_bus_ids = unique(all_bus_ids)  # This should match the scatter plot ordering

# # Create dictionaries for the ordered data
# ordered_expected_load = [get(expected_load, bus_id, 0.0) for bus_id in unique_bus_ids]
# ordered_load_served = [get(load_served, bus_id, 0.0) for bus_id in unique_bus_ids]
# ordered_load_shed = [get(load_shed, bus_id, 0.0) for bus_id in unique_bus_ids]

# # Create position indices for x-axis
# position_values = 1:length(unique_bus_ids)

# # Create bar plot with ordered buses
# lp = bar(position_values, ordered_expected_load, 
#     label="Expected Load", 
#     linestyle=:dash, 
#     color=:black, 
#     alpha=0.15)
# bar!(position_values, ordered_load_served, 
#     label="Load Served",
#     linestyle=:solid, 
#     color=:green, 
#     alpha=0.25)
# bar!(position_values, ordered_load_shed, 
#     label="Load Shed",
#     linestyle=:solid, 
#     color=:red, 
#     alpha=0.5)

# # Set x-axis ticks to show bus IDs (same as scatter plot)
# xticks!(lp, position_values, string.(Int.(unique_bus_ids)))
# xlabel!("Bus (sorted by distance from source)")
# ylabel!("Power (MW)")
# title!("Active Power Delivered per Bus")

# Create scatter plot grouped by phase
using Plots
svm = scatter()
unique_bus_ids = unique(all_bus_ids)
bus_to_position = Dict(bus_id => i for (i, bus_id) in enumerate(unique_bus_ids))
all_positions = [bus_to_position[id] for id in all_bus_ids]
# Group by phase and hops away from the source, then plot
unique_phases = unique(phase_labels)
markers = [:circle, :square, :diamond, :star, :cross]
colors = [:red, :blue, :green, :orange, :purple]

#for (hops,buses) in enumerate(exclusive_hops)
for (i, phase) in enumerate(unique_phases)
    phase_mask = phase_labels .== phase
    marker_idx = min(i, length(markers))
    color_idx = min(i, length(colors))
    
#       scatter!(svm, all_bus_ids[phase_mask], all_v_mags[phase_mask],
        scatter!(svm, all_positions[phase_mask], all_v_mags[phase_mask],
            label=phase,
            marker=markers[marker_idx],
            color=colors[color_idx],
            markersize=6)
end
#end
unique_bus_ids = unique(all_bus_ids)
xticks!(svm, unique_bus_ids, string.(unique_bus_ids))
xlabel!(svm, "Buses sorted by distance from the sourcebus")
ylabel!(svm, "Voltage Magnitude (p.u.)")
title!(svm, "Voltage Magnitudes by Bus and Phase")

# Set x-axis ticks to show bus IDs
position_values = 1:length(unique_bus_ids)
xticks!(svm, position_values, string.(Int.(unique_bus_ids)))

# Plot vmin/vmax lines using positions
plot!(svm, position_values, (0.9*ones(length(unique_bus_ids))).^2, label="vmin")
plot!(svm, position_values, (1.1*ones(length(unique_bus_ids))).^2, label="vmax")


# x_vals = collect(LinRange(minimum(node_locations[6]), maximum(node_locations[6]), length(node_locations[6])))
# x_vals = [x_vals[1]; x_vals[end]]
# plot!(unique_bus_ids, (0.9*ones(length(unique_bus_ids))).^2, label="vmin")#.^2
# plot!(unique_bus_ids, (1.1*ones(length(unique_bus_ids))).^2, label="vmax")
ylabel!("W (pu)")
# plot!([0; length(res["bus"])], [0.9; 0.9].^2, label="vmin")#.^2
# plot!([0; length(res["bus"])], [1.1; 1.1].^2, label="vmax")
#length(node_locations[6])]

#xlabel!("bus id (-)")
#display(svm)
# Save the scatter plot
savefig(svm,"$(voltage_folder)/squared_voltage_magnitudes_$test_condition.png")

# Collect branch power flow data
all_branch_ids = []
all_power_from = []
all_power_to = []
all_power_loss = []
phase_labels_branch = []
branch_s_max = []
node_balance_max = []  # Max gen - Max load
node_balance_min = []  # Min gen - Max load
all_node_ids = []
phase_labels_node = []
all_power_balance = []
# Iterate through exclusive_hops to get branches in order
for (hop, branches) in sort(collect(exclusive_branch_hops))
    for branch_id in branches
        if haskey(ref[:branch], branch_id)
            branch = ref[:branch][branch_id]
            # Get power flow data for each phase
            for c in 1:length(branch["f_connections"])
                push!(all_branch_ids, branch_id)
                # Assuming you have solution data - adjust variable names as needed
                pf = res["branch"][string(branch_id)]["pf"][c]
                pt = res["branch"][string(branch_id)]["pt"][c]
                push!(all_power_from, pf)
                push!(all_power_to, pt)
                push!(all_power_loss, pf + pt)  # Loss = pf + pt (pt is negative)
                push!(phase_labels_branch, string(branch["f_connections"][c]))
                if haskey(branch, "rate_a")
                    #println(branch["rate_a"])
                    s_max = branch["rate_a"][c]  # Adjust field name as needed
                    push!(branch_s_max, s_max)
                else
                    s_max = Inf
                    push!(branch_s_max, s_max)
                end
            end
        elseif haskey(ref[:switch], branch_id)
            switch = ref[:switch][branch_id]
            # Get power flow data for switches
            for c in 1:length(switch["f_connections"])
                push!(all_branch_ids, branch_id)
                psw = _PMD.sol(pm, nw, :switch, branch_id, :psw)[c]
                push!(all_power_from, psw)
                push!(all_power_to, -psw)
                push!(all_power_loss, 0.0)  # Switches typically have no loss
                push!(phase_labels_branch, string(switch["f_connections"][c]))
                s_max = switch["rating"][c]  # Adjust field name as needed
                push!(branch_s_max, s_max)
            end
        end
    end
end

# Create position mapping for branches
unique_branch_ids = unique(all_branch_ids)
branch_to_position = Dict(branch_id => i for (i, branch_id) in enumerate(unique_branch_ids))
all_branch_positions = [branch_to_position[id] for id in all_branch_ids]

# Plot power flow
spf = scatter()

unique_phases = unique(phase_labels_branch)
markers = [:circle, :square, :diamond, :star, :cross]
colors = [:red, :blue, :green, :orange, :purple]

# First, plot the whiskers (vertical lines for limits)
for i in 1:length(all_branch_positions)
    plot!(spf, [all_branch_positions[i], all_branch_positions[i]], 
          [-branch_s_max[i], branch_s_max[i]],
          color=:gray,
          alpha=0.3,
          linewidth=2,
          label=(i==1 ? "Power Limits" : ""))
    
    # Add horizontal caps at the ends of whiskers
    cap_width = 0.2
    plot!(spf, [all_branch_positions[i]-cap_width, all_branch_positions[i]+cap_width],
          [branch_s_max[i], branch_s_max[i]],
          color=:gray,
          alpha=0.3,
          linewidth=2,
          label="")
    plot!(spf, [all_branch_positions[i]-cap_width, all_branch_positions[i]+cap_width],
          [-branch_s_max[i], -branch_s_max[i]],
          color=:gray,
          alpha=0.7,
          linewidth=2,
          label="")
end

for (i, phase) in enumerate(unique_phases)
    phase_mask = phase_labels_branch .== phase
    marker_idx = min(i, length(markers))
    color_idx = min(i, length(colors))
    
    scatter!(spf, all_branch_positions[phase_mask], all_power_from[phase_mask],
        label="Phase $phase (from)",
        marker=markers[marker_idx],
        color=colors[color_idx],
        markersize=6)
end

# Set x-axis ticks to show branch IDs
position_values = 1:length(unique_branch_ids)
xticks!(spf, position_values, string.(Int.(unique_branch_ids)))

xlabel!(spf, "Branches sorted by distance from sourcebus")
ylabel!(spf, "Power Flow (kW)")
title!(spf, "Power Flow by Branch and Phase")
#display(spf)

# Save the figure
savefig(spf,"$(power_folder)/branch_limits_inf_$test_condition.png")

# Iterate through exclusive_hops to get nodes in order
for (hop, buses) in sort(exclusive_hops)
    for bus_id in buses
        bus = ref[:bus][bus_id]
        
        # Get connections/phases for this bus
        terminals = bus["terminals"]
        
        for (idx, term) in enumerate(terminals)
            push!(all_node_ids, bus_id)
            push!(phase_labels_node, string(term))
            
            # Calculate actual power balance (generation - load)
            gen_power = 0.0
            load_power = 0.0
            max_gen = 0.0
            min_gen = 0.0
            max_load = 0.0
            
            # Sum generation at this bus and phase
            if haskey(ref[:bus_gens], bus_id)
                for gen_id in ref[:bus_gens][bus_id]
                    gen = ref[:gen][gen_id]
                    # Find the phase index
                    if term in gen["connections"]
                        phase_idx = findfirst(x -> x == term, gen["connections"])
                        if !isnothing(phase_idx)
                            # Get actual generation from solution
                            gen_power += res["gen"][string(gen_id)]["pg"][phase_idx]
                            # Get generation limits
                            max_gen += gen["pmax"][phase_idx]
                            min_gen += gen["pmin"][phase_idx]
                        end
                    end
                end
            end
            
            # Sum load at this bus and phase
            if haskey(ref[:bus_loads], bus_id)
                for load_id in ref[:bus_loads][bus_id]
                    load = ref[:load][load_id]
                    if term in load["connections"]
                        phase_idx = findfirst(x -> x == term, load["connections"])
                        #println(load_id)
                        #println(phase_idx)
                        if !isnothing(phase_idx)
                            # Get actual load from solution
                            load_power += res["load"][string(load_id)]["pd"][phase_idx]
                            # Get max load
                            max_load += load["pd"][phase_idx]
                        end
                    end
                end
            end
            
            # Power balance = generation - load
            push!(all_power_balance, gen_power - load_power)
            push!(node_balance_max, max_gen - max_load)
            push!(node_balance_min, min_gen - max_load)
        end
    end
end

# Create position mapping for nodes
unique_node_ids = unique(all_node_ids)
node_to_position = Dict(node_id => i for (i, node_id) in enumerate(unique_node_ids))
all_node_positions = [node_to_position[id] for id in all_node_ids]

# Plot power balance with limits
spb = plot()

unique_phases = unique(phase_labels_node)
markers = [:circle, :square, :diamond, :star, :cross]
colors = [:red, :blue, :green, :orange, :purple]

# First, plot the whiskers (vertical lines for limits)
for i in 1:length(all_node_positions)
    plot!(spb, [all_node_positions[i], all_node_positions[i]], 
          [node_balance_min[i], node_balance_max[i]],
          color=:red,
          alpha=0.3,
          linewidth=2,
          label=(i==1 ? "Power Balance Limits" : ""))
    
    # Add horizontal caps at the ends of whiskers
    cap_width = 0.2
    plot!(spb, [all_node_positions[i]-cap_width, all_node_positions[i]+cap_width],
          [node_balance_max[i], node_balance_max[i]],
          color=:red,
          alpha=0.9,
          linewidth=2,
          label="")
    plot!(spb, [all_node_positions[i]-cap_width, all_node_positions[i]+cap_width],
          [node_balance_min[i], node_balance_min[i]],
          color=:red,
          alpha=0.9,
          linewidth=2,
          label="")
end

# Add zero line for reference
plot!(spb, [0.5, length(unique_node_ids)+0.5], [0, 0],
      color=:black,
      linestyle=:dash,
      linewidth=1,
      label="Zero Balance")

# Then plot the actual power balance points
for (i, phase) in enumerate(unique_phases)
    phase_mask = phase_labels_node .== phase
    marker_idx = min(i, length(markers))
    color_idx = min(i, length(colors))
    
    scatter!(spb, all_node_positions[phase_mask], all_power_balance[phase_mask],
        label="Phase $phase",
        marker=markers[marker_idx],
        color=colors[color_idx],
        markersize=6)
end

# Set x-axis ticks to show bus IDs
position_values = 1:length(unique_node_ids)
xticks!(spb, position_values, string.(Int.(unique_node_ids)))

xlabel!(spb, "Buses sorted by distance from sourcebus")
ylabel!(spb, "Power Balance (Gen - Load) (kW)")
title!(spb, "Power Balance by Bus and Phase with Limits")
savefig(spb,"$(power_folder)/bus_power_balance_$test_condition.png")
#display(spb)

# # Optional: Plot for power losses
# sloss = scatter()

# for (i, phase) in enumerate(unique_phases)
#     phase_mask = phase_labels_branch .== phase
#     marker_idx = min(i, length(markers))
#     color_idx = min(i, length(colors))
    
#     scatter!(sloss, all_branch_positions[phase_mask], all_power_loss[phase_mask],
#         label="Phase $phase",
#         marker=markers[marker_idx],
#         color=colors[color_idx],
#         markersize=6)
# end

# xticks!(sloss, position_values, string.(Int.(unique_branch_ids)))
# xlabel!(sloss, "Branches sorted by distance from sourcebus")
# ylabel!(sloss, "Power Loss (p.u.)")
# title!(sloss, "Power Losses by Branch and Phase")

# # Display plots
# display(sloss)

# Plot the load block switch states
lb_states = []
for (ind, lb) in enumerate(res["block"])
	push!(lb_states, lb[2]["status"])
end
lb = bar(collect(keys(res["block"])), lb_states, label="Load Block States")
xlabel!("Load Block")
ylabel!("State")
title!("Load Block States")

# Save the figure
savefig(lb, "$(switch_folder)/block_switch_states_$test_condition.png")

# Plot the switch states

z_switch =  Dict{Int, Float64}()
for (ind, z) in res["switch"]
    z_switch[parse(Int, ind)] = z["state"]
    #z_switch[parse(Int,ind)] = ["switch"][ind]["state"]
    println("Switch ", ind, " state: ", z_switch[parse(Int,ind)])   
end

ss = bar(collect(keys(z_switch)), collect(values(z_switch)), label="Switch States")
xlabel!("Switch")
ylabel!("State")
title!("Switch States")

# Save the figure
savefig(ss, "$(switch_folder)/switch_states_$test_condition.png")

# Plot and save the demand states per bus
z_demand =  Dict{Int, Float64}()
for (ind, z) in res["load"]
    z_demand[parse(Int, ind)] = z["status"]
end

sd = bar(collect(keys(z_demand)), collect(values(z_demand)), label="Demand States")
xlabel!("Demand")
ylabel!("State")
title!("Demand States")

# Save the figure
savefig(sd, "$(switch_folder)/demand_states_$test_condition.png")