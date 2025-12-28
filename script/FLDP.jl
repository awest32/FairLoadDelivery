using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS, Juniper
using HSL_jll
using Plots
using Random
using Distributions
using DiffOpt
using JuMP
using LinearAlgebra,SparseArrays
using PowerPlots
using DataFrames
using CSV
using Plots
# using DataFrames
ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# Inputs: case file path, percentage of load shed, list of critical load IDs
eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_a.dss", 0.5, [])

#Initial fair load weights
fair_weights = Float64[]
for (load_id, load) in (math["load"])
    push!(fair_weights, load["weight"])
end


dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, mld_soln  = lower_level_soln(math, fair_weights, 1)
# plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids, 1)

# fair_weights = Float64[]
# for i in weight_ids
#     load = math["load"][string(i)]
#     push!(fair_weights, load["weight"])
# end

# Make a copy of the math dictionary
math_new = deepcopy(math)

total_pshed = []
pshed_lower_level = []
pshed_upper_level = []
weight_ids_fin = []
iterations = 1
fair_func = "proportional"
function relaxed_fldp(data::Dict{String, Any}, iterations::Int, fair_weights::Vector{Float64}, fair_func::String)
    # Build and solve the relaxed Fair Load Delivery Problem (FLDP)
    ref_out = []
    for k in 1:iterations
        @info "Starting iteration $k"

        # Solve lower-level problem and get sensitivities
        dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, lower_level_ref = lower_level_soln(data, fair_weights, k)
        push!(ref_out,lower_level_ref)
        # Plot the heatmap of sensitivities
        plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids, k)
        
        # Plot load shed per bus
        plot_load_shed_per_bus(pshed_val, pshed_ids, k)
        
        # Update weights using Lin-PALMA-W with gradient input
        # pshed_new, fair_weight_vals, sigma = lin_palma_w_grad_input(dpshed, pshed_val, weight_vals, pd)
        if fair_func == "proportional"
            pshed_new, fair_weight_vals = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals)
        elseif fair_func == "efficiency"
            pshed_new, fair_weight_vals = complete_efficiency_load_shed(dpshed, pshed_val, weight_vals)
        elseif fair_func == "min_max"
            pshed_new, fair_weight_vals = min_max_load_shed(dpshed, pshed_val, weight_vals) 
        elseif fair_func == "jain"
            pshed_new, fair_weight_vals = jains_fairness_index(dpshed, pshed_val, weight_vals)
        elseif fair_func == "palma"
             # Order the load using the indices from the pshed_ids
            pd = Float64[]
            for i in pshed_ids
                push!(pd, sum(math_new["load"][string(i)]["pd"]))
            end
            pshed_new, fair_weight_vals = lin_palma_w_grad_input(dpshed, pshed_val, weight_vals, pd)
        end
        #pshed_new, fair_weight_vals = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals)
        #pshed_new, fair_weight_vals = complete_efficiency_load_shed(dpshed, pshed_val, weight_vals)
        #pshed_new, fair_weight_vals = min_max_load_shed(dpshed, pshed_val, weight_vals)
        #pshed_new, fair_weight_vals = jains_fairness_index(dpshed, pshed_val, weight_vals)
        
        plot_weights_per_load(fair_weight_vals, weight_ids, k)
        
        # Update the fair load weights in the math dictionary
        for (i, w) in zip(weight_ids, fair_weight_vals)
            math_new["load"][string(i)]["weight"] = w
        end
        
        # Store the total load shed for this iteration
        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))
        push!(weight_ids_fin, weight_ids)
    end
    return math, pshed_lower_level, pshed_upper_level, weight_ids_fin, ref_out[1]
end

math_relaxed, pshed_lower_level, pshed_upper_level, weight_ids_fin, ref = relaxed_fldp(math_new, 1, fair_weights, fair_func)

# Find the switch samples from the relaxed solution using solve_mc_mld_shed_implicit_diff
mld_implicit_diff_relaxed = solve_mc_mld_shed_implicit_diff(math_relaxed, ipopt; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])

# Extract the switch states and load block statuses from the math dictionary, save as seperate dictionaries 
function extract_switch_block_states(relaxed_soln::Dict{String,Any})
    # Create the switch state dictionary
    switch_states = Dict{Int64,Float64}([])
    for (s_id, s_data) in relaxed_soln["switch"]
        switch_states[parse(Int,s_id)] = s_data["state"]
    end

    #Create the block status dictionary
    block_status = Dict{Int64,Float64}([])
    for (b_id, b_data) in relaxed_soln["block"]
        block_status[parse(Int,b_id)] = b_data["status"]
    end
    return switch_states, block_status
end

switch_states, block_status = extract_switch_block_states(mld_implicit_diff_relaxed["solution"])

# Determine the number of rounding rounds and bernoulli samples per round
n_rounds = 3 # Change the randome seed per round to get different results
n_bernoulli_samples = 10

bernoulli_switch_selection_exp = Vector{Dict{Int, Float64}}()
bernoulli_block_selection_exp = Vector{Dict{Int, Float64}}()
bernoulli_load_selection_exp = Vector{Dict{Int, Float64}}()

bernoulli_selection_index = []
for r in 1:n_rounds
    #seed = 100*r
    rng = Random.MersenneTwister(100 * r)
    bernoulli_samples = Dict{Int,Vector{Dict{Int, Float64}}}()
    # Generate bernoulli samples for switches and blocks
    bernoulli_samples[r] = generate_bernoulli_samples(switch_states, n_bernoulli_samples, rng)

    # Find the best bernoulli sample be topology feasible and closes to the relaxed solution
    index, selection, block_ids, block_status, load_ids, load_status = radiality_check(ref, switch_states, bernoulli_samples[r])
    @info "Round $r: Best radial sample index: $index"
    @info "Round $r: Best radial sample selection: $selection"
    @info "Round $r: Best radial sample block status: $block_status"
    @info "Round $r: Best radial sample load status: $load_status"
    push!(bernoulli_selection_index, index)
    push!(bernoulli_switch_selection_exp, selection)
    push!(bernoulli_block_selection_exp, zip(block_ids, block_status) |> Dict)
    push!(bernoulli_load_selection_exp, zip(load_ids, load_status) |> Dict)
end

# Create R copies of the math dictionary for random rounding testing
math_random_test = Vector{Dict{String, Any}}()
for r in 1:n_rounds
    math_copy = deepcopy(math_relaxed)
    push!(math_random_test, math_copy)
end
math_out = Vector{Dict{String, Any}}()
# Apply the best switch configuration from each round to the respective math dictionary
for r in 1:n_rounds
    math_out = update_network(math_random_test[r], bernoulli_switch_selection_exp[r], bernoulli_load_selection_exp[r], bernoulli_block_selection_exp[r], ref, r)
    # best_switch_config = bernoulli_switch_selection_exp[r]
    # @info "The best switch configuration for round $r is $best_switch_config"
    # for (switch_id, switch_state) in best_switch_config
    #     @info "Setting switch $switch_id to state $switch_state in math dictionary for round $r"
    #     math_random_test[r]["switch"][string(switch_id)]["state"] = switch_state
    #     math_random_test[r]["switch"][string(switch_id)]["status"] = switch_state
    #     math_random_test[r]["switch"][string(switch_id)]["dispatchable"] = 0.0
    #     @info "Switch $switch_id state in math dictionary is now $(math_random_test[r]["switch"][string(switch_id)]["state"])"
    # end
    # # De-energize load blocks based on the block status from the relaxed solution
    # for (load_id, load_data) in math_random_test[r]["load"]
    #     if bernoulli_load_selection_exp[r][parse(Int,load_id)] <= 0.0
    #         @info "De-energizing load $load_id in round $r"
    #         math_random_test[r]["load"][load_id]["pd"] = 0.0
    #         math_random_test[r]["load"][load_id]["qd"] = 0.0
    #         math_random_test[r]["load"][load_id]["status"] = 0.0
    #     end
    # end
    # # De-energize shunts based on the block status from the relaxed solution
    # for (shunt_id, shunt_data) in math_random_test[r]["shunt"]
    #     block_id = ref[:shunt_block_map][parse(Int,shunt_id)]
    #     if bernoulli_block_selection_exp[r][block_id] <= 0.0
    #         @info "De-energizing shunt $shunt_id in round $r"
    #         math_random_test[r]["shunt"][shunt_id]["status"] = 0.0
    #     end
    # end

    # # De-energize branches based on the block status from the relaxed solution
    # for (block_id, branches) in ref[:block_branches]
    #     for branch_id in branches
    #         @info "Branch $branch_id is in block $block_id"
    #         if bernoulli_block_selection_exp[r][block_id] <= 0.0
    #             @info "De-energizing branch $branch_id in round $r"
    #             math_random_test[r]["branch"][string(branch_id)]["status"] = 0.0
    #         end
    #     end
    # end
end

# Test the AC feasibility of each rounded solution
# Use the PMD IVRUPowerModel for AC power flow testing
function ac_feasibility_test(math_list::Vector{Dict{String, Any}}, bernoulli_samples::Vector{Dict{Int, Float64}}, switch_ids::Vector{Int}; optimizer=ipopt)
    ac_feasible_solutions = Vector{Dict{String, Any}}()
    ac_feasible_values = Float64[]
    for (i, math) in enumerate(math_list)
        # Build the AC power flow model with fixed switches
        # pm_ac = instantiate_mc_model(
        #     math,
        #     PowerModelsDistribution.IVRUPowerModel,
        #     build_mc_opf;
        #     ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
        # )
        pm_ac = solve_mc_opf(math, IVRUPowerModel, ipopt)
        # Fix the switch states according to the bernoulli sample
        #FairLoadDelivery.constraint_rounded_switch_states_f(pm_ac; z_bern=bernoulli_samples[i])
        
        # Optimize the AC power flow model
        #optimize_model!(pm_ac; optimizer=optimizer)
        
        # Check feasibility
        status = pm_ac["termination_status"]# JuMP.termination_status(pm_ac.model)
        if status == MOI.OPTIMAL || status == MOI.FEASIBLE_POINT
            @info "Round $i: AC power flow feasible with status $status"
            push!(ac_feasible_solutions, pm_ac)
            push!(ac_feasible_values, total_pshed)
        else
            @warn "Round $i: AC power flow not feasible with status $status"
        end
    end
    return ac_feasible_solutions, ac_feasible_values
end

ac_feasible_solutions, ac_feasible_values = ac_feasibility_test(math_out, bernoulli_switch_selection_exp, collect(keys(switch_states)), optimizer=ipopt)


math_fin = deepcopy(math_new)
# Plot total load shed over iterations
plot(iterations, pshed_lower_level, title = "Total Load Shed over Iterations", xlabel = "Iteration", ylabel = "Total Load Shed (kW)", marker = :o)
savefig("total_load_shed_over_iterations_$fair_func.svg")
# Save load shed data to CSV
df = DataFrame(Iteration = iterations, Total_Load_Shed = pshed_lower_level, Lower_Level_Load_Shed = pshed_lower_level, Upper_Level_Load_Shed = pshed_upper_level)
CSV.write("load_shed_data_$fair_func.csv", df)
# Display the plot
#display(plot(iterations, total_pshed, title = "Total Load Shed over Iterations", xlabel = "Iteration", ylabel = "Total Load Shed (kW)", marker = :o))
# Display the plot with all three lines
display(plot(iterations, [pshed_lower_level pshed_upper_level], labels = ["Total Load Shed" "Lower-Level Load Shed" "Upper-Level Load Shed"], title = "Load Shed over Iterations", xlabel = "Iteration", ylabel = "Load Shed (kW)", marker = :o))
savefig("load_shed_comparison_over_iterations_$fair_func.svg")
# Plot the load shed comparison, y-axis upper level load shed, x-axis lower level load shed
# color each iteration differently
iter_annotation = []
for i in iterations
    push!(iter_annotation, string(i))
end
pshed_comparison = scatter(pshed_lower_level, pshed_upper_level, title = "Load Shed Comparison", xlabel = "Lower-Level Load Shed (kW)", ylabel = "Upper-Level Load Shed (kW)", marker = :o, label = "Iteration")
for i in iterations
    annotate!(pshed_comparison, pshed_lower_level[i], pshed_upper_level[i], text(string(i), :left, :green, 30))
end
# add a 45 degree line to the pshed_comparison plot 
plot!(pshed_comparison, [minimum([pshed_lower_level,pshed_upper_level]) maximum([pshed_lower_level,pshed_upper_level])], [minimum([pshed_lower_level,pshed_upper_level]) maximum([pshed_lower_level,pshed_upper_level])], label = "y=x", line = (:dash, :red))

display(pshed_comparison)
savefig("load_shed_comparison_$fair_func.svg")