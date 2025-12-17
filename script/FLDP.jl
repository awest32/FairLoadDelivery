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
eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.5, ["675a"])

#Initial fair load weights
fair_weights = Float64[]
for (load_id, load) in (math["load"])
    push!(fair_weights, load["weight"])
end


dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math, fair_weights, 1)
# plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids, 1)

fair_weights = Float64[]
for i in weight_ids
    load = math["load"][string(i)]
    push!(fair_weights, load["weight"])
end

# Make a copy of the math dictionary
math_new = deepcopy(math)

total_pshed = []
pshed_lower_level = []
pshed_upper_level = []
iterations = 1
for k in 1:iterations
    @info "Starting iteration $k"

    # Solve lower-level problem and get sensitivities
    dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math_new, fair_weights, k)
    
    # Plot the heatmap of sensitivities
    plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids, k)
    
    # Plot load shed per bus
    plot_load_shed_per_bus(pshed_val, pshed_ids, k)
    
    # Order the load using the indices from the pshed_ids
    pd = Float64[]
    for i in pshed_ids
        push!(pd, sum(math_new["load"][string(i)]["pd"]))
    end
    
    # Update weights using Lin-PALMA-W with gradient input
    pshed_new, fair_weight_vals, sigma = lin_palma_w_grad_input(dpshed, pshed_val, weight_vals, pd)
    
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
    push!(total_pshed, sum(pshed_val))
    push!(pshed_lower_level, sum(pshed_val))
    push!(pshed_upper_level, sum(pshed_new))
end

math_fin = deepcopy(math_new)
# Plot total load shed over iterations
iterations = collect(1:length(total_pshed))
plot(iterations, total_pshed, title = "Total Load Shed over Iterations", xlabel = "Iteration", ylabel = "Total Load Shed (kW)", marker = :o)
savefig("total_load_shed_over_iterations.svg")
# Save load shed data to CSV
df = DataFrame(Iteration = iterations, Total_Load_Shed = total_pshed, Lower_Level_Load_Shed = pshed_lower_level, Upper_Level_Load_Shed = pshed_upper_level)
CSV.write("load_shed_data.csv", df)
# Display the plot
#display(plot(iterations, total_pshed, title = "Total Load Shed over Iterations", xlabel = "Iteration", ylabel = "Total Load Shed (kW)", marker = :o))
# Display the plot with all three lines
display(plot(iterations, [pshed_lower_level pshed_upper_level], labels = ["Total Load Shed" "Lower-Level Load Shed" "Upper-Level Load Shed"], title = "Load Shed over Iterations", xlabel = "Iteration", ylabel = "Load Shed (kW)", marker = :o))
savefig("load_shed_comparison_over_iterations.svg")
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
savefig("results/load_shed_comparison.svg")