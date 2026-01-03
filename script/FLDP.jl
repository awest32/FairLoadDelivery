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

ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# Inputs: case file path, percentage of load shed, list of critical load IDs
eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.9, [])

#Initial fair load weights
fair_weights = Float64[]
for (load_id, load) in (math["load"])
    push!(fair_weights, load["weight"])
end

# Gather control MLD results 
# Integer MLD
  pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_integer; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
# Relaxed MLD
 pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_relaxed; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]

          
dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, mld_soln  = lower_level_soln(math, fair_weights, 1)

# Make a copy of the math dictionary
math_new = deepcopy(math)

total_pshed = []
pshed_lower_level = []
pshed_upper_level = []
weight_ids_fin = []
iterations = 10
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
math_relaxed, pshed_lower_level, pshed_upper_level, weight_ids_fin, model = relaxed_fldp(math_new, iterations, fair_weights, fair_func)

# Plot total load shed over iterations
plot(1:iterations, pshed_lower_level, title = "Lower Level Load Shed over Iterations", xlabel = "Iteration", ylabel = "Total Load Shed (kW)", marker = :o)
savefig("lower_level_load_shed_over_iterations_$fair_func.svg")
# Save load shed data to CSV
df = DataFrame(Iteration = 1:iterations, Total_Load_Shed = pshed_lower_level, Lower_Level_Load_Shed = pshed_lower_level, Upper_Level_Load_Shed = pshed_upper_level)
CSV.write("load_shed_data_$fair_func.csv", df)
# Display the plot with all three lines
display(plot(1:iterations, [pshed_lower_level pshed_upper_level], labels = ["Total Load Shed" "Lower-Level Load Shed" "Upper-Level Load Shed"], title = "Load Shed over Iterations", xlabel = "Iteration", ylabel = "Load Shed (kW)", marker = :o))
savefig("load_shed_comparison_over_iterations_$fair_func.svg")
# Plot the load shed comparison, y-axis upper level load shed, x-axis lower level load shed
# color each iteration differently
iter_annotation = []
for i in 1:iterations
    push!(iter_annotation, string(i))
end
pshed_comparison = scatter(pshed_lower_level, pshed_upper_level, title = "Load Shed Comparison", xlabel = "Lower-Level Load Shed (kW)", ylabel = "Upper-Level Load Shed (kW)", marker = :o, label = "Iteration")
for i in 1:iterations
    annotate!(pshed_comparison, pshed_lower_level[i], pshed_upper_level[i], text(string(i), :left, :green, 30))
end
# add a 45 degree line to the pshed_comparison plot 
plot!(pshed_comparison, [minimum([pshed_lower_level,pshed_upper_level]) maximum([pshed_lower_level,pshed_upper_level])], [minimum([pshed_lower_level,pshed_upper_level]) maximum([pshed_lower_level,pshed_upper_level])], label = "y=x", line = (:dash, :red))
display(pshed_comparison)
savefig("load_shed_comparison_$fair_func.svg")

# Find the switch samples from the relaxed solution using solve_mc_mld_shed_implicit_diff
mld_implicit_diff_relaxed = solve_mc_mld_shed_implicit_diff(math_relaxed, ipopt; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
imp_diff_model = instantiate_mc_model(
    math_relaxed,
    LinDist3FlowPowerModel,
    build_mc_mld_shedding_implicit_diff;
    ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
)
ref = imp_diff_model.ref[:it][:pmd][:nw][0]
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
n_rounds = 5 # Change the randome seed per round to get different results
n_bernoulli_samples = 10

bernoulli_switch_selection_exp = Vector{Dict{Int, Float64}}()
bernoulli_block_selection_exp = Vector{Dict{Int, Float64}}()
bernoulli_load_selection_exp = Vector{Dict{Int, Float64}}()

bernoulli_selection_index = []
bernoulli_samples = Dict{Int,Vector{Dict{Int, Float64}}}()
for r in 1:n_rounds
    rng = 100*r
    #rng = Random.MersenneTwister(100 * r)
    # Generate bernoulli samples for switches and blocks
    bernoulli_samples[r] = generate_bernoulli_samples(block_status, n_bernoulli_samples, rng)

    # Find the best bernoulli sample be topology feasible and closes to the relaxed solution
    index, block_states, switch_ids, switch_status, load_ids, load_status = radiality_check(ref, switch_states, block_status,bernoulli_samples[r])
    @info "Round $r: Best radial sample index: $index"
    @info "Round $r: Best radial sample switch status: $switch_status"
    @info "Round $r: Best radial sample block status: $block_status"
    @info "Round $r: Best radial sample load status: $load_status"
    push!(bernoulli_selection_index, index)
    push!(bernoulli_block_selection_exp, block_states)
    push!(bernoulli_switch_selection_exp, zip(switch_ids, switch_status) |> Dict)
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
    push!(math_out, update_network(math_random_test[r], bernoulli_block_selection_exp[r], bernoulli_load_selection_exp[r], bernoulli_switch_selection_exp[r], ref, r))
end

# Test the AC feasibility of each rounded solution
# Use the PMD IVRUPowerModel for AC power flow testing
ac_feas = Vector{Dict{String, Any}}()
# Allowing the solution to be reached iteration limit
for r in 1:n_rounds
    @info "Testing AC feasibility for rounded solution from round $r"
    feas_dict = ac_feasibility_test(math_out[r], r)
    push!(ac_feas, feas_dict)
end

# Find the feasibility solution which serves the most load
max_load_served = -Inf
best_feasibility = nothing
for feas in ac_feas
    if haskey(feas, "feas_obj") && feas["feas_obj"] != nothing
        load_served = feas["feas_obj"]
        if load_served > max_load_served
            max_load_served = load_served
            best_feasibility = feas
        end
        @info "Maximum load served among feasible AC solutions: $max_load_served"
        if best_feasibility != nothing
            @info "Best feasibility solution details: $best_feasibility"        
        end
    else
        @warn "Feasibility dictionary does not contain an AC feasible solution."
        @info "Finding best MLD objective"
    end
end

best_mld = Dict{String, Any}()
best_obj = -Inf
best_set = 0 
# data = math_out[1]
# mld = instantiate_mc_model(
#     data,
#     LinDist3FlowPowerModel,
#     build_mc_mld_shedding_random_rounding;
#     ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
# )
for (id, data) in enumerate(math_out)
    #@info id
    mld = FairLoadDelivery.solve_mc_mld_shed_random_round(data, ipopt)
    if best_obj <= mld["objective"] 
        best_obj = mld["objective"]
        best_set = id
        best_mld = mld
    end
end



# plot the best solution 
eng_out = PowerModelsDistribution.transform_data_model(math_out[best_feasibility["set_id"]])

p = powerplot(eng_out, bus = (:data=>"bus_type", :data_type=>"nominal"),
                    branch = (:data=>"index", :data_type=>"ordinal"),
                    gen    = (:data=>"pmax", :data_type=>"quantitative"),
                    load   = (:data=>"pd",  :data_type=>"quantitative"),
                    shunt = (:data=>"gs", :data_type=>"quantitative"),
                    title = "Best AC Feasible Solution from Random Rounding",
                    width = 300, height=300
)
