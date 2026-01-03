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
using Dates
"""
This is for the experimental cases.
Select the desired network.
Run the control MLD with switches and load blocks for the integer and the relaxed MLD.
Extract the results from these cases.
    1. Table: blocks, block states
    2. Table: switches, switch states
    3. Plot: pshed v. block
    4. Plot: qshed v. block
    5. Plot: pshed v. switch buses
    6. Plot: qshed v. switch buses
    7. Plot: v^2 v. switch buses
Save these results in the location:
    results/date/case_name/exp
"""
ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

case = "motivation_a"
gen_cap = 0.9
# Inputs: case file path, percentage of load shed, list of critical load IDs
eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/$case.dss", gen_cap, [])

# Create the folder to store the results
    today = Dates.today()

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

    # Create a folder for the control experiment results
    case_name = "results/$(today)/$case"
    if !isdir(case_name)
        mkdir(case_name)
    end

    # Create a folder for the control experiment results
    exp_folder = "results/$(today)/$case/exp"
    if !isdir(exp_folder)
        mkdir(exp_folder)
    end
 

# Create the structure of the output tables
block_df = DataFrame(BlockID=Int64[], GenCap=Float64[], Prioritization=Float64[], RelaxStatus=Float64[], BlockSwitches=Set{Int64}[])
switch_df = DataFrame(SwitchID=Int64[], GenCap=Float64[], RelaxStatus=Float64[], SwitchName=String[])
switch_voltage_df = DataFrame(SwitchID=Int64[], GenCap=Float64[], RelaxTBus=Vector{Float64}[], RelaxFBus=Vector{Float64}[])
# Create the structure of the Plots
pshed_block = plot()
qshed_block = plot()
pshed_switch = plot()
qshed_switch = plot()
v_squared_switch = plot()

# Prepare the bounds to plot for the squared voltage of each switch
n_sw = length(math["switch"])
v_squared_max = (1.1^2)*ones(n_sw)
v_squared_min = (0.9^2)*ones(n_sw)

# Get the reference data for the blocks
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_relaxed; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
ref = mld_model.ref[:it][:pmd][:nw][0]

# Run the relaxed FLDP

#Initial fair load weights
fair_weights = Float64[]
for (load_id, load) in (math["load"])
    push!(fair_weights, load["weight"])
end


# Make a copy of the math dictionary
math_new = deepcopy(math)

total_pshed = []
pshed_lower_level = []
pshed_upper_level = []
weight_ids_fin = []
weight_vals_fin = []
iterations = 10
fair_func = "proportional" # Options: "proportional", "efficiency", "min_max", "jain", "palma"
   
# Create folder for the specific experiment
fair_exp = "results/$(today)/$case/exp/$fair_func"
if !isdir(fair_exp)
    mkdir(fair_exp)
end

function relaxed_fldp(data::Dict{String, Any}, iterations::Int, fair_weights::Vector{Float64}, fair_func::String, save_folder::String)
    # Build and solve the relaxed Fair Load Delivery Problem (FLDP)
    mod_out = []
    math = deepcopy(data)
    for k in 1:iterations
        @info "Starting iteration $k"

        # Solve lower-level problem and get sensitivities
        dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, lower_level_mod = lower_level_soln(math, fair_weights, k)
        push!(mod_out,lower_level_mod)
        # Plot the heatmap of sensitivities
        plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids, k, fair_exp)
        
        # Plot load shed per bus
        plot_load_shed_per_bus(pshed_val, pshed_ids, k, fair_exp)
        
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
       
        plot_weights_per_load(fair_weight_vals, weight_ids, k, fair_exp)
        
        # Update the fair load weights in the math dictionary
        for (i, w) in zip(weight_ids, fair_weight_vals)
            math["load"][string(i)]["weight"] = w
        end
        
        # Store the total load shed for this iteration
        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))
        push!(weight_ids_fin, weight_ids)
        push!(weight_vals_fin, fair_weight_vals)
    end
    return math, pshed_lower_level, pshed_upper_level, weight_ids_fin, weight_vals_fin, mod_out[1]
end
math_relaxed, pshed_lower_level, pshed_upper_level, weight_ids_fin, weight_vals_fin, model = relaxed_fldp(math_new, iterations, fair_weights, fair_func, fair_exp)

# Plot total load shed over iterations
plot(1:iterations, pshed_lower_level, title = "Lower Level Load Shed over Iterations ($fair_func)", xlabel = "Iteration", ylabel = "Total Load Shed (kW)", marker = :o)
savefig("$exp_folder/lower_level_load_shed_over_iterations_$fair_func.svg")
# Save load shed data to CSV
df = DataFrame(Iteration = 1:iterations, Lower_Level_Load_Shed = pshed_lower_level, Upper_Level_Load_Shed = pshed_upper_level)
CSV.write("$exp_folder/load_shed_data_$fair_func.csv", df)
# Display the plot with all three lines
display(plot(1:iterations, [pshed_lower_level pshed_upper_level], labels = ["Total Load Shed" "Lower-Level Load Shed" "Upper-Level Load Shed"], title = "Load Shed over Iterations ($fair_func)", xlabel = "Iteration", ylabel = "Load Shed (kW)", marker = :o))
savefig("$exp_folder/load_shed_comparison_over_iterations_$fair_func.svg")
# Plot the load shed comparison, y-axis upper level load shed, x-axis lower level load shed
# color each iteration differently
iter_annotation = []
for i in 1:iterations
    push!(iter_annotation, string(i))
end
pshed_comparison = scatter(pshed_lower_level, pshed_upper_level, title = "Load Shed Comparison ($fair_func)", xlabel = "Lower-Level Load Shed (kW)", ylabel = "Upper-Level Load Shed (kW)", marker = :o, label = "Iteration")
for i in 1:iterations
    annotate!(pshed_comparison, pshed_lower_level[i], pshed_upper_level[i], text(string(i), :left, :green, 30))
end
# add a 45 degree line to the pshed_comparison plot 
lo = minimum(vcat(pshed_lower_level, pshed_upper_level))
hi = maximum(vcat(pshed_lower_level, pshed_upper_level))

plot!(pshed_comparison, [lo, hi], [lo, hi],
      label = "y = x",
      line = (:dash, :red))
display(pshed_comparison)
savefig("$exp_folder/load_shed_comparison_$fair_func.svg")

# Solve the relaxed MLD with load blocks and switches
mld = solve_mc_mld_shed_random_round(math_relaxed, ipopt; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
mld_mod = instantiate_mc_model(math_relaxed, LinDist3FlowPowerModel, build_mc_mld_shedding_random_rounding; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
ref = mld_mod.ref[:it][:pmd][:nw][0]
block = mld["solution"]["block"]
switch = mld["solution"]["switch"]
for (id, block) in (block)
    push!(block_df.BlockID, parse(Int,id))
    push!(block_df.RelaxStatus, block["status"])
    push!(block_df.GenCap, gen_cap)
    push!(block_df.BlockSwitches, ref[:block_switches][parse(Int,id)])
    push!(block_df.Prioritization, ref[:block_weights][parse(Int,id)])
end

for (id, switch) in switch
    push!(switch_df.SwitchID, parse(Int,id))
    push!(switch_voltage_df.SwitchID, parse(Int,id))
    push!(switch_df.GenCap, gen_cap)
    push!(switch_voltage_df.GenCap, gen_cap)
    push!(switch_df.RelaxStatus, switch["state"])
    push!(switch_df.SwitchName, math["switch"][id]["name"])
end

switch_ids = sort(parse.(Int, collect(keys(mld["solution"]["switch"]))))

fbus_w = Vector{Vector{Float64}}()
tbus_w = Vector{Vector{Float64}}()

for sid in switch_ids
    sw = math["switch"][string(sid)]

    fbus = string(sw["f_bus"])
    tbus = string(sw["t_bus"])

    push!(fbus_w, mld["solution"]["bus"][fbus]["w"])
    push!(tbus_w, mld["solution"]["bus"][tbus]["w"])
end



# Plot the active power load shed vs the block id
# Calculate the total active power in each block
lbs = ref[:block_loads]
block_p = []
block_q = []
for (id, block) in sort(lbs)
    block_load_p = 0
    block_load_q = 0

    for load in block
        block_load_p += mld["solution"]["load"][string(load)]["pshed"]
        block_load_q += mld["solution"]["load"][string(load)]["qshed"]
            end
    push!(block_p, block_load_p)
    push!(block_q, block_load_q)

end
scatter!(pshed_block, collect(keys(block)), block_p, label="Integer Solution")
xlabel!(pshed_block, "Load Block")
ylabel!(pshed_block, "Load Shed (kW)")
title!(pshed_block, "Active Power Load Shed per Block ($fair_func)")

scatter!(qshed_block, collect(keys(block)), block_q, label="Integer Solution")
xlabel!(qshed_block, "Load Block")
ylabel!(qshed_block, "Load Shed (kW)")
title!(qshed_block, "Reactive Power Load Shed per Block ($fair_func)")


phases = 1:3
phase_labels = ["Phase A", "Phase B", "Phase C"]
markers = [:circle, :diamond, :utriangle]

plt = scatter(
    xlabel = "Switch ID",
    ylabel = "Voltage² (p.u.)",
    title = "Squared Voltage at Switch Buses ($fair_func)"
)

for (p, lab, m) in zip(phases, phase_labels, markers)
    scatter!(plt, switch_ids, [w[p] for w in fbus_w],
        label = "FBus $lab",
        marker = m
    )
    scatter!(plt, switch_ids, [w[p] for w in tbus_w],
        label = "TBus $lab",
        marker = m,
        linestyle = :dash
    )
end

# Voltage limits
plot!(plt, switch_ids, fill(0.9^2, length(switch_ids)),
      label = "Min Limit", line = (:dot, :black))
plot!(plt, switch_ids, fill(1.1^2, length(switch_ids)),
      label = "Max Limit", line = (:dot, :black))

display(plt)
savefig(plt, joinpath(exp_folder, "v_squared_switch_3phase_$fair_func.svg"))



# # Plot the switch power flow results
# scatter!(pshed_switch, switch_keys, switch_p, label="Integer Solution")
# scatter!(qshed_switch, switch_keys, switch_q, label="Integer Solution")

# Save all the plots and the tables into the results folder
savefig(pshed_block, joinpath(exp_folder,"pshed_block_$fair_func"))
savefig(qshed_block, joinpath(exp_folder,"qshed_block_$fair_func"))
# savefig(pshed_switch, joinpath(exp_folder, "pshed_switch_$fair_func"))
# savefig(qshed_switch, joinpath(exp_folder, "qshed_switch_$fair_func"))
CSV.write(joinpath(exp_folder,"block_summary_$fair_func"), block_df)
CSV.write(joinpath(exp_folder,"switch_summary_$fair_func"), switch_df)

# Calculate and plot fairness indices for the final solution
served = Float64[]
plot_fairness_indices(mld["solution"], collect(mld["solution"]["load"][string(i)]["pshed"] for i in 1:length(mld["solution"]["load"])), collect(parse.(Int,collect(keys(mld["solution"]["load"])))), iterations, fair_exp, fair_func)