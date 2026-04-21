using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS
using JuMP
using LinearAlgebra, SparseArrays
using DataFrames
using CSV
using Dates
using Plots
using StatsPlots
using Statistics

# Load the test case
case = "ieee_13_aw_edit/motivation_c.dss"
dir = joinpath(@__DIR__,"../", "data")
file = joinpath(dir, case)

# Test parameters
gen_cap = 0.8
source_pu = 1.03
switch_rating = 600.0
critical_loads = []

# Load the data
eng, math = setup_network(file, gen_cap, source_pu, switch_rating, critical_loads)

# Solve the lower level problem
# Get initial weights
fair_weights = Float64[]
for (load_id, load) in math["load"]
    push!(fair_weights, load["weight"])
end

# Run initial lower-level solve
dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, ref, mld_paramed = lower_level_soln(math, fair_weights, 1)

# Test the upper level problem with the fixed inputs from the lower level solution
pshed_new_efficiency, fair_weight_vals_efficiency, status_efficiency = efficient_load_shed(dpshed, pshed_val, weight_vals)

pd = Float64[sum(math["load"][string(i)]["pd"]) for i in pshed_ids]
pshed_new_proportional, fair_weight_vals_proportional, status_proportional = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals, pd)
pshed_new_min_max, fair_weight_vals_min_max, status_min_max = min_max_load_shed(dpshed, pshed_val, weight_vals)
pshed_new_jain, fair_weight_vals_jain, status_jain = jains_fairness_index(dpshed, pshed_val, weight_vals)
pshed_new_palman, fair_weight_vals_palman, status_palman = lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd)

# Create assessment of the different in total load shed and the outputs of the fairness weights for this first iteration.
# Plot scatter plot of the weight values for each fairness function
fair_funcs = ["proportional", "efficiency", "min_max", "jain", "palma"]
weight_vals_df = DataFrame(
    load_id = weight_ids,
    proportional = fair_weight_vals_proportional,
    efficiency = fair_weight_vals_efficiency,
    min_max = fair_weight_vals_min_max,
    jain = fair_weight_vals_jain,
    palma = fair_weight_vals_palman
)
@df weight_vals_df scatter(:load_id, :proportional, label="Proportional", markershape=:circle)
@df weight_vals_df scatter!(:load_id, :efficiency, label="Efficiency", markershape=:square)
@df weight_vals_df scatter!(:load_id, :min_max, label="Min-Max", markershape=:diamond)
@df weight_vals_df scatter!(:load_id, :jain, label="Jain", markershape=:star)
@df weight_vals_df scatter!(:load_id, :palma, label="Palma", markershape=:hexagon)
xlabel!("Load ID")

# Plot bar plot of total load shed difference from the lower level solution for each fairness function
pshed_diff_df = DataFrame(
    fairness_function = fair_funcs,
    total_pshed = [sum(pshed_new_proportional), sum(pshed_new_efficiency), sum(pshed_new_min_max), sum(pshed_new_jain), sum(pshed_new_palman)],
    pshed_diff = [sum(pshed_new_proportional) - sum(pshed_val), sum(pshed_new_efficiency) - sum(pshed_val), sum(pshed_new_min_max) - sum(pshed_val), sum(pshed_new_jain) - sum(pshed_val), sum(pshed_new_palman) - sum(pshed_val)]
)
@df pshed_diff_df bar(:fairness_function, :pshed_diff, label="Difference in Total Load Shed", legend=:top)
xlabel!("Fairness Function")

# Extract the switch states from the lower level solution and plot their values.
# Note: `mld_paramed.sol` is only populated by the `solve_*` wrapper; since we called
# `instantiate_mc_model` + `optimize!` directly, read the JuMP variables from `.var`.
switch_state_ids = mld_paramed.sol[:it][:pmd][:nw][0][:switch]
switch_ids_sorted = sort(collect(keys(switch_state_ids)))
switch_states_df = DataFrame(
    switch_id = switch_ids_sorted,
    state = Float64[JuMP.value(mld_paramed.var[:it][:pmd][:nw][0][:switch_state][sid]) for sid in switch_ids_sorted],
)
@df switch_states_df scatter(:switch_id, :state,
    label="Switch state", markershape=:circle,
    xlabel="Switch ID", ylabel="State (relaxed)",
    title="Lower-level Switch States",
    xticks=switch_ids_sorted)