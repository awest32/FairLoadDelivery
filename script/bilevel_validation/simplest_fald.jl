using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS, Juniper
using HSL_jll
using Random
using Distributions
using DiffOpt
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra, SparseArrays
using DataFrames
using CSV
using Dates

const CASE = "motivation_c"
const CASE_FILE = "ieee_13_aw_edit/$CASE.dss"
const LS_PERCENT = 0.8
const ITERATIONS = 5
const FAIR_FUNCS = "palma"#"efficiency", "proportional", "equality_min", "min_max", "jain"]#min_max throws error for motivation_c
const N_ROUNDS = 2
const N_BERNOULLI_SAMPLES = 6

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
    CASE_name = "results/$(today)/$CASE"
    if !isdir(CASE_name)
        mkdir(CASE_name)
    end

    # Create a folder for the control experiment results
    bilevel_exp_folder = "results/$(today)/$CASE/bilevel_exp"
    if !isdir(bilevel_exp_folder)
        mkdir(bilevel_exp_folder)
    end


fair_func = FAIR_FUNCS
# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = Gurobi.Optimizer
highs_solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT, [])

math_new = deepcopy(math)
fair_weights = Float64[]
for (load_id, load_dict) in math["load"]
    push!(fair_weights, load_dict["weight"])
end
    
pshed_lower_level = Float64[]
pshed_upper_level = Float64[]
final_weight_ids = Int[]
final_weights = Float64[]
for k in 1:ITERATIONS
    println("\n  --- Iteration $k ---")
    # Solve lower-level problem and get sensitivities
    dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, _ = lower_level_soln(math_new, fair_weights, 1)
    # Apply fairness function
    if fair_func == "proportional"
        pshed_new, fair_weight_vals = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals, math_new)
    elseif fair_func == "efficiency"
        pshed_new, fair_weight_vals = complete_efficiency_load_shed(dpshed, pshed_val, weight_vals, math_new)
    elseif fair_func == "min_max"
        pshed_new, fair_weight_vals = min_max_load_shed(dpshed, pshed_val, weight_vals)
    elseif fair_func == "equality_min"
        pshed_new, fair_weight_vals = FairLoadDelivery.equality_min(dpshed, pshed_val, weight_vals)
    elseif fair_func == "jain"
        pshed_new, fair_weight_vals = jains_fairness_index(dpshed, pshed_val, weight_vals)
    elseif fair_func == "palma"
        pd = Float64[]
        for (load_id, load_dict) in math_new["load"]
            push!(pd, sum(load_dict["pd"]))
        end
        pshed_new, fair_weight_vals = lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd)
    else
        error("Unknown fairness function: $fair_func")
    end

    # Update weights in math dict
    for (i, w) in zip(weight_ids, fair_weight_vals)
        math_new["load"][string(i)]["weight"] = w
    end

    push!(pshed_lower_level, sum(pshed_val))
    push!(pshed_upper_level, sum(pshed_new))

    println("    Lower-level shed: $(sum(pshed_val)), Upper-level shed: $(sum(pshed_new))")
    println("    Weights: $fair_weight_vals, with type: $(typeof(fair_weight_vals))")
end

mld_relaxed_final = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math_new, ipopt_solver; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])

    # Create a folder for the fairness experiment results
    fairness_exp_folder = "results/$(today)/$CASE/bilevel_exp/$fair_func"
    if !isdir(fairness_exp_folder)
        mkdir(fairness_exp_folder)
    end
plot_network_load_shed(mld_relaxed_final["solution"], math_new;
    output_file=joinpath(fairness_exp_folder, "FALD_relaxed.svg"),
    layout=:ieee13)





