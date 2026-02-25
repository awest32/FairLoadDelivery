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
const FAIR_FUNC = "efficiency"  # simplest fairness function for testing
const N_ROUNDS = 2
const N_BERNOULLI_SAMPLES = 6

# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = Gurobi.Optimizer
highs_solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT, [])

pd=[]
all_pshed_lower = []
all_pshed_upper = []
math_new = deepcopy(math)
for k in 1:ITERATIONS
    global fair_weights_k
    println("\n  --- Iteration $k ---")
  
    # Solve lower-level
    dpshed_k, pshed_val_k, pshed_ids_k, weight_vals_k, weight_ids_k, model_k = lower_level_soln(math_new, fair_weights, k)

    # Apply upper-level fairness function
    pshed_new, fair_weight_vals = complete_efficiency_load_shed(dpshed_k, pshed_val_k, weight_vals_k, math_new)

    for i in pshed_ids_k
        push!(pd, sum(math_new["load"][string(i)]["pd"]))
    end

    # Update weights in math dict
    math_before_update = deepcopy(math_new)
    for (i, w) in zip(weight_ids_k, fair_weight_vals)
        math_new["load"][string(i)]["weight"] = w
    end

    push!(all_pshed_lower, sum(pshed_val_k))
    push!(all_pshed_upper, sum(pshed_new))

    fair_weights_k = fair_weight_vals  # Update for next iteration

    println("    Lower-level shed: $(sum(pshed_val_k)), Upper-level shed: $(sum(pshed_new))")
    println("    Weights: $fair_weight_vals")
end
