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

const CASE =  "motivation_c" #"pmonm_13_bus_mod"
const CASE_FILE = "ieee_13_aw_edit/$CASE.dss"
const LS_PERCENT = 0.8
const ITERATIONS = 5
const FAIR_FUNCS = "efficiency"#"efficiency", "proportional", "equality_min", "min_max", "jain"]#min_max throws error for motivation_c
const N_ROUNDS = 5
const N_BERNOULLI_SAMPLES = 50

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

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT, []);
