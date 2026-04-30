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

include("../../src/implementation/visualization.jl")

# Set the network path
case_name = "../../data/pmd_opendss/case4_unbalanced_switch.dss"
dir = @__DIR__
case_path = joinpath(dir,case_name)

eng,math = setup_network(case_path, 0.8)
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_min_max; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
mld_model_int = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_min_max_integer; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
ref = mld_model.ref[:it][:pmd][:nw][0]

# set alpha sweep for the functions
alpha_points = 20
loadshed = zeros(alpha_points,length(ref[:load])+1)
for (index,alpha) in enumerate(LinRange(0,1,alpha_points))
    # set the objective for the min_max, efficiency trade-off with alpha*efficiency + (1-alpha)*fairness
    FairLoadDelivery.objective_min_max(mld_model; alpha=alpha)

    JuMP.set_optimizer(mld_model.model, Gurobi.Optimizer)
    # solve the problem
    JuMP.optimize!(mld_model.model)
    # inspect the termination status
    status = JuMP.termination_status(mld_model.model)
    println("alpha=$alpha  status=$status")
    if JuMP.primal_status(mld_model.model) !=
    MOI.FEASIBLE_POINT
    println("Termination status is $status")
        break
    end
    loads = mld_model.sol[:it][:pmd][:nw][0][:load]
    switch = mld_model.sol[:it][:pmd][:nw][0][:switch]
    block =  mld_model.sol[:it][:pmd][:nw][0][:block]
    for (load_id, load_data) in loads
        loadshed[index, end] = alpha
        loadshed[index, load_id] = sum(value.(load_data[:pd]))
    end
end

