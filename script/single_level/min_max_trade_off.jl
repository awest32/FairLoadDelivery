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
loadshed = zeros(alpha_points,length(ref[:load])+2)
JuMP.set_optimizer(mld_model_int.model, Gurobi.Optimizer)

for (index,alpha) in enumerate(LinRange(0,1,alpha_points))
    # set the objective for the min_max, efficiency trade-off with alpha*fairness + (1-alpha)*efficiency
    FairLoadDelivery.objective_min_max(mld_model_int; alpha=alpha)

    # solve the problem
    JuMP.optimize!(mld_model_int.model)
    # inspect the termination status
    status = JuMP.termination_status(mld_model_int.model)
    println("alpha=$alpha  status=$status")
    if JuMP.primal_status(mld_model_int.model) !=
    MOI.FEASIBLE_POINT
    println("Termination status is $status")
        break
    end
    loads = mld_model_int.sol[:it][:pmd][:nw][0][:load]
    switch = mld_model_int.sol[:it][:pmd][:nw][0][:switch]
    block =  mld_model_int.sol[:it][:pmd][:nw][0][:block]
    for (load_id, load_data) in loads
        loadshed[index, load_id] = sum(value.(load_data[:pshed]))
        loadshed[index, length(loads)+1] = sum(loadshed[index,1:length(loads)])
        loadshed[index, end] = alpha
    end
end

# Plot Pareto curve 
n=length(ref[:load])
total_shed = loadshed[:,n+1]
max_shed   = [maximum(loadshed[i, 1:n])  for i in 1:alpha_points]
alphas     = loadshed[:, end]

# Pareto curve: total shed (efficiency) vs max load shed (fairness)
p1 = plot(total_shed,max_shed,
    seriestype = :line,
    lc = :grey,
    marker = :circle,
    marker_z = alphas, colorbar_title = "alpha", color = :cividis,
    ylabel = "max load shed (fairness)",
    xlabel = "total load shed (efficiency)",
    title  = "Pareto front: fairness vs efficiency",
    legend = true)
# for i in eachindex(alphas)
#     annotate!(p1, total_shed[i], max_shed[i],
#             text(" a=$(round(alphas[i], digits=2))", 7, :left))
# end
plot!(p1)#, total_shed, max_shed)

# Metrics vs alpha
p2 = plot(alphas, total_shed, label="total shed (kW)", lw=2, marker=:circle,
        xlabel="alpha", ylabel=" load shed (kW)")
plot!(p2, alphas, max_shed, label="max load shed (kW)", lw=2, marker=:square)

plot(p1, p2, layout=(1,2), size=(900,400))