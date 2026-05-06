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
case_name = "../../data/pmd_opendss/case6_unbalanced_switch_meshed_good4integer.dss"
#case_name = "../../data/ieee_13_aw_edit/motivation_c.dss"
dir = @__DIR__
case_path = joinpath(dir,case_name)
date = Dates.format(now(), "yyyy-mm-dd")  
LS_PERCENT = 0.8
pshed_type = "proportional"  # "absolute" or "proportional"
min_max_obj = pshed_type == "proportional" ? FairLoadDelivery.objective_min_max_proportional :
                                             FairLoadDelivery.objective_min_max_absolute
eng,math,lbs, critical_id  = setup_network(case_path, LS_PERCENT; switch_rating=sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT)
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_min_max; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
mld_model_int = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_min_max_integer; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
ref = mld_model.ref[:it][:pmd][:nw][0]
#pf_soln = PowerModelsDistribution.solve_mc_pf(math, ACRUPowerModel, Ipopt.Optimizer)
# set alpha sweep for the functions
alpha_points = 10
loadshed = zeros(alpha_points,length(ref[:load])+2)

output_dir = joinpath(@__DIR__, "../../results/$date/trade_off")
if !isdir(output_dir)
    mkpath(output_dir)
end

# Integer check
loadshed = zeros(alpha_points,length(ref[:load])+2)
JuMP.set_optimizer(mld_model_int.model, Gurobi.Optimizer)
loadshed_keys = []

for (index,alpha) in enumerate(LinRange(0,1,alpha_points))
    # set the objective for the min_max, efficiency trade-off with alpha*fairness + (1-alpha)*efficiency
    min_max_obj(mld_model_int; alpha=alpha)

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
    push!(loadshed_keys,keys(loads))
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
p3 = plot(total_shed,max_shed,label="solution (kW)",
    seriestype = :line,
    lc = :grey,
    marker = :circle,
    marker_z = alphas, colorbar_title = "alpha", color = :cividis,
    ylabel = "max load shed (fairness)",
    xlabel = "total load shed (efficiency)",
    #title  = "Pareto front: integer problem fairness vs efficiency",
    legend = true)

plot!(p3)#, total_shed, max_shed)
# Metrics vs alpha
p4 = plot(alphas, total_shed, label="total shed (kW)", lw=2, marker=:circle,
        xlabel="alpha", ylabel=" load shed (kW)")
plot!(p4, alphas, max_shed, label="max load shed (kW)", lw=2, marker=:square)


savefig(plot(p3, p4, layout=(1,2), size=(900,400)),
          joinpath(output_dir, "pareto_summary_integer_$(pshed_type).svg"))

n = length(ref[:load])
load_labels = [load_data["name"] for (id, load_data) in sort(ref[:load])]

function build_dist_plot(pshed_per_load, title_str)
    p = bar(load_labels, pshed_per_load,
        xlabel = "load ID",
        ylabel = "load shed (kW)",
        title  = title_str,
        legend = false,
        color  = :steelblue,
        linecolor = :black,
    )
    for (i, v) in enumerate(pshed_per_load)
        annotate!(p, i, v + maximum(pshed_per_load)*0.02,
                text("$(round(v, digits=1))", 8, :center))
    end
    return p
end

p_dist_a0 = build_dist_plot(loadshed[1, 1:n],   "alpha = 0 (efficiency)")
p_dist_a1 = build_dist_plot(loadshed[end, 1:n], "alpha = 1 (fairness)")

savefig(p_dist_a0, joinpath(output_dir, "loadshed_distribution_integer_alpha0.svg"))
savefig(p_dist_a1, joinpath(output_dir, "loadshed_distribution_integer_alpha1.svg"))

combined = plot(p_dist_a0, p_dist_a1, p4, p3, layout=(2,2), size=(1400, 900),
    left_margin=10Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=10Plots.mm)
savefig(combined, joinpath(output_dir, "summary_integer_all_$(pshed_type).svg"))
display(combined)

JuMP.set_optimizer(mld_model.model, Gurobi.Optimizer)
for (index,alpha) in enumerate(LinRange(0,1,alpha_points))
    # set the objective for the min_max, efficiency trade-off with alpha*fairness + (1-alpha)*efficiency
    min_max_obj(mld_model; alpha=alpha)

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
    title  = "Pareto front: relaxed fairness vs efficiency",
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
savefig(plot(p1, p2, layout=(1,2), size=(900,400)),
          joinpath(output_dir, "pareto_summary_relaxed_$(pshed_type).svg"))