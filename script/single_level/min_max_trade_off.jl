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
using Statistics
using PowerPlots
using DataFrames
using CSV
using Plots
using Dates

include("../../src/implementation/visualization.jl")

# Set the network path
#case_name = "../../data/pmd_opendss/case6_unbalanced_switch_meshed_good4integer.dss"
case_name = "../../data/pmd_opendss/case6_unbalanced_switch_more_meshed_good4integer.dss"
#case_name = "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss"

case ="more_meshed_6bus"
dir = @__DIR__
case_path = joinpath(dir,case_name)
date = Dates.format(now(), "yyyy-mm-dd")  
LS_PERCENT = 0.8
pshed_type = "absolute"  # "absolute" or "proportional"
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
load_summary_path = joinpath(output_dir, "load_summary_$(pshed_type)_$case.csv")
isfile(load_summary_path) && rm(load_summary_path)

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
    pshed_by_load = Dict(lid => sum(value.(ld[:pshed])) for (lid, ld) in loads)
    append_load_summary!(load_summary_path,
        load_summary_rows(math, pshed_by_load; extra=(stage="integer", alpha=alpha)))
end

# ----------------------------------------------------------------------------
# Per-α aggregates and per-load-shed-vector norms
# ----------------------------------------------------------------------------
n          = length(ref[:load])
total_shed = loadshed[:, n + 1]
max_shed   = [maximum(loadshed[i, 1:n])  for i in 1:alpha_points]
alphas     = loadshed[:, end]

# Norms of the per-load shed vector x = [pshed_i]_i (kW).
# CoV = std/mean; NaN when mean ≈ 0 (the α=0 corner may shed essentially
# nothing on some loads, but total_shed ≈ 0 only if the problem is trivially
# feasible — kept defensive so the plot doesn't blow up).
function shed_norms(shed_vec::AbstractVector{<:Real})
    m = mean(shed_vec)
    s = std(shed_vec)
    return (
        l1   = norm(shed_vec, 1),
        l2   = norm(shed_vec, 2),
        linf = norm(shed_vec, Inf),
        cov  = m > 1e-9 ? s / m : NaN,
    )
end

norms_per_alpha = [shed_norms(loadshed[i, 1:n]) for i in 1:alpha_points]
l1_vec   = [nm.l1   for nm in norms_per_alpha]
l2_vec   = [nm.l2   for nm in norms_per_alpha]
linf_vec = [nm.linf for nm in norms_per_alpha]
cov_vec  = [nm.cov  for nm in norms_per_alpha]

# ----------------------------------------------------------------------------
# Figure 1: load-shed distribution at α=0, at α=1, and total+max-shed vs α
# ----------------------------------------------------------------------------
load_labels = [load_data["name"] for (id, load_data) in sort(ref[:load])]

const FONT_KW = (tickfontsize = 16, guidefontsize = 22,
                 titlefontsize = 18, legendfontsize = 16)

function build_dist_plot(pshed_per_load, title_str)
    p = bar(load_labels, pshed_per_load,
        xlabel = "load ID",
        ylabel = "load shed (kW)",
        title  = title_str,
        legend = false,
        color  = :steelblue,
        linecolor = :black;
        FONT_KW...,
    )
    ymax = maximum(pshed_per_load)
    for (i, v) in enumerate(pshed_per_load)
        annotate!(p, i, v + (ymax > 0 ? ymax : 1.0) * 0.02,
            text("$(round(v, digits = 1))", 14, :center))
    end
    return p
end

p_dist_a0 = build_dist_plot(loadshed[1, 1:n],   "alpha = 0 (efficiency)")
p_dist_a1 = build_dist_plot(loadshed[end, 1:n], "alpha = 1 (fairness)")

savefig(p_dist_a0, joinpath(output_dir, "loadshed_distribution_integer_alpha0.svg"))
savefig(p_dist_a1, joinpath(output_dir, "loadshed_distribution_integer_alpha1.svg"))

p_metrics = plot(alphas, total_shed, label = "total shed (kW)",
    lw = 2, marker = :circle, xlabel = "alpha", ylabel = "load shed (kW)",
    title = "Total + max per-load shed vs alpha"; FONT_KW...)
plot!(p_metrics, alphas, max_shed, label = "max load shed (kW)",
    lw = 2, marker = :square)

fig1 = plot(p_dist_a0, p_dist_a1, p_metrics,
    layout = (1, 3), size = (1900, 600),
    left_margin = 14Plots.mm, right_margin = 6Plots.mm,
    top_margin = 8Plots.mm, bottom_margin = 14Plots.mm)
savefig(fig1, joinpath(output_dir, "summary_integer_$(pshed_type)_$case.svg"))
display(fig1)

# ----------------------------------------------------------------------------
# Figure 2: Pareto fronts (total shed vs L1 / L2 / L∞ / CoV of shed vector),
# α encoded by marker color. Colorbar lives in a dedicated narrow subplot so
# the four data panels stay equally sized.
# ----------------------------------------------------------------------------
function pareto_norm_plot(total_shed_vec, norm_vec, alphas_vec, ylab)
    plot(total_shed_vec, norm_vec,
        seriestype = :line, lc = :grey,
        marker = :circle, marker_z = alphas_vec, color = :cividis,
        clims = (0.0, 1.0), colorbar = false,
        xlabel = "total load shed (kW)", ylabel = ylab,
        legend = false; FONT_KW...)
end

p_l1   = pareto_norm_plot(total_shed, l1_vec,   alphas, "L1 norm of shed (kW)")
p_l2   = pareto_norm_plot(total_shed, l2_vec,   alphas, "L2 norm of shed (kW)")
p_linf = pareto_norm_plot(total_shed, linf_vec, alphas, "L∞ norm of shed (kW)")
p_cov  = pareto_norm_plot(total_shed, cov_vec,  alphas, "CoV (stdev/mean)")

# Dedicated colorbar strip: heatmap of α∈[0,1] in :cividis, ticked at 0/0.5/1.
p_cbar = heatmap(reshape(collect(LinRange(0.0, 1.0, 256)), :, 1);
    color = :cividis, colorbar = false,
    xticks = false, yticks = ([1, 128, 256], ["0", "0.5", "1"]),
    ylabel = "alpha", title = "", framestyle = :box,
    tickfontsize = 16, guidefontsize = 22)

fig2 = plot(p_l1, p_l2, p_linf, p_cov, p_cbar,
    layout = @layout([a b c d e{0.02w}]),
    size = (2200, 600),
    left_margin = 14Plots.mm, right_margin = 6Plots.mm,
    top_margin = 8Plots.mm, bottom_margin = 14Plots.mm)
savefig(fig2, joinpath(output_dir, "pareto_norms_integer_$(pshed_type)_$case.svg"))
display(fig2)

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
    pshed_by_load = Dict(lid => sum(value.(ld[:pshed])) for (lid, ld) in loads)
    append_load_summary!(load_summary_path,
        load_summary_rows(math, pshed_by_load; extra=(stage="relaxed", alpha=alpha)))
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
          joinpath(output_dir, "pareto_summary_relaxed_$(pshed_type)_$case.svg"))