
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
# using DataFrames
# ipopt = Ipopt.Optimizer
# gurobi = Gurobi.Optimizer

# ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
# highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# Inputs: case file path, percentage of load shed, list of critical load IDs
#eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.5, ["675a"])

function diff_forward_full_jacobian(model::JuMP.Model, fair_load_weights::Vector{Float64})
    weight_params = model[:fair_load_weights]
    pshed_vars = model[:pshed]
    
    weight_keys = (collect(eachindex(weight_params)))
    pshed_keys = (collect(eachindex(pshed_vars)))
    weight_ids = collect(axes(weight_params, 1))
    pshed_ids = collect(axes(pshed_vars, 1))

    n_weights = length(weight_keys)
    n_pshed = length(pshed_keys)
    
    # Get current pshed
   # pshed = [JuMP.value(pshed_vars[k]) for k in pshed_keys]
    
    # Build Jacobian column by colum
    
    jacobian = zeros(n_pshed, n_weights)
    #dpshed = Dict{Any,Any}()
    for j in 1:n_weights
        # Perturb ONLY weight j (standard basis vector e_j)
        for (k, wkey) in enumerate(weight_keys)
            perturbation = (k == j) ? fair_load_weights[j] : 0.0
            DiffOpt.set_forward_parameter(model, weight_params[wkey], perturbation)
        end

        optimize!(model)

        # Compute derivatives
        DiffOpt.forward_differentiate!(model)
        
        # Extract column j: [∂pshed[1]/∂weight[j], ∂pshed[2]/∂weight[j], ...]
        for (i, pkey) in enumerate(pshed_keys)
            jacobian[i, j] = DiffOpt.get_forward_variable(model, pshed_vars[pkey])
            #dpshed[i,pkey] = jacobian[i, j]
        end
    end
    
    return jacobian, Array(value.(pshed_vars)), pshed_ids, Array(value.(weight_params)), weight_ids, model
end

function lower_level_soln(math, weights_new, k)
    # Solve the parameterized MLD 
    # mld_paramed_soln = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math, ipopt)
    # # Extract the pshed from the loads in the solution
    # pshed = Dict{String,Float64}()
    # for (load_id, load) in (mld_paramed_soln["solution"]["load"])
    #     pshed[string(load_id)] = load["pshed"]
    # end

    mld_paramed = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_shedding_implicit_diff; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
    ref = mld_paramed.ref[:it][:pmd][:nw][0]


    # Initial weights (iteration 0)
    if k == 1
        weights_prev = ones(length(ref[:load])) * 10.0
    else
        weights_prev = weights_new
    end

    # Use the parameterized MLD solution to perform implicit differentiation with DiffOpt.jl 
    dpshed_mat, pshed_val, pshed_ids, weight_vals, weight_ids, model = diff_forward_full_jacobian(mld_paramed.model, weights_prev)
    return dpshed_mat, pshed_val, pshed_ids, weight_vals, weight_ids, ref
end

#dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math, ipopt)
function plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids, k)
    # Labels for rows/columns (use pshed keys)
    # original ordering
    # keys_paramed = [string(k[1]) for k in keys(mld_paramed.model[:fair_load_weights])]
    # row_labels = keys_paramed  # loads
    # col_labels = keys_paramed  # weights

    # Plot heatmap
    heatmap_plot = heatmap(
        string.(weight_ids), string.(pshed_ids), dpshed;
        xlabel = "Load Weight ID",
        ylabel = "Load Shed ID",
        title = "Sensitivity of Load Shedding (dP/dw)",
        c = :viridis,
        #colorbar_title = "∂P/∂w",
        size = (600, 500),
        right_margin = 5Plots.mm,
        top_margin = 5Plots.mm
    )
    annotate!(16.5, 16, text("∂P/∂w", 9, "Computer Modern"))


    # Save the heatmap
    savefig(heatmap_plot, "load_shed_sensitivities_heatmap_k$(k).svg")  # save as SVG
    println("Heatmap saved as load_shed_sensitivities_heatmap_k$(k).svg")
end

# Plot the load shed per bus  with the buses sorted by distance from the sourcebus
function plot_load_shed_per_bus(pshed_val, pshed_ids, k)
    bar_plot = bar(
        string.(pshed_ids), pshed_val;
        xlabel = "Load ID",
        ylabel = "Load Shedding (p.u.)",
        title = "Load Shedding per Load ID",
        legend = false,
        size = (600, 400),
        right_margin = 5Plots.mm,
        top_margin = 5Plots.mm
    )
    savefig(bar_plot, "load_shedding_per_load_k$(k).svg")  # save as SVG
    println("Bar plot saved as load_shedding_per_load_k$(k).svg")
end
#plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids)

# Solve the MILP MLD problem
# mld_mip_soln = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)

# Extract the fixed variables from the MLD solution
# math_mip = deepcopy(math)
# for (switch_id, switch) in enumerate(math_mip["switch"])
#     #@info "Setting switch $switch_id to state to $(mld_mip_soln["solution"]["switch"][string(switch_id)]["state"]) from MIP solution."
#     switch_var = mld_mip_soln["solution"]["switch"][string(switch_id)]["state"]
#     #@info "Switch variable before rounding: $switch_var"
#     math_mip["switch"][string(switch_id)]["state"] = round(Int, switch_var)
# end
# for (block_id, block) in enumerate(math_mip["block"])
#     block_var = mld_mip_soln["solution"]["block"][string(block_id)]["status"]
#     math_mip["block"][string(block_id)]["state"] = round(Int, block_var)
# end

# # Solve the AC OPF MLD with fixed discrete variables from the MIP solution
# #ac_mld_fixed_soln = FairLoadDelivery.solve_mc_pf_aw(math_mip, ipopt)

# # Solve the continues rounded mld using the mip solutions
# mld_fixed_soln = FairLoadDelivery.solve_mc_mld_shed_random_round(math_mip, ipopt)

# #print the switch and load block states
# println("Switch states from MLD MIP solution:")
# for (switch_id, switch) in (math_mip["switch"])
#     println("Switch $switch_id state: $(switch["state"])")
# end
# println("Load block states from MLD MIP solution:")
# for (block_id, block) in (math_mip["block"])
#     println("Block $block_id state: $(block["state"])")
# end
