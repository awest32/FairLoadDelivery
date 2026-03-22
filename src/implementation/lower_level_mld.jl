
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
#eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.5, [])

function diff_forward_full_jacobian(model::JuMP.Model, fair_load_weights::Vector{Float64})
    weight_params = model[:fair_load_weights]
    pshed_vars = model[:pshed]
    #pserved_vars = model[:pd]
    
    weight_keys = (collect(eachindex(weight_params)))
    pshed_keys = (collect(eachindex(pshed_vars)))
    #pserved_keys = (collect(eachindex(pserved_vars)))
    weight_ids = collect(axes(weight_params, 1))
    pshed_ids = collect(axes(pshed_vars, 1))
    #pserved_ids = collect(axes(pserved_vars, 1))

    n_weights = length(weight_keys)
    n_pshed = length(pshed_keys)
    #n_pserved = length(pserved_keys)

    # Solve once — perturbations only affect differentiation direction, not the optimal solution
    optimize!(model)

    # Build Jacobian column by column
    jacobian = zeros(n_pshed, n_weights)
    for j in 1:n_weights
        # Clear previous perturbation before setting next direction
        DiffOpt.empty_input_sensitivities!(model)

        # Perturb ONLY weight j (standard basis vector e_j)
        # Use 1.0 (not fair_load_weights[j]) so that get_forward_variable
        # returns the true partial derivative ∂pshed/∂w_j
        for (k, wkey) in enumerate(weight_keys)
            perturbation = (k == j) ? 1.0 : 0.0
            DiffOpt.set_forward_parameter(model, weight_params[wkey], perturbation)
        end

        # Compute derivatives
        DiffOpt.forward_differentiate!(model)

        # Extract column j: [∂pshed[1]/∂weight[j], ∂pshed[2]/∂weight[j], ...]
        for (i, pkey) in enumerate(pshed_keys)
            jacobian[i, j] = DiffOpt.get_forward_variable(model, pshed_vars[pkey])
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

    mld_paramed = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_shedding_implicit_diff; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
    ref = mld_paramed.ref[:it][:pmd][:nw][0]


    # On the first iteration, use the weights from the math dictionary.
    # On subsequent iterations, use the weights from the upper-level fairness function.
    if k == 1
        load_ids_sorted = sort(parse.(Int, collect(keys(math["load"]))))
        weights_prev = Float64[math["load"][string(i)]["weight"] for i in load_ids_sorted]
    else
        weights_prev = weights_new
    end
    @info "Iteration $k: Solving lower-level MLD with weights: $weights_prev, with type: $(typeof(weights_prev))"
    # Use the parameterized MLD solution to perform implicit differentiation with DiffOpt.jl
    dpshed_mat, pshed_val, pshed_ids, weight_vals, weight_ids, ref = diff_forward_full_jacobian(mld_paramed.model, weights_prev)
    return dpshed_mat, pshed_val, pshed_ids, weight_vals, weight_ids, ref
end

# dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math, 10*ones(length(math["load"])), 1)
# mld_relaxed = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)
# mld_relaxed["solution"]["switch"]
"""
Multiperiod Jacobian computation via DiffOpt forward differentiation.
Returns Jacobian of shape (T*N) x N where:
  - Rows: pshed for each load in each period (flattened: period 0 loads, period 1 loads, ...)
  - Columns: global fair_load_weights (N weights)
Also returns flattened pshed values, pshed IDs (tuples of (nw, load_id)),
weight values, weight IDs, and the model.
"""
function diff_forward_full_jacobian_mn(model::JuMP.Model, fair_load_weights::Vector{Float64})
    weight_params = model[:fair_load_weights]
    nw_ids = model[:nw_ids]

    weight_keys = collect(eachindex(weight_params))
    weight_ids = collect(axes(weight_params, 1))
    n_weights = length(weight_keys)

    # Collect all pshed variables across periods (flattened)
    all_pshed_vars = []      # JuMP variable references
    all_pshed_keys = []      # keys for indexing
    all_pshed_nw_ids = []    # (nw, load_id) tuples for identification

    for n in nw_ids
        pshed_nw = model[Symbol("pshed_nw_$(n)")]
        pshed_keys_nw = collect(eachindex(pshed_nw))
        pshed_ids_nw = collect(axes(pshed_nw, 1))
        for (key, lid) in zip(pshed_keys_nw, pshed_ids_nw)
            push!(all_pshed_vars, pshed_nw[key])
            push!(all_pshed_keys, (n, key))
            push!(all_pshed_nw_ids, (n, lid))
        end
    end

    n_pshed_total = length(all_pshed_vars)

    # Solve once — perturbations only affect differentiation direction, not the optimal solution
    optimize!(model)

    # Build Jacobian column by column: (T*N) x N
    jacobian = zeros(n_pshed_total, n_weights)
    for j in 1:n_weights
        DiffOpt.empty_input_sensitivities!(model)

        # Perturb ONLY weight j (standard basis vector e_j)
        # Use 1.0 (not fair_load_weights[j]) so that get_forward_variable
        # returns the true partial derivative ∂pshed/∂w_j
        for (k, wkey) in enumerate(weight_keys)
            perturbation = (k == j) ? 1.0 : 0.0
            DiffOpt.set_forward_parameter(model, weight_params[wkey], perturbation)
        end

        DiffOpt.forward_differentiate!(model)

        # Extract column j: [∂pshed_t0_l1/∂w_j, ∂pshed_t0_l2/∂w_j, ..., ∂pshed_t1_l1/∂w_j, ...]
        for (i, var) in enumerate(all_pshed_vars)
            jacobian[i, j] = DiffOpt.get_forward_variable(model, var)
        end
    end

    # Collect pshed values (flattened across periods)
    pshed_vals = Float64[JuMP.value(var) for var in all_pshed_vars]

    return jacobian, pshed_vals, all_pshed_nw_ids, Array(JuMP.value.(weight_params)), weight_ids, model
end

"""
Multiperiod lower-level solution: instantiates the multiperiod implicit diff model,
sets weights, computes Jacobian via DiffOpt.
Returns: dpshed_mat (T*N x N), pshed_val (T*N), pshed_nw_ids, weight_vals, weight_ids, refs
"""
function lower_level_soln_mn(mn_data::Dict{String,Any}, weights_new, k)
    mld_paramed = instantiate_mc_model(
        mn_data,
        LinDist3FlowPowerModel,
        build_mn_mc_mld_shedding_implicit_diff;
        multinetwork=true,
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!]
    )

    nw_ids = sort(collect(_PMD.nw_ids(mld_paramed)))
    first_nw = nw_ids[1]
    refs = Dict(n => mld_paramed.ref[:it][:pmd][:nw][n] for n in nw_ids)

    # On first iteration, use weights from the data; subsequently use updated weights
    if k == 1
        first_nw_data = mn_data["nw"][string(first_nw)]
        load_ids_sorted = sort(parse.(Int, collect(keys(first_nw_data["load"]))))
        weights_prev = Float64[first_nw_data["load"][string(i)]["weight"] for i in load_ids_sorted]
    else
        weights_prev = weights_new
    end

    @info "Iteration $k: Solving multiperiod lower-level MLD with weights: $weights_prev"

    dpshed_mat, pshed_val, pshed_nw_ids, weight_vals, weight_ids, _ = diff_forward_full_jacobian_mn(mld_paramed.model, weights_prev)
    return dpshed_mat, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs
end

function plot_dpshed_heatmap(dpshed, pshed_ids, weight_ids, k, save_path)
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
    Plots.annotate!(16.5, 16, Plots.text("∂P/∂w", 9, "Computer Modern"))


    # Save the heatmap
    savefig(heatmap_plot, "$save_path/load_shed_sensitivities_heatmap_k$(k).svg")  # save as SVG
    println("Heatmap saved as load_shed_sensitivities_heatmap_k$(k).svg")
end

# Plot the load shed per bus  with the buses sorted by distance from the sourcebus
function plot_load_shed_per_bus(pshed_val, pshed_ids, k, save_path)
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
    savefig(bar_plot, "$save_path/load_shedding_per_load_k$(k).svg")  # save as SVG
    println("Bar plot saved as load_shedding_per_load_k$(k).svg")
end
