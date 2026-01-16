#=
OpenDSS Network Experiment: Reformulated Palma Ratio Optimization
=================================================================

This script integrates the reformulated Palma ratio optimization with real
OpenDSS network data from PowerModelsDistribution.jl.

Workflow:
1. Parse OpenDSS (.dss) file using PowerModelsDistribution
2. Set up network with load shedding capacity constraint
3. Run bilevel optimization iterations:
   - Lower level: Solve MLD to get pshed and compute Jacobian via DiffOpt
   - Upper level: Optimize Palma ratio using reformulated model
4. Generate plots and export results

Run with:
    julia --project=. script/reformulation/opendss_experiment.jl

Author: Claude (with guidance from Sam)
Date: 2026-01-14
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Revise
using FairLoadDelivery
using PowerModelsDistribution
using Ipopt, Gurobi
using JuMP
using DiffOpt
using LinearAlgebra
using Plots
using DataFrames
using CSV
using Dates
using Printf
using Random

# Include the reformulated model and network setup
include("load_shed_as_parameter.jl")
include(joinpath(@__DIR__, "..", "..", "src", "implementation", "network_setup.jl"))

# Solver configurations
const ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
const gurobi_solver = Gurobi.Optimizer

#=============================================================================
 Core Functions: Lower Level Solver with DiffOpt
=============================================================================#

"""
    compute_jacobian_diffopt(math::Dict, weights::Vector{Float64})

Solve the lower-level MLD problem and compute the Jacobian ∂pshed/∂w
using DiffOpt implicit differentiation.

Returns: (jacobian, pshed_values, pshed_ids, weight_ids, model_ref)
"""
function compute_jacobian_diffopt(math::Dict, weights::Vector{Float64})
    # Instantiate parameterized MLD model
    mld_model = instantiate_mc_model(
        math,
        LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
    )

    model = mld_model.model
    ref = mld_model.ref[:it][:pmd][:nw][0]

    # Get variable references
    weight_params = model[:fair_load_weights]
    pshed_vars = model[:pshed]

    weight_keys = collect(eachindex(weight_params))
    pshed_keys = collect(eachindex(pshed_vars))
    weight_ids = collect(axes(weight_params, 1))
    pshed_ids = collect(axes(pshed_vars, 1))

    n_weights = length(weight_keys)
    n_pshed = length(pshed_keys)

    # Build Jacobian column by column using forward-mode differentiation
    jacobian = zeros(n_pshed, n_weights)

    for j in 1:n_weights
        # Set unit perturbation for weight j (must be 1.0 for correct Jacobian)
        for (k, wkey) in enumerate(weight_keys)
            perturbation = (k == j) ? 1.0 : 0.0  # Unit vector in direction j
            DiffOpt.set_forward_parameter(model, weight_params[wkey], perturbation)
        end

        optimize!(model)
        DiffOpt.forward_differentiate!(model)

        # Extract column j of Jacobian
        for (i, pkey) in enumerate(pshed_keys)
            jacobian[i, j] = DiffOpt.get_forward_variable(model, pshed_vars[pkey])
        end
    end

    pshed_values = Array(value.(pshed_vars))

    return jacobian, pshed_values, pshed_ids, weight_ids, ref
end

#=============================================================================
 Main Interface: solve_palma_ratio_minimization
=============================================================================#

"""
    solve_palma_ratio_minimization(
        network_file::String;
        ls_percent::Float64 = 0.9,
        critical_loads::Vector{String} = String[],
        iterations::Int = 10,
        trust_radius::Float64 = 0.1,
        output_dir::String = "script/reformulation/experiments",
        experiment_name::String = "palma_experiment",
        plot_results::Bool = true,
        verbose::Bool = true
    )

Run the reformulated Palma ratio bilevel optimization on an OpenDSS network.

# Arguments
- `network_file::String`: Path to OpenDSS .dss file (relative to data/ folder)
- `ls_percent::Float64`: Load shedding capacity as fraction of total load (default 0.9)
- `critical_loads::Vector{String}`: List of critical load names (default empty)
- `iterations::Int`: Number of bilevel iterations (default 10)
- `trust_radius::Float64`: Trust region radius for weight updates (default 0.1)
- `output_dir::String`: Directory for output files (default "script/reformulation/experiments")
- `experiment_name::String`: Name prefix for output files (default "palma_experiment")
- `plot_results::Bool`: Whether to generate plots (default true)
- `verbose::Bool`: Print progress information (default true)

# Returns
NamedTuple with:
- `math`: Final math dictionary with updated weights
- `palma_history`: Vector of Palma ratios per iteration
- `pshed_history`: Vector of load shed vectors per iteration
- `weights_history`: Vector of weight vectors per iteration
- `output_path`: Path to output directory
"""
function solve_palma_ratio_minimization(
    network_file::String;
    ls_percent::Float64 = 0.9,
    critical_loads::Vector{String} = String[],
    iterations::Int = 10,
    trust_radius::Float64 = 0.1,
    output_dir::String = joinpath(@__DIR__, "experiments"),
    experiment_name::String = "palma_experiment",
    plot_results::Bool = true,
    verbose::Bool = true
)
    # Create timestamped output directory (day_hour format only)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH")
    output_path = joinpath(output_dir, "$(experiment_name)_$(timestamp)")
    mkpath(output_path)

    verbose && println("="^70)
    verbose && println("REFORMULATED PALMA RATIO OPTIMIZATION")
    verbose && println("="^70)
    verbose && println("Network file: $network_file")
    verbose && println("Load shed capacity: $(ls_percent * 100)%")
    verbose && println("Iterations: $iterations")
    verbose && println("Trust radius: $trust_radius")
    verbose && println("Output path: $output_path")
    verbose && println()

    #=========================================================================
    # Step 1: Parse and setup network
    =========================================================================#
    verbose && println("Step 1: Setting up network...")

    eng, math, lbs, critical_id = setup_network(network_file, ls_percent, critical_loads)

    # Extract initial weights and load demands
    n_loads = length(math["load"])
    load_ids = sort(parse.(Int, collect(keys(math["load"]))))

    weights = Float64[]
    pd = Float64[]
    for id in load_ids
        load = math["load"][string(id)]
        push!(weights, load["weight"])
        push!(pd, sum(load["pd"]))  # Sum across phases
    end

    verbose && println("  Number of loads: $n_loads")
    verbose && println("  Number of load blocks: $(length(lbs))")
    verbose && println("  Initial weights: $(weights[1]) (all equal)")
    verbose && println()

    #=========================================================================
    # Step 2: Run bilevel iterations
    =========================================================================#
    verbose && println("Step 2: Running bilevel optimization...")
    verbose && println()
    verbose && println("Iteration  Palma Ratio   Total Shed    Solve Time")
    verbose && println("---------  -----------   ----------    ----------")

    # History storage
    palma_history = Float64[]
    pshed_history = Vector{Float64}[]
    weights_history = Vector{Float64}[]
    solve_times = Float64[]

    math_working = deepcopy(math)

    for k in 1:iterations
        # Update weights in math dictionary
        for (i, id) in enumerate(load_ids)
            math_working["load"][string(id)]["weight"] = weights[i]
        end

        # Lower level: Solve MLD and get Jacobian
        jacobian, pshed_val, pshed_ids, weight_ids, ref = compute_jacobian_diffopt(math_working, weights)

        # Reorder to match load_ids ordering
        pshed_ordered = zeros(n_loads)
        for (i, pid) in enumerate(pshed_ids)
            idx = findfirst(==(pid), load_ids)
            if idx !== nothing
                pshed_ordered[idx] = pshed_val[i]
            end
        end

        # Reorder Jacobian to match load_ids
        jacobian_ordered = zeros(n_loads, n_loads)
        for (i, pid) in enumerate(pshed_ids)
            for (j, wid) in enumerate(weight_ids)
                i_new = findfirst(==(pid), load_ids)
                j_new = findfirst(==(wid), load_ids)
                if i_new !== nothing && j_new !== nothing
                    jacobian_ordered[i_new, j_new] = jacobian[i, j]
                end
            end
        end

        # Store initial Palma if first iteration
        if k == 1
            initial_palma = palma_ratio(pshed_ordered)
            push!(palma_history, initial_palma)
            push!(pshed_history, copy(pshed_ordered))
            push!(weights_history, copy(weights))
        else
            # For subsequent iterations, show ACTUAL vs PREDICTED comparison
            actual_palma_from_mld = palma_ratio(pshed_ordered)
            if length(palma_history) > 0
                predicted_palma = palma_history[end]
                verbose && println("  [DEBUG] Predicted Palma (Taylor): $predicted_palma")
                verbose && println("  [DEBUG] Actual Palma (MLD solve): $actual_palma_from_mld")
                verbose && println("  [DEBUG] Prediction error: $(abs(predicted_palma - actual_palma_from_mld) / actual_palma_from_mld * 100)%")
            end
        end

        # Debug: Show Jacobian diagonal (should be negative)
        if k == 1 && verbose
            println("  [DEBUG] Jacobian diagonal (first 5): ", round.(diag(jacobian_ordered)[1:min(5,n_loads)], digits=4))
            println("  [DEBUG] Expected: NEGATIVE values (increasing weight → decreasing shed)")
        end

        # Upper level: Solve reformulated Palma ratio minimization
        t_start = time()

        # Check if Palma is well-defined before attempting optimization
        if !is_palma_well_defined(pshed_ordered; min_denom=1e-3 * sum(pd))
            verbose && println("  Note: Palma ratio not well-defined (bottom 40% ≈ 0)")
            verbose && println("        Many loads have zero shedding - this is actually good!")
            verbose && println("        Skipping optimization, using current weights.")

            # Keep current state - no optimization needed when load shedding is minimal
            push!(palma_history, palma_ratio(pshed_ordered))
            push!(pshed_history, copy(pshed_ordered))
            push!(weights_history, copy(weights))
            push!(solve_times, 0.0)
            continue
        end

        result = palma_ratio_minimization(
            jacobian_ordered,
            pshed_ordered,
            weights,
            pd;
            trust_radius = trust_radius,
            w_bounds = (0.0, 10.0),
            relax_binary = false,  # Must use integer permutation for correct sorting
            silent = false  # Show Gurobi output to debug
        )
        solve_time = time() - t_start

        # Update weights for next iteration
        old_weights = copy(weights)
        weights = result.weights_new

        # Debug: Show weight changes
        if verbose
            delta_w = weights - old_weights
            println("  [DEBUG] Weight changes (Δw): min=$(minimum(delta_w)), max=$(maximum(delta_w))")
            println("  [DEBUG] Loads with Δw > 0: ", sum(delta_w .> 0.001))
            println("  [DEBUG] Loads with Δw < 0: ", sum(delta_w .< -0.001))
        end

        # Store results
        push!(palma_history, result.palma_ratio)
        push!(pshed_history, result.pshed_new)
        push!(weights_history, copy(weights))
        push!(solve_times, solve_time)

        verbose && println(@sprintf("   %2d       %8.4f      %8.4f     %7.3f s",
            k, result.palma_ratio, sum(result.pshed_new), solve_time))
    end

    verbose && println()

    #=========================================================================
    # Step 3: Generate plots and export results
    =========================================================================#
    if plot_results
        verbose && println("Step 3: Generating plots...")

        # Plot 1: Palma ratio convergence (handle Inf values)
        palma_finite = [isinf(p) ? NaN : p for p in palma_history]
        if all(isnan.(palma_finite))
            verbose && println("  Warning: All Palma ratios are Inf (bottom 40% ≈ 0)")
            p1 = plot(title = "Palma Ratio Convergence\n(Undefined - bottom 40% has zero shedding)",
                      xlabel = "Iteration", ylabel = "Palma Ratio")
        else
            p1 = plot(0:iterations, palma_finite,
                xlabel = "Iteration",
                ylabel = "Palma Ratio",
                title = "Palma Ratio Convergence",
                marker = :circle,
                legend = false,
                linewidth = 2
            )
        end
        savefig(p1, joinpath(output_path, "palma_convergence.png"))
        savefig(p1, joinpath(output_path, "palma_convergence.svg"))

        # Plot 2: Total load shed per iteration (values are in kW, not p.u.)
        total_shed = [sum(ps) for ps in pshed_history]
        p2 = plot(0:iterations, total_shed,
            xlabel = "Iteration",
            ylabel = "Total Load Shed (kW)",
            title = "Total Load Shed per Iteration",
            marker = :circle,
            legend = false,
            linewidth = 2
        )
        savefig(p2, joinpath(output_path, "total_shed.png"))
        savefig(p2, joinpath(output_path, "total_shed.svg"))

        # Plot 3: Final load shed distribution (bar chart)
        final_pshed = pshed_history[end]
        p3 = bar(string.(load_ids), final_pshed,
            xlabel = "Load ID",
            ylabel = "Load Shed (kW)",
            title = "Final Load Shed Distribution",
            legend = false
        )
        savefig(p3, joinpath(output_path, "final_pshed_distribution.png"))
        savefig(p3, joinpath(output_path, "final_pshed_distribution.svg"))

        # Plot 4: Final weight distribution (bar chart)
        final_weights = weights_history[end]
        p4 = bar(string.(load_ids), final_weights,
            xlabel = "Load ID",
            ylabel = "Weight",
            title = "Final Weight Distribution",
            legend = false
        )
        savefig(p4, joinpath(output_path, "final_weights.png"))
        savefig(p4, joinpath(output_path, "final_weights.svg"))

        # Plot 5: Weight evolution heatmap
        weights_matrix = hcat(weights_history...)'
        p5 = heatmap(string.(load_ids), string.(0:iterations), weights_matrix,
            xlabel = "Load ID",
            ylabel = "Iteration",
            title = "Weight Evolution",
            c = :viridis
        )
        savefig(p5, joinpath(output_path, "weights_heatmap.png"))
        savefig(p5, joinpath(output_path, "weights_heatmap.svg"))

        # Plot 6: Load shed evolution heatmap
        pshed_matrix = hcat(pshed_history...)'
        p6 = heatmap(string.(load_ids), string.(0:iterations), pshed_matrix,
            xlabel = "Load ID",
            ylabel = "Iteration",
            title = "Load Shed Evolution (kW)",
            c = :hot
        )
        savefig(p6, joinpath(output_path, "pshed_heatmap.png"))
        savefig(p6, joinpath(output_path, "pshed_heatmap.svg"))

        verbose && println("  Plots saved to: $output_path")
    end

    # Export data to CSV
    verbose && println("Step 4: Exporting results...")

    # Convergence data
    df_convergence = DataFrame(
        iteration = 0:iterations,
        palma_ratio = palma_history,
        total_shed = [sum(ps) for ps in pshed_history]
    )
    CSV.write(joinpath(output_path, "convergence.csv"), df_convergence)

    # Final results
    df_final = DataFrame(
        load_id = load_ids,
        demand = pd,
        final_pshed = pshed_history[end],
        final_weight = weights_history[end],
        shed_fraction = pshed_history[end] ./ pd
    )
    CSV.write(joinpath(output_path, "final_results.csv"), df_final)

    # Fairness metrics comparison
    initial_pshed = pshed_history[1]
    final_pshed = pshed_history[end]
    df_fairness = DataFrame(
        metric = ["Palma Ratio", "Gini Index", "Jain's Index", "Min-Max Range"],
        initial = [
            palma_ratio(initial_pshed),
            gini_index(initial_pshed),
            jains_index(initial_pshed),
            maximum(initial_pshed) - minimum(initial_pshed)
        ],
        final = [
            palma_ratio(final_pshed),
            gini_index(final_pshed),
            jains_index(final_pshed),
            maximum(final_pshed) - minimum(final_pshed)
        ]
    )
    df_fairness.improvement = (df_fairness.initial .- df_fairness.final) ./ df_fairness.initial .* 100
    CSV.write(joinpath(output_path, "fairness_metrics.csv"), df_fairness)

    verbose && println("  CSV files saved")
    verbose && println()

    #=========================================================================
    # Summary
    =========================================================================#
    verbose && println("="^70)
    verbose && println("RESULTS SUMMARY")
    verbose && println("="^70)
    verbose && println(@sprintf("Initial Palma Ratio: %.4f", palma_history[1]))
    verbose && println(@sprintf("Final Palma Ratio:   %.4f", palma_history[end]))
    verbose && println(@sprintf("Improvement:         %.2f%%",
        (palma_history[1] - palma_history[end]) / palma_history[1] * 100))
    verbose && println(@sprintf("Total solve time:    %.2f s", sum(solve_times)))
    verbose && println("Output directory:    $output_path")
    verbose && println("="^70)

    # Update final weights in math dictionary
    for (i, id) in enumerate(load_ids)
        math_working["load"][string(id)]["weight"] = weights[i]
    end

    return (
        math = math_working,
        palma_history = palma_history,
        pshed_history = pshed_history,
        weights_history = weights_history,
        output_path = output_path
    )
end

#=============================================================================
 Fairness Metric Functions (for comparison)
=============================================================================#

"""Compute Gini index of a distribution (0 = perfect equality, 1 = max inequality)"""
function gini_index(x::Vector{Float64})
    x = sort(x)
    n = length(x)
    sum_x = sum(x)
    if sum_x == 0
        return 0.0
    end
    gini_top = 1 - 1/n + 2 * sum(sum(x[j] for j in 1:i) for i in 1:n-1) / (n * sum_x)
    gini_bottom = 2 * (1 - 1/n)
    return gini_top / gini_bottom
end

"""Compute Jain's fairness index (1 = perfect fairness)"""
function jains_index(x::Vector{Float64})
    n = length(x)
    sum_x = sum(x)
    sum_x2 = sum(xi^2 for xi in x)
    if sum_x2 == 0
        return 1.0
    end
    return (sum_x^2) / (n * sum_x2)
end

#=============================================================================
 Minimum Working Example
=============================================================================#

"""
Run a minimal working example using the IEEE 13-bus motivation_b network.
"""
function run_mwe(; ls_percent::Float64=0.5)
    println("\n" * "="^70)
    println("MINIMUM WORKING EXAMPLE: IEEE 13-Bus Network")
    println("="^70)
    println()
    println("Note: Using $(Int(ls_percent*100))% capacity constraint to force load shedding.")
    println("      Palma ratio requires shedding across most loads to be meaningful.")
    println()

    result = solve_palma_ratio_minimization(
        "ieee_13_aw_edit/motivation_a.dss";
        ls_percent = ls_percent,  # Lower = more shedding required
        iterations = 3,           # Fewer iterations for faster testing
        trust_radius = 0.1,
        experiment_name = "mwe_ieee13a_$(Int(ls_percent*100))pct",
        verbose = true
    )

    println("\nMWE completed successfully!")
    println("Check output at: $(result.output_path)")

    return result
end

#=============================================================================
 Entry Point
=============================================================================#

if abspath(PROGRAM_FILE) == @__FILE__
    # Run the MWE by default
    run_mwe()
end
