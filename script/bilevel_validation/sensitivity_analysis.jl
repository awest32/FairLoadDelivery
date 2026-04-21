"""
    sensitivity_analysis.jl

    Sensitivity analysis for weight limits and step size in the bilevel formulation.
    Analyzes how these parameters affect load shed performance across fairness functions.
"""

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS
using JuMP
using LinearAlgebra, SparseArrays
using DataFrames
using CSV
using Dates
using Plots

include("../../src/implementation/network_setup.jl")
include("../../src/implementation/lower_level_mld.jl")
include("../../src/implementation/random_rounding.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASE = "motivation_a"  # Use single case for sensitivity analysis
const LS_PERCENT = 0.9
const ITERATIONS = 5
const N_ROUNDS = 2
const N_BERNOULLI_SAMPLES = 5

# Sensitivity analysis parameters
const WEIGHT_MIN_VALUES = [1.0]  # Lower bound (keep fixed)
const WEIGHT_MAX_VALUES = [5.0, 10.0, 20.0, 50.0]  # Upper bound variations
const STEP_SIZE_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0]  # Step size variations
const FAIR_FUNCS = ["proportional", "efficiency", "min_max", "equality_min", "jain"]

# Solver
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

# ============================================================
# PARAMETERIZED FAIRNESS FUNCTIONS
# ============================================================

function jains_fairness_index_param(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64},
                                     weights_prev::Vector{Float64};
                                     weight_min=1.0, weight_max=10.0, step_size=0.1)
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(pshed_prev)
    @variable(model, weights_new[1:n] >= weight_min)
    @constraint(model, [i=1:n], weights_new[i] <= weight_max)
    @constraint(model, [i=1:n], weights_new[i] - weights_prev[i] <= step_size)
    @expression(model, pshed_new[i=1:n],
        pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:n)
    )
    sum_pshed = sum(pshed_new)
    sum_pshed_squared = sum(pshed_new[i]^2 for i in 1:n)
    fairness_index = (sum_pshed^2) / (n * sum_pshed_squared)
    @objective(model, Max, fairness_index)
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

function min_max_load_shed_param(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64},
                                  weights_prev::Vector{Float64};
                                  weight_min=1.0, weight_max=10.0, step_size=0.1)
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(weights_prev)
    @variable(model, weights_new[1:n] >= weight_min)
    @constraint(model, [i=1:n], weights_new[i] <= weight_max)
    @constraint(model, [i=1:n], weights_new[i] - weights_prev[i] <= step_size)
    @variable(model, t >= 1)
    @expression(model, pshed_new[i=1:length(pshed_prev)],
        pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:n)
    )
    @constraint(model, t >= maximum(pshed_new))
    @objective(model, Min, maximum(pshed_new))
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

function proportional_fairness_load_shed_param(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64},
                                                weights_prev::Vector{Float64};
                                                weight_min=1.0, weight_max=10.0, step_size=0.1)
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(weights_prev)
    @variable(model, weights_new[1:n] >= weight_min)
    @constraint(model, weights_new[1:n] .<= weight_max)
    @constraint(model, [i=1:n], weights_new[i] - weights_prev[i] <= step_size)
    @expression(model, pshed_new[i=1:length(pshed_prev)],
        pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:n)
    )
    @objective(model, Max, sum(log(pshed_new[i]) for i in 1:length(pshed_new)))
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

function efficient_load_shed_param(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64},
                                              weights_prev::Vector{Float64}, math::Dict{String,Any};
                                              weight_min=1.0, weight_max=10.0, step_size=0.1)
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(weights_prev)
    @variable(model, weights_new[1:n] >= weight_min)
    @constraint(model, weights_new[1:n] .<= weight_max)
    @constraint(model, [i=1:n], weights_new[i] - weights_prev[i] <= step_size)
    @expression(model, pshed_new[i=1:length(pshed_prev)],
        pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:n)
    )
    @objective(model, Min, sum(pshed_new))
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

function equality_min_param(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64},
                            weights_prev::Vector{Float64};
                            weight_min=1.0, weight_max=10.0, step_size=0.1)
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(weights_prev)
    @variable(model, weights_new[1:n] >= weight_min)
    @variable(model, t >= 0)
    @constraint(model, weights_new[1:n] .<= weight_max)
    @constraint(model, [i=1:n], weights_new[i]^2 - weights_prev[i]^2 <= step_size^2)
    @expression(model, pshed_new[i=1:length(pshed_prev)],
        pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:n)
    )
    @constraint(model, [i=1:length(pshed_new)], t == pshed_new[i])
    @objective(model, Max, t)
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# ============================================================
# HELPER FUNCTIONS
# ============================================================

function extract_switch_block_states(relaxed_soln::Dict{String,Any})
    switch_states = Dict{Int64,Float64}()
    for (s_id, s_data) in relaxed_soln["switch"]
        switch_states[parse(Int, s_id)] = s_data["state"]
    end
    block_status = Dict{Int64,Float64}()
    for (b_id, b_data) in relaxed_soln["block"]
        block_status[parse(Int, b_id)] = b_data["status"]
    end
    return switch_states, block_status
end

function get_total_demand(math::Dict)
    total_pd = 0.0
    for (_, load) in math["load"]
        total_pd += sum(load["pd"])
    end
    return total_pd
end

# ============================================================
# BILEVEL WITH PARAMETERIZED FAIRNESS
# ============================================================

function run_bilevel_param(data::Dict{String, Any}, iterations::Int, fair_weights_init::Vector{Float64},
                           fair_func::String; weight_min=1.0, weight_max=10.0, step_size=0.1)
    math_new = deepcopy(data)
    fair_weights = copy(fair_weights_init)
    pshed_lower_level = Float64[]
    pshed_upper_level = Float64[]
    final_weights = Float64[]

    for k in 1:iterations
        dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, _ = lower_level_soln(data, fair_weights, k)

        # Apply parameterized fairness function
        if fair_func == "proportional"
            pshed_new, fair_weight_vals = proportional_fairness_load_shed_param(
                dpshed, pshed_val, weight_vals; weight_min=weight_min, weight_max=weight_max, step_size=step_size)
        elseif fair_func == "efficiency"
            pshed_new, fair_weight_vals = efficient_load_shed_param(
                dpshed, pshed_val, weight_vals; weight_min=weight_min, weight_max=weight_max, step_size=step_size)
        elseif fair_func == "min_max"
            pshed_new, fair_weight_vals = min_max_load_shed_param(
                dpshed, pshed_val, weight_vals; weight_min=weight_min, weight_max=weight_max, step_size=step_size)
        elseif fair_func == "equality_min"
            pshed_new, fair_weight_vals = equality_min_param(
                dpshed, pshed_val, weight_vals; weight_min=weight_min, weight_max=weight_max, step_size=step_size)
        elseif fair_func == "jain"
            pshed_new, fair_weight_vals = jains_fairness_index_param(
                dpshed, pshed_val, weight_vals; weight_min=weight_min, weight_max=weight_max, step_size=step_size)
        end

        for (i, w) in zip(weight_ids, fair_weight_vals)
            math_new["load"][string(i)]["weight"] = w
        end

        fair_weights = fair_weight_vals
        final_weights = fair_weight_vals

        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))
    end

    return math_new, pshed_lower_level, pshed_upper_level, final_weights
end

function run_random_rounding(math_relaxed::Dict, n_rounds::Int, n_samples::Int, ipopt)
    mld_implicit = solve_mc_mld_shed_implicit_diff(math_relaxed, ipopt;
        ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])

    imp_model = instantiate_mc_model(
        math_relaxed,
        LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
    )
    ref = imp_model.ref[:it][:pmd][:nw][0]

    switch_states, block_status = extract_switch_block_states(mld_implicit["solution"])

    math_out = Vector{Dict{String, Any}}()
    mld_results = Vector{Dict{String, Any}}()

    for r in 1:n_rounds
        rng = 100 * r
        bernoulli_samples = generate_bernoulli_samples(switch_states, n_samples, rng)

        index, switch_states_radial, block_ids, block_status_radial, load_ids, load_status =
            radiality_check(ref, switch_states, block_status, bernoulli_samples)

        if index === nothing
            continue
        end

        math_copy = deepcopy(math_relaxed)
        math_rounded = update_network(
            math_copy,
            Dict(zip(block_ids, block_status_radial)),
            Dict(zip(load_ids, load_status)),
            switch_states_radial,
            ref, r
        )

        mld_rounded = FairLoadDelivery.solve_mc_mld_shed_random_round(math_rounded, ipopt)

        if mld_rounded["termination_status"] in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
            push!(math_out, math_rounded)
            push!(mld_results, mld_rounded)
        end
    end

    return math_out, mld_results
end

function find_best_mld_solution(mlds::Vector{Dict{String, Any}})
    best_obj = -Inf
    best_idx = 0
    for (idx, mld) in enumerate(mlds)
        if mld["objective"] > best_obj
            best_obj = mld["objective"]
            best_idx = idx
        end
    end
    return best_idx, best_idx > 0 ? mlds[best_idx] : nothing
end

function extract_load_shed(mld::Dict)
    if mld === nothing || !haskey(mld, "solution") || !haskey(mld["solution"], "load")
        return NaN, NaN
    end
    total_pshed = 0.0
    total_pd_served = 0.0
    for (load_id, load_data) in mld["solution"]["load"]
        if haskey(load_data, "pshed")
            total_pshed += sum(load_data["pshed"])
        end
        if haskey(load_data, "pd")
            total_pd_served += sum(load_data["pd"])
        end
    end
    return total_pshed, total_pd_served
end

# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

function run_sensitivity_analysis()
    results = DataFrame(
        fair_func = String[],
        weight_max = Float64[],
        step_size = Float64[],
        total_demand = Float64[],
        final_pshed = Float64[],
        final_pd_served = Float64[],
        pct_shed = Float64[],
        pct_served = Float64[],
        weight_spread = Float64[],  # max - min of final weights
        objective = Float64[]
    )

    println("=" ^ 70)
    println("SENSITIVITY ANALYSIS: Weight Limits and Step Size")
    println("Case: $CASE")
    println("=" ^ 70)

    # Setup network once
    eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/$CASE.dss", LS_PERCENT, [])
    fair_weights_init = Float64[load["weight"] for (_, load) in math["load"]]
    total_demand = get_total_demand(math)

    println("Total demand: $(round(total_demand, digits=4))")
    println("Number of parameter combinations: $(length(WEIGHT_MAX_VALUES) * length(STEP_SIZE_VALUES) * length(FAIR_FUNCS))")

    for weight_max in WEIGHT_MAX_VALUES
        for step_size in STEP_SIZE_VALUES
            println("\n--- Weight max: $weight_max, Step size: $step_size ---")

            for fair_func in FAIR_FUNCS
                print("  $fair_func: ")

                try
                    # Run bilevel with parameters
                    math_relaxed, pshed_lower, pshed_upper, final_weights = run_bilevel_param(
                        math, ITERATIONS, fair_weights_init, fair_func;
                        weight_min=1.0, weight_max=weight_max, step_size=step_size
                    )

                    # Run random rounding
                    math_out, mld_results = run_random_rounding(
                        math_relaxed, N_ROUNDS, N_BERNOULLI_SAMPLES, ipopt_solver
                    )

                    if isempty(mld_results)
                        println("No feasible solution")
                        push!(results, (fair_func, weight_max, step_size, total_demand,
                                       NaN, NaN, NaN, NaN, NaN, NaN))
                        continue
                    end

                    best_idx, best_mld = find_best_mld_solution(mld_results)
                    final_pshed, final_pd_served = extract_load_shed(best_mld)

                    pct_shed = (final_pshed / total_demand) * 100
                    pct_served = (final_pd_served / total_demand) * 100
                    weight_spread = maximum(final_weights) - minimum(final_weights)

                    println("shed=$(round(pct_shed, digits=2))%, spread=$(round(weight_spread, digits=2))")

                    push!(results, (
                        fair_func,
                        weight_max,
                        step_size,
                        total_demand,
                        final_pshed,
                        final_pd_served,
                        pct_shed,
                        pct_served,
                        weight_spread,
                        best_mld["objective"]
                    ))

                catch e
                    println("ERROR: $e")
                    push!(results, (fair_func, weight_max, step_size, total_demand,
                                   NaN, NaN, NaN, NaN, NaN, NaN))
                end
            end
        end
    end

    return results
end

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

function plot_sensitivity_results(results::DataFrame, save_dir::String)
    # Plot 1: Percent served vs step size for each fairness function (at weight_max=10)
    p1 = plot(
        xlabel = "Step Size",
        ylabel = "Percent Served (%)",
        title = "Load Served vs Step Size (weight_max=10)",
        legend = :outertopright,
        size = (900, 500)
    )

    colors = Dict("proportional" => :green, "efficiency" => :blue,
                  "min_max" => :red, "equality_min" => :orange, "jain" => :purple)

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && row.weight_max == 10.0, results)
        if !isempty(subset)
            plot!(p1, subset.step_size, subset.pct_served,
                  label=fair_func, marker=:circle, linewidth=2, color=colors[fair_func])
        end
    end
    savefig(p1, joinpath(save_dir, "sensitivity_step_size.svg"))

    # Plot 2: Percent served vs weight_max for each fairness function (at step_size=0.1)
    p2 = plot(
        xlabel = "Weight Max",
        ylabel = "Percent Served (%)",
        title = "Load Served vs Weight Max (step_size=0.1)",
        legend = :outertopright,
        size = (900, 500)
    )

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && row.step_size == 0.1, results)
        if !isempty(subset)
            plot!(p2, subset.weight_max, subset.pct_served,
                  label=fair_func, marker=:circle, linewidth=2, color=colors[fair_func])
        end
    end
    savefig(p2, joinpath(save_dir, "sensitivity_weight_max.svg"))

    # Plot 3: Heatmap for efficiency function
    p3 = plot_heatmap_for_func(results, "efficiency", save_dir)

    # Plot 4: Heatmap for proportional function
    p4 = plot_heatmap_for_func(results, "proportional", save_dir)

    # Plot 5: Weight spread vs performance
    p5 = plot(
        xlabel = "Weight Spread (max - min)",
        ylabel = "Percent Served (%)",
        title = "Performance vs Weight Differentiation",
        legend = :outertopright,
        size = (900, 500)
    )

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.weight_spread), results)
        if !isempty(subset)
            scatter!(p5, subset.weight_spread, subset.pct_served,
                     label=fair_func, marker=:circle, markersize=6, color=colors[fair_func])
        end
    end
    savefig(p5, joinpath(save_dir, "sensitivity_weight_spread.svg"))

    return p1, p2, p5
end

function plot_heatmap_for_func(results::DataFrame, fair_func::String, save_dir::String)
    subset = filter(row -> row.fair_func == fair_func && !isnan(row.pct_served), results)

    if isempty(subset)
        return nothing
    end

    # Create matrix for heatmap
    weight_maxs = sort(unique(subset.weight_max))
    step_sizes = sort(unique(subset.step_size))

    matrix = fill(NaN, length(step_sizes), length(weight_maxs))

    for row in eachrow(subset)
        i = findfirst(==(row.step_size), step_sizes)
        j = findfirst(==(row.weight_max), weight_maxs)
        if i !== nothing && j !== nothing
            matrix[i, j] = row.pct_served
        end
    end

    p = heatmap(
        string.(weight_maxs),
        string.(step_sizes),
        matrix,
        xlabel = "Weight Max",
        ylabel = "Step Size",
        title = "Percent Served: $fair_func",
        color = :viridis,
        size = (700, 500)
    )

    savefig(p, joinpath(save_dir, "heatmap_$(fair_func).svg"))
    return p
end

# ============================================================
# RUN ANALYSIS
# ============================================================

println("\nStarting sensitivity analysis at $(now())...\n")

results = run_sensitivity_analysis()

# Save results
save_dir = joinpath("results", Dates.format(today(), "yyyy-mm-dd"), "sensitivity")
mkpath(save_dir)

csv_path = joinpath(save_dir, "sensitivity_results.csv")
CSV.write(csv_path, results)
println("\nResults saved to: $csv_path")

# Write summary to text file
summary_path = joinpath(save_dir, "sensitivity_summary.txt")
open(summary_path, "w") do io
    println(io, "SENSITIVITY ANALYSIS SUMMARY")
    println(io, "Generated: $(now())")
    println(io, "Case: $CASE")
    println(io, "=" ^ 70)

    println(io, "\nBEST PARAMETERS BY FAIRNESS FUNCTION")
    println(io, "-" ^ 70)
    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.pct_served), results)
        if !isempty(subset)
            best_row = subset[argmax(subset.pct_served), :]
            println(io, "\n$fair_func:")
            println(io, "  Best: weight_max=$(best_row.weight_max), step_size=$(best_row.step_size)")
            println(io, "  Served: $(round(best_row.pct_served, digits=2))%, Shed: $(round(best_row.pct_shed, digits=2))%")
            println(io, "  Weight spread: $(round(best_row.weight_spread, digits=2))")
        end
    end

    println(io, "\n" * "=" ^ 70)
    println(io, "EFFICIENCY FUNCTION ANALYSIS")
    println(io, "-" ^ 70)
    eff_results = filter(row -> row.fair_func == "efficiency" && !isnan(row.pct_served), results)
    if !isempty(eff_results)
        println(io, "\nEfficiency performance across parameter settings (sorted by % served):")
        for row in eachrow(sort(eff_results, :pct_served, rev=true))
            println(io, "  weight_max=$(row.weight_max), step=$(row.step_size) => $(round(row.pct_served, digits=2))% served, spread=$(round(row.weight_spread, digits=2))")
        end
    end

    println(io, "\n" * "=" ^ 70)
    println(io, "ALL RESULTS BY FAIRNESS FUNCTION")
    println(io, "-" ^ 70)
    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.pct_served), results)
        if !isempty(subset)
            println(io, "\n$fair_func:")
            for row in eachrow(sort(subset, :pct_served, rev=true))
                println(io, "  wmax=$(row.weight_max), step=$(row.step_size) => $(round(row.pct_served, digits=2))% served, spread=$(round(row.weight_spread, digits=2))")
            end
        end
    end
end
println("Summary saved to: $summary_path")

# Print summary table to console
println("\n" * "=" ^ 70)
println("SUMMARY: Best parameters by fairness function")
println("=" ^ 70)

for fair_func in FAIR_FUNCS
    subset = filter(row -> row.fair_func == fair_func && !isnan(row.pct_served), results)
    if !isempty(subset)
        best_row = subset[argmax(subset.pct_served), :]
        println("\n$fair_func:")
        println("  Best: weight_max=$(best_row.weight_max), step_size=$(best_row.step_size)")
        println("  Served: $(round(best_row.pct_served, digits=2))%, Shed: $(round(best_row.pct_shed, digits=2))%")
        println("  Weight spread: $(round(best_row.weight_spread, digits=2))")
    end
end

# Generate plots
println("\n" * "=" ^ 70)
println("GENERATING PLOTS")
println("=" ^ 70)

plot_sensitivity_results(results, save_dir)
println("Plots saved to: $save_dir")

println("\n" * "=" ^ 70)
println("SENSITIVITY ANALYSIS COMPLETE")
println("=" ^ 70)
