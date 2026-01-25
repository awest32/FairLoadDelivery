"""
    compare_load_shed.jl

    Efficient script to compare load shed results from the bilevel formulation
    across all cases and fairness functions.
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
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/random_rounding.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASES = ["motivation_a", "motivation_b", "motivation_c"]
const FAIR_FUNCS = ["proportional", "efficiency", "min_max", "equality_min", "jain"]
const LS_PERCENT = 0.9
const ITERATIONS = 5
const N_ROUNDS = 2
const N_BERNOULLI_SAMPLES = 5

# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

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

# ============================================================
# STREAMLINED BILEVEL OPTIMIZATION (no plotting)
# ============================================================
function run_bilevel_relaxed(data::Dict{String, Any}, iterations::Int, fair_weights_init::Vector{Float64}, fair_func::String)
    math_new = deepcopy(data)
    fair_weights = copy(fair_weights_init)  # Mutable copy for updates
    pshed_lower_level = Float64[]
    pshed_upper_level = Float64[]

    for k in 1:iterations
        # Solve lower-level problem and get sensitivities
        dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, _ = lower_level_soln(data, fair_weights, k)

        # Apply fairness function
        if fair_func == "proportional"
            pshed_new, fair_weight_vals = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals)
        elseif fair_func == "efficiency"
            pshed_new, fair_weight_vals = complete_efficiency_load_shed(dpshed, pshed_val, weight_vals, math_new)
        elseif fair_func == "min_max"
            pshed_new, fair_weight_vals = min_max_load_shed(dpshed, pshed_val, weight_vals)
        elseif fair_func == "equality_min"
            pshed_new, fair_weight_vals = equality_min(dpshed, pshed_val, weight_vals)
        elseif fair_func == "jain"
            pshed_new, fair_weight_vals = jains_fairness_index(dpshed, pshed_val, weight_vals)
        end

        # Update weights in math dictionary
        for (i, w) in zip(weight_ids, fair_weight_vals)
            math_new["load"][string(i)]["weight"] = w
        end

        # Update fair_weights for next iteration
        fair_weights = fair_weight_vals

        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))
    end

    return math_new, pshed_lower_level, pshed_upper_level
end

# ============================================================
# RANDOM ROUNDING AND FINAL SOLUTION
# ============================================================
function run_random_rounding(math_relaxed::Dict, n_rounds::Int, n_samples::Int, ipopt)
    # Solve implicit diff to get switch/block states
    mld_implicit = solve_mc_mld_shed_implicit_diff(math_relaxed, ipopt; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])

    imp_model = instantiate_mc_model(
        math_relaxed,
        LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
    )
    ref = imp_model.ref[:it][:pmd][:nw][0]

    switch_states, block_status = extract_switch_block_states(mld_implicit["solution"])

    # Storage for each round
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

        # Apply rounded states
        math_copy = deepcopy(math_relaxed)
        math_rounded = update_network(
            math_copy,
            Dict(zip(block_ids, block_status_radial)),
            Dict(zip(load_ids, load_status)),
            switch_states_radial,
            ref, r
        )

        # Solve rounded MLD
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

function extract_per_load_data(mld::Dict)
    """Extract per-load shed and served values, sorted by load ID."""
    if mld === nothing || !haskey(mld, "solution") || !haskey(mld["solution"], "load")
        return Int[], Float64[], Float64[]
    end

    load_ids = sort(parse.(Int, collect(keys(mld["solution"]["load"]))))
    pshed_per_load = Float64[]
    pd_per_load = Float64[]

    for lid in load_ids
        load_data = mld["solution"]["load"][string(lid)]
        push!(pshed_per_load, haskey(load_data, "pshed") ? sum(load_data["pshed"]) : 0.0)
        push!(pd_per_load, haskey(load_data, "pd") ? sum(load_data["pd"]) : 0.0)
    end

    return load_ids, pshed_per_load, pd_per_load
end

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
const FAIR_FUNC_COLORS = Dict(
    "proportional" => :green,
    "efficiency" => :blue,
    "min_max" => :red,
    "equality_min" => :orange,
    "jain" => :purple
)

const FAIR_FUNC_LABELS = Dict(
    "proportional" => "Proportional",
    "efficiency" => "Efficiency",
    "min_max" => "Min-Max",
    "equality_min" => "Equality Min",
    "jain" => "Jain's Index"
)

function create_grouped_bar_chart(case::String, per_load_data::Dict, data_type::Symbol, save_dir::String)
    """
    Create grouped bar chart for load shed or load served.
    data_type: :pshed or :pd_served
    """
    # Get union of all load IDs across all fairness functions
    all_load_ids = Set{Int}()
    for (_, data) in per_load_data
        if !isempty(data[:load_ids])
            union!(all_load_ids, data[:load_ids])
        end
    end

    if isempty(all_load_ids)
        @warn "No load data available for $case"
        return nothing
    end

    load_ids = sort(collect(all_load_ids))

    # Count functions with actual data
    n_funcs_with_data = count(ff -> haskey(per_load_data, ff) && !isempty(per_load_data[ff][:load_ids]), FAIR_FUNCS)
    if n_funcs_with_data == 0
        @warn "No valid data for $case"
        return nothing
    end

    bar_width = 0.8 / n_funcs_with_data
    x_offset = bar_width

    if data_type == :pshed
        title = "Load Shed Comparison: $case"
        ylabel = "Load Shed (kW)"
        filename = "load_shed_per_bus_$case.svg"
    else
        title = "Load Served Comparison: $case"
        ylabel = "Load Served (kW)"
        filename = "load_served_per_bus_$case.svg"
    end

    p = plot(
        xlabel = "Load ID",
        ylabel = ylabel,
        title = title,
        legend = :outertopright,
        xticks = load_ids,
        size = (1100, 500),
        bottom_margin = 5Plots.mm
    )

    # Calculate offsets to center the bars
    offsets = collect(range(-(n_funcs_with_data-1)/2 * x_offset, (n_funcs_with_data-1)/2 * x_offset, length=n_funcs_with_data))

    func_idx = 0
    for fair_func in FAIR_FUNCS
        if !haskey(per_load_data, fair_func)
            continue
        end

        data = per_load_data[fair_func]
        func_load_ids = data[:load_ids]
        func_values = data_type == :pshed ? data[:pshed] : data[:pd_served]

        if isempty(func_load_ids)
            continue
        end

        func_idx += 1

        # Create a mapping from load_id to value for this function
        id_to_value = Dict(zip(func_load_ids, func_values))

        # Build values array aligned with the common load_ids (0 for missing)
        aligned_values = [get(id_to_value, lid, 0.0) for lid in load_ids]

        bar!(p, load_ids .+ offsets[func_idx], aligned_values,
            bar_width = bar_width * 0.9,
            label = FAIR_FUNC_LABELS[fair_func],
            color = FAIR_FUNC_COLORS[fair_func]
        )
    end

    savefig(p, joinpath(save_dir, filename))
    return p
end

# ============================================================
# MAIN COMPARISON LOOP
# ============================================================
function get_total_demand(math::Dict)
    total_pd = 0.0
    for (_, load) in math["load"]
        total_pd += sum(load["pd"])
    end
    return total_pd
end

function run_comparison()
    # Results storage
    results = DataFrame(
        case = String[],
        fair_func = String[],
        total_demand = Float64[],
        final_pshed = Float64[],
        final_pd_served = Float64[],
        pct_shed = Float64[],
        pct_served = Float64[],
        relaxed_pshed_final = Float64[],
        objective = Float64[]
    )

    # Per-load data storage: case => fair_func => {load_ids, pshed, pd_served}
    per_load_results = Dict{String, Dict{String, Dict{Symbol, Vector}}}()

    println("=" ^ 60)
    println("LOAD SHED COMPARISON ACROSS CASES AND FAIRNESS FUNCTIONS")
    println("=" ^ 60)

    for case in CASES
        println("\n>>> Processing case: $case")

        # Setup network
        eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/$case.dss", LS_PERCENT, [])
        fair_weights = Float64[load["weight"] for (_, load) in math["load"]]
        total_demand = get_total_demand(math)

        println("    Total demand: $(round(total_demand, digits=4))")

        # Initialize per-load storage for this case
        per_load_results[case] = Dict{String, Dict{Symbol, Vector}}()

        for fair_func in FAIR_FUNCS
            print("  $fair_func: ")

            try
                # Run bilevel relaxation
                math_relaxed, pshed_lower, pshed_upper = run_bilevel_relaxed(
                    math, ITERATIONS, fair_weights, fair_func
                )

                # Run random rounding
                math_out, mld_results = run_random_rounding(
                    math_relaxed, N_ROUNDS, N_BERNOULLI_SAMPLES, ipopt_solver
                )

                if isempty(mld_results)
                    println("No feasible solution found")
                    push!(results, (case, fair_func, total_demand, NaN, NaN, NaN, NaN, pshed_upper[end], NaN))
                    per_load_results[case][fair_func] = Dict(:load_ids => Int[], :pshed => Float64[], :pd_served => Float64[])
                    continue
                end

                # Find best solution
                best_idx, best_mld = find_best_mld_solution(mld_results)
                final_pshed, final_pd_served = extract_load_shed(best_mld)

                # Extract per-load data
                load_ids, pshed_per_load, pd_per_load = extract_per_load_data(best_mld)
                per_load_results[case][fair_func] = Dict(
                    :load_ids => load_ids,
                    :pshed => pshed_per_load,
                    :pd_served => pd_per_load
                )

                # Calculate percentages
                pct_shed = (final_pshed / total_demand) * 100
                pct_served = (final_pd_served / total_demand) * 100

                println("shed=$(round(pct_shed, digits=2))%, served=$(round(pct_served, digits=2))%")

                push!(results, (
                    case,
                    fair_func,
                    total_demand,
                    final_pshed,
                    final_pd_served,
                    pct_shed,
                    pct_served,
                    pshed_upper[end],
                    best_mld["objective"]
                ))

            catch e
                println("ERROR: $e")
                push!(results, (case, fair_func, total_demand, NaN, NaN, NaN, NaN, NaN, NaN))
                per_load_results[case][fair_func] = Dict(:load_ids => Int[], :pshed => Float64[], :pd_served => Float64[])
            end
        end
    end

    return results, per_load_results
end

# ============================================================
# RUN AND SAVE RESULTS
# ============================================================
println("\nStarting comparison at $(now())...\n")

results, per_load_results = run_comparison()

# Print summary table
println("\n" * "=" ^ 60)
println("SUMMARY TABLE")
println("=" ^ 60)
println(results)

# Pivot table for easier comparison
println("\n" * "=" ^ 60)
println("LOAD SHED BY CASE AND FAIRNESS FUNCTION")
println("=" ^ 60)

for case in CASES
    println("\n$case:")
    case_results = filter(row -> row.case == case, results)
    if !isempty(case_results)
        println("  Total demand: $(round(first(case_results.total_demand), digits=4))")
        println("  " * "-"^50)
    end
    for row in eachrow(case_results)
        println("  $(rpad(row.fair_func, 15)) => shed: $(round(row.pct_shed, digits=2))%, served: $(round(row.pct_served, digits=2))%")
    end
end

# Save results
save_dir = joinpath("results", Dates.format(today(), "yyyy-mm-dd"))
mkpath(save_dir)
csv_path = joinpath(save_dir, "load_shed_comparison.csv")
CSV.write(csv_path, results)
println("\nResults saved to: $csv_path")

# ============================================================
# GENERATE GROUPED BAR CHARTS
# ============================================================
println("\n" * "=" ^ 60)
println("GENERATING BAR CHARTS")
println("=" ^ 60)

for case in CASES
    if !haskey(per_load_results, case)
        continue
    end

    println("\n  Creating charts for $case...")

    # Create load shed chart
    p_shed = create_grouped_bar_chart(case, per_load_results[case], :pshed, save_dir)
    if p_shed !== nothing
        println("    Saved load_shed_per_bus_$case.svg")
    end

    # Create load served chart
    p_served = create_grouped_bar_chart(case, per_load_results[case], :pd_served, save_dir)
    if p_served !== nothing
        println("    Saved load_served_per_bus_$case.svg")
    end
end

println("\n" * "=" ^ 60)
println("COMPARISON COMPLETE")
println("Results and charts saved in: $save_dir")
println("=" ^ 60)
