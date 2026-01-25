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
using Statistics

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
    final_weight_ids = Int[]
    final_weights = Float64[]

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
        final_weight_ids = weight_ids
        final_weights = fair_weight_vals

        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))
    end

    return math_new, pshed_lower_level, pshed_upper_level, final_weight_ids, final_weights
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

function check_binary_solution(mld::Dict)
    """Check that switch states and block status in final MLD solution are binary (0.0 or 1.0)."""
    if mld === nothing || !haskey(mld, "solution")
        return true, []
    end

    violations = String[]

    # Check switch states
    if haskey(mld["solution"], "switch")
        for (s_id, s_data) in mld["solution"]["switch"]
            state = s_data["state"]
            if !(isapprox(state, 0.0, atol=1e-6) || isapprox(state, 1.0, atol=1e-6))
                push!(violations, "Switch $s_id: state=$state")
            end
        end
    end

    # Check block status
    if haskey(mld["solution"], "block")
        for (b_id, b_data) in mld["solution"]["block"]
            status = b_data["status"]
            if !(isapprox(status, 0.0, atol=1e-6) || isapprox(status, 1.0, atol=1e-6))
                push!(violations, "Block $b_id: status=$status")
            end
        end
    end

    return isempty(violations), violations
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

function create_shed_distribution_plot(case::String, per_load_data::Dict, save_dir::String)
    """
    Create scatter plot showing shed distribution by fairness function.
    Similar to mld_comparison_experiment.jl shed_distribution.svg
    """
    # Get union of all load IDs
    all_load_ids = Set{Int}()
    for (_, data) in per_load_data
        if !isempty(data[:load_ids])
            union!(all_load_ids, data[:load_ids])
        end
    end

    if isempty(all_load_ids)
        return nothing
    end

    load_ids = sort(collect(all_load_ids))

    p = plot(
        xlabel = "Load ID",
        ylabel = "Load Shed (kW)",
        title = "Shed Distribution by Fairness Function: $case",
        legend = :outertopright,
        size = (1100, 500)
    )

    markers = Dict(
        "proportional" => :square,
        "efficiency" => :circle,
        "min_max" => :star5,
        "equality_min" => :diamond,
        "jain" => :utriangle
    )

    for fair_func in FAIR_FUNCS
        if !haskey(per_load_data, fair_func)
            continue
        end

        data = per_load_data[fair_func]
        if isempty(data[:load_ids])
            continue
        end

        id_to_value = Dict(zip(data[:load_ids], data[:pshed]))
        aligned_values = [get(id_to_value, lid, NaN) for lid in load_ids]

        scatter!(p, load_ids, aligned_values,
            label = FAIR_FUNC_LABELS[fair_func],
            marker = markers[fair_func],
            markersize = 8,
            color = FAIR_FUNC_COLORS[fair_func]
        )
    end

    # Add horizontal line for equality_min max shed (fairness target)
    if haskey(per_load_data, "equality_min") && !isempty(per_load_data["equality_min"][:pshed])
        max_eq_shed = maximum(per_load_data["equality_min"][:pshed])
        hline!(p, [max_eq_shed], label="EqMin Max Shed", linestyle=:dash, color=:orange, linewidth=2)
    end

    savefig(p, joinpath(save_dir, "shed_distribution_$case.svg"))
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

    # Final weights storage: case => fair_func => {weight_ids, weights}
    final_weights_results = Dict{String, Dict{String, Dict{Symbol, Vector}}}()

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
        final_weights_results[case] = Dict{String, Dict{Symbol, Vector}}()

        for fair_func in FAIR_FUNCS
            print("  $fair_func: ")

            try
                # Run bilevel relaxation
                math_relaxed, pshed_lower, pshed_upper, weight_ids, final_wts = run_bilevel_relaxed(
                    math, ITERATIONS, fair_weights, fair_func
                )

                # Store final weights
                final_weights_results[case][fair_func] = Dict(
                    :weight_ids => weight_ids,
                    :weights => final_wts
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

                # Check binary values in final solution
                binary_ok, binary_violations = check_binary_solution(best_mld)
                if !binary_ok
                    @warn "$case/$fair_func: Non-binary values in final MLD solution: $(join(binary_violations[1:min(3,length(binary_violations))], "; "))"
                end

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
                final_weights_results[case][fair_func] = Dict(:weight_ids => Int[], :weights => Float64[])
            end
        end
    end

    return results, per_load_results, final_weights_results
end

# ============================================================
# RUN AND SAVE RESULTS
# ============================================================
println("\nStarting comparison at $(now())...\n")

results, per_load_results, final_weights_results = run_comparison()

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
save_dir = "results/$(Dates.today())/"
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

    # Create shed distribution scatter plot
    p_dist = create_shed_distribution_plot(case, per_load_results[case], save_dir)
    if p_dist !== nothing
        println("    Saved shed_distribution_$case.svg")
    end
end

# ============================================================
# FINAL WEIGHTS COMPARISON
# ============================================================
println("\n" * "=" ^ 60)
println("FINAL WEIGHTS COMPARISON")
println("=" ^ 60)

for case in CASES
    if !haskey(final_weights_results, case)
        continue
    end

    println("\n$case:")

    # Get all load IDs
    all_weight_ids = Set{Int}()
    for (ff, data) in final_weights_results[case]
        union!(all_weight_ids, data[:weight_ids])
    end
    load_ids_sorted = sort(collect(all_weight_ids))

    if isempty(load_ids_sorted)
        println("  No weight data available")
        continue
    end

    # Build weights DataFrame
    weights_df = DataFrame(load_id = load_ids_sorted)
    for fair_func in FAIR_FUNCS
        if !haskey(final_weights_results[case], fair_func) || isempty(final_weights_results[case][fair_func][:weight_ids])
            weights_df[!, fair_func] = fill(NaN, length(load_ids_sorted))
        else
            data = final_weights_results[case][fair_func]
            id_to_weight = Dict(zip(data[:weight_ids], data[:weights]))
            weights_df[!, fair_func] = [get(id_to_weight, lid, NaN) for lid in load_ids_sorted]
        end
    end

    # Print weight statistics only
    for fair_func in FAIR_FUNCS
        wts = weights_df[!, fair_func]
        valid_wts = filter(!isnan, wts)
        if !isempty(valid_wts)
            println("  $(rpad(fair_func, 15)): min=$(round(minimum(valid_wts), digits=2)), max=$(round(maximum(valid_wts), digits=2)), spread=$(round(maximum(valid_wts)-minimum(valid_wts), digits=2))")
        end
    end

    # Save CSV
    CSV.write(joinpath(save_dir, "final_weights_$case.csv"), weights_df)

    # Create bar chart
    p_weights = plot(xlabel="Load ID", ylabel="Final Weight", title="Final Weights: $case",
                     legend=:outertopright, xticks=load_ids_sorted, size=(1100, 500))
    n_funcs = length(FAIR_FUNCS)
    bar_width = 0.8 / n_funcs
    offsets = collect(range(-(n_funcs-1)/2 * bar_width, (n_funcs-1)/2 * bar_width, length=n_funcs))

    for (i, fair_func) in enumerate(FAIR_FUNCS)
        wts = weights_df[!, fair_func]
        valid_mask = .!isnan.(wts)
        if any(valid_mask)
            bar!(p_weights, load_ids_sorted[valid_mask] .+ offsets[i], wts[valid_mask],
                bar_width=bar_width*0.9, label=FAIR_FUNC_LABELS[fair_func], color=FAIR_FUNC_COLORS[fair_func])
        end
    end
    savefig(p_weights, joinpath(save_dir, "final_weights_$case.svg"))
    println("  Saved: final_weights_$case.csv, final_weights_$case.svg")
end

# ============================================================
# FAIRNESS METRICS SUMMARY CSV
# ============================================================
println("\n" * "=" ^ 60)
println("FAIRNESS METRICS SUMMARY")
println("=" ^ 60)

for case in CASES
    if !haskey(per_load_results, case)
        continue
    end

    println("\n$case:")

    # Get total demand from results DataFrame
    case_rows = filter(row -> row.case == case, results)
    if isempty(case_rows)
        continue
    end
    total_demand = first(case_rows).total_demand

    # Build summary DataFrame
    summary_df = DataFrame(
        Formulation = String[],
        TotalServed_pct = Float64[],
        TotalShed_kW = Float64[],
        MaxShed_kW = Float64[],
        ShedVariance = Float64[],
        JainsIndex = Float64[],
        PalmaRatio = Float64[],
        GiniCoeff = Float64[],
        CV_Served = Float64[]
    )

    for fair_func in FAIR_FUNCS
        if !haskey(per_load_results[case], fair_func)
            continue
        end

        data = per_load_results[case][fair_func]
        pshed_vals = data[:pshed]
        pd_vals = data[:pd_served]

        if isempty(pshed_vals) || isempty(pd_vals)
            continue
        end

        # Calculate metrics
        total_shed = sum(pshed_vals)
        total_served = sum(pd_vals)
        total_served_pct = (total_served / total_demand) * 100
        max_shed = maximum(pshed_vals)
        shed_variance = Statistics.var(pshed_vals)

        # Fairness metrics on served values (filter zeros)
        served_nonzero = filter(x -> x > 0, pd_vals)
        if length(served_nonzero) >= 2
            jains_idx = jains_index(served_nonzero)
            palma = length(served_nonzero) >= 3 ? palma_ratio(served_nonzero) : NaN
            gini = gini_index(served_nonzero)
            cv_served = std(served_nonzero) / mean(served_nonzero)
        else
            jains_idx = NaN
            palma = NaN
            gini = NaN
            cv_served = NaN
        end

        push!(summary_df, (
            FAIR_FUNC_LABELS[fair_func],
            round(total_served_pct, digits=2),
            round(total_shed, digits=2),
            round(max_shed, digits=2),
            round(shed_variance, digits=2),
            round(jains_idx, digits=4),
            round(palma, digits=4),
            round(gini, digits=4),
            round(cv_served, digits=4)
        ))

        println("  $(rpad(FAIR_FUNC_LABELS[fair_func], 15)): $(round(total_served_pct, digits=1))% served, Jain=$(round(jains_idx, digits=3)), Gini=$(round(gini, digits=3))")
    end

    # Save summary CSV
    summary_path = joinpath(save_dir, "fairness_summary_$case.csv")
    CSV.write(summary_path, summary_df)
    println("  Saved: $summary_path")
end

println("\n" * "=" ^ 60)
println("COMPARISON COMPLETE")
println("Results and charts saved in: $save_dir")
println("=" ^ 60)
