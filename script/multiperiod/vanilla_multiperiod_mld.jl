"""
    vanilla_multiperiod_mld.jl

    Multiperiod MLD problem using vanilla PowerModelsDistribution constraints.
    Creates simulated time-varying loads across multiple periods and solves the
    combined optimization problem.

    NOTE: This uses standard PMD constraints WITHOUT FairLoadDelivery's custom
    radiality, switch ampacity, and load block constraints. As a result, it will
    serve more load than the FLDP formulation.
"""

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution
const PMD = PowerModelsDistribution
using Ipopt, Gurobi
using JuMP
using DataFrames
using CSV
using Dates
using Plots
using Statistics

# Include the shared network setup function
include("../../src/implementation/network_setup.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASE_FILE = "ieee_13_aw_edit/motivation_a.dss"
const N_PERIODS = 3  # Number of time periods
const LS_PERCENT = 0.9  # Generation limit as fraction of total load

# Load scaling factors for each period (simulating daily load curve)
# Period 1: Morning (80% of peak)
# Period 2: Afternoon peak (100%)
# Period 3: Evening (90% of peak)
const LOAD_SCALE_FACTORS = [0.8, 1.0, 0.9]

# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

"""
Create multinetwork data structure from a single network.
Replicates the network N times with different load scaling factors.
"""
function create_multinetwork_data(base_math::Dict{String,Any}, n_periods::Int, load_scales::Vector{Float64})
    @assert length(load_scales) == n_periods "Load scale factors must match number of periods"

    # Create the multinetwork structure
    mn_data = Dict{String,Any}(
        "multinetwork" => true,
        "per_unit" => get(base_math, "per_unit", true),
        "name" => get(base_math, "name", "multiperiod_mld"),
        "data_model" => get(base_math, "data_model", PMD.MATHEMATICAL),
        "nw" => Dict{String,Any}()
    )

    # Copy global keys that should be at the top level
    for key in ["baseMVA", "basekv", "bus_lookup", "settings"]
        if haskey(base_math, key)
            mn_data[key] = deepcopy(base_math[key])
        end
    end

    # Create each time period network
    for t in 1:n_periods
        nw_id = string(t - 1)  # 0-indexed like PMD convention
        nw_data = deepcopy(base_math)

        # Remove multinetwork flag from individual networks
        delete!(nw_data, "multinetwork")

        # Scale loads for this time period
        scale = load_scales[t]
        for (load_id, load) in nw_data["load"]
            load["pd"] = load["pd"] .* scale
            load["qd"] = load["qd"] .* scale
        end

        # Add period identifier
        nw_data["time_period"] = t
        nw_data["load_scale"] = scale

        mn_data["nw"][nw_id] = nw_data
    end

    return mn_data
end

"""
Build the multiperiod MLD problem using vanilla PMD constraints.
This creates variables and constraints for each time period WITHOUT
FairLoadDelivery's custom radiality and load block constraints.
"""
function build_mn_mc_mld_vanilla(pm::PMD.AbstractUnbalancedPowerModel)
    # Get all network IDs and sort them
    nw_ids = sort(collect(PMD.nw_ids(pm)))

    println("Building multiperiod MLD (vanilla PMD) with $(length(nw_ids)) periods...")

    for n in nw_ids
        println("  Building period $n...")

        # Variables
        PMD.variable_mc_bus_voltage(pm; nw=n)
        PMD.variable_mc_branch_power(pm; nw=n)
        PMD.variable_mc_branch_current(pm; nw=n)
        PMD.variable_mc_switch_power(pm; nw=n)
        PMD.variable_mc_transformer_power(pm; nw=n)
        PMD.variable_mc_generator_power(pm; nw=n)
        PMD.variable_mc_load_indicator(pm; nw=n, relax=true)
        PMD.variable_mc_shunt_indicator(pm; nw=n, relax=true)

        # Constraints
        PMD.constraint_mc_model_voltage(pm; nw=n)

        for i in PMD.ids(pm, n, :ref_buses)
            PMD.constraint_mc_theta_ref(pm, i; nw=n)
        end

        for i in PMD.ids(pm, n, :gen)
            PMD.constraint_mc_generator_power(pm, i; nw=n)
        end

        for i in PMD.ids(pm, n, :bus)
            PMD.constraint_mc_power_balance_shed(pm, i; nw=n)
        end

        for i in PMD.ids(pm, n, :branch)
            PMD.constraint_mc_power_losses(pm, i; nw=n)
            PMD.constraint_mc_model_voltage_magnitude_difference(pm, i; nw=n)
            PMD.constraint_mc_voltage_angle_difference(pm, i; nw=n)
            PMD.constraint_mc_thermal_limit_from(pm, i; nw=n)
            PMD.constraint_mc_thermal_limit_to(pm, i; nw=n)
        end

        for i in PMD.ids(pm, n, :switch)
            PMD.constraint_mc_switch_state(pm, i; nw=n)
            PMD.constraint_mc_switch_thermal_limit(pm, i; nw=n)
        end

        for i in PMD.ids(pm, n, :transformer)
            PMD.constraint_mc_transformer_power(pm, i; nw=n)
        end
    end

    # Objective: minimize total load shed across all periods
    objective_mn_min_load_shed(pm)
end

"""
Objective function: minimize load shedding across all time periods.
"""
function objective_mn_min_load_shed(pm::PMD.AbstractUnbalancedPowerModel)
    nw_ids = PMD.nw_ids(pm)

    obj_expr = JuMP.AffExpr(0.0)

    for n in nw_ids
        for (i, load) in PMD.ref(pm, n, :load)
            z_demand = PMD.var(pm, n, :z_demand, i)
            pd = load["pd"]

            # Weight by load magnitude
            weight = get(load, "weight", 10.0)

            for (idx, p) in enumerate(pd)
                # Penalty for NOT serving load (1 - z_demand means shed)
                JuMP.add_to_expression!(obj_expr, weight * p * (1 - z_demand))
            end
        end
    end

    JuMP.@objective(pm.model, Min, obj_expr)
end

"""
Solve the multiperiod MLD problem using vanilla PMD formulation.
"""
function solve_mn_mc_mld(data::Dict{String,Any}, solver)
    return PMD.solve_mc_model(
        data,
        PMD.LPUBFDiagPowerModel,
        solver,
        build_mn_mc_mld_vanilla;
        multinetwork=true
    )
end

"""
Extract and summarize results from multiperiod solution.
"""
function summarize_mn_results(result::Dict, mn_data::Dict)
    println("\n" * "=" ^ 70)
    println("MULTIPERIOD MLD RESULTS")
    println("=" ^ 70)

    if result["termination_status"] != MOI.OPTIMAL &&
       result["termination_status"] != MOI.LOCALLY_SOLVED
        println("ERROR: Problem did not solve successfully")
        println("Status: $(result["termination_status"])")
        return nothing
    end

    println("Termination Status: $(result["termination_status"])")
    println("Objective Value: $(round(result["objective"], digits=2))")

    # Create results dataframe
    results_df = DataFrame(
        Period = Int[],
        LoadScale = Float64[],
        TotalLoadRef = Float64[],
        TotalServed = Float64[],
        TotalShed = Float64[],
        PctServed = Float64[],
        PctShed = Float64[]
    )

    nw_ids = sort(collect(keys(mn_data["nw"])))

    for nw_id in nw_ids
        nw_data = mn_data["nw"][nw_id]
        period = nw_data["time_period"]
        scale = nw_data["load_scale"]

        # Get solution for this period
        if !haskey(result["solution"], "nw") || !haskey(result["solution"]["nw"], nw_id)
            println("  Period $period: No solution found")
            continue
        end

        nw_soln = result["solution"]["nw"][nw_id]

        # Calculate totals
        total_load_ref = 0.0
        total_served = 0.0

        for (load_id, load_data) in nw_data["load"]
            pd_ref = sum(load_data["pd"])
            total_load_ref += pd_ref

            if haskey(nw_soln, "load") && haskey(nw_soln["load"], load_id)
                pd_served = sum(nw_soln["load"][load_id]["pd"])
                total_served += pd_served
            end
        end

        total_shed = total_load_ref - total_served
        pct_served = total_load_ref > 0 ? (total_served / total_load_ref) * 100 : 0.0
        pct_shed = total_load_ref > 0 ? (total_shed / total_load_ref) * 100 : 0.0

        push!(results_df, (
            Period = period,
            LoadScale = scale,
            TotalLoadRef = round(total_load_ref, digits=2),
            TotalServed = round(total_served, digits=2),
            TotalShed = round(total_shed, digits=2),
            PctServed = round(pct_served, digits=2),
            PctShed = round(pct_shed, digits=2)
        ))

        println("\n  Period $period (scale=$scale):")
        println("    Total load reference: $(round(total_load_ref, digits=2)) MW")
        println("    Total load served: $(round(total_served, digits=2)) MW")
        println("    Total load shed: $(round(total_shed, digits=2)) MW")
        println("    Percent served: $(round(pct_served, digits=2))%")
    end

    # Summary statistics
    println("\n" * "-" ^ 70)
    println("SUMMARY ACROSS ALL PERIODS:")
    println("  Total load (all periods): $(round(sum(results_df.TotalLoadRef), digits=2)) MW")
    println("  Total served (all periods): $(round(sum(results_df.TotalServed), digits=2)) MW")
    println("  Total shed (all periods): $(round(sum(results_df.TotalShed), digits=2)) MW")
    println("  Average percent served: $(round(mean(results_df.PctServed), digits=2))%")

    return results_df
end

"""
Create visualization of multiperiod results.
"""
function plot_mn_results(results_df::DataFrame, save_path::String)
    # Stacked bar chart of load served/shed by period
    periods = results_df.Period
    served = results_df.TotalServed
    shed = results_df.TotalShed
    total = served .+ shed

    p = bar(
        periods,
        total,
        label = "Total Load (Reference)",
        xlabel = "Time Period",
        ylabel = "Load (MW)",
        title = "Multiperiod Load Shedding Results",
        legend = :topright,
        color = :lightgray,
        alpha = 0.5,
        bar_width = 0.6
    )

    bar!(
        periods,
        served,
        label = "Load Served",
        color = :green,
        bar_width = 0.4
    )

    # Add percentage labels
    for i in 1:length(periods)
        pct = round(results_df.PctServed[i], digits=1)
        annotate!(periods[i], served[i] + 100, text("$(pct)%", 8, :center))
    end

    savefig(p, save_path)
    println("Plot saved to: $save_path")

    return p
end

# ============================================================
# MAIN EXECUTION
# ============================================================
function run_multiperiod_mld()
    println("=" ^ 70)
    println("VANILLA MULTIPERIOD MLD PROBLEM (no FLD constraints)")
    println("Case: $CASE_FILE")
    println("Periods: $N_PERIODS")
    println("Load scales: $LOAD_SCALE_FACTORS")
    println("=" ^ 70)

    # Step 1: Setup base network (using shared setup_network function)
    println("\n[1] Setting up base network...")
    eng, math, lbs, _ = setup_network(CASE_FILE, LS_PERCENT, [])

    println("  Loads: $(length(math["load"]))")
    println("  Buses: $(length(math["bus"]))")
    println("  Branches: $(length(math["branch"]))")
    println("  Switches: $(length(math["switch"]))")
    println("  Load blocks: $(length(lbs))")

    # Step 2: Create multinetwork data
    println("\n[2] Creating multinetwork data for $N_PERIODS periods...")
    mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)

    for nw_id in sort(collect(keys(mn_data["nw"])), by=x->parse(Int, x))
        nw_data = mn_data["nw"][nw_id]
        total_load = sum(sum(l["pd"]) for (_, l) in nw_data["load"])
        println("  Period $(nw_data["time_period"]): scale=$(nw_data["load_scale"]), total_load=$(round(total_load, digits=2)) MW")
    end

    # Step 3: Solve multiperiod problem
    println("\n[3] Solving multiperiod MLD problem...")
    result = solve_mn_mc_mld(mn_data, ipopt_solver)

    # Step 4: Summarize results
    results_df = summarize_mn_results(result, mn_data)

    # Step 5: Save results
    if results_df !== nothing
        # Create output directory
        today = Dates.format(Dates.today(), "yyyy-mm-dd")
        output_dir = joinpath(@__DIR__, "../..", "results", today)
        mkpath(output_dir)

        # Save CSV
        csv_path = joinpath(output_dir, "vanilla_multiperiod_mld_results.csv")
        CSV.write(csv_path, results_df)
        println("\nResults saved to: $csv_path")

        # Create plot
        plot_path = joinpath(output_dir, "vanilla_multiperiod_mld_results.svg")
        plot_mn_results(results_df, plot_path)
    end

    println("\n" * "=" ^ 70)
    println("MULTIPERIOD MLD COMPLETE")
    println("=" ^ 70)

    return result, mn_data, results_df
end

# Run the analysis
result, mn_data, results_df = run_multiperiod_mld();  # Semicolon suppresses output
