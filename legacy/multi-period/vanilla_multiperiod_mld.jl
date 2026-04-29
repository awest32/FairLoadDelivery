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
const LS_PERCENT = 0.9  # Generation limit as fraction of total load

# 24-hour load profile (p.u.) - typical residential/commercial demand curve
# Morning ramp, midday plateau, evening peak, night valley
const LOAD_PROFILE_24H = [
    0.45,  # 00:00 - night minimum
    0.40,  # 01:00 - night minimum
    0.38,  # 02:00 - lowest demand
    0.38,  # 03:00 - lowest demand
    0.40,  # 04:00 - slight uptick
    0.50,  # 05:00 - early risers
    0.65,  # 06:00 - morning ramp begins
    0.80,  # 07:00 - morning ramp
    0.90,  # 08:00 - morning peak starting
    0.95,  # 09:00 - business hours
    0.95,  # 10:00 - business hours
    0.92,  # 11:00 - midday
    0.90,  # 12:00 - lunch dip
    0.92,  # 13:00 - afternoon
    0.95,  # 14:00 - afternoon
    0.98,  # 15:00 - afternoon peak building
    1.00,  # 16:00 - late afternoon peak
    1.00,  # 17:00 - evening peak
    0.98,  # 18:00 - evening peak
    0.95,  # 19:00 - evening
    0.85,  # 20:00 - evening decline
    0.75,  # 21:00 - late evening
    0.60,  # 22:00 - night begins
    0.50   # 23:00 - night
]

# Full 24-hour simulation
const N_PERIODS = 24
const LOAD_SCALE_FACTORS = LOAD_PROFILE_24H
const GEN_SCALE_FACTORS = GEN_PROFILE_24H

# 24-hour generation profile (p.u.) - base generation + solar variation
# Base generation always available, with solar boost during daylight hours
const GEN_PROFILE_24H = [
    0.50,  # 00:00 - base only
    0.50,  # 01:00 - base only
    0.50,  # 02:00 - base only
    0.50,  # 03:00 - base only
    0.50,  # 04:00 - base only
    0.52,  # 05:00 - dawn
    0.55,  # 06:00 - sunrise
    0.65,  # 07:00 - early morning ramp
    0.75,  # 08:00 - morning ramp
    0.85,  # 09:00 - mid-morning
    0.92,  # 10:00 - approaching peak
    0.98,  # 11:00 - near peak
    1.00,  # 12:00 - solar noon (peak)
    0.98,  # 13:00 - just past peak
    0.92,  # 14:00 - early afternoon
    0.85,  # 15:00 - mid-afternoon decline
    0.75,  # 16:00 - late afternoon
    0.65,  # 17:00 - evening decline
    0.55,  # 18:00 - sunset ramp down
    0.52,  # 19:00 - dusk
    0.50,  # 20:00 - base only
    0.50,  # 21:00 - base only
    0.50,  # 22:00 - base only
    0.50   # 23:00 - base only
]

# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

"""
Create multinetwork data structure from a single network.
Replicates the network N times with different load and generation scaling factors.
"""
function create_multinetwork_data(base_math::Dict{String,Any}, n_periods::Int, load_scales::Vector{Float64}, gen_scales::Vector{Float64})
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
        load_scale = load_scales[t]
        for (load_id, load) in nw_data["load"]
            load["pd"] = load["pd"] .* load_scale
            load["qd"] = load["qd"] .* load_scale
        end

        # Scale sourcebus generation capacity for this time period
        gen_scale = gen_scales[t]
        for (gen_id, gen) in nw_data["gen"]
            gen["pmax"] = gen["pmax"] .* gen_scale
            gen["qmax"] = gen["qmax"] .* gen_scale
        end

        # Add period identifier
        nw_data["time_period"] = t
        nw_data["load_scale"] = load_scale
        nw_data["gen_scale"] = gen_scale

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
        GenScale = Float64[],
        GenCapacity = Float64[],
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
        load_scale = nw_data["load_scale"]
        gen_scale = nw_data["gen_scale"]

        # Get solution for this period
        nw_soln = result["solution"]["nw"][nw_id]

        # Calculate totals
        total_load_ref = 0.0
        total_served = 0.0
        total_gen_capacity = sum(sum(g["pmax"]) for (_, g) in nw_data["gen"])

        for (load_id, load_data) in nw_data["load"]
            pd_ref = sum(load_data["pd"])
            total_load_ref += pd_ref
            pd_served = sum(nw_soln["load"][load_id]["pd"])
            total_served += pd_served
        end

        total_shed = total_load_ref - total_served
        pct_served = (total_served / total_load_ref) * 100
        pct_shed = (total_shed / total_load_ref) * 100

        push!(results_df, (
            Period = period,
            LoadScale = load_scale,
            GenScale = gen_scale,
            GenCapacity = round(total_gen_capacity, digits=2),
            TotalLoadRef = round(total_load_ref, digits=2),
            TotalServed = round(total_served, digits=2),
            TotalShed = round(total_shed, digits=2),
            PctServed = round(pct_served, digits=2),
            PctShed = round(pct_shed, digits=2)
        ))

        println("\n  Period $period (load_scale=$load_scale, gen_scale=$gen_scale):")
        println("    Gen capacity: $(round(total_gen_capacity, digits=2)) MW")
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
    println("Gen scales:  $GEN_SCALE_FACTORS")
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
    mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS, GEN_SCALE_FACTORS)

    for nw_id in sort(collect(keys(mn_data["nw"])), by=x->parse(Int, x))
        nw_data = mn_data["nw"][nw_id]
        total_load = sum(sum(l["pd"]) for (_, l) in nw_data["load"])
        total_gen = sum(sum(g["pmax"]) for (_, g) in nw_data["gen"])
        println("  Period $(nw_data["time_period"]): load_scale=$(nw_data["load_scale"]), gen_scale=$(nw_data["gen_scale"]), load=$(round(total_load, digits=2)) MW, gen=$(round(total_gen, digits=2)) MW")
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
