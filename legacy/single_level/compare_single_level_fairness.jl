"""
    compare_single_level_fairness.jl

    Compare single-level MLD results across fairness function formulations
    for all motivation cases. Solves integer (Gurobi) formulations only.
    Exports total network load shed and post-hoc fairness metrics
    (Jain, Palma, Gini) for each formulation.
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
using StatsPlots
using Statistics

include("../../src/implementation/network_setup.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASES = ["motivation_c"]
const GEN_CAP = 5000.0
const SOURCE_PU = 1.03
const SWITCH_RATING = 700.0

# Integer formulations (MIP, Gurobi)
const FAIR_SOLVE_INTEGER = [
    "efficiency"    => FairLoadDelivery.solve_mc_mld_switch_integer,
    "equality_min"  => FairLoadDelivery.solve_mc_mld_equality_min_integer,
    "proportional"  => FairLoadDelivery.solve_mc_mld_proportional_fairness_integer,
    "min_max"       => FairLoadDelivery.solve_mc_mld_min_max_integer,
    "jain"          => FairLoadDelivery.solve_mc_mld_jain_integer,
    "palma"         => FairLoadDelivery.solve_mc_mld_palma_integer,
]

# Solver
gurobi_solver = Gurobi.Optimizer

# Save results
save_dir = "results/$(Dates.today())/single_level_fairness"
integer_dir = joinpath(save_dir, "integer")
mkpath(integer_dir)

# Target buses for voltage plots (load buses of interest)
const TARGET_BUSES = ["670","632","645","671","634","646","611","675","652","692"]
#["primary", "sourcebus", "loadbus"]# 
# ============================================================
# POST-HOC FAIRNESS METRICS
# ============================================================
function gini_index(x)
    x = sort(x)
    n = length(x)
    return (2 * sum(i * x[i] for i in 1:n) / (n * sum(x))) - (n + 1) / n
end

function jains_index(x)
    n = length(x)
    sum_x = sum(x)
    sum_x2 = sum(xi^2 for xi in x)
    return (sum_x^2) / (n * sum_x2)
end

function palma_ratio_posthoc(x)
    sorted_x = sort(x)
    n = length(x)
    top_10 = sum(sorted_x[ceil(Int, 0.9n):end])
    bot_40 = sum(sorted_x[1:floor(Int, 0.4n)])
    return bot_40 > 0 ? top_10 / bot_40 : Inf
end

# ============================================================
# SOLVE + EXTRACT + COMPUTE METRICS
# ============================================================
function run_integer_comparison(cases, fair_solve_funcs, solver, out_dir::String)
    results_df = DataFrame(
        case = String[],
        fair_func = String[],
        total_demand = Float64[],
        total_pshed = Float64[],
        total_pd_served = Float64[],
        pct_shed = Float64[],
        pct_served = Float64[],
        objective = Float64[],
        jain = Float64[],
        palma = Float64[],
        gini = Float64[]
    )

    for case in cases
        println("\n>>> Processing case: $case")

        eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/$case.dss", GEN_CAP, SOURCE_PU, SWITCH_RATING, [])

        total_demand = sum(sum(load["pd"]) for (_, load) in math["load"])
        println("    Total demand: $(round(total_demand, digits=4))")

        voltage_data_per_func = Dict{String, Dict{String, Dict{Int, Float64}}}()
        loadshed_data_per_func = Dict{String, Tuple{Vector{String}, Vector{Float64}, Vector{Float64}}}()

        for (fair_func, solve_func) in fair_solve_funcs
            print("  $fair_func: ")

            mld_result = solve_func(math, solver)

            if !(mld_result["termination_status"] in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED])
                @warn "Non-optimal termination for $case/$fair_func: $(mld_result["termination_status"])"
                continue
            end

            solution = mld_result["solution"]

            # Extract per-bus voltage
            voltage_data_per_func[fair_func] = extract_voltage_by_bus_name(solution, math)

            # Extract per-bus loadshed
            bus_names, pshed_pct, qshed_pct = extract_per_bus_loadshed(solution, math)
            loadshed_data_per_func[fair_func] = (bus_names, pshed_pct, qshed_pct)

            # Compute per-load pshed fractions for fairness metrics
            pshed_fractions = Float64[]
            total_pshed = 0.0
            total_pd_served = 0.0
            for (lid, load_sol) in solution["load"]
                pd_orig = sum(math["load"][lid]["pd"])
                pd_s = load_sol["pd"]
                pd_served = isa(pd_s, AbstractArray) ? sum(pd_s) : pd_s
                total_pd_served += pd_served
                pshed_load = pd_orig - pd_served
                total_pshed += pshed_load
                push!(pshed_fractions, pd_orig > 0 ? pshed_load / pd_orig : 0.0)
            end

            pct_shed = (total_pshed / total_demand) * 100
            pct_served = (total_pd_served / total_demand) * 100

            # Post-hoc fairness metrics on per-load shed fractions
            jain_val = jains_index(pshed_fractions)
            palma_val = palma_ratio_posthoc(pshed_fractions)
            gini_val = gini_index(pshed_fractions)

            println("shed=$(round(pct_shed, digits=2))%, jain=$(round(jain_val, digits=4)), palma=$(round(palma_val, digits=4)), gini=$(round(gini_val, digits=4))")

            push!(results_df, (
                case, fair_func, total_demand, total_pshed, total_pd_served,
                pct_shed, pct_served, mld_result["objective"],
                jain_val, palma_val, gini_val
            ))

            # Save solution CSV
            load_df = DataFrame(
                load_id = Int[], pd_served = Float64[], pshed = Float64[],
                qd_served = Float64[], qshed = Float64[]
            )
            for (lid, ldata) in solution["load"]
                pd_s = ldata["pd"]
                pd_served = isa(pd_s, AbstractArray) ? sum(pd_s) : pd_s
                qd_s = ldata["qd"]
                qd_served = isa(qd_s, AbstractArray) ? sum(qd_s) : qd_s
                pd_orig = sum(math["load"][lid]["pd"])
                qd_orig = sum(math["load"][lid]["qd"])
                push!(load_df, (parse(Int, lid), pd_served, pd_orig - pd_served, qd_served, qd_orig - qd_served))
            end
            sort!(load_df, :load_id)
            CSV.write(joinpath(out_dir, "solution_load_$(case)_$(fair_func).csv"), load_df)
        end

        # Generate voltage comparison plot
        plot_voltage_per_bus_comparison(
            voltage_data_per_func,
            joinpath(out_dir, "voltage_per_phase_$case.svg");
            title = "Bus Voltage Per Phase (Integer): $case",
            target_buses = TARGET_BUSES
        )

        # Generate loadshed comparison plot
        plot_loadshed_per_bus_comparison(
            loadshed_data_per_func,
            joinpath(out_dir, "loadshed_per_bus_$case.svg");
            title = "Load Shed Per Bus (Integer): $case"
        )
    end

    # Save summary CSV
    csv_path = joinpath(out_dir, "single_level_comparison.csv")
    CSV.write(csv_path, results_df)
    println("\n  Results saved to: $csv_path")

    return results_df
end

# ============================================================
# RUN
# ============================================================
println("=" ^ 60)
println("SINGLE-LEVEL FAIRNESS COMPARISON (Integer only)")
println("=" ^ 60)

results_integer = run_integer_comparison(CASES, FAIR_SOLVE_INTEGER, gurobi_solver, integer_dir)

# ============================================================
# SUMMARY
# ============================================================
println("\n" * "=" ^ 60)
println("RESULTS SUMMARY")
println("=" ^ 60)
println(results_integer)

println("\n" * "=" ^ 60)
println("COMPARISON COMPLETE")
println("Results in: $integer_dir")
println("=" ^ 60)
