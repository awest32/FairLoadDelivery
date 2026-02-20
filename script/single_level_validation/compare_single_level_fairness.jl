"""
    compare_single_level_fairness.jl

    Compare single-level MLD results across fairness function formulations
    for all motivation cases. Solves both relaxed (Ipopt) and integer (Gurobi)
    formulations. Produces voltage and loadshed comparison plots.
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
const CASES = ["motivation_c"]#["motivation_a", "motivation_b",
const GEN_CAP = 0.8

# Relaxed formulations (continuous, Ipopt)
const FAIR_SOLVE_RELAXED = [
    "efficiency"    => FairLoadDelivery.solve_mc_mld_switch_relaxed,
    "equality_min"  => FairLoadDelivery.solve_mc_mld_equality_min,
    "proportional"  => FairLoadDelivery.solve_mc_mld_proportional_fairness,
     "min_max"       => FairLoadDelivery.solve_mc_mld_min_max,
    #"jain"          => FairLoadDelivery.solve_mc_mld_jain,
]

# Integer formulations (MIP, Gurobi)
const FAIR_SOLVE_INTEGER = [
    "efficiency"    => FairLoadDelivery.solve_mc_mld_switch_integer,
    "equality_min"  => FairLoadDelivery.solve_mc_mld_equality_min_integer,
    "proportional"  => FairLoadDelivery.solve_mc_mld_proportional_fairness_integer,
    "min_max"       => FairLoadDelivery.solve_mc_mld_min_max_integer,
   # "jain"          => FairLoadDelivery.solve_mc_mld_jain_integer,
]

# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = Gurobi.Optimizer

# Save results
save_dir = "results/$(Dates.today())/single_level_fairness"
relaxed_dir = joinpath(save_dir, "relaxed")
integer_dir = joinpath(save_dir, "integer")
mkpath(relaxed_dir)
mkpath(integer_dir)

# Target buses for voltage plots (load buses of interest)
const TARGET_BUSES = ["670","632","645","671","634","646","611","675","652","692"]

# ============================================================
# SOLVE + EXTRACT + PLOT FOR ONE FORMULATION TYPE
# ============================================================
function run_formulation(
    cases, fair_solve_funcs, solver, formulation_label::String, out_dir::String
)
    results_df = DataFrame(
        case = String[],
        fair_func = String[],
        total_demand = Float64[],
        total_pshed = Float64[],
        total_pd_served = Float64[],
        pct_shed = Float64[],
        pct_served = Float64[],
        objective = Float64[]
    )

    for case in cases
        println("\n>>> Processing case: $case ($formulation_label)")

        eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/$case.dss", GEN_CAP, [])

        total_demand = sum(sum(load["pd"]) for (_, load) in math["load"])
        println("    Total demand: $(round(total_demand, digits=4))")

        voltage_data_per_func = Dict{String, Dict{String, Dict{Int, Float64}}}()
        loadshed_data_per_func = Dict{String, Tuple{Vector{String}, Vector{Float64}, Vector{Float64}}}()

        for (fair_func, solve_func) in fair_solve_funcs
            print("  $fair_func: ")

            mld_result = solve_func(math, solver)

            if !(mld_result["termination_status"] in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED])
                @warn "Non-optimal termination for $case/$fair_func ($formulation_label): $(mld_result["termination_status"])"
                continue
            end

            solution = mld_result["solution"]

            # Extract per-bus voltage
            voltage_data_per_func[fair_func] = extract_voltage_by_bus_name(solution, math)

            # Extract per-bus loadshed
            bus_names, pshed_pct, qshed_pct = extract_per_bus_loadshed(solution, math)
            loadshed_data_per_func[fair_func] = (bus_names, pshed_pct, qshed_pct)

            # Compute totals for summary
            total_pshed = 0.0
            total_pd_served = 0.0
            for (lid, load_sol) in solution["load"]
                pd_orig = sum(math["load"][lid]["pd"])
                pd_s = load_sol["pd"]
                pd_served = isa(pd_s, AbstractArray) ? sum(pd_s) : pd_s
                total_pd_served += pd_served
                total_pshed += (pd_orig - pd_served)
            end

            pct_shed = (total_pshed / total_demand) * 100
            pct_served = (total_pd_served / total_demand) * 100
            println("shed=$(round(pct_shed, digits=2))%, served=$(round(pct_served, digits=2))%")

            push!(results_df, (
                case, fair_func, total_demand, total_pshed, total_pd_served,
                pct_shed, pct_served, mld_result["objective"]
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
            title = "Bus Voltage Per Phase ($formulation_label): $case",
            target_buses = TARGET_BUSES
        )

        # Generate loadshed comparison plot
        plot_loadshed_per_bus_comparison(
            loadshed_data_per_func,
            joinpath(out_dir, "loadshed_per_bus_$case.svg");
            title = "Load Shed Per Bus ($formulation_label): $case"
        )
    end

    # Save summary CSV
    csv_path = joinpath(out_dir, "single_level_comparison.csv")
    CSV.write(csv_path, results_df)
    println("\n  Results saved to: $csv_path")

    return results_df
end

# ============================================================
# RUN BOTH FORMULATIONS
# ============================================================
println("=" ^ 60)
println("SINGLE-LEVEL FAIRNESS COMPARISON")
println("=" ^ 60)

println("\n" * "=" ^ 60)
println("RELAXED FORMULATIONS (Ipopt)")
println("=" ^ 60)
results_relaxed = run_formulation(CASES, FAIR_SOLVE_RELAXED, ipopt_solver, "Relaxed", relaxed_dir)

println("\n" * "=" ^ 60)
println("INTEGER FORMULATIONS (Gurobi)")
println("=" ^ 60)
results_integer = run_formulation(CASES, FAIR_SOLVE_INTEGER, gurobi_solver, "Integer", integer_dir)

# ============================================================
# COMBINED SUMMARY
# ============================================================
println("\n" * "=" ^ 60)
println("RELAXED SUMMARY")
println("=" ^ 60)
println(results_relaxed)

println("\n" * "=" ^ 60)
println("INTEGER SUMMARY")
println("=" ^ 60)
println(results_integer)

println("\n" * "=" ^ 60)
println("COMPARISON COMPLETE")
println("Relaxed results in: $relaxed_dir")
println("Integer results in: $integer_dir")
println("=" ^ 60)
