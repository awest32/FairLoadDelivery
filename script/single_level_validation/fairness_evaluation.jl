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
using LinearAlgebra, SparseArrays
using Statistics
using PowerPlots
using DataFrames
using CSV
using Dates

include("../../src/implementation/network_setup.jl")
include("../../src/implementation/lower_level_mld.jl")
include("../../src/implementation/load_shed_as_parameter.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/random_rounding.jl")
include("../../src/implementation/export_results.jl")
include("../../src/implementation/visualization.jl")

"""
Flexible fairness evaluation script.

This script allows you to:
1. Select which MLD formulations to run
2. Compute multiple fairness metrics on each solution
3. Compare results across formulations

Available formulations:
- :efficient_relaxed    - Max total load served (relaxed)
- :efficient_integer    - Max total load served (integer)
- :equality_min_relaxed - Min-max fairness (relaxed)
- :equality_min_integer - Min-max fairness (integer)
- :prop_fair_relaxed    - Proportional fairness / Nash bargaining (relaxed)
- :prop_fair_integer    - Proportional fairness / Nash bargaining (integer)
- :jain_relaxed         - Maximize Jain's fairness index (relaxed)
- :jain_integer         - Maximize Jain's fairness index (integer)
- :palma_relaxed        - Minimize Palma ratio (relaxed)
- :palma_integer        - Minimize Palma ratio (integer)

Usage:
    Set `formulations_to_run` to a vector of symbols for the formulations you want to compare.
"""

# ============================================================
# CONFIGURATION - Edit this section
# ============================================================

# Select which formulations to run
formulations_to_run = [
    :efficient_integer,
    #:equality_min_integer,
    :min_max_integer,
    :prop_fair_integer,
    :jain_integer,
    #:palma_integer,
]

case = "motivation_c"
gen_cap = 0.6  # Limited generation to force load shedding

# ============================================================
# SETUP
# ============================================================

ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# Inputs: case file path, generation capacity, list of critical load IDs
eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/$case.dss", gen_cap, 1.03, [])

# Create the folder structure for results
today = Dates.today()

for folder in ["results", "results/$(today)", "results/$(today)/$case",
               "results/$(today)/$case/fairness_eval"]
    !isdir(folder) && mkdir(folder)
end

eval_folder = "results/$(today)/$case/fairness_eval"

# Get the reference data for the blocks
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_min_max_integer;
                                  ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
ref = mld_model.ref[:it][:pmd][:nw][0]

#mld_soln = FairLoadDelivery.solve_mc_mld_min_max_integer(math, gurobi)
# Build bus ID to name mapping
bus_id_to_name = Dict{Int,String}()
for (bus_id_str, bus) in math["bus"]
    bus_id = parse(Int, bus_id_str)
    bus_name = split(string(bus["source_id"]), ".")[end]
    bus_id_to_name[bus_id] = bus_name
end

# ============================================================
# FAIRNESS METRICS FUNCTIONS
# ============================================================

"""
Compute Jain's fairness index on a vector of allocations.
J(x) = (sum(x))^2 / (n * sum(x^2))
Returns 1 for perfect equality, 1/n for maximum inequality.
"""
function compute_jains_index(x::Vector{Float64})
    n = length(x)
    n == 0 && return NaN
    sum_x = sum(x)
    sum_x2 = sum(xi^2 for xi in x)
    sum_x2 == 0 && return NaN
    return (sum_x^2) / (n * sum_x2)
end

"""
Compute Palma ratio: ratio of top 10% to bottom 40% allocations.
Lower is more fair (1.0 means equal share).
"""
function compute_palma_ratio(x::Vector{Float64})
    n = length(x)
    n < 3 && return NaN  # Need enough data points
    sorted_x = sort(x)
    top_10_idx = max(1, ceil(Int, 0.9 * n))
    bottom_40_idx = max(1, floor(Int, 0.4 * n))
    top_10_sum = sum(sorted_x[top_10_idx:end])
    bottom_40_sum = sum(sorted_x[1:bottom_40_idx])
    bottom_40_sum == 0 && return Inf
    return top_10_sum / bottom_40_sum
end

"""
Compute Gini coefficient.
0 = perfect equality, 1 = perfect inequality.
"""
function compute_gini(x::Vector{Float64})
    n = length(x)
    n == 0 && return NaN
    sorted_x = sort(x)
    sum_x = sum(sorted_x)
    sum_x == 0 && return NaN
    gini = (2 * sum(i * sorted_x[i] for i in 1:n) / (n * sum_x)) - ((n + 1) / n)
    return gini
end

"""
Compute coefficient of variation (CV = std/mean).
Lower is more fair.
"""
function compute_cv(x::Vector{Float64})
    n = length(x)
    n == 0 && return NaN
    mean_x = mean(x)
    mean_x == 0 && return Inf
    return std(x) / mean_x
end

"""
Compute all fairness metrics for a solution.
Returns a Dict with all metrics.
"""
function compute_all_metrics(result::Dict, math::Dict)
    # Extract load served fractions
    served_fractions = Float64[]
    shed_amounts = Float64[]
    served_amounts = Float64[]

    for (load_id, load) in math["load"]
        original_demand = sum(load["pd"])
        pshed = sum(result["solution"]["load"][load_id]["pshed"])
        served = original_demand - pshed

        push!(shed_amounts, pshed)
        push!(served_amounts, served)
        if original_demand > 1e-6
            push!(served_fractions, served / original_demand)
        end
    end

    total_demand = sum(sum(load["pd"]) for (_, load) in math["load"])
    total_shed = sum(shed_amounts)
    total_served = total_demand - total_shed

    return Dict(
        # Basic metrics
        "total_demand" => total_demand,
        "total_shed" => total_shed,
        "total_served" => total_served,
        "percent_served" => 100 * total_served / total_demand,

        # Shed-based metrics
        "max_shed" => maximum(shed_amounts),
        "min_shed" => minimum(shed_amounts),
        "shed_variance" => Statistics.var(shed_amounts),
        "shed_std" => Statistics.std(shed_amounts),

        # Fairness metrics on served fractions
        "jains_index" => compute_jains_index(served_fractions),
        "palma_ratio" => compute_palma_ratio(served_fractions),
        "gini_coefficient" => compute_gini(served_fractions),
        "cv_served" => compute_cv(served_fractions),

        # Fairness metrics on shed amounts
        "jains_index_shed" => compute_jains_index(shed_amounts),
        "cv_shed" => compute_cv(shed_amounts),

        # Raw data for further analysis
        "served_fractions" => served_fractions,
        "shed_amounts" => shed_amounts,
        "served_amounts" => served_amounts
    )
end

# ============================================================
# FORMULATION SOLVERS
# ============================================================

formulation_solvers = Dict(
    :efficient_integer => () -> FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi),
    :equality_min_integer => () -> FairLoadDelivery.solve_mc_mld_equality_min_integer(math, gurobi),
    :min_max_integer => () -> FairLoadDelivery.solve_mc_mld_min_max_integer(math, gurobi),
    :prop_fair_integer => () -> FairLoadDelivery.solve_mc_mld_proportional_fairness_integer(math, gurobi),
    :jain_integer => () -> FairLoadDelivery.solve_mc_mld_jain_integer(math, gurobi),
    #:palma_integer => () -> FairLoadDelivery.solve_mc_mld_palma_integer(math, gurobi),
)

formulation_names = Dict(
    :efficient_integer => "Efficient (Integer)",
    :equality_min_integer => "Equality Min (Integer)",
    :min_max_integer => "Min Max (Integer)",
    :prop_fair_integer => "Prop Fair (Integer)",
    :jain_integer => "Jain Max (Integer)",
    # :palma_relaxed => "Palma Min (Relaxed)",
    # :palma_integer => "Palma Min (Integer)",
)

# ============================================================
# RUN FORMULATIONS
# ============================================================

println("\n" * "="^70)
println("FAIRNESS EVALUATION EXPERIMENT")
println("Case: $case | Generation Capacity: $gen_cap kW")
println("Formulations: $(length(formulations_to_run))")
println("="^70)

results = Dict{Symbol, Dict}()
metrics = Dict{Symbol, Dict}()

for (i, form) in enumerate(formulations_to_run)
    name = get(formulation_names, form, string(form))
    println("\n[$i/$(length(formulations_to_run))] Running $name...")

    if !haskey(formulation_solvers, form)
        println("  ⚠ Formulation not implemented yet, skipping")
        continue
    end

    result = formulation_solvers[form]()
    results[form] = result

    println("  Termination: $(result["termination_status"])")
    println("  Objective: $(round(result["objective"], digits=4))")

    # Compute metrics
    metrics[form] = compute_all_metrics(result, math)
    println("  Total Served: $(round(metrics[form]["percent_served"], digits=1))%")
    println("  Jain's Index: $(round(metrics[form]["jains_index"], digits=4))")
end

# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

println("\n" * "="^70)
println("FAIRNESS METRICS COMPARISON")
println("="^70)

# Create summary DataFrame
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

for form in formulations_to_run
    !haskey(metrics, form) && continue
    m = metrics[form]
    push!(summary_df, (
        get(formulation_names, form, string(form)),
        round(m["percent_served"], digits=2),
        round(m["total_shed"], digits=2),
        round(m["max_shed"], digits=2),
        round(m["shed_variance"], digits=2),
        round(m["jains_index"], digits=4),
        round(m["palma_ratio"], digits=4),
        round(m["gini_coefficient"], digits=4),
        round(m["cv_served"], digits=4)
    ))
end

# Print table
println("\n")
println(summary_df)

# ============================================================
# INTERPRETATION GUIDE
# ============================================================

println("\n" * "="^70)
println("METRIC INTERPRETATION GUIDE")
println("="^70)
println("""
┌─────────────────┬──────────────────────────────────────────────────────┐
│ Metric          │ Interpretation                                       │
├─────────────────┼──────────────────────────────────────────────────────┤
│ TotalServed_%   │ Higher = more efficient (less total shed)            │
│ MaxShed_kW      │ Lower = better min-max fairness                      │
│ ShedVariance    │ Lower = more equal distribution of shed              │
│ Jain's Index    │ Higher = more fair (1.0 = perfect equality)          │
│ Palma Ratio     │ Lower = more fair (1.0 = equal top/bottom shares)    │
│ Gini Coeff      │ Lower = more fair (0.0 = perfect equality)           │
│ CV (Served)     │ Lower = more equal served fractions                  │
└─────────────────┴──────────────────────────────────────────────────────┘
""")

# ============================================================
# SAVE RESULTS
# ============================================================

println("\n" * "="^70)
println("SAVING RESULTS")
println("="^70)

CSV.write(joinpath(eval_folder, "fairness_metrics_comparison.csv"), summary_df)
println("  Saved fairness_metrics_comparison.csv")

# Per-load shed comparison
load_shed_df = DataFrame(LoadID = Int64[], BusName = String[], OriginalDemand = Float64[])
for form in formulations_to_run
    load_shed_df[!, Symbol(form)] = Float64[]
end

for (load_id, load) in math["load"]
    row = [parse(Int, load_id), get(bus_id_to_name, load["load_bus"], string(load["load_bus"])), sum(load["pd"])]
    for form in formulations_to_run
        if haskey(results, form)
            push!(row, sum(results[form]["solution"]["load"][load_id]["pshed"]))
        else
            push!(row, NaN)
        end
    end
    push!(load_shed_df, tuple(row...))
end

CSV.write(joinpath(eval_folder, "load_shed_by_formulation.csv"), load_shed_df)
println("  Saved load_shed_by_formulation.csv")

# ============================================================
# SWITCH UTILIZATION SUMMARY
# ============================================================

println("\n" * "="^70)
println("SWITCH CURRENT RATING SUMMARY")
println("="^70)

# Print header
println("\nSwitch ratings from network data:")
println("-"^50)

for (switch_id, switch) in math["switch"]
    sw_name = get(switch, "name", switch_id)
    f_bus = switch["f_bus"]
    t_bus = switch["t_bus"]
    current_rating = get(switch, "current_rating", [Inf])
    f_bus_name = get(bus_id_to_name, f_bus, string(f_bus))
    t_bus_name = get(bus_id_to_name, t_bus, string(t_bus))

    println("Switch $sw_name ($f_bus_name -> $t_bus_name): Rating = $(round.(current_rating, digits=1)) A")
end

# Calculate and print utilization for each formulation
println("\n" * "-"^70)
println("Switch Utilization by Formulation:")
println("-"^70)

for form in formulations_to_run
    !haskey(results, form) && continue
    solution = results[form]["solution"]
    name = get(formulation_names, form, string(form))

    println("\n[$name]")

    for (switch_id, switch) in math["switch"]
        sw_name = get(switch, "name", switch_id)
        f_bus = switch["f_bus"]
        current_rating = get(switch, "current_rating", [Inf])
        f_connections = get(switch, "f_connections", [1, 2, 3])

        if !haskey(solution, "switch") || !haskey(solution["switch"], switch_id)
            println("  $sw_name: No solution data")
            continue
        end

        switch_sol = solution["switch"][switch_id]
        state = get(switch_sol, "state", [1.0])
        is_closed = all(state .> 0.5)

        if !is_closed
            println("  $sw_name: OPEN")
            continue
        end

        # Get power flow
        pf = get(switch_sol, "psw_fr", get(switch_sol, "pf", [0.0]))
        qf = get(switch_sol, "qsw_fr", get(switch_sol, "qf", [0.0]))

        # Get voltage at from bus
        w_vals = Dict{Int,Float64}()
        if haskey(solution, "bus") && haskey(solution["bus"], string(f_bus))
            bus_sol = solution["bus"][string(f_bus)]
            if haskey(bus_sol, "w")
                w_vals = bus_sol["w"]
            elseif haskey(bus_sol, "vm")
                for (k, v) in bus_sol["vm"]
                    w_vals[k] = v^2
                end
            end
        end

        # Calculate utilization per phase
        util_per_phase = Float64[]
        for (idx, (p, q)) in enumerate(zip(pf, qf))
            s_squared = p^2 + q^2
            conn = idx <= length(f_connections) ? f_connections[idx] : idx
            w = get(w_vals, conn, 1.0)
            rating = idx <= length(current_rating) ? current_rating[idx] : current_rating[1]

            if rating > 0 && rating < Inf && w > 1e-6
                util = sqrt(s_squared) / (sqrt(w) * rating) * 100
                push!(util_per_phase, util)
            end
        end

        if !isempty(util_per_phase)
            max_util = maximum(util_per_phase)
            println("  $sw_name: $(round(max_util, digits=1))% utilization (max across phases)")
        else
            println("  $sw_name: Unable to calculate utilization")
        end
    end
end

println("\n" * "="^70)

# ============================================================
# VISUALIZATION
# ============================================================

# Bar chart comparing key metrics
if length(formulations_to_run) > 0 && length(metrics) > 0
    form_names = [get(formulation_names, f, string(f)) for f in formulations_to_run if haskey(metrics, f)]

    # Jain's Index comparison
    jain_values = [metrics[f]["jains_index"] for f in formulations_to_run if haskey(metrics, f)]
    p1 = bar(form_names, jain_values,
             title="Jain's Fairness Index (higher = more fair)",
             ylabel="Jain's Index", xlabel="",
             legend=false, rotation=15, color=:blue)

    # Total Served comparison
    served_values = [metrics[f]["percent_served"] for f in formulations_to_run if haskey(metrics, f)]
    p2 = bar(form_names, served_values,
             title="Total Load Served %",
             ylabel="% Served", xlabel="",
             legend=false, rotation=15, color=:green)

    # Max Shed comparison
    maxshed_values = [metrics[f]["max_shed"] for f in formulations_to_run if haskey(metrics, f)]
    p3 = bar(form_names, maxshed_values,
             title="Maximum Load Shed (lower = more fair)",
             ylabel="Max Shed (kW)", xlabel="",
             legend=false, rotation=15, color=:orange)

    # Palma Ratio comparison
    palma_values = [metrics[f]["palma_ratio"] for f in formulations_to_run if haskey(metrics, f)]
    p4 = bar(form_names, palma_values,
             title="Palma Ratio (lower = more fair)",
             ylabel="Palma Ratio", xlabel="",
             legend=false, rotation=15, color=:red)

    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 700))
    savefig(combined_plot, joinpath(eval_folder, "fairness_metrics_comparison.svg"))
    println("  Saved fairness_metrics_comparison.svg")
end

# Network visualizations for each formulation
for form in formulations_to_run
    !haskey(results, form) && continue
    filename = "network_$(form).svg"
    plot_network_load_shed(results[form]["solution"], math;
        output_file=joinpath(eval_folder, filename),
        layout=:ieee13)
    println("  Saved $filename")
end

println("\n" * "="^70)
println("EVALUATION COMPLETE!")
println("Results saved in: $eval_folder")
println("="^70)
