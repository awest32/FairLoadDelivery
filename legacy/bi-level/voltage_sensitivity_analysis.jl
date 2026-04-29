"""
    voltage_sensitivity_analysis.jl

    Sensitivity analysis for sourcebus voltage in the bilevel formulation.
    Analyzes how varying the source voltage affects load shedding and bus voltages.
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
include("../../src/implementation/lower_level_mld.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/random_rounding.jl")
include("../../src/implementation/visualization.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASE = "motivation_a"
const LS_PERCENT = 0.9
const ITERATIONS = 5
const N_ROUNDS = 2
const N_BERNOULLI_SAMPLES = 5

# Voltage sensitivity parameters
const VOLTAGE_SCALE_VALUES = 0.7:0.05:1.8  # From 0.9 pu to 1.1 pu
const FAIR_FUNCS = ["proportional", "efficiency", "min_max", "equality_min", "jain"]

# Solver
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

"""
    apply_voltage_scale!(math::Dict, vscale::Float64)

Modify the math dictionary to scale the source voltage.
Updates the generator voltage setpoint and source bus voltage.
"""
function apply_voltage_scale!(math::Dict, vscale::Float64)
    source_bus_id = nothing

    # Update generator voltage setpoint
    for (i, gen) in math["gen"]
        if gen["source_id"] == "voltage_source.source"
            gen["vg"] = gen["vg"] .* vscale
            source_bus_id = gen["gen_bus"]
            println("    Applied vscale=$vscale to generator $i, vg=$(gen["vg"])")
        end
    end

    # Update source bus voltage magnitude if present
    if source_bus_id !== nothing && haskey(math["bus"], string(source_bus_id))
        bus = math["bus"][string(source_bus_id)]
        if haskey(bus, "vm")
            bus["vm"] = bus["vm"] .* vscale
            println("    Updated bus $source_bus_id vm=$(bus["vm"])")
        end
        # Also store vscale in bus for reference
        bus["vscale"] = vscale
    end

    # Store vscale in math dict for use in update_network
    math["vscale"] = vscale

    return math
end

function get_total_demand(math::Dict)
    total_pd = 0.0
    for (_, load) in math["load"]
        total_pd += sum(load["pd"])
    end
    return total_pd
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

"""
    extract_bus_voltages(solution::Dict, math::Dict)

Extract bus voltage statistics from solution.
Returns (min_voltage, max_voltage, avg_voltage, voltage_dict)
"""
function extract_bus_voltages(solution::Dict, math::Dict)
    if !haskey(solution, "bus")
        return NaN, NaN, NaN, Dict()
    end

    voltage_dict = Dict{String, Float64}()
    all_voltages = Float64[]

    # Build bus ID to name mapping
    bus_id_to_name = Dict{Int,String}()
    for (bus_id_str, bus) in math["bus"]
        bus_id = parse(Int, bus_id_str)
        bus_name = split(string(bus["source_id"]), ".")[end]
        bus_id_to_name[bus_id] = bus_name
    end

    for (bus_id_str, bus_sol) in solution["bus"]
        bus_id = parse(Int, bus_id_str)
        bus_name = get(bus_id_to_name, bus_id, bus_id_str)

        v_pu = 1.0
        if haskey(bus_sol, "w")
            w_vals = values(bus_sol["w"])
            v_pu = sqrt(mean(collect(w_vals)))
        elseif haskey(bus_sol, "vm")
            vm_vals = values(bus_sol["vm"])
            v_pu = mean(collect(vm_vals))
        end

        voltage_dict[bus_name] = v_pu
        push!(all_voltages, v_pu)
    end

    if isempty(all_voltages)
        return NaN, NaN, NaN, voltage_dict
    end

    return minimum(all_voltages), maximum(all_voltages), mean(all_voltages), voltage_dict
end

"""
    extract_per_bus_shed(solution::Dict, math::Dict)

Extract load shed aggregated by bus.
Returns Dict{bus_name => shed_kw}
"""
function extract_per_bus_shed(solution::Dict, math::Dict)
    if !haskey(solution, "load")
        return Dict{String, Float64}()
    end

    # Build bus ID to name mapping
    bus_id_to_name = Dict{Int,String}()
    for (bus_id_str, bus) in math["bus"]
        bus_id = parse(Int, bus_id_str)
        bus_name = split(string(bus["source_id"]), ".")[end]
        bus_id_to_name[bus_id] = bus_name
    end

    bus_shed = Dict{String, Float64}()

    for (load_id, load) in math["load"]
        load_bus = load["load_bus"]
        bus_name = get(bus_id_to_name, load_bus, string(load_bus))

        if !haskey(bus_shed, bus_name)
            bus_shed[bus_name] = 0.0
        end

        if haskey(solution["load"], load_id) && haskey(solution["load"][load_id], "pshed")
            bus_shed[bus_name] += sum(solution["load"][load_id]["pshed"])
        end
    end

    return bus_shed
end

# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

function run_voltage_sensitivity_analysis()
    results = DataFrame(
        vscale = Float64[],
        fair_func = String[],
        total_demand = Float64[],
        final_pshed = Float64[],
        final_pd_served = Float64[],
        pct_shed = Float64[],
        pct_served = Float64[],
        min_voltage = Float64[],
        max_voltage = Float64[],
        avg_voltage = Float64[],
        objective = Float64[]
    )

    # Per-bus results for detailed analysis
    per_bus_voltages = Dict{Float64, Dict{String, Dict{String, Float64}}}()  # vscale => fair_func => bus => voltage
    per_bus_shed = Dict{Float64, Dict{String, Dict{String, Float64}}}()      # vscale => fair_func => bus => shed

    # Store solutions for visualization
    solutions_for_viz = Dict{Float64, Dict{String, Tuple{Dict, Dict}}}()  # vscale => fair_func => (solution, math)

    println("=" ^ 70)
    println("VOLTAGE SENSITIVITY ANALYSIS")
    println("Case: $CASE")
    println("Voltage scales: $VOLTAGE_SCALE_VALUES")
    println("=" ^ 70)

    for vscale in VOLTAGE_SCALE_VALUES
        println("\n>>> Source Voltage Scale: $(vscale) pu")

        # Setup network and apply voltage scale
        eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/$CASE.dss", LS_PERCENT, [])
        apply_voltage_scale!(math, vscale)

        fair_weights = Float64[load["weight"] for (_, load) in math["load"]]
        total_demand = get_total_demand(math)

        println("    Total demand: $(round(total_demand, digits=4))")

        per_bus_voltages[vscale] = Dict{String, Dict{String, Float64}}()
        per_bus_shed[vscale] = Dict{String, Dict{String, Float64}}()
        solutions_for_viz[vscale] = Dict{String, Tuple{Dict, Dict}}()

        for fair_func in FAIR_FUNCS
            print("  $fair_func: ")

            try
                # Run bilevel relaxation
                math_relaxed, pshed_lower, pshed_upper, weight_ids, final_wts = run_bilevel_relaxed(
                    math, ITERATIONS, fair_weights, fair_func
                )

                # Run random rounding
                math_out, mld_results = run_random_rounding(
                    math_relaxed, N_ROUNDS, N_BERNOULLI_SAMPLES, ipopt_solver
                )

                if isempty(mld_results)
                    println("No feasible solution")
                    push!(results, (vscale, fair_func, total_demand, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN))
                    continue
                end

                # Find best solution
                best_idx, best_mld = find_best_mld_solution(mld_results)
                final_pshed, final_pd_served = extract_load_shed(best_mld)

                # Extract voltage data
                min_v, max_v, avg_v, v_dict = extract_bus_voltages(best_mld["solution"], math)
                per_bus_voltages[vscale][fair_func] = v_dict

                # Extract per-bus shed
                shed_dict = extract_per_bus_shed(best_mld["solution"], math)
                per_bus_shed[vscale][fair_func] = shed_dict

                # Store for visualization
                solutions_for_viz[vscale][fair_func] = (best_mld["solution"], math)

                pct_shed = (final_pshed / total_demand) * 100
                pct_served = (final_pd_served / total_demand) * 100

                println("shed=$(round(pct_shed, digits=2))%, V_min=$(round(min_v, digits=3)), V_max=$(round(max_v, digits=3))")

                push!(results, (
                    vscale,
                    fair_func,
                    total_demand,
                    final_pshed,
                    final_pd_served,
                    pct_shed,
                    pct_served,
                    min_v,
                    max_v,
                    avg_v,
                    best_mld["objective"]
                ))

            catch e
                println("ERROR: $e")
                push!(results, (vscale, fair_func, total_demand, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN))
            end
        end
    end

    return results, per_bus_voltages, per_bus_shed, solutions_for_viz
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

function plot_voltage_sensitivity_results(results::DataFrame, per_bus_voltages::Dict,
                                          per_bus_shed::Dict, save_dir::String)

    # Plot 1: Percent served vs source voltage for each fairness function
    p1 = plot(
        xlabel = "Source Voltage (pu)",
        ylabel = "Percent Served (%)",
        title = "Load Served vs Source Voltage",
        legend = :outertopright,
        size = (900, 500)
    )

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.pct_served), results)
        if !isempty(subset)
            plot!(p1, subset.vscale, subset.pct_served,
                  label=FAIR_FUNC_LABELS[fair_func], marker=:circle, linewidth=2,
                  color=FAIR_FUNC_COLORS[fair_func])
        end
    end
    savefig(p1, joinpath(save_dir, "voltage_sensitivity_served.svg"))

    # Plot 2: Min bus voltage vs source voltage
    p2 = plot(
        xlabel = "Source Voltage (pu)",
        ylabel = "Minimum Bus Voltage (pu)",
        title = "Minimum Bus Voltage vs Source Voltage",
        legend = :outertopright,
        size = (900, 500)
    )

    # Add voltage limit lines
    hline!(p2, [0.95], color=:red, linestyle=:dash, label="Lower limit (0.95)")
    hline!(p2, [1.05], color=:red, linestyle=:dash, label="Upper limit (1.05)")

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.min_voltage), results)
        if !isempty(subset)
            plot!(p2, subset.vscale, subset.min_voltage,
                  label=FAIR_FUNC_LABELS[fair_func], marker=:circle, linewidth=2,
                  color=FAIR_FUNC_COLORS[fair_func])
        end
    end
    savefig(p2, joinpath(save_dir, "voltage_sensitivity_min_voltage.svg"))

    # Plot 3: Max bus voltage vs source voltage
    p3 = plot(
        xlabel = "Source Voltage (pu)",
        ylabel = "Maximum Bus Voltage (pu)",
        title = "Maximum Bus Voltage vs Source Voltage",
        legend = :outertopright,
        size = (900, 500)
    )

    hline!(p3, [0.95], color=:red, linestyle=:dash, label="Lower limit (0.95)")
    hline!(p3, [1.05], color=:red, linestyle=:dash, label="Upper limit (1.05)")

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.max_voltage), results)
        if !isempty(subset)
            plot!(p3, subset.vscale, subset.max_voltage,
                  label=FAIR_FUNC_LABELS[fair_func], marker=:circle, linewidth=2,
                  color=FAIR_FUNC_COLORS[fair_func])
        end
    end
    savefig(p3, joinpath(save_dir, "voltage_sensitivity_max_voltage.svg"))

    # Plot 4: Voltage range (max - min) vs source voltage
    p4 = plot(
        xlabel = "Source Voltage (pu)",
        ylabel = "Voltage Range (max - min) (pu)",
        title = "Voltage Spread vs Source Voltage",
        legend = :outertopright,
        size = (900, 500)
    )

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.min_voltage), results)
        if !isempty(subset)
            voltage_range = subset.max_voltage .- subset.min_voltage
            plot!(p4, subset.vscale, voltage_range,
                  label=FAIR_FUNC_LABELS[fair_func], marker=:circle, linewidth=2,
                  color=FAIR_FUNC_COLORS[fair_func])
        end
    end
    savefig(p4, joinpath(save_dir, "voltage_sensitivity_range.svg"))

    # Plot 5: Total shed vs source voltage
    p5 = plot(
        xlabel = "Source Voltage (pu)",
        ylabel = "Total Load Shed (kW)",
        title = "Total Load Shed vs Source Voltage",
        legend = :outertopright,
        size = (900, 500)
    )

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.final_pshed), results)
        if !isempty(subset)
            plot!(p5, subset.vscale, subset.final_pshed,
                  label=FAIR_FUNC_LABELS[fair_func], marker=:circle, linewidth=2,
                  color=FAIR_FUNC_COLORS[fair_func])
        end
    end
    savefig(p5, joinpath(save_dir, "voltage_sensitivity_total_shed.svg"))

    # Plot 6: Per-bus voltage heatmap for efficiency function
    plot_per_bus_voltage_heatmap(per_bus_voltages, "efficiency", save_dir)

    # Plot 7: Per-bus shed heatmap for efficiency function
    plot_per_bus_shed_heatmap(per_bus_shed, "efficiency", save_dir)

    return p1, p2, p3, p4, p5
end

function plot_per_bus_voltage_heatmap(per_bus_voltages::Dict, fair_func::String, save_dir::String)
    # Get all voltage scales and bus names
    vscales = sort(collect(keys(per_bus_voltages)))

    # Get bus names from first available result
    bus_names = String[]
    for vscale in vscales
        if haskey(per_bus_voltages[vscale], fair_func)
            bus_names = sort(collect(keys(per_bus_voltages[vscale][fair_func])))
            break
        end
    end

    if isempty(bus_names)
        return nothing
    end

    # Build matrix
    matrix = fill(NaN, length(bus_names), length(vscales))
    for (j, vscale) in enumerate(vscales)
        if haskey(per_bus_voltages[vscale], fair_func)
            for (i, bus) in enumerate(bus_names)
                if haskey(per_bus_voltages[vscale][fair_func], bus)
                    matrix[i, j] = per_bus_voltages[vscale][fair_func][bus]
                end
            end
        end
    end

    p = heatmap(
        string.(vscales),
        bus_names,
        matrix,
        xlabel = "Source Voltage (pu)",
        ylabel = "Bus",
        title = "Bus Voltages: $(FAIR_FUNC_LABELS[fair_func])",
        color = :RdYlGn,
        clims = (0.9, 1.1),
        size = (800, 600)
    )

    savefig(p, joinpath(save_dir, "heatmap_voltage_$(fair_func).svg"))
    return p
end

function plot_per_bus_shed_heatmap(per_bus_shed::Dict, fair_func::String, save_dir::String)
    vscales = sort(collect(keys(per_bus_shed)))

    bus_names = String[]
    for vscale in vscales
        if haskey(per_bus_shed[vscale], fair_func)
            bus_names = sort(collect(keys(per_bus_shed[vscale][fair_func])))
            break
        end
    end

    if isempty(bus_names)
        return nothing
    end

    matrix = fill(NaN, length(bus_names), length(vscales))
    for (j, vscale) in enumerate(vscales)
        if haskey(per_bus_shed[vscale], fair_func)
            for (i, bus) in enumerate(bus_names)
                if haskey(per_bus_shed[vscale][fair_func], bus)
                    matrix[i, j] = per_bus_shed[vscale][fair_func][bus]
                end
            end
        end
    end

    p = heatmap(
        string.(vscales),
        bus_names,
        matrix,
        xlabel = "Source Voltage (pu)",
        ylabel = "Bus",
        title = "Load Shed per Bus (kW): $(FAIR_FUNC_LABELS[fair_func])",
        color = :YlOrRd,
        size = (800, 600)
    )

    savefig(p, joinpath(save_dir, "heatmap_shed_$(fair_func).svg"))
    return p
end

# ============================================================
# RUN ANALYSIS
# ============================================================

println("\nStarting voltage sensitivity analysis at $(now())...\n")

results, per_bus_voltages, per_bus_shed, solutions_for_viz = run_voltage_sensitivity_analysis()

# Save results
save_dir = joinpath("results", Dates.format(today(), "yyyy-mm-dd"), "voltage_sensitivity")
mkpath(save_dir)

csv_path = joinpath(save_dir, "voltage_sensitivity_results.csv")
CSV.write(csv_path, results)
println("\nResults saved to: $csv_path")

# Write summary
summary_path = joinpath(save_dir, "voltage_sensitivity_summary.txt")
open(summary_path, "w") do io
    println(io, "VOLTAGE SENSITIVITY ANALYSIS SUMMARY")
    println(io, "Generated: $(now())")
    println(io, "Case: $CASE")
    println(io, "Voltage scales tested: $VOLTAGE_SCALE_VALUES")
    println(io, "=" ^ 70)

    println(io, "\nRESULTS BY SOURCE VOLTAGE")
    println(io, "-" ^ 70)

    for vscale in VOLTAGE_SCALE_VALUES
        subset = filter(row -> row.vscale == vscale && !isnan(row.pct_served), results)
        if !isempty(subset)
            println(io, "\nSource Voltage: $(vscale) pu")
            for row in eachrow(subset)
                println(io, "  $(rpad(FAIR_FUNC_LABELS[row.fair_func], 15)): " *
                           "$(round(row.pct_served, digits=1))% served, " *
                           "V_min=$(round(row.min_voltage, digits=3)), " *
                           "V_max=$(round(row.max_voltage, digits=3))")
            end
        end
    end

    println(io, "\n" * "=" ^ 70)
    println(io, "BEST SOURCE VOLTAGE BY FAIRNESS FUNCTION")
    println(io, "-" ^ 70)

    for fair_func in FAIR_FUNCS
        subset = filter(row -> row.fair_func == fair_func && !isnan(row.pct_served), results)
        if !isempty(subset)
            best_row = subset[argmax(subset.pct_served), :]
            println(io, "\n$(FAIR_FUNC_LABELS[fair_func]):")
            println(io, "  Best voltage: $(best_row.vscale) pu")
            println(io, "  Served: $(round(best_row.pct_served, digits=2))%")
            println(io, "  V_min: $(round(best_row.min_voltage, digits=3)), V_max: $(round(best_row.max_voltage, digits=3))")
        end
    end
end
println("Summary saved to: $summary_path")

# Print summary to console
println("\n" * "=" ^ 70)
println("SUMMARY: Results by source voltage")
println("=" ^ 70)

for vscale in VOLTAGE_SCALE_VALUES
    subset = filter(row -> row.vscale == vscale && !isnan(row.pct_served), results)
    if !isempty(subset)
        println("\nSource Voltage: $(vscale) pu")
        for row in eachrow(subset)
            println("  $(rpad(FAIR_FUNC_LABELS[row.fair_func], 15)): " *
                   "$(round(row.pct_served, digits=1))% served, " *
                   "V_min=$(round(row.min_voltage, digits=3)), " *
                   "V_max=$(round(row.max_voltage, digits=3))")
        end
    end
end

# Generate plots
println("\n" * "=" ^ 70)
println("GENERATING PLOTS")
println("=" ^ 70)

plot_voltage_sensitivity_results(results, per_bus_voltages, per_bus_shed, save_dir)
println("Plots saved to: $save_dir")

# Generate network visualizations for extreme voltage cases
println("\n" * "=" ^ 70)
println("GENERATING NETWORK VISUALIZATIONS")
println("=" ^ 70)

viz_vscales = [minimum(VOLTAGE_SCALE_VALUES), 1.0, maximum(VOLTAGE_SCALE_VALUES)]
for vscale in viz_vscales
    if haskey(solutions_for_viz, vscale)
        for fair_func in ["efficiency"]  # Just efficiency for visualization
            if haskey(solutions_for_viz[vscale], fair_func)
                solution, math = solutions_for_viz[vscale][fair_func]
                filename = "network_v$(vscale)_$(fair_func).svg"
                plot_network_load_shed(solution, math;
                    output_file=joinpath(save_dir, filename),
                    layout=:ieee13)
                println("  Saved $filename")
            end
        end
    end
end

println("\n" * "=" ^ 70)
println("VOLTAGE SENSITIVITY ANALYSIS COMPLETE")
println("=" ^ 70)
