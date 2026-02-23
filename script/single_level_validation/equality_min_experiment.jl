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
using PowerPlots
using DataFrames
using CSV
using Dates

include("../src/implementation/network_setup.jl")
include("../src/implementation/lower_level_mld.jl")
include("../src/implementation/palma_relaxation.jl")
include("../src/implementation/other_fair_funcs.jl")
include("../src/implementation/random_rounding.jl")
include("../src/implementation/export_results.jl")
include("../src/implementation/visualization.jl")

"""
This experiment tests the equality_min (min-max fairness) objective.
It runs the MLD with the equality_min objective and compares results
with the standard relaxed MLD formulation.

The equality_min objective minimizes the maximum load shed across all loads,
promoting fair distribution of load shedding.

Results are saved in:
    results/date/case/control_exp/equality_min/
"""

ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

case = "motivation_a"
gen_cap = 1000.0  # Limited generation to force load shedding

# Inputs: case file path, generation capacity, list of critical load IDs
eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/$case.dss", gen_cap, [])
powerplot(eng)

# Create the folder structure for results
today = Dates.today()

if !isdir("results")
    mkdir("results")
end

date_folder = "results/$(today)"
if !isdir(date_folder)
    mkdir(date_folder)
end

case_folder = "results/$(today)/$case"
if !isdir(case_folder)
    mkdir(case_folder)
end

control_exp_folder = "results/$(today)/$case/control_exp"
if !isdir(control_exp_folder)
    mkdir(control_exp_folder)
end

equality_min_folder = "results/$(today)/$case/control_exp/equality_min"
if !isdir(equality_min_folder)
    mkdir(equality_min_folder)
end

# Create output DataFrames
block_df = DataFrame(BlockID=Int64[], GenCap=Float64[], EqualityMinStatus=Float64[], RelaxedStatus=Float64[], BlockSwitches=Set{Int64}[])
switch_df = DataFrame(SwitchID=Int64[], GenCap=Float64[], EqualityMinStatus=Float64[], RelaxedStatus=Float64[], SwitchName=String[])
load_shed_df = DataFrame(LoadID=Int64[], BusName=String[], OriginalDemand=Float64[], EqualityMinShed=Float64[], RelaxedShed=Float64[])

# Get the reference data for the blocks
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_equality_min; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
ref = mld_model.ref[:it][:pmd][:nw][0]

# Build bus ID to name mapping
bus_id_to_name = Dict{Int,String}()
for (bus_id_str, bus) in math["bus"]
    bus_id = parse(Int, bus_id_str)
    bus_name = split(string(bus["source_id"]), ".")[end]
    bus_id_to_name[bus_id] = bus_name
end

# Run the equality_min MLD
println("\n" * "="^60)
println("Running Equality Min (Min-Max Fairness) MLD")
println("="^60)
mld_equality_min = FairLoadDelivery.solve_mc_mld_equality_min(math, ipopt)

println("Termination status: $(mld_equality_min["termination_status"])")
println("Objective value: $(mld_equality_min["objective"])")

# Run the standard relaxed MLD for comparison
println("\n" * "="^60)
println("Running Standard Relaxed MLD (for comparison)")
println("="^60)
mld_relaxed = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)

println("Termination status: $(mld_relaxed["termination_status"])")
println("Objective value: $(mld_relaxed["objective"])")

# Extract and compare results
println("\n" * "="^60)
println("RESULTS COMPARISON")
println("="^60)

# Block status comparison
block_eq = mld_equality_min["solution"]["block"]
block_relax = mld_relaxed["solution"]["block"]

for (id, block) in block_eq
    block_id = parse(Int, id)
    push!(block_df.BlockID, block_id)
    push!(block_df.EqualityMinStatus, block["status"])
    push!(block_df.RelaxedStatus, block_relax[id]["status"])
    push!(block_df.GenCap, gen_cap)
    push!(block_df.BlockSwitches, ref[:block_switches][block_id])
end

# Switch status comparison
switch_eq = mld_equality_min["solution"]["switch"]
switch_relax = mld_relaxed["solution"]["switch"]

for (id, switch) in switch_eq
    push!(switch_df.SwitchID, parse(Int, id))
    push!(switch_df.GenCap, gen_cap)
    push!(switch_df.EqualityMinStatus, switch["state"])
    push!(switch_df.RelaxedStatus, switch_relax[id]["state"])
    push!(switch_df.SwitchName, math["switch"][id]["name"])
end

# Load shed comparison
println("\nLoad Shed Comparison:")
println("-"^80)
println("Load ID | Bus Name | Original (kW) | Equality Min Shed | Relaxed Shed")
println("-"^80)

global total_eq_shed = 0.0
global total_relax_shed = 0.0
global total_demand = 0.0

for (load_id, load) in math["load"]
    global total_eq_shed, total_relax_shed, total_demand
    load_bus = load["load_bus"]
    bus_name = get(bus_id_to_name, load_bus, string(load_bus))
    original_demand = sum(load["pd"])

    eq_shed = sum(mld_equality_min["solution"]["load"][load_id]["pshed"])
    relax_shed = sum(mld_relaxed["solution"]["load"][load_id]["pshed"])

    push!(load_shed_df.LoadID, parse(Int, load_id))
    push!(load_shed_df.BusName, bus_name)
    push!(load_shed_df.OriginalDemand, original_demand)
    push!(load_shed_df.EqualityMinShed, eq_shed)
    push!(load_shed_df.RelaxedShed, relax_shed)

    total_eq_shed += eq_shed
    total_relax_shed += relax_shed
    total_demand += original_demand

    println("   $load_id    |   $bus_name   |    $(round(original_demand, digits=2))    |      $(round(eq_shed, digits=2))      |    $(round(relax_shed, digits=2))")
end

println("-"^80)
println("TOTAL   |          |    $(round(total_demand, digits=2))    |      $(round(total_eq_shed, digits=2))      |    $(round(total_relax_shed, digits=2))")
println("="^60)

# Check voltage bounds
println("\n=== Voltage Check (v² should be in [0.81, 1.21]) ===")
v_min_sq = 0.81
v_max_sq = 1.21
voltage_violations = 0

for (bus_id, bus_soln) in mld_equality_min["solution"]["bus"]
    if haskey(bus_soln, "w")
        for (idx, w) in enumerate(bus_soln["w"])
            if w < v_min_sq - 1e-4 || w > v_max_sq + 1e-4
                voltage_violations += 1
                bus_name = get(bus_id_to_name, parse(Int, bus_id), bus_id)
                println("  Voltage violation at bus $bus_name (phase $idx): w = $(round(w, digits=4))")
            end
        end
    end
end

if voltage_violations == 0
    println("  ✓ No voltage violations detected")
else
    println("  ✗ Total voltage violations: $voltage_violations")
end

# Check switch states and power flow consistency
println("\n=== Switch State and Power Flow Check ===")
for (sw_id, sw_soln) in mld_equality_min["solution"]["switch"]
    sw_name = math["switch"][sw_id]["name"]
    state = sw_soln["state"]
    println("  Switch $sw_id ($sw_name): state = $(round(state, digits=4))")
end

# Check if only generator block loads are served when switches are off
all_switches_off = all(sw["state"] < 0.5 for (_, sw) in mld_equality_min["solution"]["switch"])
if all_switches_off
    println("\n  All switches are OFF. Checking if only generator block loads are served...")
    gen_blocks = [b for (b, gens) in ref[:block_gens] if !isempty(gens)]
    println("  Generator blocks: $gen_blocks")

    for (load_id, load_soln) in mld_equality_min["solution"]["load"]
        load_bus = math["load"][load_id]["load_bus"]
        load_block = ref[:bus_block_map][load_bus]
        bus_name = get(bus_id_to_name, load_bus, string(load_bus))

        pshed = sum(load_soln["pshed"])
        original_demand = sum(math["load"][load_id]["pd"])
        served = original_demand - pshed

        if load_block in gen_blocks
            println("    Load $load_id (bus $bus_name, block $load_block): GENERATOR BLOCK - served = $(round(served, digits=2)) kW")
        else
            if served > 1e-4
                println("    ✗ Load $load_id (bus $bus_name, block $load_block): NON-GEN BLOCK but served = $(round(served, digits=2)) kW (UNEXPECTED)")
            else
                println("    ✓ Load $load_id (bus $bus_name, block $load_block): NON-GEN BLOCK - correctly shed")
            end
        end
    end
end

# Save results
println("\n=== Saving Results ===")
CSV.write(joinpath(equality_min_folder, "block_comparison.csv"), block_df)
CSV.write(joinpath(equality_min_folder, "switch_comparison.csv"), switch_df)
CSV.write(joinpath(equality_min_folder, "load_shed_comparison.csv"), load_shed_df)
println("  Saved block_comparison.csv")
println("  Saved switch_comparison.csv")
println("  Saved load_shed_comparison.csv")

# Create network visualization
plot_network_load_shed(mld_equality_min["solution"], math;
    output_file=joinpath(equality_min_folder, "network_load_shed.svg"),
    layout=:ieee13)
println("  Saved network_load_shed.svg")

# Create comparison bar chart
load_ids = sort(parse.(Int, collect(keys(mld_equality_min["solution"]["load"]))))
eq_sheds = [sum(mld_equality_min["solution"]["load"][string(id)]["pshed"]) for id in load_ids]
relax_sheds = [sum(mld_relaxed["solution"]["load"][string(id)]["pshed"]) for id in load_ids]

# Create side-by-side comparison using scatter plot
x_eq = load_ids .- 0.15
x_relax = load_ids .+ 0.15

comparison_plot = plot(
    xlabel = "Load ID",
    ylabel = "Load Shed (kW)",
    title = "Load Shed Comparison: Equality Min vs Relaxed",
    legend = :topright,
    xticks = load_ids
)
bar!(comparison_plot, x_eq, eq_sheds, bar_width=0.3, label="Equality Min", color=:blue)
bar!(comparison_plot, x_relax, relax_sheds, bar_width=0.3, label="Relaxed", color=:orange)
savefig(comparison_plot, joinpath(equality_min_folder, "load_shed_comparison.svg"))
println("  Saved load_shed_comparison.svg")

println("\n" * "="^60)
println("Experiment Complete!")
println("Results saved in: $equality_min_folder")
println("="^60)
