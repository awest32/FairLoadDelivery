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

include("../src/implementation/network_setup.jl")
include("../src/implementation/lower_level_mld.jl")
include("../src/implementation/palma_relaxation.jl")
include("../src/implementation/other_fair_funcs.jl")
include("../src/implementation/random_rounding.jl")
include("../src/implementation/export_results.jl")
include("../src/implementation/visualization.jl")

"""
This experiment compares different MLD formulations:
1. Efficient MLD (Relaxed) - Standard weighted max load served objective
2. Efficient MLD (Integer) - Integer version with binary switches/blocks
3. Equality Min MLD (Relaxed) - Min-max fairness objective (continuous)
4. Equality Min MLD (Integer) - Min-max fairness objective with binary switches/blocks
5. Proportional Fairness MLD (Relaxed) - Nash bargaining / log-sum objective (continuous)
6. Proportional Fairness MLD (Integer) - Nash bargaining / log-sum objective with binary switches/blocks

Results are saved in:
    results/date/case/control_exp/mld_comparison/
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

comparison_folder = "results/$(today)/$case/control_exp/mld_comparison"
if !isdir(comparison_folder)
    mkdir(comparison_folder)
end

# Create output DataFrames
block_df = DataFrame(
    BlockID=Int64[],
    GenCap=Float64[],
    EfficientRelaxed=Float64[],
    EfficientInteger=Float64[],
    EqualityMinRelaxed=Float64[],
    EqualityMinInteger=Float64[],
    PropFairRelaxed=Float64[],
    PropFairInteger=Float64[],
    BlockSwitches=Set{Int64}[]
)

switch_df = DataFrame(
    SwitchID=Int64[],
    GenCap=Float64[],
    EfficientRelaxed=Float64[],
    EfficientInteger=Float64[],
    EqualityMinRelaxed=Float64[],
    EqualityMinInteger=Float64[],
    PropFairRelaxed=Float64[],
    PropFairInteger=Float64[],
    SwitchName=String[]
)

load_shed_df = DataFrame(
    LoadID=Int64[],
    BusName=String[],
    OriginalDemand=Float64[],
    EfficientRelaxedShed=Float64[],
    EfficientIntegerShed=Float64[],
    EqualityMinRelaxedShed=Float64[],
    EqualityMinIntegerShed=Float64[],
    PropFairRelaxedShed=Float64[],
    PropFairIntegerShed=Float64[]
)

summary_df = DataFrame(
    Formulation=String[],
    TotalDemand=Float64[],
    TotalShed=Float64[],
    TotalServed=Float64[],
    PercentServed=Float64[],
    MaxShed=Float64[],
    MinShed=Float64[],
    ShedVariance=Float64[],
    TerminationStatus=String[],
    ObjectiveValue=Float64[]
)

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

# ============================================================
# Run all MLD formulations
# ============================================================

println("\n" * "="^70)
println("RUNNING MLD COMPARISON EXPERIMENT")
println("Case: $case | Generation Capacity: $gen_cap kW")
println("="^70)

# 1. Efficient MLD (Relaxed)
println("\n[1/6] Running Efficient MLD (Relaxed)...")
mld_eff_relaxed = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
println("  Termination: $(mld_eff_relaxed["termination_status"])")
println("  Objective: $(round(mld_eff_relaxed["objective"], digits=4))")

# 2. Efficient MLD (Integer)
println("\n[2/6] Running Efficient MLD (Integer)...")
mld_eff_integer = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)
println("  Termination: $(mld_eff_integer["termination_status"])")
println("  Objective: $(round(mld_eff_integer["objective"], digits=4))")

# 3. Equality Min MLD (Relaxed)
println("\n[3/6] Running Equality Min MLD (Relaxed)...")
mld_eq_relaxed = FairLoadDelivery.solve_mc_mld_equality_min(math, ipopt)
println("  Termination: $(mld_eq_relaxed["termination_status"])")
println("  Objective: $(round(mld_eq_relaxed["objective"], digits=4))")

# 4. Equality Min MLD (Integer)
println("\n[4/6] Running Equality Min MLD (Integer)...")
mld_eq_integer = FairLoadDelivery.solve_mc_mld_equality_min_integer(math, gurobi)
println("  Termination: $(mld_eq_integer["termination_status"])")
println("  Objective: $(round(mld_eq_integer["objective"], digits=4))")

# 5. Proportional Fairness MLD (Relaxed)
println("\n[5/6] Running Proportional Fairness MLD (Relaxed)...")
mld_pf_relaxed = FairLoadDelivery.solve_mc_mld_proportional_fairness(math, ipopt)
println("  Termination: $(mld_pf_relaxed["termination_status"])")
println("  Objective: $(round(mld_pf_relaxed["objective"], digits=4))")

# 6. Proportional Fairness MLD (Integer)
println("\n[6/6] Running Proportional Fairness MLD (Integer)...")
mld_pf_integer = FairLoadDelivery.solve_mc_mld_proportional_fairness_integer(math, gurobi)
println("  Termination: $(mld_pf_integer["termination_status"])")
println("  Objective: $(round(mld_pf_integer["objective"], digits=4))")

# ============================================================
# Extract and compare results
# ============================================================

println("\n" * "="^70)
println("RESULTS COMPARISON")
println("="^70)

# Block status comparison
println("\n--- Block Status Comparison ---")
for (id, block) in mld_eff_relaxed["solution"]["block"]
    block_id = parse(Int, id)
    eff_relax = block["status"]
    eff_int = mld_eff_integer["solution"]["block"][id]["status"]
    eq_relax = mld_eq_relaxed["solution"]["block"][id]["status"]
    eq_int = mld_eq_integer["solution"]["block"][id]["status"]
    pf_relax = mld_pf_relaxed["solution"]["block"][id]["status"]
    pf_int = mld_pf_integer["solution"]["block"][id]["status"]

    push!(block_df.BlockID, block_id)
    push!(block_df.GenCap, gen_cap)
    push!(block_df.EfficientRelaxed, eff_relax)
    push!(block_df.EfficientInteger, eff_int)
    push!(block_df.EqualityMinRelaxed, eq_relax)
    push!(block_df.EqualityMinInteger, eq_int)
    push!(block_df.PropFairRelaxed, pf_relax)
    push!(block_df.PropFairInteger, pf_int)
    push!(block_df.BlockSwitches, ref[:block_switches][block_id])

    println("  Block $block_id: Eff=$(round(eff_relax, digits=2))/$(round(eff_int, digits=2)), EqMin=$(round(eq_relax, digits=2))/$(round(eq_int, digits=2)), PropFair=$(round(pf_relax, digits=2))/$(round(pf_int, digits=2))")
end

# Switch status comparison
println("\n--- Switch Status Comparison ---")
for (id, switch) in mld_eff_relaxed["solution"]["switch"]
    sw_name = math["switch"][id]["name"]
    eff_relax = switch["state"]
    eff_int = mld_eff_integer["solution"]["switch"][id]["state"]
    eq_relax = mld_eq_relaxed["solution"]["switch"][id]["state"]
    eq_int = mld_eq_integer["solution"]["switch"][id]["state"]
    pf_relax = mld_pf_relaxed["solution"]["switch"][id]["state"]
    pf_int = mld_pf_integer["solution"]["switch"][id]["state"]

    push!(switch_df.SwitchID, parse(Int, id))
    push!(switch_df.GenCap, gen_cap)
    push!(switch_df.EfficientRelaxed, eff_relax)
    push!(switch_df.EfficientInteger, eff_int)
    push!(switch_df.EqualityMinRelaxed, eq_relax)
    push!(switch_df.EqualityMinInteger, eq_int)
    push!(switch_df.PropFairRelaxed, pf_relax)
    push!(switch_df.PropFairInteger, pf_int)
    push!(switch_df.SwitchName, sw_name)

    println("  Switch $id ($sw_name): Eff=$(round(eff_relax, digits=2))/$(round(eff_int, digits=2)), EqMin=$(round(eq_relax, digits=2))/$(round(eq_int, digits=2)), PropFair=$(round(pf_relax, digits=2))/$(round(pf_int, digits=2))")
end

# Load shed comparison
println("\n--- Load Shed Comparison ---")
println("-"^140)
println("Load ID | Bus Name | Demand (kW) | Eff-Relax | Eff-Int | EqMin-Relax | EqMin-Int | PF-Relax | PF-Int")
println("-"^140)

global total_demand = 0.0
global eff_relax_total_shed = 0.0
global eff_int_total_shed = 0.0
global eq_relax_total_shed = 0.0
global eq_int_total_shed = 0.0
global pf_relax_total_shed = 0.0
global pf_int_total_shed = 0.0

eff_relax_sheds = Float64[]
eff_int_sheds = Float64[]
eq_relax_sheds = Float64[]
eq_int_sheds = Float64[]
pf_relax_sheds = Float64[]
pf_int_sheds = Float64[]

for (load_id, load) in math["load"]
    global total_demand, eff_relax_total_shed, eff_int_total_shed, eq_relax_total_shed, eq_int_total_shed, pf_relax_total_shed, pf_int_total_shed

    load_bus = load["load_bus"]
    bus_name = get(bus_id_to_name, load_bus, string(load_bus))
    original_demand = sum(load["pd"])

    eff_relax_shed = sum(mld_eff_relaxed["solution"]["load"][load_id]["pshed"])
    eff_int_shed = sum(mld_eff_integer["solution"]["load"][load_id]["pshed"])
    eq_relax_shed = sum(mld_eq_relaxed["solution"]["load"][load_id]["pshed"])
    eq_int_shed = sum(mld_eq_integer["solution"]["load"][load_id]["pshed"])
    pf_relax_shed = sum(mld_pf_relaxed["solution"]["load"][load_id]["pshed"])
    pf_int_shed = sum(mld_pf_integer["solution"]["load"][load_id]["pshed"])

    push!(load_shed_df.LoadID, parse(Int, load_id))
    push!(load_shed_df.BusName, bus_name)
    push!(load_shed_df.OriginalDemand, original_demand)
    push!(load_shed_df.EfficientRelaxedShed, eff_relax_shed)
    push!(load_shed_df.EfficientIntegerShed, eff_int_shed)
    push!(load_shed_df.EqualityMinRelaxedShed, eq_relax_shed)
    push!(load_shed_df.EqualityMinIntegerShed, eq_int_shed)
    push!(load_shed_df.PropFairRelaxedShed, pf_relax_shed)
    push!(load_shed_df.PropFairIntegerShed, pf_int_shed)

    push!(eff_relax_sheds, eff_relax_shed)
    push!(eff_int_sheds, eff_int_shed)
    push!(eq_relax_sheds, eq_relax_shed)
    push!(eq_int_sheds, eq_int_shed)
    push!(pf_relax_sheds, pf_relax_shed)
    push!(pf_int_sheds, pf_int_shed)

    total_demand += original_demand
    eff_relax_total_shed += eff_relax_shed
    eff_int_total_shed += eff_int_shed
    eq_relax_total_shed += eq_relax_shed
    eq_int_total_shed += eq_int_shed
    pf_relax_total_shed += pf_relax_shed
    pf_int_total_shed += pf_int_shed

    println("   $load_id    |   $bus_name   |   $(round(original_demand, digits=2))   |   $(round(eff_relax_shed, digits=2))   |   $(round(eff_int_shed, digits=2))   |    $(round(eq_relax_shed, digits=2))    |   $(round(eq_int_shed, digits=2))   |   $(round(pf_relax_shed, digits=2))   |   $(round(pf_int_shed, digits=2))")
end

println("-"^140)
println("TOTAL   |          |   $(round(total_demand, digits=2))   |   $(round(eff_relax_total_shed, digits=2))   |   $(round(eff_int_total_shed, digits=2))   |    $(round(eq_relax_total_shed, digits=2))    |   $(round(eq_int_total_shed, digits=2))   |   $(round(pf_relax_total_shed, digits=2))   |   $(round(pf_int_total_shed, digits=2))")

# Calculate statistics for summary
function calc_stats(sheds, total_shed, total_demand, result)
    served = total_demand - total_shed
    pct_served = (served / total_demand) * 100
    max_shed = maximum(sheds)
    min_shed = minimum(filter(x -> x > 1e-6, sheds); init=0.0)
    variance = length(sheds) > 1 ? Statistics.var(sheds) : 0.0
    return (served, pct_served, max_shed, min_shed, variance)
end

# Add to summary DataFrame
eff_relax_stats = calc_stats(eff_relax_sheds, eff_relax_total_shed, total_demand, mld_eff_relaxed)
eff_int_stats = calc_stats(eff_int_sheds, eff_int_total_shed, total_demand, mld_eff_integer)
eq_relax_stats = calc_stats(eq_relax_sheds, eq_relax_total_shed, total_demand, mld_eq_relaxed)
eq_int_stats = calc_stats(eq_int_sheds, eq_int_total_shed, total_demand, mld_eq_integer)
pf_relax_stats = calc_stats(pf_relax_sheds, pf_relax_total_shed, total_demand, mld_pf_relaxed)
pf_int_stats = calc_stats(pf_int_sheds, pf_int_total_shed, total_demand, mld_pf_integer)

push!(summary_df, ("Efficient (Relaxed)", total_demand, eff_relax_total_shed, eff_relax_stats[1], eff_relax_stats[2], eff_relax_stats[3], eff_relax_stats[4], eff_relax_stats[5], string(mld_eff_relaxed["termination_status"]), mld_eff_relaxed["objective"]))
push!(summary_df, ("Efficient (Integer)", total_demand, eff_int_total_shed, eff_int_stats[1], eff_int_stats[2], eff_int_stats[3], eff_int_stats[4], eff_int_stats[5], string(mld_eff_integer["termination_status"]), mld_eff_integer["objective"]))
push!(summary_df, ("Equality Min (Relaxed)", total_demand, eq_relax_total_shed, eq_relax_stats[1], eq_relax_stats[2], eq_relax_stats[3], eq_relax_stats[4], eq_relax_stats[5], string(mld_eq_relaxed["termination_status"]), mld_eq_relaxed["objective"]))
push!(summary_df, ("Equality Min (Integer)", total_demand, eq_int_total_shed, eq_int_stats[1], eq_int_stats[2], eq_int_stats[3], eq_int_stats[4], eq_int_stats[5], string(mld_eq_integer["termination_status"]), mld_eq_integer["objective"]))
push!(summary_df, ("Prop Fair (Relaxed)", total_demand, pf_relax_total_shed, pf_relax_stats[1], pf_relax_stats[2], pf_relax_stats[3], pf_relax_stats[4], pf_relax_stats[5], string(mld_pf_relaxed["termination_status"]), mld_pf_relaxed["objective"]))
push!(summary_df, ("Prop Fair (Integer)", total_demand, pf_int_total_shed, pf_int_stats[1], pf_int_stats[2], pf_int_stats[3], pf_int_stats[4], pf_int_stats[5], string(mld_pf_integer["termination_status"]), mld_pf_integer["objective"]))

# Print summary
println("\n" * "="^70)
println("SUMMARY STATISTICS")
println("="^70)
println("\nFormulation          | Total Shed | % Served | Max Shed | Min Shed | Variance")
println("-"^80)
for row in eachrow(summary_df)
    println("$(rpad(row.Formulation, 20)) | $(round(row.TotalShed, digits=2)) kW   | $(round(row.PercentServed, digits=1))%    | $(round(row.MaxShed, digits=2))    | $(round(row.MinShed, digits=2))    | $(round(row.ShedVariance, digits=2))")
end

# ============================================================
# Voltage Check
# ============================================================
println("\n" * "="^70)
println("VOLTAGE BOUNDS CHECK (v² ∈ [0.81, 1.21])")
println("="^70)

v_min_sq = 0.81
v_max_sq = 1.21

for (name, result) in [("Efficient (Relaxed)", mld_eff_relaxed), ("Efficient (Integer)", mld_eff_integer), ("Equality Min (Relaxed)", mld_eq_relaxed), ("Equality Min (Integer)", mld_eq_integer), ("Prop Fair (Relaxed)", mld_pf_relaxed), ("Prop Fair (Integer)", mld_pf_integer)]
    violations = 0
    for (bus_id, bus_soln) in result["solution"]["bus"]
        if haskey(bus_soln, "w")
            for (idx, w) in enumerate(bus_soln["w"])
                if w < v_min_sq - 1e-4 || w > v_max_sq + 1e-4
                    violations += 1
                end
            end
        end
    end
    status = violations == 0 ? "✓ PASS" : "✗ FAIL ($violations violations)"
    println("  $name: $status")
end

# ============================================================
# Switch Consistency Check
# ============================================================
println("\n" * "="^70)
println("SWITCH CONSISTENCY CHECK")
println("="^70)

for (name, result) in [("Efficient (Relaxed)", mld_eff_relaxed), ("Efficient (Integer)", mld_eff_integer), ("Equality Min (Relaxed)", mld_eq_relaxed), ("Equality Min (Integer)", mld_eq_integer), ("Prop Fair (Relaxed)", mld_pf_relaxed), ("Prop Fair (Integer)", mld_pf_integer)]
    all_off = all(sw["state"] < 0.5 for (_, sw) in result["solution"]["switch"])

    if all_off
        println("\n  $name: All switches OFF")
        gen_blocks = [b for (b, gens) in ref[:block_gens] if !isempty(gens)]

        correctly_shed = true
        for (load_id, load_soln) in result["solution"]["load"]
            load_bus = math["load"][load_id]["load_bus"]
            load_block = ref[:bus_block_map][load_bus]
            served = sum(math["load"][load_id]["pd"]) - sum(load_soln["pshed"])

            if !(load_block in gen_blocks) && served > 1e-4
                correctly_shed = false
                bus_name = get(bus_id_to_name, load_bus, string(load_bus))
                println("    ✗ Load $load_id (bus $bus_name) in non-gen block but served $(round(served, digits=2)) kW")
            end
        end

        if correctly_shed
            println("    ✓ Only generator block loads are served (correct)")
        end
    else
        println("\n  $name: Some switches ON")
        for (sw_id, sw) in result["solution"]["switch"]
            if sw["state"] >= 0.5
                sw_name = math["switch"][sw_id]["name"]
                println("    Switch $sw_id ($sw_name): ON (state=$(round(sw["state"], digits=3)))")
            end
        end
    end
end

# ============================================================
# Save Results
# ============================================================
println("\n" * "="^70)
println("SAVING RESULTS")
println("="^70)

CSV.write(joinpath(comparison_folder, "block_comparison.csv"), block_df)
CSV.write(joinpath(comparison_folder, "switch_comparison.csv"), switch_df)
CSV.write(joinpath(comparison_folder, "load_shed_comparison.csv"), load_shed_df)
CSV.write(joinpath(comparison_folder, "summary_statistics.csv"), summary_df)
println("  Saved CSV files")

# Create network visualizations for each formulation
for (name, result, filename) in [
    ("Efficient Relaxed", mld_eff_relaxed, "network_efficient_relaxed.svg"),
    ("Efficient Integer", mld_eff_integer, "network_efficient_integer.svg"),
    ("Equality Min Relaxed", mld_eq_relaxed, "network_equality_min_relaxed.svg"),
    ("Equality Min Integer", mld_eq_integer, "network_equality_min_integer.svg"),
    ("Prop Fair Relaxed", mld_pf_relaxed, "network_prop_fair_relaxed.svg"),
    ("Prop Fair Integer", mld_pf_integer, "network_prop_fair_integer.svg")
]
    plot_network_load_shed(result["solution"], math;
        output_file=joinpath(comparison_folder, filename),
        layout=:ieee13)
    println("  Saved $filename")
end

# Create comparison bar chart
load_ids = sort(parse.(Int, collect(keys(mld_eff_relaxed["solution"]["load"]))))

comparison_plot = plot(
    xlabel = "Load ID",
    ylabel = "Load Shed (kW)",
    title = "Load Shed Comparison: All Formulations",
    legend = :outertopright,
    xticks = load_ids,
    size = (1100, 500)
)

x_offset = 0.14
bar!(comparison_plot, load_ids .- 2.5*x_offset, eff_relax_sheds, bar_width=0.12, label="Efficient (Relaxed)", color=:blue)
bar!(comparison_plot, load_ids .- 1.5*x_offset, eff_int_sheds, bar_width=0.12, label="Efficient (Integer)", color=:lightblue)
bar!(comparison_plot, load_ids .- 0.5*x_offset, eq_relax_sheds, bar_width=0.12, label="Equality Min (Relaxed)", color=:orange)
bar!(comparison_plot, load_ids .+ 0.5*x_offset, eq_int_sheds, bar_width=0.12, label="Equality Min (Integer)", color=:yellow)
bar!(comparison_plot, load_ids .+ 1.5*x_offset, pf_relax_sheds, bar_width=0.12, label="Prop Fair (Relaxed)", color=:green)
bar!(comparison_plot, load_ids .+ 2.5*x_offset, pf_int_sheds, bar_width=0.12, label="Prop Fair (Integer)", color=:lightgreen)

savefig(comparison_plot, joinpath(comparison_folder, "load_shed_comparison_all.svg"))
println("  Saved load_shed_comparison_all.svg")

# Create fairness comparison plot (shed distribution)
fairness_plot = plot(
    xlabel = "Load ID",
    ylabel = "Load Shed (kW)",
    title = "Shed Distribution by Formulation",
    legend = :outertopright,
    size = (1100, 450)
)

scatter!(fairness_plot, load_ids, eff_relax_sheds, label="Efficient (Relaxed)", marker=:circle, markersize=8, color=:blue)
scatter!(fairness_plot, load_ids, eff_int_sheds, label="Efficient (Integer)", marker=:circle, markersize=6, color=:lightblue)
scatter!(fairness_plot, load_ids, eq_relax_sheds, label="Equality Min (Relaxed)", marker=:diamond, markersize=8, color=:orange)
scatter!(fairness_plot, load_ids, eq_int_sheds, label="Equality Min (Integer)", marker=:diamond, markersize=6, color=:yellow)
scatter!(fairness_plot, load_ids, pf_relax_sheds, label="Prop Fair (Relaxed)", marker=:square, markersize=8, color=:green)
scatter!(fairness_plot, load_ids, pf_int_sheds, label="Prop Fair (Integer)", marker=:square, markersize=6, color=:lightgreen)

# Add horizontal lines for equality min target (max shed)
if !isempty(eq_relax_sheds)
    max_eq_relax_shed = maximum(eq_relax_sheds)
    hline!(fairness_plot, [max_eq_relax_shed], label="EqMin-Relax Max", linestyle=:dash, color=:orange, linewidth=1)
end

savefig(fairness_plot, joinpath(comparison_folder, "shed_distribution.svg"))
println("  Saved shed_distribution.svg")

println("\n" * "="^70)
println("EXPERIMENT COMPLETE!")
println("Results saved in: $comparison_folder")
println("="^70)
