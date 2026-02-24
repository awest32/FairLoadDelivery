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
using LinearAlgebra,SparseArrays
using PowerPlots
using DataFrames
using CSV
using Plots
using Dates

include("../../src/implementation/visualization.jl")

"""
This is the control case.
Select the desired network.
Run the control MLD with switches and load blocks for the integer and the relaxed MLD.
Extract the results from these cases.
    1. Table: blocks, block states: integer and continuous
    2. Table: switches, switch states: integer and continuous
    3. Plot: pshed v. block: integer and continuous
    4. Plot: qshed v. block: integer and continuous
    5. Plot: pshed v. switch: integer and continuous
    6. Plot: qshed v. switch: integer and continuous
    7. Plot: v^2 v. switch: integer and continuous
Save these results in the location:
    results/date/control_exp/
"""

include("../../src/implementation/network_setup.jl")
include("../../src/implementation/export_results.jl")
include("../../src/implementation/visualization.jl")
ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

case = "motivation_c"
gen_cap = 0.8
# Inputs: case file path, percentage of load shed, list of critical load IDs
eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/$case.dss", gen_cap, [])
# Create the folder to store the results
    today = Dates.today()

    # Create a results folder
    if !isdir("results")
        mkdir("results")
    end

    # Create a folder per date
    using Dates
    today = Dates.today()
    date_folder = "results/$(today)"
    if !isdir(date_folder)
        mkdir(date_folder)
    end

    # Create a folder for the control experiment results
    case_name = "results/$(today)/$case"
    if !isdir(case_name)
        mkdir(case_name)
    end

    # Create a folder for the control experiment results
    control_exp_folder = "results/$(today)/$case/control_exp"
    if !isdir(control_exp_folder)
        mkdir(control_exp_folder)
    end

# Create the structure of the output tables
block_df = DataFrame(BlockID=Int64[], GenCap=Float64[], IntStatus=Int64[], RelaxStatus=Float64[], BlockSwitches=Set{Int64}[])
switch_df = DataFrame(SwitchID=Int64[], GenCap=Float64[], IntStatus=Int64[], RelaxStatus=Float64[], SwitchName=String[])
switch_voltage_df = DataFrame(SwitchID=Int64[], GenCap=Float64[], IntTBus=Vector{Float64}[], IntFBus=Vector{Float64}[], RelaxTBus=Vector{Float64}[], RelaxFBus=Vector{Float64}[])
# Create the structure of the Plots
pshed_block = plot()
qshed_block = plot()
pshed_switch = plot()
qshed_switch = plot()
v_squared_switch = plot()

# Prepare the bounds to plot for the squared voltage of each switch
n_sw = length(math["switch"])
v_squared_max = (1.05^2)*ones(n_sw)
v_squared_min = (0.95^2)*ones(n_sw)

# Get the reference data for the blocks
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_relaxed; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
ref = mld_model.ref[:it][:pmd][:nw][0]

# Run the integer case of the mld
mld_int = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)

block_int = mld_int["solution"]["block"]
switch_int = mld_int["solution"]["switch"]
for (id, block) in (block_int)
    push!(block_df.BlockID, parse(Int,id))
    push!(block_df.IntStatus, block["status"])
    push!(block_df.GenCap, gen_cap)
    push!(block_df.BlockSwitches, ref[:block_switches][parse(Int,id)])
end

for (id, switch) in switch_int
    push!(switch_df.SwitchID, parse(Int,id))
    push!(switch_voltage_df.SwitchID, parse(Int,id))
    push!(switch_df.GenCap, gen_cap)
    push!(switch_voltage_df.GenCap, gen_cap)
    push!(switch_df.IntStatus, Int(round(switch["state"])))
    push!(switch_df.SwitchName, math["switch"][id]["name"])
end


# Run the continuous case of the mld
mld_relaxed = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
block_relax = mld_relaxed["solution"]["block"]
switch_relax = mld_relaxed["solution"]["switch"]
for (id, block) in (block_relax)
    push!(block_df.RelaxStatus, block["status"])
end

for (id, switch) in switch_relax
    push!(switch_df.RelaxStatus, switch["state"])
end

# Debug: Check if switch ampacity constraint is binding
println("\n=== Switch Ampacity Debug ===")
for (sid, sw) in math["switch"]
    sw_soln = mld_relaxed["solution"]["switch"][sid]
    fbus = string(sw["f_bus"])
    w_fbus = mld_relaxed["solution"]["bus"][fbus]["w"]

    # Get switch power flow (psw, qsw)
    psw = sw_soln["pf"]
    qsw = sw_soln["qf"]

    # Current rating
    i_rating = sw["current_rating"]

    println("\nSwitch $(sid) ($(sw["name"])):")
    println("  State: $(sw_soln["state"])")
    println("  Current Rating: $(i_rating) A")
    println("  Voltage squared (w) at f_bus: $(w_fbus)")

    for (idx, phase) in enumerate(sw["f_connections"])
        p = psw[idx]
        q = qsw[idx]
        w = w_fbus[idx]
        s_squared = p^2 + q^2
        limit = sw_soln["state"] * w * i_rating[idx]^2
        utilization = s_squared / limit * 100

        println("  Phase $(phase): P=$(round(p, digits=2)) kW, Q=$(round(q, digits=2)) kVar")
        println("           S²=$(round(s_squared, digits=2)), Limit=$(round(limit, digits=2)), Utilization=$(round(utilization, digits=1))%")
    end
end
println("=== End Debug ===")

# Plot the squared voltage at each switch for both the integer and relaxed cases
switch_ids = sort(parse.(Int, collect(keys(mld_relaxed["solution"]["switch"]))))

fbus_w = Vector{Vector{Float64}}()
tbus_w = Vector{Vector{Float64}}()

for sid in switch_ids
    sw = math["switch"][string(sid)]

    fbus = string(sw["f_bus"])
    tbus = string(sw["t_bus"])

    push!(fbus_w, mld_relaxed["solution"]["bus"][fbus]["w"])
    push!(tbus_w, mld_relaxed["solution"]["bus"][tbus]["w"])
end



# Plot the active power load shed vs the block id
# Calculate the total active power in each block
lbs = ref[:block_loads]
block_p_int = []
block_q_int = []
block_p_relaxed = []
block_q_relaxed = []
for (id, block) in sort(lbs)
    block_load_p_int = 0
    block_load_q_int = 0
    block_load_p_relaxed = 0
    block_load_q_relaxed = 0
    for load in block
        block_load_p_int += mld_int["solution"]["load"][string(load)]["pshed"]
        block_load_q_int += mld_int["solution"]["load"][string(load)]["qshed"]
        block_load_p_relaxed += mld_relaxed["solution"]["load"][string(load)]["pshed"]
        block_load_q_relaxed += mld_relaxed["solution"]["load"][string(load)]["qshed"]
    end
    push!(block_p_int, block_load_p_int)
    push!(block_q_int, block_load_q_int)
    push!(block_p_relaxed, block_load_p_relaxed)
    push!(block_q_relaxed, block_load_q_relaxed)
end
scatter!(pshed_block, collect(keys(block_int)), block_p_int, label="Integer Solution")
scatter!(pshed_block, collect(keys(block_int)), block_p_relaxed, label="Relaxed Solution")
xlabel!(pshed_block, "Load Block")
ylabel!(pshed_block, "Load Shed (kW)")
title!(pshed_block, "Active Power Load Shed per Block")

scatter!(qshed_block, collect(keys(block_int)), block_q_int, label="Integer Solution")
scatter!(qshed_block, collect(keys(block_int)), block_q_relaxed, label="Relaxed Solution")
xlabel!(qshed_block, "Load Block")
ylabel!(qshed_block, "Load Shed (kW)")
title!(qshed_block, "Reactive Power Load Shed per Block")



phases = 1:3
phase_labels = ["Phase A", "Phase B", "Phase C"]
markers = [:circle, :diamond, :utriangle]

plt = scatter(
    xlabel = "Switch ID",
    ylabel = "Voltage² (p.u.)",
    title = "Squared Voltage at Switch Buses"
)

for (p, lab, m) in zip(phases, phase_labels, markers)
    scatter!(plt, switch_ids, [w[p] for w in fbus_w],
        label = "FBus $lab",
        marker = m
    )
    scatter!(plt, switch_ids, [w[p] for w in tbus_w],
        label = "TBus $lab",
        marker = m,
        linestyle = :dash
    )
end

# Voltage limits
plot!(plt, switch_ids, fill(0.9^2, length(switch_ids)),
      label = "Min Limit", line = (:dot, :black))
plot!(plt, switch_ids, fill(1.1^2, length(switch_ids)),
      label = "Max Limit", line = (:dot, :black))

# Save all the plots and the tables into the results folder
savefig(pshed_block, joinpath(control_exp_folder,"pshed_block"))
savefig(qshed_block, joinpath(control_exp_folder,"qshed_block"))
savefig(plt, joinpath(control_exp_folder, "v_squared_switch_3phase.svg"))
CSV.write(joinpath(control_exp_folder,"block_summary"), block_df)
CSV.write(joinpath(control_exp_folder,"switch_summary"), switch_df)

# Calculate and plot fairness indices for the final solution
# served = Float64[]
# plot_fairness_indices(mld["solution"], collect(mld["solution"]["load"][string(i)]["pshed"] for i in 1:length(mld["solution"]["load"])), collect(parse.(Int,collect(keys(mld["solution"]["load"])))), iterations, fair_exp, fair_func)


plot_network_load_shed(mld_relaxed["solution"], math;
    output_file=joinpath(control_exp_folder, "network_load_shed_relaxed.svg"),
    layout=:ieee13)

plot_network_load_shed(mld_int["solution"], math;
    output_file=joinpath(control_exp_folder, "network_load_shed_integer.svg"),
    layout=:ieee13)

# --- Single-level test: force switches 671692 and 646611 open ---
println("\n" * "="^60)
println("BLOCK DE-ENERGIZATION TEST: forcing 671692 and 646611 open")
println("="^60)
math_test = deepcopy(math)
for (i, sw) in math_test["switch"]
    if sw["name"] in ("671692", "646611")
        sw["state"] = 0
        println("  Forced switch $(sw["name"]) (id=$i) to state=0")
    end
end
mld_test = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(math_test, gurobi)
println("Termination: $(mld_test["termination_status"])")
plot_network_load_shed(mld_test["solution"], math_test;
    output_file=joinpath(control_exp_folder, "network_block_deenergize_test.svg"),
    layout=:ieee13)

