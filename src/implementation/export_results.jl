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

"""
Save the load shed per block and the voltage of the switches to CSV files for further analysis.

Select the desired network.
Run the  FLDP with switches and load blocks for the integer and the relaxed MLD.
Extract the results from these cases.
    1. Table: blocks, block states
    2. Table: switches, switch states
    3. Plot: pshed v. block
    4. Plot: qshed v. block
    5. Plot: pshed v. switch
    6. Plot: qshed v. switch
    7. Plot: v^2 v. switch
Save these results in the location:
    results/date/fair_exp/
"""
function create_save_folder(case::String, fair_func::String)
# Create the folder to store the results
    today = Dates.today()

    # Create a results folder
    if !isdir("results")
        mkdir("results")
    end

    # Create a folder per date
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
    exp_folder = "results/$(today)/$case/$fair_func"
    if !isdir(exp_folder)
        mkdir(exp_folder)
    end
end
function export_results(math::Dict, exp_folder::String, gen_cap::Float64, fair_func::String)
    # Create the structure of the output tables
    block_df = DataFrame(BlockID=Int64[], GenCap=Float64[], Status=Float64[], BlockSwitches=Set{Int64}[])
    switch_df = DataFrame(SwitchID=Int64[], GenCap=Float64[], Status=Float64[], SwitchName=String[])
    switch_voltage_df = DataFrame(SwitchID=Int64[], GenCap=Float64[], RelaxTBus=Vector{Float64}[], RelaxFBus=Vector{Float64}[])
    # Create the structure of the Plots
    pshed_block = plot()
    qshed_block = plot()
    pshed_switch = plot()
    qshed_switch = plot()
    v_squared_switch = plot()

    # Prepare the bounds to plot for the squared voltage of each switch
    n_sw = length(math["switch"])
    v_squared_max = (1.1^2)*ones(n_sw)
    v_squared_min = (0.9^2)*ones(n_sw)

    # Get the reference data for the blocks
    mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_shedding_random_rounding; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
    ref = mld_model.ref[:it][:pmd][:nw][0]

    # Run the integer case of the mld
    mld = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)

    block = mld["solution"]["block"]
    switch = mld["solution"]["switch"]
    for (id, block) in (block)
        push!(block_df.BlockID, parse(Int,id))
        push!(block_df.Status, block["status"])
        push!(block_df.GenCap, gen_cap)
        push!(block_df.BlockSwitches, ref[:block_switches][parse(Int,id)])
    end

    for (id, switch) in switch
        push!(switch_df.SwitchID, parse(Int,id))
        push!(switch_voltage_df.SwitchID, parse(Int,id))
        push!(switch_df.GenCap, gen_cap)
        push!(switch_voltage_df.GenCap, gen_cap)
        push!(switch_df.Status, switch["state"])
        push!(switch_df.SwitchName, math["switch"][id]["name"])
    end


    # Plot the squared voltage at each switch for both the integer and relaxed cases
    switch_ids = sort(parse.(Int, collect(keys(mld["solution"]["switch"]))))

    fbus_w = Vector{Vector{Float64}}()
    tbus_w = Vector{Vector{Float64}}()

    for sid in switch_ids
        sw = math["switch"][string(sid)]

        fbus = string(sw["f_bus"])
        tbus = string(sw["t_bus"])

        push!(fbus_w, mld["solution"]["bus"][fbus]["w"])
        push!(tbus_w, mld["solution"]["bus"][tbus]["w"])
    end



    # Plot the active power load shed vs the block id
    # Calculate the total active power in each block
    lbs = ref[:block_loads]
    block_p = []
    block_q = []
    for (id, block) in sort(lbs)
        block_load_p = 0
        block_load_q = 0
        for load in block
            block_load_p += mld["solution"]["load"][string(load)]["pshed"]
            block_load_q += mld["solution"]["load"][string(load)]["qshed"]
        end
        push!(block_p, block_load_p)
        push!(block_q, block_load_q)
    end
    scatter!(pshed_block, collect(keys(block)), block_p)
    xlabel!(pshed_block, "Load Block")
    ylabel!(pshed_block, "Load Shed (kW)")
    title!(pshed_block, "Active Power Load Shed per Block")

    scatter!(qshed_block, collect(keys(block)), block_q)
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
    savefig(pshed_block, joinpath(exp_folder,"pshed_block"))
    savefig(qshed_block, joinpath(exp_folder,"qshed_block"))
    savefig(plt, joinpath(exp_folder, "v_squared_switch_3phase.svg"))
    CSV.write(joinpath(exp_folder,"block_summary"), block_df)
    CSV.write(joinpath(exp_folder,"switch_summary"), switch_df)

    # Calculate and plot fairness indices for the final solution
    # served = Float64[]
    # plot_fairness_indices(mld["solution"], collect(mld["solution"]["load"][string(i)]["pshed"] for i in 1:length(mld["solution"]["load"])), collect(parse.(Int,collect(keys(mld["solution"]["load"])))), iterations, fair_exp, fair_func)
    return nothing


end
# Plot the weights per load 
function plot_weights_per_load(weights_new, weight_ids, k, save_path)
    weights_plot = bar(weight_ids, weights_new, title = "Fair Load Weights per Load - Iteration $k", xlabel = "Load ID", ylabel = "Fair Load Weight", legend = false)
    savefig(weights_plot, "$save_path/fair_load_weights_per_load_k$(k).svg")
    println("Weights plot saved as $save_path/fair_load_weights_per_load_k$(k).svg")
end
