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

#--------------------------------------------------------------------------------------------------------
# This script brute-force checks the feasibility of all 2^n switch configruations
# for the 13-bus motivation case. This is used to verify that the integer formulations are correctly.
#--------------------------------------------------------------------------------------------------------

case = "ieee_13_aw_edit/motivation_c.dss"
dir = @__DIR__
casepath = joinpath(dir, "../data/", case)
const SWITCH_RATING = 700.0
const GEN_CAP = 5000.0
const SOURCE_PU = 1.03

eng, math = setup_network(casepath, GEN_CAP, SOURCE_PU, SWITCH_RATING,[])

switch_2_eng_map = Dict{String, String}()
for (switch_id, switch_data) in math["switch"]
    switch_2_eng_map[switch_id] = switch_data["name"]
end
# Make every switch dispatchable so the assigned state takes effect
for (_, switch) in math["switch"]
    switch["dispatchable"] = 1.0
end

sorted_switch_ids = sort(collect(keys(math["switch"])); by = x -> parse(Int, x))
n_switches = length(sorted_switch_ids)

ac_results = DataFrame(
    config_bits = String[],
    switch_states = String[],
    termination_status = Symbol[],
    objective = Float64[],
    feasible = Bool[],
)

mld_results = DataFrame(
    config_bits = String[],
    switch_states = String[],
    termination_status = Symbol[],
    objective = Float64[],
    feasible = Bool[],
)

radiality_results = DataFrame(
    config_bits = String[],
    switch_states = String[],
    termination_status = Symbol[],
    objective = Float64[],
    feasible = Bool[],
)

# Get the reference data for the blocks
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_integer; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
ref_round = mld_model.ref[:it][:pmd][:nw][0]

# Radiality check function
function radiality_check(ref_round, math, optimizer)
    model = JuMP.Model()
    set_optimizer(model, optimizer)
    @variable(model, z_switch[1:length(ref_round[:switch])].>= 0)
    @variable(model, z_block[1:length(ref_round[:block])] .>= 0)
    @variable(model, z_demand[1:length(ref_round[:load])].>= 0)
    @variable(model, z_gen[1:length(ref_round[:gen])].>= 0)
    @variable(model, z_storage[1:length(ref_round[:storage])].>= 0)
    @variable(model, z_voltage[1:length(ref_round[:bus])].>= 0)
    @variable(model, z_shunt[1:length(ref_round[:shunt])].>= 0)

    for (s, s_data)in math["switch"]
        @constraint(model, z_switch[parse(Int, s)] == s_data["state"])
    end
    @constraint(model, z_block[1:length(ref_round[:block])] .<= 1)
    @constraint(model, z_demand[1:length(ref_round[:load])] .<= 1)
    @constraint(model, z_gen[1:length(ref_round[:gen])] .<= 1)
    @constraint(model, z_storage[1:length(ref_round[:storage])] .<= 1)
    @constraint(model, z_voltage[1:length(ref_round[:bus])] .<= 1)
    @constraint(model, z_shunt[1:length(ref_round[:shunt])] .<= 1)

    # radiality
    FairLoadDelivery.constraint_radial_topology_jump(model, ref_round, z_switch)
    FairLoadDelivery.constraint_mc_isolate_block_jump(model, ref_round)
    for b in ref_round[:substation_blocks]
        @constraint(model, z_block[b] == 1)
    end

    FairLoadDelivery.constraint_block_budget_jump(model, ref_round)
    FairLoadDelivery.constraint_switch_budget_jump(model, ref_round)

    FairLoadDelivery.constraint_connect_block_load_jump(model, ref_round)
    FairLoadDelivery.constraint_connect_block_gen_jump(model, ref_round)
    FairLoadDelivery.constraint_connect_block_voltage_jump(model, ref_round)
    FairLoadDelivery.constraint_connect_block_shunt_jump(model, ref_round)
    FairLoadDelivery.constraint_connect_block_storage_jump(model, ref_round)

    optimize!(model);
    return termination_status(model)
end
empty!(radiality_results);
for config in 0:(2^n_switches - 1)
    states = Dict(sid => Float64((config >> (i - 1)) & 1)
                  for (i, sid) in enumerate(sorted_switch_ids))

    trial = deepcopy(math)
    for (sid, s) in states
        trial["switch"][sid]["state"]  = s
        trial["switch"][sid]["status"] = s
    end
    
    mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_integer; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
    ref_round = mld_model.ref[:it][:pmd][:nw][0]
    tstat = radiality_check(ref_round, trial, Ipopt.Optimizer)
    feasible = tstat in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)

    bits = join((Int(states[sid]) for sid in sorted_switch_ids), "")
    labeled = join(("$(sid)=$(Int(states[sid]))" for sid in sorted_switch_ids), ",")

    push!(radiality_results, (bits, labeled, Symbol(tstat), NaN, feasible))
                        #push!(radiality_results, (bits, labeled, Symbol(tstat),
                    #get(radiality_result, "objective", NaN), feasible))
    @info "config $bits → $tstat" feasible
end

radiality_feasible_networks = filter(row -> row.feasible, radiality_results)

empty!(mld_results); 
# For each radial config, fix switch states and test MLD feasibility.
for row in eachrow(radiality_feasible_networks)
    states = Dict(sid => Float64(parse(Int, row.config_bits[i]))
                  for (i, sid) in enumerate(sorted_switch_ids))

    trial = deepcopy(math)
    for (sid, s) in states
        trial["switch"][sid]["state"]        = s
        trial["switch"][sid]["status"]       = s
        trial["switch"][sid]["dispatchable"] = 1.0  # fix switch at this state
    end

    mld_res = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(trial, Gurobi.Optimizer)
    tstat = mld_res["termination_status"]
    feasible = tstat in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)

    push!(mld_results, (row.config_bits, row.switch_states, Symbol(tstat),
                        get(mld_res, "objective", NaN), feasible))
    @info "mld config $(row.config_bits) → $tstat" feasible
end
 
mld_feasible_networks = filter(row -> row.feasible, mld_results)


empty!(ac_results)
# For each MLD-feasible config, apply the MLD shed decisions and run AC OPF
# to check whether the post-shed network is AC-power-flow feasible.
for row in eachrow(mld_feasible_networks)
    states = Dict(sid => Float64(parse(Int, row.config_bits[i]))
                  for (i, sid) in enumerate(sorted_switch_ids))

    trial = deepcopy(math)
    for (sid, s) in states
        trial["switch"][sid]["state"]        = s
        trial["switch"][sid]["status"]       = s
        trial["switch"][sid]["dispatchable"] = 1.0
    end

    # Re-solve MLD to retrieve the per-load shed decisions for this config
    mld_res = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(trial, Gurobi.Optimizer)
    load_sol = mld_res["solution"]["load"]

    # Apply MLD load-shed decisions: reduce pd/qd by the solution's pshed/qshed
    for (lid, load) in trial["load"]
        if haskey(load_sol, lid)
            pshed = get(load_sol[lid], "pshed", zeros(length(load["pd"])))
            qshed = get(load_sol[lid], "qshed", zeros(length(load["qd"])))
            load["pd"] = max.(load["pd"] .- pshed, 0.0)
            load["qd"] = max.(load["qd"] .- qshed, 0.0)
        end
    end

    # Open the branch associated with any switch that is open (state == 0)
    # so the AC model sees an actually disconnected topology.
    for (sid, s) in states
        if s == 0.0
            bid = trial["switch"][sid]["branch_id"]
            if bid != 0
                trial["branch"][string(bid)]["br_status"] = 0
            end
        end
    end

    ac_res = PowerModelsDistribution.solve_mc_opf(trial, IVRUPowerModel, Ipopt.Optimizer)
    tstat = ac_res["termination_status"]
    feasible = tstat in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)

    push!(ac_results, (row.config_bits, row.switch_states, Symbol(tstat),
                       get(ac_res, "objective", NaN), feasible))
    @info "ac config $(row.config_bits) → $tstat" feasible
end

ac_feasible_networks = filter(row -> row.feasible, ac_results)

