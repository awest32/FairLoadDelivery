"""
Debug AC feasibility failures for the multiperiod bilevel.
Runs a single period through the full pipeline and prints diagnostics.
"""

using Revise
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi
using JuMP
import MathOptInterface as MOI

include("../../src/implementation/network_setup.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

const LS_PERCENT = 0.8
const SOURCE_PU = 1.03
const N_PERIODS = 6
const LOAD_SCALE_FACTORS = [round(0.8 + 1.0 * exp(-((t - N_PERIODS/2)^2) / (2 * 4^2)), digits=3) for t in 0:N_PERIODS-1]

# Setup network
eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/motivation_c.dss", LS_PERCENT, SOURCE_PU, [])

# Set extreme weights to mimic equality_min / palma outcome
# Loads that should be prioritized get high weight, others get low
sorted_load_ids = sort(parse.(Int, collect(keys(math["load"]))))
for lid in sorted_load_ids
    math["load"][string(lid)]["weight"] = 10.0  # high weight = serve this load
end
# De-prioritize most loads (mimic fairness pushing weights to extremes)
for lid in sorted_load_ids[4:end]
    math["load"][string(lid)]["weight"] = 1.0
end

println("Weights: ", [math["load"][string(lid)]["weight"] for lid in sorted_load_ids])

# Step 1: Solve relaxed MLD
println("\n=== Step 1: Relaxed MLD ===")
relaxed = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math, Ipopt.Optimizer;
    ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
println("Status: $(relaxed["termination_status"])")

sol = relaxed["solution"]
println("\nSwitch states:")
for s in sort(collect(keys(sol["switch"])))
    println("  Switch $s: state=$(round(sol["switch"][s]["state"], digits=3))")
end
println("\nBlock states:")
for b in sort(collect(keys(sol["block"])))
    println("  Block $b: status=$(round(sol["block"][b]["status"], digits=3))")
end
println("\nLoad shed:")
for l in sort(collect(keys(sol["load"])), by=x->parse(Int,x))
    pd = sum(sol["load"][l]["pd"])
    pshed = sum(sol["load"][l]["pshed"])
    println("  Load $l: served=$(round(pd, digits=1)), shed=$(round(pshed, digits=1))")
end

# Step 2: Extract and round
switch_states, block_status = FairLoadDelivery.extract_switch_block_states(sol)

pm = PowerModelsDistribution.instantiate_mc_model(
    math, PowerModelsDistribution.LinDist3FlowPowerModel,
    FairLoadDelivery.build_mc_mld_shedding_implicit_diff;
    ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
ref = pm.ref[:it][:pmd][:nw][0]

samples = FairLoadDelivery.generate_bernoulli_samples(switch_states, 2000, 100)
index, sw_radial, block_ids, blk_radial, load_ids, load_radial, top_candidates =
    FairLoadDelivery.radiality_check(ref, switch_states, block_status, samples; top_k=5)

println("\n=== Step 2: Top radial candidates ===")
for (k, cand) in enumerate(top_candidates[1:min(3, length(top_candidates))])
    println("  Candidate $k (sample $(cand.index), dist=$(round(cand.dist, digits=4))):")
    sw = samples[cand.index]
    for s in sort(collect(keys(sw)))
        println("    Switch $s: $(sw[s])")
    end
    println("    Block status: $(round.(cand.block_status, digits=2))")
end

# Step 3: Integer MLD on best candidate
println("\n=== Step 3: Integer MLD ===")
math_rounded = FairLoadDelivery.update_network(deepcopy(math), samples[top_candidates[1].index], ref)
mld = FairLoadDelivery.solve_mc_mld_shed_random_round(math_rounded, Ipopt.Optimizer)
println("MLD status: $(mld["termination_status"])")
mld_sol = mld["solution"]

if haskey(mld_sol, "switch")
    println("\nMLD Switch states:")
    for s in sort(collect(keys(mld_sol["switch"])))
        println("  Switch $s: state=$(round(get(mld_sol["switch"][s], "state", NaN), digits=3))")
    end
else
    println("\nMLD solution has no 'switch' key. Available keys: $(collect(keys(mld_sol)))")
end
if haskey(mld_sol, "block")
    println("\nMLD Block states:")
    for b in sort(collect(keys(mld_sol["block"])))
        println("  Block $b: status=$(round(get(mld_sol["block"][b], "status", NaN), digits=3))")
    end
else
    println("\nMLD solution has no 'block' key.")
end
println("\nMLD Load results:")
mld_total_shed = 0.0
mld_total_served = 0.0
for l in sort(collect(keys(mld_sol["load"])), by=x->parse(Int,x))
    pd = sum(mld_sol["load"][l]["pd"])
    pshed = sum(mld_sol["load"][l]["pshed"])
    global mld_total_shed += pshed
    global mld_total_served += pd
    println("  Load $l: served=$(round(pd, digits=1)), shed=$(round(pshed, digits=1))")
end
println("Total: served=$(round(mld_total_served, digits=1)), shed=$(round(mld_total_shed, digits=1))")

# Step 4: AC network update — inspect what goes into the PF
println("\n=== Step 4: AC Network Update ===")
math_ac = FairLoadDelivery.ac_network_update(math_rounded, ref; mld_solution=mld)

println("\nAC network buses:")
for (bid, bus) in sort(collect(math_ac["bus"]), by=x->parse(Int,x[1]))
    println("  Bus $bid: type=$(bus["bus_type"]), status=$(get(bus, "status", "N/A"))")
end
println("\nAC network switches:")
for (sid, sw) in sort(collect(math_ac["switch"]), by=x->parse(Int,x[1]))
    println("  Switch $sid: state=$(sw["state"]), status=$(sw["status"]), dispatchable=$(sw["dispatchable"])")
end
println("\nAC network branches:")
for (bid, br) in sort(collect(math_ac["branch"]), by=x->parse(Int,x[1]))
    println("  Branch $bid: status=$(get(br, "br_status", get(br, "status", "N/A"))), f_bus=$(br["f_bus"])→t_bus=$(br["t_bus"])")
end
println("\nAC network loads:")
for (lid, load) in sort(collect(math_ac["load"]), by=x->parse(Int,x[1]))
    println("  Load $lid: status=$(get(load, "status", "N/A")), pd=$(round.(load["pd"], digits=1))")
end
println("\nAC network generators:")
for (gid, gen) in sort(collect(math_ac["gen"]), by=x->parse(Int,x[1]))
    println("  Gen $gid: status=$(gen["gen_status"]), source=$(gen["source_id"]), pmax=$(gen["pmax"])")
end

# Step 5: Run AC PF with verbose Ipopt
println("\n=== Step 5: AC Power Flow ===")
ipopt_verbose = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 5)
pf_result = PowerModelsDistribution.solve_mc_pf(math_ac, IVRUPowerModel, ipopt_verbose)
println("PF status: $(pf_result["termination_status"])")
