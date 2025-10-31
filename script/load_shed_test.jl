using Revise
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS, Juniper
using Plots
using Random
using Distributions
using DiffOpt
using JuMP
using LinearAlgebra,SparseArrays
# using DataFrames
ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# To make a bilevel JuMP model, we need to create a BilevelJuMP model here 
juniper = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt, "mip_solver"=>highs)


## Main loop
dir = dirname(@__FILE__)

#case = "ieee_13_pmd_mod.dss"
#case = "three_bus_constrained_line_capacity.dss"
#case = "three_bus_constrained_generation.dss"
case = "load_shed_test_single_phase.dss"
casepath = "data/$case"
file = joinpath(dir, "..", casepath)

data = case 
vscale = 1
loadscale = 1   

eng = PowerModelsDistribution.parse_file(file)#, transformations=[PowerModelsDistribution.transform_loops!,PowerModelsDistribution.remove_all_bounds!])

eng["settings"]["sbase_default"] = 1
eng["voltage_source"]["source"]["rs"] *=0
eng["voltage_source"]["source"]["xs"] *=0
eng["voltage_source"]["source"]["vm"] *=vscale


"Ensure use the reduce lines function in Fred's basecase script"
PowerModelsDistribution.reduce_line_series!(eng)

math = PowerModelsDistribution.transform_data_model(eng)
lbs = PowerModelsDistribution.identify_load_blocks(math)
get(eng, "time_series", Dict())

for (i,bus) in math["bus"]

		bus["vmax"][:] .= 1.1
		bus["vmin"][:] .= 0.9
end

# Ensure the generation from the source bus is less than the max load
# First calculate the total load
# ls_percent = .39 # ensure not inf
# for (i,gen) in math["gen"]
# 	if gen["source_id"] == "voltage_source.source"
# 		gen["pmax"] .= ls_percent*sum(load["pd"][idx] for (i,load) in math["load"] for (idx,c) in enumerate(load["connections"]))
# 		gen["qmax"] .= ls_percent*sum(load["qd"][idx] for (j,load) in math["load"] for (idx,c) in enumerate(load["connections"]))
# 		gen["pmin"] .= -ls_percent*sum(load["pd"][idx] for (i,load) in math["load"] for (idx,c) in enumerate(load["connections"]))
# 		gen["qmin"] .= -ls_percent*sum(load["qd"][idx] for (j,load) in math["load"] for (idx,c) in enumerate(load["connections"]))

# 	end
# end

# Create the critical load set
#critical_load = ["645", "652", "675a", "675b", "675c"]
# critical_load = ["l1"]
# for (i,load) in math["load"]
# 	if load["name"] in critical_load
# 		load["critical"] = 1
# 		load["weight"] = 20
# 		println("Load $(load["name"]) at math load node $(i) is critical.")
# 	else
# 		load["critical"] = 0
# 		load["weight"] = 10
# 		println("Load $(load["name"]) at math load node $(i) is not critical.")

# 	end
# end

for (id, load) in math["load"]
    if occursin("Critical", load["name"])
        load["critical"] = 1.0  # Highest
        load["weight"] = 10
    elseif occursin("Medium", load["name"])
        load["weight"] = 10
        load["critical"] = 0  # Medium
    else
        load["weight"] = 10 # Lowest - shed first
        load["critical"] = 0
    end
end

for (switch_id, switch) in enumerate(math["switch"])
    math["switch"][string(switch_id)]["state"] = 0
    math["switch"][string(switch_id)]["branch_id"] = 0
    for (branch_id, branch) in enumerate(math["branch"])
        # println("Branch $branch_id")
        # println("Branch dict $(branch[2]["source_id"])")
        # println("Branch dict $(math["switch"][string(switch_id)]["source_id"])")
        # println(" Switch dict $(switch[2]["source_id"])")
            if branch[2]["source_id"] == switch[2]["source_id"]
                switch[2]["branch_id"] = branch_id  # Assuming you have this mapping
            end
    end
end
math["switch"]["1"]["state"] = 0
math["switch"]["2"]["state"] = 1
math["switch"]["3"]["state"] = 0

math["block"] = Dict{String,Any}()
for (block, loads) in enumerate(lbs)
	math["block"][string(block)] = Dict("id"=>block, "state"=>0)
end
math["block"]["1"]["state"] = 1
math["block"]["2"]["state"] = 0
math["block"]["3"]["state"] = 1



# Ensure that all branches have some bounds. Currently, they are infinite
# this produces an error for the constraint_mc_switch_power_on_off, because
# the switch power variable is unbounded, when the reference bounds are infinite


#pm_acopf_soln = solve_mc_opf(math, ACPUPowerModel, ipopt)

#pm_mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
# JuMP.set_optimizer(pm_mld.model, Ipopt.Optimizer)
# JuMP.optimize!(pm_mld.model)
#reference = pm_mld.ref[:it][:pmd][:nw][0]    

#pm_pf = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_pf_switchable; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
#ref_pf = pm_pf.ref[:it][:pmd][:nw][0]    
#pm_pf_soln = FairLoadDelivery.solve_mc_pf_aw(math, ipopt)#; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
# pm_mld_soln = solve_mc_mld(math, LinDist3FlowPowerModel, ipopt)
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
set_optimizer(mld_model.model, ipopt)
optimize!(mld_model.model)
con_ref = mld_model.con[:it][:pmd][:nw][0]
var = mld_model.var[:it][:pmd][:nw][0]
soln_ref = mld_model.sol[:it][:pmd][:nw][0]
ref = mld_model.ref[:it][:pmd][:nw][0]

pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch(math, ipopt)
res = pm_mld_soln["solution"]
# println("Load served: $(sum(load["pd"] for load in math["load"] if load["critical"] == 1))")
load_ref = sum(load["pd"] for (i,load) in ref[:load])
println("Total load in reference solution: $load_ref")
gen_ref = sum(gen["pg"] for (i,gen) in ref[:gen])
println("Total generation in reference solution: $gen_ref")
load_served = sum((load["pd"]) for (i,load) in res["load"])
println("Total load served in MLD solution: $load_served")
println("Load served percentage: $(load_served/load_ref*100) %")
# After optimize!(pm.model)

function diagnose_infeasibility(mld_model, ref, var, con_ref)
    println("\n" * "="^60)
    println("COMPREHENSIVE INFEASIBILITY DIAGNOSTICS")
    println("="^60)
    
    # 1. Model status
    println("\n=== MODEL STATUS ===")
    println("Termination status: ", JuMP.termination_status(mld_model.model))
    println("Primal status: ", JuMP.primal_status(mld_model.model))
    
    # 2. Switch diagnostics
    println("\n=== SWITCH STATE DIAGNOSTICS ===")
    switch_violations = 0
    for (s, switch) in ref[:switch]
        var_ref = var[:switch_state][s]
        var_value = JuMP.value(var_ref)
        ref_value = switch["state"]
        error = abs(var_value - ref_value)
        
        if error > 0.001
            switch_violations += 1
            println("\n⚠️  Switch $s VIOLATION:")
            println("  Reference: $ref_value, Solved: $var_value, Error: $error")
        end
    end
    println("Total switch violations: $switch_violations")
    
    # 3. Block diagnostics
    println("\n=== BLOCK STATE DIAGNOSTICS ===")
    block_violations = 0
    for (b, block) in ref[:block]
        var_ref = var[:z_block][b]
        var_value = JuMP.value(var_ref)
        ref_value = block["state"]
        error = abs(var_value - ref_value)
        
        if error > 0.001
            block_violations += 1
            println("\n⚠️  Block $b VIOLATION:")
            println("  Reference: $ref_value, Solved: $var_value, Error: $error")
        end
    end
    println("Total block violations: $block_violations")
    
    # 4. Topology consistency
    println("\n=== TOPOLOGY CONSISTENCY CHECK ===")
    topology_violations = 0
    for (s, switch) in ref[:switch]
        f_bus = switch["f_bus"]
        t_bus = switch["t_bus"]
        block_fr = ref[:bus_block_map][f_bus]
        block_to = ref[:bus_block_map][t_bus]
        
        switch_state = switch["state"]
        block_fr_state = ref[:block][block_fr]["state"]
        block_to_state = ref[:block][block_to]["state"]
        
        # Check reference data consistency
        if switch_state == 1 && block_fr_state != block_to_state
            topology_violations += 1
            println("\n⚠️  TOPOLOGY CONFLICT in reference data:")
            println("  Switch $s is CLOSED (state=1)")
            println("  But connects block $block_fr (state=$block_fr_state) to block $block_to (state=$block_to_state)")
        end
    end
    println("Total topology violations in reference data: $topology_violations")
    
    # 5. Summary
    println("\n" * "="^60)
    println("SUMMARY")
    println("="^60)
    println("Switch violations: $switch_violations")
    println("Block violations: $block_violations")
    println("Topology conflicts: $topology_violations")
    
    if switch_violations + block_violations + topology_violations == 0
        println("\n✓ No violations found - model should be feasible")
    else
        println("\n✗ Violations detected - this explains infeasibility")
    end
end

# Run the diagnostic
#diagnose_infeasibility(mld_model, ref, var, con_ref)