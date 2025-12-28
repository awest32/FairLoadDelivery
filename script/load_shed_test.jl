using Revise
using FairLoadDelivery
using MKL
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

# using DataFrames
ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# To make a bilevel JuMP model, we need to create a BilevelJuMP model here 
juniper = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt, "mip_solver"=>highs)

global solver = gurobi

## Main loop
dir = dirname(@__FILE__)

#case = "ieee_13_clean/ieee13.dss"
#case = "ieee_13_aw_edit/case_file_1trans_kron_reduced_3ph3wr_all_switches.dss"
case = "ieee_13_aw_edit/motivation_b.dss"
#case = "13_bus_load_shed_test.dss"
#case = "ieee_aw.dss"
#case = "load_shed_test_single_phase.dss"
#case = "control_case.dss"
#case = "load_shed_test_three_phase.dss"
#case = "Master.dss"
#casepath = "data/network_10/Feeder_1/$case"
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
#PowerModelsDistribution.reduce_line_series!(eng)


math = PowerModelsDistribution.transform_data_model(eng)


for (idx, switch) in math["switch"]
    switch["state"] = 1
end

# lbs = Dict{Int64, Any}()
# bbs = Dict{Int64, Any}()

# lbs_eng = Dict{Int64, Any}()
# lb1 = [645, 646, 611, 652, 671]
# lb2 = [670]#really 632
# lb3 = [634, 675, 692]
# lbs_eng[1] = lb1
# lbs_eng[2] = lb2
# lbs_eng[3] = lb3

# for (lb_id, lb) in lbs_eng
#     lb_vect = []
#     bb_vect = []
#     for bus in lb
#         for cons in 1:length(ref[:bus_loads][math["bus_lookup"][string(bus)]])
#             push!(lb_vect, ref[:bus_loads][math["bus_lookup"][string(bus)]][cons])
#         end
#         push!(bb_vect, math["bus_lookup"][string(bus)])
#     end
#     lbs[lb_id] = lb_vect
#     bbs[lb_id] = bb_vect
# end
lbs = PowerModelsDistribution.identify_load_blocks(math)
get(eng, "time_series", Dict())

for (i,bus) in math["bus"]

		bus["vmax"][:] .= 1.1
		bus["vmin"][:] .= 0.9
end

# Save for the relaxed version when using nonlinear terms in objective
#add_start_vrvi!(math)


# p = powerplot(eng, bus    = (:data=>"bus_type", :data_type=>"nominal"),
#                     branch = (:data=>"index", :data_type=>"ordinal"),
#                     gen    = (:data=>"pmax", :data_type=>"quantitative"),
#                     load   = (:data=>"pd",  :data_type=>"quantitative"),
#                     width = 300, height=300
# )
# save("powerplot.pdf", p)

# Ensure the generation from the source bus is less than the max load
# First calculate the total load
served = [] #Dict{Any,Any}()
ls_percent = 0.9
for (i,gen) in math["gen"]
    if gen["source_id"] == "voltage_source.source"
        pd_phase1=0
        pd_phase2=0
        pd_phase3=0
        qd_phase1=0
        qd_phase2=0
        qd_phase3=0
        for (ind, d) in math["load"]
            # @info d
            # @info d["connections"]
            for (idx, con) in enumerate(d["connections"])
                # @info "Load at connection $(d["connections"][idx]) has pd=$(d["pd"][idx]) and qd=$(d["qd"][idx])"
                if 1 == con# d["connections"] 
                    pd_phase1 += d["pd"][idx]
                    qd_phase1 += d["qd"][idx]
                end
                if 2 == con
                    pd_phase2 += d["pd"][idx]
                    qd_phase2 += d["qd"][idx]
                end 
                if 3 == con
                    pd_phase3 += d["pd"][idx]
                    qd_phase3 += d["qd"][idx]
                end
            end
        end
        gen["pmax"][1] = pd_phase1 * ls_percent
        gen["qmax"][1] = qd_phase1 * ls_percent
        gen["pmax"][2] = pd_phase2 * ls_percent
        gen["qmax"][2] = qd_phase2 * ls_percent
        gen["pmax"][3] = pd_phase3 * ls_percent
        gen["qmax"][3] = qd_phase3 * ls_percent
        gen["pmin"][:] .= 0
        gen["qmin"][:] .= 0
    end
end

# Create the critical load set
critical_load = ["675a"]
#critical_load = ["l4"]
for (i,load) in math["load"]
    if load["name"] in critical_load
        load["critical"] = 0
        load["weight"] = 10
        println("Load $(load["name"]) at math load node $(i) is critical.")
    else
        load["critical"] = 0
        load["weight"] = 10
        println("Load $(load["name"]) at math load node $(i) is not critical.")

    end
end

for (switch_id, switch) in enumerate(math["switch"])
    math["switch"][string(switch_id)]["branch_id"] = 0
    for (branch_id, branch) in enumerate(math["branch"])
            if branch[2]["source_id"] == switch[2]["source_id"]
                switch[2]["branch_id"] = branch_id  # Assuming you have this mapping
            end
    end
end
math["switch"]["1"]["state"] = 0 # Open the switch to force load shedding
math["switch"]["2"]["state"] = 1 # Open the switch to force load shedding
math["switch"]["3"]["state"] = 1 # Open the switch to force load shedding
math["block"] = Dict{String,Any}()
for (block, loads) in enumerate(lbs)
    math["block"][string(block)] = Dict("id"=>block, "state"=>0)
end
# math["block"]["1"]["state"] = 0 # Open the switch to force load shedding
# math["block"]["2"]["state"] = 1 # Open the switch to force load shedding
# math["block"]["3"]["state"] = 1 
#pm_ivr_soln = solve_mc_pf(math, IVRUPowerModel, ipopt)
# pm_ivr_opf_soln = solve_mc_opf(math, IVRUPowerModel, ipopt)
#pf_ivrup_aw = solve_mc_pf_aw(math, ipopt)
#ivrup_aw_mod = instantiate_mc_model(math, IVRUPowerModel, build_mc_pf_switch; ref_extensions=[ref_add_load_blocks!])

# pm_acrup_mld_feas = FairLoadDelivery.solve_mc_opf_acp(math, ipopt)
# pm_acopf_soln = solve_mc_opf(math, ACRUPowerModel, ipopt)
# pm_acopf_soln = solve_mc_pf(math, ACRUPowerModel, ipopt)

##################################
# Build and solve the MLD model
##################################
mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_integer; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
pf_model = instantiate_mc_model(math, IVRUPowerModel, build_mc_pf_switch; ref_extensions=[ref_add_load_blocks!])
mld = mld_model.model
set_optimizer(mld, solver)
if solver == ipopt
    set_attribute(mld, "hsllib", HSL_jll.libhsl_path)
    set_attribute(mld, "linear_solver", "ma27")
end
optimize!(mld)
con_ref = mld_model.con[:it][:pmd][:nw][0]
var = mld_model.var[:it][:pmd][:nw][0]
soln_ref = mld_model.sol[:it][:pmd][:nw][0]
ref = mld_model.ref[:it][:pmd][:nw][0]
#ref = pf_model.ref[:it][:pmd][:nw][0]
tests = ["ContSwitch_ModelA", "IntSwitch_ModelA", "ImpDiff_ModelA", "LLSoln_ModelA", "ContSwitch_ModelB", "IntSwitch_ModelB", "ImpDiff_ModelB", "LLSoln_ModelB"]
df_load_results = DataFrame(TestName=String[], GenSoln=Float64[], LoadReference=Float64[], LoadServedSetting=Float64[],LoadServed=Float64[], LoadServedPercentage=Float64[]) #, GiniIndex=Float64[], JainsIndex=Float64[], PalmaRatio=Float64[], AlphaFairness1=Float64[])
df_fairness_results = DataFrame(TestName=String[], GiniIndex=Float64[], JainsIndex=Float64[], PalmaRatio=Float64[], AlphaFairness1=Float64[])
df_decision_results = DataFrame(TestName=String[], LoadBlocks=Vector{Dict{Any,Any}}(), Switches=Vector{Dict{String, Any}}(), SwitchNames=Vector{Dict{String,Any}}(), BlockBusMap=Vector{Dict{Int64, Set}}(), BusMap=Vector{Dict{String, Any}}())

#Gini index
function gini_index(x)
    x = sort(x)
    n = length(x)
    #mean_x = _PMD.mean(x)
    #sum_diff = sum(abs(x[i] - x[j]) for i in 1:n for j in 1:n)
    gini_top = 1 - 1/n + 2*sum(sum(x[j] for j in 1:i) for i in 1:n-1)/(n*sum(x))
    gini_bottom = 2*(1-1/n)
    return gini_top/gini_bottom
end

#Jain's index
function jains_index(x)
    n = length(x)
    sum_x = sum(x)
    sum_x2 = sum(xi^2 for xi in x)
    return (sum_x^2) / (n * sum_x2)
end

#Palma Ratio
function palma_ratio(x)
    sorted_x = sort(x)
    n = length(x)
    top_10_percent = sum(sorted_x[ceil(Int, 0.9n):end])
    bottom_40_percent = sum(sorted_x[1:floor(Int, 0.4n)])
    return top_10_percent / bottom_40_percent
end
#Alpha fairness for alpha=1
function alpha_fairness(x, alpha=1)
    if alpha == 1
        return sum(log(xi) for xi in x)
    else
        return sum((xi^(1 - alpha)) / (1 - alpha) for xi in x)
    end
end

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


for test_name in tests
    if test_name == "ContSwitch_ModelA" || test_name == "IntSwitch_ModelA" || test_name == "ImpDiff_ModelA" || test_name == "LLSoln_ModelA"
        println("\nRunning test: $test_name")
        # Setup the network for Model A 
        casepath = "ieee_13_aw_edit/motivation_a.dss"
        eng, math, lbs, critical_id = setup_network(casepath, ls_percent, critical_load)
        # Continuous switch variables, Model A
        if test_name == "ContSwitch_ModelA"
            @info "Testing continuous switch mld for model A."
            pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_relaxed; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
        elseif test_name == "IntSwitch_ModelA"
            @info "Testing the integer switch mld for model A."
            pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_integer; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
        elseif test_name == "ImpDiff_ModelA"
            @info "Testing the continuous implicit diff enabled mld for model A."
            pm_mld_soln = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math, ipopt)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_shedding_implicit_diff; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
        elseif test_name == "LLSoln_ModelA"
            @info "Testing the lower level solution script for model A."
            dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math, 10*ones(length(math["load"])), 1)
            pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_relaxed; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
        end
    elseif test_name == "ContSwitch_ModelB" || test_name == "IntSwitch_ModelB" || test_name == "ImpDiff_ModelB" || test_name == "LLSoln_ModelB"
        println("\nRunning test: $test_name")
        # Setup the network for Model B
        casepath = "ieee_13_aw_edit/motivation_b.dss"
        eng, math, lbs, critical_id = setup_network(casepath, ls_percent, critical_load)
        if test_name == "ContSwitch_ModelB"
            @info "Testing the continous switch mld for model B."
            pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_relaxed; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
        elseif test_name == "IntSwitch_ModelB"
            @info "Testing the integer switch model for model B."
            pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_integer; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
        elseif test_name == "ImpDiff_ModelB"
            @info "Testing the continuous switch implicit diff mld for model B."
            pm_mld_soln = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math, ipopt)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_shedding_implicit_diff; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
        elseif test_name == "LLSoln_ModelB"
            @info "Testing the lower level solution for model B."
            dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math, 10*ones(length(math["load"])), 1)
            pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_relaxed(math, ipopt)
            mld = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable_relaxed; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
            ref = mld.ref[:it][:pmd][:nw][0]
        end
    end
    #pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)
    #mld_round_soln = FairLoadDelivery.solve_mc_mld_shed_random_round(math, ipopt)
    ##############################################
    # Extract results
    ##############################################
    #res = pf_ivrup_aw["solution"]
    #res = mld_round_soln["solution"]
    res = pm_mld_soln["solution"]
    # println("Load served: $(sum(load["pd"] for load in math["load"] if load["critical"] == 1))")
    #load_ref = sum(load["pd"][idx] for (idx, con) in enumerate(load["connections"]) for (i,load) in ref[:load] )
    load_ref = []
    for (i, load) in sort(ref[:load])
        cons = load["connections"]
        for idx in 1:length(cons)
            push!(load_ref, load["pd"][idx])
        end
    end
    load_ref_sum = sum(load_ref)
    println("Total load in reference: $load_ref_sum")

    gen_ref = []# sum(gen["pg"] for (i,gen) in ref[:gen])
    for (i, gen) in ref[:gen]
        cons = gen["connections"]
        for idx in 1:length(cons)
            push!(gen_ref, gen["pg"][idx])
        end
    end
    gen_ref_sum = sum(gen_ref)
    println("Total generation in reference: $gen_ref_sum")

    gen_soln = []# sum(gen["pg"] for (i,gen) in ref[:gen])
    for (i, gen) in res["gen"]
        for idx in 1:length(gen["pg"])
            push!(gen_soln, gen["pg"][idx])
        end
    end
    gen_soln_sum = sum(gen_soln)
    println("Total generation in solution: $gen_soln_sum")

    #load_served = sum((load["pd"]) for (i,load) in res["load"])
    load_served = []
    load_shed = []
    idxs = sort(parse.(Int,collect(keys(res["load"]))))
    for i in 1:length(idxs)
        load = res["load"][string(i)]
        for idx in 1:length(load["pd"])
            push!(load_served, load["pd"][idx])
        end
        # push!(load_shed, load["pshed"])
    end
    load_served_sum = sum(load_served)

    switch_statuses = Dict{String, Any}()
    for (id, switch) in res["switch"]
        switch_statuses[id] = switch["state"]
    end

    switch_names = Dict{String, Any}()
    for (id, switch) in ref[:switch]
        @info id
        @info switch
        switch_names[string(id)] = switch["name"]
    end

    println("Total load served in ACPF solution: $load_served_sum")
    #println("Total load shed in ACPF solution: $(sum(load_shed))")
    println("Load served percentage: $(load_served_sum/load_ref_sum*100) %")
    push!(served, (load_served_sum/load_ref_sum)*100)
    #end
    #println(served)

    # Calculate and print fairness indices
    served_array = collect(load_served./load_ref)
    # println("Gini Index: ", gini_index(served_array))
    # println("Jain's Index: ", jains_index(served_array))    
    # println("Palma Ratio: ", palma_ratio(served_array))
    # println("Alpha Fairness (alpha=1): ", alpha_fairness(served_array, 1))
    # println("Alpha Fairness (alpha=0.5): ", alpha_fairness(served_array, 0.5))
    # println("Alpha Fairness (alpha=5): ", alpha_fairness(served_array, 5))

    # Put the fairness results into a dataframe, save as csv and print to the terminal
    push!(df_load_results.TestName, "$test_name")
    push!(df_load_results.LoadServed, load_served_sum)
    push!(df_load_results.LoadReference, load_ref_sum)
    push!(df_load_results.LoadServedSetting, ls_percent)
    push!(df_load_results.LoadServedPercentage, (load_served_sum/load_ref_sum)*100)
    push!(df_load_results.GenSoln, gen_soln_sum)
    println(df_load_results)
    

    push!(df_fairness_results.TestName, "$test_name")
    push!(df_fairness_results.GiniIndex, gini_index(served_array))
    push!(df_fairness_results.JainsIndex, jains_index(served_array))
    push!(df_fairness_results.PalmaRatio, palma_ratio(served_array))
    push!(df_fairness_results.AlphaFairness1, alpha_fairness(served_array, 1))
    println(df_fairness_results)

    push!(df_decision_results.TestName, "$test_name")
    push!(df_decision_results.LoadBlocks, res["block"])
    push!(df_decision_results.Switches, switch_statuses)
    push!(df_decision_results.SwitchNames, switch_names)
    push!(df_decision_results.BlockBusMap, ref[:block_loads])
    push!(df_decision_results.BusMap, math["bus_lookup"])
end
CSV.write("fairness_summary.csv", df_load_results)
CSV.write("fairness_indices.csv", df_fairness_results)
CSV.write("fairness_decisions.csv", df_decision_results)
    # Run the diagnostic
    #diagnose_infeasibility(mld_model, ref, var, con_ref)
      # Print the following fairness indices: Gini index, Jain's index, Palma Ratio, Alpha fairness for alpha=1
