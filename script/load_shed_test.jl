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
using PowerPlots
# using DataFrames
ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# To make a bilevel JuMP model, we need to create a BilevelJuMP model here 
juniper = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt, "mip_solver"=>highs)


## Main loop
dir = dirname(@__FILE__)

#case = "ieee_13_clean/ieee13.dss"
case = "ieee_13_aw_edit/case_file_1trans_kron_reduced_3ph3wr.dss"
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

# Add a switch 
#add_switch!(eng,"632633", "632.1.2.3", "633.1.2.3", [1,2,3], [1,2,3]; linecode="mtx601")
#eng["switch"]["632671"]["dispatchable"] = "YES"

# Make a deepcopy of the switch and change the fields
# Make a deepcopy of a line and change the fields, make this line have the same buses

 #function add_gens!(eng)
        #for (d, load) in eng["load"]
            #if p[!,pen][load["index"]]
                #phases = length(load["connections"])-1
          
            #     switch_name = 632671
            #     eng["switch"]["$switch_name"] = deepcopy(eng["switch"]["671692"])
            #    eng["switch"]["$switch_name"]["f_bus"] = "632"#.1.2.3"
            #     eng["switch"]["$switch_name"]["t_bus"] = "671"# .1.2.3"
            
            # switch_name = 632645
            #     eng["switch"]["$switch_name"] = deepcopy(eng["switch"]["671692"])
            #    eng["switch"]["$switch_name"]["f_bus"] = "632"#.1.2.3"
            #     eng["switch"]["$switch_name"]["t_bus"] = "645"# .1.2.3"
    #end
        #end
       # @show "added $(switch_counter-1) PV systems at penetration $(pen) for $(length(eng["load"])) loads"

    # end
    # add_gens!(eng)

#function add_gens!(eng)
        #for (d, load) in eng["load"]
            #if p[!,pen][load["index"]]
                #phases = length(load["connections"])-1
                
                # line_name = 632671
                # eng["line"]["$line_name"] = deepcopy(eng["line"]["671680"])
                # eng["line"]["$line_name"]["f_bus"] = "632"#.1.2.3"
                # eng["line"]["$line_name"]["t_bus"] = "671"# .1.2.3"
                
                # # line_name = 632645
                # eng["line"]["$line_name"] = deepcopy(eng["line"]["671680"])
                # eng["line"]["$line_name"]["f_bus"] = "632"#.1.2.3"
                # eng["line"]["$line_name"]["t_bus"] = "645"# .1.2.3"
            #end
        #end
       # @show "added $(switch_counter-1) PV systems at penetration $(pen) for $(length(eng["load"])) loads"

    # end
    # add_gens!(eng)

math = PowerModelsDistribution.transform_data_model(eng)
# for (i,branch) in math["branch"]
#     if branch["name"] == "632671"
#         delete!(math["branch"],string(i))
#     end
#     # if branch["name"] == "632645"
#     #     delete!(math["branch"],string(i))
#     # end
# end
   
# Remove the lines parallel to the switches
# Look up the correct branch
# Use the delete! command in julia to remove from the math dictionary
for (idx, switch) in math["switch"]
    switch["state"] = 1
end
lbs = PowerModelsDistribution.identify_load_blocks(math)
get(eng, "time_series", Dict())

for (i,bus) in math["bus"]

		bus["vmax"][:] .= 1.1
		bus["vmin"][:] .= 0.9
end

# Save for the relaxed version when using nonlinear terms in objective
#add_start_vrvi!(math)


p = powerplot(eng, bus    = (:data=>"bus_type", :data_type=>"nominal"),
                    branch = (:data=>"index", :data_type=>"ordinal"),
                    gen    = (:data=>"pmax", :data_type=>"quantitative"),
                    load   = (:data=>"pd",  :data_type=>"quantitative"),
                    width = 300, height=300
)
 save("powerplot.pdf", p)
# Ensure the generation from the source bus is less than the max load
# First calculate the total load
#ls_percent = 0. # ensure not inf
served = [] #Dict{Any,Any}()
ls_percent = 1
#for ls_percent in LinRange(0,1,11)
    tot_pd = sum(load["pd"][idx] for (i,load) in math["load"] for (idx,c) in enumerate(load["connections"]))
    tot_qd = sum(load["qd"][idx] for (i,load) in math["load"] for (idx,c) in enumerate(load["connections"]))
    println(tot_pd)
    for (i,gen) in math["gen"]
        if gen["source_id"] == "voltage_source.source"
            pd_phase1=0
            pd_phase2=0
            pd_phase3=0
            qd_phase1=0
            qd_phase2=0
            qd_phase3=0
            for (ind, d) in math["load"]
                @info d
                @info d["connections"]
                for (idx, con) in enumerate(d["connections"])
                    @info "Load at connection $(d["connections"][idx]) has pd=$(d["pd"][idx]) and qd=$(d["qd"][idx])"
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
        end
    end

    # Create the critical load set
    #critical_load = ["645", "652", "675a", "675b", "675c"]
    critical_load = ["l4"]
    for (i,load) in math["load"]
        if load["name"] in critical_load
            load["critical"] = 1
            load["weight"] = 10
            println("Load $(load["name"]) at math load node $(i) is critical.")
        else
            load["critical"] = 0
            load["weight"] = 10
            println("Load $(load["name"]) at math load node $(i) is not critical.")

        end
    end

    for (switch_id, switch) in enumerate(math["switch"])
    # math["switch"][string(switch_id)]["state"] = 0
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
    # math["switch"]["1"]["state"] = 0
    # math["switch"]["2"]["state"] = 0
    # math["switch"]["3"]["state"] = 0

    math["block"] = Dict{String,Any}()
    for (block, loads) in enumerate(lbs)
        math["block"][string(block)] = Dict("id"=>block, "state"=>0)
    end
    # math["block"]["1"]["state"] = 0
    # math["block"]["2"]["state"] = 0
    #math["block"]["3"]["state"] = 0

    mld_model = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_switchable; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
    mld = mld_model.model
    set_optimizer(mld, gurobi)
    optimize!(mld)
    con_ref = mld_model.con[:it][:pmd][:nw][0]
    var = mld_model.var[:it][:pmd][:nw][0]
    soln_ref = mld_model.sol[:it][:pmd][:nw][0]
    ref = mld_model.ref[:it][:pmd][:nw][0]


    pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch(math, gurobi)
    res = pm_mld_soln["solution"]
    # println("Load served: $(sum(load["pd"] for load in math["load"] if load["critical"] == 1))")
    #load_ref = sum(load["pd"][idx] for (idx, con) in enumerate(load["connections"]) for (i,load) in ref[:load] )
    load_ref = []
    for (i, load) in ref[:load]
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
    for (i, load) in res["load"]
        for idx in 1:length(load["pd"])
            push!(load_served, load["pd"][idx])
        end
    end
    load_served_sum = sum(load_served)
    println("Total load served in MLD solution: $load_served_sum")
    println("Load served percentage: $(load_served_sum/load_ref_sum*100) %")
    push!(served, (load_served_sum/load_ref_sum)*100)
#end
#println(served)


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