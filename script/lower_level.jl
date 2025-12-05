
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

# using DataFrames
ipopt = Ipopt.Optimizer
gurobi = Gurobi.Optimizer

ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# To make a bilevel JuMP model, we need to create a BilevelJuMP model here 
juniper = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>ipopt, "mip_solver"=>highs)

#global solver = ipopt

## Main loop
dir = dirname(@__FILE__)

case = "ieee_13_aw_edit/motivation_b.dss"

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
lbs = PowerModelsDistribution.identify_load_blocks(math)
get(eng, "time_series", Dict())

for (i,bus) in math["bus"]

		bus["vmax"][:] .= 1.1
		bus["vmin"][:] .= 0.9
end


# Ensure the generation from the source bus is less than the max load
# First calculate the total load
served = [] #Dict{Any,Any}()
ls_percent = 0.98
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
        load["critical"] = 1
        load["weight"] = 10
        #println("Load $(load["name"]) at math load node $(i) is critical.")
    else
        load["critical"] = 0
        load["weight"] = 10
        #println("Load $(load["name"]) at math load node $(i) is not critical.")

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

# Manual feasibility test
math["switch"]["1"]["state"] = 0 # Open the switch to force load shedding
math["switch"]["2"]["state"] = 1 # Open the switch to force load shedding
math["switch"]["3"]["state"] = 1 # Open the switch to force load shedding
math["block"] = Dict{String,Any}()
for (block, loads) in enumerate(lbs)
    math["block"][string(block)] = Dict("id"=>block, "state"=>0)
end

# Solve the MILP MLD problem
mld_mip_soln = FairLoadDelivery.solve_mc_mld_switch(math, gurobi)

# Extract the fixed variables from the MLD solution
math_mip = deepcopy(math)
for (switch_id, switch) in enumerate(math_mip["switch"])
    #@info "Setting switch $switch_id to state to $(mld_mip_soln["solution"]["switch"][string(switch_id)]["state"]) from MIP solution."
    switch_var = mld_mip_soln["solution"]["switch"][string(switch_id)]["state"]
    #@info "Switch variable before rounding: $switch_var"
    math_mip["switch"][string(switch_id)]["state"] = round(Int, switch_var)
end
for (block_id, block) in enumerate(math_mip["block"])
    block_var = mld_mip_soln["solution"]["block"][string(block_id)]["status"]
    math_mip["block"][string(block_id)]["state"] = round(Int, block_var)
end

# Solve the continues rounded mld using the mip solutions
mld_fixed_soln = FairLoadDelivery.solve_mc_mld_shed_random_round(math_mip, ipopt)

#print the switch and load block states
println("Switch states from MLD MIP solution:")
for (switch_id, switch) in (math_mip["switch"])
    println("Switch $switch_id state: $(switch["state"])")
end
println("Load block states from MLD MIP solution:")
for (block_id, block) in (math_mip["block"])
    println("Block $block_id state: $(block["state"])")
end
# pm_ivr_soln = solve_mc_pf(math, IVRUPowerModel, ipopt)
# pm_ivr_opf_soln = solve_mc_opf(math, IVRUPowerModel, ipopt)

# model_bern = JuMP.Model()
# set_attribute(model_bern, "hsllib", HSL_jll)
# set_attribute(model_bern, "linear_solver", "ma27")

# res,ref = solve_mld_relaxed(math; optimizer=ipopt)
# z_relaxed = res["switch_states"]
# pm_ivr_soln = solve_mc_pf(math, IVRUPowerModel, ipopt)

# bernoulli_samples = generate_bernoulli_samples(res["switch_states"]; n_samples=10, seed=42)
# optimizer=ipopt

# bernoulli_selection,switch_ids = radiality_check(ref, z_relaxed,
#                                       bernoulli_samples;
#                                       optimizer=ipopt)    

# # ac_bernoulli,ac_bernoulli_val = ac_feasibility_test(math, bernoulli_selection, switch_ids; optimizer=ipopt)
# # if length(ac_bernoulli) == 0
# #     error("No feasible AC solutions found among Bernoulli samples.")
# # else
# #     println("Number of feasible AC solutions from Bernoulli samples: $(length(ac_bernoulli))")
# # end

# best_sample_idx, best_switch_config = find_best_switch_set(math, ac_bernoulli, switch_ids; optimizer=ipopt)
# if best_sample_idx == 0
#     error("No feasible MLD solutions found among AC feasible Bernoulli samples.")
# end
# println("Best feasible switch configuration found at sample index: $best_sample_idx")
# println("Best switch configuration: $best_switch_config")

# mld_rounded_soln = solve_mc_mld_shed_random_round(math_fin, optimizer)
