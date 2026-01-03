"""
Script to test the ACOPF and ensure it can solve with zero loads
"""

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

#case = "ieee_13_aw_edit/case_file_1trans_kron_reduced_3ph3wr_all_switches.dss"
#case = "ieee_13_aw_edit/motivation_b.dss"
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

math["block"] = Dict{String,Any}()
for (block, loads) in enumerate(lbs)
    math["block"][string(block)] = Dict("id"=>block, "state"=>0)
end
switch_selection = Dict{Int,Int}()
load_selection = Dict{Int,Float64}()
block_selection = Dict{Int,Float64}()
# for (switch_id, switch) in math["switch"]
#     @info switch
#     switch_selection[parse(Int,switch_id)] = 0.0
# end
# for (load_id, load) in  (math["load"])
#     load_selection[parse(Int,load_id)] = 0.0
# end
# for (block_id, block) in (math["block"])
#     block_selection[parse(Int,block_id)] = 0.0
# end
block_selection[1] = 1.0
block_selection[2] = 0.0
block_selection[3] = 1.0

switch_selection[1] = 1.0
switch_selection[2] = 1.0
switch_selection[3] = 0.0

opf = instantiate_mc_model(math, IVRUPowerModel, build_mc_opf; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
set_optimizer(opf.model, ipopt)
optimize!(opf.model)
objective_value(opf.model)
ref = opf.ref[:it][:pmd][:nw][0]
for (block, loads) in ref[:block_loads]
    for load in loads
        load_selection[load] = block_selection[block]
    end
end
math_out = update_network(math,block_selection, load_selection, switch_selection, ref, 1)

opf_new = instantiate_mc_model(math_out, IVRUPowerModel, build_mc_opf; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
set_optimizer(opf_new.model, ipopt)
optimize!(opf_new.model)
objective_value(opf_new.model)

pm_ivr_soln = solve_mc_pf(math_out, IVRUPowerModel, ipopt)
pm_ivr_opf_soln = solve_mc_opf(math_out, IVRUPowerModel, ipopt)
print("ACOPF termination status: $(pm_ivr_opf_soln["termination_status"])\n")
print("ACPF termination status: $(pm_ivr_soln["termination_status"])\n")