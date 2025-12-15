
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
#case = "ieee_13_aw_edit/case_file_1trans_kron_reduced_3ph3wr_all_switches.dss"
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
ls_percent = 0.5
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
critical_id =[]
#critical_load = ["l4"]
for (i,load) in math["load"]
    if load["name"] in critical_load
        load["critical"] = 1
        load["weight"] = 1000
        push!(critical_id,parse(Int,i))
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

# Solve the parameterized MLD 
mld_paramed_soln = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math, ipopt)
# Extract the pshed from the loads in the solution
pshed = Dict{String,Float64}()
for (load_id, load) in (mld_paramed_soln["solution"]["load"])
    pshed[string(load_id)] = load["pshed"]
end

mld_paramed = instantiate_mc_model(math, LinDist3FlowPowerModel, build_mc_mld_shedding_implicit_diff; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
ref = mld_paramed.ref[:it][:pmd][:nw][0]

function diff_forward_full_jacobian(model::JuMP.Model, fair_load_weights::Vector{Float64})
    weight_params = model[:fair_load_weights]
    pshed_vars = model[:pshed]
    
    weight_keys = (collect(eachindex(weight_params)))
    pshed_keys = (collect(eachindex(pshed_vars)))
    
    n_weights = length(weight_keys)
    n_pshed = length(pshed_keys)
    
    # Get current pshed
   # pshed = [JuMP.value(pshed_vars[k]) for k in pshed_keys]
    
    # Build Jacobian column by colum
    
    jacobian = zeros(n_pshed, n_weights)
    dpshed = Dict{Any,Any}()
    for j in 1:n_weights
        # Perturb ONLY weight j (standard basis vector e_j)
        for (k, wkey) in enumerate(weight_keys)
            perturbation = (k == j) ? fair_load_weights[j] : 0.0
            DiffOpt.set_forward_parameter(model, weight_params[wkey], perturbation)
        end

        optimize!(model)

        # Compute derivatives
        DiffOpt.forward_differentiate!(model)
        
        # Extract column j: [∂pshed[1]/∂weight[j], ∂pshed[2]/∂weight[j], ...]
        for (i, pkey) in enumerate(pshed_keys)
            jacobian[i, j] = DiffOpt.get_forward_variable(model, pshed_vars[pkey])
            dpshed[i,pkey] = jacobian[i, j]
        end
    end
    
    return jacobian, dpshed
end
function diff_forward(model::JuMP.Model, weights_prev::Vector{Float64}; ϵ::Float64 = 1.0)
    # Initialization of parameters and references to simplify the notation
    vect_ref = model[:pshed]
    weights_new = collect(eachindex(weights_prev))
    weight_vars = collect(eachindex(model[:fair_load_weights]))
	DiffOpt.set_forward_parameter(model, weight_vars, 10)
    DiffOpt.forward_differentiate!(model)
	
	dvect = DiffOpt.get_forward_variable(model, model[:pshed])
	vect = JuMP.value.(vect_ref)

    # Return the values as a vector
    return [vect, dvect]
end
# Initial weights (iteration 0)
weights_prev = ones(length(ref[:load])) * 10.0
# Use the parameterized MLD solution to perform implicit differentiation with DiffOpt.jl 
dpshed_mat, dpshed_dict = diff_forward_full_jacobian(mld_paramed.model, weights_prev)

using Plots

# Labels for rows/columns (use pshed keys)
 # original ordering
keys_paramed = [string(k[1]) for k in keys(mld_paramed.model[:fair_load_weights])]
row_labels = keys_paramed  # loads
col_labels = keys_paramed  # weights

# Plot heatmap
heatmap_plot = heatmap(
    col_labels, row_labels, dpshed_mat;
    xlabel = "Load Weight ID",
    ylabel = "Load Shed ID",
    title = "Sensitivity of Load Shedding (dP/dw)",
    c = :viridis,
    #colorbar_title = "∂P/∂w",
    size = (600, 500),
    right_margin = 5Plots.mm,
    top_margin = 5Plots.mm
)
annotate!(16.5, 16, text("∂P/∂w", 9, "Computer Modern"))


# Save the heatmap
savefig(heatmap_plot, "load_shed_sensitivities_heatmap.svg")  # save as SVG
println("Heatmap saved as load_shed_sensitivities_heatmap.svg")


# Solve the MILP MLD problem
# mld_mip_soln = FairLoadDelivery.solve_mc_mld_switch_integer(math, gurobi)

# Extract the fixed variables from the MLD solution
# math_mip = deepcopy(math)
# for (switch_id, switch) in enumerate(math_mip["switch"])
#     #@info "Setting switch $switch_id to state to $(mld_mip_soln["solution"]["switch"][string(switch_id)]["state"]) from MIP solution."
#     switch_var = mld_mip_soln["solution"]["switch"][string(switch_id)]["state"]
#     #@info "Switch variable before rounding: $switch_var"
#     math_mip["switch"][string(switch_id)]["state"] = round(Int, switch_var)
# end
# for (block_id, block) in enumerate(math_mip["block"])
#     block_var = mld_mip_soln["solution"]["block"][string(block_id)]["status"]
#     math_mip["block"][string(block_id)]["state"] = round(Int, block_var)
# end

# # Solve the AC OPF MLD with fixed discrete variables from the MIP solution
# #ac_mld_fixed_soln = FairLoadDelivery.solve_mc_pf_aw(math_mip, ipopt)

# # Solve the continues rounded mld using the mip solutions
# mld_fixed_soln = FairLoadDelivery.solve_mc_mld_shed_random_round(math_mip, ipopt)

# #print the switch and load block states
# println("Switch states from MLD MIP solution:")
# for (switch_id, switch) in (math_mip["switch"])
#     println("Switch $switch_id state: $(switch["state"])")
# end
# println("Load block states from MLD MIP solution:")
# for (block_id, block) in (math_mip["block"])
#     println("Block $block_id state: $(block["state"])")
# end
