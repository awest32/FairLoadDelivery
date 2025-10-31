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
case = "three_bus_constrained_generation.dss"
casepath = "data/$case"
file = joinpath(dir, "..", casepath)


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
ls_percent = .39 # ensure not inf
for (i,gen) in math["gen"]
	if gen["source_id"] == "voltage_source.source"
		gen["pmax"] .= ls_percent*sum(load["pd"][idx] for (i,load) in math["load"] for (idx,c) in enumerate(load["connections"]))
		gen["qmax"] .= ls_percent*sum(load["qd"][idx] for (j,load) in math["load"] for (idx,c) in enumerate(load["connections"]))
		gen["pmin"] .= -ls_percent*sum(load["pd"][idx] for (i,load) in math["load"] for (idx,c) in enumerate(load["connections"]))
		gen["qmin"] .= -ls_percent*sum(load["qd"][idx] for (j,load) in math["load"] for (idx,c) in enumerate(load["connections"]))

	end
end

# Create the critical load set
#critical_load = ["645", "652", "675a", "675b", "675c"]
critical_load = ["l6"]
for (i,load) in math["load"]
	if load["name"] in critical_load
		load["critical"] = 1
		load["weight"] = 20
		println("Load $(load["name"]) at math load node $(i) is critical.")
	else
		load["critical"] = 0
		load["weight"] = 10
		println("Load $(load["name"]) at math load node $(i) is not critical.")

	end
end

math["block"] = Dict{String,Any}()
for (block, loads) in enumerate(lbs)
	math["block"][string(block)] = Dict("id"=>block, "state"=>0)
end

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
soln_ref = mld_model.sol[:it][:pmd][:nw][0]
ref = mld_model.ref[:it][:pmd][:nw][0]

pm_mld_soln = FairLoadDelivery.solve_mc_mld_switch(math, ipopt)
pm_mld_soln = mld_model.sol[:it][:pmd][:nw][0]
res = pm_mld_soln


"""
	Solve the relaxed model using a variable (alpha) to control the probability of rounding up.
	The goal is to generate integer solutions that are uniformly distributed 
"""

function random_rounding(pm::JuMP.Model; z_switch::Dict{String,Any}, z_block::Dict{String,Any}, reference::Dict{Symbol,Any}, math::Dict{String,Any}, bern_trade_off, bern_trade_off_max::Float64=2.0, max_iter::Int=10)
	opt_switch_vals = Dict{Int,Any}()
	opt_switch_val_vec =[]
	opt_switch_obj_vec = []
	opt_switch_iter_vec = []
	for (iter, i) in enumerate(bern_trade_off)
		bern_switch = Dict()
		switch_vals = Dict()
		bern_prob = []
		z_bn_switch = []
		for (ind,z) in z_switch 
			switch_val = z["state"]
			bern_prob = minimum([i * switch_val, 1.0])
			println("The original switch value is: $switch_val")
			println("Alpha = $i")
			println("Alpha*z_switch = $(i*switch_val)")
			println("The Bernoulli probability is: $bern_prob")
			if bern_prob > 1.0 || bern_prob < 0.0
				p = clamp(bern_prob, 0.0, 1.0)
			else
				p = bern_prob
			end
			bern_switch[ind] = rand(Bernoulli(p))  # Initial probability of rounding up
			println(bern_switch)
			# Sample from the Bernoulli distribution and compare
			push!(z_bn_switch, Int64(bern_switch[ind]))
		end

		bern_block = Dict()
		z_bn_block = []
		for (ind, lb) in z_block
			block_val = lb["status"]
			trade_off = rand(bern_trade_off)
			bern_prob = minimum([i * block_val, 1.0])
			if bern_prob > 1.0 || bern_prob < 0.0
				p_block = clamp(bern_prob, 0.0, 1.0)
			else
				p_block	 = bern_prob
			end
			bern_block[ind] = Bernoulli(p_block)  # Initial probability of rounding up
			# Sample from the Bernoulli distribution and compare
			bern_sample = rand(bern_block[ind])
			bern_block[ind] = rand(Bernoulli(p_block))  # Initial probability of rounding up
			println(bern_block)
			# Sample from the Bernoulli distribution and compare
			push!(z_bn_block, Int64(bern_block[ind]))
		end
		z_bern_block = Dict(parse.(Int,collect(keys(z_block))) .=> values(z_bn_block))
		z_bern_switch = Dict(parse.(Int,collect(keys(z_switch))) .=> z_bn_switch)

		"""
			Determine the feasiblity of the rounded solution by using the LD3F MLD
		"""
		# Create the radiality constrained problem
		model = JuMP.Model()

		# Create the variables for the Bernoulli switch states
		bern_switch = Dict{Int,Any}()
		bern_block = Dict{Int,Any}()
		bern_switch = JuMP.@variable(model, bern_switch[i in collect(keys(z_bern_switch))], base_name="bern_switch",
				lower_bound = 0,
				upper_bound = 1,
				start = 0.0
			)

		bern_block = JuMP.@variable(model, bern_block[i in collect(keys(z_bern_block))], base_name="bern_block",
				lower_bound = 0,
				upper_bound = 1,
				start = 0.0
			)

		"""
			Assign the rounded values to the switch states
		"""
			for s in collect(keys(z_bern_switch))
					JuMP.@constraint(model, bern_switch[s] == z_bern_switch[s])
			end
			for b in collect(keys(z_bern_block))
					JuMP.@constraint(model, bern_block[b] == z_bern_block[b])
			end

			FairLoadDelivery.constraint_radial_topology_jump(model,reference,bern_switch)

		"""
			Objective to minimize the distance from the rounded solution to the relaxed solution
		"""
		switch_dist = sum( (bern_switch[s] - z_switch[string(s)]["state"])^2 for s in keys(z_bern_switch))
		block_dist = sum( (bern_block[b] - z_block[string(b)]["status"])^2 for b in keys(z_bern_block))
		JuMP.@NLobjective(model, Min, switch_dist + block_dist)


		JuMP.set_optimizer(model, ipopt)
		optimize!(model)
		if JuMP.termination_status(model) == MOI.OPTIMAL || JuMP.termination_status(model) == MOI.LOCALLY_SOLVED
			println("Rounded solution is feasible for radiality constraints at iteration $iter")
			println("The switch states are:  $(value.(bern_switch))")
			
			"""
			Check for ACOPF feasibility with chosen switch values
			"""
			# Create a copy of the data file to use for the acopf problem
			math_cp = math

			# Change the switch data in the copy to match the rounded switch values
			# If status==0, set math["switch"][i]["status"]=0, math["switch"][i]["dispatchable"]=1
			# If status==1, set math["switch"][i]["status"]=1, math["switch"][i]["dispatchable"]=0
			for (i,switch) in z_bern_switch
				math_cp["switch"][string(i)]["state"] = switch
			end
			for (i,block) in z_bern_block
				math_cp["block"][string(i)]["state"] = block
			end

			pm_acopf_soln = solve_mc_opf_acp(math_cp, ipopt)
			#@info pm_acopf_soln["termination_status"] typeof(pm_acopf_soln["termination_status"])
			if pm_acopf_soln["termination_status"] == MOI.OPTIMAL || pm_acopf_soln["termination_status"] == MOI.LOCALLY_SOLVED ||  pm_acopf_soln["termination_status"] == MOI.ALMOST_LOCALLY_SOLVED
				println("ACOPF solution is feasible at iteration $iter")
		
				pm_mld_shed_soln = solve_mc_mld_shed_random_round(math_cp,ipopt)
				if pm_mld_shed_soln["termination_status"] == MOI.OPTIMAL || pm_mld_shed_soln["termination_status"] == MOI.LOCALLY_SOLVED #||  pm_mld_shed_soln["termination_status"] == MOI.ALMOST_LOCALLY_SOLVED
					obj_val = pm_mld_shed_soln["objective"]	
					println("MLD solution is objective value is $obj_val at iteration $iter")
					push!(opt_switch_val_vec, z_bern_switch)
					push!(opt_switch_obj_vec, obj_val)
					push!(opt_switch_iter_vec, iter)
				else
					println("MLD solution is not feasible at iteration $iter. Checking for a new rounded solution.")
				end
				### Run the MLD and get the objective with the new switch values
				### Store the set of switches for this solution and the objective value
				### Compare the new solution objective with the old objective if second time or greater in this loop
			else
				println("ACOPF solution is infeasible at iteration $iter")
				println("Checking a new Bernoulli value")
			end
		else
			println("Rounded solution is infeasible for radiality constraints at iteration $iter")
			println("Checking a new Bernoulli value")
		end
	end
	return opt_switch_val_vec, opt_switch_obj_vec, opt_switch_iter_vec
end

function diff_forward(model::JuMP.Model, fair_load_weights::Vector{Float64}; ϵ::Float64 = 1.0)
    # Initialization of parameters and references to simplify the notation
    vect_ref = model[:pshed]

	DiffOpt.set_forward_parameter(model, model[:fair_load_weights][:], fair_load_weights)
    DiffOpt.forward_differentiate!(model)
	
	dvect = DiffOpt.get_forward_variable(model, model[:pshed])
	vect = JuMP.value.(vect_ref)

    # Return the values as a vector
    return [vect, dvect]
end
function diff_forward_full_jacobian(model::JuMP.Model, fair_load_weights::Vector{Float64})
    weight_params = model[:fair_load_weights]
    pshed_vars = model[:pshed]
    
    weight_keys = sort(collect(eachindex(weight_params)))
    pshed_keys = sort(collect(eachindex(pshed_vars)))
    
    n_weights = length(weight_keys)
    n_pshed = length(pshed_keys)
    
    # Get current pshed
    pshed = [JuMP.value(pshed_vars[k]) for k in pshed_keys]
    
    # Build Jacobian column by column
    jacobian = zeros(n_pshed, n_weights)
    
    for j in 1:n_weights
        # Perturb ONLY weight j (standard basis vector e_j)
        for (k, wkey) in enumerate(weight_keys)
            perturbation = (k == j) ? 1.0 : 0.0
            DiffOpt.set_forward_parameter(model, weight_params[wkey], perturbation)
        end
        
        # Compute derivatives
        DiffOpt.forward_differentiate!(model)
        
        # Extract column j: [∂pshed[1]/∂weight[j], ∂pshed[2]/∂weight[j], ...]
        for (i, pkey) in enumerate(pshed_keys)
            jacobian[i, j] = DiffOpt.get_forward_variable(model, pshed_vars[pkey])
        end
    end
    
    return pshed, jacobian
end
"""
Most robust version that handles all container types
"""
function diff_forward_robust(model::JuMP.Model, fair_load_weights::Vector{Float64})
    weight_params = model[:fair_load_weights]
    pshed_vars = model[:pshed]
    
    # Convert to lists of VariableRefs
    if isa(weight_params, JuMP.Containers.DenseAxisArray)
        weight_keys = collect(eachindex(weight_params))
        weight_refs = [weight_params[k] for k in weight_keys]
    else
        weight_refs = collect(weight_params)
        weight_keys = 1:length(weight_refs)
    end
    
    if isa(pshed_vars, JuMP.Containers.DenseAxisArray)
        pshed_keys = collect(eachindex(pshed_vars))
        pshed_refs = [pshed_vars[k] for k in pshed_keys]
    else
        pshed_refs = collect(pshed_vars)
        pshed_keys = 1:length(pshed_refs)
    end
    
    @assert length(weight_refs) == length(fair_load_weights) "Weight dimension mismatch"
    
    # Set forward parameters
    for (var_ref, val) in zip(weight_refs, fair_load_weights)
        DiffOpt.set_forward_parameter(model, var_ref, val)
    end
    
    # Forward differentiate
    DiffOpt.forward_differentiate!(model)
    
    # Extract pshed values
    pshed = [JuMP.value(ref) for ref in pshed_refs]
    
    # Extract Jacobian: J[i,j] = ∂pshed[i]/∂weight[j]
    n_pshed = length(pshed_refs)
    n_weights = length(weight_refs)
    jacobian = zeros(n_pshed, n_weights)
    
    for i in 1:n_pshed
        dvars = DiffOpt.get_forward_variable(model, pshed_refs[i])
        
        # dvars should be a vector or container with derivatives w.r.t. each weight
        if isa(dvars, Number)
            # Single value - diagonal element only
            if i <= n_weights
                jacobian[i, i] = dvars
            end
        elseif isa(dvars, Vector)
            jacobian[i, :] = dvars
        elseif isa(dvars, JuMP.Containers.DenseAxisArray)
            for j in 1:n_weights
                jacobian[i, j] = dvars[weight_keys[j]]
            end
        elseif isa(dvars, Dict)
            for j in 1:n_weights
                jacobian[i, j] = get(dvars, weight_keys[j], 0.0)
            end
        end
    end
    
    return pshed, jacobian
end

"""
Helper to convert to Dict preserving indices
"""
function convert_to_dict(data)
    if isa(data, Dict)
        return data
    elseif isa(data, JuMP.Containers.DenseAxisArray)
        return Dict(idx => data[idx] for idx in eachindex(data))
    else
        error("Cannot convert $(typeof(data)) to Dict")
    end
end

"""
Construct the fairness problem
"""
function variable_fair_load_weights(model::JuMP.Model, ref::Dict{Symbol,Any})
    flw_val = ref[:load_weights]
    critical_loads = ref[:critical_loads]
    
    # Define bounds
    w_min = 10.0
    w_max_noncritical = 20.0
    w_max_critical = 4.0 * w_max_noncritical  # 80.0
    
    # Create variables with dynamic bounds
    model[:fair_load_weights_upper] = JuMP.@variable(
        model,
        [j in keys(flw_val)],
        lower_bound = w_min,
        upper_bound = (haskey(critical_loads, j)) ? 
                      w_max_critical : w_max_noncritical,
        base_name = "fair_load_weights_upper"
    )
    
    return model[:fair_load_weights_upper]
end

function constraint_critical_load_weights(model::JuMP.Model, ref::Dict{Symbol,Any})
    fair_load_weights_upper = model[:fair_load_weights_upper]
    for (d, crit_load) in ref[:load]
		if crit_load["critical"] == 1
			for (j, crit_load) in ref[:load]
				if d != j
					JuMP.@constraint(model, fair_load_weights_upper[d] == 2*fair_load_weights_upper[j])
				end
			end
		end
	end
end

function objective_maxmin_fairness(model::JuMP.Model, pshed; report::Bool=true)
    #maxmin fairness: infinity norm

    return JuMP.@objective(model, Max,
        LinearAlgebra.norm((pshed),Inf) #for i in _PMD.ids(pm, nw, :load)
    )
end

function objective_equal_fairness(model::JuMP.Model; report::Bool=true)
   alpha_val = 10.0
    return JuMP.@objective(model, Max,
         sum((pshed[i].^(1-alpha_val) for i in keys(model[:load]))./(1-alpha_val) )
    )
end

function objective_jain_fairness_pshed(model::JuMP.Model, pshed; ref::Dict{Symbol,Any}, report::Bool=true)
    load_ids = keys(ref[:load])
    n = length(load_ids)
    
    # Numerator: (sum of load served)^2
    numerator = sum(pshed[i] for i in load_ids
    )^2
    
    # Denominator: n * sum of (load shed)^2
    denominator = n * sum(
        (pshed[i])^2
        for i in load_ids
    )
    
    return JuMP.@objective(model, Max, numerator / denominator)
end

# Solve the maxmin fairness problem
function build_max_fairness_problem(ref::Dict{Symbol,Any}, pshed_update; report::Bool=true)
	model = JuMP.Model()
	variable_fair_load_weights(model, ref)
	constraint_critical_load_weights(model, ref)
	#objective_maxmin_fairness(model, pshed_update)
	objective_jain_fairness_pshed(model, pshed_update, ref=ref)
	JuMP.set_optimizer(model, ipopt)
	return model
end


"""
 Frank-Wolfe Algorithm to update the fairness weights
"""
# Update the sensitivites with the change in fairness weighted

K = 5  # Frank-Wolfe iterations
function frank_wolfe(ref::Dict{Symbol,Any}, math::Dict{String,Any}; K::Int=5)
	
	fair_model_out = JuMP.Model()
	mld_mod_out = JuMP.Model()
	mld_rr_out = JuMP.Model()
	n_loads = length(ref[:load_weights])
	load_ids = sort(collect(keys(ref[:load])))

	# gather the network data file for changes later 
	math_weight_update = math
	math_switch_update = math
	# Initialize storage
	flw_prev = Dict{Int, Vector{Float64}}()
	pshed_new = Dict{Int, Vector{Float64}}()
	pshed_approx = Dict{Int, Vector{Float64}}()
	dpshed_new = Dict{Int, Matrix{Float64}}()
	# Initial weights (iteration 0)
	flw_prev[0] = ones(n_loads) * 10.0

	# Rounding settings
	Random.seed!(1234) # Sets the seed to 1234
	bern_trade_off_max = 2
	max_iter = 10
	bern_trade_off = range(0.0, stop=bern_trade_off_max, length=max_iter)
	# Frank-Wolfe loop
	for k in 1:K
		# Get weights from previous iteration
		k=1
		weights_prev = flw_prev[k-1]

		# Assign the new weights as parameters to the loads
		for (d, load) in math_weight_update["load"]
			load["weight"] = weights_prev[parse(Int,d)]
		end

		# Solve the MLD with the updated fairness priorization weights
		pm_mld_shed_soln = solve_mc_mld_switch(math_weight_update, ipopt)
		z_switch = pm_mld_shed_soln["solution"]["switch"]
		z_block = pm_mld_shed_soln["solution"]["block"]
		######################################################################################
		# --- Random Rounding to get integer switch values ---
		# Force the switch values to be integer using random rounding 
		pm_mld_random_round = instantiate_mc_model(math_weight_update, LinDist3FlowPowerModel, build_mc_mld_shedding_random_rounding; ref_extensions=[FairLoadDelivery.ref_add_load_blocks!])
		mld_round_ref = pm_mld_random_round.ref[:it][:pmd][:nw][0]
		######################################################################################
		# TODO: Round for the load blocks too, add it to the distance objective and set the data accordingly
		######################################################################################
		(opt_switch_val_vec, opt_switch_obj_vec, opt_switch_iter_vec) = random_rounding(pm_mld_random_round.model; z_switch=z_switch, z_block=z_block, reference=mld_round_ref, math=math_switch_update, bern_trade_off=bern_trade_off)
		max_idx = argmax(opt_switch_obj_vec)
		# Get the corresponding values
		max_obj_val = opt_switch_obj_vec[max_idx]
		max_id = opt_switch_iter_vec[max_idx]
		max_switch_val = opt_switch_val_vec[max_idx]
		for (i,switch) in max_switch_val
			math_switch_update["switch"][string(i)]["state"] = switch
			for (block,switches) in ref[:block_switches]
				if i in switches && switch==1
					math_switch_update["block"][string(block)]["state"] = switch
				end
			end
		end

		# Solve the MLD using the rounded switch values and the fair weight parameter
		pm_mld_implicit_diff = instantiate_mc_model(math_switch_update, LinDist3FlowPowerModel, build_mc_mld_shedding_implicit_diff; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
		ref = pm_mld_implicit_diff.ref[:it][:pmd][:nw][0]
		modmod = pm_mld_implicit_diff
		mld_mod_out = modmod.model
		JuMP.optimize!(modmod.model)

		# Forward differentiation roundedMLD at previous weights
		(pshed, dpshed) = diff_forward_full_jacobian(modmod.model, weights_prev)
		pshed_new[k] = pshed
		dpshed_new[k] = dpshed
		######################################################################################
		# --- Build the fairness problem ---
		######################################################################################
		
		fair_mod = JuMP.Model()
		weights = variable_fair_load_weights(fair_mod, ref)

		# Set the critical load weight constraints
		for (d, crit_load) in ref[:load]
			if crit_load["critical"] == 1
				for (j, crit_load) in ref[:load]
					if d != j
						JuMP.@constraint(fair_mod, weights[d] >= 2*weights[j])
					end
				end
				println("Added critical load constraints for load $d")
			end
		end
	 	#constraint_critical_load_weights(fair_mod, ref)
		#objective_maxmin_fairness(model, pshed_update)

		# Create the pshed update using the pshed sensitivities (dpshed) and the previous weights
		pshed_update = Dict{Any, Any}()
		for (i, load_id) in enumerate(load_ids)
			pshed_update[load_id] = sum(dpshed[i,j]*weights[i] - dpshed[i,j]*weights_prev[i] for j in load_ids) #+ pshed[i]
		end

		objective_jain_fairness_pshed(fair_mod, pshed_update, ref=ref)
		JuMP.set_optimizer(fair_mod, ipopt)
		fair_model_out = fair_mod
		optimize!(fair_mod)
		weights_new = JuMP.value.(weights)
		pshed_new_k_vec = [JuMP.value(pshed_update[id]) for id in sort(collect(keys(pshed_update)))]
		@info weights_new
		# Store results
		pshed_approx[k] = pshed_new_k_vec
		flw_prev[k] = weights_new
		
		println("Iteration $k: weights = $(weights_new)")
	end

	return flw_prev, pshed_approx, pshed_new, dpshed_new, math_switch_update, math_weight_update, fair_model_out, mld_mod_out
end
#upper_model = build_max_fairness_problem(reference, pshed)
#JuMP.optimize!(upper_model)
(flw_prev, pshed_approx, pshed_new, dpshed_new, math_switch_update, math_weight_update, fair_model_out, mld_mod_out) = frank_wolfe(ref, math; K=1)

# 		######################################################################################
# 		# --- LMO: 1. Compute gradient of the objective and 2. solve LP to find a vertex and 3. take convex combination to get new weights ---
# 		######################################################################################
# 		#TODO: Gradient of the upper level objective w.r.t. the weights
# 		# We want to solve: min_w || pshed(w) ||_inf s.t. w ∈[w_min,w_max]ᵈ
# 		@assert n_loads == length(weights_prev)
# 		∇F = zeros(n_loads)  # gradient wrt weights (length d)

# 		# indices of entries attaining the max absolute value
# 		Mabs = maximum(abs.(pshed))
# 		tol = 1e-12 * (1 + Mabs)
# 		maxima = findall(i -> abs(pshed[i]) ≥ Mabs - tol, eachindex(pshed))

# 		if length(maxima) == 1
# 			j = maxima[1]
# 			# full gradient is sign(pshed[j]) times the j-th row of the Jacobian
# 			∇F .= sign(pshed[j]) .* view(dpshed, j, :)
# 		else
# 			@info "Multiple maxima found: $maxima"
# 			# choose any convex combination over the tie set (here: uniform)
# 			α = fill(1.0/length(maxima), length(maxima))  # or use a Dirichlet draw
# 			# form s in R^m, then J' * s
# 			s = zeros(length(pshed))
# 			@inbounds for (a, i) in zip(α, maxima)
# 				s[i] = a * sign(pshed[i])
# 			end
# 			∇F .= dpshed' * s
# 		end

# 		# Determine the upper level fairness problem objective function
# 		# upper_model = build_max_fairness_problem(reference, pshed)
# 		# upper_obj_diff = ForwardDiff.gradient(objective_function(upper_model), upper_model[:fair_load_weights_upper])
# 		# println("Upper level objective gradient: $upper_obj_diff")
# 		# Compute new weights (this would come from your FW algorithm)
# 		# TODO: LMO
# 		w_min, w_max = 10.0, 20.0 #TODO: set these based on your problem
# 		lmo = JuMP.Model(Gurobi.Optimizer)
# 		load_ids = sort(collect(keys(ref[:load])))
# 		critical_loads = [id for id in load_ids if get(ref[:load][id], "critical", 0) == 1]
# 		non_critical_loads = [id for id in load_ids if get(ref[:load][id], "critical", 0) == 0]

# 		@variable(lmo, w[1:n_loads])
# 		for (d, crit_load) in ref[:load]
# 			if crit_load["critical"] == 1
# 				for (j, crit_load) in ref[:load]
# 					if d != j
# 						JuMP.@constraint(lmo, w[d] >= 2*w[j])
# 					end
# 				end
# 				JuMP.@constraint(lmo,[i=1:n_loads], w[i] <= 4*w_max)
# 			else
# 				JuMP.@constraint(lmo,[i=1:n_loads], w[i] >= w_min)
# 				JuMP.@constraint(lmo,[i=1:n_loads], w[i] <= w_max)
# 			end
# 		end

# 		# Linear objective: minimize the inner product of the gradient and the weights
# 		@objective(lmo, Min, sum(∇F[i]*w[i] for i in 1:n_loads))
# 		# Solve the LP
# 		JuMP.optimize!(lmo)

# 		# Test if ANY feasible point exists
# println("Testing feasibility with simple point...")
# test_weights = fill(w_min, n_loads)
# w_max_critical = 4.0 * w_max
# # Set critical loads to max
# for (i, load_id) in enumerate(load_ids)
#     if get(ref[:load][load_id], "critical", 0) == 1
#         test_weights[i] = w_max_critical
#     else
#         test_weights[i] = w_min
#     end
# end

# println("Test weights: ", test_weights)

# # Check if satisfies constraints
# for crit_id in critical_loads
#     crit_idx = findfirst(==(crit_id), load_ids)
#     for noncrit_id in non_critical_loads
#         noncrit_idx = findfirst(==(noncrit_id), load_ids)
#         satisfied = test_weights[crit_idx] >= 2.0 * test_weights[noncrit_idx]
#         println("  w[$crit_idx]($(test_weights[crit_idx])) >= 2*w[$noncrit_idx]($(2*test_weights[noncrit_idx])): $satisfied")
#     end
# end
# 		# update the weights with a montonic stepsize
# 		vertex_new = JuMP.value.(w)
# 		γ = 2.0/(k+2.0)  
# 		weights_new = (1-γ)*weights_prev .+ γ*vertex_new	
# 		# --- End LMO ---
# 		# Original notes:
# 		# For now, using model variable as example
# 		# weights_new = JuMP.value.(model[:fair_load_weights_upper])	
# 		######################################################################################

# 		# Compute linear approximation of pshed at new weights
# 		pshed_step_k = Float64[]
# 		for (idx, i) in enumerate(load_ids)
# 			# pshed_new ≈ pshed_old + dpshed * (w_new - w_old)
# 			pshed_new_i = pshed[idx] + dpshed[idx] * (weights_new[idx] - weights_prev[idx])
# 			push!(pshed_step_k, pshed_new_i)
# 		end
		
# 		# Store results
# 		pshed_step[k] = pshed_step_k
# 		flw_prev[k] = weights_new
		
# 		println("Iteration $k: weights = $(weights_new)")
# 	end

# 	return flw_prev, pshed_step
# end
# #upper_model = build_max_fairness_problem(reference, pshed)
# #JuMP.optimize!(upper_model)
# (flw_prev, pshed_step) = frank_wolfe(reference, math; K=1)


# using Plots

# # Jain Fairness function
# function jain_fairness(x)
#     n = length(x)
#     sum_x = sum(x)
#     sum_x_sq = sum(x.^2)
#     return sum_x^2 / (n * sum_x_sq)
# end

# # X values from 1 to 10
# x_range = 1:10

# # Different scenarios for 2 users
# scenario1 = [jain_fairness([x, 5.0]) for x in x_range]  # User 2 = 5
# scenario2 = [jain_fairness([x, x]) for x in x_range]    # Both equal
# scenario3 = [jain_fairness([x, 11.0-x]) for x in x_range]  # Constant sum
# scenario0 = [jain_fairness(x) for x in x_range]  # User 2 = 10
# # Plot
# plot(x_range, scenario1, label="User 2 = 5", marker=:circle, lw=2)
# plot!(x_range, scenario2, label="Both equal", marker=:square, lw=2, ls=:dash)
# plot!(x_range, scenario3, label="Sum = 11", marker=:diamond, lw=2, ls=:dot)
# plot!(x_range, scenario0, label="User 2 = 10", marker=:triangle, lw=2, ls=:dot)
# hline!([1.0], label="Perfect Fairness", ls=:dash, color=:gray)

# xlabel!("User 1 Allocation (x)")
# ylabel!("Jain Fairness Index")
# title!("Jain Fairness: x ∈ [1,10]")
# ylims!(0, 1.1)
# n_loads = length(ref[:load_weights])
# load_ids = sort(collect(keys(ref[:load])))

# # Initialize storage
# flw_prev = Dict{Int, Vector{Float64}}()
# pshed_step = Dict{Int, Vector{Float64}}()

# # Initial weights (iteration 0)
# flw_prev[0] = ones(n_loads) * 10.0

# # Frank-Wolfe loop
# for k in 1:K
#     # Get weights from previous iteration
#     weights_prev = flw_prev[k-1]
    
#     # Forward differentiation at previous weights
#     pshed, dpshed = diff_forward(model, weights_prev)
    
#     # Compute new weights (this would come from your FW algorithm)
#     # For now, using model variable as example
#     weights_new = JuMP.value.(model[:fair_load_weights_upper])
    
#     # Compute linear approximation of pshed at new weights
#     pshed_step_k = Float64[]
#     for (idx, i) in enumerate(load_ids)
#         # pshed_new ≈ pshed_old + dpshed * (w_new - w_old)
#         pshed_new_i = pshed[idx] + dpshed[idx] * (weights_new[idx] - weights_prev[idx])
#         push!(pshed_step_k, pshed_new_i)
#     end
    
#     # Store results
#     pshed_step[k] = pshed_step_k
#     flw_prev[k] = weights_new
    
#     println("Iteration $k: weights = $(weights_new)")
# end

# # Access all previous weights
# for k in 0:K
#     println("Iteration $k weights: ", flw_prev[k])
# end

# K = length(ref[:load_weights])
# pshed_step = []
# flw_prev = ones(length(ref[:load_weights]))*10.0
# for k in 2:K
# 	for (i,load) in ref[:load]
# 		push!(pshed_step,  pshed[i] + dpshed[i]*(model[:fair_load_weights_upper][k]-flw_prev[k-1][i]))
# 	end
# end
# Fairness objectives
 # +
          #  - sum(_PMD.var(pm, nw)[:pshed][i][c]^2 for (i,bus) in _PMD.ref(pm, nw)[:load] for c in bus["connections"]) / (length(ref_onm[:load]) * sum(pshed[i][c] for (i,bus) in _PMD.ref(pm,nw)[:load] for c in bus["connections"])^2)
		#- sum(pd_zblock[i][c]^(1-alpha_val) for (i,bus) in ref_onm[:load] for c in bus["connections"])/(1-alpha_val) 
		#- sum(log(pd_zblock[i][c]) for (i,bus) in ref_onm[:load] for c in bus["connections"])

#@objective(model, Min, sum(var(pm, :pg)[1]))


# # Iterate through all constraint types and print constraints
# for T in JuMP.list_of_constraint_types(model)
#     constraints = all_constraints(model, T...)
#     println("Constraint Type: $T")
#     for con in constraints
#         println("\t$con")
#     end
# end

