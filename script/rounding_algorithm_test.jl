
function solve_mld_relaxed(data::Dict{String, Any}; optimizer=Ipopt.Optimizer)
    soln = FairLoadDelivery.solve_mc_mld_switch_relaxed(data, optimizer)
    # Build the MLD problem with relaxed integers (NLP formulation)
    pm = instantiate_mc_model(
        data, 
        LinDist3FlowPowerModel,
        build_mc_mld_switchable_relaxed;
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!]
    )
    ref = pm.ref[:it][:pmd][:nw][0]
    # Set optimizer
    set_optimizer(pm.model, optimizer)
    set_optimizer_attribute(pm.model, "print_level", 0)
    # Solve the relaxed problem
    optimize!(pm.model)

    # Extract results
    result = Dict{String, Any}()
    # result["termination_status"] = termination_status(pm.model)
    # result["objective"] = objective_value(pm.model)
    # result["solve_time"] = solve_time(pm.model)
    result["switch"] = soln["solution"]["switch"]

    # Extract switch states (relaxed values)
    result["switch_states"] = Dict{Int, Any}()
    for (i, var) in result["switch"]
        result["switch_states"][parse(Int,i)] = var["state"]
    end
    # Store the full power model for potential warm-starting
    return result,ref
end
# res,ref = solve_mld_relaxed(math; optimizer=ipopt)
# z_relaxed = res["switch_states"]
# pm_ivr_soln = solve_mc_pf(math, IVRUPowerModel, ipopt)

"""
    generate_bernoulli_samples(switch_states::Dict{Int, Float64}; 
                               n_samples=10, 
                               seed=nothing)

Generate sets of Bernoulli variables based on relaxed switch probabilities.
Each switch i with relaxed value p_i is sampled as Bernoulli(p_i).
Returns an array of sample dictionaries.
"""

function generate_bernoulli_samples(switch_states::Dict{Int64, Float64}, n_samples::Int64, seed::Int64)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    samples = Vector{Dict{Int, Float64}}()
    
    for sample_idx in 1:n_samples
        sample = Dict{Int, Float64}()
        for (switch_id, p) in switch_states
            # Generate Bernoulli random variable with probability p
            sample[switch_id] = Float64(rand(Bernoulli(p)))
        end
        push!(samples, sample)
    end
    
    return samples
end

"""
    radiality_check(ref_round::Dict{Symbol,Any}, z_relaxed::Dict{Int, Any},
                    bernoulli_samples::Vector{Dict{Int, Float64}};
                    optimizer=Ipopt.Optimizer)
"""

function radiality_check(ref_round::Dict{Symbol,Any}, z_relaxed::Dict{Int, Float64},bernoulli_samples::Vector{Dict{Int, Float64}}; optimizer=Gurobi.Optimizer)
    
    n_samples = length(bernoulli_samples)
    n_switches = length(z_relaxed)
    switch_ids = sort(collect(keys(z_relaxed)))
    
    println("\n[Optimization] Finding best single Bernoulli sample...")
    println("  Number of samples: $n_samples")
    println("  Number of switches: $n_switches")

    best_i = nothing
    best_dist = Inf
    best_block_status = nothing
    best_load_status = nothing

    for i in 1:n_samples
        model = JuMP.Model()
        set_optimizer(model, optimizer)
        set_attribute(model, "DualReductions", 0)
        @variable(model, switch_state[1:length(switch_ids)])
        @variable(model, z_block[1:length(ref_round[:block])], Bin)
        @variable(model, z_demand[1:length(ref_round[:load])])
        @variable(model, z_gen[1:length(ref_round[:gen])])
        @variable(model, z_storage[1:length(ref_round[:storage])])
        @variable(model, z_voltage[1:length(ref_round[:bus])])
        @variable(model, z_shunt[1:length(ref_round[:shunt])])
        for s in switch_ids
            @constraint(model, switch_state[s] == bernoulli_samples[i][s])
        end
        # radiality
        FairLoadDelivery.constraint_radial_topology_jump(model, ref_round, switch_state)
        FairLoadDelivery.constraint_mc_isolate_block_jump(model, ref_round)
        #FairLoadDelivery.constraint_mc_block_energization_consistency_bigm_jump(model, ref_round)

        # Must be disabled if there is no generation in the network
        FairLoadDelivery.constraint_block_budget_jump(model, ref_round)
        FairLoadDelivery.constraint_switch_budget_jump(model, ref_round)

        FairLoadDelivery.constraint_connect_block_load_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_gen_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_voltage_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_shunt_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_storage_jump(model, ref_round)

        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            d = sum((bernoulli_samples[i][s] - z_relaxed[s])^2 for s in switch_ids)
            if d < best_dist
                best_dist = d
                best_i = i
                best_block_status = value.(z_block)
                best_load_status = value.(z_demand)
            end
        end
    end
    block_ids = sort(collect(keys(ref_round[:block])))
    load_ids = sort(collect(keys(ref_round[:load])))
    return best_i, bernoulli_samples[best_i], block_ids, best_block_status, load_ids, best_load_status
end
# bernoulli_selection,switch_ids = radiality_check(reference, z_relaxed,
#                                       bernoulli_samples;
#                                       optimizer=ipopt)    
# if length(bernoulli_selection) == 0
#     error("No radial solutions found among Bernoulli samples.")
# else
#     println("Number of radial solutions from Bernoulli samples: $(length(bernoulli_selection))")
# end
# Test the acpf feasibility for the new set of switches that passed radiality
function ac_feasibility_test(math::Dict{String, Any}, set_id)
    pf_ivrup = PowerModelsDistribution.solve_mc_opf(math, IVRUPowerModel, ipopt)#solve_mc_pf_aw(math_round, ipopt)
    term_status = pf_ivrup["termination_status"]
    feas_dict = Dict{String, Any}()
    feas_dict["set_id"] = set_id
    if term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED
        println("AC Optimization converged to optimality for sample $set_id.")
        feas_dict["feas_status"] = true
        feas_dict["feas_obj"] = pf_ivrup["objective"]
    else
        println("AC Optimization did not converge to optimality for sample $set_id. Status: $term_status")
        feas_dict["feas_status"] = false
        feas_dict["feas_obj"] = nothing
        # println("Optimization did not converge to optimality for sample $set_id. Status: $term_status")
    end

    return set_id, feas_dict
end
# ac_bernoulli,ac_bernoulli_val = ac_feasibility_test(math, bernoulli_selection, switch_ids; optimizer=ipopt)
# if length(ac_bernoulli) == 0
#     error("No feasible AC solutions found among Bernoulli samples.")
# else
#     println("Number of feasible AC solutions from Bernoulli samples: $(length(ac_bernoulli))")
# end


# Find the set of switches with best mld objective value among feasible ac solutions
function find_best_switch_set(math::Dict{String, Any}, 
                          ac_bernoulli::Vector{Any},
                          switch_ids::Vector{Int};
                          optimizer=Ipopt.Optimizer)
    best_obj = -Inf
    best_sample_idx = 0
    best_switch_config = Dict{Int, Float64}()
    math_round = deepcopy(math)
    for (set_id, bernoulli_set) in enumerate(ac_bernoulli)
        for s in switch_ids
            math_round["switch"][string(s)]["state"] = value(bernoulli_set[s])
        end
        mld_rounded_soln = solve_mc_mld_shed_random_round(math_round, optimizer)
        obj_val = mld_rounded_soln["objective"]
        term_status = mld_rounded_soln["termination_status"]
        @info "Sample $set_id: Objective = $obj_val, Status = $term_status"
        if (term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED) && obj_val > best_obj
            best_obj = obj_val
            best_sample_idx = set_id
            for s in switch_ids
                best_switch_config[s] = value(bernoulli_set[s])
            end
        end
    end

    return best_sample_idx, best_switch_config
end
