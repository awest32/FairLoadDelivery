using FairLoadDelivery
using JuMP
using PowerModelsDistribution
using Random
using Distributions
"""
    solve_mld_relaxed(data::Dict{String, Any}; optimizer=Ipopt.Optimizer)

Solve the MLD problem with relaxed integer variables using parsed network data.
Returns the result dictionary with continuous switch variables.
"""
function solve_mld_relaxed(data::Dict{String, Any}; optimizer=Ipopt.Optimizer)
    soln = FairLoadDelivery.solve_mc_mld_switch(data, optimizer)
    # Build the MLD problem with relaxed integers (NLP formulation)
    pm = instantiate_mc_model(
        data, 
        LinDist3FlowPowerModel,
        build_mc_mld_switchable;
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

"""
    random_rounding_switches_blocks(math::Dict{String, Any}, 
                                   n_samples::Int; 
                                   seed=nothing)
Generate Bernoulli samples for switch and block variables based on their
relaxed values in the math dictionary.
"""
function random_rounding_switches_blocks(math::Dict{String, Any}, 
                                        n_samples::Int,
                                        seed=nothing)
    # Solve the relaxed MLD problem to get switch and block probabilities using the new weights
    relaxed_soln = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math, Ipopt.Optimizer)
    res = relaxed_soln["solution"]
    # Extract switch states
    switch_states = Dict{Int, Any}()
       for (s_id, s_data) in enumerate(res["switch"])
        switch_states[s_id] = s_data[2]["state"]
    end
    # # Extract block states
    # block_states = Dict{Int, Any}()
    # for (b_id, b_data) in enumerate(res["block"])
    #     block_states[b_id] = b_data[2]["status"]
    # end

    # Generate Bernoulli samples for switches
    switch_samples = generate_bernoulli_samples(switch_states; 
                                               n_samples=n_samples, 
                                               seed=seed)
    # Generate Bernoulli samples for blocks
    # block_samples = generate_bernoulli_samples(block_states; 
    #                                           n_samples=n_samples, 
    #                                           seed=seed)

    # Find the best switch sample by minimizing the distane to the relaxed values
    pm = instantiate_mc_model(
        math, 
        LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_load_blocks!]
    )
    ref = pm.ref[:it][:pmd][:nw][0]
    bernoulli_selection_exp, switch_ids = radiality_check(ref, switch_states, switch_samples)

    return bernoulli_selection_exp, switch_ids
end

"""
    generate_bernoulli_samples(switch_states::Dict{Int, Float64}; 
                               n_samples=10, 
                               seed=nothing)

Generate sets of Bernoulli variables based on relaxed switch probabilities.
Each switch i with relaxed value p_i is sampled as Bernoulli(p_i).
Returns an array of sample dictionaries.
"""
function generate_bernoulli_samples(switch_states::Dict{Int, Float64},n_samples::Int, rng::Int)
    samples = Vector{Dict{Int, Float64}}(undef, n_samples)

    for i in 1:n_samples
        sample = Dict{Int, Float64}()
        for (switch_id, p) in switch_states
            p_clamped = clamp(p, 0.0, 1.0)
            sample[switch_id] = rand(Bernoulli(p_clamped))
        end
        samples[i] = sample
    end

    return samples
end

"""
    radiality_test(math::Dict{String, Any}, 
                    bernoulli_selection_exp::Any;
                    optimizer=Ipopt.Optimizer)

Test radiality of the network for each set of switch configurations provided in bernoulli_selection_exp.
Returns a vector of feasible switch configurations that maintain radiality.
"""
function radiality_check(ref_round::Dict{Symbol,Any}, zs_relaxed::Dict{Int, Float64}, zb_relaxed::Dict{Int, Float64}, bernoulli_samples::Vector{Dict{Int, Float64}}; optimizer=Gurobi.Optimizer)
    
    n_samples = length(bernoulli_samples)
    n_switches = length(zs_relaxed)
    switch_ids = sort(collect(keys(zs_relaxed)))
    n_blocks = length(zb_relaxed)
    block_ids = sort(collect(keys(zb_relaxed)))
    
    println("\n[Optimization] Finding best single Bernoulli sample...")
    println("  Number of samples: $n_samples")
    println("  Number of switches: $n_switches")
    println("  Number of blocks: $n_blocks")


    best_i = nothing
    best_dist = Inf
    best_switch_status = nothing
    best_load_status = nothing

    for i in 1:n_samples
        model = JuMP.Model()
        set_optimizer(model, optimizer)
        set_attribute(model, "DualReductions", 0)
        @variable(model, z_block[1:length(block_ids)])
        @variable(model, switch_state[1:length(ref_round[:switch])], Bin)
        @variable(model, z_demand[1:length(ref_round[:load])])
        @variable(model, z_gen[1:length(ref_round[:gen])])
        @variable(model, z_storage[1:length(ref_round[:storage])])
        @variable(model, z_voltage[1:length(ref_round[:bus])])
        @variable(model, z_shunt[1:length(ref_round[:shunt])])
        for b in block_ids
            @constraint(model, z_block[b] == bernoulli_samples[i][b])
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

        # Constraint to ensure generation matches the load served
        # @constraint(model, sum(ref_round[g]["pg"][t]*z_gen[g] for (g, conns) in ref_round[:bus_conns_gen] for t in conns)
        #     >= sum(ref_round[s]["ps"][t]*z_storage[s] for (s, conns) in ref_round[:bus_conns_storage] for t in conns)
        #     - sum(ref_round[l]["pd"][t]*z_demand[l] for (l, conns) in ref_round[:bus_conns_load] for t in conns)
        #     ) 
        print(model)
        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            d = sum((bernoulli_samples[i][b] - zb_relaxed[b])^2 for b in block_ids)
            if d < best_dist
                best_dist = d
                best_i = i
                best_switch_status = value.(switch_state)
                best_load_status = value.(z_demand)
            end
        end
    end
    switch_ids = sort(collect(keys(ref_round[:switch])))
    load_ids = sort(collect(keys(ref_round[:load])))
    return best_i, bernoulli_samples[best_i], switch_ids, best_switch_status, load_ids, best_load_status
end

"""
    ac_feasibility_test(math::Dict{String, Any}, 
                        bernoulli_selection_exp::Any,
                        switch_ids::Vector{Int};
                        optimizer=Ipopt.Optimizer)
"""
# Test the acpf feasibility for the new set of switches that passed radiality
function ac_feasibility_test(math::Dict{String, Any}, set_id)
    pf_ivrup = PowerModelsDistribution.solve_mc_pf(math, IVRUPowerModel, ipopt)#solve_mc_pf_aw(math_round, ipopt)
    term_status = pf_ivrup["termination_status"]
    feas_dict = Dict{String, Any}()
    feas_dict["set_id"] = set_id
    if term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED || term_status == MOI.ITERATION_LIMIT
        println("AC Optimization converged to optimality for sample $set_id.")
        feas_dict["feas_status"] = true
        feas_dict["feas_obj"] = pf_ivrup["objective"]
    else
        println("AC Optimization did not converge to optimality for sample $set_id. Status: $term_status")
        feas_dict["feas_status"] = false
        feas_dict["feas_obj"] = nothing
        # println("Optimization did not converge to optimality for sample $set_id. Status: $term_status")
    end

    return feas_dict
end
# # Find the set of switches with best mld objective value among feasible ac solutions
# function find_best_switch_set(math::Dict{String, Any}, 
#                           ac_bernoulli::Vector{Any},
#                           switch_ids::Vector{Int};
#                           optimizer=Ipopt.Optimizer)
#     best_obj = -Inf
#     best_sample_idx = 0
#     best_switch_config = Dict{Int, Float64}()
#     math_round = deepcopy(math)
#     for (set_id, bernoulli_set) in enumerate(ac_bernoulli)
#         for s in switch_ids
#             math_round["switch"][string(s)]["state"] = value(bernoulli_set[s])
#         end
#         mld_rounded_soln = solve_mc_mld_shed_random_round(math_round, optimizer)
#         obj_val = mld_rounded_soln["objective"]
#         term_status = mld_rounded_soln["termination_status"]
#         @info "Sample $set_id: Objective = $obj_val, Status = $term_status"
#         if (term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED) && obj_val > best_obj
#             best_obj = obj_val
#             best_sample_idx = set_id
#             for s in switch_ids
#                 best_switch_config[s] = value(bernoulli_set[s])
#             end
#         end
#     end

#     return best_sample_idx, best_switch_config
# end

