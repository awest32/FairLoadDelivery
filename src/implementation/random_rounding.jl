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
function generate_bernoulli_samples(switch_states_bern::Dict{Int, Float64}, n_samples::Int, rng_seed::Int)
    local_rng = Random.MersenneTwister(rng_seed)
    samples = Vector{Dict{Int, Float64}}(undef, n_samples)

    for i in 1:n_samples
        sample = Dict{Int, Float64}()
        for (switch_id, p) in switch_states_bern
            p_clamped = clamp(p, 0.0, 1.0)
            # @info " The Bernoulli switch state for round $i is $p"
            # @info " The clamped Bernoulli switch state for round $i is $p_clamped"
            sample[switch_id] = rand(local_rng, Bernoulli(p_clamped))
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
    best_block_status = nothing
    best_load_status = nothing

    for i in 1:n_samples
        model = JuMP.Model()
        set_optimizer(model, optimizer)
        #set_attribute(model, "DualReductions", 0)
        @variable(model, z_switch[1:length(ref_round[:switch])].>= 0)
        @variable(model, z_block[1:length(block_ids)] .>= 0)
        @variable(model, z_demand[1:length(ref_round[:load])].>= 0)
        @variable(model, z_gen[1:length(ref_round[:gen])].>= 0)
        @variable(model, z_storage[1:length(ref_round[:storage])].>= 0)
        @variable(model, z_voltage[1:length(ref_round[:bus])].>= 0)
        @variable(model, z_shunt[1:length(ref_round[:shunt])].>= 0)
        
        for s in switch_ids
            @constraint(model, z_switch[s] == bernoulli_samples[i][s])
        end
        @constraint(model, z_block[1:length(block_ids)] .<= 1)
        @constraint(model, z_demand[1:length(ref_round[:load])] .<= 1)
        @constraint(model, z_gen[1:length(ref_round[:gen])] .<= 1)
        @constraint(model, z_storage[1:length(ref_round[:storage])] .<= 1)
        @constraint(model, z_voltage[1:length(ref_round[:bus])] .<= 1)
        @constraint(model, z_shunt[1:length(ref_round[:shunt])] .<= 1)

        # radiality
        FairLoadDelivery.constraint_radial_topology_jump(model, ref_round, z_switch)
        FairLoadDelivery.constraint_mc_isolate_block_jump(model, ref_round)
        for b in ref_round[:substation_blocks]
            @constraint(model, z_block[b] == 1)
        end
        #FairLoadDelivery.constraint_mc_block_energization_consistency_bigm_jump(model, ref_round)

        # Must be disabled if there is no generation in the network
        FairLoadDelivery.constraint_block_budget_jump(model, ref_round)
        FairLoadDelivery.constraint_switch_budget_jump(model, ref_round)

        FairLoadDelivery.constraint_connect_block_load_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_gen_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_voltage_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_shunt_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_storage_jump(model, ref_round)

        optimize!(model);
        #@info "Sample $i: Radiality check status: $(termination_status(model))"
        if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED || termination_status(model) == MOI.ALMOST_LOCALLY_SOLVED
            d = sum((bernoulli_samples[i][s] - zs_relaxed[s])^2 for s in switch_ids)
            if d < best_dist
                best_dist = d
                best_i = i
                best_block_status = value.(z_block)
                best_load_status = value.(z_demand)
            end
        end
        #   JuMP._CONSTRAINT_LIMIT_FOR_PRINTING[] = 1E9
        #   open("radiality_model.txt", "w") do io
        #     redirect_stdout(io) do
        #         print(model)
        #     end
        #end
    end
    block_ids = sort(collect(keys(ref_round[:block])))
    load_ids = sort(collect(keys(ref_round[:load])))

    if best_i === nothing
        @warn "[radiality_check] No feasible radial topology found in any of $n_samples Bernoulli samples"
        return nothing, Dict{Int,Float64}(), block_ids, zeros(length(block_ids)), load_ids, zeros(length(load_ids))
    else
        return best_i, bernoulli_samples[best_i], block_ids, best_block_status, load_ids, best_load_status
    end
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
    if term_status == MOI.OPTIMAL || term_status == MOI.LOCALLY_SOLVED || term_status == MOI.ALMOST_LOCALLY_SOLVED || term_status == MOI.ALMOST_OPTIMAL || term_status == MOI.ITERATION_LIMIT
        println("AC Optimization converged to optimality for sample $set_id.")
        feas_dict["feas_status"] = true
        feas_dict["feas_obj"] = pf_ivrup["objective"]
        feas_dict["solution"] = pf_ivrup["solution"]
    else
        println("AC Optimization did not converge to optimality for sample $set_id. Status: $term_status")
        feas_dict["feas_status"] = false
        feas_dict["feas_obj"] = pf_ivrup["objective"]
        feas_dict["solution"] = pf_ivrup["solution"]
        # println("Optimization did not converge to optimality for sample $set_id. Status: $term_status")
    end

    return feas_dict
end
"""
    extract_switch_block_states(solution::Dict{String,Any})

Extract switch states and block statuses from an MLD solution dictionary.
Returns `(switch_states, block_status)` as `Dict{Int64,Float64}` each.
"""
function extract_switch_block_states(solution::Dict{String,Any})
    switch_states = Dict{Int64,Float64}()
    for (s_id, s_data) in solution["switch"]
        switch_states[parse(Int, s_id)] = s_data["state"]
    end

    block_status = Dict{Int64,Float64}()
    for (b_id, b_data) in solution["block"]
        block_status[parse(Int, b_id)] = b_data["status"]
    end
    return switch_states, block_status
end

"""
    find_best_mld_solution(math_out::Vector{Dict{String, Any}}, solver)

Solve the rounded MLD problem for each candidate network in `math_out` and
return `(best_set, best_mld)` — the index and solution dictionary of the
topology with the highest objective value.
"""
function find_best_mld_solution(math_out::Vector{Dict{String, Any}}, solver)
    best_obj = -Inf
    best_set = 0
    best_mld = Dict{String, Any}()
    for (id, data) in enumerate(math_out)
        mld = solve_mc_mld_shed_random_round(data, solver)
        @info "Rounded solution from set $id has termination status: $(mld["termination_status"]) and objective value: $(mld["objective"])"
        if best_obj <= mld["objective"]
            best_obj = mld["objective"]
            best_set = id
            best_mld = mld
        end
    end
    return best_set, best_mld
end

"""
    set_source_gen_capacity!(math::Dict{String,Any}; ls_percent::Real=1000)

Set the source bus generation capacity to `ls_percent` times the total per-phase
demand. This simulates an effectively infinite slack bus for AC feasibility testing.
"""
function set_source_gen_capacity!(math::Dict{String,Any}; ls_percent::Real=1000)
    for (i, gen) in math["gen"]
        if gen["source_id"] == "voltage_source.source"
            pd_phase = zeros(3)
            qd_phase = zeros(3)
            for (_, d) in math["load"]
                for (idx, con) in enumerate(d["connections"])
                    if con in 1:3
                        pd_phase[con] += d["pd"][idx]
                        qd_phase[con] += d["qd"][idx]
                    end
                end
            end
            for phase in 1:3
                gen["pmax"][phase] = pd_phase[phase] * ls_percent
                gen["qmax"][phase] = qd_phase[phase] * ls_percent
            end
            gen["pmin"][:] .= 0
            gen["qmin"][:] .= 0
        end
    end
end

"""
    round_and_select_topology_mn(mn_data::Dict{String,Any};
        n_samples::Int=1000, n_rounds::Int=1, seed_base::Int=100,
        solver=ipopt) -> Dict{String, Any}

Per-period rounding and topology selection for multiperiod FALD.

For each period in `mn_data["nw"]`:
1. Solve relaxed MLD with the period's final weights
2. Generate Bernoulli samples of switch states
3. Find the best radial topology via radiality check
4. Apply the rounded switch configuration to the network
5. Set source generation capacity and run AC feasibility test
6. Select the best topology via rounded MLD objective

Returns a `Dict` keyed by `nw_id` with per-period results.
"""
function round_and_select_topology_mn(mn_data::Dict{String,Any};
        n_samples::Int=1000, n_rounds::Int=1, seed_base::Int=100,
        solver=ipopt)

    nw_ids = sort(collect(keys(mn_data["nw"])), by=x -> parse(Int, x))
    results = Dict{String, Any}()

    for nw_id in nw_ids
        println("\n" * "=" ^ 60)
        println("[round_and_select_topology_mn] Processing period $nw_id")
        println("=" ^ 60)

        period_math = deepcopy(mn_data["nw"][nw_id])

        # Step 1: Solve relaxed MLD with the period's final weights
        relaxed_soln = solve_mc_mld_shed_implicit_diff(period_math, solver;
            ref_extensions=[ref_add_rounded_load_blocks!])

        # Step 2: Extract switch/block states from relaxed solution
        switch_states, block_status = extract_switch_block_states(relaxed_soln["solution"])
        @info "[Period $nw_id] Relaxed switch states: $switch_states"
        @info "[Period $nw_id] Relaxed block status: $block_status"

        # Step 3: Instantiate model to get ref dict for radiality check
        pm = _PMD.instantiate_mc_model(
            period_math,
            _PMD.LinDist3FlowPowerModel,
            build_mc_mld_shedding_implicit_diff;
            ref_extensions=[ref_add_rounded_load_blocks!]
        )
        ref = pm.ref[:it][:pmd][:nw][0]

        # Step 4: Bernoulli sampling + radiality check for each rounding round
        bernoulli_switch_selection = Vector{Dict{Int, Float64}}()
        bernoulli_block_selection = Vector{Dict{Int, Float64}}()
        bernoulli_load_selection = Vector{Dict{Int, Float64}}()
        bernoulli_selection_index = []

        for r in 1:n_rounds
            rng_seed = seed_base * r
            samples = generate_bernoulli_samples(switch_states, n_samples, rng_seed)

            index, switch_states_radial, block_ids, block_status_radial, load_ids, load_status =
                radiality_check(ref, switch_states, block_status, samples)

            @info "[Period $nw_id, Round $r] Best radial sample index: $index"
            push!(bernoulli_selection_index, index)
            push!(bernoulli_switch_selection, switch_states_radial)
            push!(bernoulli_block_selection, Dict(zip(block_ids, block_status_radial)))
            push!(bernoulli_load_selection, Dict(zip(load_ids, load_status)))
        end

        # Step 5: Apply rounded switch configurations (switches + voltage source only)
        math_out = Vector{Dict{String, Any}}()
        for r in 1:n_rounds
            if bernoulli_selection_index[r] === nothing
                @warn "[Period $nw_id, Round $r] No feasible radial topology — skipping"
                continue
            end
            math_rounded = update_network(
                deepcopy(period_math),
                bernoulli_switch_selection[r],
                ref)
            push!(math_out, math_rounded)
        end

        if isempty(math_out)
            @warn "[Period $nw_id] No feasible radial topology found in any round"
            results[nw_id] = Dict(
                "best_math" => nothing,
                "best_mld" => nothing,
                "ac_feas" => Dict{String,Any}[],
                "total_load_shed" => NaN,
                "n_feasible_samples" => 0
            )
            continue
        end

        # Step 6: Solve rounded MLD for each topology, select best, and run AC feasibility
        best_obj = -Inf
        best_set = 0
        best_mld = Dict{String, Any}()
        mld_solutions = Vector{Dict{String, Any}}()
        for (id, data) in enumerate(math_out)
            mld = solve_mc_mld_shed_random_round(data, solver)
            push!(mld_solutions, mld)
            @info "[Period $nw_id] Rounded MLD set $id: status=$(mld["termination_status"]), obj=$(mld["objective"])"
            if best_obj <= mld["objective"]
                best_obj = mld["objective"]
                best_set = id
                best_mld = mld
            end
        end

        # Step 7: AC feasibility test — use ac_network_update with MLD solution
        # to properly de-energize loads/shunts/branches/buses on disconnected blocks
        ac_feas = Vector{Dict{String, Any}}()
        for (r, math_r) in enumerate(math_out)
            @info "[Period $nw_id] Testing AC feasibility for rounded solution $r"
            math_ac = ac_network_update(math_r, ref; mld_solution=mld_solutions[r])
            feas_dict = ac_feasibility_test(math_ac, r)
            push!(ac_feas, feas_dict)
        end

        total_load_shed = NaN
        if !isempty(best_mld) && haskey(best_mld, "solution") && haskey(best_mld["solution"], "load")
            total_load_shed = sum(
                sum(best_mld["solution"]["load"][string(i)]["pshed"])
                for i in 1:length(best_mld["solution"]["load"])
            )
        end

        @info "[Period $nw_id] Best MLD from set $best_set, objective=$(get(best_mld, "objective", NaN)), load_shed=$total_load_shed"

        results[nw_id] = Dict(
            "best_math" => best_set > 0 ? math_out[best_set] : nothing,
            "best_mld" => best_mld,
            "ac_feas" => ac_feas,
            "total_load_shed" => total_load_shed,
            "n_feasible_samples" => length(math_out),
            "relaxed_switch_states" => switch_states,
            "relaxed_block_status" => block_status
        )
    end

    return results
end

