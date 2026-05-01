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
function radiality_check(ref_round::Dict{Symbol,Any}, zs_relaxed::Dict{Int, Float64}, zb_relaxed::Dict{Int, Float64}, bernoulli_samples::Vector{Dict{Int, Float64}}; optimizer=Gurobi.Optimizer, top_k::Int=5)

    n_samples = length(bernoulli_samples)
    n_switches = length(zs_relaxed)
    switch_ids = sort(collect(keys(zs_relaxed)))
    n_blocks = length(zb_relaxed)
    block_ids_init = sort(collect(keys(zb_relaxed)))

    println("\n[Optimization] Finding top-$top_k radial Bernoulli samples...")
    println("  Number of samples: $n_samples")
    println("  Number of switches: $n_switches")
    println("  Number of blocks: $n_blocks")

    # Collect all feasible samples with their distance to relaxed solution
    feasible_candidates = []  # (distance, index, block_status, load_status)

    for i in 1:n_samples
        model = JuMP.Model()
        set_optimizer(model, optimizer)
        @variable(model, z_switch[1:length(ref_round[:switch])].>= 0)
        @variable(model, z_block[1:length(block_ids_init)] .>= 0)
        @variable(model, z_demand[1:length(ref_round[:load])].>= 0)
        @variable(model, z_gen[1:length(ref_round[:gen])].>= 0)
        @variable(model, z_storage[1:length(ref_round[:storage])].>= 0)
        @variable(model, z_voltage[1:length(ref_round[:bus])].>= 0)
        @variable(model, z_shunt[1:length(ref_round[:shunt])].>= 0)

        for s in switch_ids
            @constraint(model, z_switch[s] == bernoulli_samples[i][s])
        end
        @constraint(model, z_block[1:length(block_ids_init)] .<= 1)
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

        FairLoadDelivery.constraint_block_budget_jump(model, ref_round)
        FairLoadDelivery.constraint_switch_budget_jump(model, ref_round)

        FairLoadDelivery.constraint_connect_block_load_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_gen_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_voltage_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_shunt_jump(model, ref_round)
        FairLoadDelivery.constraint_connect_block_storage_jump(model, ref_round)

        optimize!(model);
        if termination_status(model) == MOI.OPTIMAL || termination_status(model) == MOI.LOCALLY_SOLVED || termination_status(model) == MOI.ALMOST_LOCALLY_SOLVED
            d = sum((bernoulli_samples[i][s]^2 + zs_relaxed[s]^2)^2 for s in switch_ids)
            push!(feasible_candidates, (dist=d, index=i, block_status=value.(z_block), load_status=value.(z_demand)))
        end
    end

    block_ids = sort(collect(keys(ref_round[:block])))
    load_ids = sort(collect(keys(ref_round[:load])))

    if isempty(feasible_candidates)
        @warn "[radiality_check] No feasible radial topology found in any of $n_samples Bernoulli samples"
        return nothing, Dict{Int,Float64}(), block_ids, zeros(length(block_ids)), load_ids, zeros(length(load_ids))
    end

    # Sort by distance and return top-k (or best if top_k=1 for backward compat)
    sort!(feasible_candidates, by=c -> c.dist)
    n_return = min(top_k, length(feasible_candidates))
    println("  Found $(length(feasible_candidates)) feasible samples, returning top $n_return")

    best = feasible_candidates[1]
    return best.index, bernoulli_samples[best.index], block_ids, best.block_status, load_ids, best.load_status, feasible_candidates[1:n_return]
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
        if best_obj >= mld["objective"]
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
        n_samples::Int=1000, n_rounds::Int=1, n_top_k::Int=5, seed_base::Int=100,
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

        # ================================================================
        # Filter pipeline: N_bern → R → MLD_INT → AC_FEAS
        # Each stage filters to feasible solutions from the previous stage
        # ================================================================
        n_bern = n_samples

        # Stage 1: Bernoulli sampling → Radiality check (N_bern → R)
        rng_seed = seed_base
        samples = generate_bernoulli_samples(switch_states, n_bern, rng_seed)

        _, _, block_ids, _, load_ids, _, radial_candidates =
            radiality_check(ref, switch_states, block_status, samples; top_k=n_bern)

        n_radial = length(radial_candidates)
        @info "[Period $nw_id] Filter: N_bern=$n_bern → R=$n_radial"

        if n_radial == 0
            error("[Period $nw_id] Filter failed: N_bern=$n_bern → R=0 (no radial-feasible topology)")
        end

        # Stage 2: Radiality → Integer MLD check (R → MLD_INT)
        # Use top-K radial candidates (sorted by distance to relaxed)
        n_mld_candidates = min(n_top_k, n_radial)
        math_out = Vector{Dict{String, Any}}()
        mld_solutions = Vector{Dict{String, Any}}()
        mld_feasible_idx = Int[]
        feasible_statuses = [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]

        for (k, cand) in enumerate(radial_candidates[1:n_mld_candidates])
            math_rounded = update_network(deepcopy(period_math), samples[cand.index], ref)
            push!(math_out, math_rounded)

            mld = solve_mc_mld_shed_random_round(math_rounded, solver)
            push!(mld_solutions, mld)
            @info "[Period $nw_id] MLD candidate $k (sample $(cand.index)): status=$(mld["termination_status"]), obj=$(mld["objective"])"

            if mld["termination_status"] in feasible_statuses
                push!(mld_feasible_idx, k)
            end
        end

        n_mld_feas = length(mld_feasible_idx)
        @info "[Period $nw_id] Filter: R=$n_radial → MLD_INT=$n_mld_feas (of $n_mld_candidates tested)"

        if n_mld_feas == 0
            error("[Period $nw_id] Filter failed: N_bern=$n_bern → R=$n_radial → MLD_INT=0")
        end

        # Stage 3: Integer MLD → AC feasibility check (MLD_INT → AC_FEAS)
        ac_feas = Vector{Dict{String, Any}}()
        ac_feasible_idx = Int[]

        for k in mld_feasible_idx
            @info "[Period $nw_id] AC feasibility test for MLD candidate $k"
            math_ac = ac_network_update(math_out[k], ref; mld_solution=mld_solutions[k])
            feas_dict = ac_feasibility_test(math_ac, k)
            push!(ac_feas, feas_dict)

            if feas_dict["feas_status"]
                push!(ac_feasible_idx, k)
            end
        end

        n_ac_feas = length(ac_feasible_idx)
        @info "[Period $nw_id] Filter: N_bern=$n_bern → R=$n_radial → MLD_INT=$n_mld_feas → AC_FEAS=$n_ac_feas"

        if n_ac_feas == 0
            # Diagnostic: dump topology of all MLD-feasible candidates that failed AC
            for k in mld_feasible_idx
                sol = mld_solutions[k]["solution"]
                sw_str = join(["s$(s)=$(round(sol["switch"][s]["state"], digits=2))" for s in sort(collect(keys(sol["switch"])))], ", ")
                blk_str = join(["b$(b)=$(round(sol["block"][b]["status"], digits=2))" for b in sort(collect(keys(sol["block"])))], ", ")
                load_shed = sum(sum(sol["load"][l]["pshed"]) for l in keys(sol["load"]))
                load_served = sum(sum(sol["load"][l]["pd"]) for l in keys(sol["load"]))
                @warn "[Period $nw_id] AC-failed candidate $k: switches=[$sw_str], blocks=[$blk_str], shed=$(round(load_shed, digits=1)), served=$(round(load_served, digits=1))"
            end
            error("[Period $nw_id] Filter failed: N_bern=$n_bern → R=$n_radial → MLD_INT=$n_mld_feas → AC_FEAS=0")
        end

        # Select best AC-feasible MLD (highest objective)
        best_obj = -Inf
        best_set = 0
        best_mld = Dict{String, Any}()
        for k in ac_feasible_idx
            if mld_solutions[k]["objective"] > best_obj
                best_obj = mld_solutions[k]["objective"]
                best_set = k
                best_mld = mld_solutions[k]
            end
        end

        total_load_shed = NaN
        if !isempty(best_mld) && haskey(best_mld, "solution") && haskey(best_mld["solution"], "load")
            total_load_shed = sum(
                sum(best_mld["solution"]["load"][string(i)]["pshed"])
                for i in 1:length(best_mld["solution"]["load"])
            )
        end

        @info "[Period $nw_id] Best AC-feasible MLD from candidate $best_set, objective=$(get(best_mld, "objective", NaN)), load_shed=$total_load_shed"

        results[nw_id] = Dict(
            "best_math" => best_set > 0 ? math_out[best_set] : nothing,
            "best_mld" => best_mld,
            "ac_feas" => ac_feas,
            "total_load_shed" => total_load_shed,
            "n_feasible_samples" => n_ac_feas,
            "n_radial" => n_radial,
            "n_mld_feasible" => n_mld_feas,
            "n_ac_feasible" => n_ac_feas,
            "relaxed_switch_states" => switch_states,
            "relaxed_block_status" => block_status
        )
    end

    return results
end

