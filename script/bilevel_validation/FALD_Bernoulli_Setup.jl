
using FairLoadDelivery
using Ipopt, Gurobi, HiGHS, Juniper


ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

function extract_switch_block_states(relaxed_soln::Dict{String,Any})
    switch_states = Dict{Int64,Float64}()
    for (s_id, s_data) in relaxed_soln["switch"]
        switch_states[parse(Int, s_id)] = s_data["state"]
    end

    block_status = Dict{Int64,Float64}()
    for (b_id, b_data) in relaxed_soln["block"]
        block_status[parse(Int, b_id)] = b_data["status"]
    end
    return switch_states, block_status
end

imp_model = instantiate_mc_model(
    math_relaxed,
    LinDist3FlowPowerModel,
    build_mc_mld_shedding_implicit_diff;
    ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
)
ref = imp_model.ref[:it][:pmd][:nw][0]

switch_states, block_status = extract_switch_block_states(mld_relaxed_final["solution"])

# Storage for each round
math_radial = Vector{Dict{String, Any}}()
math_out = Vector{Dict{String, Any}}()
mld_results = Vector{Dict{String, Any}}()
math_out_ac = Vector{Dict{String, Any}}()
ac_results = Vector{Dict{String, Any}}()
round_tested = Int[]
stage = "random_rounding"

function generate_bernoulli_samples(switch_states_bern::Dict{Int, Float64}, n_samples::Int, rng_seed::Int)
    local_rng = Random.MersenneTwister(rng_seed)
    samples = Vector{Dict{Int, Float64}}(undef, n_samples)

    for i in 1:n_samples
        sample = Dict{Int, Float64}()
        for (switch_id, p) in switch_states_bern
            p_clamped = clamp(p, 0.0, 1.0)
            sample[switch_id] = rand(local_rng, Bernoulli(p_clamped))
        end
        samples[i] = sample
    end

    return samples
end



for r in 1:N_ROUNDS
    rng = 100 * r
    bernoulli_samples = generate_bernoulli_samples(switch_states, N_BERNOULLI_SAMPLES, rng);

    index, switch_states_radial, block_ids, block_status_radial, load_ids, load_status =
        radiality_check(ref, switch_states, block_status, bernoulli_samples)

    if index === nothing
        @warn "Round $r failed at: RADIAL FEASIBILITY — no Bernoulli sample produced a feasible radial topology" 
    end

    # Apply rounded states
    math_rounded = update_network(math_relaxed, switch_states_radial, ref);
    push!(math_radial, math_rounded)

    # Solve rounded MLD
    if !isempty(math_rounded)
        mld_rounded = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(math_rounded, gurobi);
        if mld_rounded["termination_status"] in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
            push!(math_out, math_rounded)
            push!(mld_results, mld_rounded)
        else
            @warn "Round $r failed at: ROUNDED MLD SOLVE — termination status: $(mld_rounded["termination_status"])"
        end
    else
        @warn "Round $r failed: no valid radial topology found, skipping rounded MLD solve and ACPF."
    end

    # Updated network data dictionary for AC power flow
    if !isempty(math_out)
        math_ac = ac_network_update(math_out[r], ref)
         # Solve AC power flow
        pf_ac = solve_mc_pf(math_ac, IVRUPowerModel, ipopt);
        if pf_ac["termination_status"] in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
            @info "Round $r succeeded with AC power flow solve after random rounding."
            push!(math_out_ac, math_ac)
            push!(ac_results, pf_ac)
            push!(ac_tested, r)
        else
            @warn "Round $r failed at: AC POWER FLOW SOLVE — termination status: $(pf_ac["termination_status"])"
        end
    else
        @warn "No valid rounded MLD solution found in round $r, skipping AC power flow solve."
    end
end

if  isempty(math_radial)
    @error "All rounds failed to produce a valid radial topology. No results to display."
elseif isempty(math_out)
    @error "All rounds failed to produce a valid rounded MLD solution. No results to display."
elseif isempty(math_out_ac)
    @error "Rounded MLD solutions were found, but all failed AC power flow validation. No results to display."
end