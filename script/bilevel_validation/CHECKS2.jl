using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS, Juniper
using HSL_jll
using Random
using Distributions
using DiffOpt
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra, SparseArrays
using DataFrames
using CSV
using Dates

# Load validation utilities
include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/random_rounding.jl")

# ============================================================
# STEP 4: RANDOM ROUNDING
# ============================================================
print_validation_header("Step 4: Random Rounding")

rounding_checks = Dict{String, Any}()

# Build the implicit diff model to get ref
imp_diff_model = instantiate_mc_model(
    math_new,
    LinDist3FlowPowerModel,
    build_mc_mld_shedding_implicit_diff;
    ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
)
ref = imp_diff_model.ref[:it][:pmd][:nw][0]

# Extract switch and block states from relaxed solution
switch_states = Dict{Int64, Float64}()
for (s_id, s_data) in mld_relaxed_final["solution"]["switch"]
    switch_states[parse(Int, s_id)] = s_data["state"]
end

block_status = Dict{Int64, Float64}()
for (b_id, b_data) in mld_relaxed_final["solution"]["block"]
    block_status[parse(Int, b_id)] = b_data["status"]
end

println("\n  Relaxed switch states:")
for (s_id, state) in sort(collect(switch_states))
    println("    Switch $s_id: state = $(round(state, digits=4))")
end
println("  Relaxed block status:")
for (b_id, status) in sort(collect(block_status))
    println("    Block $b_id: status = $(round(status, digits=4))")
end


# Storage for results from each round
bernoulli_switch_selection_exp = Vector{Dict{Int, Float64}}()
bernoulli_block_selection_exp = Vector{Dict{Int, Float64}}()
bernoulli_load_selection_exp = Vector{Dict{Int, Float64}}() 
bernoulli_selection_index = []
bernoulli_samples = Dict{Int, Vector{Dict{Int, Float64}}}()

# Run multiple rounds of bernoulli rounding with different RNG seeds
for r in 1:N_ROUNDS
    rng = 100 * r
    # Generate bernoulli samples for switches and blocks
    bernoulli_samples[r] = generate_bernoulli_samples(switch_states, N_BERNOULLI_SAMPLES, rng)

    # Find the best bernoulli sample that is topology feasible and closest to the relaxed solution
    index, switch_states_radial, block_ids, block_status_radial, load_ids, load_status = radiality_check(ref, switch_states, block_status, bernoulli_samples[r])

    if index === nothing
        @warn "[$CASE/$FAIR_FUNC] Round $r failed at: RADIAL FEASIBILITY — no Bernoulli sample produced a feasible radial topology"
    end

    println("  Round $r: Best radial sample index: $index")
    println("  Round $r: Best radial sample switch status: $switch_states_radial")
    println("  Round $r: Best radial sample block status: $block_status_radial")
    println("  Round $r: Best radial sample load status: $load_status")

    push!(bernoulli_selection_index, index)
    push!(bernoulli_switch_selection_exp, switch_states_radial)
    push!(bernoulli_block_selection_exp, Dict(zip(block_ids, block_status_radial)))
    push!(bernoulli_load_selection_exp, Dict(zip(load_ids, load_status)))
end

# Check if any round found a radial topology
passed_radial = any(idx !== nothing for idx in bernoulli_selection_index)
rounding_checks["radiality_found"] = Dict("passed" => passed_radial, "details" => ["Sample indices by round: $bernoulli_selection_index"])
print_check_result("Radial topology found", passed_radial)


# Create copies of math dictionary for each round and apply rounded states
math_random_test = Vector{Dict{String, Any}}()
for r in 1:N_ROUNDS
    math_copy = deepcopy(math_new)
    push!(math_random_test, math_copy)
end

math_out = Vector{Union{Dict{String, Any}, Nothing}}(nothing, N_ROUNDS)
for r in 1:N_ROUNDS
    if bernoulli_selection_index[r] === nothing
        println("  Skipping round $r (no feasible radial topology)")
        continue
    end

    # Get block_ids and load_ids from the stored dictionaries
    block_ids_r = collect(keys(bernoulli_block_selection_exp[r]))
    load_ids_r = collect(keys(bernoulli_load_selection_exp[r]))

    math_out[r] = update_network(
        math_random_test[r],
        bernoulli_block_selection_exp[r],
        bernoulli_load_selection_exp[r],
        bernoulli_switch_selection_exp[r],
        ref, r
    )
end
if math_out == [nothing for _ in 1:N_ROUNDS]
    @error "[$CASE/$FAIR_FUNC] FAILED — no feasible radial topology found across all $N_ROUNDS rounds. Check warnings above for failure stage RADIAL FEASIBILITY."
    error("No feasible solution — cannot proceed to rounding checks")
end

mld_rounded_r = FairLoadDelivery.solve_mc_mld_shed_random_round(math_out[1], gurobi_solver)
