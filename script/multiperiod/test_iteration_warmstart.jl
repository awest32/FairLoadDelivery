"""
    test_iteration_warmstart.jl

    Test whether warm-starting a 4-iteration bilevel run from the output of
    a 2-iteration run yields better (lower) relaxed load shed than the
    2-iteration run alone.

    Efficiency fair func only. Relaxed bilevel only (no rounding).

    Phase A: run bilevel with ITERATIONS=2 on fresh mn_data
    Phase B: run bilevel with ITERATIONS=4 starting from Phase A's mn_relaxed
             (weights already baked into each nw's load["weight"])
    Compare: total pshed (lower- and upper-level) at last iteration of each.
"""

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi, HiGHS
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra, SparseArrays
using Statistics
using DiffOpt

const PMD = PowerModelsDistribution

include("../bilevel_validation/validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

const CASE = "motivation_c"
const FAIR_FUNC = "efficiency"
const LS_PERCENT = 0.8
const N_PERIODS = 3
const SOURCE_PU = 1.03
const critical_buses = []

# Same load and cost profiles as compare_fair_funcs_multiperiod
const LOAD_SCALE_FACTORS = [round(0.8 + 1.0 * exp(-((t - N_PERIODS/2)^2) / (2 * 4^2)), digits=3) for t in 0:N_PERIODS-1]
const PEAK_TIME_COSTS    = [round(8 + 22 * exp(-((t - N_PERIODS/2)^2) / (2 * 3^2)), digits=2) for t in 0:N_PERIODS-1]

function create_multinetwork_data(base_math::Dict{String,Any}, n_periods::Int, load_scales::Vector{Float64})
    @assert length(load_scales) == n_periods
    mn_data = Dict{String,Any}(
        "multinetwork" => true,
        "per_unit" => true,
        "data_model" => PMD.MATHEMATICAL,
        "nw" => Dict{String,Any}()
    )
    for key in ["baseMVA", "basekv", "bus_lookup", "settings"]
        if haskey(base_math, key)
            mn_data[key] = deepcopy(base_math[key])
        end
    end
    for t in 1:n_periods
        nw_id = string(t - 1)
        nw_data = deepcopy(base_math)
        delete!(nw_data, "multinetwork")
        scale = load_scales[t]
        for (_, load) in nw_data["load"]
            load["pd"] = load["pd"] .* scale
            load["qd"] = load["qd"] .* scale
        end
        nw_data["time_period"] = t
        nw_data["load_scale"] = scale
        mn_data["nw"][nw_id] = nw_data
    end
    return mn_data
end

# Minimal run_bilevel_relaxed_mn copy from compare_fair_funcs_multiperiod.jl,
# restricted to fair_func="efficiency".
function run_bilevel_relaxed_mn(mn_data::Dict{String,Any}, iterations::Int, fair_weights_init::Vector{Float64},
                                critical_id::Vector{Int}=Int[];
                                peak_time_costs::Vector{Float64}=Float64[])
    mn_new = deepcopy(mn_data)
    fair_weights = copy(fair_weights_init)
    nw_ids = sort(collect(keys(mn_new["nw"])), by=x->parse(Int, x))
    T = length(nw_ids)

    pshed_lower_level = Float64[]
    pshed_upper_level = Float64[]
    final_weight_ids = Int[]
    final_weights = Float64[]
    completed_iterations = 0
    last_status = MOI.OPTIMIZE_NOT_CALLED

    for k in 1:iterations
        dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs = lower_level_soln_mn(mn_new, fair_weights, k)
        n_loads = length(weight_ids)

        pshed_new, fair_weight_vals, status = complete_efficiency_load_shed(
            dpshed, pshed_val, weight_vals, critical_id, weight_ids;
            peak_time_costs=peak_time_costs, n_loads=n_loads)

        last_status = status
        @info "Iteration $k: upper-level status = $status"
        if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.TIME_LIMIT]
            @warn "Iteration $k/$iterations: upper-level infeasible (status=$status) — stopping"
            break
        end
        completed_iterations = k

        for (t, nw_id) in enumerate(nw_ids)
            nw_data = mn_new["nw"][nw_id]
            offset = (t - 1) * n_loads
            for (j, lid) in enumerate(weight_ids)
                nw_data["load"][string(lid)]["weight"] = fair_weight_vals[offset + j]
            end
        end

        fair_weights = copy(fair_weight_vals)
        final_weight_ids = weight_ids
        final_weights = fair_weight_vals

        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))

        println("  iter $k: pshed_lower=$(round(sum(pshed_val), digits=6)), pshed_upper=$(round(sum(pshed_new), digits=6))")
    end

    summary = Dict(
        "completed_iterations" => completed_iterations,
        "last_status" => string(last_status),
    )
    return mn_new, pshed_lower_level, pshed_upper_level, final_weight_ids, final_weights, summary
end

function get_total_demand_mn(mn_data::Dict{String,Any})
    total = 0.0
    for (_, nw_data) in mn_data["nw"]
        for (_, load) in nw_data["load"]
            total += sum(load["pd"])
        end
    end
    return total
end

# ============================================================
# RUN
# ============================================================
println("=" ^ 70)
println("ITERATION WARM-START TEST  (fair_func = $FAIR_FUNC)")
println("=" ^ 70)
println("Load scale factors: $LOAD_SCALE_FACTORS")
println("Peak-time costs:    $PEAK_TIME_COSTS")

eng, math, lbs, critical_id = FairLoadDelivery.setup_network("ieee_13_aw_edit/$CASE.dss", LS_PERCENT, SOURCE_PU, critical_buses)
sorted_load_ids = sort(parse.(Int, collect(keys(math["load"]))))
fair_weights_init = Float64[math["load"][string(i)]["weight"] for i in sorted_load_ids]

mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)
total_demand = get_total_demand_mn(mn_data)
println("Total demand across periods: $(round(total_demand, digits=4))")

# ---------- Phase A: 2 iterations ----------
println("\n" * "-" ^ 70)
println("PHASE A: 2 iterations (cold start)")
println("-" ^ 70)
t0 = time()
mn_A, lower_A, upper_A, wids_A, wvals_A, sum_A =
    run_bilevel_relaxed_mn(mn_data, 2, fair_weights_init, critical_id; peak_time_costs=PEAK_TIME_COSTS)
t_A = time() - t0
println("Phase A took $(round(t_A, digits=1))s, completed $(sum_A["completed_iterations"]) iterations")

# ---------- Phase B: 4 iterations, warm-started from Phase A ----------
println("\n" * "-" ^ 70)
println("PHASE B: 4 iterations (warm-started from Phase A's final mn_data)")
println("-" ^ 70)
# Weights are already embedded in mn_A per-period. The lower-level reads
# them from the math dict on iter 1, so fair_weights_init value is ignored
# but we pass the final per-period weights for symmetry.
t0 = time()
mn_B, lower_B, upper_B, wids_B, wvals_B, sum_B =
    run_bilevel_relaxed_mn(mn_A, 4, wvals_A, critical_id; peak_time_costs=PEAK_TIME_COSTS)
t_B = time() - t0
println("Phase B took $(round(t_B, digits=1))s, completed $(sum_B["completed_iterations"]) iterations")

# ---------- Compare ----------
println("\n" * "=" ^ 70)
println("COMPARISON")
println("=" ^ 70)

final_lower_A = isempty(lower_A) ? NaN : lower_A[end]
final_upper_A = isempty(upper_A) ? NaN : upper_A[end]
final_lower_B = isempty(lower_B) ? NaN : lower_B[end]
final_upper_B = isempty(upper_B) ? NaN : upper_B[end]

println("Phase A (2 iters cold start):")
println("  pshed_lower (final iter):  $(round(final_lower_A, digits=6))   ($(round(100*final_lower_A/total_demand, digits=3))% of demand)")
println("  pshed_upper (final iter):  $(round(final_upper_A, digits=6))   ($(round(100*final_upper_A/total_demand, digits=3))% of demand)")
println("  lower trajectory: $(round.(lower_A, digits=6))")
println("  upper trajectory: $(round.(upper_A, digits=6))")

println("\nPhase B (4 iters warm start from A):")
println("  pshed_lower (final iter):  $(round(final_lower_B, digits=6))   ($(round(100*final_lower_B/total_demand, digits=3))% of demand)")
println("  pshed_upper (final iter):  $(round(final_upper_B, digits=6))   ($(round(100*final_upper_B/total_demand, digits=3))% of demand)")
println("  lower trajectory: $(round.(lower_B, digits=6))")
println("  upper trajectory: $(round.(upper_B, digits=6))")

Δlower = final_lower_B - final_lower_A
Δupper = final_upper_B - final_upper_A
println("\nΔ lower (B - A): $(round(Δlower, digits=6))   ($(Δlower < 0 ? "B BETTER (less shed)" : Δlower > 0 ? "A BETTER" : "same"))")
println("Δ upper (B - A): $(round(Δupper, digits=6))   ($(Δupper < 0 ? "B BETTER (less shed)" : Δupper > 0 ? "A BETTER" : "same"))")
