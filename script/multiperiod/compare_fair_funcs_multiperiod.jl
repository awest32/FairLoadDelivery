"""
    compare_fair_funcs_multiperiod.jl

    Multiperiod version of compare_fair_funcs_and_networks.jl.
    Compares bilevel FALD across fairness functions with T time periods.

    Key difference from single-period:
    - ONE global set of fair_load_weights shared across all T periods
    - Jacobian is (T*N) x N: sensitivities of all pshed (across periods) w.r.t. global weights
    - Upper-level fairness functions operate on the flattened (T*N) pshed vector
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
using DataFrames
using CSV
using Dates
using Plots
using Statistics
using DiffOpt
using JLD2

const PMD = PowerModelsDistribution

include("../bilevel_validation/validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASES = ["motivation_c"]
const FAIR_FUNCS = ["efficiency", "min_max", "equality_min", "proportional", "jain"]
const LS_PERCENT = 0.8
const ITERATIONS = 20
const N_ROUNDS = 2
const N_BERNOULLI_SAMPLES = 1000
const SOURCE_PU = 1.03
const critical_buses = []
const N_PERIODS = 12
# Summer daily profile: 6am-5pm (12 hourly periods)
const LOAD_SCALE_FACTORS = [
    0.6,   # 6am  - early morning
    0.75,  # 7am  - morning ramp
    0.9,   # 8am  - morning ramp
    1.0,   # 9am  - approaching peak
    1.1,   # 10am - midday peak (AC)
    1.2,   # 11am - peak
    1.2,   # 12pm - peak
    1.15,  # 1pm  - peak
    1.05,  # 2pm  - afternoon decline
    0.95,  # 3pm  - afternoon
    1.0,   # 4pm  - evening ramp
    1.1,   # 5pm  - evening peak start
]

# Save results
save_dir = "results/$(Dates.today())/bilevel_comparisons_multiperiod"
mkpath(save_dir)

# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi = optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0)

# ============================================================
# MULTINETWORK DATA CREATION
# ============================================================
"""
Create multinetwork data from a single-period math dict.
Replicates network N times with different load scaling factors.
"""
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
        for (load_id, load) in nw_data["load"]
            load["pd"] = load["pd"] .* scale
            load["qd"] = load["qd"] .* scale
        end

        nw_data["time_period"] = t
        nw_data["load_scale"] = scale
        mn_data["nw"][nw_id] = nw_data
    end

    return mn_data
end

# ============================================================
# MULTIPERIOD BILEVEL OPTIMIZATION
# ============================================================
"""
Run bilevel optimization on multiperiod data.
The Jacobian is (T*N) x N: all pshed across periods w.r.t. global weights.
Upper-level fairness functions operate on the full (T*N) pshed vector.
"""
function run_bilevel_relaxed_mn(mn_data::Dict{String,Any}, iterations::Int, fair_weights_init::Vector{Float64},
                                fair_func::String, critical_id::Vector{Int}=Int[];
                                n_periods::Int=N_PERIODS)
    mn_new = deepcopy(mn_data)
    fair_weights = copy(fair_weights_init)
    n_weights = length(fair_weights)
    nw_ids = sort(collect(keys(mn_new["nw"])), by=x->parse(Int, x))

    pshed_lower_level = Float64[]
    pshed_upper_level = Float64[]
    final_weight_ids = Int[]
    final_weights = Float64[]
    completed_iterations = 0
    last_status = MOI.OPTIMIZE_NOT_CALLED
    prev_weights = copy(fair_weights)
    prev_pshed = Float64[]
    max_delta_weights = NaN
    max_delta_pshed = NaN

    for k in 1:iterations
        # Solve multiperiod lower-level and get (T*N) x N Jacobian
        dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs = lower_level_soln_mn(mn_new, fair_weights, k)

        n_total = length(pshed_val)  # T*N

        # Collect per-period pd reference values matching pshed ordering from Jacobian
        # Use refs dict (from instantiate_mc_model) to match the lower-level constraint:
        #   pshed[d] == (1 - z_demand[d]) * sum(ref[:load][d]["pd"])
        # The math dict pd may differ from ref pd due to PMD processing/per-unit conversion.
        pd_all = Float64[]
        for (nw, lid) in pshed_nw_ids
            push!(pd_all, sum(refs[nw][:load][lid]["pd"]))
        end

        # Apply fairness function on the full per-period (T*N) pshed vector
        # dpshed is (T*N) x N, pshed_val is (T*N), weight_vals is (N)
        if fair_func == "proportional"
            pshed_new, fair_weight_vals, status = proportional_fairness_load_shed(dpshed, pshed_val, weight_vals, pd_all, critical_id, weight_ids)
        elseif fair_func == "efficiency"
            math_dummy = _create_math_dummy_mn(mn_new)
            pshed_new, fair_weight_vals, status = complete_efficiency_load_shed(dpshed, pshed_val, weight_vals, math_dummy, critical_id, weight_ids)
        elseif fair_func == "min_max"
            pshed_new, fair_weight_vals, status = min_max_load_shed(dpshed, pshed_val, weight_vals, critical_id, weight_ids)
        elseif fair_func == "equality_min"
            pshed_new, fair_weight_vals, status = FairLoadDelivery.equality_min(dpshed, pshed_val, weight_vals, critical_id, weight_ids)
        elseif fair_func == "jain"
            pshed_new, fair_weight_vals, status = jains_fairness_index(dpshed, pshed_val, weight_vals, critical_id, weight_ids)
        elseif fair_func == "palma"
            pshed_new, fair_weight_vals, status = lin_palma_reformulated(dpshed, pshed_val, weight_vals, pd_all, critical_id, weight_ids)
        else
            error("Unknown fairness function: $fair_func")
        end

        last_status = status
        @info "[$fair_func] Iteration $k: upper-level status = $status"
        if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.TIME_LIMIT]
            @warn "[$fair_func] Iteration $k/$iterations: upper-level infeasible (status=$status) — stopping"
            break
        end
        completed_iterations = k

        # Track convergence
        max_delta_weights = maximum(abs.(fair_weight_vals .- prev_weights))
        if !isempty(prev_pshed)
            max_delta_pshed = maximum(abs.(pshed_new .- prev_pshed))
        end
        prev_weights = copy(fair_weight_vals)
        prev_pshed = copy(pshed_new)

        # Update weights in ALL periods of mn_data (global weights)
        for nw_id in nw_ids
            nw_data = mn_new["nw"][nw_id]
            for (i, w) in zip(weight_ids, fair_weight_vals)
                nw_data["load"][string(i)]["weight"] = w
            end
        end

        fair_weights = fair_weight_vals
        final_weight_ids = weight_ids
        final_weights = fair_weight_vals

        push!(pshed_lower_level, sum(pshed_val))
        push!(pshed_upper_level, sum(pshed_new))
    end

    weights_converged = !isnan(max_delta_weights) && max_delta_weights <= TRUST_RADIUS
    bilevel_summary = Dict(
        "completed_iterations" => completed_iterations,
        "total_iterations" => iterations,
        "last_status" => string(last_status),
        "early_stop" => completed_iterations < iterations,
        "max_delta_weights" => max_delta_weights,
        "max_delta_pshed" => max_delta_pshed,
        "weights_converged" => weights_converged,
        "n_periods" => length(nw_ids)
    )
    @info "[$fair_func] Bilevel finished: $completed_iterations/$iterations iterations, Δw_max=$(round(max_delta_weights, digits=6))"
    return mn_new, pshed_lower_level, pshed_upper_level, final_weight_ids, final_weights, bilevel_summary
end

"""
Aggregate (T*N) Jacobian, pshed, and pd to (N) per-load values by summing over periods.
Fairness is measured across loads (customers): pshed_agg[i] = Σ_t pshed[t,i].
Returns: J_agg (N×N), pshed_agg (N), pd_agg (N) in the same load order as weight_ids.
"""
function _aggregate_by_load(dpshed::Matrix{Float64}, pshed_val::Vector{Float64},
                            pshed_nw_ids::Vector, mn_data::Dict{String,Any},
                            weight_ids::Vector{Int})
    n_w = length(weight_ids)
    # Build load_id → position mapping (matching weight_ids order)
    lid_to_pos = Dict(lid => pos for (pos, lid) in enumerate(weight_ids))

    pshed_agg = zeros(n_w)
    dpshed_agg = zeros(n_w, n_w)
    pd_agg = zeros(n_w)

    for (row_idx, (nw, lid)) in enumerate(pshed_nw_ids)
        pos = lid_to_pos[lid]
        pshed_agg[pos] += pshed_val[row_idx]
        dpshed_agg[pos, :] .+= dpshed[row_idx, :]
        pd_agg[pos] += sum(mn_data["nw"][string(nw)]["load"][string(lid)]["pd"])
    end

    return dpshed_agg, pshed_agg, pd_agg
end

"""Helper to create a dummy math dict with total load info for efficiency function."""
function _create_math_dummy_mn(mn_data::Dict{String,Any})
    # Aggregate loads across all periods for the efficiency function's total_load_ref calculation
    math_dummy = Dict{String,Any}("load" => Dict{String,Any}())
    nw_ids = sort(collect(keys(mn_data["nw"])), by=x->parse(Int, x))
    # Use first period's load structure, but sum demands across periods
    first_nw = mn_data["nw"][nw_ids[1]]
    for (lid, load) in first_nw["load"]
        total_pd = zeros(length(load["pd"]))
        for nw_id in nw_ids
            total_pd .+= mn_data["nw"][nw_id]["load"][lid]["pd"]
        end
        math_dummy["load"][lid] = Dict(
            "pd" => total_pd,
            "connections" => load["connections"]
        )
    end
    return math_dummy
end

# ============================================================
# RESULT EXTRACTION
# ============================================================
function extract_per_period_pshed(pshed_val::Vector{Float64}, pshed_nw_ids::Vector, n_loads::Int, n_periods::Int)
    """Reshape flattened (T*N) pshed vector into per-period vectors."""
    per_period = Dict{Int,Vector{Float64}}()
    for t in 0:(n_periods-1)
        start_idx = t * n_loads + 1
        end_idx = (t + 1) * n_loads
        per_period[t] = pshed_val[start_idx:end_idx]
    end
    return per_period
end

function get_total_demand_mn(mn_data::Dict{String,Any})
    total_pd = 0.0
    for (_, nw_data) in mn_data["nw"]
        for (_, load) in nw_data["load"]
            total_pd += sum(load["pd"])
        end
    end
    return total_pd
end

function get_per_period_demand(mn_data::Dict{String,Any})
    demands = Dict{String,Float64}()
    for (nw_id, nw_data) in mn_data["nw"]
        total = 0.0
        for (_, load) in nw_data["load"]
            total += sum(load["pd"])
        end
        demands[nw_id] = total
    end
    return demands
end

# ============================================================
# MAIN COMPARISON LOOP
# ============================================================
function run_comparison_mn()
    results = DataFrame(
        case = String[],
        fair_func = String[],
        n_periods = Int[],
        total_demand_all_periods = Float64[],
        total_pshed_lower = Float64[],
        total_pshed_upper = Float64[],
        pct_shed = Float64[],
        bilevel_iters_completed = Int[],
        bilevel_iters_total = Int[],
        bilevel_last_status = String[],
        bilevel_early_stop = Bool[],
        max_delta_weights = Float64[],
        max_delta_pshed = Float64[],
        weights_converged = Bool[]
    )

    per_period_results = Dict{String, Dict{String, Dict{Int, Vector{Float64}}}}()
    final_weights_results = Dict{String, Dict{String, Dict{Symbol, Any}}}()
    rounding_results = Dict{String, Dict{String, Dict{String, Any}}}()
    failed_combinations = Vector{Tuple{String, String, String}}()

    println("=" ^ 70)
    println("MULTIPERIOD LOAD SHED COMPARISON ACROSS FAIRNESS FUNCTIONS")
    println("Periods: $N_PERIODS, Load scales: $LOAD_SCALE_FACTORS")
    println("=" ^ 70)

    for case in CASES
        println("\n>>> Processing case: $case")

        # Setup base network
        eng, math, lbs, critical_id = FairLoadDelivery.setup_network("ieee_13_aw_edit/$case.dss", LS_PERCENT, SOURCE_PU, critical_buses)
        sorted_load_ids = sort(parse.(Int, collect(keys(math["load"]))))
        n_loads = length(sorted_load_ids)
        fair_weights = Float64[math["load"][string(i)]["weight"] for i in sorted_load_ids]

        # Create multiperiod data
        mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)
        total_demand = get_total_demand_mn(mn_data)
        per_period_demand = get_per_period_demand(mn_data)

        println("    Total demand (all periods): $(round(total_demand, digits=4))")
        for (nw_id, demand) in sort(collect(per_period_demand), by=x->parse(Int, x[1]))
            println("    Period $nw_id demand: $(round(demand, digits=4))")
        end

        per_period_results[case] = Dict{String, Dict{Int, Vector{Float64}}}()
        final_weights_results[case] = Dict{String, Dict{Symbol, Any}}()
        rounding_results[case] = Dict{String, Dict{String, Any}}()

        for fair_func in FAIR_FUNCS
            print("  $fair_func: ")

            mn_relaxed, pshed_lower, pshed_upper, weight_ids, final_wts, bilevel_summary =
                run_bilevel_relaxed_mn(mn_data, ITERATIONS, fair_weights, fair_func, critical_id)

            if bilevel_summary["completed_iterations"] == 0
                @warn "[$case/$fair_func] FAILED — bilevel infeasible on first iteration"
                push!(failed_combinations, (case, fair_func, "Bilevel infeasible on first iteration"))
                continue
            end

            final_weights_results[case][fair_func] = Dict(
                :weight_ids => weight_ids,
                :weights => final_wts,
                :bilevel_summary => bilevel_summary
            )

            total_pshed_lower = isempty(pshed_lower) ? NaN : pshed_lower[end]
            total_pshed_upper = isempty(pshed_upper) ? NaN : pshed_upper[end]
            pct_shed = (total_pshed_lower / total_demand) * 100

            println("shed=$(round(pct_shed, digits=2))%, iters=$(bilevel_summary["completed_iterations"])")

            push!(results, (
                case,
                fair_func,
                N_PERIODS,
                total_demand,
                total_pshed_lower,
                total_pshed_upper,
                pct_shed,
                bilevel_summary["completed_iterations"],
                bilevel_summary["total_iterations"],
                bilevel_summary["last_status"],
                bilevel_summary["early_stop"],
                bilevel_summary["max_delta_weights"],
                bilevel_summary["max_delta_pshed"],
                bilevel_summary["weights_converged"]
            ))

            # Per-period rounding and topology selection
            println("    Running per-period rounding & topology selection...")
            per_period_topology = round_and_select_topology_mn(
                mn_relaxed;
                n_samples=N_BERNOULLI_SAMPLES,
                n_rounds=N_ROUNDS,
                seed_base=100,
                solver=ipopt_solver
            )
            rounding_results[case][fair_func] = per_period_topology

            # Print per-period rounding summary
            for (nw_id, period_res) in sort(collect(per_period_topology), by=x -> parse(Int, x[1]))
                n_feas = period_res["n_feasible_samples"]
                load_shed = period_res["total_load_shed"]
                ac_statuses = [f["feas_status"] for f in period_res["ac_feas"]]
                best_mld_obj = period_res["best_mld"] !== nothing && !isempty(period_res["best_mld"]) ? get(period_res["best_mld"], "objective", NaN) : NaN
                println("      Period $nw_id: n_feasible=$n_feas, load_shed=$(round(load_shed, digits=4)), AC_feas=$ac_statuses, MLD_obj=$(round(best_mld_obj, digits=4))")
            end
        end
    end

    return results, per_period_results, final_weights_results, rounding_results, failed_combinations
end

# ============================================================
# RUN AND SAVE RESULTS
# ============================================================
println("\nStarting multiperiod comparison at $(now())...\n")

results, per_period_results, final_weights_results, rounding_results, failed_combinations = run_comparison_mn()

# Print failed combinations
if !isempty(failed_combinations)
    println("\n" * "=" ^ 60)
    println("FAILED COMBINATIONS")
    println("=" ^ 60)
    for (case, fair_func, reason) in failed_combinations
        println("  $case / $fair_func: $reason")
    end
end

# Print summary table
println("\n" * "=" ^ 70)
println("MULTIPERIOD SUMMARY TABLE")
println("=" ^ 70)
println(results)

# Save CSV
csv_path = joinpath(save_dir, "multiperiod_load_shed_comparison.csv")
CSV.write(csv_path, results)
println("\nResults saved to: $csv_path")

# Print final weights per fairness function
println("\n" * "=" ^ 70)
println("FINAL WEIGHTS PER FAIRNESS FUNCTION")
println("=" ^ 70)
for case in CASES
    if !haskey(final_weights_results, case)
        continue
    end
    println("\n$case:")
    for fair_func in FAIR_FUNCS
        if !haskey(final_weights_results[case], fair_func)
            continue
        end
        wdata = final_weights_results[case][fair_func]
        wids = wdata[:weight_ids]
        wvals = wdata[:weights]
        println("  $fair_func: $(round.(wvals, digits=3))")
    end
end

# Print per-period rounding results
println("\n" * "=" ^ 70)
println("PER-PERIOD ROUNDING & TOPOLOGY SELECTION RESULTS")
println("=" ^ 70)
for case in CASES
    if !haskey(rounding_results, case)
        continue
    end
    println("\n$case:")
    for fair_func in FAIR_FUNCS
        if !haskey(rounding_results[case], fair_func)
            continue
        end
        println("  $fair_func:")
        per_period = rounding_results[case][fair_func]
        for (nw_id, period_res) in sort(collect(per_period), by=x -> parse(Int, x[1]))
            n_feas = period_res["n_feasible_samples"]
            load_shed = round(period_res["total_load_shed"], digits=4)
            ac_statuses = [f["feas_status"] for f in period_res["ac_feas"]]
            best_mld_obj = period_res["best_mld"] !== nothing && !isempty(period_res["best_mld"]) ? round(get(period_res["best_mld"], "objective", NaN), digits=4) : NaN
            # Relaxed switch states
            sw_str = ""
            if haskey(period_res, "relaxed_switch_states")
                sw = period_res["relaxed_switch_states"]
                sw_str = join(["s$k=$(round(v, digits=3))" for (k,v) in sort(collect(sw))], ", ")
            end
            println("    Period $nw_id: feasible=$n_feas, shed=$load_shed, AC=$ac_statuses, MLD=$best_mld_obj")
            if !isempty(sw_str)
                println("             switches: $sw_str")
            end
        end
    end
end

# Save rounding results to CSV
rounding_df = DataFrame(
    case=String[], fair_func=String[], period=String[],
    n_feasible_samples=Int[], total_load_shed=Float64[],
    best_mld_obj=Float64[], any_ac_feasible=Bool[]
)
for case in CASES
    if !haskey(rounding_results, case); continue; end
    for fair_func in FAIR_FUNCS
        if !haskey(rounding_results[case], fair_func); continue; end
        for (nw_id, period_res) in rounding_results[case][fair_func]
            any_ac = isempty(period_res["ac_feas"]) ? false : any(f["feas_status"] for f in period_res["ac_feas"])
            mld_obj = period_res["best_mld"] !== nothing && !isempty(period_res["best_mld"]) ? get(period_res["best_mld"], "objective", NaN) : NaN
            push!(rounding_df, (case, fair_func, nw_id,
                period_res["n_feasible_samples"],
                period_res["total_load_shed"],
                mld_obj, any_ac))
        end
    end
end
rounding_csv_path = joinpath(save_dir, "multiperiod_rounding_results.csv")
CSV.write(rounding_csv_path, rounding_df)
println("\nRounding results saved to: $rounding_csv_path")

# Save full results to JLD2 for visualization script
jld2_path = joinpath(save_dir, "multiperiod_results.jld2")
@save jld2_path results final_weights_results rounding_results failed_combinations CASES FAIR_FUNCS N_PERIODS LOAD_SCALE_FACTORS LS_PERCENT SOURCE_PU
println("Full results saved to: $jld2_path")
println("\nRun script/multiperiod/visualize_multiperiod.jl to generate heatmaps.")
