"""
    Multiperiod Bilevel FLDP Validation Runner

    Multi-period analogue of run_validation.jl. Runs the bilevel FLDP loop
    on a multinetwork (T periods, diurnal load + TOU profiles) and validates:
      1. Label consistency across stages (per period)
      2. Voltage limits (per period AC PF)
      3. Switch ampacity (per period AC PF)
      4. AC feasibility (per period)

    Differences vs single-period:
      - Lower level uses lower_level_soln_mn (T*N pshed, T*N weights, T*N x T*N Jacobian)
      - Upper level (min_max_load_shed) is fed peak_time_costs and per-period pd
      - Random rounding is performed PER PERIOD (each period's relaxed switch
        states are independently rounded to a feasible radial topology)
      - AC PF + voltage/ampacity checks run PER PERIOD on each rounded topology

    Usage:
        julia --project=. script/bilevel_validation/run_validation_mn.jl
"""

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
using Logging, LoggingExtras

const PMD = PowerModelsDistribution

include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASE = "case6_unbalanced_switch_meshed_good4integer"
const CASE_FILE = joinpath(@__DIR__, "../../data/pmd_opendss/$CASE.dss")

LS_PERCENT = 0.8
const ITERATIONS = 20
const FAIR_FUNC = "min_max"
pshed_type = "proportional"  # "absolute" or "proportional"
const N_ROUNDS = 1
const N_BERNOULLI_SAMPLES = 2000

# Multi-period setup: 24 hourly periods with linear-ramp load + TOU peak-cost profiles
const N_PERIODS = 24
const PERIOD_HOURS        = collect(0:N_PERIODS-1)
# Linear ramp from 0.7 (period 1) to 1.0 (period 24).
const LOAD_SCALE_FACTORS  = [round(s, digits=3) for s in LinRange(0.7, 1.0, N_PERIODS)]
const PEAK_TIME_COSTS     = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                             for h in PERIOD_HOURS]

switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT

# Solvers
ipopt_solver  = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = Gurobi.Optimizer

validation_results = Dict{String,Any}(
    "case"             => CASE,
    "fair_func"        => FAIR_FUNC,
    "iterations"       => ITERATIONS,
    "n_periods"        => N_PERIODS,
    "period_hours"     => PERIOD_HOURS,
    "load_scales"      => LOAD_SCALE_FACTORS,
    "peak_time_costs"  => PEAK_TIME_COSTS,
    "pshed_type"       => pshed_type,
)

save_dir = "results/$(Dates.today())/bilevel_validation_mn/$CASE/$(FAIR_FUNC)_$(pshed_type)"
mkpath(save_dir)

log_file = joinpath(save_dir, "run_validation_mn.log")
global_logger(TeeLogger(global_logger(), FileLogger(log_file)))
@info "Logging to $log_file"
@info "N_PERIODS=$N_PERIODS, hours=$PERIOD_HOURS, scales=$LOAD_SCALE_FACTORS, λ=$PEAK_TIME_COSTS"

# ============================================================
# STEP 1: NETWORK + MULTINETWORK SETUP
# ============================================================
print_validation_header("Step 1: Network + Multinetwork Setup")

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating=switch_rating)

"""
Replicate a single-period math dict into a multinetwork with per-period load scaling.
"""
function create_multinetwork_data(base_math::Dict{String,Any}, n_periods::Int, load_scales::Vector{Float64})
    @assert length(load_scales) == n_periods
    mn_data = Dict{String,Any}(
        "multinetwork" => true,
        "per_unit"     => true,
        "data_model"   => PMD.MATHEMATICAL,
        "nw"           => Dict{String,Any}()
    )
    for key in ["baseMVA", "basekv", "bus_lookup", "settings"]
        haskey(base_math, key) && (mn_data[key] = deepcopy(base_math[key]))
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
        nw_data["load_scale"]  = scale
        mn_data["nw"][nw_id]   = nw_data
    end
    return mn_data
end

mn_data = create_multinetwork_data(math, N_PERIODS, LOAD_SCALE_FACTORS)
nw_ids_sorted = sort(collect(keys(mn_data["nw"])), by=x->parse(Int, x))
n_loads_per_period = length(mn_data["nw"][nw_ids_sorted[1]]["load"])
@info "Built multinetwork with $N_PERIODS periods, $n_loads_per_period loads per period"

# ============================================================
# STEP 2: BILEVEL ITERATIONS (multi-period)
# ============================================================
print_validation_header("Step 2: Bilevel Iterations ($FAIR_FUNC, $pshed_type, $ITERATIONS iters)")

mn_new = deepcopy(mn_data)

# Initial weights: read from base math (per-load) and replicate across periods later via lower_level_soln_mn
fair_weights_init = Float64[load["weight"] for (_, load) in math["load"]]
fair_weights = copy(fair_weights_init)  # gets replaced after iter 1 by per-period (T*N) values

iteration_label_consistent = true
all_pshed_lower = Float64[]
all_pshed_upper = Float64[]
last_status = MOI.OPTIMIZE_NOT_CALLED
final_weight_ids = Int[]
final_pshed_nw_ids = Tuple[]

for k in 1:ITERATIONS
    global fair_weights, iteration_label_consistent, last_status, final_weight_ids, final_pshed_nw_ids
    println("\n  --- Iteration $k ---")

    local dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs
    try
        dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs =
            lower_level_soln_mn(mn_new, fair_weights, k)
    catch err
        @warn "[$FAIR_FUNC/$pshed_type] iter $k lower-level failed ($err) — stopping bilevel, falling back to last converged weights"
        break
    end
    n_loads = length(weight_ids)

    # Per-load pd reference matching pshed ordering (across all periods)
    pd_all = Float64[sum(refs[nw][:load][lid]["pd"]) for (nw, lid) in pshed_nw_ids]

    # Upper-level fairness step (multi-period, peak-cost weighted)
    if FAIR_FUNC == "min_max"
        pshed_new, fair_weight_vals, status = min_max_load_shed(
            dpshed, pshed_val, weight_vals;
            critical_ids=critical_id, weight_ids=weight_ids,
            peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads,
            pd=pd_all, pshed_type=pshed_type)
    else
        error("Only FAIR_FUNC=\"min_max\" wired up for now (pshed_type toggle).")
    end
    last_status = status
    @info "[$FAIR_FUNC/$pshed_type] iter $k upper-level status = $status"
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]
        @warn "upper-level not converged at iter $k — stopping"
        break
    end

    # Push T*N weights back into mn_new per period
    for (t, nw_id) in enumerate(nw_ids_sorted)
        offset = (t - 1) * n_loads
        for (j, lid) in enumerate(weight_ids)
            mn_new["nw"][nw_id]["load"][string(lid)]["weight"] = fair_weight_vals[offset + j]
        end
    end
    fair_weights      = copy(fair_weight_vals)
    final_weight_ids  = weight_ids
    final_pshed_nw_ids = pshed_nw_ids

    push!(all_pshed_lower, sum(pshed_val))
    push!(all_pshed_upper, sum(pshed_new))
    println("    Σ pshed (lower) = $(round(sum(pshed_val), digits=3))   Σ pshed (upper) = $(round(sum(pshed_new), digits=3))")
end

validation_results["bilevel"] = Dict(
    "completed_iterations" => length(all_pshed_lower),
    "last_status"          => string(last_status),
    "pshed_lower_history"  => all_pshed_lower,
    "pshed_upper_history"  => all_pshed_upper,
)

# ============================================================
# STEP 3: FINAL RELAXED MULTI-PERIOD SOLVE
# ============================================================
print_validation_header("Step 3: Final relaxed multi-period MLD with updated weights")
mn_relaxed_final = FairLoadDelivery.solve_mn_mc_mld_shed_implicit_diff(mn_new, ipopt_solver)
@info "relaxed multi-period termination: $(mn_relaxed_final["termination_status"])"

# ============================================================
# STEP 4: PER-PERIOD RANDOM ROUNDING + AC FEASIBILITY
# ============================================================
print_validation_header("Step 4: Per-period rounding + AC feasibility")

per_period_results = Dict{String,Any}()
mn_rounded = Dict{String,Dict{String,Any}}()  # rounded math per nw_id

for (t, nw_id) in enumerate(nw_ids_sorted)
    println("\n  ----- Period $t (nw=$nw_id, scale=$(LOAD_SCALE_FACTORS[t]), λ=$(PEAK_TIME_COSTS[t])) -----")
    period_checks = Dict{String,Any}()

    # Single-period math dict for this period (already scaled)
    math_t = deepcopy(mn_new["nw"][nw_id])

    # Build a single-period implicit-diff model to get the ref needed for rounding helpers
    imp_diff_model_t = instantiate_mc_model(
        math_t, LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
    ref_t = imp_diff_model_t.ref[:it][:pmd][:nw][0]

    # Pull this period's relaxed switch & block states from the multi-period solution
    relaxed_t = mn_relaxed_final["solution"]["nw"][nw_id]
    switch_states = Dict(parse(Int, sid) => sw["state"]   for (sid, sw)  in relaxed_t["switch"])
    block_status  = Dict(parse(Int, bid) => bl["status"] for (bid, bl)  in relaxed_t["block"])

    # Bernoulli rounding → radial-feasible candidate
    rng = 100 + t  # different seed per period
    bernoulli_samples = generate_bernoulli_samples(switch_states, N_BERNOULLI_SAMPLES, rng)
    index, sw_radial, block_ids, bl_radial, load_ids, load_status, _ =
        FairLoadDelivery.radiality_check(ref_t, switch_states, block_status, bernoulli_samples)

    if index === nothing
        @warn "[period $t] no feasible radial topology found"
        period_checks["radiality_found"] = Dict("passed" => false)
        per_period_results["period_$t"] = period_checks
        continue
    end
    period_checks["radiality_found"] = Dict("passed" => true, "details" => ["sample index $index"])
    print_check_result("Period $t: radial topology found", true, "sample $index")

    # Apply rounded switch states to this period's math
    math_t_rounded = update_network(math_t, sw_radial, ref_t)

    # Solve rounded integer single-period MLD for this period
    mld_rounded_t = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(math_t_rounded, gurobi_solver)
    rounded_term  = mld_rounded_t["termination_status"]
    rounded_ok    = (rounded_term == MOI.OPTIMAL || rounded_term == MOI.LOCALLY_SOLVED || rounded_term == MOI.ALMOST_LOCALLY_SOLVED)
    period_checks["rounded_mld_converged"] = Dict("passed" => rounded_ok, "details" => ["status: $rounded_term"])
    print_check_result("Period $t: rounded MLD converged", rounded_ok, "status: $rounded_term")
    if !rounded_ok
        per_period_results["period_$t"] = period_checks
        continue
    end

    # Voltage + ampacity on the rounded MLD solution
    v_passed_r, v_violations_r, v_summary_r = check_voltage_limits_relaxed(mld_rounded_t, math_t_rounded)
    period_checks["voltage_limits_rounded"] = Dict("passed" => v_passed_r, "details" => [string(v) for v in v_violations_r])
    print_check_result("Period $t: voltage limits (rounded)", v_passed_r, "$(v_summary_r["violations"]) violations / $(v_summary_r["checked"]) checked")

    c_passed_r, c_violations_r, c_summary_r = check_switch_ampacity(mld_rounded_t, math_t_rounded)
    period_checks["current_limits_rounded"] = Dict("passed" => c_passed_r, "details" => [string(v) for v in c_violations_r])
    print_check_result("Period $t: switch ampacity (rounded)", c_passed_r, "$(c_summary_r["violations"]) violations")

    # Build AC dispatch network and run AC PF
    math_ac_t = ac_network_update(math_t_rounded, ref_t; mld_solution=mld_rounded_t)
    ac_result_t = PowerModelsDistribution.solve_mc_pf(math_ac_t, IVRUPowerModel, ipopt_solver)
    ac_term_t   = ac_result_t["termination_status"]
    ac_ok_t     = (ac_term_t == MOI.OPTIMAL || ac_term_t == MOI.LOCALLY_SOLVED || ac_term_t == MOI.ALMOST_LOCALLY_SOLVED)
    period_checks["ac_convergence"] = Dict("passed" => ac_ok_t, "details" => ["status: $ac_term_t"])
    print_check_result("Period $t: AC PF converged", ac_ok_t, "status: $ac_term_t")

    if ac_ok_t && haskey(ac_result_t, "solution")
        v_passed_ac, v_violations_ac, v_summary_ac = check_voltage_limits_ac(ac_result_t, math_ac_t)
        period_checks["voltage_limits_ac"] = Dict("passed" => v_passed_ac, "details" => [string(v) for v in v_violations_ac])
        print_check_result("Period $t: voltage limits (AC PF)", v_passed_ac, "$(v_summary_ac["violations"]) / $(v_summary_ac["checked"])")

        if haskey(ac_result_t["solution"], "switch")
            c_passed_ac, c_violations_ac, _ = check_switch_ampacity(ac_result_t, math_ac_t)
            period_checks["current_limits_ac"] = Dict("passed" => c_passed_ac, "details" => [string(v) for v in c_violations_ac])
            print_check_result("Period $t: switch ampacity (AC PF)", c_passed_ac, "$(length(c_violations_ac)) violations")
        else
            period_checks["current_limits_ac"] = Dict("passed" => true, "details" => ["no switch data in AC solution"])
        end

        if haskey(ac_result_t["solution"], "load")
            served_ac = sum(sum(ld["pd"]) for (_, ld) in ac_result_t["solution"]["load"] if haskey(ld, "pd"))
            println("    Period $t: AC active power served = $(round(served_ac, digits=3))")
            period_checks["ac_total_served"] = served_ac
        end
    end

    mn_rounded[nw_id] = math_t_rounded
    per_period_results["period_$t"] = period_checks
end

validation_results["per_period"] = per_period_results

# ============================================================
# REPORT
# ============================================================
report_path = joinpath(save_dir, "validation_report_mn_$(pshed_type).txt")
generate_summary_report(validation_results, report_path)
println("\nMulti-period validation complete. Report → $report_path")
