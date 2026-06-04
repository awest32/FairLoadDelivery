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
using Printf

const PMD = PowerModelsDistribution

include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

# ============================================================
# CONFIGURATION
# ============================================================
 #CASE = "case6_unbalanced_switch_more_meshed_bd_good4integer"
CASE = "case6_unbalanced_switch_more_meshed_good4integer"  # no-bd — matches single-level + finals
 #CASE = "motivation_c_good4integer"
case = "6_bus" #"13_bus"#"6_bus"
#critical_load = ["611"]
CASE_FILE = joinpath(@__DIR__,"../../data/pmd_opendss/$CASE.dss")
#CASE_FILE = joinpath(@__DIR__, "../../data/ieee_13_aw_edit/$CASE.dss")
LS_PERCENT = 0.8
ITERATIONS = 20
FAIR_FUNC = "palma"  # "min_max", "palma", or "efficiency"
# Which quantity the Palma objective sorts: "served" (income-Palma analogy; default)
# or "shed" (fairness of the shed burden — matches the single-level shed-objective).
PALMA_ON = get(ENV, "PALMA_TARGET", "served") == "shed" ? :shed : :served
pshed_type = "absolute"  # "absolute" or "proportional"
N_ROUNDS = 1
N_BERNOULLI_SAMPLES = 2000

# Multi-period setup: per-load Hamilton & Aliprantis (PECI 2023) schedules.
# Each load name is deterministically mapped to (schedule_idx, ±1h shift) so
# loads peak at different periods — replaces the old uniform LOAD_SCALE_FACTORS.
#
# SELECTED_HOURS downsamples the 24h day to a representative subset. The
# DiffOpt forward Jacobian costs O((T·N)^2) per bilevel iter, so cutting T
# from 24 → 8 drops per-iter cost ~9×. Hours chosen to span the operational
# regimes: trough (4), morning ramp (6,8), midday plateau (12), pre-peak rise
# (15), evening peak (18), descent (20), late-night start (22).
 # SELECTED_HOURS    = collect(0:23)   # T=24 full diurnal cycle (was [4,6,8,12,15,18,20,22] for T=8)
 #SELECTED_HOURS    = [4, 12, 15, 18, 22]   # T=5: trough, midday, pre-peak, evening peak, descent
 #SELECTED_HOURS    = collect(0:23)   # T=24 full diurnal cycle
 SELECTED_HOURS    = [4, 6, 8, 12, 15, 18, 20, 22]   # T=8 — defense: original formal-CC MILP tractable (forward Jacobian = 72 cols, vs 216 at T=24 which hung)
 #SELECTED_HOURS    = [4, 18, 8]

 N_PERIODS         = length(SELECTED_HOURS)
PEAK_STRESS       = 1.0                            # uniform multiplier over the paper schedules
CENTER_AT_NOMINAL = true                           # divide each schedule by its daily mean
PERIOD_HOURS      = SELECTED_HOURS
 PEAK_TIME_COSTS   = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                           for h in PERIOD_HOURS]
# Override results_block_mn.jl default — pick trough/plateau/peak indices into
# SELECTED_HOURS so the grouped bar covers the 3 most distinct regimes.
REP_PERIODS = [1, 4, 6]   # T=8 indices → hours 4, 12, 18

switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT

# Solvers
# max_iter=30000 + acceptable_tol/iter mirror build_mn_mc_mld_shedding_implicit_diff
# (src/prob/mld.jl:275-279). Needed for motivation_c 13bus: Step 3's final relaxed
# solve and post-iteration AC PFs operate on weights deep into the descent, where
# the NLP becomes brittle. Default Ipopt max_iter=3000 → ITERATION_LIMIT after ~700s
# → Steps 4-6 (rounding/AC/JLD2) get skipped on a non-converged relaxation.
ipopt_solver  = optimizer_with_attributes(Ipopt.Optimizer,
    "max_iter"        => 30_000,
    "acceptable_tol"  => 1e-4,
    "acceptable_iter" => 50,
    "print_level"     => 0)
gurobi_solver = Gurobi.Optimizer

# shed-objective writes to a separate dir so it won't overwrite the served run.
obj_dir_suffix = PALMA_ON === :shed ? "_shedobj" : ""
save_dir = "results/$(Dates.today())/bilevel_validation_mn/$CASE/$(FAIR_FUNC)_$(pshed_type)$(obj_dir_suffix)"
mkpath(save_dir)

log_file = joinpath(save_dir, "run_validation_mn.log")
global_logger(TeeLogger(global_logger(), FileLogger(log_file)))
@info "Logging to $log_file"

# ============================================================
# STEP 1: NETWORK + MULTINETWORK SETUP
# ============================================================
print_validation_header("Step 1: Network + Multinetwork Setup")

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating=switch_rating)#, critical_load=critical_load

# System aggregate scale per period — used by results_block_mn.jl print rows and
# by downstream plotting. Per-load shape now comes from the H&A schedules.
LOAD_SCALE_FACTORS = FairLoadDelivery.aggregate_demand_fraction(math, N_PERIODS;
    hours = SELECTED_HOURS, center_at_nominal = CENTER_AT_NOMINAL) .* PEAK_STRESS

validation_results = Dict{String,Any}(
    "case"             => CASE,
    "fair_func"        => FAIR_FUNC,
    "iterations"       => ITERATIONS,
    "n_periods"        => N_PERIODS,
    "period_hours"     => PERIOD_HOURS,
    "load_scales"      => LOAD_SCALE_FACTORS,
    "peak_stress"      => PEAK_STRESS,
    "peak_time_costs"  => PEAK_TIME_COSTS,
    "pshed_type"       => pshed_type,
)

@info "N_PERIODS=$N_PERIODS, hours=$PERIOD_HOURS, agg_scales=$LOAD_SCALE_FACTORS, peak_stress=$PEAK_STRESS, λ=$PEAK_TIME_COSTS"

mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS, center_at_nominal = CENTER_AT_NOMINAL)

println("Load profile assignments for $CASE:")
for row in FairLoadDelivery.profile_assignment_table(math)
    println("  ", row)
end
nw_ids_sorted = sort(collect(keys(mn_data["nw"])), by=x->parse(Int, x))
n_loads_per_period = length(mn_data["nw"][nw_ids_sorted[1]]["load"])
@info "Built multinetwork with $N_PERIODS periods, $n_loads_per_period loads per period (profiled)"

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
iter_timings = Dict{Symbol,Any}[]   # one dict per completed bilevel iteration
last_status = MOI.OPTIMIZE_NOT_CALLED
final_weight_ids = Int[]
final_pshed_nw_ids = Tuple[]

for k in 1:ITERATIONS
    global fair_weights, iteration_label_consistent, last_status, final_weight_ids, final_pshed_nw_ids
    println("\n  --- Iteration $k ---")

    timing = Dict{Symbol,Any}(:iter => k)

    local dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs
    try
        # Solve multinetwork integer MLD first to fix switch topology per period,
        # then run DiffOpt lower level on the topology-fixed multinetwork. Mirrors
        # the single-period warm-start at run_validation.jl:220-225.
        #
        # Skipped for Palma: with switch topology fixed, pshed becomes near-binary
        # ({0, pd} per load) and bottom-40%-of-served can sum to zero in some
        # periods, leaving the Charnes-Cooper constraint σ[t]*bot_sum_t==1 with no
        # feasible σ — Gurobi then hits TimeLimit with no incumbent.
        if FAIR_FUNC != "palma"
            t_int = @elapsed mld_int_mn = FairLoadDelivery.solve_mn_mc_mld_switch_integer(mn_new, gurobi_solver;
                peak_time_costs=PEAK_TIME_COSTS)
            timing[:integer_warmstart_s] = t_int
            int_term = mld_int_mn["termination_status"]
            @info "[$FAIR_FUNC/$pshed_type] iter $k integer MLD status = $int_term (t=$(round(t_int,digits=1))s)"
            if int_term ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED]
                error("integer MLD did not converge (status=$int_term)")
            end
            for nw_id in nw_ids_sorted
                mn_new["nw"][nw_id] = update_network(
                    mld_int_mn["solution"]["nw"][nw_id], mn_new["nw"][nw_id])
            end
        else
            timing[:integer_warmstart_s] = 0.0
        end

        lower_timings = Dict{Symbol,Any}()
        # NOTE: `@elapsed (a, b, c = func())` interacts badly with `local`-declared
        # tuple targets under Julia 1.12.1 — the macro's let-scope holds the
        # assignment, leaving the outer locals undefined and the next access
        # triggers UndefVarError(:dpshed). Split the call out so the assignment
        # lands in the outer scope.
        _t_lower_start = time()
        (dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs) =
            lower_level_soln_mn(mn_new, fair_weights, k; timings=lower_timings)
        t_lower = time() - _t_lower_start
        timing[:lower_level_total_s] = t_lower
        merge!(timing, Dict(Symbol("lower_$(k2)") => v for (k2, v) in lower_timings))
    catch err
        @warn "[$FAIR_FUNC/$pshed_type] iter $k lower-level failed ($err) — stopping bilevel, falling back to last converged weights"
        timing[:fail_phase] = "lower_level"
        push!(iter_timings, timing)
        break
    end
    n_loads = length(weight_ids)

    # Per-load pd reference matching pshed ordering (across all periods)
    pd_all = Float64[sum(refs[nw][:load][lid]["pd"]) for (nw, lid) in pshed_nw_ids]

    # Upper-level fairness step (multi-period, peak-cost weighted).
    # Wrapped in try/catch so an upper-level error (e.g. formal-CC INFEASIBLE
    # after the bilevel pushes pserved_prev into a σ_max-degenerate corner)
    # doesn't crash the whole bilevel — fall back to previous weights and stop.
    local pshed_new, fair_weight_vals, status
    upper_timings = Dict{Symbol,Any}()
    try
        t_upper = @elapsed begin
            if FAIR_FUNC == "min_max"
                pshed_new, fair_weight_vals, status = min_max_load_shed(
                    dpshed, pshed_val, weight_vals;
                    critical_ids=critical_id, weight_ids=weight_ids,
                    peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads,
                    pd=pd_all, pshed_type=pshed_type, timings=upper_timings)
            elseif FAIR_FUNC == "palma"
                # 2026-05-19 defense: formal-CC (use_weak_cc=false default). Weak-CC
                # failed on 6-bus T=24 with NO incumbent inside 10 min; formal-CC
                # closes within ~25s/iter on this case per 2026-05-17 partial run.
                # 10-min cap is generous headroom; on rare INFEASIBLE iters (iter 20
                # hit this in the 2026-05-17 run) lin_palma_reformulated falls back
                # to weak-CC for that iter only.
                pshed_new, fair_weight_vals, status = lin_palma_reformulated(
                    dpshed, pshed_val, weight_vals, pd_all;
                    critical_ids=critical_id, weight_ids=weight_ids,
                    peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads,
                    time_limit=60*10, timings=upper_timings, palma_on=PALMA_ON)
            elseif FAIR_FUNC == "efficiency"
                pshed_new, fair_weight_vals, status = efficient_load_shed(
                    dpshed, pshed_val, weight_vals;
                    critical_ids=critical_id, weight_ids=weight_ids,
                    peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads,
                    timings=upper_timings)
            else
                error("FAIR_FUNC=\"$FAIR_FUNC\" not wired up; supported: \"min_max\", \"palma\", \"efficiency\".")
            end
        end
        timing[:upper_level_total_s] = t_upper
        merge!(timing, Dict(Symbol("upper_$(k2)") => v for (k2, v) in upper_timings))
    catch err
        @warn "[$FAIR_FUNC/$pshed_type] iter $k upper-level FAILED ($err) — stopping bilevel, keeping iter $(k-1) weights"
        timing[:fail_phase] = "upper_level"
        push!(iter_timings, timing)
        break
    end
    last_status = status
    @info "[$FAIR_FUNC/$pshed_type] iter $k upper-level status = $status"
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.TIME_LIMIT, MOI.ITERATION_LIMIT]
        @warn "upper-level not converged at iter $k — stopping"
        break
    end
    if status in [MOI.TIME_LIMIT, MOI.ITERATION_LIMIT]
        @warn "[$FAIR_FUNC/$pshed_type] iter $k upper-level hit $status — using suboptimal incumbent"
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

    timing[:total_iter_s] = get(timing, :integer_warmstart_s, 0.0) +
                            get(timing, :lower_level_total_s, 0.0) +
                            get(timing, :upper_level_total_s, 0.0)
    push!(iter_timings, timing)
    @info @sprintf("[%s/%s] iter %d timings: int=%.1fs lower=%.1fs (primal=%.1fs jac=%.1fs) upper=%.1fs total=%.1fs",
        FAIR_FUNC, pshed_type, k,
        get(timing, :integer_warmstart_s, 0.0),
        get(timing, :lower_level_total_s, 0.0),
        get(timing, :lower_primal_solve_s, 0.0),
        get(timing, :lower_jacobian_loop_s, 0.0),
        get(timing, :upper_level_total_s, 0.0),
        timing[:total_iter_s])
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
relaxed_term = mn_relaxed_final["termination_status"]
relaxed_ok = relaxed_term in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
@info "relaxed multi-period termination: $relaxed_term (converged=$relaxed_ok)"
validation_results["step3"] = Dict(
    "termination_status" => string(relaxed_term),
    "converged"          => relaxed_ok,
)

if relaxed_ok

# ============================================================
# STEP 4: PER-PERIOD RANDOM ROUNDING + AC FEASIBILITY
# ============================================================
print_validation_header("Step 4: Per-period rounding + AC feasibility")

per_period_results = Dict{String,Any}()
mn_rounded = Dict{String,Dict{String,Any}}()  # rounded math per nw_id
mn_rounded_solutions = Dict{String,Dict{String,Any}}()  # rounded MLD solution per nw_id (for plotting)
ac_solutions_by_nw   = Dict{String,Any}()      # raw AC PF result per nw_id (for JLD2 persistence)

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

    ac_solutions_by_nw[nw_id] = Dict(
        "solution"           => get(ac_result_t, "solution", Dict{String,Any}()),
        "termination_status" => string(ac_term_t),
    )

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
    mn_rounded_solutions[nw_id] = mld_rounded_t
    per_period_results["period_$t"] = period_checks
end

validation_results["per_period"] = per_period_results

# ============================================================
# STEP 5: LOAD-SHED HEATMAP + FINAL RESULT + REPORT
# (extracted so it can be re-run standalone in REPL — builds pshed_matrix,
# load_labels, period_labels, period_total, period_max, rounded_objectives)
# ============================================================

include("results_block_mn.jl")

# ============================================================
# STEP 6: PERSIST PER-RUN DATA FOR STANDALONE PLOTTING
# Filename pins (CASE, FAIR_FUNC, pshed_type) so each fair_func × case run
# lands in its own JLD2 and downstream plot scripts can target them by key.
# Reuses pshed_matrix / load_labels / etc. built by results_block_mn.jl.
# ============================================================
using JLD2
# Backfill iter_timings if running Step 6 in isolation against a pre-instrumentation
# session (variable only gets initialized inside the bilevel loop in Step 2).
if !@isdefined(iter_timings)
    @warn "iter_timings not defined — saving empty Vector. (Re-run the full script to capture per-iter timings.)"
    iter_timings = Dict{Symbol,Any}[]
end
jld_path = joinpath(save_dir, "bilevel_mn_$(CASE)_$(FAIR_FUNC)_$(pshed_type).jld2")
JLD2.jldsave(jld_path;
    pshed_matrix             = pshed_matrix,
    pd_ref_matrix            = pd_ref_matrix,
    load_labels              = load_labels,
    period_labels            = period_labels,
    bus_labels               = bus_labels,
    bus_pd_matrix            = bus_pd_matrix,
    bus_pshed_matrix         = bus_pshed_matrix,
    bus_status_matrix        = bus_status_matrix,
    relaxed_pshed_matrix     = relaxed_pshed_matrix,
    relaxed_bus_pshed_matrix = relaxed_bus_pshed_matrix,
    relaxed_bus_status_matrix = relaxed_bus_status_matrix,
    final_fair_weights       = fair_weights,
    final_weight_ids         = final_weight_ids,
    LOAD_SCALE_FACTORS       = LOAD_SCALE_FACTORS,
    PEAK_TIME_COSTS          = PEAK_TIME_COSTS,
    SELECTED_HOURS           = SELECTED_HOURS,
    PEAK_STRESS              = PEAK_STRESS,
    CENTER_AT_NOMINAL        = CENTER_AT_NOMINAL,
    LS_PERCENT               = LS_PERCENT,
    CASE                     = CASE,
    FAIR_FUNC                = FAIR_FUNC,
    pshed_type               = pshed_type,
    N_PERIODS                = N_PERIODS,
    period_total             = period_total,
    period_max               = period_max,
    rounded_objectives       = rounded_objectives,
    relaxed_mn_objective     = mn_relaxed_final["objective"],
    iter_timings             = iter_timings,
    # ---- Raw per-period solution dicts (load/block/switch/branch/bus fields). ----
    # Downstream can read pshed/status/state/pf/qf/w/vr/vi from these directly and
    # derive utilization via rating fields in the math_* snapshots below.
    mn_relaxed_solution_per_period = mn_relaxed_final["solution"]["nw"],
    mn_rounded_solution_per_period = Dict(nw_id => v["solution"] for (nw_id, v) in mn_rounded_solutions),
    mn_ac_solution_per_period      = ac_solutions_by_nw,
    # Static topology + rating metadata (identical across periods).
    math_switch              = math["switch"],
    math_branch              = math["branch"],
    math_bus                 = math["bus"],
    math_load                = math["load"],
    load_block_sets          = lbs,
)
println("Saved bilevel run data → $jld_path")

if !isempty(iter_timings)
    println("\nPer-iter timings (seconds):")
    @printf "  %3s %10s %10s %10s %10s %10s\n" "k" "int" "lower" "primal" "jac_loop" "upper"
    for t in iter_timings
        @printf "  %3d %10.2f %10.2f %10.2f %10.2f %10.2f\n" t[:iter] get(t, :integer_warmstart_s, 0.0) get(t, :lower_level_total_s, 0.0) get(t, :lower_primal_solve_s, 0.0) get(t, :lower_jacobian_loop_s, 0.0) get(t, :upper_level_total_s, 0.0)
    end
    tot_int   = sum(get(t, :integer_warmstart_s, 0.0)  for t in iter_timings)
    tot_lower = sum(get(t, :lower_level_total_s, 0.0)  for t in iter_timings)
    tot_upper = sum(get(t, :upper_level_total_s, 0.0)  for t in iter_timings)
    @printf "  ----------------------------------------------------------------\n"
    @printf "  %3s %10.2f %10.2f %10s %10s %10.2f   total=%.1fs\n" "Σ" tot_int tot_lower "" "" tot_upper (tot_int+tot_lower+tot_upper)
end

println("\nMulti-period validation complete.")

else  # !relaxed_ok — Step 3 did not converge
    @warn "[$FAIR_FUNC/$pshed_type] Step 3 final relaxed multi-period MLD did not converge (status $relaxed_term). Skipping Steps 4–6 (rounding, plots, JLD2 save) to avoid building outputs on a non-converged relaxation."
    abort_path = joinpath(save_dir, "step3_aborted.txt")
    open(abort_path, "w") do io
        println(io, "Step 3 relaxed multi-period MLD did not converge.")
        println(io, "termination_status      = $relaxed_term")
        println(io, "bilevel completed_iters = $(length(all_pshed_lower))")
        println(io, "bilevel last_status     = $last_status")
        println(io, "Steps 4–6 skipped (rounding, results block, JLD2 save).")
    end
    println("Wrote $abort_path")
    println("\nMulti-period validation halted at Step 3.")
end
