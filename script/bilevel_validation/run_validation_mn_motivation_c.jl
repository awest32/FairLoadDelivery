"""
    Multiperiod Bilevel FLDP Validation Runner — motivation_c (defense final)
    =========================================================================

    Sibling of run_validation_mn.jl, tailored for the 13-bus motivation_c case.
    Supports both Palma (weak-CC MIQCP or formal-CC indicator MILP) and min_max
    upper-level objectives; toggle FAIR_FUNC via environment variable.

    T=8. T·N²=2048 binaries for the Palma upper-level. Toggle USE_WEAK_CC to
    pick between weak-CC bilinear MIQCP and formal-CC indicator MILP.

    Expected wall time:
      - min_max: <1 hr (Ipopt upper-level QP, integer warm-start each iter)
      - palma: ~3.5 hr (20 iters × 10-min Gurobi TimeLimit per upper-level
        solve; prior weak-CC runs hit TimeLimit on every iter)

    Usage:
        # default = palma
        julia --project=. script/bilevel_validation/run_validation_mn_motivation_c.jl

        # min_max:
        \$env:FAIR_FUNC = "min_max"; julia --project=. script/bilevel_validation/run_validation_mn_motivation_c.jl
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

PMD = PowerModelsDistribution

include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

# ============================================================
# CONFIGURATION
# ============================================================
CASE = "motivation_c_good4integer"
case = "13_bus"

CASE_FILE = joinpath(@__DIR__, "../../data/ieee_13_aw_edit/motivation_c_good4integer.dss")
LS_PERCENT = 0.8
# 2026-05-19 defense run: cut to 5 after two prior min_max attempts (killed at
# iters 12 and 9) showed Σ pshed convergence by iter ~5, with iters 6+ alternating
# Δ=0 (no upper-level weight movement) and DiffOpt inertia-correction slowdowns
# pushing per-iter wall time from ~15 min to ~3 hr. Iters 11+ hit DiffOpt's
# zero-Jacobian fallback (load_shed_as_parameter.jl:519-545). 5 captures the
# meaningful bilevel convergence; the trajectory log from the killed iters10
# min_max run documents what happens beyond iter 5.
ITERATIONS = 5
# Allow FAIR_FUNC to be overridden by env var so the same script can run both
# "palma" and "min_max" defense cases without re-editing the source.
FAIR_FUNC = get(ENV, "FAIR_FUNC", "palma")
@assert FAIR_FUNC in ("palma", "min_max") "FAIR_FUNC=\"$FAIR_FUNC\" not supported here; use \"palma\" or \"min_max\""
pshed_type = "absolute"
N_ROUNDS = 1
N_BERNOULLI_SAMPLES = 2000

# T=2 quick-look — midday plateau + evening peak. The 2nd period IS the peak
# so peak_time_cost differentiation is maximal across the pair. Use [12, 18, 22]
# (3 periods, peak in middle) if T=2 runs quickly.
SELECTED_HOURS    = [4, 18, 8]   # T=3: peak in the middle position so plots show off-peak → peak → off-peak. λ=[5.0, 30.0, 5.01]. Prior: [12, 18] for T=2.
# Defense final: weak-CC bilinear MIQCP (matches prior reference run).
USE_WEAK_CC       = true
# Per-iter Gurobi TimeLimit for Palma upper-level (seconds). 10 min × 20 iters
# ≈ 3.5 hr worst case; prior 5-min run shed all sheddable loads at TimeLimit.
PALMA_TIME_LIMIT  = 60 * 10
N_PERIODS         = length(SELECTED_HOURS)
PEAK_STRESS       = 1.0
CENTER_AT_NOMINAL = true
PERIOD_HOURS      = SELECTED_HOURS
PEAK_TIME_COSTS   = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                           for h in PERIOD_HOURS]
REP_PERIODS = collect(1:N_PERIODS)   # show all periods on the grouped bar (T=2 or T=3)

switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT

# Solvers
ipopt_solver  = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = Gurobi.Optimizer

# 2026-05-24 short-period defense follow-up: drop variant suffix from the dir
# layout so post-hoc scripts (per_block_fairness_mn, loadshed_heatmap_mn,
# loadshed_grouped_mn) find the JLD2 via their default <case>/<fair>_<pshed>/
# lookup — matches the bus-case convention. Retain VARIANT_TAG for the
# in-JLD2 "palma_variant" label only.
VARIANT_TAG = USE_WEAK_CC ? "weakcc" : "formalcc"
save_dir = "results/$(Dates.today())/bilevel_validation_mn/$(CASE)/$(FAIR_FUNC)_$(pshed_type)"
mkpath(save_dir)

log_file = joinpath(save_dir, "run_validation_mn.log")
global_logger(TeeLogger(global_logger(), FileLogger(log_file)))
@info "Logging to $log_file (FAIR_FUNC=$FAIR_FUNC, VARIANT_TAG=$VARIANT_TAG, PALMA_TIME_LIMIT=$PALMA_TIME_LIMIT)"

# ============================================================
# STEP 1: NETWORK + MULTINETWORK SETUP
# ============================================================
print_validation_header("Step 1: Network + Multinetwork Setup")

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating=switch_rating)

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
    "palma_variant"    => FAIR_FUNC == "palma" ? (USE_WEAK_CC ? "weak_cc" : "formal_cc_indicator") : "n/a",
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

fair_weights_init = Float64[load["weight"] for (_, load) in math["load"]]
fair_weights = copy(fair_weights_init)

iteration_label_consistent = true
all_pshed_lower = Float64[]
all_pshed_upper = Float64[]
last_status = MOI.OPTIMIZE_NOT_CALLED
final_weight_ids = Int[]
final_pshed_nw_ids = Tuple[]
iter_timings = Dict{Symbol,Any}[]

for k in 1:ITERATIONS
    global fair_weights, iteration_label_consistent, last_status, final_weight_ids, final_pshed_nw_ids
    println("\n  --- Iteration $k ---")

    timing = Dict{Symbol,Any}(:iter => k)

    local dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs
    try
        # Mirror run_validation_mn.jl: integer warm-start for non-Palma objectives
        # to fix switch topology per period, then DiffOpt lower level. Skipped
        # for Palma per [[project_palma_skip_integer_warmstart]] — topology-fix
        # collapses pshed to {0,pd} and breaks Charnes-Cooper σ*bot=1.
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
        # tuple targets under this Julia (1.12.1) — the macro's let-scope holds
        # the assignment, leaving the outer locals undefined and the next access
        # triggers UndefVarError(:dpshed). Split the call out so the assignment
        # lands in the outer scope.
        _t_lower_start = time()
        (dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs) =
            lower_level_soln_mn(mn_new, fair_weights, k; timings=lower_timings)
        t_lower = time() - _t_lower_start
        timing[:lower_level_total_s] = t_lower
        merge!(timing, Dict(Symbol("lower_$(k2)") => v for (k2, v) in lower_timings))
    catch err
        @warn "[$FAIR_FUNC/$pshed_type] iter $k lower-level failed ($err) — stopping bilevel"
        timing[:fail_phase] = "lower_level"
        push!(iter_timings, timing)
        break
    end
    n_loads = length(weight_ids)

    pd_all = Float64[sum(refs[nw][:load][lid]["pd"]) for (nw, lid) in pshed_nw_ids]

    local pshed_new, fair_weight_vals, status
    upper_timings = Dict{Symbol,Any}()
    try
        t_upper = @elapsed begin
            if FAIR_FUNC == "palma"
                # Defense final: weak-CC MIQCP with 10-min Gurobi TimeLimit per iter.
                pshed_new, fair_weight_vals, status = lin_palma_reformulated(
                    dpshed, pshed_val, weight_vals, pd_all;
                    critical_ids=critical_id, weight_ids=weight_ids,
                    peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads,
                    use_weak_cc=USE_WEAK_CC,
                    time_limit=PALMA_TIME_LIMIT, timings=upper_timings)
            else  # "min_max"
                pshed_new, fair_weight_vals, status = min_max_load_shed(
                    dpshed, pshed_val, weight_vals;
                    critical_ids=critical_id, weight_ids=weight_ids,
                    peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads,
                    pd=pd_all, pshed_type=pshed_type, timings=upper_timings)
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
mn_rounded = Dict{String,Dict{String,Any}}()
mn_rounded_solutions = Dict{String,Dict{String,Any}}()

for (t, nw_id) in enumerate(nw_ids_sorted)
    println("\n  ----- Period $t (nw=$nw_id, scale=$(LOAD_SCALE_FACTORS[t]), λ=$(PEAK_TIME_COSTS[t])) -----")
    period_checks = Dict{String,Any}()

    math_t = deepcopy(mn_new["nw"][nw_id])

    imp_diff_model_t = instantiate_mc_model(
        math_t, LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])
    ref_t = imp_diff_model_t.ref[:it][:pmd][:nw][0]

    relaxed_t = mn_relaxed_final["solution"]["nw"][nw_id]
    switch_states = Dict(parse(Int, sid) => sw["state"]   for (sid, sw)  in relaxed_t["switch"])
    block_status  = Dict(parse(Int, bid) => bl["status"] for (bid, bl)  in relaxed_t["block"])

    rng = 100 + t
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

    math_t_rounded = update_network(math_t, sw_radial, ref_t)

    mld_rounded_t = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(math_t_rounded, gurobi_solver)
    rounded_term  = mld_rounded_t["termination_status"]
    rounded_ok    = (rounded_term == MOI.OPTIMAL || rounded_term == MOI.LOCALLY_SOLVED || rounded_term == MOI.ALMOST_LOCALLY_SOLVED)
    period_checks["rounded_mld_converged"] = Dict("passed" => rounded_ok, "details" => ["status: $rounded_term"])
    print_check_result("Period $t: rounded MLD converged", rounded_ok, "status: $rounded_term")
    if !rounded_ok
        per_period_results["period_$t"] = period_checks
        continue
    end

    v_passed_r, v_violations_r, v_summary_r = check_voltage_limits_relaxed(mld_rounded_t, math_t_rounded)
    period_checks["voltage_limits_rounded"] = Dict("passed" => v_passed_r, "details" => [string(v) for v in v_violations_r])
    print_check_result("Period $t: voltage limits (rounded)", v_passed_r, "$(v_summary_r["violations"]) violations / $(v_summary_r["checked"]) checked")

    c_passed_r, c_violations_r, c_summary_r = check_switch_ampacity(mld_rounded_t, math_t_rounded)
    period_checks["current_limits_rounded"] = Dict("passed" => c_passed_r, "details" => [string(v) for v in c_violations_r])
    print_check_result("Period $t: switch ampacity (rounded)", c_passed_r, "$(c_summary_r["violations"]) violations")

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
    mn_rounded_solutions[nw_id] = mld_rounded_t
    per_period_results["period_$t"] = period_checks
end

validation_results["per_period"] = per_period_results

# ============================================================
# STEP 5: LOAD-SHED HEATMAP + FINAL RESULT + REPORT
# ============================================================
include("results_block_mn.jl")

# ============================================================
# STEP 6: PERSIST PER-RUN DATA FOR STANDALONE PLOTTING
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
    pshed_matrix         = pshed_matrix,
    pd_ref_matrix        = pd_ref_matrix,
    load_labels          = load_labels,
    period_labels        = period_labels,
    bus_labels           = bus_labels,
    bus_pd_matrix        = bus_pd_matrix,
    bus_pshed_matrix     = bus_pshed_matrix,
    bus_status_matrix    = bus_status_matrix,
    LOAD_SCALE_FACTORS   = LOAD_SCALE_FACTORS,
    PEAK_TIME_COSTS      = PEAK_TIME_COSTS,
    CASE                 = CASE,
    FAIR_FUNC            = FAIR_FUNC,
    pshed_type           = pshed_type,
    N_PERIODS            = N_PERIODS,
    period_total         = period_total,
    period_max           = period_max,
    rounded_objectives   = rounded_objectives,
    relaxed_mn_objective = mn_relaxed_final["objective"],
    palma_variant        = FAIR_FUNC == "palma" ? (USE_WEAK_CC ? "weak_cc" : "formal_cc_indicator") : "n/a",
    iter_timings         = iter_timings,
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

println("\nMulti-period validation complete (motivation_c / $FAIR_FUNC / $VARIANT_TAG).")

else
    @warn "[$FAIR_FUNC/$pshed_type] Step 3 final relaxed multi-period MLD did not converge (status $relaxed_term). Skipping Steps 4–6."
    abort_path = joinpath(save_dir, "step3_aborted.txt")
    open(abort_path, "w") do io
        println(io, "Step 3 relaxed multi-period MLD did not converge.")
        println(io, "termination_status      = $relaxed_term")
        println(io, "bilevel completed_iters = $(length(all_pshed_lower))")
        println(io, "bilevel last_status     = $last_status")
        println(io, "Steps 4–6 skipped.")
    end
    println("Wrote $abort_path")
    println("\nMulti-period validation halted at Step 3.")
end
