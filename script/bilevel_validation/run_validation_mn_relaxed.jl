"""
    Multiperiod Bilevel FLDP — Relaxed-only Runner

    Trimmed sibling of run_validation_mn.jl. Runs the bilevel loop (lower-level
    DiffOpt → upper-level fairness step) and the Step 3 final multi-period
    relaxed MLD solve, but SKIPS:
      * per-period random rounding to a radial topology (Step 4 of the
        full runner)
      * post-rounding AC feasibility / voltage / ampacity checks
      * the results_block_mn.jl include (it depends on rounded outputs)

    What you get out:
      * mn_relaxed_final  — the multi-period relaxed MLD solution
      * a small JLD2 with relaxed_pshed_matrix + metadata, sufficient to
        plug into post_hoc_fairness_pareto.jl (relaxed-side bilevel marker)
        and the relaxed heatmap script.

    Usage:
        julia --project=. script/bilevel_validation/run_validation_mn_relaxed.jl
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
CASE = "case6_unbalanced_switch_more_meshed_good4integer"
 #CASE = "motivation_c_good4integer"
case = "6_bus" #"13_bus"#"6_bus"
#critical_load = ["611"]
CASE_FILE = joinpath(@__DIR__,"../../data/pmd_opendss/$CASE.dss")
#CASE_FILE = joinpath(@__DIR__, "../../data/ieee_13_aw_edit/$CASE.dss")
LS_PERCENT = 0.8
ITERATIONS = 20
FAIR_FUNC = "efficiciency"  # "min_max", "palma", or "efficiency"
pshed_type = "absolute"  # "absolute" or "proportional"

# Multi-period setup: per-load Hamilton & Aliprantis (PECI 2023) schedules.
 SELECTED_HOURS    = collect(0:23)   # T=24 full diurnal cycle
 #SELECTED_HOURS    = [4, 18, 8]

 N_PERIODS         = length(SELECTED_HOURS)
PEAK_STRESS       = 1.0
CENTER_AT_NOMINAL = true
PERIOD_HOURS      = SELECTED_HOURS
 PEAK_TIME_COSTS   = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                           for h in PERIOD_HOURS]

switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT

# Solvers
ipopt_solver  = optimizer_with_attributes(Ipopt.Optimizer,
    "max_iter"        => 30_000,
    "acceptable_tol"  => 1e-4,
    "acceptable_iter" => 50,
    "print_level"     => 0)
gurobi_solver = Gurobi.Optimizer

# Sibling folder of bilevel_validation_mn/ so the relaxed-only JLD2 doesn't
# collide with the full-pipeline one for the same (CASE, FAIR_FUNC, pshed).
save_dir = "results/$(Dates.today())/bilevel_validation_mn_relaxed/$CASE/$(FAIR_FUNC)_$(pshed_type)"
mkpath(save_dir)

log_file = joinpath(save_dir, "run_validation_mn_relaxed.log")
global_logger(TeeLogger(global_logger(), FileLogger(log_file)))
@info "Logging to $log_file"

# ============================================================
# STEP 1: NETWORK + MULTINETWORK SETUP
# ============================================================
print_validation_header("Step 1: Network + Multinetwork Setup")

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating=switch_rating)#, critical_load=critical_load

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

fair_weights_init = Float64[load["weight"] for (_, load) in math["load"]]
fair_weights = copy(fair_weights_init)

iteration_label_consistent = true
all_pshed_lower = Float64[]
all_pshed_upper = Float64[]
iter_timings = Dict{Symbol,Any}[]
last_status = MOI.OPTIMIZE_NOT_CALLED
final_weight_ids = Int[]
final_pshed_nw_ids = Tuple[]

for k in 1:ITERATIONS
    global fair_weights, iteration_label_consistent, last_status, final_weight_ids, final_pshed_nw_ids
    println("\n  --- Iteration $k ---")

    timing = Dict{Symbol,Any}(:iter => k)

    local dpshed, pshed_val, pshed_nw_ids, weight_vals, weight_ids, refs
    try
        # Switch-topology integer warm-start (skipped for Palma — see
        # run_validation_mn.jl for the CC σ-degeneracy reasoning).
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

    pd_all = Float64[sum(refs[nw][:load][lid]["pd"]) for (nw, lid) in pshed_nw_ids]

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
                pshed_new, fair_weight_vals, status = lin_palma_reformulated(
                    dpshed, pshed_val, weight_vals, pd_all;
                    critical_ids=critical_id, weight_ids=weight_ids,
                    peak_time_costs=PEAK_TIME_COSTS, n_loads=n_loads,
                    time_limit=60*10, timings=upper_timings)
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

if !relaxed_ok
    @warn "[$FAIR_FUNC/$pshed_type] Step 3 relaxed multi-period MLD did not converge (status $relaxed_term). Skipping results extraction + JLD2 save."
    abort_path = joinpath(save_dir, "step3_aborted.txt")
    open(abort_path, "w") do io
        println(io, "Step 3 relaxed multi-period MLD did not converge.")
        println(io, "termination_status      = $relaxed_term")
        println(io, "bilevel completed_iters = $(length(all_pshed_lower))")
        println(io, "bilevel last_status     = $last_status")
    end
    println("Wrote $abort_path")
    println("\nRelaxed-only run halted at Step 3.")
else

# ============================================================
# STEP 4: BUILD RELAXED PER-LOAD / PER-BUS MATRICES
# (inlined from results_block_mn.jl, relaxed-only — no rounded path)
# ============================================================
print_validation_header("Step 4: Extract relaxed pshed matrices")

ref_load_ids  = sort(collect(keys(mn_new["nw"][nw_ids_sorted[1]]["load"])), by=x->parse(Int, x))
load_labels   = [mn_new["nw"][nw_ids_sorted[1]]["load"][lid]["name"] for lid in ref_load_ids]
period_labels = [string(t) for t in 1:N_PERIODS]

relaxed_pshed_matrix = fill(NaN, N_PERIODS, length(ref_load_ids))
for (t, nw_id) in enumerate(nw_ids_sorted)
    haskey(mn_relaxed_final["solution"]["nw"], nw_id) || continue
    sol_t = mn_relaxed_final["solution"]["nw"][nw_id]
    haskey(sol_t, "load") || continue
    for (j, lid) in enumerate(ref_load_ids)
        if haskey(sol_t["load"], lid) && haskey(sol_t["load"][lid], "pshed")
            relaxed_pshed_matrix[t, j] = sum(sol_t["load"][lid]["pshed"])
        end
    end
end

math_ref = mn_new["nw"][nw_ids_sorted[1]]
bus_name_map = FairLoadDelivery.build_bus_name_maps(math_ref)
load_bus_set = Set(math_ref["load"][lid]["load_bus"] for lid in ref_load_ids)
all_bus_ids = sort(collect(load_bus_set))
bus_labels = [get(bus_name_map, bid, "bus_$bid") for bid in all_bus_ids]

bus_col = Dict(bid => k for (k, bid) in enumerate(all_bus_ids))
load_to_bus_col = [bus_col[math_ref["load"][lid]["load_bus"]] for lid in ref_load_ids]

pd_ref_matrix            = zeros(N_PERIODS, length(ref_load_ids))
bus_pd_matrix            = zeros(N_PERIODS, length(all_bus_ids))
relaxed_bus_pshed_matrix = zeros(N_PERIODS, length(all_bus_ids))
for (t, nw_id) in enumerate(nw_ids_sorted)
    nw_data = mn_new["nw"][nw_id]
    for (j, lid) in enumerate(ref_load_ids)
        pd_total = sum(nw_data["load"][lid]["pd"])
        pd_ref_matrix[t, j] = pd_total
        bus_pd_matrix[t, load_to_bus_col[j]] += pd_total
        vr = relaxed_pshed_matrix[t, j]
        relaxed_bus_pshed_matrix[t, load_to_bus_col[j]] += isnan(vr) ? 0.0 : vr
    end
end

relaxed_bus_status_matrix = fill(NaN, N_PERIODS, length(all_bus_ids))
for t in 1:N_PERIODS, b in 1:length(all_bus_ids)
    if bus_pd_matrix[t, b] > 1e-9
        relaxed_bus_status_matrix[t, b] = 1.0 - relaxed_bus_pshed_matrix[t, b] / bus_pd_matrix[t, b]
    end
end

valid_mask = .!isnan.(relaxed_pshed_matrix)
period_total = [sum(relaxed_pshed_matrix[t, j] for j in 1:length(ref_load_ids) if valid_mask[t, j]; init=0.0) for t in 1:N_PERIODS]
period_max   = [maximum(relaxed_pshed_matrix[t, j] for j in 1:length(ref_load_ids) if valid_mask[t, j]; init=0.0) for t in 1:N_PERIODS]

total_shed_all = sum(filter(!isnan, relaxed_pshed_matrix))
println("\n  Relaxed result (no rounding):")
println("    Total shed (Σ over all loads × periods) = $(round(total_shed_all, digits=3)) kW")
println("    Relaxed multi-period MLD objective       = $(round(mn_relaxed_final["objective"], digits=3))")
println("\n  Per-period summary:")
println("    " * rpad("t", 4) * rpad("scale", 8) * rpad("λ", 8) *
              rpad("total shed", 14) * "max shed")
for t in 1:N_PERIODS
    println("    " * rpad(string(t), 4) *
                    rpad(string(LOAD_SCALE_FACTORS[t]), 8) *
                    rpad(string(PEAK_TIME_COSTS[t]), 8) *
                    rpad(string(round(period_total[t], digits=3)), 14) *
                    string(round(period_max[t], digits=3)))
end

validation_results["final"] = Dict(
    "total_shed_all_periods" => total_shed_all,
    "period_total_shed"      => period_total,
    "period_max_shed"        => period_max,
    "relaxed_mn_objective"   => mn_relaxed_final["objective"],
)

# ============================================================
# STEP 5: PERSIST RELAXED-ONLY JLD2
# ============================================================
using JLD2
if !@isdefined(iter_timings)
    @warn "iter_timings not defined — saving empty Vector."
    iter_timings = Dict{Symbol,Any}[]
end
jld_path = joinpath(save_dir, "bilevel_mn_relaxed_$(CASE)_$(FAIR_FUNC)_$(pshed_type).jld2")
JLD2.jldsave(jld_path;
    relaxed_pshed_matrix      = relaxed_pshed_matrix,
    relaxed_bus_pshed_matrix  = relaxed_bus_pshed_matrix,
    relaxed_bus_status_matrix = relaxed_bus_status_matrix,
    pd_ref_matrix             = pd_ref_matrix,
    bus_pd_matrix             = bus_pd_matrix,
    load_labels               = load_labels,
    period_labels             = period_labels,
    bus_labels                = bus_labels,
    final_fair_weights        = fair_weights,
    final_weight_ids          = final_weight_ids,
    LOAD_SCALE_FACTORS        = LOAD_SCALE_FACTORS,
    PEAK_TIME_COSTS           = PEAK_TIME_COSTS,
    SELECTED_HOURS            = SELECTED_HOURS,
    PEAK_STRESS               = PEAK_STRESS,
    CENTER_AT_NOMINAL         = CENTER_AT_NOMINAL,
    LS_PERCENT                = LS_PERCENT,
    CASE                      = CASE,
    FAIR_FUNC                 = FAIR_FUNC,
    pshed_type                = pshed_type,
    N_PERIODS                 = N_PERIODS,
    relaxed_mn_objective      = mn_relaxed_final["objective"],
    iter_timings              = iter_timings,
)
println("Saved relaxed-only bilevel run data → $jld_path")

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

println("\nRelaxed-only multi-period run complete.")
end
