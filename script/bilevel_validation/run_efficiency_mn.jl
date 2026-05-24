"""
    Multi-period efficient-operation runner (no bilevel, no DiffOpt).

    The "efficiency" case minimizes total load shed under per-period TOU
    peak-cost weights — no fairness gradient needed, so we skip the bilevel
    loop entirely and solve the multi-period integer MLD directly with Gurobi.

    Outputs are saved in the same JLD2 schema as run_validation_mn.jl so the
    standalone plot scripts (e.g. per_period_norms_mn.jl) work uniformly.
"""

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi
using HSL_jll
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra
using DataFrames
using Dates
using Logging, LoggingExtras
using JLD2
using Plots
using Printf

# Unified 9pt font defaults for every figure in this script.
include(joinpath(@__DIR__, "../figure_defaults.jl"))

const PMD = PowerModelsDistribution

include("validation_utils.jl")

# ============================================================
# CONFIGURATION
# ============================================================
CASE = "motivation_c_good4integer"
case            = "motivation_c_13bus"
CASE_FILE = joinpath(@__DIR__, "../../data/ieee_13_aw_edit/$CASE.dss")
LS_PERCENT      = 0.8
FAIR_FUNC = "efficiency"
pshed_type      = "absolute"

# 2026-05-24 short-period defense follow-up for motivation_c (13-bus).
# T=2: midday plateau + evening peak so peak_time_cost differentiation is
# maximal. Bump to [12, 18, 22] (T=3, peak in middle) if the 2-period run is
# fast.

SELECTED_HOURS    = [12, 18]
N_PERIODS         = length(SELECTED_HOURS)

PEAK_STRESS       = 1.0
# When true, each schedule is divided by its own daily mean before applying
# peak_stress — so the daily-average per-load scale equals PEAK_STRESS exactly
# (1.4× nominal here) and the nameplate pd is the daily mean, matching the
# single-period reference.
CENTER_AT_NOMINAL = true
PERIOD_HOURS      = SELECTED_HOURS
PEAK_TIME_COSTS   = [round(5.0 + 25.0 * exp(-((h - 18)^2) / (2 * 2.5^2)), digits=2)
                           for h in PERIOD_HOURS]
REP_PERIODS = collect(1:N_PERIODS)   # show all periods on the grouped bar

switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT
ipopt_solver   = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver  = Gurobi.Optimizer

save_dir = "results/$(Dates.today())/bilevel_validation_mn/$CASE/$(FAIR_FUNC)_$(pshed_type)"
mkpath(save_dir)
log_file = joinpath(save_dir, "run_efficiency_mn.log")
global_logger(TeeLogger(global_logger(), FileLogger(log_file)))
@info "Logging to $log_file"

# ============================================================
# STEP 1: NETWORK + PROFILED MULTINETWORK
# ============================================================
print_validation_header("Step 1: Network + profiled multinetwork")

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating = switch_rating)

LOAD_SCALE_FACTORS = FairLoadDelivery.aggregate_demand_fraction(math, N_PERIODS;
    hours = SELECTED_HOURS, center_at_nominal = CENTER_AT_NOMINAL) .* PEAK_STRESS

mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS, center_at_nominal = CENTER_AT_NOMINAL)
nw_ids_sorted = sort(collect(keys(mn_data["nw"])), by = x -> parse(Int, x))
mn_new = mn_data   # alias kept so results_block_mn.jl sees the expected name

println("Load profile assignments for $CASE:")
assignment_rows = FairLoadDelivery.profile_assignment_table(math)
for row in assignment_rows
    println("  ", row)
end
@info "N_PERIODS=$N_PERIODS, agg_scales=$LOAD_SCALE_FACTORS, λ=$PEAK_TIME_COSTS, peak_stress=$PEAK_STRESS"

# ============================================================
# STEP 1b: LOAD PROFILE PLOT + ASSIGNMENT TABLE
# Mirrors Hamilton & Aliprantis (PECI 2023) Fig. 2 (3 schedule curves) and
# Table I (per-load schedule + shift). Schedules are shown with peak_stress
# and (optionally) center_at_nominal applied so the displayed magnitude
# matches the demand the MIP actually sees.
# ============================================================
print_validation_header("Step 1b: Load profile + assignment figures")

function _scaled_schedule(sched_idx::Int, shift::Int)
    # schedule_value takes 1-indexed position; SELECTED_HOURS is 0-indexed hour-of-day.
    # Bug pre-downsample: used `t in 1:N_PERIODS` which read schedule positions 1..8
    # (hours 0..7) and plotted them against PERIOD_HOURS, making the profiles appear
    # to peak at the rightmost sample.
    raw = [FairLoadDelivery.schedule_value(sched_idx, shift, h + 1) for h in SELECTED_HOURS]
    norm = CENTER_AT_NOMINAL ? FairLoadDelivery.SCHEDULE_MEANS[sched_idx] : 1.0
    return raw .* (PEAK_STRESS / norm)
end

p_profiles = plot(xlabel = "hour", ylabel = "load scale (× nominal pd)",
    title = "Hamilton & Aliprantis schedules  (peak_stress=$(PEAK_STRESS), centered=$(CENTER_AT_NOMINAL))",
    legend = :topright, lw = 2)
for s in 1:FairLoadDelivery.N_SCHEDULES
    plot!(p_profiles, PERIOD_HOURS, _scaled_schedule(s, 0),
        marker = :circle, markersize = 4, label = "schedule $s")
end
hline!(p_profiles, [1.0], linestyle = :dash, color = :gray, label = "nominal pd")
savefig(p_profiles, joinpath(save_dir, "load_profile_schedules_$case.svg"))
display(p_profiles)
println("Saved schedule plot → ", joinpath(save_dir, "load_profile_schedules_$case.svg"))

# TOU peak-cost profile λ[t] — single line, Gaussian centered at hour 18.
p_costs = plot(PERIOD_HOURS, PEAK_TIME_COSTS,
    marker = :circle, markersize = 4, lw = 2, color = :firebrick, legend = false,
    xlabel = "hour", ylabel = "peak-time cost λ (¢/kWh)",
    title  = "TOU peak-time costs (peak at hr 18, σ=2.5h)")
savefig(p_costs, joinpath(save_dir, "peak_time_costs_$case.svg"))
display(p_costs)
println("Saved peak-time-cost plot → ", joinpath(save_dir, "peak_time_costs_$case.svg"))

# Per-phase schedule overlay — each (load, phase) uses its bus-aware (sched, shift).
n_phase_lines = sum(length(r.phases) for r in assignment_rows)
p_per_load = plot(xlabel = "hour", ylabel = "load scale (× nominal pd)",
    title = "Per-phase profile after assignment ($n_phase_lines phases)",
    legend = false, lw = 1)
for row in assignment_rows, (sched_idx, shift) in row.phases
    plot!(p_per_load, PERIOD_HOURS, _scaled_schedule(sched_idx, shift))
end
hline!(p_per_load, [1.0], linestyle = :dash, color = :gray)
savefig(p_per_load, joinpath(save_dir, "load_profile_per_load_$case.svg"))
display(p_per_load)
println("Saved per-load profile plot → ", joinpath(save_dir, "load_profile_per_load_$case.svg"))

# Assignment table rendered as a plot (sorted by bus then load name).
sorted_rows = sort(assignment_rows; by = r -> (r.bus, r.name))
_phases_str(phases) = join(("(s$s,sh$(sh ≥ 0 ? "+$sh" : "$sh"))" for (s, sh) in phases), ",")
table_strs = [@sprintf("%-8s  bus %s  phases=%s  n_ph=%d  balanced=%s",
                       r.name, r.bus, _phases_str(r.phases), r.n_phases, r.balanced)
              for r in sorted_rows]
p_table = plot(framestyle = :none, legend = false,
    title = "Load → (schedule, shift) assignment",
    xlims = (0, 1), ylims = (0, length(table_strs) + 1))
for (i, str) in enumerate(table_strs)
    annotate!(p_table, 0.02, length(table_strs) + 1 - i,
              text(str, 9, :left))
end
savefig(p_table, joinpath(save_dir, "load_profile_assignments_$case.svg"))
display(p_table)
println("Saved assignment table → ", joinpath(save_dir, "load_profile_assignments_$case.svg"))

# ============================================================
# STEP 2: SOLVE MULTI-PERIOD INTEGER MLD (no DiffOpt)
# ============================================================
print_validation_header("Step 2: Multi-period integer MLD (Gurobi)")

mn_integer = FairLoadDelivery.solve_mn_mc_mld_switch_integer(mn_data, gurobi_solver;
    peak_time_costs = PEAK_TIME_COSTS)
@info "Integer MLD termination: $(mn_integer["termination_status"]), objective=$(round(mn_integer["objective"], digits=3))"

# Synthesize a `mn_relaxed_final`-shaped dict so results_block_mn.jl can report
# the "relaxed multi-period MLD objective" row consistently.
mn_relaxed_final = mn_integer

# ============================================================
# STEP 3: PACKAGE PER-PERIOD SOLUTIONS + AC FEASIBILITY
# ============================================================
print_validation_header("Step 3: Per-period AC feasibility")

mn_rounded_solutions = Dict{String,Dict{String,Any}}()
per_period_results   = Dict{String,Any}()

for (t, nw_id) in enumerate(nw_ids_sorted)
    println("\n  ----- Period $t (nw=$nw_id, λ=$(PEAK_TIME_COSTS[t]), agg_scale=$(round(LOAD_SCALE_FACTORS[t], digits=3))) -----")
    period_checks = Dict{String,Any}()

    sol_t = mn_integer["solution"]["nw"][nw_id]
    obj_t = haskey(mn_integer, "objective") ? mn_integer["objective"] / N_PERIODS : NaN
    mn_rounded_solutions[nw_id] = Dict{String,Any}(
        "solution"           => sol_t,
        "objective"          => obj_t,
        "termination_status" => mn_integer["termination_status"],
    )

    # Per-period rounded math (apply chosen switch states from the integer solution)
    math_t = deepcopy(mn_data["nw"][nw_id])
    imp_diff_model_t = instantiate_mc_model(
        math_t, LinDist3FlowPowerModel,
        FairLoadDelivery.build_mc_mld_shedding_implicit_diff;
        ref_extensions = [FairLoadDelivery.ref_add_rounded_load_blocks!])
    ref_t = imp_diff_model_t.ref[:it][:pmd][:nw][0]

    sw_states_t = Dict(parse(Int, sid) => sw["state"] for (sid, sw) in sol_t["switch"])
    math_t_rounded = update_network(math_t, sw_states_t, ref_t)

    # Voltage + ampacity on the integer MLD solution
    mld_rounded_t = Dict{String,Any}(
        "solution"           => sol_t,
        "termination_status" => mn_integer["termination_status"],
    )
    v_passed_r, v_violations_r, v_summary_r = check_voltage_limits_relaxed(mld_rounded_t, math_t_rounded)
    period_checks["voltage_limits_rounded"] = Dict("passed" => v_passed_r,
        "details" => [string(v) for v in v_violations_r])
    print_check_result("Period $t: voltage limits (integer MLD)", v_passed_r,
        "$(v_summary_r["violations"]) violations / $(v_summary_r["checked"]) checked")

    c_passed_r, c_violations_r, c_summary_r = check_switch_ampacity(mld_rounded_t, math_t_rounded)
    period_checks["current_limits_rounded"] = Dict("passed" => c_passed_r,
        "details" => [string(v) for v in c_violations_r])
    print_check_result("Period $t: switch ampacity (integer MLD)", c_passed_r,
        "$(c_summary_r["violations"]) violations")

    # AC PF
    math_ac_t = ac_network_update(math_t_rounded, ref_t; mld_solution = mld_rounded_t)
    ac_result_t = PowerModelsDistribution.solve_mc_pf(math_ac_t, IVRUPowerModel, ipopt_solver)
    ac_term_t   = ac_result_t["termination_status"]
    ac_ok_t     = ac_term_t in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED)
    period_checks["ac_convergence"] = Dict("passed" => ac_ok_t, "details" => ["status: $ac_term_t"])
    print_check_result("Period $t: AC PF converged", ac_ok_t, "status: $ac_term_t")

    if ac_ok_t && haskey(ac_result_t, "solution")
        v_passed_ac, v_violations_ac, v_summary_ac = check_voltage_limits_ac(ac_result_t, math_ac_t)
        period_checks["voltage_limits_ac"] = Dict("passed" => v_passed_ac,
            "details" => [string(v) for v in v_violations_ac])
        print_check_result("Period $t: voltage limits (AC PF)", v_passed_ac,
            "$(v_summary_ac["violations"]) / $(v_summary_ac["checked"])")
    end

    per_period_results["period_$t"] = period_checks
end

validation_results = Dict{String,Any}(
    "case"            => CASE,
    "fair_func"       => FAIR_FUNC,
    "n_periods"       => N_PERIODS,
    "period_hours"    => PERIOD_HOURS,
    "load_scales"     => LOAD_SCALE_FACTORS,
    "peak_stress"     => PEAK_STRESS,
    "peak_time_costs" => PEAK_TIME_COSTS,
    "pshed_type"      => pshed_type,
    "per_period"      => per_period_results,
)

# ============================================================
# STEP 4: HEATMAP + REPORT (shared with bilevel runner)
# ============================================================
include("results_block_mn.jl")

# ============================================================
# STEP 5: PERSIST PER-RUN DATA (same schema as run_validation_mn.jl Step 6)
# ============================================================
jld_path = joinpath(save_dir, "bilevel_mn_$(CASE)_$(FAIR_FUNC)_$(pshed_type).jld2")
JLD2.jldsave(jld_path;
    pshed_matrix         = pshed_matrix,
    load_labels          = load_labels,
    period_labels        = period_labels,
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
)
println("Saved efficiency run data → $jld_path")

println("\nEfficiency multi-period run complete.")
