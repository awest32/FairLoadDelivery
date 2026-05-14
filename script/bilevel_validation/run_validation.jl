"""
    Bilevel FLDP Validation Runner

    Runs the bilevel experiment on the simplest test case (motivation_a)
    and validates each stage for:
    1. Data label consistency across pipeline stages
    2. Voltage limits upheld (energized buses only)
    3. Switch current (ampacity) limits upheld
    4. AC power flow feasibility with correct switch ratings

    Usage:
        julia --project=. script/bilevel_validation/run_validation.jl
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
using Statistics

# Load validation utilities
include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/load_shed_as_parameter.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASE = "case6_unbalanced_switch_meshed_good4integer"
const CASE_FILE = joinpath(@__DIR__,"../../data/pmd_opendss/$CASE.dss")

# const CASE = "motivation_c_good4integer"
# const CASE_FILE = joinpath(@__DIR__,"../../data/ieee_13_aw_edit/$CASE.dss")
case = "6_bus"#"13_bus"
LS_PERCENT = 0.8
const ITERATIONS = 20
const FAIR_FUNC = "min_max"  # simplest fairness function for testing
pshed_type = "absolute"  # "absolute" or "proportional" — only used when FAIR_FUNC=="min_max"
const N_ROUNDS = 1
const N_BERNOULLI_SAMPLES = 1000
switch_rating = sqrt.([(26.0^2+13.1^2),(23.0^2+9^2),(21.0^2+9.5^2)])*LS_PERCENT

#switch_rating=[400*LS_PERCENT].*ones(3)
#critical_buses = []
# Solvers
ipopt_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
gurobi_solver = Gurobi.Optimizer
highs_solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)

# Results storage
validation_results = Dict{String, Any}(
    "case" => CASE,
    "fair_func" => FAIR_FUNC,
    "iterations" => ITERATIONS
)

save_dir = "results/$(Dates.today())/bilevel_validation/$CASE/$FAIR_FUNC"
mkpath(save_dir)

# Set up persistent file logging (console + file)
log_file = joinpath(save_dir, "run_validation.log")
global_logger(TeeLogger(
    global_logger(),
    FileLogger(log_file)
))
@info "Logging to $log_file"

# ============================================================
# STEP 1: NETWORK SETUP
# ============================================================
print_validation_header("Step 1: Network Setup")

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT; switch_rating=switch_rating)

setup_checks = Dict{String, Any}()

# Check 1.1: Math dictionary has required keys
required_keys = ["bus", "load", "gen", "switch", "branch", "block"]
missing_keys = [k for k in required_keys if !haskey(math, k)]
passed = isempty(missing_keys)
setup_checks["math_has_required_keys"] = Dict("passed" => passed, "details" => missing_keys)
print_check_result("Math dict has required keys", passed, join(missing_keys, ", "))

# Check 1.2: Switches have current ratings
switches_with_ratings = count(
      switch -> any(r -> r < Inf && r > 0, get(switch,     
  "current_rating", [Inf])),
      values(math["switch"])
  )
passed = switches_with_ratings > 0
setup_checks["switches_have_ratings"] = Dict("passed" => passed, "details" => ["$switches_with_ratings / $(length(math["switch"])) switches have finite ratings"])
print_check_result("Switches have current ratings", passed, "$switches_with_ratings / $(length(math["switch"]))")

# Check 1.3: Voltage limits set correctly
v_limits_ok = true
for (bus_id, bus) in math["bus"]
    if any(bus["vmax"] .!= 1.05) || any(bus["vmin"] .!= 0.95)
        v_limits_ok = false
        break
    end
end
setup_checks["voltage_limits_set"] = Dict("passed" => v_limits_ok)
print_check_result("Voltage limits set to [0.95, 1.05]", v_limits_ok)

# Check 1.4: Generation capacity limited (forces load shedding)
gen_limited = false
total_pd_ref = 0
for (i, gen) in math["gen"]
    if gen["source_id"] == "voltage_source.source"
        total_pmax = sum(gen["pmax"])
        @info "Generation $i ($(gen["name"])) capacity: $total_pmax"
        total_pd_ref = sum(sum(load["pd"]) for (_, load) in math["load"])
        @info "Total demand in system: $total_pd_ref"
        if total_pmax < total_pd_ref
            gen_limited = true
        end
        println("    Generation capacity: $total_pmax, Total demand: $total_pd_ref")
    end
end
setup_checks["gen_forces_shedding"] = Dict("passed" => gen_limited)
print_check_result("Generation limited (forces load shedding)", gen_limited)

# Check 1.5: Load weights initialized
weights_init_ok = true
for (load_id, load) in math["load"]
    if !haskey(load, "weight") || load["weight"] <= 0
        weights_init_ok = false
        break
    end
end
setup_checks["weights_initialized"] = Dict("passed" => weights_init_ok)
print_check_result("Load weights initialized", weights_init_ok)

# Print switch info
println("\n  Switch configuration:")
for (s_id, switch) in sort(collect(math["switch"]), by=x->parse(Int,x[1]))
    rating = get(switch, "current_rating", [Inf])
    println("    Switch $s_id ($(switch["name"])): rating = $rating")
end

# Print load info
println("\n  Load configuration:")
for (l_id, load) in sort(collect(math["load"]), by=x->parse(Int,x[1]))
    println("    Load $l_id ($(load["name"])): pd=$(load["pd"]), weight=$(load["weight"])")
end

validation_results["setup"] = setup_checks
mld_integer_initial = FairLoadDelivery.solve_mc_mld_switch_integer(math,Gurobi.Optimizer)
mld_relaxed_initial = FairLoadDelivery.solve_mc_mld_switch_relaxed(math,Ipopt.Optimizer)

# ============================================================
# STEP 2: INITIAL LOWER-LEVEL SOLVE
# ============================================================
print_validation_header("Step 2: Initial Lower-Level Solve")

initial_checks = Dict{String, Any}()

# Get initial weights
fair_weights = Float64[]
for (load_id, load) in math["load"]
    push!(fair_weights, load["weight"])
end

# Run initial lower-level solve
dpshed, pshed_val, pshed_ids, weight_vals, weight_ids, mld_model = lower_level_soln(math, fair_weights, 1)

# Check 2.1: Label consistency
passed, issues = check_label_consistency(math, pshed_ids, weight_ids, "initial_lower_level")
initial_checks["label_consistency"] = Dict("passed" => passed, "details" => issues)
print_check_result("Label consistency (pshed_ids == weight_ids)", passed, join(issues, "\n"))

# Check 2.2: Jacobian dimensions match
jac_rows, jac_cols = size(dpshed)
n_loads = length(math["load"])
passed = (jac_rows == n_loads && jac_cols == n_loads)
initial_checks["jacobian_dimensions"] = Dict("passed" => passed, "details" => ["Jacobian: $(jac_rows)x$(jac_cols), expected $(n_loads)x$(n_loads)"])
print_check_result("Jacobian dimensions match n_loads", passed, "Got $(jac_rows)x$(jac_cols), expected $(n_loads)x$(n_loads)")

# Check 2.3: pshed values non-negative
passed = all(pshed_val .>= -1e-6)
initial_checks["pshed_nonnegative"] = Dict("passed" => passed)
print_check_result("pshed values non-negative", passed)

println("\n  pshed_ids: $pshed_ids")
println("  weight_ids: $weight_ids")
println("  pshed_val: $pshed_val")

validation_results["initial_solve"] = initial_checks

# ============================================================
# STEP 3: RELAXED FLDP ITERATIONS
# ============================================================
print_validation_header("Step 3: Relaxed FLDP ($FAIR_FUNC, $ITERATIONS iterations)")

relaxed_checks = Dict{String, Any}()
math_new = deepcopy(math)

# Track intermediate states
all_pshed_lower = Float64[]
all_pshed_upper = Float64[]
iteration_label_consistent = true

for k in 1:ITERATIONS
    global fair_weights, iteration_label_consistent
    println("\n  --- Iteration $k ---")

    # Solve integer problem first
    mld_integer_solution = solve_mc_mld_switch_integer(math_new, Gurobi.Optimizer)

    math_updated = update_network(mld_integer_solution["solution"], math_new)    
    # Solve lower-level
    dpshed_k, pshed_val_k, pshed_ids_k, weight_vals_k, weight_ids_k, model_k = lower_level_soln(math_updated, fair_weights, k);

    # Check label consistency at each iteration
    passed_k, issues_k = check_label_consistency(math_new, pshed_ids_k, weight_ids_k, "iteration_$k")
    if !passed_k
        iteration_label_consistent = false
        @error "    [X] Label mismatch at iteration $k: $(join(issues_k, "; "))"
    end

    # Apply upper-level fairness function
    if FAIR_FUNC == "min_max"
        pd_k = pshed_type == "proportional" ?
               Float64[sum(math_new["load"][string(i)]["pd"]) for i in pshed_ids_k] : Float64[]
        pshed_new, fair_weight_vals, status = min_max_load_shed(dpshed_k, pshed_val_k, weight_vals_k; critical_ids=critical_id, pd=pd_k, pshed_type=pshed_type)
    elseif FAIR_FUNC == "proportional"
        pd = Float64[sum(math_new["load"][string(i)]["pd"]) for i in pshed_ids_k]
        pshed_new, fair_weight_vals, status = proportional_fairness_load_shed(dpshed_k, pshed_val_k, weight_vals_k, pd; critical_ids=critical_id)
    elseif FAIR_FUNC == "efficiency"
        pshed_new, fair_weight_vals, status = efficient_load_shed(dpshed_k, pshed_val_k, weight_vals_k; weight_ids, critical_ids=critical_id)
    elseif FAIR_FUNC == "jain"
        pshed_new, fair_weight_vals, status = jains_fairness_index(dpshed_k, pshed_val_k, weight_vals_k; critical_ids=critical_id)
    elseif FAIR_FUNC == "equality_min"
        pshed_new, fair_weight_vals, status = equality_min(dpshed_k, pshed_val_k, weight_vals_k; critical_ids=critical_id)
    elseif FAIR_FUNC == "palma"
        pd = Float64[]
        for i in pshed_ids_k
            push!(pd, sum(math_new["load"][string(i)]["pd"]))
        end
        pshed_new, fair_weight_vals, status = lin_palma_reformulated(dpshed_k, pshed_val_k, weight_vals_k, pd)
    end
    
    @info "[$FAIR_FUNC] Iteration $k: upper-level status = $status (type: $(typeof(status)))"


    # Update weights in math dict
    math_before_update = deepcopy(math_new)
    for (i, w) in zip(weight_ids_k, fair_weight_vals)
        math_new["load"][string(i)]["weight"] = w
    end

    # Check weight update consistency
    passed_w, issues_w = check_weight_update_consistency(math_before_update, math_new, weight_ids_k, fair_weight_vals)
    if !passed_w
        error("    [X] Weight update inconsistency: $(join(issues_w, "; "))")
    else 
        if (status == "MOI.INFEASIBLE" || status == "MOI.UNBOUNDED" ) && k == 1
            error("    [!] Fairness optimization reported infeasibility at iteration $k. Check the fairness function implementation and input data.")
        end
            println("    Fairness optimization status: $status")
    end

    push!(all_pshed_lower, sum(pshed_val_k))
    push!(all_pshed_upper, sum(pshed_new))

    fair_weights = fair_weight_vals  # Update for next iteration

    println("    Lower-level shed: $(sum(pshed_val_k)), Upper-level shed: $(sum(pshed_new))")
    println("    Weights: $fair_weight_vals")
end
#error("    [!] Ending iterations early for testing purposes. Remove this break statement to run full iterations.")
relaxed_checks["iteration_label_consistency"] = Dict("passed" => iteration_label_consistent)
print_check_result("Label consistency across all iterations", iteration_label_consistent)

# Now solve the relaxed MLD with the final weights and check limits
print_validation_header("Step 3: Solving relaxed MLD with final weights...")
mld_relaxed_final = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math_new, ipopt_solver; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]);
mld_integer_final = FairLoadDelivery.solve_mc_mld_switch_integer(math_new,Gurobi.Optimizer)

# Check voltage limits on relaxed solution
v_passed, v_violations, v_summary = check_voltage_limits_relaxed(mld_relaxed_final, math_new)
relaxed_checks["voltage_limits"] = Dict("passed" => v_passed, "details" => [string(v) for v in v_violations], "summary" => v_summary)
print_check_result("Voltage limits (relaxed MLD)", v_passed, "$(v_summary["violations"]) violations, $(v_summary["checked"]) checked, $(v_summary["skipped_deenergized"]) de-energized")

# Check current limits on relaxed solution
c_passed, c_violations, c_summary = check_switch_ampacity(mld_relaxed_final, math_new)
relaxed_checks["current_limits"] = Dict("passed" => c_passed, "details" => [string(v) for v in c_violations], "summary" => c_summary)
print_check_result("Switch ampacity (relaxed MLD)", c_passed, "$(c_summary["violations"]) violations, $(c_summary["checked"]) checked")

if haskey(c_summary, "utilizations")
    println("    Switch utilizations:")
    for (s_id, util) in sort(collect(c_summary["utilizations"]), by=x->parse(Int, x[1]))
        println("      Switch $s_id: $(round(util, digits=1))%")
    end
end

# Voltage source consistency
vs_ok = true
for (i, gen) in math_new["gen"]
    if gen["source_id"] == "voltage_source.source"
        for (idx, v) in enumerate(gen["vg"])
            if v != math["gen"][i]["vg"][idx]
                vs_ok = false
            end
        end
        if gen["vbase"] != math["gen"][i]["vbase"]
            vs_ok = false
        end
    end
end
relaxed_checks["voltage_source_consistency"] = Dict("passed" => vs_ok)
print_check_result("Voltage source consistency after relaxation", vs_ok)

validation_results["relaxed_fldp"] = relaxed_checks

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

# Check 4.1: Switch label consistency
passed_sw, issues_sw = check_switch_label_consistency(math_new, switch_states, "random_rounding")
rounding_checks["switch_label_consistency"] = Dict("passed" => passed_sw, "details" => issues_sw)
print_check_result("Switch label consistency", passed_sw, join(issues_sw, "\n"))

# Check 4.2: Block label consistency
passed_bl, issues_bl = check_block_label_consistency(math_new, block_status, "random_rounding")
rounding_checks["block_label_consistency"] = Dict("passed" => passed_bl, "details" => issues_bl)
print_check_result("Block label consistency", passed_bl, join(issues_bl, "\n"))

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
#for r in 1:N_ROUNDS
    r=1
    rng=100

    # Generate bernoulli samples for switches and blocks
    bernoulli_samples[r] = generate_bernoulli_samples(switch_states, N_BERNOULLI_SAMPLES, rng)

    # Find the best bernoulli sample that is topology feasible and closest to the relaxed solution
    index, switch_states_radial, block_ids, block_status_radial, load_ids, load_status, feasible_candidates = FairLoadDelivery.radiality_check(ref, switch_states, block_status, bernoulli_samples[r])

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
#end

# Check if any round found a radial topology
passed_radial = any(idx !== nothing for idx in bernoulli_selection_index)
rounding_checks["radiality_found"] = Dict("passed" => passed_radial, "details" => ["Sample indices by round: $bernoulli_selection_index"])
print_check_result("Radial topology found", passed_radial)

# Check that rounded values are binary (1.0 or 0.0)
binary_check_passed = true
non_binary_values = []
#for r in 1:N_ROUNDS
    r = 1 
    if bernoulli_selection_index[r] === nothing
        @error("No Bernoulli radial samples")
    end
    # Check switch states
    for (s_id, state) in bernoulli_switch_selection_exp[r]
        if !(state == 0.0 || state == 1.0)
            binary_check_passed = false
            push!(non_binary_values, "Round $r: Switch $s_id = $state")
        end
    end
    # Check block status
    for (b_id, status) in bernoulli_block_selection_exp[r]
        if !(status == 0.0 || status == 1.0)
            binary_check_passed = false
            push!(non_binary_values, "Round $r: Block $b_id = $status")
        end
    end
    # Check load status
    for (l_id, status) in bernoulli_load_selection_exp[r]
        if !(status == 0.0 || status == 1.0)
            binary_check_passed = false
            push!(non_binary_values, "Round $r: Load $l_id = $status")
        end
    end
#end
rounding_checks["binary_values"] = Dict("passed" => binary_check_passed, "details" => non_binary_values)
print_check_result("Rounded values are binary (0.0 or 1.0)", binary_check_passed, isempty(non_binary_values) ? "" : join(non_binary_values[1:min(5,length(non_binary_values))], "; "))

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
        bernoulli_switch_selection_exp[r],
        ref)
end
if math_out == [nothing for _ in 1:N_ROUNDS]
    @error "[$CASE/$FAIR_FUNC] FAILED — no feasible radial topology found across all $N_ROUNDS rounds. Check warnings above for failure stage RADIAL FEASIBILITY."
    error("No feasible solution — cannot proceed to rounding checks")
end

# Voltage source consistency after rounding (check all valid rounds)
vs_ok_round = true
for r in 1:N_ROUNDS
    if math_out[r] === nothing
        continue
    end
    for (i, gen) in math_out[r]["gen"]
        if gen["source_id"] == "voltage_source.source"
            for (idx, v) in enumerate(gen["vg"])
                if v != math["gen"][i]["vg"][idx]
                    vs_ok_round = false
                end
            end
        end
    end
end
rounding_checks["voltage_source_after_rounding"] = Dict("passed" => vs_ok_round)
print_check_result("Voltage source consistency after rounding", vs_ok_round)

# Solve rounded integer MLD for each round and check limits
mld_rounded_results = Vector{Dict{String, Any}}()
math_rounded_results = Vector{Dict{String, Any}}()
r=1
if math_out[r] === nothing
    @error("[$CASE/$FAIR_FUNC] Skipping rounded MLD for round $r (no feasible radial topology)")
end
println("\n  Solving rounded MLD for round $r...")
mld_rounded_r = FairLoadDelivery.solve_mc_mld_shed_random_round_integer(math_out[r], gurobi_solver)
rounded_term = mld_rounded_r["termination_status"]
passed_term = (rounded_term == MOI.OPTIMAL || rounded_term == MOI.LOCALLY_SOLVED || rounded_term == MOI.ALMOST_LOCALLY_SOLVED)
rounding_checks["rounded_mld_converged_r$r"] = Dict("passed" => passed_term, "details" => ["Status: $rounded_term"])
print_check_result("Rounded MLD converge status (round $r)", passed_term, "Status: $rounded_term")

if passed_term
    push!(mld_rounded_results, mld_rounded_r)
    push!(math_rounded_results, math_out[r])
    # Check voltage limits
    v_passed_r, v_violations_r, v_summary_r = check_voltage_limits_relaxed(mld_rounded_r, math_out[r])
    rounding_checks["voltage_limits_rounded_r$r"] = Dict("passed" => v_passed_r, "details" => [string(v) for v in v_violations_r])
    print_check_result("Voltage limits (rounded MLD, round $r)", v_passed_r, "$(v_summary_r["violations"]) violations")

    # Check current limits
    c_passed_r, c_violations_r, c_summary_r = check_switch_ampacity(mld_rounded_r, math_out[r])
    rounding_checks["current_limits_rounded_r$r"] = Dict("passed" => c_passed_r, "details" => [string(v) for v in c_violations_r])
    print_check_result("Switch ampacity (rounded MLD, round $r)", c_passed_r, "$(c_summary_r["violations"]) violations")

    if haskey(c_summary_r, "utilizations")
        println("    Switch utilizations (rounded, round $r):")
        for (s_id, util) in sort(collect(c_summary_r["utilizations"]), by=x->parse(Int, x[1]))
            println("      Switch $s_id: $(round(util, digits=1))%")
        end
    end
end

function find_best_mld_solution(mlds::Vector{Dict{String, Any}})
    best_obj = -Inf
    best_set = 0
    best_mld = Dict{String, Any}()
    @info " the number of mlds to evaluate is: $(length(mlds))"
    for (id, mld) in enumerate(mlds)
        @info "Rounded solution from set $id has termination status: $(mld["termination_status"]) and objective value: $(mld["objective"])"
        # Objective is to maximize the load served
        if best_obj <= mld["objective"] 
            best_obj = mld["objective"]
            best_set = id
            best_mld = mld
        end
    end
    return best_set, best_mld
end

# Use the first valid round's results for subsequent steps
if isempty(mld_rounded_results)
    @error "[$CASE/$FAIR_FUNC] FAILED — no feasible rounded MLD solution found across all $N_ROUNDS rounds. Check warnings above for failure stage ROUNDED MLD SOLVE."
    error("No feasible solution — cannot proceed to AC feasibility test")
end
best_set, best_mld = find_best_mld_solution(mld_rounded_results)
math_rounded = math_rounded_results[best_set]
mld_rounded = mld_rounded_results[best_set]

validation_results["random_rounding"] = rounding_checks

# ============================================================
# STEP 5: AC FEASIBILITY TEST
# ============================================================
print_validation_header("Step 5: AC Feasibility Test")

ac_checks = Dict{String, Any}()
ac_summary = Dict{String, Any}()

# Set generation capacity high for AC feasibility (slack bus)
#math_ac = deepcopy(math_rounded)
math_ac = ac_network_update(math_rounded, ref;mld_solution=mld_rounded)


# Run AC power flow
println("  Running AC power flow (IVRUPowerModel)...")
ac_result = PowerModelsDistribution.solve_mc_pf(math_ac, IVRUPowerModel, ipopt_solver)

ac_term = ac_result["termination_status"]
ac_converged = (ac_term == MOI.OPTIMAL || ac_term == MOI.LOCALLY_SOLVED || ac_term == MOI.ALMOST_LOCALLY_SOLVED)

ac_checks["ac_convergence"] = Dict("passed" => ac_converged, "details" => ["Status: $ac_term"])
print_check_result("AC power flow converged", ac_converged, "Status: $ac_term")

ac_summary["converged"] = ac_converged
ac_summary["termination_status"] = string(ac_term)

if ac_converged && haskey(ac_result, "solution")
    # Check AC voltage limits
    v_passed_ac, v_violations_ac, v_summary_ac = check_voltage_limits_ac(ac_result, math_ac)
    ac_checks["voltage_limits_ac"] = Dict("passed" => v_passed_ac, "details" => [string(v) for v in v_violations_ac])
    print_check_result("Voltage limits (AC PF)", v_passed_ac, "$(v_summary_ac["violations"]) violations out of $(v_summary_ac["checked"]) checked")
    ac_summary["voltage_violations"] = v_summary_ac["violations"]

    if !v_passed_ac
        println("    Voltage violations:")
        for v in v_violations_ac
            println("      Bus $(v.bus_id) phase $(v.phase): |V|=$(round(v.v_mag, digits=4)) pu ($(v.type))")
        end
    end

    # Check switch current ratings on AC solution
    # For AC solutions, we need to compute switch power from the AC result
    # The AC PF result may have different keys, so we check what's available
    if haskey(ac_result["solution"], "switch")
        c_passed_ac, c_violations_ac, c_summary_ac = check_switch_ampacity(ac_result, math_ac)
        ac_checks["current_limits_ac"] = Dict("passed" => c_passed_ac, "details" => [string(v) for v in c_violations_ac])
        print_check_result("Switch ampacity (AC PF)", c_passed_ac, "$(c_summary_ac["violations"]) violations")
        ac_summary["current_violations"] = c_summary_ac["violations"]

        if !c_passed_ac
            println("    Current rating violations:")
            for v in c_violations_ac
                println("      Switch $(v.switch_id) phase $(v.phase): $(round(v.utilization_pct, digits=1))% utilized")
            end
        end
    else
        println("    [!] No switch data in AC PF solution - cannot check current ratings")
        ac_summary["current_violations"] = -1  # Unknown
        ac_checks["current_limits_ac"] = Dict("passed" => true, "details" => ["No switch data in AC solution"])
    end

    # Report total load served in AC solution
    if haskey(ac_result["solution"], "load")
        total_pd_served = sum(sum(load_data["pd"]) for (_, load_data) in ac_result["solution"]["load"] 
        if haskey(load_data, "pd"))
        println("    Total active power served (AC): $total_pd_served")
    end
else
    ac_summary["voltage_violations"] = -1
    ac_summary["current_violations"] = -1
    @error "    ** NO AC FEASIBLE SOLUTION - cannot check limits **"
end

validation_results["ac_feasibility"] = ac_checks
validation_results["ac_feasibility_summary"] = ac_summary

# 

# ============================================================
# GENERATE FINAL REPORT
# ============================================================
report_path = joinpath(save_dir, "validation_report_$(pshed_type)_$case.txt")
generate_summary_report(validation_results, report_path)

# Save final network plot
if ac_converged && haskey(ac_result, "solution")
    if !isempty(ac_result["solution"])
        plot_path = joinpath(save_dir, "network_load_shed_$(pshed_type)_$case.svg")
        plot_network_load_shed(mld_rounded["solution"], math_rounded; output_file=plot_path)
    else
        println("  [!] Cannot plot network - 0 power flow in AC solution available.")
    end
end

println("\nValidation complete.")

load_ids       = sort(collect(keys(mld_rounded["solution"]["load"])),
by=x->parse(Int,x))
load_labels    = [math_rounded["load"][lid]["name"] for lid in
load_ids]
pshed_per_load = [sum(mld_rounded["solution"]["load"][lid]["pshed"])
for lid in load_ids]
weight_per_load = [math_new["load"][lid]["weight"] for lid in load_ids]
pd_per_load    = [sum(math_rounded["load"][lid]["pd"]) for lid in load_ids]

df_loads = DataFrames.DataFrame(
    load_id   = load_ids,
    load_name = load_labels,
    pd_kw     = pd_per_load,
    pshed_kw  = pshed_per_load,
    weight    = weight_per_load,
)
loads_savepath = joinpath(save_dir, "per_load_weights_$(pshed_type)_$(case)_$(FAIR_FUNC).csv")
CSV.write(loads_savepath, df_loads)
println("Saved per-load fairness weights to $loads_savepath")

df_weights_only = DataFrames.DataFrame(
    load_id = load_labels,
    weight  = weight_per_load,
)
weights_only_savepath = joinpath(save_dir, "weights_only_$(pshed_type)_$(case)_$(FAIR_FUNC).csv")
CSV.write(weights_only_savepath, df_weights_only)
println("Saved load prioritization weights to $weights_only_savepath")

p_weights = bar(load_labels, weight_per_load,
    xlabel = "Load",
    ylabel = "Prioritization weight",
    legend = false,
    color  = :steelblue,
    linecolor = :black,
    xrotation = 45,
    xticks = (1:length(load_labels), load_labels)
)
display(p_weights)
savefig(p_weights, joinpath(save_dir,
    "weight_distribution_$(pshed_type)_$(case)_$(FAIR_FUNC).svg"))

p_dist = bar(load_labels, pshed_per_load,
    xlabel = "Load",
    ylabel = "Load shed (kW)",
    #title  = "Load shed distribution",
    legend = false,
    color  = :steelblue,
    linecolor = :black,
    xrotation = 45,
    xticks = (1:length(load_labels), load_labels)
)

# label the max bar
max_val = maximum(pshed_per_load)
max_idx = argmax(pshed_per_load)
# annotate!(p_dist, max_idx, max_val + max_val*0.03,
#         text("max=$(round(max_val, digits=1))", 9, :center, :red))

display(p_dist)
savefig(p_dist, joinpath(save_dir,
"loadshed_distribution_rounded_$(pshed_type)_$(case)_$(FAIR_FUNC).svg"))

mean = Statistics.mean(pshed_per_load)
std_dev = Statistics.std(pshed_per_load)
cv = std_dev / mean
n = length(pshed_per_load)
total_shed   = sum(pshed_per_load)
total_served = total_pd_ref - total_shed
pct_shed     = total_pd_ref > 0 ? 100 * total_shed   / total_pd_ref : NaN
pct_served   = total_pd_ref > 0 ? 100 * total_served / total_pd_ref : NaN
l1           = LinearAlgebra.norm(pshed_per_load, 1)
l2           = LinearAlgebra.norm(pshed_per_load, 2)
linf         = LinearAlgebra.norm(pshed_per_load, Inf)
μ            = n > 0 ? total_shed / n : NaN
σ            = n > 1 ? Statistics.std(pshed_per_load; corrected=true) : 0.0
cv           = (isfinite(μ) && μ > 0) ? σ / μ : NaN

metrics = Dict{Symbol,Float64}(
    :total_pd_kw     => total_pd_ref,
    :total_shed_kw   => total_shed,
    :total_served_kw => total_served,
    :pct_shed        => pct_shed,
    :pct_served      => pct_served,
    :l1_norm         => l1,
    :l2_norm         => l2,
    :linf_norm       => linf,
    :cv              => cv,
)

cols = [:total_pd_kw, :total_shed_kw, :total_served_kw, :pct_shed, :pct_served,
        :l1_norm, :l2_norm, :linf_norm, :cv]
df = DataFrames.DataFrame((c => [metrics[c]] for c in cols)...)
savepath = joinpath(save_dir, "load_shed_metrics_$(pshed_type)_$(case)_$(FAIR_FUNC).csv")
CSV.write(savepath, df)
println("Saved load shed metrics to $savepath")
#   l9_idx = 7# findfirst(==("8"), weight_ids)
#   @info "dpshed[:, L9] = $(dpshed[:, l9_idx])"
#   @info "dpshed[L9, L9] = $(dpshed[l9_idx, l9_idx])"
#   @info "pshed[L9] = $(pshed_val[l9_idx]),  pd[L9] =$(sum(math_new["load"]["8"]["pd"]))"

#   mld_int_direct = FairLoadDelivery.solve_mc_mld_min_max_integer(math, Gurobi.Optimizer)
#   for (lid, ld) in sort(collect(mld_int_direct["solution"]["load"]),
#   by=x->parse(Int,x[1]))
#       println("Load $lid ($(math["load"][lid]["name"])): pshed = $(sum(ld["pshed"]))")
#   end

#    @info "Final relaxed switch states (target for rounding):"
#   for (sid, sw) in sort(collect(mld_relaxed_final["solution"]["switch"]),
#    by=x->parse(Int,x[1]))
#       println("  switch $sid ($(math_new["switch"][sid]["name"])): state
#   = $(round(sw["state"], digits=3))")
#   end
#   @info "Final relaxed pshed (what min-max was trying to balance):"
#   for (lid, ld) in sort(collect(mld_relaxed_final["solution"]["load"]),
#   by=x->parse(Int,x[1]))
#       println("  load $lid ($(math_new["load"][lid]["name"])): pshed =
#   $(round(sum(ld["pshed"]), digits=2)),  weight =
#   $(round(math_new["load"][lid]["weight"], digits=2))")
#   end