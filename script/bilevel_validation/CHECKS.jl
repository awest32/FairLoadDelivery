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

# Load validation utilities
include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")

# ============================================================
# CONFIGURATION
# ============================================================
const CASE = "motivation_a"
const CASE_FILE = "ieee_13_aw_edit/$CASE.dss"
const LS_PERCENT = 0.8
const ITERATIONS = 10
const FAIR_FUNC = "efficiency"  # simplest fairness function for testing
const N_ROUNDS = 5
const N_BERNOULLI_SAMPLES = 6

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

# ============================================================
# STEP 1: NETWORK SETUP
# ============================================================
print_validation_header("Step 1: Network Setup")

eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT, [])

# math["bus_lookup"] maps eng bus name -> math bus id (built by PowerModelsDistribution)
# Invert it to get math bus id -> eng bus name
math_to_eng_bus = Dict{Int,String}(v => k for (k, v) in math["bus_lookup"])

println("\n  Math-to-Eng bus mapping:")
for (math_id, eng_name) in sort(collect(math_to_eng_bus), by=first)
    println("    math bus $math_id => eng bus \"$eng_name\"")
end

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
for (i, gen) in math["gen"]
    if gen["source_id"] == "voltage_source.source"
        total_pmax = sum(gen["pmax"])
        @info "Generation $i ($(gen["name"])) capacity: $total_pmax"
        total_pd = sum(sum(load["pd"]) for (_, load) in math["load"])
        @info "Total demand in system: $total_pd"
        if total_pmax < total_pd
            gen_limited = true
        end
        println("    Generation capacity: $total_pmax, Total demand: $total_pd")
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

# Print load info with bus mapping
println("\n  Load configuration:")
for (l_id, load) in sort(collect(math["load"]), by=x->parse(Int,x[1]))
    eng_bus = get(math_to_eng_bus, load["load_bus"], "?")
    println("    Load $l_id ($(load["name"])): bus=$(load["load_bus"]) ($eng_bus), pd=$(load["pd"]), weight=$(load["weight"])")
end

validation_results["setup"] = setup_checks

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
    global fair_weights, iteration_label_consistent, weight_ids
    println("\n  --- Iteration $k ---")

    # Solve lower-level
    dpshed_k, pshed_val_k, pshed_ids_k, weight_vals_k, weight_ids_k, model_k = lower_level_soln(math_new, fair_weights, k)

    # Check label consistency at each iteration
    passed_k, issues_k = check_label_consistency(math_new, pshed_ids_k, weight_ids_k, "iteration_$k")
    if !passed_k
        iteration_label_consistent = false
        @error "    [X] Label mismatch at iteration $k: $(join(issues_k, "; "))"
    end

    # Apply upper-level fairness function
    if FAIR_FUNC == "min_max"
        pshed_new, fair_weight_vals = min_max_load_shed(dpshed_k, pshed_val_k, weight_vals_k)
    elseif FAIR_FUNC == "proportional"
        pshed_new, fair_weight_vals = proportional_fairness_load_shed(dpshed_k, pshed_val_k, weight_vals_k, math_new)
    elseif FAIR_FUNC == "efficiency"
        pshed_new, fair_weight_vals = complete_efficiency_load_shed(dpshed_k, pshed_val_k, weight_vals_k, math_new)
    elseif FAIR_FUNC == "jain"
        pshed_new, fair_weight_vals = jains_fairness_index(dpshed_k, pshed_val_k, weight_vals_k)
    elseif FAIR_FUNC == "equality_min"
        pshed_new, fair_weight_vals = equality_min(dpshed_k, pshed_val_k, weight_vals_k)
    elseif FAIR_FUNC == "palma"
        pd = Float64[]
        for i in pshed_ids_k
            push!(pd, sum(math_new["load"][string(i)]["pd"]))
        end
        pshed_new, fair_weight_vals = lin_palma_w_grad_input(dpshed_k, pshed_val_k, weight_vals_k, pd)
    end

    # Update weights in math dict
    math_before_update = deepcopy(math_new)
    for (i, w) in zip(weight_ids_k, fair_weight_vals)
        math_new["load"][string(i)]["weight"] = w
    end

    # Check weight update consistency
    passed_w, issues_w = check_weight_update_consistency(math_before_update, math_new, weight_ids_k, fair_weight_vals)
    if !passed_w
        @error "    [X] Weight update inconsistency: $(join(issues_w, "; "))"
    end

    push!(all_pshed_lower, sum(pshed_val_k))
    push!(all_pshed_upper, sum(pshed_new))

    fair_weights = fair_weight_vals  # Update for next iteration
    weight_ids = weight_ids_k  # Update for next iteration
    println("    Lower-level shed: $(sum(pshed_val_k)), Upper-level shed: $(sum(pshed_new))")
    println("    Weights: $fair_weight_vals")
end

relaxed_checks["iteration_label_consistency"] = Dict("passed" => iteration_label_consistent)
print_check_result("Label consistency across all iterations", iteration_label_consistent)

# Now solve the relaxed MLD with the final weights and check limits
println("\n  Solving relaxed MLD with final weights...")
mld_relaxed_final = FairLoadDelivery.solve_mc_mld_shed_implicit_diff(math_new, ipopt_solver; ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!])

fair_weights 
weight_ids
loads = math_new["load"]
loads

for (i, (pid, wid)) in enumerate(zip(pshed_ids, weight_ids))
    load = math["load"][string(pid)]
    println("$i: load $(load["name"]), pshed_id=$pid, weight_id=$wid, pd=$(load["pd"]), weight=$(fair_weights[i])")
end

for (s_id, switch) in math_new["switch"]
    sw_state = mld_relaxed_final["solution"]["switch"][s_id]["state"]
     println("Switch $s_id ($(switch["name"])): z_sw = $sw_state")
end

for (b_id, b_data) in ref[:blocks]
    n_gen = length(ref[:block_gens][b_id])
    n_strg = length(ref[:block_storages][b_id])
    switches = ref[:block_switches][b_id]
    is_sub = b_id in ref[:substation_blocks]
    println("Block $b_id: n_gen=$n_gen, n_strg=$n_strg, n_switches=$(length(switches)), substation=$is_sub")
    println("  switches: $switches")
    println("  gens: $(ref[:block_gens][b_id])")
end