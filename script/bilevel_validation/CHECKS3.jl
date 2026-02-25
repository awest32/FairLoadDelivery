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

include("validation_utils.jl")
include("../../src/implementation/other_fair_funcs.jl")
include("../../src/implementation/random_rounding.jl")

# Solve rounded MLD for each round and check limits
mld_rounded_results = Vector{Dict{String, Any}}()
math_rounded_results = Vector{Dict{String, Any}}()
for r in 1:N_ROUNDS
    if math_out[r] === nothing
        @warn "[$CASE/$FAIR_FUNC] Skipping rounded MLD for round $r (no feasible radial topology)"
        continue
    end
    println("\n  Solving rounded MLD for round $r...")
    mld_rounded_r = FairLoadDelivery.solve_mc_mld_shed_random_round(math_out[r], gurobi_solver)
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
#        print_check_result("Switch ampacity (rounded MLD, round $r)", c_passed_r, "$(c_summary_r["violations"]) violations")

        if haskey(c_summary_r, "utilizations")
            println("    Switch utilizations (rounded, round $r):")
            for (s_id, util) in sort(collect(c_summary_r["utilizations"]), by=x->parse(Int, x[1]))
                println("      Switch $s_id: $(round(util, digits=1))%")
            end
        end
    end
end
function find_best_mld_solution(mlds::Vector{Dict{String, Any}}, ipopt)
    best_obj = -Inf
    best_set = 0
    best_mld = Dict{String, Any}()
    @info " the number of mlds to evaluate is: $(length(mlds))"
    for (id, mld) in enumerate(mlds)
        @info "Rounded solution from set $id has termination status: $(mld["termination_status"]) and objective value: $(mld["objective"])"
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
best_set, best_mld = find_best_mld_solution(mld_rounded_results, ipopt)
math_rounded = math_rounded_results[best_set]
mld_rounded = mld_rounded_results[best_set]

validation_results["random_rounding"] = rounding_checks
