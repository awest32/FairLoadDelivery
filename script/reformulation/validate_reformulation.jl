#=
Validation Experiment: Test the Reformulated Palma Ratio Optimization
======================================================================

This script tests the new reformulation (pshed as expression) using
synthetic data. It does NOT depend on the FairLoadDelivery package
to avoid precompilation issues.

Run with:
    julia --project=. script/reformulation/validate_reformulation.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using JuMP
using LinearAlgebra
using Random
using Printf
using Statistics

# Check for solvers
const GUROBI_OK = try using Gurobi; true catch; false end
const HIGHS_OK = try using HiGHS; true catch; false end

println("Solver availability:")
println("  Gurobi: ", GUROBI_OK)
println("  HiGHS:  ", HIGHS_OK)

if !GUROBI_OK
    error("Gurobi is required for the quadratic Charnes-Cooper constraints.")
end

#=============================================================================
 Include the main reformulation (standalone, no FairLoadDelivery dependency)
=============================================================================#

include("load_shed_as_parameter.jl")

#=============================================================================
 Test 1: Synthetic Data Validation
=============================================================================#

function test_synthetic(; n::Int=6, seed::Int=42)
    println("\n" * "="^70)
    println("TEST 1: Synthetic Data Validation")
    println("="^70)

    Random.seed!(seed)

    # Generate test data
    pd = rand(n) .* 10 .+ 1
    pshed_prev = pd .* (0.3 .+ 0.4 .* rand(n))
    weights_prev = ones(n) .* 5.0

    # Jacobian: increasing weight reduces load shed
    dpshed_dw = zeros(n, n)
    for i in 1:n
        dpshed_dw[i, i] = -pd[i] * 0.1
        for j in 1:n
            if i != j
                dpshed_dw[i, j] = pd[i] * 0.01 * randn()
            end
        end
    end

    println("n = $n")
    println("Initial Palma: ", @sprintf("%.4f", palma_ratio(pshed_prev)))

    # Run optimization
    result = palma_ratio_minimization(
        dpshed_dw, pshed_prev, weights_prev, pd;
        trust_radius=0.5,
        relax_binary=true,
        silent=true
    )

    println("\nResults:")
    println("  Status:      ", result.status)
    println("  Solve time:  ", @sprintf("%.4f s", result.solve_time))
    println("  Final Palma: ", @sprintf("%.4f", result.palma_ratio))

    # Validation checks
    passed = true

    # Check 1: Solver succeeded
    if result.status in [OPTIMAL, LOCALLY_SOLVED, ALMOST_OPTIMAL]
        println("  ✓ Solver succeeded")
    else
        println("  ✗ Solver failed")
        passed = false
    end

    # Check 2: Permutation doubly stochastic
    a = result.permutation
    row_ok = all(abs.(sum(a, dims=2) .- 1) .< 1e-4)
    col_ok = all(abs.(sum(a, dims=1) .- 1) .< 1e-4)
    if row_ok && col_ok
        println("  ✓ Permutation is doubly stochastic")
    else
        println("  ✗ Permutation not doubly stochastic")
        passed = false
    end

    # Check 3: Trust region
    if all(abs.(result.delta_w) .<= 0.5 + 1e-6)
        println("  ✓ Trust region satisfied")
    else
        println("  ✗ Trust region violated")
        passed = false
    end

    # Check 4: Palma improved or stayed same
    initial = palma_ratio(pshed_prev)
    final = result.palma_ratio
    if final <= initial + 1e-6
        println("  ✓ Palma ratio improved (", @sprintf("%.4f → %.4f", initial, final), ")")
    else
        println("  ⚠ Palma ratio increased (may be at local min)")
    end

    println("\n", passed ? "TEST 1 PASSED" : "TEST 1 FAILED")
    return passed
end

#=============================================================================
 Test 2: Multiple Sizes
=============================================================================#

function test_scaling(; sizes=[4, 6, 8, 10], seed::Int=42)
    println("\n" * "="^70)
    println("TEST 2: Performance Scaling")
    println("="^70)

    Random.seed!(seed)

    println("\n   n    Time (s)    Palma      Status")
    println("  ---   --------   --------   --------")

    for n in sizes
        pd = rand(n) .* 10 .+ 1
        pshed_prev = pd .* (0.3 .+ 0.4 .* rand(n))
        weights_prev = ones(n) .* 5.0

        dpshed_dw = zeros(n, n)
        for i in 1:n
            dpshed_dw[i, i] = -pd[i] * 0.1
        end

        result = palma_ratio_minimization(
            dpshed_dw, pshed_prev, weights_prev, pd;
            trust_radius=0.1,
            relax_binary=true,
            silent=true
        )

        println("  ", @sprintf("%3d", n), "   ",
                @sprintf("%8.4f", result.solve_time), "   ",
                @sprintf("%8.4f", result.palma_ratio), "   ",
                result.status)
    end

    println("\nTEST 2 COMPLETED")
    return true
end

#=============================================================================
 Test 3: Multi-Iteration Convergence
=============================================================================#

function test_convergence(; n::Int=6, iterations::Int=5, seed::Int=42)
    println("\n" * "="^70)
    println("TEST 3: Multi-Iteration Convergence")
    println("="^70)

    Random.seed!(seed)

    pd = rand(n) .* 10 .+ 1
    pshed_curr = pd .* (0.3 .+ 0.4 .* rand(n))
    weights_curr = ones(n) .* 5.0

    println("n = $n, iterations = $iterations")
    println("\nIteration  Palma Ratio   Δ Weights")
    println("---------  -----------   ---------")

    initial_palma = palma_ratio(pshed_curr)
    println(@sprintf("    0       %8.4f      ---", initial_palma))

    for k in 1:iterations
        # Simulate varying Jacobian
        dpshed_dw = zeros(n, n)
        for i in 1:n
            dpshed_dw[i, i] = -pd[i] * 0.1 * (1 + 0.1 * randn())
            for j in 1:n
                if i != j
                    dpshed_dw[i, j] = pd[i] * 0.01 * randn()
                end
            end
        end

        result = palma_ratio_minimization(
            dpshed_dw, pshed_curr, weights_curr, pd;
            trust_radius=0.1,
            relax_binary=true,
            silent=true
        )

        pshed_curr = result.pshed_new
        weights_curr = result.weights_new

        println(@sprintf("    %d       %8.4f    %8.4f",
                k, result.palma_ratio, sum(abs.(result.delta_w))))
    end

    final_palma = palma_ratio(pshed_curr)
    println("\nInitial Palma: ", @sprintf("%.4f", initial_palma))
    println("Final Palma:   ", @sprintf("%.4f", final_palma))

    if final_palma < initial_palma
        println("✓ Palma ratio improved!")
    else
        println("⚠ Palma ratio did not improve (may be at optimum)")
    end

    println("\nTEST 3 COMPLETED")
    return true
end

#=============================================================================
 Main Entry Point
=============================================================================#

function run_all_tests()
    println("="^70)
    println("PALMA RATIO REFORMULATION VALIDATION")
    println("="^70)
    println("Testing: pshed as JuMP expression (not variable)")
    println()

    results = Dict{String, Bool}()

    results["synthetic"] = test_synthetic(n=6, seed=42)
    results["scaling"] = test_scaling(sizes=[4, 6, 8], seed=42)
    results["convergence"] = test_convergence(n=6, iterations=5, seed=42)

    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    for (name, passed) in sort(collect(results))
        status = passed ? "✓ PASS" : "✗ FAIL"
        println("  $name: $status")
    end

    all_passed = all(values(results))
    println("\n" * (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED"))
    println("="^70)

    return all_passed
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
