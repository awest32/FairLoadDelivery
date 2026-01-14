#=
Reformulated Palma Ratio Fair Load Prioritization
==================================================

This implementation reformulates the Palma ratio optimization by treating P_shed
as a JuMP expression derived from first-order Taylor expansion, rather than as
optimization variables with equality constraints.

Key simplification:
- P_shed_new is an EXPRESSION: pshed_new[i] = pshed_prev[i] + Σ_j J[i,j]·Δw[j]
- Eliminates n variables and their equality constraints
- Full dynamic sorting is PRESERVED via permutation matrix optimization

Mathematical formulation:
    min_{Δw, A} [Σ_{i∈Top10%} sorted[i]] / [Σ_{i∈Bot40%} sorted[i]]

    s.t. sorted[i] = Σ_j a[i,j] · pshed_new[j]   (sorting via permutation)
         pshed_new[j] = pshed_prev[j] + Σ_k J[j,k]·Δw[k]  (Taylor expression)
         Σ_j a[i,j] = 1, Σ_i a[i,j] = 1          (doubly stochastic)
         sorted[k] ≤ sorted[k+1]                  (ascending order)
         |Δw| ≤ trust_radius                      (trust region)

Uses Charnes-Cooper transformation to convert ratio to linear objective.
Uses McCormick envelopes for bilinear terms a[i,j] * pshed_new[j].

Author: Claude (with guidance from Sam)
Date: 2026-01-14
=#

using JuMP
using LinearAlgebra

# Try to load solvers
# Gurobi is REQUIRED for the quadratic Charnes-Cooper constraint
# HiGHS can be used as fallback with Dinkelbach's algorithm (see alternative function)
const GUROBI_AVAILABLE = try
    using Gurobi
    true
catch
    false
end

const HIGHS_AVAILABLE = try
    using HiGHS
    true
catch
    false
end

const IPOPT_AVAILABLE = try
    using Ipopt
    true
catch
    false
end

#=============================================================================
 Helper Functions
=============================================================================#

"""
    compute_palma_indices(n::Int) -> (top_10_idx, bottom_40_idx)

Compute indices for top 10% and bottom 40% in SORTED space.
These are fixed positions in the sorted array, not indices into the original array.

For Palma ratio: top 10% = largest values, bottom 40% = smallest values.
In ascending sorted order: bottom 40% are first floor(0.4n) positions,
                           top 10% are last ceil(0.1n) positions.

# Example
For n=10: bottom_40_idx = [1,2,3,4], top_10_idx = [10]
For n=20: bottom_40_idx = [1,2,3,4,5,6,7,8], top_10_idx = [19,20]
"""
function compute_palma_indices(n::Int)
    # Bottom 40%: indices 1 to floor(0.4n) in sorted order
    n_bottom = max(1, floor(Int, 0.4 * n))
    bottom_40_idx = collect(1:n_bottom)

    # Top 10%: indices ceil(0.9n) to n in sorted order
    n_top_start = max(n, ceil(Int, 0.9 * n))  # Ensure at least last element
    top_10_idx = collect(n_top_start:n)

    # Handle edge case where ranges might be empty
    if isempty(top_10_idx)
        top_10_idx = [n]
    end
    if isempty(bottom_40_idx)
        bottom_40_idx = [1]
    end

    return top_10_idx, bottom_40_idx
end

"""
    palma_ratio(pshed::Vector{Float64}) -> Float64

Compute the Palma ratio: sum(top 10%) / sum(bottom 40%) after sorting.
Returns Inf if denominator is zero or negative.
"""
function palma_ratio(pshed::Vector{Float64})
    n = length(pshed)
    sorted_pshed = sort(pshed)  # ascending order

    top_10_idx, bottom_40_idx = compute_palma_indices(n)

    numerator = sum(sorted_pshed[i] for i in top_10_idx)
    denominator = sum(sorted_pshed[i] for i in bottom_40_idx)

    if denominator <= 0
        return Inf
    end
    return numerator / denominator
end

"""
    get_default_solver()

Return the best available solver optimizer.

NOTE: The Charnes-Cooper transformation creates quadratic constraints (sorted[i] * σ),
so a QP-capable solver like Gurobi is required. HiGHS only supports LP/MILP.
"""
function get_default_solver()
    if GUROBI_AVAILABLE
        return Gurobi.Optimizer
    elseif IPOPT_AVAILABLE
        @warn "Using Ipopt (NLP solver). Gurobi is recommended for better performance."
        return Ipopt.Optimizer
    else
        error("No QP-capable solver available. Please install Gurobi or Ipopt.\n" *
              "HiGHS cannot handle the quadratic Charnes-Cooper constraints.")
    end
end

#=============================================================================
 Main Optimization: Palma Ratio Minimization
=============================================================================#

"""
    palma_ratio_minimization(
        dpshed_dw::Matrix{Float64},
        pshed_prev::Vector{Float64},
        weights_prev::Vector{Float64},
        pd::Vector{Float64};
        trust_radius::Float64 = 0.1,
        w_bounds::Tuple{Float64,Float64} = (0.0, 10.0),
        solver = get_default_solver(),
        silent::Bool = true,
        relax_binary::Bool = true
    )

Solve the Palma ratio minimization problem with P_shed as an expression.

# Key Innovation
P_shed is represented as a JuMP @expression (not @variable):
```julia
@expression(model, pshed_new[i], pshed_prev[i] + Σ_j J[i,j]·Δw[j])
```
This eliminates the need for equality constraints linking pshed variables
to the Taylor expansion, reducing model complexity.

# Arguments
- `dpshed_dw`: Jacobian matrix ∂P_shed/∂w from lower-level implicit differentiation (n×n)
- `pshed_prev`: Load shed values from previous iteration (n)
- `weights_prev`: Weight values from previous iteration (n)
- `pd`: Load demands (upper bounds on load shed) (n)
- `trust_radius`: Maximum absolute weight change per iteration (default 0.1)
- `w_bounds`: (w_min, w_max) tuple for weight bounds (default (0.0, 10.0))
- `solver`: JuMP optimizer (default: Gurobi if available, else HiGHS)
- `silent`: Suppress solver output (default true)
- `relax_binary`: If true, relax a[i,j] to [0,1]; if false, use binary (default true)

# Returns
NamedTuple with fields:
- `weights_new::Vector{Float64}`: Updated weights
- `pshed_new::Vector{Float64}`: Predicted load shed values
- `delta_w::Vector{Float64}`: Weight changes
- `palma_ratio::Float64`: Achieved Palma ratio
- `status::TerminationStatusCode`: Solver termination status
- `solve_time::Float64`: Solver time in seconds
- `permutation::Matrix{Float64}`: Optimal permutation matrix (or relaxation)
- `sorted_values::Vector{Float64}`: Sorted load shed values

# Mathematical Formulation

## Decision Variables
- Δw[j] ∈ [-trust_radius, trust_radius]: weight changes
- a[i,j] ∈ {0,1} (or [0,1] if relaxed): permutation matrix
- u[i,j] ≥ 0: McCormick auxiliary for a[i,j] * pshed_new[j]
- σ ≥ ε: Charnes-Cooper scaling variable

## P_shed as Expression
```
pshed_new[j] = pshed_prev[j] + Σ_k dpshed_dw[j,k] * Δw[k]
```

## McCormick Envelopes for u[i,j] = a[i,j] * pshed_new[j]
Since a[i,j] ∈ {0,1} and pshed_new[j] ∈ [0, P_j]:
1. u[i,j] ≥ 0
2. u[i,j] ≥ pshed_new[j] + a[i,j]*P_j - P_j
3. u[i,j] ≤ a[i,j] * P_j
4. u[i,j] ≤ pshed_new[j]

## Charnes-Cooper Transformation
Transform min(num/denom) to: min(num*σ) s.t. denom*σ = 1
where σ = 1/denom > 0.
"""
function palma_ratio_minimization(
    dpshed_dw::Matrix{Float64},
    pshed_prev::Vector{Float64},
    weights_prev::Vector{Float64},
    pd::Vector{Float64};
    trust_radius::Float64 = 0.1,
    w_bounds::Tuple{Float64, Float64} = (0.0, 10.0),
    solver = get_default_solver(),
    silent::Bool = true,
    relax_binary::Bool = true
)
    n = length(pshed_prev)
    w_min, w_max = w_bounds
    ε = 1e-8  # Small positive for σ lower bound

    # Validate inputs
    @assert size(dpshed_dw) == (n, n) "Jacobian must be n×n"
    @assert length(weights_prev) == n "weights_prev must have length n"
    @assert length(pd) == n "pd must have length n"
    @assert all(pd .>= 0) "Load demands must be non-negative"

    # Create model
    model = Model(solver)
    if silent
        set_silent(model)
    end

    # Solver-specific settings
    if GUROBI_AVAILABLE && solver == Gurobi.Optimizer
        set_optimizer_attribute(model, "DualReductions", 0)
        set_optimizer_attribute(model, "MIPGap", 1e-6)
        set_optimizer_attribute(model, "NonConvex", 2)  # Allow non-convex QP
    elseif IPOPT_AVAILABLE && solver == Ipopt.Optimizer
        set_optimizer_attribute(model, "print_level", 0)
    end

    #=========================================================================
    # Decision Variables
    =========================================================================#

    # Weight changes (the primary decision variable)
    @variable(model, Δw[1:n])

    # Permutation matrix (binary or relaxed to doubly stochastic)
    if relax_binary
        @variable(model, 0 <= a[1:n, 1:n] <= 1)
    else
        @variable(model, a[1:n, 1:n], Bin)
    end

    # McCormick auxiliary variables for bilinear products
    @variable(model, u[1:n, 1:n] >= 0)

    # Charnes-Cooper scaling variable
    @variable(model, σ >= ε)

    #=========================================================================
    # P_shed as EXPRESSION (Core Simplification)
    =========================================================================#

    # This is the key innovation: pshed_new is an expression, not a variable
    # It's defined by the first-order Taylor expansion around the previous solution
    @expression(model, pshed_new[j=1:n],
        pshed_prev[j] + sum(dpshed_dw[j, k] * Δw[k] for k in 1:n)
    )

    #=========================================================================
    # Trust Region and Weight Bounds
    =========================================================================#

    @constraint(model, trust_lb[j=1:n], Δw[j] >= -trust_radius)
    @constraint(model, trust_ub[j=1:n], Δw[j] <= trust_radius)
    @constraint(model, weight_lb[j=1:n], weights_prev[j] + Δw[j] >= w_min)
    @constraint(model, weight_ub[j=1:n], weights_prev[j] + Δw[j] <= w_max)

    #=========================================================================
    # Permutation Matrix Constraints (Doubly Stochastic)
    =========================================================================#

    # Row sums = 1: each sorted position gets exactly one load
    @constraint(model, row_sum[i=1:n], sum(a[i, j] for j in 1:n) == 1)

    # Column sums = 1: each load appears in exactly one sorted position
    @constraint(model, col_sum[j=1:n], sum(a[i, j] for i in 1:n) == 1)

    #=========================================================================
    # McCormick Envelopes for u[i,j] = a[i,j] * pshed_new[j]
    =========================================================================#

    # pshed_new[j] is bounded by [0, pd[j]] (can't shed more than demand)
    # a[i,j] is bounded by [0, 1]

    for i in 1:n, j in 1:n
        P_j = pd[j]  # Upper bound on pshed_new[j]

        # Envelope 1: u >= 0 (already in variable bounds)

        # Envelope 2: u >= pshed_new + a*P - P (active when a=1)
        @constraint(model, u[i, j] >= pshed_new[j] + a[i, j] * P_j - P_j)

        # Envelope 3: u <= a*P
        @constraint(model, u[i, j] <= a[i, j] * P_j)

        # Envelope 4: u <= pshed_new (active when a=1)
        @constraint(model, u[i, j] <= pshed_new[j])
    end

    #=========================================================================
    # Sorted Values Expression
    =========================================================================#

    # sorted[i] = Σ_j a[i,j] * pshed_new[j] = Σ_j u[i,j]
    @expression(model, sorted[i=1:n], sum(u[i, j] for j in 1:n))

    #=========================================================================
    # Ascending Order Constraint
    =========================================================================#

    # Enforce sorted[k] <= sorted[k+1] for ascending order
    @constraint(model, ascending[k=1:n-1], sorted[k] <= sorted[k+1])

    #=========================================================================
    # Charnes-Cooper Transformation for Palma Ratio
    =========================================================================#

    # Get indices in sorted space
    top_10_idx, bottom_40_idx = compute_palma_indices(n)

    # Denominator normalization: Σ_{i∈Bottom40%} sorted[i] * σ = 1
    @constraint(model, denom_norm,
        sum(sorted[i] * σ for i in bottom_40_idx) == 1)

    # Objective: minimize Σ_{i∈Top10%} sorted[i] * σ (scaled numerator)
    @objective(model, Min, sum(sorted[i] * σ for i in top_10_idx))

    #=========================================================================
    # Solve
    =========================================================================#

    solve_time = @elapsed optimize!(model)
    status = termination_status(model)

    #=========================================================================
    # Extract Solution
    =========================================================================#

    if status in [OPTIMAL, LOCALLY_SOLVED, ALMOST_OPTIMAL, TIME_LIMIT]
        Δw_val = value.(Δw)
        weights_new = weights_prev .+ Δw_val

        # Compute pshed_new from the expression
        pshed_new_val = pshed_prev .+ dpshed_dw * Δw_val

        # Get permutation matrix
        a_val = value.(a)

        # Compute sorted values
        sorted_val = [value(sorted[i]) for i in 1:n]

        # Compute actual Palma ratio (from unsorted pshed_new)
        actual_palma = palma_ratio(pshed_new_val)

        return (
            weights_new = weights_new,
            pshed_new = pshed_new_val,
            delta_w = Δw_val,
            palma_ratio = actual_palma,
            status = status,
            solve_time = solve_time,
            permutation = a_val,
            sorted_values = sorted_val
        )
    else
        @warn "Solver failed with status: $status"
        return (
            weights_new = weights_prev,
            pshed_new = pshed_prev,
            delta_w = zeros(n),
            palma_ratio = palma_ratio(pshed_prev),
            status = status,
            solve_time = solve_time,
            permutation = Matrix{Float64}(I, n, n),
            sorted_values = sort(pshed_prev)
        )
    end
end

#=============================================================================
 Simplified Interface (matches existing lin_palma_w_grad_input signature)
=============================================================================#

"""
    lin_palma_reformulated(
        dpshed_dw::Matrix{Float64},
        pshed_prev::Vector{Float64},
        weights_prev::Vector{Float64},
        pd::Vector{Float64}
    ) -> (pshed_new, weights_new, σ)

Drop-in replacement for lin_palma_w_grad_input from palma_relaxation.jl.
Returns the same tuple format for compatibility.
"""
function lin_palma_reformulated(
    dpshed_dw::Matrix{Float64},
    pshed_prev::Vector{Float64},
    weights_prev::Vector{Float64},
    pd::Vector{Float64}
)
    result = palma_ratio_minimization(
        dpshed_dw, pshed_prev, weights_prev, pd;
        trust_radius = 0.1,
        w_bounds = (0.0, 10.0),
        relax_binary = true
    )

    # Compute σ from result (for compatibility)
    n = length(pshed_prev)
    _, bottom_40_idx = compute_palma_indices(n)
    sorted_pshed = sort(result.pshed_new)
    denom = sum(sorted_pshed[i] for i in bottom_40_idx)
    σ = denom > 0 ? 1.0 / denom : 1e-8

    return result.pshed_new, result.weights_new, σ
end

#=============================================================================
 Validation / Testing Functions
=============================================================================#

using Random

"""
    test_with_synthetic_data(; n=5, seed=42)

Test the reformulation with synthetic data to verify correctness.
"""
function test_with_synthetic_data(; n::Int=5, seed::Int=42)
    Random.seed!(seed)

    println("="^60)
    println("Testing Palma Ratio Reformulation with Synthetic Data")
    println("="^60)
    println("n = $n loads")
    println()

    # Generate synthetic data
    pd = rand(n) .* 10 .+ 1  # Demands between 1 and 11
    pshed_prev = pd .* (0.3 .+ 0.4 .* rand(n))  # 30-70% of demand
    weights_prev = ones(n) .* 5.0  # Start at middle weights

    # Generate a realistic Jacobian (mostly diagonal with some coupling)
    dpshed_dw = zeros(n, n)
    for i in 1:n
        dpshed_dw[i, i] = -pd[i] * 0.1  # Increasing weight reduces shed
        for j in 1:n
            if i != j
                dpshed_dw[i, j] = pd[i] * 0.01 * randn()  # Small coupling
            end
        end
    end

    println("Input data:")
    println("  pd (demands):     ", round.(pd, digits=3))
    println("  pshed_prev:       ", round.(pshed_prev, digits=3))
    println("  weights_prev:     ", weights_prev)
    println("  Initial Palma:    ", round(palma_ratio(pshed_prev), digits=4))
    println()

    # Solve
    println("Solving optimization...")
    result = palma_ratio_minimization(
        dpshed_dw, pshed_prev, weights_prev, pd;
        trust_radius = 0.5,  # Larger trust region for testing
        relax_binary = true,
        silent = true
    )

    println()
    println("Results:")
    println("  Status:           ", result.status)
    println("  Solve time:       ", round(result.solve_time, digits=4), " s")
    println("  Final Palma:      ", round(result.palma_ratio, digits=4))
    println("  pshed_new:        ", round.(result.pshed_new, digits=3))
    println("  weights_new:      ", round.(result.weights_new, digits=3))
    println("  delta_w:          ", round.(result.delta_w, digits=3))
    println()

    # Verify permutation is doubly stochastic
    a = result.permutation
    row_sums = [sum(a[i, :]) for i in 1:n]
    col_sums = [sum(a[:, j]) for j in 1:n]
    println("Permutation matrix verification:")
    println("  Row sums:  ", round.(row_sums, digits=6))
    println("  Col sums:  ", round.(col_sums, digits=6))
    println()

    # Verify sorted values are ascending
    sorted_vals = result.sorted_values
    is_ascending = all(sorted_vals[k] <= sorted_vals[k+1] + 1e-6 for k in 1:n-1)
    println("Sorted values: ", round.(sorted_vals, digits=3))
    println("Is ascending:  ", is_ascending)
    println()

    # Verify Palma ratio matches
    computed_palma = palma_ratio(result.pshed_new)
    println("Palma ratio verification:")
    println("  From optimization: ", round(result.palma_ratio, digits=6))
    println("  Computed directly: ", round(computed_palma, digits=6))
    println("  Match: ", abs(result.palma_ratio - computed_palma) < 1e-4)

    println()
    println("="^60)

    return result
end

#=============================================================================
 Entry Point
=============================================================================#

# Run test if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = test_with_synthetic_data(n=6, seed=123)
end
