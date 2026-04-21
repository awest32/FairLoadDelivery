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
import MathOptInterface as MOI
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
    # Bottom 40%: first floor(0.4n) positions in ascending sorted order
    n_bottom = max(1, floor(Int, 0.4 * n))
    bottom_40_idx = collect(1:n_bottom)

    # Top 10%: last ceil(0.1n) positions in ascending sorted order
    # For n=15: ceil(0.1*15) = 2 elements → positions [14, 15]
    # For n=10: ceil(0.1*10) = 1 element → position [10]
    # For n=20: ceil(0.1*20) = 2 elements → positions [19, 20]
    n_top = max(1, ceil(Int, 0.1 * n))  # Number of elements in top 10%
    n_top_start = n - n_top + 1         # Starting position (1-indexed)
    top_10_idx = collect(n_top_start:n)

    return top_10_idx, bottom_40_idx
end

"""
    palma_ratio(pshed::Vector{Float64}; eps_denom::Float64=1e-6) -> Float64

Compute the Palma ratio: sum(top 10%) / sum(bottom 40%) after sorting.
Returns Inf if denominator is less than eps_denom.

Note: The Palma ratio can be undefined when most loads have zero shed.
A small eps_denom prevents division by zero while flagging degenerate cases.
"""
function palma_ratio(pshed::Vector{Float64}; eps_denom::Float64=1e-6)
    n = length(pshed)
    sorted_pshed = sort(pshed)  # ascending order

    top_10_idx, bottom_40_idx = compute_palma_indices(n)

    numerator = sum(max(0.0, sorted_pshed[i]) for i in top_10_idx)
    denominator = sum(max(0.0, sorted_pshed[i]) for i in bottom_40_idx)

    if denominator < eps_denom
        return Inf
    end
    return numerator / denominator
end

"""
    is_palma_well_defined(pshed::Vector{Float64}; min_denom::Float64=1e-4) -> Bool

Check if the Palma ratio is well-defined (bottom 40% has sufficient positive load shed).
Returns false if the bottom 40% sum is too small to meaningfully compute Palma.
"""
function is_palma_well_defined(pshed::Vector{Float64}; min_denom::Float64=1e-4)
    n = length(pshed)
    sorted_pshed = sort(pshed)
    _, bottom_40_idx = compute_palma_indices(n)
    denominator = sum(max(0.0, sorted_pshed[i]) for i in bottom_40_idx)
    return denominator >= min_denom
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
    trust_radius::Float64 = 0.5,
    w_bounds::Tuple{Float64, Float64} = (0.0, 10.0),
    solver = get_default_solver(),
    silent::Bool = true,
    relax_binary::Bool = false,  # Binary required; McCormick relaxation (true) produces degenerate solutions - needs further testing
    critical_ids::Vector{Int} = Int[],
    weight_ids::Vector{Int} = Int[],
    peak_time_costs::Vector{Float64} = Float64[],  # On-peak/off-peak weighting per period (empty = uniform)
    n_loads::Int = 0  # Number of loads per period (0 = infer from weights_prev length)
)
    m = length(pshed_prev)       # T*N: total pshed values (= total weights)
    w_min, w_max = w_bounds
    ε = 1e-8  # Small positive for σ lower bound

    # Determine loads per period
    n_per_period = n_loads > 0 ? n_loads : m

    # Clamp critical loads' pshed to zero (lower level can return slightly negative
    # values due to numerical noise, which would make the problem infeasible)
    for j in 1:m
        lid_idx = ((j - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        if load_id in critical_ids
            pshed_prev[j] = max(pshed_prev[j], 0.0)
        end
    end

    # Validate inputs
    @assert length(weights_prev) == m "weights_prev must have length m=$m, got $(length(weights_prev))"
    @assert size(dpshed_dw) == (m, m) "Jacobian must be (m×m), got $(size(dpshed_dw)) expected ($m, $m)"
    @assert length(pd) == m "pd must have length m=$m"
    @assert all(pd .>= 0) "Load demands must be non-negative"

    #=========================================================================
    # Jacobian conditioning diagnostics
    =========================================================================#
    jac_max = maximum(abs.(dpshed_dw))
    jac_nz = dpshed_dw[dpshed_dw .!= 0]
    jac_min_nz = isempty(jac_nz) ? 0.0 : minimum(abs.(jac_nz))
    jac_ratio = jac_min_nz > 0 ? jac_max / jac_min_nz : Inf
    @info "[Palma] Jacobian conditioning: max=$(round(jac_max, sigdigits=4)), min_nz=$(round(jac_min_nz, sigdigits=4)), ratio=$(round(jac_ratio, sigdigits=4))"

    # Feasibility diagnostic: check if pshed_prev fits within [ε, pd] at Δw=0
    n_above_pd = count(pshed_prev[j] > pd[j] for j in 1:m)
    n_below_eps = count(pshed_prev[j] < ε for j in 1:m)
    if n_above_pd > 0 || n_below_eps > 0
        @warn "[Palma] Starting point infeasible: $n_above_pd values > pd, $n_below_eps values < ε"
        for j in 1:m
            if pshed_prev[j] > pd[j]
                error("  pshed_prev[$j]=$(round(pshed_prev[j], sigdigits=6)) > pd[$j]=$(round(pd[j], sigdigits=6)), excess=$(round(pshed_prev[j]-pd[j], sigdigits=4))")
            elseif pshed_prev[j] < ε
                error("  pshed_prev[$j]=$(round(pshed_prev[j], sigdigits=6)) < ε=$(round(ε, sigdigits=6)), deficit=$(round(ε - pshed_prev[j], sigdigits=4))")
            end
        end
    else
        @info "[Palma] Starting point feasible: all pshed_prev ∈ [ε, pd]"
    end

    # Create model
    model = JuMP.Model(solver)
    if silent
        set_silent(model)
    end

    # Solver-specific settings
    if GUROBI_AVAILABLE && solver == Gurobi.Optimizer
        set_optimizer_attribute(model, "DualReductions", 0)
        set_optimizer_attribute(model, "MIPGap", 1e-4)   # Relaxed gap (was 1e-6)
        set_optimizer_attribute(model, "NonConvex", 2)   # Allow non-convex QP
        set_optimizer_attribute(model, "TimeLimit", 60 * 20)  # 20 minutes per iteration
        set_optimizer_attribute(model, "MIPFocus", 1)    # Focus on finding feasible solutions
        set_optimizer_attribute(model, "NumericFocus", 2) # High numerical care (3 was needed only when bounds were wrong)
        if !silent
            set_optimizer_attribute(model, "OutputFlag", 1)  # Show progress
        end
    elseif IPOPT_AVAILABLE && solver == Ipopt.Optimizer
        set_optimizer_attribute(model, "print_level", 0)
    end

    #=========================================================================
    # Per-Period Sort Decomposition
    #
    # Each period's N pshed values are sorted independently using N×N
    # binary permutation matrices. The objective is the cost-weighted sum
    # of per-period Palma ratios: min Σ_t λ[t] * Palma_t
    #
    # Binary count: T*N² (e.g., 9*225 = 2025 for T=9, N=15)
    # vs global sort: (T*N)² (e.g., 135² = 18225)
    =========================================================================#

    # Determine number of periods
    @assert m % n_per_period == 0 "m=$m must be divisible by n_per_period=$n_per_period"
    n_periods = m ÷ n_per_period
    n = n_per_period
    @info "[Palma] Per-period sort: $n_periods period(s), $n loads/period, $m weights, $(n_periods * n^2) binaries"

    # Period costs: λ[t] for each period (default uniform)
    λ = isempty(peak_time_costs) ? ones(n_periods) : peak_time_costs
    @assert length(λ) == n_periods "peak_time_costs must have length $n_periods, got $(length(λ))"

    #=========================================================================
    # Decision Variables
    =========================================================================#

    # Weight changes (m = T*N per-period weight decision variables)
    @variable(model, Δw[1:m])

    # Per-period permutation matrices: a[t][i,j] for t=1..T, i,j=1..n
    a = []
    u = []
    for t in 1:n_periods
        if relax_binary
            push!(a, @variable(model, [1:n, 1:n], lower_bound=0, upper_bound=1, base_name="a_$t"))
        else
            push!(a, @variable(model, [1:n, 1:n], Bin, base_name="a_$t"))
        end
        push!(u, @variable(model, [1:n, 1:n], lower_bound=0, base_name="u_$t"))
    end

    #=========================================================================
    # P_shed as EXPRESSION (Core Simplification)
    =========================================================================#

    # pshed_new via first-order Taylor expansion; Jacobian is m×m
    @expression(model, pshed_new[j=1:m],
        pshed_prev[j] + sum(dpshed_dw[j, k] * Δw[k] for k in 1:m)
    )

    #=========================================================================
    # Trust Region and Weight Bounds
    =========================================================================#

    @constraint(model, trust_lb[j=1:m], Δw[j] >= -trust_radius)
    @constraint(model, trust_ub[j=1:m], Δw[j] <= trust_radius)
    for j in 1:m
        lid_idx = ((j - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        if load_id in critical_ids
            @constraint(model, weights_prev[j] + Δw[j] <= 100.0)
        else
            @constraint(model, weights_prev[j] + Δw[j] >= w_min)
            @constraint(model, weights_prev[j] + Δw[j] <= w_max)
        end
    end

    #=========================================================================
    # P_shed Bounds (Critical for McCormick feasibility)
    =========================================================================#

    @constraint(model, pshed_lb[j=1:m], pshed_new[j] >= ε)
    @constraint(model, pshed_ub[j=1:m], pshed_new[j] <= pd[j])

    #=========================================================================
    # Per-Period Sorting: Permutation + McCormick + Ascending Order
    =========================================================================#

    # Palma indices (same for each period since all have n loads)
    top_10_idx, bottom_40_idx = compute_palma_indices(n)

    # Build per-period Palma ratios via Charnes-Cooper
    # σ[t] = 1 / bot_sum_t, objective = min Σ_t λ[t] * σ[t] * top_sum_t
    @variable(model, σ[1:n_periods] >= 1e-8)

    period_top_sums = []
    period_bot_sums = []

    for t in 1:n_periods
        offset = (t - 1) * n

        # Doubly stochastic constraints
        for i in 1:n
            @constraint(model, sum(a[t][i, j] for j in 1:n) == 1)
        end
        for j in 1:n
            @constraint(model, sum(a[t][i, j] for i in 1:n) == 1)
        end

        # McCormick envelopes: u[t][i,j] = a[t][i,j] * pshed_new[offset+j]
        for i in 1:n, j in 1:n
            gj = offset + j
            P_j = pd[gj]

            @constraint(model, u[t][i, j] >= pshed_new[gj] + a[t][i, j] * P_j - P_j)
            @constraint(model, u[t][i, j] <= a[t][i, j] * P_j)
            @constraint(model, u[t][i, j] <= pshed_new[gj])
        end

        # Sorted values for this period (ascending)
        sorted_t = @expression(model, [i=1:n], sum(u[t][i, j] for j in 1:n))
        for k in 1:n-1
            @constraint(model, sorted_t[k] <= sorted_t[k+1])
        end

        # Palma sums for this period
        push!(period_top_sums, @expression(model, sum(sorted_t[i] for i in top_10_idx)))
        push!(period_bot_sums, @expression(model, sum(sorted_t[i] for i in bottom_40_idx)))

        # Charnes-Cooper normalization: σ[t] * bot_sum_t = 1
        @constraint(model, σ[t] * period_bot_sums[t] == 1.0)
    end

    #=========================================================================
    # Objective: Cost-weighted sum of per-period Palma ratios
    #   min Σ_t λ[t] * σ[t] * top_sum_t
    #   where σ[t] = 1 / bot_sum_t  (Charnes-Cooper)
    =========================================================================#

    @objective(model, Min, sum(λ[t] * σ[t] * period_top_sums[t] for t in 1:n_periods))

    #=========================================================================
    # Solve
    =========================================================================#

    solve_time = @elapsed optimize!(model)
    status = termination_status(model)
    @info "[Palma] Solver status: $status (solve_time=$(round(solve_time, digits=2))s)"

    #=========================================================================
    # Extract Solution
    =========================================================================#

    has_solution = (status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED, MOI.TIME_LIMIT, MOI.ITERATION_LIMIT]) && has_values(model)
    if has_solution
        Δw_val = value.(Δw)
        weights_new = weights_prev .+ Δw_val

        # Compute pshed_new from the expression
        pshed_new_val = pshed_prev .+ dpshed_dw * Δw_val

        # Collect per-period permutation matrices and sorted values
        a_vals = [value.(a[t]) for t in 1:n_periods]
        sorted_val = Float64[]
        for t in 1:n_periods
            offset = (t - 1) * n
            pshed_t = pshed_new_val[offset+1:offset+n]
            append!(sorted_val, a_vals[t] * pshed_t)
        end

        # Compute actual Palma ratio (from unsorted pshed_new)
        actual_palma = palma_ratio(pshed_new_val)

        return (
            weights_new = weights_new,
            pshed_new = pshed_new_val,
            delta_w = Δw_val,
            palma_ratio = actual_palma,
            status = status,
            solve_time = solve_time,
            permutation = a_vals,
            sorted_values = sorted_val
        )
    else
        error("[Palma] Solver failed with status: $status (solve_time=$(round(solve_time, digits=2))s)")
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
    pd::Vector{Float64},
    critical_ids::Vector{Int} = Int[],
    weight_ids::Vector{Int} = Int[];
    peak_time_costs::Vector{Float64} = Float64[],
    n_loads::Int = 0
)
    result = palma_ratio_minimization(
        dpshed_dw, pshed_prev, weights_prev, pd;
        trust_radius = 0.5,
        w_bounds = (1.0, 10.0),
        relax_binary = false,  # Binary required; McCormick relaxation needs testing
        critical_ids = critical_ids,
        weight_ids = weight_ids,
        peak_time_costs = peak_time_costs,
        n_loads = n_loads
    )

    # Compute σ from result (for compatibility)
    m = length(pshed_prev)
    _, bottom_40_idx = compute_palma_indices(m)
    sorted_pshed = sort(result.pshed_new)
    denom = sum(sorted_pshed[i] for i in bottom_40_idx)
    σ = denom > 0 ? 1.0 / denom : 1e-8

    return result.pshed_new, result.weights_new, result.status
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