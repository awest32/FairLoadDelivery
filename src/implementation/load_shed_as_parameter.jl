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

Mathematical formulation (served-Palma):
    min_{Δw, A} [Σ_{i∈Top10%} sorted[i]] / [Σ_{i∈Bot40%} sorted[i]]

    s.t. sorted[i]    = Σ_j a[i,j] · pserved_new[j]  (sorting via permutation)
         pserved_new[j] = pd[j] − pshed_new[j]       (served = demand − shed)
         pshed_new[j]   = pshed_prev[j] + Σ_k J[j,k]·Δw[k]  (Taylor expression)
         Σ_j a[i,j] = 1, Σ_i a[i,j] = 1              (doubly stochastic)
         sorted[k] ≤ sorted[k+1]                      (ascending order)
         |Δw| ≤ trust_radius                          (trust region)

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

## P_shed / P_served as Expressions
```
pshed_new[j]   = pshed_prev[j] + Σ_k dpshed_dw[j,k] * Δw[k]
pserved_new[j] = pd[j] − pshed_new[j]
```
Palma sort + Charnes-Cooper operates on `pserved_new` (top10% served / bot40% served).
The objective is pure Palma — no efficiency term, no regularizer.

## McCormick Envelopes for u[i,j] = a[i,j] * pserved_new[j]
Since a[i,j] ∈ {0,1} and pserved_new[j] ∈ [0, P_j]:
1. u[i,j] ≥ 0
2. u[i,j] ≥ pserved_new[j] + a[i,j]*P_j - P_j
3. u[i,j] ≤ a[i,j] * P_j
4. u[i,j] ≤ pserved_new[j]

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
    n_loads::Int = 0,  # Number of loads per period (0 = infer from weights_prev length)
    weight_budget::Float64 = Inf  # Per-period upper bound on Σ_i weights_{t,i}; Inf = no constraint
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

    # Feasibility diagnostic: check if pshed_prev fits within [ε, pd] at Δw=0.
    # Tolerate numerical noise (Jacobian multiply + solver ε); only fail if the
    # violation is large enough to be a real feasibility problem.
    feas_tol = 1e-5  # absolute tolerance; violations below this are silently clamped
    n_above_pd = count(pshed_prev[j] > pd[j] + feas_tol for j in 1:m)
    n_below_eps = count(pshed_prev[j] < ε - feas_tol for j in 1:m)
    # Clamp small numerical drifts so the problem stays feasible.
    for j in 1:m
        if pshed_prev[j] > pd[j] && (pshed_prev[j] - pd[j]) <= feas_tol
            pshed_prev[j] = pd[j]
        end
        if pshed_prev[j] < ε && (ε - pshed_prev[j]) <= feas_tol
            pshed_prev[j] = ε
        end
    end
    if n_above_pd > 0 || n_below_eps > 0
        @warn "[Palma] Starting point infeasible beyond tolerance: $n_above_pd values > pd+$feas_tol, $n_below_eps values < ε-$feas_tol"
        for j in 1:m
            if pshed_prev[j] > pd[j] + feas_tol
                error("  pshed_prev[$j]=$(round(pshed_prev[j], sigdigits=6)) > pd[$j]=$(round(pd[j], sigdigits=6)), excess=$(round(pshed_prev[j]-pd[j], sigdigits=4))")
            elseif pshed_prev[j] < ε - feas_tol
                error("  pshed_prev[$j]=$(round(pshed_prev[j], sigdigits=6)) < ε=$(round(ε, sigdigits=6)), deficit=$(round(ε - pshed_prev[j], sigdigits=4))")
            end
        end
    else
        @info "[Palma] Starting point feasible: all pshed_prev ∈ [ε, pd] (±$feas_tol)"
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
        set_optimizer_attribute(model, "TimeLimit", 60 * 15)  # 15 minutes per iteration
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
    # P_shed and P_served as EXPRESSIONS (Core Simplification)
    #
    # pshed_new is the first-order Taylor estimate from the lower-level Jacobian.
    # pserved_new = pd − pshed_new is the served counterpart; the Palma sort /
    # Charnes-Cooper machinery below operates on pserved_new (top10% / bot40%
    # of *served*, not shed). Efficiency term still uses pshed_new.
    =========================================================================#

    # pshed_new via first-order Taylor expansion; Jacobian is m×m
    @expression(model, pshed_new[j=1:m],
        pshed_prev[j] + sum(dpshed_dw[j, k] * Δw[k] for k in 1:m)
    )
    @expression(model, pserved_new[j=1:m], pd[j] - pshed_new[j])

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

    # Per-period weight budget (upper bound only): Σ_i (weights_prev + Δw)_{t,i} ≤ weight_budget
    if isfinite(weight_budget)
        for t in 1:n_periods
            offset = (t - 1) * n
            @constraint(model,
                sum(weights_prev[offset + i] + Δw[offset + i] for i in 1:n) <= weight_budget)
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

        # McCormick envelopes: u[t][i,j] = a[t][i,j] * pserved_new[offset+j]
        # with bounds 0 ≤ pserved_new[j] ≤ pd[j] (pshed ∈ [ε, pd] ⇒ pserved ∈ [0, pd−ε];
        # we use the looser [0, pd] envelope here for symmetry with the single-level
        # served-Palma scripts).
        for i in 1:n, j in 1:n
            gj = offset + j
            P_j = pd[gj]

            @constraint(model, u[t][i, j] >= pserved_new[gj] + a[t][i, j] * P_j - P_j)
            @constraint(model, u[t][i, j] <= a[t][i, j] * P_j)
            @constraint(model, u[t][i, j] <= pserved_new[gj])
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
    # Objective: pure cost-weighted sum of per-period Palma ratios
    #   min Σ_t λ[t] * σ[t] * top_sum_t       (σ[t] = 1 / bot_sum_t)
    #
    # No efficiency term, no regularizer — the upper level is pure served-Palma.
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
        if status in [MOI.TIME_LIMIT, MOI.ITERATION_LIMIT]
            @warn "[Palma] Solver hit $status with an incumbent — returning suboptimal solution (solve_time=$(round(solve_time, digits=2))s)"
        end
        Δw_val = value.(Δw)
        weights_new = weights_prev .+ Δw_val

        # Compute pshed_new from the expression, then pserved_new = pd − pshed_new
        pshed_new_val   = pshed_prev .+ dpshed_dw * Δw_val
        pserved_new_val = pd .- pshed_new_val

        # Collect per-period permutation matrices and sorted SERVED values
        a_vals = [value.(a[t]) for t in 1:n_periods]
        sorted_val = Float64[]
        for t in 1:n_periods
            offset = (t - 1) * n
            pserved_t = pserved_new_val[offset+1:offset+n]
            append!(sorted_val, a_vals[t] * pserved_t)
        end

        # Compute actual Palma ratio over SERVED (top10% / bot40% of pserved_new)
        actual_palma = palma_ratio(pserved_new_val)

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
    elseif status in [MOI.TIME_LIMIT, MOI.ITERATION_LIMIT]
        # Hit the limit with no incumbent — return no-progress so the bilevel
        # can decide whether to continue rather than crashing mid-run.
        @warn "[Palma] Solver hit $status with NO incumbent — returning no-progress (Δw=0, weights unchanged) (solve_time=$(round(solve_time, digits=2))s)"
        Δw_val = zeros(m)
        pshed_new_val = copy(pshed_prev)
        pserved_new_val = pd .- pshed_new_val
        actual_palma = palma_ratio(pserved_new_val)
        # Identity permutation per period as a placeholder
        a_vals = [Matrix{Float64}(I, n, n) for _ in 1:n_periods]
        sorted_val = Float64[]
        for t in 1:n_periods
            offset = (t - 1) * n
            append!(sorted_val, sort(pserved_new_val[offset+1:offset+n]))
        end
        return (
            weights_new = copy(weights_prev),
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
 Formal Charnes-Cooper variant (MILP, no bilinearity)
=============================================================================#

"""
    palma_ratio_minimization_formal_cc(
        dpshed_dw, pshed_prev, weights_prev, pd; ...
    )

Formal-CC reformulation of [`palma_ratio_minimization`](@ref). The weak-CC
version multiplies σ in at only two places, leaving `σ·u` bilinear terms that
force Gurobi `NonConvex=2`. The formal CC rescales **every** original decision
variable by σ (`z = σ·y`) — the resulting model is a pure MILP:

  * `Δw → Δw_z = σ_t · Δw`
  * `u   → u_z = σ_t · u  =  a · pserved_z` (McCormick on binary × bounded cont.)
  * `pserved_new → pserved_z = σ_t · pserved_new` (linear expression in Δw_z and σ)
  * `σ_t · bot_sum = 1`        → `Σ_{i∈bot40,j} u_z[t][i,j] = 1` (linear)
  * `min σ_t · top_sum`        → `min Σ_{i∈top10,j} u_z[t][i,j]` (linear)
  * every RHS constant `g` becomes `g · σ_t` under rescaling (trust region,
    weight bounds, weight budget, pshed bounds).

Requires the Jacobian to be **block-diagonal in periods**: `J[j,k] = 0` whenever
`t(j) ≠ t(k)`. The lower-level multinetwork MLD is fully separable across
periods (see `build_mn_mc_mld_shedding_implicit_diff`), so this holds by
construction; a runtime guard logs a warning if off-block magnitudes exceed
`block_tol`.

Recovery of original-space variables:
    Δw[k]       = Δw_z[k] / σ_{t(k)}
    pshed_new   = pshed_prev + J · Δw
    weights_new = weights_prev + Δw

σ-bounds: McCormick on `a · pserved_z` needs a finite upper bound on
`pserved_z`. We bound `σ_t ∈ [σ_min, σ_max]` where `σ_max` defaults to
`10 / bot40_sum(pd − pshed_prev)_t` (~10× the previous-iter σ) so the big-M is
data-driven and tight rather than artificial.
"""
function palma_ratio_minimization_formal_cc(
    dpshed_dw::Matrix{Float64},
    pshed_prev::Vector{Float64},
    weights_prev::Vector{Float64},
    pd::Vector{Float64};
    trust_radius::Float64 = 0.5,
    w_bounds::Tuple{Float64, Float64} = (1.0, 10.0),
    solver = get_default_solver(),
    silent::Bool = true,
    critical_ids::Vector{Int} = Int[],
    weight_ids::Vector{Int} = Int[],
    peak_time_costs::Vector{Float64} = Float64[],
    n_loads::Int = 0,
    weight_budget::Float64 = Inf,
    sigma_max_scale::Float64 = 10.0,     # σ_max_t = sigma_max_scale / bot40_sum_prev_t
    sigma_min::Float64 = 1e-8,
    block_tol::Float64 = 1e-6,           # warning threshold on off-block Jacobian entries
)
    m = length(pshed_prev)
    w_min, w_max = w_bounds
    ε = 1e-8

    n_per_period = n_loads > 0 ? n_loads : m
    @assert m % n_per_period == 0 "m=$m must be divisible by n_per_period=$n_per_period"
    n_periods = m ÷ n_per_period
    n = n_per_period

    # Same input preconditioning as the weak-CC version --------------------------
    for j in 1:m
        lid_idx = ((j - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        if load_id in critical_ids
            pshed_prev[j] = max(pshed_prev[j], 0.0)
        end
    end

    @assert length(weights_prev) == m
    @assert size(dpshed_dw) == (m, m)
    @assert length(pd) == m
    @assert all(pd .>= 0)

    feas_tol = 1e-5
    for j in 1:m
        if pshed_prev[j] > pd[j] && (pshed_prev[j] - pd[j]) <= feas_tol
            pshed_prev[j] = pd[j]
        end
        if pshed_prev[j] < ε && (ε - pshed_prev[j]) <= feas_tol
            pshed_prev[j] = ε
        end
    end

    # Block-diagonal Jacobian guard ---------------------------------------------
    n_off_block_diag = 0
    max_off_block_diag = 0.0
    for j in 1:m, k in 1:m
        tj = ((j - 1) ÷ n) + 1
        tk = ((k - 1) ÷ n) + 1
        if tj != tk
            v = abs(dpshed_dw[j, k])
            v > max_off_block_diag && (max_off_block_diag = v)
            v > block_tol && (n_off_block_diag += 1)
        end
    end
    if n_off_block_diag > 0
        @warn "[Palma formal CC] Jacobian off-block-diagonal entries exceed $block_tol: $n_off_block_diag entries, max=$max_off_block_diag. Formal CC assumes block-diagonality — solution may be inexact."
    else
        @info "[Palma formal CC] Jacobian is block-diagonal (max off-block-diag = $max_off_block_diag)"
    end

    # Per-period σ bounds derived from previous-iter bot_sum --------------------
    top_10_idx, bottom_40_idx = compute_palma_indices(n)
    σ_max = zeros(n_periods)
    for t in 1:n_periods
        offset = (t - 1) * n
        pserved_prev_t = sort([max(pd[offset + i] - pshed_prev[offset + i], 0.0) for i in 1:n])
        bot_sum_prev = sum(pserved_prev_t[i] for i in bottom_40_idx)
        # Default: 10x previous σ; if bot_sum_prev is degenerate, fall back to a safe ceiling.
        σ_max[t] = bot_sum_prev > 1e-6 ? sigma_max_scale / bot_sum_prev : 1e6
    end

    λ = isempty(peak_time_costs) ? ones(n_periods) : peak_time_costs
    @assert length(λ) == n_periods

    @info "[Palma formal CC] T=$n_periods, N=$n, m=$m, σ_max range=[$(round(minimum(σ_max), sigdigits=3)), $(round(maximum(σ_max), sigdigits=3))]"

    # Build the MILP ------------------------------------------------------------
    model = JuMP.Model(solver)
    silent && set_silent(model)

    if GUROBI_AVAILABLE && solver == Gurobi.Optimizer
        set_optimizer_attribute(model, "MIPGap",       1e-4)
        set_optimizer_attribute(model, "TimeLimit",    60 * 15)
        set_optimizer_attribute(model, "MIPFocus",     1)
        set_optimizer_attribute(model, "NumericFocus", 2)
        # NB: NonConvex=2 NOT needed — formal CC is a MILP.
        if !silent
            set_optimizer_attribute(model, "OutputFlag", 1)
        end
    end

    # σ_t ∈ [σ_min, σ_max[t]]
    @variable(model, σ[t = 1:n_periods])
    for t in 1:n_periods
        JuMP.set_lower_bound(σ[t], sigma_min)
        JuMP.set_upper_bound(σ[t], σ_max[t])
    end

    # Binary permutation matrices — unchanged from weak CC
    a = Any[]
    for t in 1:n_periods
        push!(a, @variable(model, [1:n, 1:n], Bin, base_name = "a_$t"))
    end

    # Rescaled weight changes
    @variable(model, Δw_z[1:m])

    # Rescaled pserved (linear expression). By block-diagonality, only k with
    # t(k) == t(j) contributes — we still sum over all k since J[j,k] ≈ 0 off-block.
    @expression(model, pserved_z[j = 1:m],
        σ[((j - 1) ÷ n) + 1] * (pd[j] - pshed_prev[j])
        - sum(dpshed_dw[j, k] * Δw_z[k] for k in 1:m)
    )

    # Rescaled u: u_z[t][i,j] = a[t][i,j] · pserved_z[offset+j] via McCormick
    u_z = Any[]
    for t in 1:n_periods
        push!(u_z, @variable(model, [1:n, 1:n], lower_bound = 0, base_name = "u_z_$t"))
    end

    for t in 1:n_periods
        offset = (t - 1) * n
        # Doubly stochastic on the binary a (unchanged)
        for i in 1:n
            @constraint(model, sum(a[t][i, j] for j in 1:n) == 1)
        end
        for j in 1:n
            @constraint(model, sum(a[t][i, j] for i in 1:n) == 1)
        end

        # McCormick for u_z = a · pserved_z with bound pserved_z ≤ pd[gj] · σ_max[t]
        for i in 1:n, j in 1:n
            gj = offset + j
            P_max = pd[gj] * σ_max[t]
            @constraint(model, u_z[t][i, j] >= pserved_z[gj] + a[t][i, j] * P_max - P_max)
            @constraint(model, u_z[t][i, j] <= a[t][i, j] * P_max)
            @constraint(model, u_z[t][i, j] <= pserved_z[gj])
        end

        # Ascending sort on rescaled sorted values (ordering preserved by σ > 0)
        sorted_z_t = @expression(model, [i = 1:n], sum(u_z[t][i, j] for j in 1:n))
        for k in 1:(n - 1)
            @constraint(model, sorted_z_t[k] <= sorted_z_t[k + 1])
        end

        # Formal-CC denominator normalization (LINEAR): Σ_{i∈bot40,j} u_z[t][i,j] = 1
        @constraint(model, sum(u_z[t][i, j] for i in bottom_40_idx, j in 1:n) == 1)
    end

    # Rescaled trust region
    for j in 1:m
        t = ((j - 1) ÷ n) + 1
        @constraint(model, Δw_z[j] >= -trust_radius * σ[t])
        @constraint(model, Δw_z[j] <=  trust_radius * σ[t])
    end

    # Rescaled weight bounds
    for j in 1:m
        lid_idx = ((j - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        t = ((j - 1) ÷ n) + 1
        if load_id in critical_ids
            @constraint(model, weights_prev[j] * σ[t] + Δw_z[j] <= 100.0 * σ[t])
        else
            @constraint(model, weights_prev[j] * σ[t] + Δw_z[j] >= w_min * σ[t])
            @constraint(model, weights_prev[j] * σ[t] + Δw_z[j] <= w_max * σ[t])
        end
    end

    # Rescaled per-period weight budget
    if isfinite(weight_budget)
        for t in 1:n_periods
            offset = (t - 1) * n
            sum_prev = sum(weights_prev[offset + i] for i in 1:n)
            @constraint(model,
                sum(Δw_z[offset + i] for i in 1:n) <= (weight_budget - sum_prev) * σ[t])
        end
    end

    # Rescaled pshed bounds: pshed ∈ [ε, pd] ⇔ pserved ∈ [0, pd-ε] ⇒
    #     pserved_z ∈ [0, (pd-ε)·σ_t]
    for j in 1:m
        t = ((j - 1) ÷ n) + 1
        @constraint(model, pserved_z[j] >= 0)
        @constraint(model, pserved_z[j] <= (pd[j] - ε) * σ[t])
    end

    # Linear objective: min Σ_t λ_t · Σ_{i∈top10,j} u_z[t][i,j]
    @objective(model, Min,
        sum(λ[t] * sum(u_z[t][i, j] for i in top_10_idx, j in 1:n)
            for t in 1:n_periods))

    solve_time = @elapsed optimize!(model)
    status = termination_status(model)
    @info "[Palma formal CC] Solver status: $status (solve_time=$(round(solve_time, digits=2))s)"

    has_solution = (status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL,
                               MOI.ALMOST_LOCALLY_SOLVED, MOI.TIME_LIMIT, MOI.ITERATION_LIMIT]) && has_values(model)

    if has_solution
        if status in [MOI.TIME_LIMIT, MOI.ITERATION_LIMIT]
            @warn "[Palma formal CC] Solver hit $status with an incumbent — returning suboptimal solution"
        end

        σ_val   = value.(σ)
        Δw_z_val = value.(Δw_z)
        Δw_val   = similar(Δw_z_val)
        for k in 1:m
            t = ((k - 1) ÷ n) + 1
            Δw_val[k] = Δw_z_val[k] / σ_val[t]
        end

        weights_new     = weights_prev .+ Δw_val
        pshed_new_val   = pshed_prev   .+ dpshed_dw * Δw_val
        pserved_new_val = pd           .- pshed_new_val

        a_vals = [value.(a[t]) for t in 1:n_periods]
        sorted_val = Float64[]
        for t in 1:n_periods
            offset = (t - 1) * n
            append!(sorted_val, a_vals[t] * pserved_new_val[offset+1:offset+n])
        end

        actual_palma = palma_ratio(pserved_new_val)

        return (
            weights_new   = weights_new,
            pshed_new     = pshed_new_val,
            delta_w       = Δw_val,
            palma_ratio   = actual_palma,
            status        = status,
            solve_time    = solve_time,
            permutation   = a_vals,
            sorted_values = sorted_val,
        )
    elseif status in [MOI.TIME_LIMIT, MOI.ITERATION_LIMIT]
        @warn "[Palma formal CC] Solver hit $status with NO incumbent — returning no-progress"
        Δw_val = zeros(m)
        pshed_new_val = copy(pshed_prev)
        pserved_new_val = pd .- pshed_new_val
        actual_palma = palma_ratio(pserved_new_val)
        a_vals = [Matrix{Float64}(I, n, n) for _ in 1:n_periods]
        sorted_val = Float64[]
        for t in 1:n_periods
            offset = (t - 1) * n
            append!(sorted_val, sort(pserved_new_val[offset+1:offset+n]))
        end
        return (
            weights_new   = copy(weights_prev),
            pshed_new     = pshed_new_val,
            delta_w       = Δw_val,
            palma_ratio   = actual_palma,
            status        = status,
            solve_time    = solve_time,
            permutation   = a_vals,
            sorted_values = sorted_val,
        )
    else
        error("[Palma formal CC] Solver failed with status: $status (solve_time=$(round(solve_time, digits=2))s)")
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
    pd::Vector{Float64};
    critical_ids::Vector{Int} = Int[],
    weight_ids::Vector{Int} = Int[],
    peak_time_costs::Vector{Float64} = Float64[],
    n_loads::Int = 0,
    weight_budget::Float64 = Inf
)
    result = palma_ratio_minimization(
        dpshed_dw, pshed_prev, weights_prev, pd;
        trust_radius = 0.5,
        w_bounds = (1.0, 10.0),
        relax_binary = false,  # Binary required; McCormick relaxation needs testing
        critical_ids = critical_ids,
        weight_ids = weight_ids,
        peak_time_costs = peak_time_costs,
        n_loads = n_loads,
        weight_budget = weight_budget
    )

    # Compute σ from result (for compatibility). σ is the Charnes-Cooper scaling
    # 1/bot_sum, where bot_sum is now the bottom-40% of SERVED (pd − pshed).
    m = length(pshed_prev)
    _, bottom_40_idx = compute_palma_indices(m)
    pserved_new = pd .- result.pshed_new
    sorted_pserved = sort(pserved_new)
    denom = sum(sorted_pserved[i] for i in bottom_40_idx)
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
    println("  Initial Palma:    ", round(palma_ratio(pd .- pshed_prev), digits=4))
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

    # Verify Palma ratio matches (served-Palma: pd − pshed_new)
    computed_palma = palma_ratio(pd .- result.pshed_new)
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