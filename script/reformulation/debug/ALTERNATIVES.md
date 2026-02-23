# Alternatives to Fix DiffOpt Jacobian Computation

## Problem Summary

DiffOpt's forward differentiation fails when optimization variables hit bounds:
- Returns values 10^6 × wrong magnitude
- Wrong sign (positive instead of negative)
- "Inertia correction" warnings indicate KKT matrix issues

## Option Comparison

| Option | Complexity | Preserves Diff. Opt. | Performance | Recommendation |
|--------|------------|---------------------|-------------|----------------|
| 1. Barrier terms | Low | ✓ Yes | Good | **Try first** |
| 2. Reverse mode | Low | ✓ Yes | Good | Try second |
| 3. Larger regularization | Low | ✓ Yes | Good | May distort solution |
| 4. Different NLP solver | Medium | ✓ Yes | Variable | Try KNITRO |
| 5. Implicit diff (custom) | High | ✓ Yes | Best | Research contribution |
| 6. Finite differences | Low | ✗ No | O(n) slower | Fallback only |
| 7. Alternative fairness | Low | ✗ No | Good | Different metric |

## Option 1: Barrier Terms (Recommended First Try)

Add log-barrier to keep variables strictly interior to bounds:

```julia
μ = 1.0  # Barrier weight (tune this)
@objective(model, Max,
    sum(w[i] * pd[i] for i in 1:n)
    - ε * sum(pd[i]^2 for i in 1:n)  # Regularization
    + μ * sum(log(pd[i] + δ) + log(demand[i] - pd[i] + δ) for i in 1:n)  # Barrier
)
```

**Pros**: Simple to implement, preserves differentiable optimization
**Cons**: Changes optimal solution slightly, need to tune μ

## Option 2: Reverse Mode Differentiation

DiffOpt supports reverse mode - may handle bounds differently:

```julia
# Instead of forward mode:
# DiffOpt.set_forward_parameter(model, w[j], 1.0)
# DiffOpt.forward_differentiate!(model)
# dx = DiffOpt.get_forward_variable(model, x)

# Try reverse mode:
DiffOpt.set_reverse_variable(model, pshed[i], 1.0)  # Seed output
DiffOpt.reverse_differentiate!(model)
dw = DiffOpt.get_reverse_parameter(model, w[j])  # Get input sensitivity
```

**Pros**: Different numerical path, may avoid bound issues
**Cons**: Same KKT matrix issues may persist

## Option 3: Larger Quadratic Regularization

Current ε=0.001 may be too small. Try ε=0.01 or 0.1:

```julia
ε = 0.01  # Stronger regularization
@objective(model, Max, sum(w[i] * pd[i]) - ε * sum(pd[i]^2))
```

**Pros**: Simplest change, makes problem more convex
**Cons**: Larger ε distorts the economic dispatch solution

## Option 4: Different NLP Solver

KNITRO may have better sensitivity computation than Ipopt:

```julia
using KNITRO
model = Model(() -> DiffOpt.diff_optimizer(KNITRO.Optimizer))
```

Or try with Ipopt's different linear solvers:
```julia
set_attribute(model, "linear_solver", "ma57")  # Requires HSL
```

**Pros**: May resolve numerical issues
**Cons**: KNITRO is commercial, HSL requires license

## Option 5: Custom Implicit Differentiation

Implement implicit differentiation directly on the KKT conditions, handling bound constraints explicitly. This is the most robust but requires significant development.

Key idea: At bounds, the sensitivity is zero (variable can't move). The current DiffOpt implementation may not handle this correctly for NLPs.

```julia
# Pseudocode for custom implicit diff
function custom_jacobian(model, w, pshed)
    # Identify active constraints (which bounds are binding)
    active_bounds = find_active_bounds(model)

    # Partition variables into free and fixed
    # Compute reduced KKT system for free variables only
    # Sensitivities for fixed variables = 0
end
```

**Pros**: Most robust, handles bounds correctly
**Cons**: High implementation effort, but could be a paper contribution

## Option 6: Finite Differences (Fallback)

If differentiable optimization can't be made to work:

```julia
function finite_diff_jacobian(math, weights, δ=0.01)
    baseline = solve_mld(math, weights)
    jacobian = zeros(n, n)
    for j in 1:n
        w_perturbed = copy(weights)
        w_perturbed[j] += δ
        perturbed = solve_mld(math, w_perturbed)
        jacobian[:, j] = (perturbed.pshed - baseline.pshed) / δ
    end
    return jacobian
end
```

**Pros**: Always works, simple
**Cons**: O(n) more solves per iteration, loses diff. opt. contribution

## Option 7: Alternative Fairness Metrics

The Palma ratio requires sorting, which needs the Jacobian. Alternative metrics don't:

| Metric | Formula | Needs Jacobian? |
|--------|---------|-----------------|
| Proportional Fairness | max Σ log(pd) | No (direct NLP) |
| Jain's Index | max (Σpd)²/(n·Σpd²) | No (direct NLP) |
| Min-Max Fairness | max min(pd) | No (LP) |

These can be optimized directly without bilevel structure:

```julia
@objective(model, Max,
    sum(w[i] * pd[i]) + λ * sum(log(pd[i] + ε))  # Proportional fairness term
)
```

**Pros**: Avoids the problem entirely, simpler optimization
**Cons**: Different fairness metric, not Palma ratio

## Recommended Investigation Order

1. **Run `debug_diffopt_barrier.jl`** - Test if barrier terms fix the issue
2. **Try reverse mode** - Add to barrier test, compare results
3. **Test KNITRO** - If available, try different solver
4. **Consider custom implicit diff** - If above fail, this is the robust solution
5. **Finite differences as validation** - Always useful for checking

## Debug Scripts

| Script | Purpose |
|--------|---------|
| `debug_diffopt_minimal.jl` | 7 isolated tests of DiffOpt behavior |
| `debug_diffopt_stripped.jl` | MLD structure without PowerModelsDistribution |
| `debug_diffopt_barrier.jl` | Tests barrier/regularization fixes |
| `debug_diffopt_reverse.jl` | Tests reverse mode (to be created) |

## Key Insight

The fundamental issue is that DiffOpt's NLP sensitivity computation assumes the KKT system is non-degenerate. When variables hit bounds:

1. The active set changes (different constraints binding)
2. The KKT Jacobian becomes singular at the boundary
3. Ipopt's interior-point method approaches bounds asymptotically
4. Numerical errors explode when computing sensitivities

This is a known limitation of KKT-based sensitivity analysis. The solutions are:
- Keep variables interior (barriers)
- Handle bounds explicitly (custom implementation)
- Use different differentiation approach (autodiff through iterations)
