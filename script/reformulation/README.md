# Palma Ratio Reformulation

This directory contains the reformulated Palma ratio optimization for fair load shedding.

## Overview

The Palma ratio measures inequality as:
```
Palma = sum(top 10% load shed) / sum(bottom 40% load shed)
```

Lower Palma ratio = more equal distribution of load shedding.

## Files

| File | Description |
|------|-------------|
| `load_shed_as_parameter.jl` | Core optimization: Palma ratio minimization with McCormick envelopes and Charnes-Cooper transformation |
| `opendss_experiment.jl` | Bilevel optimization loop integrating with OpenDSS network data via DiffOpt |
| `validate_reformulation.jl` | Unit tests for the reformulation |
| `diagnose_jacobian.jl` | Diagnostic script to verify Jacobian computation |

## Mathematical Formulation

### Decision Variables
- `Δw[j]`: Weight changes (continuous, bounded by trust region)
- `a[i,j]`: Permutation matrix (binary or relaxed)
- `u[i,j]`: McCormick auxiliary for `a[i,j] * pshed_new[j]`
- `σ`: Charnes-Cooper scaling variable

### Key Expressions
```julia
# P_shed as Taylor expansion (NOT a variable)
pshed_new[j] = pshed_prev[j] + Σₖ (∂pshed[j]/∂w[k]) * Δw[k]

# Sorted values via permutation
sorted[i] = Σⱼ u[i,j]  where u[i,j] ≈ a[i,j] * pshed_new[j]
```

### Charnes-Cooper Transformation
Converts ratio minimization to linear objective:
```
min (top 10% sum) / (bottom 40% sum)
→  min (top 10% sum) * σ   s.t. (bottom 40% sum) * σ = 1
```

## Known Issues and Limitations

### 1. McCormick Relaxation vs Binary Permutation

**Problem**: McCormick envelopes are tight only at binary (0,1) points.

| Setting | Pros | Cons |
|---------|------|------|
| `relax_binary=true` | Fast LP solve | **Sorting breaks** - u values collapse to zero |
| `relax_binary=false` | Correct sorting | **Slow MIQP** - 225 binary variables for n=15 |

**Current approach**: Use `relax_binary=false` with Gurobi time limits.

### 2. Taylor Approximation Validity

The bilevel optimization relies on:
```
pshed_new ≈ pshed_prev + Jacobian * Δw
```

This is only valid for **small** weight changes. If:
- Trust region is too large
- System is highly nonlinear
- Jacobian is inaccurate

The predicted Palma ratio may be **very different** from actual.

### 3. Jacobian Computation (CRITICAL BUG)

**Status: DiffOpt sensitivities are BROKEN for this NLP**

The Jacobian `∂pshed/∂w` computed via DiffOpt forward differentiation produces **WRONG** values:

| Issue | Expected | Actual (DiffOpt) |
|-------|----------|------------------|
| Sign | Negative | **Positive** (all 15 loads) |
| Magnitude | O(demand) ≈ 17-1155 | **O(100,000)** - 100x too large |

**Root cause**: DiffOpt's KKT-based differentiation fails on this non-convex NLP because:
1. "Inertia correction needed" warnings indicate ill-conditioned KKT matrix
2. Non-convex power flow constraints create saddle points
3. Relaxed binary indicators (z_demand) cause degeneracy

**Workaround**: Use **finite differences** instead of DiffOpt:
```julia
# Actually perturb weights and re-solve MLD
δ = 0.01
for j in 1:n_loads
    math_perturbed = deepcopy(math)
    math_perturbed["load"][j]["weight"] += δ
    # ... re-solve and compute (pshed_new - pshed_baseline) / δ
end
```

The `diagnose_jacobian.jl` script now includes both DiffOpt and finite difference comparison.
Run it to verify:
```bash
julia --project=. script/reformulation/diagnose_jacobian.jl
```

### 4. Charnes-Cooper Creates Quadratic Constraints

The constraint `(bottom 40% sum) * σ = 1` is bilinear, requiring:
- Gurobi with `NonConvex=2`, or
- Ipopt (NLP solver)

HiGHS cannot be used (LP/MILP only).

## Alternative Approaches

Located in `src/implementation/other_fair_funcs.jl`:

| Metric | Formula | Complexity | Recommendation |
|--------|---------|------------|----------------|
| **Proportional Fairness** | `max Σ log(pshed + ε)` | NLP | ⭐ Best alternative |
| **Jain's Index** | `max (Σpshed)² / (n·Σpshed²)` | NLP | Good for bounded metric |
| **Min-Max** | `max min(pshed)` | LP | Extreme fairness |
| **Palma Ratio** | `min top10% / bottom40%` | MIQP | Current (slow) |

**Recommendation**: If Palma MIQP is too slow, use **Proportional Fairness** - it has similar fairness properties without requiring sorting/permutation matrices.

## Usage

### Basic experiment
```julia
include("script/reformulation/opendss_experiment.jl")
run_mwe(ls_percent=0.5)
```

### With custom settings
```julia
result = solve_palma_ratio_minimization(
    "ieee_13_aw_edit/motivation_a.dss";
    ls_percent = 0.5,        # 50% capacity → forces load shedding
    iterations = 5,
    trust_radius = 0.1,      # Smaller = more conservative
    experiment_name = "my_experiment"
)
```

### Diagnose Jacobian issues
```bash
julia --project=. script/reformulation/diagnose_jacobian.jl
```

## Debugging

### Palma ratio exploding?
1. Check Jacobian diagonal: should be **negative**
2. Check predicted vs actual Palma after each iteration
3. Reduce trust radius (try 0.01 instead of 0.1)
4. Check if bottom 40% sum → 0 (makes Palma undefined)

### Gurobi timing out?
1. Increase `MIPGap` to 0.01 (1%)
2. Reduce time limit
3. Consider switching to Proportional Fairness

### Segmentation fault?
Usually caused by Julia/solver memory issues. Try:
- Restart Julia session
- Reduce number of iterations
- Use fewer loads

## References

- Charnes-Cooper transformation: Charnes & Cooper (1962)
- McCormick envelopes: McCormick (1976)
- Palma ratio: Cobham & Sumner (2013)
- DiffOpt.jl: https://github.com/jump-dev/DiffOpt.jl
