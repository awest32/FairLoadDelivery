# DiffOpt Jacobian Debugging Plan

## Problem Statement

DiffOpt returns **wrong sensitivities** for `∂pshed/∂weight`:
- **Sign**: All positive (should be negative for diagonal)
- **Magnitude**: ~100,000 (should be ~0-1000, same order as demands)
- **Ratio vs finite diff**: 30x to 20,000x off

## Architecture Understanding

### Gradient Flow Path
```
fair_load_weights (parameter)
        ↓
    Objective: Max Σ weight[d] * pd[d]
        ↓
    Optimal pd[d] (power delivered)
        ↓
    z_demand[d] = pd[d] / demand[d]  (implicit via load model)
        ↓
    Constraint: pshed[d] = (1 - z_demand[d]) * demand[d]
        ↓
    pshed[d] (what we want sensitivity of)
```

### Expected Sensitivity
```
∂pshed/∂weight = ∂pshed/∂z_demand × ∂z_demand/∂pd × ∂pd/∂weight
               = (-demand) × (1/demand) × (∂pd/∂weight)
               = -∂pd/∂weight

Since ∂pd/∂weight ≥ 0 (higher weight → more delivery),
we expect ∂pshed/∂weight ≤ 0 (NEGATIVE)
```

## Debugging Steps

### Step 1: Run Minimal DiffOpt Tests
```bash
julia --project=. script/reformulation/debug_diffopt_minimal.jl
```

This tests DiffOpt on increasingly complex problems:
1. Simple LP (constraint binding)
2. LP with upper bound
3. Two-resource allocation (trade-off)
4. MLD-like structure (pshed = demand - pd)
5. Two competing loads with capacity
6. Same with quadratic regularization (NLP)
7. Re-optimization effect

**Expected outcome**: Tests 5-6 should show negative `dpshed/dw` if DiffOpt works correctly on the basic structure.

### Step 2: Check Parameter-to-Objective Connection

Verify the weight parameter appears in the objective:

```julia
# In build_mc_mld_shedding_implicit_diff:
objective_fairly_weighted_max_load_served(pm)

# This creates: Max Σ fair_load_weights[d] * pd[d]
```

**Potential issue**: The parameter might not be properly registered in the computational graph.

**Debug**: Add diagnostic prints in the objective function to verify:
```julia
println("Weight for load $d: ", fair_load_weights[d])
println("pd for load $d: ", _PMD.var(pm, nw, :pd)[d])
```

### Step 3: Check pshed Constraint Registration

The constraint `pshed = (1 - z_demand) * demand` must be part of the model for DiffOpt to differentiate through it.

**Verify**: In `build_mc_mld_shedding_implicit_diff`, check that `constraint_load_shed_definition(pm)` is called.

**Found**: Line 118 in `src/prob/mld.jl` confirms it's called.

### Step 4: Investigate z_demand ↔ pd Coupling

The load indicator `z_demand` controls how much load is served. Check how `pd` is constrained:

```julia
# In PowerModelsDistribution, typically:
# pd[d] = z_demand[d] * pd_ref[d]  for on/off model
# or
# 0 ≤ pd[d] ≤ z_demand[d] * pd_ref[d]  for partial shedding
```

**Potential issue**: If `z_demand` is relaxed (continuous 0-1) but `pd` has different bounds, the relationship may be more complex.

### Step 5: Test DiffOpt API Directly

Create a stripped-down MLD model without PowerModelsDistribution:

```julia
# Simplified MLD model
model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))

n_loads = 3
demand = [100.0, 150.0, 200.0]
capacity = 300.0  # Can't serve all

@variable(model, w[1:n_loads] in Parameter(1.0))
@variable(model, 0 <= z[1:n_loads] <= 1)  # load indicator
@variable(model, pd[1:n_loads] >= 0)
@variable(model, pshed[1:n_loads] >= 0)

# pd = z * demand
@constraint(model, [i=1:n_loads], pd[i] == z[i] * demand[i])
# pshed = (1-z) * demand
@constraint(model, [i=1:n_loads], pshed[i] == (1 - z[i]) * demand[i])
# Capacity
@constraint(model, sum(pd) <= capacity)

# Objective: max weighted delivery
@objective(model, Max, sum(w[i] * pd[i] for i in 1:n_loads))

optimize!(model)

# Now differentiate
for j in 1:n_loads
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(model, w[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(model)
    for i in 1:n_loads
        println("∂pshed[$i]/∂w[$j] = ", DiffOpt.get_forward_variable(model, pshed[i]))
    end
end
```

### Step 6: Compare to Reverse Mode

DiffOpt supports both forward and reverse mode. Try reverse mode to see if it gives different (correct) results:

```julia
# Reverse mode: set output sensitivity, get input sensitivity
DiffOpt.set_reverse_variable(model, pshed[i], 1.0)
DiffOpt.reverse_differentiate!(model)
grad_w = DiffOpt.get_reverse_parameter(model, w[j])
```

### Step 7: Check NLP Backend Issues

The "Inertia correction needed" warnings suggest numerical issues. Options:

1. **Use a different solver**: Try KNITRO or another NLP solver
2. **Check tolerances**: Ipopt's default tolerances might be too loose
3. **Regularize the problem**: Add small quadratic terms to improve conditioning

### Step 8: Verify Finite Difference Results

The finite difference results show mixed signs (6 negative, 9 positive). This could be due to:

1. **Network coupling**: Loads share network resources
2. **Block constraints**: Multiple loads in same block
3. **Solution degeneracy**: Multiple optimal solutions with different sensitivities

**Test**: Run finite differences with smaller δ (0.001 instead of 0.01) to check consistency.

## Key Files to Investigate

| File | Purpose | Lines of Interest |
|------|---------|-------------------|
| `src/core/variable.jl` | Weight parameter definition | 160-170 |
| `src/core/objective.jl` | Objective function | 206-216 |
| `src/core/constraint.jl` | pshed constraint | 18-25 |
| `src/prob/mld.jl` | MLD problem build | 28-137 |
| `src/implementation/lower_level_mld.jl` | Jacobian computation | 28-68 |

## Hypotheses to Test

1. **API misuse**: DiffOpt API changed or is being called incorrectly
2. **Parameter registration**: Weight not properly linked to objective
3. **NLP non-convexity**: KKT-based sensitivity fails on non-convex problem
4. **Relaxed indicators**: Continuous z_demand creates degenerate KKT
5. **Constraint structure**: pshed not in the "active" part of the model

## Success Criteria

A successful fix will show:
- Diagonal of Jacobian is **negative** (or mostly negative)
- Magnitude is **O(demand)** not O(100,000)
- DiffOpt matches finite differences within 10%
- Palma ratio **decreases** (not explodes) over iterations

## References

- [DiffOpt.jl Documentation](https://jump.dev/DiffOpt.jl/dev/)
- [DiffOpt Usage Guide](https://jump.dev/DiffOpt.jl/dev/usage/)
- [DiffOpt Thermal Generation Example](https://jump.dev/DiffOpt.jl/dev/examples/Thermal_Generation_Dispatch_Example/)
