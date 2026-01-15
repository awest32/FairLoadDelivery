# DiffOpt Debugging Scripts

This folder contains scripts to diagnose and fix the DiffOpt Jacobian computation issue.

## Problem

DiffOpt returns wrong sensitivities (∂pshed/∂weight) when optimization variables hit their bounds:
- Sign: All positive instead of negative
- Magnitude: ~100,000× too large
- Root cause: KKT-based sensitivity fails at bound-active solutions

## Scripts

| Script | Purpose | Run |
|--------|---------|-----|
| `diagnose_jacobian.jl` | Full MLD diagnostic with DiffOpt vs finite diff | `julia --project=. script/reformulation/debug/diagnose_jacobian.jl` |
| `debug_diffopt_minimal.jl` | 7 isolated tests of increasing complexity | `julia --project=. script/reformulation/debug/debug_diffopt_minimal.jl` |
| `debug_diffopt_stripped.jl` | MLD structure without PowerModelsDistribution | `julia --project=. script/reformulation/debug/debug_diffopt_stripped.jl` |
| `debug_diffopt_barrier.jl` | Tests barrier/regularization fixes | `julia --project=. script/reformulation/debug/debug_diffopt_barrier.jl` |
| `debug_diffopt_reverse.jl` | Tests reverse mode differentiation | `julia --project=. script/reformulation/debug/debug_diffopt_reverse.jl` |

## Recommended Order

1. **`debug_diffopt_minimal.jl`** - Confirms DiffOpt works on simple problems
2. **`debug_diffopt_stripped.jl`** - Shows failure when variables hit bounds
3. **`debug_diffopt_barrier.jl`** - Tests potential fixes
4. **`debug_diffopt_reverse.jl`** - Tests alternative differentiation mode
5. **`diagnose_jacobian.jl`** - Apply fix to full MLD model

## Key Finding

Test 6 in `debug_diffopt_minimal.jl` works perfectly (interior solution):
```
dpshed1/dw1 = -250.0 ✓ (correct negative)
```

But `debug_diffopt_stripped.jl` Model 3 fails (pd[1] at bound):
```
dpshed1/dw1 = +727,951 ✗ (should be -0.22)
```

## Documentation

- `DEBUG_PLAN.md` - Detailed investigation checklist
- `ALTERNATIVES.md` - Comparison of fix options

## Fix Options (see ALTERNATIVES.md)

1. **Barrier terms** - Add log-barrier to keep variables interior
2. **Reverse mode** - Try reverse differentiation instead of forward
3. **Larger regularization** - Increase ε in quadratic term
4. **Custom implicit diff** - Handle bounds explicitly (research contribution)
5. **Finite differences** - Fallback (loses diff. opt. contribution)
