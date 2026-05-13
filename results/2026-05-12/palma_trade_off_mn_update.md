# `palma_trade_off_mn.jl` — silent-zero fix + warmstart

Status: fix landed and re-run. Multi-period single-level Palma sweep now produces a valid Pareto curve across the full α grid instead of silently reporting 0 shed at the efficiency end.

## What was broken

`palma_trade_off_mn/palma_sweep_mn_aggregate_absolute.csv` (pre-fix):

| alpha | agg_total_shed | palma_ratio |
|---|---|---|
| 0.0 | **0.0** | NaN |
| 0.2 | **0.0** | NaN |
| 0.4 | **0.0** | NaN |
| 0.6 | 38661.66 | 2.9393 |
| 0.8 | 38661.66 | 2.9393 |
| 1.0 | 40067.51 | 2.8277 |

The α = 0/0.2/0.4 rows look like "no shed needed," but the **single-period** script (same network, same `LS_PERCENT = 0.8`, same `c_rating_a = Inf`) shows 633 kW shed at α = 0 — gen.pmax is the binding constraint, so shed is required wherever the period's load-scale exceeds 0.8. Twelve periods (t = 9…24) of the multi-period case have load-scale > 0.8 and should be shedding.

## Root cause

`palma_trade_off_mn.jl:299-302` pre-allocated result arrays as `zeros(...)`, then at line 314-317:

```julia
if JuMP.primal_status(mld_mn.model) != MOI.FEASIBLE_POINT
    @warn "non-feasible at alpha=$alpha — skipping"
    continue
end
```

When Gurobi hit the 5-min `TimeLimit` without producing any feasible incumbent, `continue` left the pre-allocated zeros in place. The CSV/plots then reported "0 shed" indistinguishably from "no incumbent found."

Why Gurobi couldn't find an incumbent at low α: the Palma machinery (16×16 binary permutation matrix `a`, McCormick `u`, and the bilinear `σ · bot_sum = 1` constraint) sits on top of the 24-period MLD constraint set even when α = 0 drops the Palma term from the objective. NonConvex=2 spatial branching has to subdivide σ inside the full B&B tree; the root LP relaxation is loose and 5 minutes is not enough to find a first incumbent.

## Fix (two parts)

### 1. NaN-init the result arrays

```julia
total_shed       = fill(NaN, alpha_points, N_PERIODS)
max_shed         = fill(NaN, alpha_points, N_PERIODS)
per_load_dist_a0 = fill(NaN, n_loads, N_PERIODS)
per_load_dist_a1 = fill(NaN, n_loads, N_PERIODS)
```

A skipped α now propagates NaN through `agg_total_shed`, `cost_weighted_*`, and the line/panel plots. Skipped α is visually distinct from "α with real 0-shed solution."

### 2. Pure-efficiency warmstart before the α sweep

A second model `mld_eff` is instantiated from the same `mn_data` using `build_mn_mc_mld_min_max_integer(...; alpha = 0.0, peak_time_costs = PEAK_TIME_COSTS)` — same per-period MLD constraints as `mld_mn` but **no Palma machinery, no σ-bilinearity, no permutation matrix**. It's a clean MIP that Gurobi solves to optimality in ~11 s.

Its solution is then pushed onto `mld_mn` via `JuMP.set_start_value`:

- per-period `pshed` (continuous, broadcast)
- per-period `switch_state`, `z_block`, `z_demand` (binary, broadcast over JuMP containers)
- Palma-specific `a, u, σ`: `perm = sortperm(pserved_warm)` gives ascending positions, `u[i,j] = a[i,j] · pserved[j]`, `σ = 1 / Σ(bot40 pserved)`. The sort constraint, doubly-stochastic constraint, all McCormick inequalities, and `σ · bot_sum = 1` are satisfied by construction.

Broadcast (`JuMP.set_start_value.(palma_var, JuMP.value.(eff_var))`) is required because PMD returns a scalar `VariableRef` for some load/switch families and a JuMP container for others; a naïve `for k in eachindex(...)` loop crashes with `MethodError: no method matching keys(::VariableRef)` on single-phase loads.

## Result after fix

`palma_trade_off_mn/palma_sweep_mn_aggregate_absolute.csv` (post-fix):

| alpha | status | agg_total_shed | palma_ratio | notes |
|---|---|---|---|---|
| 0.0 | OPTIMAL | 6087.52 | 2.734 | warmstart proven optimal within 1% MIPGap |
| 0.2 | TIME_LIMIT | 6087.52 | 2.734 | no improvement on warmstart in 5 min |
| 0.4 | TIME_LIMIT | 6087.52 | 2.734 | same |
| 0.6 | TIME_LIMIT | 6087.52 | 2.734 | same |
| 0.8 | TIME_LIMIT | 6087.52 | 2.734 | same |
| 1.0 | TIME_LIMIT | 18562.69 | 2.297 | pivots to different incumbent at pure-Palma |

Two things worth noting:

1. **The pre-fix α=0.6/0.8 incumbents (38,661 kW shed, Palma 2.94) were dominated on both axes** by the warmstart's incumbent (6,087 kW, Palma 2.73). Gurobi from a cold start was getting trapped in a poor region of the search tree; the warmstart lifts it out.
2. **The Pareto curve is stepped**: flat for α ∈ [0, 0.8], one corner at α = 1. This is consistent with the integer MLD's discrete configuration space (finitely many `(switch_state, z_block, z_demand)` configs); the warmstart config dominates until α = 1 puts zero weight on efficiency and Gurobi accepts ~3× more shed to push Palma from 2.73 down to 2.30.

## Files changed

- `script/single_level/palma_trade_off_mn.jl` — added warmstart block between `palma = add_palma_machinery_aggregate!(...)` and the α-sweep loop; switched result-array init from `zeros` to `fill(NaN, ...)`; updated the skip-branch warning to mention NaN propagation.

## What this doesn't fix

- **Single-period `palma_trade_off.jl` was not touched.** It already finds 633 kW shed at α = 0 because the bilinear MIP is dramatically smaller with one period and one σ, so the 20-min `TimeLimit` is enough.
- **The TIME_LIMIT status at α ∈ [0.2, 0.8]** isn't proof of optimality at those points — it just means Gurobi couldn't beat the warmstart in 5 min. Bumping the limit might surface intermediate Pareto points if they exist; the stepped front suggests they don't, but it's not formally verified.

## Open follow-ups

1. Apply the same warmstart pattern to the bilevel validation runners (`script/bilevel_validation/run_validation_mn.jl`) if they show similar silent-zero behavior at low α — the bilevel upper level should be tractable anyway via the Jacobian sever, but warmstarting the lower-level MLD per α point is cheap insurance.
2. Consider running the α sweep with `TimeLimit = 15 min` for α ∈ {0.2, 0.4, 0.6, 0.8} to formally confirm the flat Pareto region; if Gurobi still returns the warmstart, that's strong evidence the front really is stepped.
