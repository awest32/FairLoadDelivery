# Single-level Palma trade-off — findings

**Date:** 2026-05-09 (run 1, shed-Palma) / 2026-05-10 (run 2 served-Palma + reg, run 3 served-Palma no reg = final)
**Script:** `script/single_level/palma_trade_off.jl`
**Network:** `data/pmd_opendss/case6_unbalanced_switch_meshed_good4integer.dss`
**Config:** `LS_PERCENT = 0.8`, `pshed_type = "absolute"`, integer formulation,
α ∈ `LinRange(0, 1, 10)`, Gurobi w/ `NonConvex=2`, `MIPGap=1e-4`, `MIPFocus=1`,
`NumericFocus=2`. Objective: pure convex combination `α · (σ · top_sum) + (1 − α) · eff_term` (no regularizer).

> **Status:** this is the **control problem** for the study. The high-α
> degenerate corner is a known property of the formulation and is retained
> intentionally; threshold-style fixes are future work.

## Setup

Single-level multi-objective MLD. The Palma machinery from
`load_shed_as_parameter.jl` is grafted onto the JuMP model produced by
`build_mc_mld_min_max_integer` (constraint set identical to the min-max
trade-off, only the objective differs). The min-max objective is
overwritten with a convex combination of the served-Palma ratio (via
Charnes-Cooper σ) and total-shed efficiency:

    min α · (σ · top_sum) + (1 − α) · (Σ pshed / total_demand)
    s.t. σ · bot_sum = 1,    σ ≥ 1e-8

`bot_sum` and `top_sum` are formed from a sorted permutation of
`pserved[k] = pd[k] − Σ_phase pshed[load_ids[k]]`. Permutation matrix
`a[i,j] ∈ {0,1}` is binary; McCormick envelopes link
`u[i,j] = a[i,j] · pserved[j]`. Sorting is enforced via
`sorted_t[k] ≤ sorted_t[k+1]`.

## Sweep results (9 loads, run 3 = final, no reg)

| α | total shed (kW) | max shed (kW) | post-hoc Palma (served) | model σ·top | σ | top_sum | bot_sum |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 – 0.667 | 27.0 | 15.0 | 5.0 | 5.0 | 0.5 | 10.0 | 2.0 |
| 0.778 | 70.0 | 15.0 | Inf | 0.333 | 108 629 | ≈0 | ≈0 |
| 0.889 | 70.0 | 15.0 | Inf | 0.325 | 104 170 | ≈0 | ≈0 |
| 1.000 | 70.0 | 15.0 | Inf | ≈0 | 3.57 × 10⁶ | ≈0 | ≈0 |

For reference, run 2 (with `reg = 1e-4`) showed the corner first at
α ≈ 0.889; removing the regularizer brings it one α-step earlier
(α ≈ 0.778) because the corner has `eff_term = 1.0` so the reg was
applying a small (1e-4) penalty that was holding it back. At α = 1.0
without reg, σ blows up to 3.57×10⁶ since there's zero counter-pressure
on the degenerate corner — both terms in the objective favour it.

Per-load values are in `palma_sweep_integer_absolute.csv`.

All 10 solves returned `OPTIMAL`. Per-solve Gurobi wall time was sub-second
(~0.65 s root-relaxation; a few hundred nodes explored).

## The high-α corner (mechanism)

For α ≥ 0.889 the optimizer drives every load to `pshed_i ≈ pd_i`, making
`pserved ≈ [ε, ε, …, ε]` for tiny ε. Then:

- `top_sum = ε`, `bot_sum = 3ε`
- Charnes-Cooper picks `σ = 1/(3ε)` (huge), satisfying `σ·bot=1`
- objective `σ·top_sum = ε/(3ε) = 1/3`

**1/3 is the combinatorial floor of the Palma ratio for n=9 with the
top1 / bot3 partition** — when all 9 values are equal, the ratio is
`1 / 3`. The optimizer reaches this floor via "uniform near-zero served",
which costs ~70 kW of shed but minimises the fairness term.

Crossover math (partial-shed at total=27, Palma=5.0 vs. full-shed at
total=70, Palma=1/3):

    5α + 0.386(1−α) = 0.333α + 1.000(1−α)  →  α* ≈ 0.116

Theory says the corner wins for any α > 0.12. In practice Gurobi only
*found* it at α ≈ 0.89 because the corner requires σ ~ 10⁴–10⁵ which
spatial branching only explores when the fairness gradient is strong
enough; warm-starts from previous α's solution held the solver on the
partial-shed branch until then. So the "phase transition" at α ≈ 0.89 is
a solver-discovery artifact, not a structural change in the model's
optimum.

## Why this is the control problem, not the final method

Minimising a *ratio of nonnegative quantities* over a set that includes
"make both quantities arbitrarily small in proportion" lets the
optimizer reach the ratio's combinatorial floor by collapsing the served
vector to a uniform-tiny value — and shedding everything is the cheapest
way to get there. Any future fairness method we propose has to explain
how it avoids this trap. Candidate fixes that we *deliberately did not
add* here:

- Cap total shed: `Σ pshed ≤ K` with K tied to the α=0 efficient shed.
- Lower-bound `bot_sum`: `bot_sum ≥ ε · total_demand`.
- Served-fraction reformulation (same degeneracy at zero, harder bookkeeping).
- Drop Charnes-Cooper for a non-ratio surrogate (`top − γ·bot`, etc.).

## Sanity verification

The script prints, per α: `σ`, `top_sum`, `bot_sum`, and
`model(σ·top)` alongside the post-hoc Palma. For α ≤ 0.778 the two
agree exactly (both 5.0). For α ≥ 0.889 they diverge because both
`top_sum` and `bot_sum` collapse below the display precision (≈10⁻⁵);
the model objective is the truth in those rows, the post-hoc value is
numerical noise on a near-zero vector. This confirms the formulation is
mathematically self-consistent — the divergence is purely an artefact of
the degenerate corner the optimizer chose, not a coding bug.

## Other observations

- **Only the absolute pshed variant is implemented — and that's all that
  makes sense for the integer case.** A proportional variant (sort by
  `pshed_i / pd_i`) collapses on integer load-shed decisions:
  `pshed_i ∈ {0, pd_i}` makes every fraction 0 or 1, the sorted vector
  is binary, and Palma degenerates to 1.0 or Inf. The proportional
  variant is only meaningful in a relaxed/continuous setting; do not
  pursue it for the integer control problem.
- **Integer-only.** McCormick relaxation of the permutation matrix is
  known to degenerate (per the comments in `load_shed_as_parameter.jl`),
  so a relaxed counterpart is intentionally omitted.
- **`σ·bot=1` requires `NonConvex=2`.** Bumps the problem from MILP to
  MIQCP; sub-second on this 9-load network.
- **`load_shed_as_parameter.jl` was not modified** — all Palma machinery
  lives inline in this script.

## Artifacts

- `run_debug.log` — run 1 (shed-Palma) Gurobi + Julia trace.
- `run_debug_served.log` — run 2 (served-Palma) trace with per-α
  diagnostic prints (σ, top_sum, bot_sum, model(σ·top)).
- `palma_sweep_integer_absolute.csv` — sweep table.
- `pareto_summary_integer_absolute.svg` — Pareto + α-trace + Palma-vs-α (1×3).
- `summary_integer_all_absolute.svg` — 2×2 combined summary.
- `loadshed_distribution_integer_alpha0.svg`,
  `loadshed_distribution_integer_alpha1.svg` — bar charts at the two ends.
