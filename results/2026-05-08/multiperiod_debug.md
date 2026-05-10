# Multi-period script debug log — 2026-05-08

Tracking decisions and fixes while running both multiperiod scripts with the new linear load ramp.

## Setup change applied to both scripts

- **`LOAD_SCALE_FACTORS`** changed from a two-bump diurnal Gaussian (baseline 0.85, peaks 1.06 / 1.35) to a **linear ramp `LinRange(0.7, 1.0, 24)`**: first period at 0.7, last period at 1.0.
- `LS_PERCENT = 0.8` left as-is. With this, periods 1–8 have load_scale ≤ 0.8 → no shedding required for those periods; periods ≥ 9 require shed (up to ~20% at the last period).
- `PEAK_TIME_COSTS` left as the Gaussian centered on h=18 — TOU prices remain decoupled from load shape.
- Files touched:
  - `script/single_level/min_max_trade_off_mn.jl`
  - `script/bilevel_validation/run_validation_mn.jl`

## Run plan

1. `min_max_trade_off_mn.jl` — `pshed_type = "absolute"`
2. `min_max_trade_off_mn.jl` — `pshed_type = "proportional"`
3. `run_validation_mn.jl` — `pshed_type = "absolute"`
4. `run_validation_mn.jl` — `pshed_type = "proportional"`

## Run log

### 1. trade_off_mn — absolute
- Status: **passed**
- All 20 alpha sweep points solved `OPTIMAL` for the integer multi-period MILP.
- Plots written to `results/2026-05-08/trade_off_mn/` (3D Pareto, per-period panel, metrics, summary).
- No fixes needed.

### 2. trade_off_mn — proportional
- Status: **passed**
- All 20 alpha sweep points solved `OPTIMAL`. Final objective = 234.56 (kW; constant across alphas at the integer plateau).
- Plots written to `results/2026-05-08/trade_off_mn/`.
- Note: the proportional integer Pareto is expected to degenerate (max shed fraction = 1 whenever any load is shed, so alpha doesn't differentiate solutions) — same observation as on prior runs. No fix needed; this is structural, documented elsewhere.

### 3. validation_mn — absolute
- Status: **passed (with 1 AC ampacity violation at peak)**
- All 20 bilevel iterations succeeded (`LOCALLY_SOLVED` on every iter); no fallback needed.
- Σ pshed (lower) ramped 114.76 → 127.79 across iterations; Σ pshed (upper) stayed in 111.6–116.6 range.
- All 24 periods passed: radial topology, rounded MLD convergence, voltage limits (rounded), switch ampacity (rounded), AC PF convergence, AC voltage limits.
- **One failure**: Period 24 (scale=1.0, peak load) — `switch ampacity (AC PF)` FAIL. Switch currents exceed ratings under AC PF at full demand.
- Output: `results/2026-05-08/bilevel_validation_mn/.../min_max_absolute/`.
- No code-level fixes needed for this run; the AC ampacity violation is a physical-modeling result (LinDist3Flow planning vs AC reality at high demand), not a bug.

### 4. validation_mn — proportional
- Status: **skipped (by request)**
- Decision: proportional integer min-max is structurally degenerate — per-load shed fraction is 0 or 1 in the integer regime, so the max fraction collapses to a constant and the rounded solution converges to the same integer optimum regardless of α. Running this case would only re-confirm that, so we're skipping the run.
