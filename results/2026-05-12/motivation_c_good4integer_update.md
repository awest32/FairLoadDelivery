# `motivation_c_good4integer.dss` — setup update

Status: load rescaling + setup_network wiring done. Switch ratings populated from ACPF probe output. **Important caveat below — the rating values for 7 of the 9 switches are physically degenerate and need a sanity-check re-run before being trusted.**

## Files changed

- `data/ieee_13_aw_edit/motivation_c_good4integer.dss` — rescaled all 18 loads. Total ≈ 1860 kW (was ~7100 kW); per-block totals span 150–370 kW; intra-block max-loads span 60–220 kW.
- `src/implementation/network_setup.jl` — added an `elseif case == "ieee_13_aw_edit/motivation_c_good4integer.dss"` branch (around line 114) that sets per-phase `current_rating` for all 9 switches via `calc_apparent_power(P, Q)`.
- `script/single_level/acpf_probe_motivation_c_good4integer.jl` — new probe that lifts gen pmax, loosens (or pins) bus vmax/vmin, picks a switch configuration, and prints per-phase ACPF flows.

## Block partition (induced by switch topology)

```
A (root: RG60, 632, 670)         loads 670a/b/c              total 150 kW   max-load 60
B (633, 634)                      loads 634a/b/c              total 210 kW   max-load 80
C (645, 646)                      loads 645b, 646bc           total 200 kW   max-load 120
D (671, 680, 684, 611, 652)       loads 671, 611c, 652        total 330 kW   max-load 180
E (692, 675)                      loads 692, 675a/b/c         total 370 kW   max-load 110
F (700)                           load 700                    total 200 kW   max-load 200
G (701)                           load 701                    total 180 kW   max-load 180
H (702)                           load 702                    total 220 kW   max-load 220
```

Shed target at `LS_PERCENT=0.8`: **≈ 372 kW**.

## Expected integer Pareto sketch (ignoring radiality cascade)

| Combination | Total shed (kW) | Max load shed (kW) | Position |
|---|---|---|---|
| `{B, C}` | 410 | 120 | fairness end |
| `{G, H}` | 400 | 220 | efficiency end |
| `{B, E}` | 570 | 110 | deeper fairness |

`F → G → H` is a switch-chain; opening 671-700 cascades F+G+H, opening 700-701 cascades G+H. So shedding only G (without H) is not feasible under radiality. Worth verifying against optimizer output.

## ⚠ Switch-rating data quality issue

The ACPF probe was run twice:

1. **First run** (radial: `634675` and `646611` open, `vmax/vmin` = 1.4/0.6) — converged cleanly. Produced sensible kW-scale flows for the two switches it visited before crashing on a `KeyError` (open switches missing from solution dict). This is the source of **632633** and **632645** ratings.
2. **Second run** (linter edited probe: all switches closed, `vmax/vmin` = 1.03/1.03 — voltages pinned, network meshed). Solver reported `LOCALLY_SOLVED` but the resulting P/Q values are **6+ orders of magnitude larger than physically possible** (e.g., switch 634675 phase 1 P = −5.6 GW for a 1.86 MW network). This is what fed the remaining 7 switch ratings.

**Effect:** ratings for `634675`, `646611`, `670671`, `671692`, `671700`, `700701`, `701702` are effectively **non-binding** (10³–10⁶ × actual flow). Ratings for `632633` and `632645` are tight (≈ 90 and 165 |S|).

This may actually be the intended outcome — a "clean" integer Pareto driven by gen-pmax + block topology with switch ampacity relaxed everywhere except the two upstream-of-loads switches. But it's worth confirming. If we want realistic ratings everywhere, re-run the probe with:

- `RADIAL_OPEN = ("634675", "646611")` (revert the linter edit)
- `vmax[:] .= 1.05; vmin[:] .= 0.95` (or `1.4 / 0.6` to be safe)
- Apply the `haskey(...switch, sid) || continue` guard I had drafted (currently rejected — the guard is what allows open switches to be skipped cleanly)

## How to verify the Pareto front

```powershell
julia --project=. script/single_level/min_max_trade_off.jl
```

After updating `case_name` in that script from `motivation_c.dss` to `motivation_c_good4integer.dss` (currently still pointing at the original file at `min_max_trade_off.jl:23`). Look for distinct integer-Pareto points between `(total_shed, max_shed)` extremes; if there are ≥ 2 non-dominated points and they differ from the relaxed solution, the case is working.

## Open follow-ups (for tomorrow)

1. Re-run the ACPF probe in radial config to replace the degenerate ratings.
2. Update `case_name` in `script/single_level/min_max_trade_off.jl:23` (and any companion scripts: `palma_trade_off.jl`, `min_max_trade_off_mn.jl`, etc.) to point at the new dss.
3. Run min_max_trade_off + palma_trade_off on the new case to confirm the Pareto front has multiple non-dominated points.
4. If F-G-H cascade collapses too many options, consider rerouting 701, 702 to attach directly to 671 (independent shed-able blocks).
5. Check that the relaxed lower-level (DiffOpt) still produces good gradients with these load magnitudes — the bilevel loop may be sensitive to the scale change.
