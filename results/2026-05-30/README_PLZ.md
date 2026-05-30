# README

These results are switch-configuration exploration, not paper inputs.

The `_bd_` variant of case6 adds a new switched branch from bus **d** to bus
**b** on top of `case6_unbalanced_switch_more_meshed_good4integer`. The
purpose of this run was to check whether the extra b-d branch unlocks new
radial trees and changes the integer load-shed solution.

**Result: no effect on the integer solution.** Adding the b-d branch did
not change which loads were shed under any of the integer formulations
(efficiency, min-max, palma) at T=5. The bilevel solver picks the same
topology and the same load-shed pattern as the non-BD baseline, so the
new switch is non-binding in the integer regime.

Related sibling artifacts from the same exploration (T=5):

- `results/2026-05-30/trade_off_mn/efficiency_relaxed_trade_off_mn_more_meshed_bd_6_bus_absolute.jld2`
- `results/2026-05-30/trade_off_mn/min_max_relaxed_trade_off_mn_more_meshed_bd_6_bus_absolute.jld2`

These are kept on disk for reference but should not be loaded into the
paper Pareto plots. The post-hoc fairness pareto script
(`script/post_hoc_fairness_pareto.jl`) targets `more_meshed_6_bus` (the
non-BD variant) and ignores these files by design.
