# Throwaway: per-iteration total-shed trajectory from palma bilevel JLD2s.
using JLD2, Printf
for (label, jld) in [
  ("RELAXED palma bilevel", "results/2026-06-05/bilevel_validation_mn_relaxed/case6_unbalanced_switch_more_meshed_good4integer/palma_absolute/bilevel_mn_relaxed_case6_unbalanced_switch_more_meshed_good4integer_palma_absolute.jld2"),
  ("INTEGER palma bilevel", "results/2026-06-05/bilevel_validation_mn/case6_unbalanced_switch_more_meshed_good4integer/palma_absolute/bilevel_mn_case6_unbalanced_switch_more_meshed_good4integer_palma_absolute.jld2"),
]
  d = JLD2.load(jld)
  lo = get(d, "pshed_lower_history", nothing)
  up = get(d, "pshed_upper_history", nothing)
  println("\n== $label  (completed_iters=$(get(d,"completed_iterations","?"))) ==")
  if lo !== nothing
    print("  iter : "); for i in 1:length(lo); @printf("%7d", i); end; println()
    print("  lower: "); for v in lo; @printf("%7.1f", v); end; println()
    up !== nothing && (print("  upper: "); for v in up; @printf("%7.1f", v); end; println())
  else
    println("  keys: ", join(keys(d), ", "))
  end
end
