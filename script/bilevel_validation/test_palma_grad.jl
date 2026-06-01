"""
    Standalone finite-difference test of palma_grad_pshed (no solve, no Marguerite).

    Verifies the analytic ∂(palma_value)/∂pshed against central finite differences
    on random data, including the λ weighting and multiple periods.

    Usage:
        julia --project=. script/bilevel_validation/test_palma_grad.jl
"""

using Random
using LinearAlgebra

# Pull in compute_palma_indices / palma_ratio, then the FW analytics.
include("../../src/implementation/load_shed_as_parameter.jl")
include("../../src/implementation/frank_wolfe_palma.jl")

function fd_check(; T=5, n=9, seed=7, h=1e-6)
    Random.seed!(seed)
    m = T * n
    pd     = rand(m) .* 5 .+ 1                      # demands in [1,6]
    pshed  = pd .* (0.2 .+ 0.5 .* rand(m))          # 20–70% shed → served > 0
    λ      = round.(5.0 .+ 25.0 .* rand(T), digits=2)

    g_analytic = palma_grad_pshed(pshed, pd; n_loads=n, peak_time_costs=λ)

    g_fd = zeros(m)
    for j in 1:m
        pp = copy(pshed); pp[j] += h
        pm = copy(pshed); pm[j] -= h
        fp = palma_value(pp, pd; n_loads=n, peak_time_costs=λ)
        fm = palma_value(pm, pd; n_loads=n, peak_time_costs=λ)
        g_fd[j] = (fp - fm) / (2h)
    end

    abs_err = maximum(abs.(g_analytic .- g_fd))
    rel_err = abs_err / (maximum(abs.(g_fd)) + 1e-12)
    println("T=$T, n=$n, m=$m, seed=$seed")
    println("  ‖g_analytic‖_∞ = $(round(maximum(abs.(g_analytic)), sigdigits=5))")
    println("  max |g_analytic − g_fd| = $(round(abs_err, sigdigits=4))")
    println("  relative                = $(round(rel_err, sigdigits=4))")
    pass = rel_err < 1e-4
    println("  $(pass ? "PASS" : "FAIL")")
    return pass
end

println("="^60)
println("palma_grad_pshed finite-difference test")
println("="^60)
ok = true
for s in (7, 42, 101)
    global ok &= fd_check(seed=s)
    println()
end
# Also exercise a different (T,n).
ok &= fd_check(T=3, n=10, seed=5)
println("\nOVERALL: $(ok ? "PASS" : "FAIL")")
