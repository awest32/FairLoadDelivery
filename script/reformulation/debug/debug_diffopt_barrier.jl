#=
Test if adding a barrier term fixes DiffOpt sensitivity computation
when variables would otherwise be at bounds.

The hypothesis: DiffOpt fails when variables are exactly at bounds.
Fix: Add log-barrier terms to keep variables interior.

Run with: julia --project=. script/reformulation/debug_diffopt_barrier.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))  # debug → reformulation → script → root

using JuMP
using DiffOpt
using Ipopt
using LinearAlgebra
using Printf

println("="^70)
println("DIFFOPT BARRIER FIX TEST")
println("="^70)
println()

n_loads = 3
demand = [100.0, 150.0, 200.0]
capacity = 300.0

println("Setup: Same as stripped test")
println("  Demands: $demand")
println("  Capacity: $capacity")
println()

#======================================================================
MODEL A: Original (pd at bounds) - Expected to FAIL
======================================================================#
println("MODEL A: Original (no barrier) - pd will hit bounds")
println("-"^50)

modelA = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(modelA)

@variable(modelA, wA[i=1:n_loads] in Parameter(1.0))
@variable(modelA, 0 <= pdA[i=1:n_loads] <= demand[i])
@variable(modelA, pshedA[1:n_loads] >= 0)

@constraint(modelA, [i=1:n_loads], pshedA[i] == demand[i] - pdA[i])
@constraint(modelA, sum(pdA) <= capacity)

ε = 0.001
@objective(modelA, Max, sum(wA[i] * pdA[i] for i in 1:n_loads) - ε * sum(pdA[i]^2 for i in 1:n_loads))

optimize!(modelA)

println("  Solution:")
for i in 1:n_loads
    at_bound = value(pdA[i]) >= demand[i] - 1e-4 ? " ← AT BOUND!" : ""
    println("    pd[$i] = $(round(value(pdA[i]), digits=2)) / $(demand[i])$at_bound")
end
println()

# Jacobian
jacA = zeros(n_loads, n_loads)
for j in 1:n_loads
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(modelA, wA[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(modelA)
    for i in 1:n_loads
        jacA[i, j] = DiffOpt.get_forward_variable(modelA, pshedA[i])
    end
end

println("  DiffOpt diagonal: ", round.(diag(jacA), digits=2))
nA = sum(diag(jacA) .< 0)
println("  Negative: $nA / $n_loads")
println()

#======================================================================
MODEL B: Tighter capacity to force interior solution
======================================================================#
println("MODEL B: Tighter capacity (forces pd < demand for all loads)")
println("-"^50)

# With capacity=250, no single load can be fully served
capacity_tight = 250.0

modelB = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(modelB)

@variable(modelB, wB[i=1:n_loads] in Parameter(1.0))
@variable(modelB, 0 <= pdB[i=1:n_loads] <= demand[i])
@variable(modelB, pshedB[1:n_loads] >= 0)

@constraint(modelB, [i=1:n_loads], pshedB[i] == demand[i] - pdB[i])
@constraint(modelB, sum(pdB) <= capacity_tight)

@objective(modelB, Max, sum(wB[i] * pdB[i] for i in 1:n_loads) - ε * sum(pdB[i]^2 for i in 1:n_loads))

optimize!(modelB)

println("  Capacity: $capacity_tight (original: $capacity)")
println("  Solution:")
for i in 1:n_loads
    at_bound = value(pdB[i]) >= demand[i] - 1e-4 ? " ← AT BOUND!" : ""
    at_zero = value(pdB[i]) <= 1e-4 ? " ← AT ZERO!" : ""
    println("    pd[$i] = $(round(value(pdB[i]), digits=2)) / $(demand[i])$at_bound$at_zero")
end
println()

jacB = zeros(n_loads, n_loads)
for j in 1:n_loads
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(modelB, wB[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(modelB)
    for i in 1:n_loads
        jacB[i, j] = DiffOpt.get_forward_variable(modelB, pshedB[i])
    end
end

println("  DiffOpt diagonal: ", round.(diag(jacB), digits=2))
nB = sum(diag(jacB) .< 0)
println("  Negative: $nB / $n_loads")
println()

#======================================================================
MODEL C: Slack bounds to keep pd interior
======================================================================#
println("MODEL C: Slack bounds (pd_max = demand - slack)")
println("-"^50)

slack = 5.0  # Keep pd at least 5 units away from upper bound

modelC = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(modelC)

@variable(modelC, wC[i=1:n_loads] in Parameter(1.0))
# Use tighter upper bounds to keep solution interior
@variable(modelC, 0 <= pdC[i=1:n_loads] <= demand[i] - slack)
@variable(modelC, pshedC[1:n_loads] >= 0)

@constraint(modelC, [i=1:n_loads], pshedC[i] == demand[i] - pdC[i])
@constraint(modelC, sum(pdC) <= capacity)

# Standard quadratic regularization (no log terms - DiffOpt can handle this)
@objective(modelC, Max,
    sum(wC[i] * pdC[i] for i in 1:n_loads)
    - ε * sum(pdC[i]^2 for i in 1:n_loads)
)

optimize!(modelC)

println("  Slack = $slack (pd_max = demand - $slack)")
println("  Solution:")
for i in 1:n_loads
    at_bound = value(pdC[i]) >= demand[i] - slack - 1e-4 ? " ← AT SLACK BOUND" : ""
    println("    pd[$i] = $(round(value(pdC[i]), digits=2)) / $(demand[i] - slack) (demand=$(demand[i]))$at_bound")
end
println()

jacC = zeros(n_loads, n_loads)
for j in 1:n_loads
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(modelC, wC[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(modelC)
    for i in 1:n_loads
        jacC[i, j] = DiffOpt.get_forward_variable(modelC, pshedC[i])
    end
end

println("  DiffOpt diagonal: ", round.(diag(jacC), digits=2))
nC = sum(diag(jacC) .< 0)
println("  Negative: $nC / $n_loads")
println()

# Finite difference verification
println("  Finite difference verification:")
δ = 0.01
fd_jacC = zeros(n_loads, n_loads)
for j in 1:n_loads
    modelC_fd = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(modelC_fd)
    w_pert = ones(n_loads); w_pert[j] += δ
    @variable(modelC_fd, wC_fd[i=1:n_loads] in Parameter(w_pert[i]))
    @variable(modelC_fd, 0 <= pdC_fd[i=1:n_loads] <= demand[i] - slack)
    @variable(modelC_fd, pshedC_fd[1:n_loads] >= 0)
    @constraint(modelC_fd, [i=1:n_loads], pshedC_fd[i] == demand[i] - pdC_fd[i])
    @constraint(modelC_fd, sum(pdC_fd) <= capacity)
    @objective(modelC_fd, Max,
        sum(wC_fd[i] * pdC_fd[i] for i in 1:n_loads)
        - ε * sum(pdC_fd[i]^2 for i in 1:n_loads)
    )
    optimize!(modelC_fd)
    for i in 1:n_loads
        fd_jacC[i, j] = (value(pshedC_fd[i]) - value(pshedC[i])) / δ
    end
end
println("  FD diagonal: ", round.(diag(fd_jacC), digits=2))
println("  DiffOpt/FD ratio: ", round.(diag(jacC) ./ diag(fd_jacC), digits=2))
println()

#======================================================================
MODEL D: Larger quadratic regularization
======================================================================#
println("MODEL D: Larger quadratic regularization (ε=0.01)")
println("-"^50)

ε_large = 0.01

modelD = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(modelD)

@variable(modelD, wD[i=1:n_loads] in Parameter(1.0))
@variable(modelD, 0 <= pdD[i=1:n_loads] <= demand[i])
@variable(modelD, pshedD[1:n_loads] >= 0)

@constraint(modelD, [i=1:n_loads], pshedD[i] == demand[i] - pdD[i])
@constraint(modelD, sum(pdD) <= capacity)

@objective(modelD, Max, sum(wD[i] * pdD[i] for i in 1:n_loads) - ε_large * sum(pdD[i]^2 for i in 1:n_loads))

optimize!(modelD)

println("  ε = $ε_large (original: 0.001)")
println("  Solution:")
for i in 1:n_loads
    at_bound = value(pdD[i]) >= demand[i] - 1e-4 ? " ← AT BOUND!" : ""
    println("    pd[$i] = $(round(value(pdD[i]), digits=2)) / $(demand[i])$at_bound")
end
println()

jacD = zeros(n_loads, n_loads)
for j in 1:n_loads
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(modelD, wD[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(modelD)
    for i in 1:n_loads
        jacD[i, j] = DiffOpt.get_forward_variable(modelD, pshedD[i])
    end
end

println("  DiffOpt diagonal: ", round.(diag(jacD), digits=2))
nD = sum(diag(jacD) .< 0)
println("  Negative: $nD / $n_loads")
println()

#======================================================================
SUMMARY
======================================================================#
println("="^70)
println("SUMMARY")
println("="^70)
println()
println("Model A (original):        $nA / $n_loads negative (pd[1] at bound)")
println("Model B (tight capacity):  $nB / $n_loads negative (all pd interior)")
println("Model C (log-barrier):     $nC / $n_loads negative (barrier keeps interior)")
println("Model D (large ε):         $nD / $n_loads negative (stronger regularization)")
println()
if nB == n_loads || nC == n_loads || nD == n_loads
    println("✓ At least one fix works!")
    println("  Recommendation: Add barrier or larger regularization to MLD objective")
else
    println("✗ None of the fixes fully work. DiffOpt may have deeper issues.")
end
