#=
Test reverse mode differentiation as alternative to forward mode.
Reverse mode seeds the output and retrieves input sensitivities.

Run with: julia --project=. script/reformulation/debug/debug_diffopt_reverse.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))  # debug → reformulation → script → root

using JuMP
using DiffOpt
using Ipopt
using LinearAlgebra
using Printf

println("="^70)
println("DIFFOPT REVERSE MODE TEST")
println("="^70)
println()

n_loads = 3
demand = [100.0, 150.0, 200.0]
capacity = 300.0
ε = 0.001

println("Testing reverse mode vs forward mode on same model")
println("  Demands: $demand, Capacity: $capacity")
println()

#======================================================================
Build model
======================================================================#
model = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model)

@variable(model, w[i=1:n_loads] in Parameter(1.0))
@variable(model, 0 <= pd[i=1:n_loads] <= demand[i])
@variable(model, pshed[1:n_loads] >= 0)

@constraint(model, [i=1:n_loads], pshed[i] == demand[i] - pd[i])
@constraint(model, sum(pd) <= capacity)
@objective(model, Max, sum(w[i] * pd[i] for i in 1:n_loads) - ε * sum(pd[i]^2 for i in 1:n_loads))

optimize!(model)

println("Solution:")
for i in 1:n_loads
    at_bound = value(pd[i]) >= demand[i] - 1e-4 ? " ← AT BOUND" : ""
    println("  pd[$i] = $(round(value(pd[i]), digits=2))$at_bound, pshed[$i] = $(round(value(pshed[i]), digits=2))")
end
println()

#======================================================================
Forward mode Jacobian (what we've been using)
======================================================================#
println("FORWARD MODE (seed parameter, get variable sensitivity)")
println("-"^50)

fwd_jacobian = zeros(n_loads, n_loads)
for j in 1:n_loads
    # Clear previous sensitivities
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(model, w[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(model)
    for i in 1:n_loads
        fwd_jacobian[i, j] = DiffOpt.get_forward_variable(model, pshed[i])
    end
end

println("Forward mode Jacobian ∂pshed/∂w:")
println("             w[1]        w[2]        w[3]")
for i in 1:n_loads
    row = [@sprintf("%12.2f", fwd_jacobian[i,j]) for j in 1:n_loads]
    println("  pshed[$i]: ", join(row, " "))
end
println("  Diagonal: ", round.(diag(fwd_jacobian), digits=2))
println()

#======================================================================
Reverse mode Jacobian
======================================================================#
println("REVERSE MODE (seed variable, get parameter sensitivity)")
println("-"^50)

rev_jacobian = zeros(n_loads, n_loads)

for i in 1:n_loads
    # Clear and set seed for pshed[i]
    for k in 1:n_loads
        DiffOpt.set_reverse_variable(model, pshed[k], k == i ? 1.0 : 0.0)
    end
    DiffOpt.reverse_differentiate!(model)
    for j in 1:n_loads
        # Get ∂pshed[i]/∂w[j] via reverse mode
        rev_jacobian[i, j] = DiffOpt.get_reverse_parameter(model, w[j])
    end
end

println("Reverse mode Jacobian ∂pshed/∂w:")
println("             w[1]        w[2]        w[3]")
for i in 1:n_loads
    row = [@sprintf("%12.2f", rev_jacobian[i,j]) for j in 1:n_loads]
    println("  pshed[$i]: ", join(row, " "))
end
println("  Diagonal: ", round.(diag(rev_jacobian), digits=2))
println()

#======================================================================
Comparison
======================================================================#
println("COMPARISON")
println("-"^50)
println("Forward vs Reverse diagonal:")
for i in 1:n_loads
    f = fwd_jacobian[i,i]
    r = rev_jacobian[i,i]
    match = abs(f - r) < 1e-6 ? "✓ match" : "✗ DIFFER"
    println("  pshed[$i]: Fwd=$(round(f, digits=2)), Rev=$(round(r, digits=2)) $match")
end
println()

#======================================================================
Finite difference ground truth
======================================================================#
println("FINITE DIFFERENCE (ground truth)")
println("-"^50)

δ = 0.01
fd_jacobian = zeros(n_loads, n_loads)

for j in 1:n_loads
    model_fd = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model_fd)
    w_pert = ones(n_loads); w_pert[j] += δ
    @variable(model_fd, w_fd[i=1:n_loads] in Parameter(w_pert[i]))
    @variable(model_fd, 0 <= pd_fd[i=1:n_loads] <= demand[i])
    @variable(model_fd, pshed_fd[1:n_loads] >= 0)
    @constraint(model_fd, [i=1:n_loads], pshed_fd[i] == demand[i] - pd_fd[i])
    @constraint(model_fd, sum(pd_fd) <= capacity)
    @objective(model_fd, Max, sum(w_fd[i] * pd_fd[i] for i in 1:n_loads) - ε * sum(pd_fd[i]^2 for i in 1:n_loads))
    optimize!(model_fd)
    for i in 1:n_loads
        fd_jacobian[i, j] = (value(pshed_fd[i]) - value(pshed[i])) / δ
    end
end

println("Finite diff Jacobian ∂pshed/∂w:")
println("             w[1]        w[2]        w[3]")
for i in 1:n_loads
    row = [@sprintf("%12.4f", fd_jacobian[i,j]) for j in 1:n_loads]
    println("  pshed[$i]: ", join(row, " "))
end
println("  Diagonal: ", round.(diag(fd_jacobian), digits=4))
println()

#======================================================================
Summary
======================================================================#
println("="^70)
println("SUMMARY")
println("="^70)
println()

fwd_neg = sum(diag(fwd_jacobian) .< 0)
rev_neg = sum(diag(rev_jacobian) .< 0)
fd_neg = sum(diag(fd_jacobian) .< 0)

println("Negative diagonal entries (expected: $n_loads):")
println("  Forward mode:  $fwd_neg / $n_loads")
println("  Reverse mode:  $rev_neg / $n_loads")
println("  Finite diff:   $fd_neg / $n_loads")
println()

# Check accuracy vs finite diff
fwd_error = norm(diag(fwd_jacobian) - diag(fd_jacobian)) / norm(diag(fd_jacobian))
rev_error = norm(diag(rev_jacobian) - diag(fd_jacobian)) / norm(diag(fd_jacobian))

println("Relative error vs finite diff:")
println("  Forward mode:  $(round(fwd_error * 100, digits=1))%")
println("  Reverse mode:  $(round(rev_error * 100, digits=1))%")
println()

if rev_neg == n_loads && rev_error < 0.1
    println("✓ REVERSE MODE WORKS! Use this instead of forward mode.")
elseif rev_error < fwd_error
    println("⚠ Reverse mode is better but still has issues.")
else
    println("✗ Neither mode works well. Need barrier/regularization fix.")
end
