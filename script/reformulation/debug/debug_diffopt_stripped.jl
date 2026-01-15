#=
Stripped-down MLD model to test DiffOpt without PowerModelsDistribution complexity.
This isolates the core structure:
- Weights as parameters
- pd = z * demand
- pshed = (1-z) * demand
- Capacity constraint
- Objective: Max Σ w * pd

Run with: julia --project=. script/reformulation/debug_diffopt_stripped.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))  # debug → reformulation → script → root

using JuMP
using DiffOpt
using Ipopt
using LinearAlgebra
using Printf

println("="^70)
println("STRIPPED MLD MODEL - DIFFOPT TEST")
println("="^70)
println()

# Parameters
n_loads = 3
demand = [100.0, 150.0, 200.0]
capacity = 300.0  # Can only serve 300 out of 450 total

println("Setup:")
println("  Loads: $n_loads")
println("  Demands: $demand (total: $(sum(demand)))")
println("  Capacity: $capacity")
println("  Shortfall: $(sum(demand) - capacity) kW must be shed")
println()

#======================================================================
MODEL 1: Bilinear formulation (z_demand relaxed)
This mimics the actual MLD structure
======================================================================#
println("MODEL 1: Bilinear (z * demand)")
println("-"^50)

model1 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model1)

# Weight parameters
@variable(model1, w[i=1:n_loads] in Parameter(1.0))

# Load indicator (relaxed: continuous 0-1)
@variable(model1, 0 <= z[1:n_loads] <= 1)

# Power delivered and shed
@variable(model1, pd[1:n_loads] >= 0)
@variable(model1, pshed[1:n_loads] >= 0)

# Bilinear constraints (this is the challenge!)
@constraint(model1, [i=1:n_loads], pd[i] == z[i] * demand[i])
@constraint(model1, [i=1:n_loads], pshed[i] == (1 - z[i]) * demand[i])

# Capacity constraint
@constraint(model1, sum(pd) <= capacity)

# Objective: maximize weighted delivery
@objective(model1, Max, sum(w[i] * pd[i] for i in 1:n_loads))

optimize!(model1)

println("  Solution:")
for i in 1:n_loads
    println("    Load $i: z=$(round(value(z[i]),digits=4)), pd=$(round(value(pd[i]),digits=2)), pshed=$(round(value(pshed[i]),digits=2))")
end
println()

# Compute Jacobian
println("  DiffOpt Jacobian (∂pshed/∂w):")
jacobian1 = zeros(n_loads, n_loads)
for j in 1:n_loads
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(model1, w[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(model1)
    for i in 1:n_loads
        jacobian1[i, j] = DiffOpt.get_forward_variable(model1, pshed[i])
    end
end

println("             w[1]      w[2]      w[3]")
for i in 1:n_loads
    row = [@sprintf("%10.4f", jacobian1[i,j]) for j in 1:n_loads]
    println("  pshed[$i]: ", join(row, " "))
end
println()
println("  Diagonal: ", round.(diag(jacobian1), digits=4))
n_neg = sum(diag(jacobian1) .< 0)
println("  Negative diagonals: $n_neg / $n_loads (expected: $n_loads)")
println()

#======================================================================
MODEL 2: Linear formulation (pd directly bounded)
Avoids bilinear z * demand
======================================================================#
println("MODEL 2: Linear (pd bounded directly)")
println("-"^50)

model2 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model2)

@variable(model2, w2[i=1:n_loads] in Parameter(1.0))
@variable(model2, 0 <= pd2[i=1:n_loads] <= demand[i])
@variable(model2, pshed2[1:n_loads] >= 0)

# Linear constraint: pshed = demand - pd
@constraint(model2, [i=1:n_loads], pshed2[i] == demand[i] - pd2[i])

# Capacity
@constraint(model2, sum(pd2) <= capacity)

# Objective
@objective(model2, Max, sum(w2[i] * pd2[i] for i in 1:n_loads))

optimize!(model2)

println("  Solution:")
for i in 1:n_loads
    println("    Load $i: pd=$(round(value(pd2[i]),digits=2)), pshed=$(round(value(pshed2[i]),digits=2))")
end
println()

# Jacobian
println("  DiffOpt Jacobian (∂pshed/∂w):")
jacobian2 = zeros(n_loads, n_loads)
for j in 1:n_loads
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(model2, w2[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(model2)
    for i in 1:n_loads
        jacobian2[i, j] = DiffOpt.get_forward_variable(model2, pshed2[i])
    end
end

println("             w[1]      w[2]      w[3]")
for i in 1:n_loads
    row = [@sprintf("%10.4f", jacobian2[i,j]) for j in 1:n_loads]
    println("  pshed[$i]: ", join(row, " "))
end
println()
println("  Diagonal: ", round.(diag(jacobian2), digits=4))
n_neg2 = sum(diag(jacobian2) .< 0)
println("  Negative diagonals: $n_neg2 / $n_loads (expected: $n_loads)")
println()

#======================================================================
MODEL 3: Quadratic regularization (smooth NLP)
======================================================================#
println("MODEL 3: Quadratic regularization")
println("-"^50)

model3 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model3)

@variable(model3, w3[i=1:n_loads] in Parameter(1.0))
@variable(model3, 0 <= pd3[i=1:n_loads] <= demand[i])
@variable(model3, pshed3[1:n_loads] >= 0)

@constraint(model3, [i=1:n_loads], pshed3[i] == demand[i] - pd3[i])
@constraint(model3, sum(pd3) <= capacity)

# Small quadratic regularization makes the problem strictly convex
ε = 0.001
@objective(model3, Max, sum(w3[i] * pd3[i] for i in 1:n_loads) - ε * sum(pd3[i]^2 for i in 1:n_loads))

optimize!(model3)

println("  Solution:")
for i in 1:n_loads
    println("    Load $i: pd=$(round(value(pd3[i]),digits=2)), pshed=$(round(value(pshed3[i]),digits=2))")
end
println()

# Jacobian
println("  DiffOpt Jacobian (∂pshed/∂w):")
jacobian3 = zeros(n_loads, n_loads)
for j in 1:n_loads
    for k in 1:n_loads
        DiffOpt.set_forward_parameter(model3, w3[k], k == j ? 1.0 : 0.0)
    end
    DiffOpt.forward_differentiate!(model3)
    for i in 1:n_loads
        jacobian3[i, j] = DiffOpt.get_forward_variable(model3, pshed3[i])
    end
end

println("             w[1]      w[2]      w[3]")
for i in 1:n_loads
    row = [@sprintf("%10.4f", jacobian3[i,j]) for j in 1:n_loads]
    println("  pshed[$i]: ", join(row, " "))
end
println()
println("  Diagonal: ", round.(diag(jacobian3), digits=4))
n_neg3 = sum(diag(jacobian3) .< 0)
println("  Negative diagonals: $n_neg3 / $n_loads (expected: $n_loads)")
println()

#======================================================================
FINITE DIFFERENCE VERIFICATION for Model 3
======================================================================#
println("FINITE DIFFERENCE VERIFICATION (Model 3)")
println("-"^50)

δ = 0.01
fd_jacobian = zeros(n_loads, n_loads)

for j in 1:n_loads
    # Perturb w[j]
    model_fd = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
    set_silent(model_fd)

    # Set perturbed weights
    w_perturbed = ones(n_loads)
    w_perturbed[j] += δ

    @variable(model_fd, w_fd[i=1:n_loads] in Parameter(w_perturbed[i]))
    @variable(model_fd, 0 <= pd_fd[i=1:n_loads] <= demand[i])
    @variable(model_fd, pshed_fd[1:n_loads] >= 0)

    @constraint(model_fd, [i=1:n_loads], pshed_fd[i] == demand[i] - pd_fd[i])
    @constraint(model_fd, sum(pd_fd) <= capacity)
    @objective(model_fd, Max, sum(w_fd[i] * pd_fd[i] for i in 1:n_loads) - ε * sum(pd_fd[i]^2 for i in 1:n_loads))

    optimize!(model_fd)

    pshed_perturbed = [value(pshed_fd[i]) for i in 1:n_loads]
    pshed_baseline = [value(pshed3[i]) for i in 1:n_loads]

    fd_jacobian[:, j] = (pshed_perturbed .- pshed_baseline) ./ δ
end

println("  Finite Difference Jacobian:")
println("             w[1]      w[2]      w[3]")
for i in 1:n_loads
    row = [@sprintf("%10.4f", fd_jacobian[i,j]) for j in 1:n_loads]
    println("  pshed[$i]: ", join(row, " "))
end
println()

println("  Comparison (DiffOpt / FiniteDiff):")
println("             w[1]      w[2]      w[3]")
for i in 1:n_loads
    row = [@sprintf("%10.2f", abs(fd_jacobian[i,j]) > 1e-8 ? jacobian3[i,j] / fd_jacobian[i,j] : Inf) for j in 1:n_loads]
    println("  pshed[$i]: ", join(row, " "))
end
println()

#======================================================================
SUMMARY
======================================================================#
println("="^70)
println("SUMMARY")
println("="^70)
println()
println("Model 1 (Bilinear):   $n_neg / $n_loads negative diagonals")
println("Model 2 (Linear LP):  $n_neg2 / $n_loads negative diagonals")
println("Model 3 (Regularized): $n_neg3 / $n_loads negative diagonals")
println()

if n_neg3 == n_loads
    println("✓ DiffOpt works correctly on simplified MLD model!")
    println("  Issue is likely in PowerModelsDistribution integration or constraint structure.")
else
    println("✗ DiffOpt shows issues even on simplified model.")
    println("  This suggests a fundamental problem with the API usage or NLP backend.")
end
