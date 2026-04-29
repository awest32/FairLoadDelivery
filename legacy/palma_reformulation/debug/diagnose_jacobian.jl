#=
Diagnostic script to check Jacobian computation
Run with: julia --project=. script/reformulation/diagnose_jacobian.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))  # debug → reformulation → script → root

using FairLoadDelivery
using PowerModelsDistribution
using Ipopt
using JuMP
using DiffOpt
using LinearAlgebra
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "..", "..", "src", "implementation", "network_setup.jl"))

println("="^70)
println("JACOBIAN DIAGNOSTIC")
println("="^70)
println()

# Setup network
eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/motivation_a.dss", 0.5, String[])

n_loads = length(math["load"])
load_ids = sort(parse.(Int, collect(keys(math["load"]))))

# Get initial weights and demands
weights = Float64[]
pd = Float64[]
for id in load_ids
    load = math["load"][string(id)]
    push!(weights, load["weight"])
    push!(pd, sum(load["pd"]))
end

println("Number of loads: $n_loads")
println("Load IDs: $load_ids")
println("Initial weights: $(weights[1]) (all equal)")
println("Load demands (pd): ", round.(pd, digits=1))
println()

# Solve MLD once to get baseline
mld_model = instantiate_mc_model(
    math,
    LinDist3FlowPowerModel,
    build_mc_mld_shedding_implicit_diff;
    ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
)

optimize!(mld_model.model)
# Extract pshed in sorted load_id order (pshed is indexed by integer load IDs)
pshed_baseline = Float64[]
for id in load_ids
    push!(pshed_baseline, value(mld_model.model[:pshed][id]))
end
println("Baseline pshed (sorted by load_id): ", round.(pshed_baseline, digits=2))
println("Baseline Palma: ", round(sum(sort(pshed_baseline)[end-1:end]) / sum(sort(pshed_baseline)[1:6]), digits=4))
println()

# Now compute Jacobian column by column
println("Computing Jacobian (n=$(n_loads) MLD solves)...")
println()

weight_params = mld_model.model[:fair_load_weights]
pshed_vars = mld_model.model[:pshed]
# Sort keys to ensure consistent ordering with load_ids (keys are integers)
weight_keys = sort(collect(eachindex(weight_params)))
pshed_keys = sort(collect(eachindex(pshed_vars)))

println("Key alignment check:")
println("  load_ids: $load_ids")
println("  weight_keys: $weight_keys")
println("  pshed_keys: $pshed_keys")
println()

jacobian = zeros(n_loads, n_loads)

for j in 1:n_loads
    # Set unit perturbation for weight j
    for (k, wkey) in enumerate(weight_keys)
        perturbation = (k == j) ? 1.0 : 0.0
        DiffOpt.set_forward_parameter(mld_model.model, weight_params[wkey], perturbation)
    end

    optimize!(mld_model.model)
    DiffOpt.forward_differentiate!(mld_model.model)

    for (i, pkey) in enumerate(pshed_keys)
        jacobian[i, j] = DiffOpt.get_forward_variable(mld_model.model, pshed_vars[pkey])
    end
end

println("Jacobian computed!")
println()

# Analyze Jacobian
println("="^70)
println("JACOBIAN ANALYSIS")
println("="^70)
println()

println("Jacobian diagonal (∂pshed[i]/∂w[i]):")
diag_J = diag(jacobian)
for i in 1:n_loads
    sign_str = diag_J[i] < 0 ? "✓ (correct)" : "✗ (WRONG - should be negative!)"
    println("  Load $i: $(round(diag_J[i], digits=4)) $sign_str")
end
println()

n_negative_diag = sum(diag_J .< 0)
n_positive_diag = sum(diag_J .> 0)
n_zero_diag = sum(abs.(diag_J) .< 1e-6)

println("Summary:")
println("  Negative diagonal entries: $n_negative_diag (expected: $n_loads)")
println("  Positive diagonal entries: $n_positive_diag (expected: 0)")
println("  Near-zero diagonal entries: $n_zero_diag")
println()

if n_positive_diag > 0
    println("⚠️  WARNING: Positive diagonal entries mean increasing weight INCREASES shed!")
    println("    This is likely a bug in the MLD formulation or DiffOpt usage.")
end

if n_zero_diag > n_loads / 2
    println("⚠️  WARNING: Many near-zero diagonal entries suggest Jacobian is degenerate.")
    println("    DiffOpt may not be computing gradients correctly.")
end

# Show full Jacobian matrix (small enough for n=15)
println()
println("Full Jacobian matrix (rounded):")
println("Rows = ∂pshed[i], Columns = ∂w[j]")
for i in 1:n_loads
    row_str = join([@sprintf("%8.2f", jacobian[i,j]) for j in 1:n_loads], " ")
    println("  Row $i: $row_str")
end
println()

# Test: What would happen with a small weight change?
println("="^70)
println("PREDICTION TEST")
println("="^70)
println()

# Increase weight on load 1 by 0.1
Δw = zeros(n_loads)
Δw[1] = 0.1
predicted_pshed = pshed_baseline + jacobian * Δw

println("Test: Increase weight[1] by 0.1")
println("  Δpshed[1] predicted: $(round(jacobian[1,1] * 0.1, digits=4))")
println("  Expected: negative (more weight → less shed)")
println()

if jacobian[1,1] > 0
    println("❌ BUG CONFIRMED: Jacobian has wrong sign!")
    println("   The optimization will move weights in the WRONG direction.")
else
    println("✓ Jacobian sign looks correct for load 1")
end

# ======================================================================
# FINITE DIFFERENCE VERIFICATION
# ======================================================================
println()
println("="^70)
println("FINITE DIFFERENCE VERIFICATION")
println("="^70)
println()
println("Computing TRUE Jacobian via finite differences...")
println("This actually perturbs weights and re-solves the MLD problem.")
println()

δ = 0.01  # Perturbation size
fd_jacobian = zeros(n_loads, n_loads)

for j in 1:n_loads
    # Create new math dict with perturbed weight for load j
    math_perturbed = deepcopy(math)
    load_key = string(load_ids[j])
    math_perturbed["load"][load_key]["weight"] += δ

    # Re-solve MLD with perturbed weights
    mld_perturbed = instantiate_mc_model(
        math_perturbed,
        LinDist3FlowPowerModel,
        build_mc_mld_shedding_implicit_diff;
        ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
    )
    JuMP.set_silent(mld_perturbed.model)
    optimize!(mld_perturbed.model)

    # Extract pshed in sorted load_id order (pshed is indexed by integer load IDs)
    pshed_perturbed = Float64[]
    for id in load_ids
        push!(pshed_perturbed, value(mld_perturbed.model[:pshed][id]))
    end

    # Finite difference: (pshed_perturbed - pshed_baseline) / δ
    fd_jacobian[:, j] = (pshed_perturbed .- pshed_baseline) ./ δ

    print("  Column $j done...")
    if j % 5 == 0 || j == n_loads
        println()
    end
end

println()
println("Finite Difference Jacobian diagonal (TRUE ∂pshed[i]/∂w[i]):")
fd_diag = diag(fd_jacobian)
for i in 1:n_loads
    sign_str = fd_diag[i] < 0 ? "✓ (correct)" : "✗ (should be negative)"
    println("  Load $i: $(round(fd_diag[i], digits=4)) $sign_str")
end
println()

# Compare DiffOpt vs Finite Difference
println("="^70)
println("COMPARISON: DiffOpt vs Finite Difference")
println("="^70)
println()
println("Diagonal comparison:")
println(@sprintf("  %-8s %15s %15s %15s", "Load", "DiffOpt", "FiniteDiff", "Ratio"))
println("-"^60)
for i in 1:n_loads
    do_val = diag_J[i]
    fd_val = fd_diag[i]
    ratio = abs(fd_val) > 1e-8 ? do_val / fd_val : Inf
    println(@sprintf("  %-8d %15.2f %15.4f %15.2f", i, do_val, fd_val, ratio))
end
println()

# Summary statistics
mean_ratio = mean(abs.(diag_J) ./ max.(abs.(fd_diag), 1e-8))
println("Average |DiffOpt/FiniteDiff| ratio: $(round(mean_ratio, digits=2))")
println()

fd_negative = sum(fd_diag .< 0)
fd_positive = sum(fd_diag .> 0)
fd_zero = sum(abs.(fd_diag) .< 1e-6)

println("Finite Difference Summary:")
println("  Negative diagonal entries: $fd_negative")
println("  Positive diagonal entries: $fd_positive")
println("  Near-zero diagonal entries: $fd_zero")
println()

if fd_negative > 0
    println("✓ Finite difference shows CORRECT sign (negative)")
    println("  This confirms DiffOpt is computing WRONG sensitivities!")
    println()
    println("RECOMMENDATION: Use finite differences for Jacobian computation")
    println("  in the bilevel optimization loop instead of DiffOpt.")
else
    println("⚠️  Finite difference also shows wrong sign.")
    println("  The issue may be in the MLD formulation itself.")
end
