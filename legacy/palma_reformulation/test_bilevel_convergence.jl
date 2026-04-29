#=
Test bilevel optimization convergence with DiffOpt fix
======================================================

This script runs a few iterations of the bilevel Palma ratio optimization
to verify that DiffOpt now computes correct Jacobians with regularization.

Run with: julia --project=. script/reformulation/test_bilevel_convergence.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using FairLoadDelivery
using PowerModelsDistribution
using Ipopt
using Gurobi
using JuMP
using DiffOpt
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "..", "..", "src", "implementation", "network_setup.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "implementation", "palma_relaxation.jl"))

println("="^70)
println("BILEVEL OPTIMIZATION CONVERGENCE TEST")
println("="^70)
println()

# Setup network
ls_percent = 0.5
eng, math, lbs, critical_id = setup_network("ieee_13_aw_edit/motivation_a.dss", ls_percent, String[])

n_loads = length(math["load"])
load_ids = sort(parse.(Int, collect(keys(math["load"]))))

# Get demands for Palma optimization
pd = Float64[]
for id in load_ids
    push!(pd, sum(math["load"][string(id)]["pd"]))
end

println("Setup: $n_loads loads")
println("Demands: ", round.(pd, digits=1))
println()

function run_bilevel_test(math, load_ids, pd, n_loads)
    # Initialize weights
    current_weights = 10.0 * ones(n_loads)

    # Run bilevel optimization
    n_iterations = 5
    palma_history = Float64[]

    for k in 1:n_iterations
        println("-"^50)
        println("Iteration $k")
        println("-"^50)

        # Update weights in math dict
        for (i, id) in enumerate(load_ids)
            math["load"][string(id)]["weight"] = current_weights[i]
        end

        # Solve lower-level MLD with DiffOpt
        mld_model = instantiate_mc_model(
            math,
            LinDist3FlowPowerModel,
            build_mc_mld_shedding_implicit_diff;
            ref_extensions=[FairLoadDelivery.ref_add_rounded_load_blocks!]
        )
        JuMP.set_silent(mld_model.model)
        optimize!(mld_model.model)

        # Extract pshed in sorted order
        pshed = Float64[]
        for id in load_ids
            push!(pshed, value(mld_model.model[:pshed][id]))
        end

        # Compute current Palma ratio
        sorted_pshed = sort(pshed)
        n_top = max(1, ceil(Int, 0.1 * n_loads))
        n_bottom = max(1, floor(Int, 0.4 * n_loads))
        top_sum = sum(sorted_pshed[end-n_top+1:end])
        bottom_sum = sum(sorted_pshed[1:n_bottom])
        palma = bottom_sum > 1e-6 ? top_sum / bottom_sum : Inf
        push!(palma_history, palma)

        println("  pshed: ", round.(pshed, digits=1))
        println("  Palma ratio: ", @sprintf("%.4f", palma))

        if k == n_iterations
            break  # Don't compute Jacobian on last iteration
        end

        # Compute Jacobian via DiffOpt
        weight_params = mld_model.model[:fair_load_weights]
        pshed_vars = mld_model.model[:pshed]

        jacobian = zeros(n_loads, n_loads)
        for j in 1:n_loads
            # Unit perturbation for weight j
            for i in 1:n_loads
                DiffOpt.set_forward_parameter(mld_model.model, weight_params[load_ids[i]], i == j ? 1.0 : 0.0)
            end
            DiffOpt.forward_differentiate!(mld_model.model)
            for i in 1:n_loads
                jacobian[i, j] = DiffOpt.get_forward_variable(mld_model.model, pshed_vars[load_ids[i]])
            end
        end

        # Check Jacobian diagonal
        diag_J = diag(jacobian)
        n_negative = sum(diag_J .< 0)
        println("  Jacobian diagonal: $n_negative / $n_loads negative")

        # Solve upper-level Palma optimization
        println("  Solving upper-level optimization...")
        pshed_new, weights_new, sigma = lin_palma_w_grad_input(jacobian, pshed, current_weights, pd)

        # Update weights for next iteration
        current_weights = weights_new
        println("  New weights: ", round.(current_weights, digits=2))
    end

    return palma_history
end

# Run the test
palma_history = run_bilevel_test(math, load_ids, pd, n_loads)

println()
println("="^70)
println("CONVERGENCE SUMMARY")
println("="^70)
println()

println("Palma ratio history:")
for (k, p) in enumerate(palma_history)
    println("  Iteration $k: ", @sprintf("%.4f", p))
end
println()

# Check if Palma ratio decreased
if length(palma_history) >= 2
    initial = palma_history[1]
    final = palma_history[end]
    improvement = (initial - final) / initial * 100

    if final < initial
        println("✓ SUCCESS: Palma ratio decreased from $(@sprintf("%.4f", initial)) to $(@sprintf("%.4f", final))")
        println("  Improvement: $(@sprintf("%.1f%%", improvement))")
    else
        println("✗ FAILURE: Palma ratio did not decrease")
        println("  Initial: $(@sprintf("%.4f", initial)), Final: $(@sprintf("%.4f", final))")
    end
end
