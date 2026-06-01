"""
    Reverse-mode DiffOpt spike (Frank-Wolfe option-b gate)
    =======================================================

    Goal: confirm DiffOpt REVERSE mode works on the LinDist3Flow multinetwork
    MLD, so the Frank-Wolfe upper level can get ∇_w f = Jᵀv in ONE adjoint
    solve instead of the T·N forward solves the current full-Jacobian build does.

    This script writes NOTHING to src/. It only reads the model and compares.

    Checks (case6, the working bilevel case):
      1. CROSS-CHECK: for a random seed v over pshed, reverse-mode Jᵀv must
         equal transpose(J_forward) * v, where J_forward is the existing,
         in-production forward Jacobian. This is the primary gate.
      2. FD-CHECK: pick one weight j, finite-difference Σ_i v_i·pshed_i(w)
         w.r.t. w_j by re-solving the MLD at w ± h·e_j, compare to (Jᵀv)_j.

    PASS criteria printed at the end. If this passes, lower_level_vjp_mn is safe
    to add to lower_level_mld.jl.

    Usage:
        julia --project=. script/bilevel_validation/spike_reverse_mode.jl
"""

using Revise
using MKL
using FairLoadDelivery
using PowerModelsDistribution, PowerModels
using Ipopt, Gurobi
using HSL_jll
using Random
using DiffOpt
using JuMP
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra

const PMD = PowerModelsDistribution

include("../../src/implementation/lower_level_mld.jl")

# ------------------------------------------------------------------
# Reverse-mode bridge override, mirroring the forward override at
# lower_level_mld.jl:124-126. The NonLinearProgram backend's
# reverse_differentiate!(::Model; tol) defaults to tol=1e-6, which trips the
# same near-degenerate dual-sign assertion the forward path hit; 1e-4 absorbs it.
# ------------------------------------------------------------------
function DiffOpt.reverse_differentiate!(
        model::MOI.Bridges.LazyBridgeOptimizer{DiffOpt.NonLinearProgram.Model})
    return DiffOpt.reverse_differentiate!(model.model; tol = 1e-4)
end

# ------------------------------------------------------------------
# Config — mirror run_validation_mn.jl Step 1 (case6).
# ------------------------------------------------------------------
CASE       = "case6_unbalanced_switch_more_meshed_bd_good4integer"
CASE_FILE  = joinpath(@__DIR__, "../../data/pmd_opendss/$CASE.dss")
LS_PERCENT = 0.8
SELECTED_HOURS    = [4, 12, 15, 18, 22]   # T=5, same as the runner
N_PERIODS         = length(SELECTED_HOURS)
PEAK_STRESS       = 1.0
CENTER_AT_NOMINAL = true
switch_rating = sqrt.([(26.0^2 + 13.1^2), (23.0^2 + 9^2), (21.0^2 + 9.5^2)]) * LS_PERCENT

println("="^70)
println("Reverse-mode DiffOpt spike — $CASE, T=$N_PERIODS")
println("="^70)

# ------------------------------------------------------------------
# Build network + multinetwork (Step 1 of the runner).
# ------------------------------------------------------------------
eng, math, lbs, critical_id = FairLoadDelivery.setup_network(CASE_FILE, LS_PERCENT;
    switch_rating = switch_rating)

mn_data = FairLoadDelivery.create_multinetwork_data_profiled(math, N_PERIODS;
    hours = SELECTED_HOURS, peak_stress = PEAK_STRESS, center_at_nominal = CENTER_AT_NOMINAL)

# ------------------------------------------------------------------
# Helper: instantiate the implicit-diff multinetwork model and collect the
# per-period weight parameters and pshed variables in the SAME flattened order
# diff_forward_full_jacobian_mn uses (period-major, load order = container axes).
# ------------------------------------------------------------------
function build_model_and_handles(mn_data)
    mld = instantiate_mc_model(
        mn_data, LinDist3FlowPowerModel, build_mn_mc_mld_shedding_implicit_diff;
        multinetwork = true, ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])
    model = mld.model
    nw_ids = model[:nw_ids]

    weight_params = []   # flattened JuMP parameters, period-major
    weight_ids    = Int[]
    for (t, n) in enumerate(nw_ids)
        wp = model[Symbol("weights_nw_$(n)")]
        if t == 1
            weight_ids = collect(axes(wp, 1))
        end
        for key in eachindex(wp)
            push!(weight_params, wp[key])
        end
    end

    pshed_vars = []      # flattened pshed vars, period-major
    for n in nw_ids
        pv = model[Symbol("pshed_nw_$(n)")]
        for key in eachindex(pv)
            push!(pshed_vars, pv[key])
        end
    end
    return model, weight_params, pshed_vars, weight_ids
end

# ------------------------------------------------------------------
# FORWARD ground truth: full Jacobian via the existing production function.
# ------------------------------------------------------------------
println("\n[1/3] Forward full Jacobian (ground truth, existing code)...")
Jf, pshed_val, pshed_nw_ids, weight_vals, weight_ids, _ =
    diff_forward_full_jacobian_mn(build_model_and_handles(mn_data)[1], Float64[])
m = length(pshed_val)
println("      J is $(size(Jf)),  Σpshed = $(round(sum(pshed_val), digits=4))")
println("      ‖J‖_∞ = $(round(maximum(abs.(Jf)), sigdigits=4)),  nnz = $(count(!iszero, Jf))/$(length(Jf))")

# ------------------------------------------------------------------
# REVERSE: single adjoint solve for a random seed v → g_rev = Jᵀv.
# Fresh model instance (avoid any state carryover from the forward loop).
# ------------------------------------------------------------------
println("\n[2/3] Reverse adjoint solve (the gate)...")
Random.seed!(1234)
v = randn(m)

model_r, wparams_r, pshed_r, _ = build_model_and_handles(mn_data)
optimize!(model_r)
rev_status = termination_status(model_r)
println("      primal status = $rev_status")

g_rev = fill(NaN, m)
gate1_ok = false
try
    DiffOpt.empty_input_sensitivities!(model_r)
    for (i, var) in enumerate(pshed_r)
        DiffOpt.set_reverse_variable(model_r, var, v[i])
    end
    DiffOpt.reverse_differentiate!(model_r)
    for (j, p) in enumerate(wparams_r)
        g_rev[j] = DiffOpt.get_reverse_parameter(model_r, p)
    end
    global gate1_ok = true
    println("      reverse_differentiate! succeeded; ‖Jᵀv‖_∞ = $(round(maximum(abs.(g_rev)), sigdigits=4))")
catch err
    println("      reverse_differentiate! THREW: $err")
end

# ------------------------------------------------------------------
# Compare reverse Jᵀv against transpose(forward J) * v.
# ------------------------------------------------------------------
if gate1_ok
    g_fwd = transpose(Jf) * v
    abs_err = maximum(abs.(g_rev .- g_fwd))
    den = maximum(abs.(g_fwd)) + 1e-12
    rel_err = abs_err / den
    println("      max |Jᵀv_rev − Jᵀv_fwd|      = $(round(abs_err, sigdigits=4))")
    println("      relative (÷‖Jᵀv_fwd‖_∞)      = $(round(rel_err, sigdigits=4))")
    global gate1_pass = rel_err < 1e-3
else
    global gate1_pass = false
end

# ------------------------------------------------------------------
# [3/3] Independent finite-difference check on one weight component.
# φ(w) = Σ_i v_i · pshed_i(w);  dφ/dw_j = (Jᵀv)_j.
# ------------------------------------------------------------------
println("\n[3/3] Finite-difference cross-check on one weight...")
function phi_at(weights::Vector{Float64}, v::Vector{Float64})
    mdl, _, pv, _ = build_model_and_handles(mn_data)
    # set all weight params
    nw_ids = mdl[:nw_ids]
    idx = 0
    for n in nw_ids
        wp = mdl[Symbol("weights_nw_$(n)")]
        for key in eachindex(wp)
            idx += 1
            JuMP.set_parameter_value(wp[key], weights[idx])
        end
    end
    optimize!(mdl)
    ps = Float64[JuMP.value(var) for var in pv]
    return dot(v, ps)
end

gate2_pass = false
if gate1_ok
    j_test = argmax(abs.(g_rev))          # most informative component
    h = 1e-4
    wp = copy(weight_vals)
    wm = copy(weight_vals)
    wp[j_test] += h
    wm[j_test] -= h
    fd = (phi_at(wp, v) - phi_at(wm, v)) / (2h)
    analytic = g_rev[j_test]
    fd_abs = abs(fd - analytic)
    fd_rel = fd_abs / (abs(analytic) + 1e-12)
    println("      weight j=$j_test:  FD = $(round(fd, sigdigits=5)),  Jᵀv = $(round(analytic, sigdigits=5))")
    println("      |FD − Jᵀv| = $(round(fd_abs, sigdigits=4)),  rel = $(round(fd_rel, sigdigits=4))")
    global gate2_pass = fd_rel < 5e-2     # FD is looser; KKT corner noise + O(h²)
end

# ------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------
println("\n" * "="^70)
println("GATE RESULTS")
println("  [1] reverse mode runs ......... $(gate1_ok ? "PASS" : "FAIL")")
println("  [2] Jᵀv_rev == Jᵀv_fwd ........ $(gate1_pass ? "PASS" : "FAIL")")
println("  [3] FD cross-check ............ $(gate2_pass ? "PASS" : "FAIL")")
overall = gate1_ok && gate1_pass && gate2_pass
println("  OVERALL: $(overall ? "PASS — option (b) is viable" : "FAIL — reverse mode not usable as-is")")
println("="^70)
