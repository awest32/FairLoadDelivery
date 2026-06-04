#=
Frank-Wolfe upper level for served-Palma (option b)
===================================================

Replaces the MILP upper level (`lin_palma_reformulated`) with a projection-free
Frank-Wolfe loop over the weight box. The FW algorithm mirrors Marguerite.jl's
`solve(f, Box, w0; grad, monotonic=false)` path (box LMO + 2/(t+2) step + FW gap),
but is inlined rather than imported: Marguerite pins ForwardDiff 1.x, which
conflicts with PowerModelsDistribution 0.16's ForwardDiff<1 cap in this env. Our
path uses only the LMO + step, never Marguerite's AD machinery, so the inline
version is behaviorally identical for this problem.

Per FW iteration the cost is exactly TWO lower-level MLD solves:
  1. PRIMAL  — solve MLD at current weights w → pshed*(w). Gives the objective
     value f(w) AND leaves the model at its KKT point (the adjoint's lin. point).
  2. ADJOINT — one DiffOpt reverse solve: seed pshed-sensitivities with
     v = ∂f/∂pshed (analytic, `palma_grad_pshed`), read weight-sensitivities
     = Jᵀv = ∇_w f. ONE solve yields the whole T·N gradient — J is never formed.

This is the win over the MILP path's `1 primal + T·N forward columns + Gurobi MILP`.
The reverse mode was validated to machine precision against the forward Jacobian
in script/bilevel_validation/spike_reverse_mode.jl (2026-05-31).

Objective (cost-weighted served-Palma, ratio of sums — matches the single-level
trade-off `add_palma_machinery_cw_aggregate!`):
    f(w) = ( Σ_t λ_t · Σ_{top10%} sorted(served_t) )
           ────────────────────────────────────────────
           ( Σ_t λ_t · Σ_{bot40%} sorted(served_t) )
    served = pd − pshed*(w)

Requires `compute_palma_indices` / `palma_ratio` from load_shed_as_parameter.jl
(include that first).

Author: Claude (with guidance from Amanda); 2026-05-31.
=#

using JuMP
import MathOptInterface as MOI
using DiffOpt
using LinearAlgebra
using Printf

# ------------------------------------------------------------------
# Reverse-mode bridge override (twin of the forward one at
# lower_level_mld.jl:124). NonLinearProgram's reverse_differentiate!(::Model; tol)
# defaults to tol=1e-6, which trips the near-degenerate dual-sign assertion on the
# MLD active set; 1e-4 absorbs it. Verified necessary+sufficient by the spike.
# ------------------------------------------------------------------
function DiffOpt.reverse_differentiate!(
        model::MOI.Bridges.LazyBridgeOptimizer{DiffOpt.NonLinearProgram.Model})
    return DiffOpt.reverse_differentiate!(model.model; tol = 1e-4)
end

#=============================================================================
 Analytic Palma objective + gradient w.r.t. pshed (no solve)
=============================================================================#

"""
    palma_value(pshed, pd; n_loads, peak_time_costs=Float64[], eps_denom=1e-6)

Cost-weighted served-Palma objective used by the SLP / Frank-Wolfe upper level —
the **ratio of cost-weighted sums** (matched to the single-level trade-off):

    f = ( Σ_t λ_t · top10%(sorted served_t) ) / ( Σ_t λ_t · bot40%(sorted served_t) )
      = T(w) / B(w),    served = pd − pshed.

Each period is sorted independently (a true per-period top-10% / bot-40% decile),
λ-weighted, then summed into ONE numerator T and ONE denominator B; the objective
is their single ratio. This is `(Σλ·top)/(Σλ·bot)`, NOT the old `Σλ·(top/bot)`
(sum of per-period ratios) — it matches `add_palma_machinery_cw_aggregate!` in
`script/single_level/palma_trade_off_mn.jl` (σ·top_sum with σ·bot_sum=1).

The single GLOBAL denominator B is softened to `max(B, eps_denom)`; because B sums
over all periods it is far more robust to the bot40→0 corner than any per-period
bot_t (B vanishes only if EVERY period's bottom-40% is fully shed). For REPORTING
use `palma_ratio` (hard cutoff, Inf).
"""
function palma_value(pshed::Vector{Float64}, pd::Vector{Float64};
                     n_loads::Int, peak_time_costs::Vector{Float64} = Float64[],
                     eps_denom::Float64 = 1e-6, palma_on::Symbol = :served)
    @assert palma_on in (:served, :shed) "palma_on must be :served or :shed"
    m = length(pshed)
    n = n_loads
    @assert m % n == 0 "length(pshed)=$m not divisible by n_loads=$n"
    T = m ÷ n
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length T=$T"
    top_idx, bot_idx = compute_palma_indices(n)

    Tsum = 0.0   # Σ_t λ_t · top10%(q_t)
    Bsum = 0.0   # Σ_t λ_t · bot40%(q_t)
    for t in 1:T
        off = (t - 1) * n
        q = palma_on === :shed ? Float64[pshed[off + j] for j in 1:n] :
                                 Float64[pd[off + j] - pshed[off + j] for j in 1:n]
        s = sort(q)
        Tsum += λ[t] * sum(max(0.0, s[i]) for i in top_idx)
        Bsum += λ[t] * sum(max(0.0, s[i]) for i in bot_idx)
    end
    return Tsum / max(Bsum, eps_denom)
end

"""
    palma_grad_pshed(pshed, pd; n_loads, peak_time_costs=Float64[], eps_denom=1e-6)

Analytic v = ∂(palma_value)/∂pshed, length T·N, for the RATIO-OF-SUMS objective
f = T/B with T = Σ_t λ_t·top_t, B = Σ_t λ_t·bot_t. This is the seed handed to the
adjoint solve; `∇_w f = Jᵀ v`.

T and B are GLOBAL scalars, so the per-served derivatives couple all periods:
  ∂f/∂served_i = +λ_t / B          for i in period t's top10% positions,
  ∂f/∂served_i = −λ_t · T / B²     for i in period t's bot40% positions,
scattered back via the per-period sort permutation, then ∂/∂pshed = −∂/∂served.
B softened to max(B, eps_denom); when floored the bottom term is dropped (its
derivative is 0 under the constant floor), avoiding the bot→0 gradient blow-up.
A valid (sub)gradient under ties.
"""
function palma_grad_pshed(pshed::Vector{Float64}, pd::Vector{Float64};
                          n_loads::Int, peak_time_costs::Vector{Float64} = Float64[],
                          eps_denom::Float64 = 1e-6, palma_on::Symbol = :served)
    @assert palma_on in (:served, :shed) "palma_on must be :served or :shed"
    m = length(pshed)
    n = n_loads
    @assert m % n == 0 "length(pshed)=$m not divisible by n_loads=$n"
    T = m ÷ n
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length T=$T"
    top_idx, bot_idx = compute_palma_indices(n)
    # q is the sorted quantity: served = pd−pshed (default) or shed = pshed.
    # ∂q/∂pshed = −1 (served) or +1 (shed); the chain-rule sign below follows.
    qsgn = palma_on === :shed ? 1.0 : -1.0
    qvec(off) = palma_on === :shed ? Float64[pshed[off + j] for j in 1:n] :
                                     Float64[pd[off + j] - pshed[off + j] for j in 1:n]

    # Pass 1: the GLOBAL cost-weighted numerator/denominator.
    Tsum = 0.0; Bsum = 0.0
    for t in 1:T
        off = (t - 1) * n
        s = sort(qvec(off))
        Tsum += λ[t] * sum(max(0.0, s[i]) for i in top_idx)
        Bsum += λ[t] * sum(max(0.0, s[i]) for i in bot_idx)
    end
    Beff = max(Bsum, eps_denom)
    floored = Bsum ≤ eps_denom

    # Pass 2: scatter ∂f/∂q back to pshed, period by period.
    v = zeros(m)
    for t in 1:T
        off = (t - 1) * n
        perm = sortperm(qvec(off))             # perm[i] = original load at sorted position i

        gsorted = zeros(n)                      # ∂f/∂sorted[i]
        for i in top_idx
            gsorted[i] += λ[t] / Beff
        end
        if !floored
            for i in bot_idx
                gsorted[i] += -λ[t] * Tsum / (Beff^2)
            end
        end

        for i in 1:n
            j = perm[i]                         # back to original load index in period
            v[off + j] = qsgn * gsorted[i]      # ∂/∂pshed = (∂q/∂pshed)·∂f/∂q
        end
    end
    return v
end

#=============================================================================
 Lower-level model handles, primal solve, and adjoint VJP
=============================================================================#

"""
    build_fw_implicit_model(mn_data)
        -> (model, weight_params, pshed_vars, weight_ids, pshed_nw_ids, pd_all)

Instantiate the multinetwork implicit-diff MLD and collect handles in the SAME
period-major flattened order `diff_forward_full_jacobian_mn` uses, so the analytic
v (over pshed) and the recovered g (over weights) align with the rest of the code.
`pd_all` is the per-(period,load) demand matching pshed ordering; it is independent
of the weights, so callers compute it once.
"""
function build_fw_implicit_model(mn_data::Dict{String,Any})
    mld = instantiate_mc_model(
        mn_data, LinDist3FlowPowerModel, build_mn_mc_mld_shedding_implicit_diff;
        multinetwork = true, ref_extensions = [FairLoadDelivery.ref_add_load_blocks!])
    model = mld.model
    nw_ids = model[:nw_ids]

    weight_params = JuMP.VariableRef[]
    weight_ids = Int[]
    for (t, nw) in enumerate(nw_ids)
        wp = model[Symbol("weights_nw_$(nw)")]
        if t == 1
            weight_ids = collect(axes(wp, 1))
        end
        for key in eachindex(wp)
            push!(weight_params, wp[key])
        end
    end

    pshed_vars = JuMP.VariableRef[]
    pshed_nw_ids = Tuple[]
    for nw in nw_ids
        pv = model[Symbol("pshed_nw_$(nw)")]
        ids = collect(axes(pv, 1))
        for (key, lid) in zip(eachindex(pv), ids)
            push!(pshed_vars, pv[key])
            push!(pshed_nw_ids, (nw, lid))
        end
    end

    refs = Dict(nw => mld.ref[:it][:pmd][:nw][nw] for nw in nw_ids)
    pd_all = Float64[sum(refs[nw][:load][lid]["pd"]) for (nw, lid) in pshed_nw_ids]

    return model, weight_params, pshed_vars, weight_ids, pshed_nw_ids, pd_all
end

"""
    fw_solve_primal!(model, weight_params, w) -> (pshed_val, status)

Set the per-period weight parameters to `w` and solve the lower-level MLD (the
PRIMAL). Returns the flattened pshed values and the termination status.
"""
function fw_solve_primal!(model::JuMP.Model, weight_params::Vector{JuMP.VariableRef},
                          pshed_vars::Vector{JuMP.VariableRef}, w::Vector{Float64})
    @assert length(w) == length(weight_params)
    for (idx, p) in enumerate(weight_params)
        JuMP.set_parameter_value(p, w[idx])
    end
    optimize!(model)
    status = termination_status(model)
    pshed_val = Float64[JuMP.value(var) for var in pshed_vars]
    return pshed_val, status
end

"""
    fw_vjp!(model, pshed_vars, weight_params, v) -> g

The ADJOINT: one DiffOpt reverse solve. Seeds each pshed variable's downstream
sensitivity with `v`, runs `reverse_differentiate!`, and reads the resulting
weight-parameter sensitivities `g = Jᵀv = ∇_w f`. Assumes `model` is already at
the primal solution for the current `w` (call `fw_solve_primal!` first).
"""
function fw_vjp!(model::JuMP.Model, pshed_vars::Vector{JuMP.VariableRef},
                 weight_params::Vector{JuMP.VariableRef}, v::Vector{Float64})
    @assert length(v) == length(pshed_vars)
    DiffOpt.empty_input_sensitivities!(model)
    for (i, var) in enumerate(pshed_vars)
        DiffOpt.set_reverse_variable(model, var, v[i])
    end
    DiffOpt.reverse_differentiate!(model)
    return Float64[DiffOpt.get_reverse_parameter(model, p) for p in weight_params]
end

#=============================================================================
 Frank-Wolfe driver (Marguerite)
=============================================================================#

"""
    period_bot_sums(pshed, pd; n_loads) -> Vector{Float64}

Per-period bottom-40%-of-served sums — the Palma denominators. A period sum near 0
means its poorest 40% are fully shed (the corner-collapse degeneracy); the Palma
objective diverges there. Diagnostic for FW step health.
"""
function period_bot_sums(pshed::Vector{Float64}, pd::Vector{Float64}; n_loads::Int,
                         palma_on::Symbol = :served)
    m = length(pshed); n = n_loads; T = m ÷ n
    _, bot_idx = compute_palma_indices(n)
    out = zeros(T)
    for t in 1:T
        off = (t - 1) * n
        q = palma_on === :shed ? sort(Float64[pshed[off + j] for j in 1:n]) :
                                 sort(Float64[pd[off + j] - pshed[off + j] for j in 1:n])
        out[t] = sum(max(0.0, q[i]) for i in bot_idx)
    end
    return out
end

"""
    frank_wolfe_palma(mn_data; critical_ids, peak_time_costs, w_bounds=(1.0,10.0),
                      critical_cap=100.0, trust_radius=0.5, max_iters=50, tol=1e-4,
                      verbose=true)
        -> NamedTuple

Trust-region conditional-gradient (Frank-Wolfe) on the λ-weighted served-Palma
objective. Each iteration the LMO minimizes ⟨g,v⟩ over the box ∩ an L∞ ball of
radius `trust_radius` around the current weights:

    v[i] = g[i] ≥ 0 ? max(lo[i], x[i]−Δ) : min(hi[i], x[i]+Δ)

The trust region is the SAME safeguard the MILP upper level uses (`trust_radius=0.5`):
without it, vanilla FW leaps to box corners where the lower level drives a period's
bottom-40% served → 0 and the Palma objective blows up (~1e8) — see the inline
note and project_palma_skip_integer_warmstart. With it, weights stay where DiffOpt's
linearization is valid.

`f` and `∇f!` share ONE primal solve per distinct weight vector via an internal
cache. The step is a MONOTONE backtracking line search: start at γ_t = 2/(t+2) and
halve until f(x+γd) ≤ f(x), or until γ < `min_step` → the iterate is TR-stationary
and the loop stops. The lowest-obj iterate seen is tracked and RETURNED (the
DiffOpt-linearized objective is nonconvex, so the raw FW path can wander uphill).
Each iteration costs 1 adjoint + (1 + #backtracks) primals.

Returns `(; weights, pshed, palma_value, palma_ratio_reported, converged, fw_iters,
           fw_gap, history, n_primal, n_adjoint, weight_ids, pshed_nw_ids)`.
"""
function frank_wolfe_palma(mn_data::Dict{String,Any};
                           critical_ids::Vector{Int} = Int[],
                           peak_time_costs::Vector{Float64} = Float64[],
                           w_bounds::Tuple{Float64,Float64} = (1.0, 10.0),
                           critical_cap::Float64 = 100.0,
                           trust_radius::Float64 = 0.5,
                           max_iters::Int = 50,
                           tol::Real = 1e-4,
                           min_step::Float64 = 1e-3,
                           verbose::Bool = true)
    w_min, w_max = w_bounds

    # Discover ordering, box, pd, and initial weights from one build.
    model0, wparams0, pshed0, weight_ids, pshed_nw_ids, pd_all =
        build_fw_implicit_model(mn_data)
    m = length(wparams0)
    n = length(weight_ids)
    @assert m % n == 0
    T = m ÷ n
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs

    # Weight box (per-coordinate); critical loads get the higher ceiling.
    lo = fill(w_min, m)
    hi = fill(w_max, m)
    for idx in 1:m
        lid = weight_ids[((idx - 1) % n) + 1]
        if lid in critical_ids
            hi[idx] = critical_cap
        end
    end

    # Initial weights = the model's default parameter values, clamped into the box.
    w0 = Float64[JuMP.parameter_value(p) for p in wparams0]
    w0 .= clamp.(w0, lo, hi)

    # --- shared-primal cache --------------------------------------------------
    # f(w) and ∇f!(g,w) are called at the same w within an FW iteration; build a
    # fresh solved model on each NEW w and reuse it for the adjoint.
    cur_w      = fill(NaN, m)
    cur_model  = model0
    cur_wp     = wparams0
    cur_pshed_vars = pshed0
    cur_pshed  = Float64[]
    n_primal   = Ref(0)
    n_adjoint  = Ref(0)

    function ensure!(w)
        if cur_w != w
            model, wp, pv, _, _, _ = build_fw_implicit_model(mn_data)
            ps, status = fw_solve_primal!(model, wp, pv, collect(Float64, w))
            if status ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED,
                         MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
                @warn "[FW] primal terminated $status — gradient may be unreliable"
            end
            n_primal[] += 1
            cur_model = model
            cur_wp = wp
            cur_pshed_vars = pv
            cur_pshed = ps
            cur_w .= w
        end
        return nothing
    end

    f = function (w)
        ensure!(w)
        return palma_value(cur_pshed, pd_all; n_loads = n, peak_time_costs = λ)
    end

    ∇f! = function (g, w)
        ensure!(w)
        v = palma_grad_pshed(cur_pshed, pd_all; n_loads = n, peak_time_costs = λ)
        gw = fw_vjp!(cur_model, cur_pshed_vars, cur_wp, v)
        n_adjoint[] += 1
        g .= gw
        return g
    end

    # --- trust-region conditional gradient (Frank-Wolfe) --------------------
    # Per-iter LMO minimizes ⟨g,v⟩ over box ∩ L∞-ball(x, Δ), separable:
    #   v[i] = g[i] ≥ 0 ? max(lo[i], x[i]−Δ) : min(hi[i], x[i]+Δ).
    # The trust region Δ mirrors the MILP upper level's trust_radius and keeps the
    # iterate off the singular box corners (where a period's bot-40% served → 0 and
    # the Palma objective blows up). Step γ_t = 2/(t+2); gap = ⟨g, x−v⟩ (local).
    Δ = trust_radius
    tr_lmo!(vbuf, g, x) = (@inbounds for i in eachindex(g)
                               vbuf[i] = g[i] ≥ 0 ? max(lo[i], x[i] - Δ) :
                                                    min(hi[i], x[i] + Δ)
                           end; vbuf)

    x       = copy(w0)
    g       = zeros(m)
    vbuf    = zeros(m)
    xtrial  = zeros(m)
    obj     = f(x)
    gap     = Inf
    converged = false
    fw_iter = 0
    history = NamedTuple{(:iter, :obj, :gap, :min_bot, :gnorm),
                         Tuple{Int,Float64,Float64,Float64,Float64}}[]

    # --- best-iterate tracking (fix A) ---------------------------------------
    # The DiffOpt-linearized objective is NONconvex (the lower-level active set
    # shifts with w), so the iterate can wander uphill after finding a good point.
    # Keep the lowest-obj weights/pshed seen and return THOSE, not the last iterate.
    best_x     = copy(x)
    best_obj   = obj
    best_pshed = copy(cur_pshed)

    for t in 0:(max_iters - 1)
        ∇f!(g, x)
        if !all(isfinite, g)
            @warn "[FW] non-finite gradient at iter $t — stopping"
            fw_iter = t
            break
        end
        tr_lmo!(vbuf, g, x)
        gap = dot(g, x .- vbuf)               # local FW gap ≥ 0 at a TR-stationary point
        obj = f(x)
        min_bot = minimum(period_bot_sums(cur_pshed, pd_all; n_loads = n))
        gnorm = maximum(abs.(g))
        push!(history, (iter = t, obj = obj, gap = gap, min_bot = min_bot, gnorm = gnorm))
        if obj < best_obj                     # fix A: remember the best point
            best_obj = obj
            best_x .= x
            best_pshed .= cur_pshed
        end
        verbose && (t % 5 == 0 || t == max_iters - 1) &&
            @info @sprintf("[FW] iter %3d  obj=%.6e  gap=%.4e  min_bot=%.3e  ‖g‖=%.3e",
                           t, obj, gap, min_bot, gnorm)
        if isfinite(gap) && gap ≤ tol * (1 + abs(obj))
            converged = true
            fw_iter = t
            break
        end
        # --- monotone backtracking step (fix B) ------------------------------
        # FW direction d = vbuf − x is a descent direction for the linear model;
        # the open-loop γ_t = 2/(t+2) does NOT guarantee f decreases on the true
        # nonconvex objective (it drove the old run back uphill). Backtrack γ by
        # halving until f(x + γ d) ≤ f(x), or until γ < min_step → declare the
        # iterate stationary within the trust region and stop.
        γ = 2.0 / (t + 2.0)
        accepted = false
        while γ ≥ min_step
            @. xtrial = x + γ * (vbuf - x)    # convex step stays in box ∩ TR
            objtrial = f(xtrial)              # one primal solve (cached for next ∇f!)
            if isfinite(objtrial) && objtrial ≤ obj + tol * (1 + abs(obj))
                x .= xtrial
                accepted = true
                break
            end
            γ *= 0.5
        end
        if !accepted
            verbose && @info @sprintf("[FW] iter %3d  no descent along FW direction (γ<%.1e) — stationary",
                                      t, min_step)
            converged = true                  # TR-stationary: no improving step exists
            fw_iter = t
            break
        end
        fw_iter = t + 1
    end

    # Return the BEST iterate found (fix A), not the last one.
    pserved = pd_all .- best_pshed
    verbose && @info @sprintf("[FW] done: %d iters, %s, best_obj=%.6e, gap=%.4e, primal=%d adjoint=%d",
                              fw_iter, converged ? "converged" : "max_iters",
                              best_obj, gap, n_primal[], n_adjoint[])
    return (; weights = collect(Float64, best_x),
              pshed = copy(best_pshed),
              palma_value = best_obj,
              palma_ratio_reported = palma_ratio(pserved),
              converged = converged,
              fw_iters = fw_iter,
              fw_gap = gap,
              history = history,
              n_primal = n_primal[],
              n_adjoint = n_adjoint[],
              weight_ids = weight_ids,
              pshed_nw_ids = pshed_nw_ids)
end
# NOTE: slp_cc_palma (Sequential Charnes-Cooper LP, variant b) moved to
# src/implementation/slp_cc_palma.jl (include this file first; it reuses the helpers above).
