#=
Sequential Charnes–Cooper LP upper level for served-Palma  (variant b)
======================================================================
An alternative to the Frank-Wolfe driver in `frank_wolfe_palma.jl`. FW is a
smooth conditional-gradient method, but Palma is hostile to it on two counts:
the objective is FRACTIONAL (top/bot — FW must soften it with eps_denom and the
gradient blows up as bot→0) and it requires a SORT (nonsmooth; the FW subgradient
silently flips when the ranking reshuffles). This driver embraces that structure
instead of fighting it:

  • Charnes–Cooper handles the fraction EXACTLY (linear-fractional → LP; no
    surrogate, σ unbounded absorbs bot→0).
  • The sort is made EXPLICIT: hold it fixed per iteration (membership sets +
    boundary inequalities), re-sort between iterations (active-set / SLP).

So this is Sequential Linear(-fractional) Programming, NOT Frank-Wolfe: each
iteration rebuilds a local CC-LP over a trust region and solves it to optimality,
rather than gradient-linearizing a fixed feasible set and taking damped convex
steps. The gradients are cheap reverse adjoints (4 per iteration, T-independent).

INCLUDE ORDER: this file reuses helpers from `frank_wolfe_palma.jl`
(`build_fw_implicit_model`, `fw_solve_primal!`, `fw_vjp!`, `palma_value`,
`period_bot_sums`) and from `load_shed_as_parameter.jl` (`compute_palma_indices`,
`palma_ratio`). Include BOTH of those first.

Author: Claude (with guidance from Amanda); 2026-05-31.
=#

using JuMP
import MathOptInterface as MOI
using Printf

"""
    palma_sets_and_cuts(served, n) -> (T_set, B_set, (A, D), (C, E))

For one period's `served` vector (length `n`), return the ORIGINAL-load index
sets for the top-10% (`T_set`) and bottom-40% (`B_set`) of the ascending sort,
plus the two boundary pairs used by the variant-(b) ordering cuts:

  bottom cut (A, D): D = largest load INSIDE  the bottom set (sorted pos n_bot),
                     A = smallest load OUTSIDE the bottom set (sorted pos n_bot+1).
                     `served_D ≤ served_A` holds the bottom set fixed.
  top cut    (C, E): C = smallest load INSIDE the top set (sorted pos n−n_top+1),
                     E = largest load OUTSIDE the top set (sorted pos n−n_top).
                     `served_E ≤ served_C` holds the top set fixed.

Indices are 1..n within the period; the caller adds the period offset.
"""
function palma_sets_and_cuts(served::Vector{Float64}, n::Int)
    perm = sortperm(served)                     # ascending; perm[pos] = orig load
    top_idx, bot_idx = compute_palma_indices(n)
    n_bot = length(bot_idx)
    n_top = length(top_idx)
    T_set = perm[top_idx]
    B_set = perm[bot_idx]
    D = perm[n_bot]                             # largest in bottom set
    A = perm[n_bot + 1]                         # smallest outside bottom set
    C = perm[n - n_top + 1]                     # smallest in top set
    E = perm[n - n_top]                         # largest outside top set
    return T_set, B_set, (A, D), (C, E)
end

"""
    cc_reverse_grads(model, pshed_vars, weight_params, n, T, sets)
        -> (gtop, gbot, gbotcut, gtopcut)

Four adjoint solves (independent of T) giving ∇_w of, per period:
  gtop    — Σ_{i∈T_set} served_i        (top-10% sum)
  gbot    — Σ_{i∈B_set} served_i        (bottom-40% sum)
  gbotcut — served_A − served_D         (bottom boundary gap)
  gtopcut — served_C − served_E         (top boundary gap)
Each is length m = T·n over the weight parameters; by block-diagonality the
gradient for period t is nonzero only on period t's weight block. Signs use
∂served/∂w = −∂pshed/∂w = −Jᵀ·seed. `model` must already be at the primal
solution for the current weights. `sets[t]` is the tuple from `palma_sets_and_cuts`.
"""
function cc_reverse_grads(model::JuMP.Model, pshed_vars::Vector{JuMP.VariableRef},
                          weight_params::Vector{JuMP.VariableRef},
                          n::Int, T::Int, sets)
    m = n * T
    v_top = zeros(m); v_bot = zeros(m); v_botcut = zeros(m); v_topcut = zeros(m)
    for t in 1:T
        off = (t - 1) * n
        T_set, B_set, (A, D), (C, E) = sets[t]
        for i in T_set; v_top[off + i] = 1.0; end
        for i in B_set; v_bot[off + i] = 1.0; end
        v_botcut[off + A] += 1.0; v_botcut[off + D] -= 1.0    # served_A − served_D
        v_topcut[off + C] += 1.0; v_topcut[off + E] -= 1.0    # served_C − served_E
    end
    gtop    = -fw_vjp!(model, pshed_vars, weight_params, v_top)
    gbot    = -fw_vjp!(model, pshed_vars, weight_params, v_bot)
    gbotcut = -fw_vjp!(model, pshed_vars, weight_params, v_botcut)
    gtopcut = -fw_vjp!(model, pshed_vars, weight_params, v_topcut)
    return gtop, gbot, gbotcut, gtopcut
end

"""
    cc_lp_step(w_k, served, sets, grads, λ, n, T; w_lo, w_hi, Δ, σ_min, lp_optimizer)
        -> w_new

One fixed-sort Charnes–Cooper LP. Per period t with σ_t = 1/bot_t(w) and the
rescaled step `dz = σ_t·Δw`:

  minimize   Σ_t λ_t (a_t·σ_t + gtop_t·dz_t)            (= Σ_t λ_t·σ_t·top_t(w))
  subject to b_t·σ_t + gbot_t·dz_t = 1                  (σ_t·bot_t(w) = 1)
             (sA−sD)·σ_t + gbotcut_t·dz_t ≥ 0           (variant-b bottom cut)
             (sC−sE)·σ_t + gtopcut_t·dz_t ≥ 0           (variant-b top cut)
             |dz_t| ≤ Δ·σ_t                              (trust region)
             (w_lo−w_k)·σ_t ≤ dz_t ≤ (w_hi−w_k)·σ_t      (weight box)
             σ_t ≥ σ_min

where a_t = top_t(w_k), b_t = bot_t(w_k) are constants from `served`. Recovers
Δw_t = dz_t/σ_t and returns w_k + Δw clamped to the box. On non-optimal LP
status returns w_k (zero step → the outer loop reads it as stationary).
"""
function cc_lp_step(w_k::Vector{Float64}, served::Vector{Float64}, sets,
                    grads, λ::Vector{Float64}, n::Int, T::Int;
                    w_lo::Vector{Float64}, w_hi::Vector{Float64},
                    Δ::Float64, σ_min::Float64, lp_optimizer)
    gtop, gbot, gbotcut, gtopcut = grads
    m = n * T
    model = JuMP.Model(lp_optimizer)
    set_silent(model)
    @variable(model, σ[1:T] >= σ_min)
    @variable(model, dz[1:m])
    obj = JuMP.AffExpr(0.0)
    for t in 1:T
        off = (t - 1) * n
        T_set, B_set, (A, D), (C, E) = sets[t]
        servt = view(served, (off + 1):(off + n))
        a_t = sum(servt[i] for i in T_set)        # top_t(w_k)
        b_t = sum(servt[i] for i in B_set)        # bot_t(w_k)
        # σ_t · bot_t(w) = 1
        @constraint(model, b_t * σ[t] + sum(gbot[off + j] * dz[off + j] for j in 1:n) == 1)
        # variant-(b) boundary cuts (keep the sort self-consistent over the step)
        @constraint(model, (servt[A] - servt[D]) * σ[t] +
                           sum(gbotcut[off + j] * dz[off + j] for j in 1:n) >= 0)
        @constraint(model, (servt[C] - servt[E]) * σ[t] +
                           sum(gtopcut[off + j] * dz[off + j] for j in 1:n) >= 0)
        for j in 1:n
            gj = off + j
            @constraint(model, dz[gj] <=  Δ * σ[t])             # trust region
            @constraint(model, dz[gj] >= -Δ * σ[t])
            @constraint(model, dz[gj] <= (w_hi[gj] - w_k[gj]) * σ[t])  # box
            @constraint(model, dz[gj] >= (w_lo[gj] - w_k[gj]) * σ[t])
        end
        JuMP.add_to_expression!(obj, λ[t] * a_t, σ[t])          # λ_t·a_t·σ_t
        for j in 1:n
            JuMP.add_to_expression!(obj, λ[t] * gtop[off + j], dz[off + j])
        end
    end
    @objective(model, Min, obj)
    optimize!(model)
    st = termination_status(model)
    if st ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        @warn "[SLP-CC] LP status $st — taking zero step this iter"
        return copy(w_k)
    end
    σv = JuMP.value.(σ); dzv = JuMP.value.(dz)
    w_new = copy(w_k)
    for t in 1:T
        off = (t - 1) * n
        for j in 1:n
            gj = off + j
            w_new[gj] = w_k[gj] + dzv[gj] / σv[t]
        end
    end
    @inbounds for gj in 1:m
        w_new[gj] = clamp(w_new[gj], w_lo[gj], w_hi[gj])
    end
    return w_new
end

"""
    period_palma_ratios(pshed, pd; n_loads) -> Vector{Float64}

Per-period EXACT served-Palma ratios top_t/bot_t (eps_denom-free; Inf if a
period's bottom-40% served is 0). For reporting against the equality floor
`ceil(0.1N)/floor(0.4N)`.
"""
function period_palma_ratios(pshed::Vector{Float64}, pd::Vector{Float64}; n_loads::Int)
    m = length(pshed); n = n_loads; T = m ÷ n
    top_idx, bot_idx = compute_palma_indices(n)
    out = zeros(T)
    for t in 1:T
        off = (t - 1) * n
        s = sort(Float64[pd[off + j] - pshed[off + j] for j in 1:n])
        top = sum(max(0.0, s[i]) for i in top_idx)
        bot = sum(max(0.0, s[i]) for i in bot_idx)
        out[t] = bot > 0 ? top / bot : Inf
    end
    return out
end

"""
    palma_equality_floor(n) -> Float64

The lowest achievable raw-sum Palma ratio (all loads served equally):
`ceil(0.1n) / floor(0.4n)`. 1/3 for n=9 or 16; 0.25 for n=10 or 20.
"""
function palma_equality_floor(n::Int)
    top_idx, bot_idx = compute_palma_indices(n)
    return length(top_idx) / length(bot_idx)
end

"""
    slp_cc_palma(mn_data; lp_optimizer, critical_ids, peak_time_costs, w_bounds,
                 critical_cap, trust_radius, max_iters, tol, σ_min, min_step, verbose)
        -> NamedTuple

Sequential Charnes–Cooper LP upper level (variant b). Per outer iteration:
  1. PRIMAL solve the MLD at the current weights → pshed → served.
  2. Fix the per-period sort; build membership sets + boundary pairs.
  3. FOUR adjoint solves → structured gradients (top/bot sums + the two cuts).
  4. Solve the fixed-sort CC-LP for a trust-region step.
  5. Backtracking line search on the TRUE Palma objective (monotone descent),
     best-iterate tracking. Re-sort next iteration (active-set update).

Minimizes Σ_t λ_t·top_t/bot_t. Returns `(; weights, pshed, palma_value,
palma_ratio_reported, period_ratios, equality_floor, converged, slp_iters,
n_primal, n_adjoint, history, weight_ids, pshed_nw_ids)`.
"""
function slp_cc_palma(mn_data::Dict{String,Any};
                      lp_optimizer,
                      critical_ids::Vector{Int} = Int[],
                      peak_time_costs::Vector{Float64} = Float64[],
                      w_bounds::Tuple{Float64,Float64} = (1.0, 10.0),
                      critical_cap::Float64 = 100.0,
                      trust_radius::Float64 = 0.5,
                      max_iters::Int = 30,
                      tol::Real = 1e-4,
                      σ_min::Float64 = 1e-8,
                      min_step::Float64 = 1e-3,
                      verbose::Bool = true)
    w_min, w_max = w_bounds

    # Discover ordering / box / pd from one build.
    _, wparams0, _, weight_ids, pshed_nw_ids, pd_all = build_fw_implicit_model(mn_data)
    m = length(wparams0)
    n = length(weight_ids)
    @assert m % n == 0
    T = m ÷ n
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T

    w_lo = fill(w_min, m); w_hi = fill(w_max, m)
    for idx in 1:m
        lid = weight_ids[((idx - 1) % n) + 1]
        if lid in critical_ids
            w_hi[idx] = critical_cap
        end
    end
    w = clamp.(Float64[JuMP.parameter_value(p) for p in wparams0], w_lo, w_hi)

    n_primal  = Ref(0)
    n_adjoint = Ref(0)

    # Build + solve a fresh model at weights `wv`; returns handles for the adjoint.
    function solve_at(wv::Vector{Float64})
        mdl, wp, pv, _, _, _ = build_fw_implicit_model(mn_data)
        ps, status = fw_solve_primal!(mdl, wp, pv, wv)
        n_primal[] += 1
        return mdl, wp, pv, ps, status
    end
    merit(ps) = palma_value(ps, pd_all; n_loads = n, peak_time_costs = λ)

    cur_model, cur_wp, cur_pv, pshed, st0 = solve_at(w)
    if st0 ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        @warn "[SLP-CC] initial primal status $st0"
    end
    obj = merit(pshed)

    best_w     = copy(w)
    best_obj   = obj
    best_pshed = copy(pshed)
    converged  = false
    slp_iter   = 0
    history = NamedTuple{(:iter, :obj, :step, :min_bot),
                         Tuple{Int,Float64,Float64,Float64}}[]
    floor_val = palma_equality_floor(n)

    for it in 0:(max_iters - 1)
        # (2) fix the sort: per-period membership + boundary pairs
        sets = Vector{Any}(undef, T)
        for t in 1:T
            off = (t - 1) * n
            servt = Float64[pd_all[off + j] - pshed[off + j] for j in 1:n]
            sets[t] = palma_sets_and_cuts(servt, n)
        end
        # (3) structured reverse gradients at the current KKT point
        grads = cc_reverse_grads(cur_model, cur_pv, cur_wp, n, T, sets)
        n_adjoint[] += 4
        served_full = pd_all .- pshed
        # (4) CC-LP trust-region step
        w_lp = cc_lp_step(w, served_full, sets, grads, λ, n, T;
                          w_lo = w_lo, w_hi = w_hi, Δ = trust_radius,
                          σ_min = σ_min, lp_optimizer = lp_optimizer)
        # (5) backtracking line search on the TRUE objective
        d = w_lp .- w
        γ = 1.0
        accepted = false
        obj_prev = obj
        while γ ≥ min_step
            w_try = w .+ γ .* d
            tm, twp, tpv, tps, tst = solve_at(w_try)
            # Reject trials whose lower-level solve did NOT converge: a
            # LOCALLY_INFEASIBLE/iteration-limited Ipopt solve still returns finite
            # pshed (its last iterate), so a merit-only test would accept it — and
            # the NEXT iteration's reverse_differentiate! then throws on the
            # non-optimal model. Only converged, descending steps are accepted.
            feas = tst in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
            to = feas ? merit(tps) : Inf
            if feas && isfinite(to) && to < obj - 1e-10
                w = w_try; cur_model = tm; cur_wp = twp; cur_pv = tpv
                pshed = tps; obj = to; accepted = true
                break
            end
            γ *= 0.5
        end
        min_bot = minimum(period_bot_sums(pshed, pd_all; n_loads = n))
        push!(history, (iter = it, obj = obj, step = accepted ? γ : 0.0, min_bot = min_bot))
        if obj < best_obj
            best_obj = obj; best_w .= w; best_pshed .= pshed
        end
        verbose && (it % 5 == 0 || it == max_iters - 1 || !accepted) &&
            @info @sprintf("[SLP-CC] iter %3d  obj=%.6e  step=%.3e  min_bot=%.3e",
                           it, obj, accepted ? γ : 0.0, min_bot)
        slp_iter = it + 1
        if !accepted
            verbose && @info @sprintf("[SLP-CC] iter %3d  no descent (γ<%.1e) — stationary", it, min_step)
            converged = true
            break
        end
        if abs(obj_prev - obj) ≤ tol * (1 + abs(obj))
            converged = true
            break
        end
    end

    verbose && @info @sprintf("[SLP-CC] done: %d iters, %s, best_obj=%.6e, primal=%d adjoint=%d",
                              slp_iter, converged ? "converged" : "max_iters",
                              best_obj, n_primal[], n_adjoint[])
    return (; weights = copy(best_w),
              pshed = copy(best_pshed),
              palma_value = best_obj,
              palma_ratio_reported = palma_ratio(pd_all .- best_pshed),
              period_ratios = period_palma_ratios(best_pshed, pd_all; n_loads = n),
              equality_floor = floor_val,
              converged = converged,
              slp_iters = slp_iter,
              n_primal = n_primal[],
              n_adjoint = n_adjoint[],
              history = history,
              weight_ids = weight_ids,
              pshed_nw_ids = pshed_nw_ids)
end
