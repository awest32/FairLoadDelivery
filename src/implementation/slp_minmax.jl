#=
Sequential epigraph-LP min-max upper level  (reverse-mode SLP)
==============================================================
Min-max sibling of `slp_cc_palma` (slp_cc_palma.jl). Min-max has NO fraction, so
there is NO Charnes–Cooper here — the upper level minimizes the SINGLE GLOBAL
maximum shed over all (load, period):

    min_w  z   s.t.  z ≥ pshed_{i,t}(w)   ∀ (i,t)

This is the single global epigraph form mandated by project_feedback_bilevel_min_max_epigraph:
the per-period max_shed[t] + λ-weighted-sum "textbook" form fails to descend
(verified worse on case6 T=24) — do NOT switch to it.

Per outer iteration (SLP / active-set, mirroring slp_cc_palma):
  1. PRIMAL solve the MLD at w → pshed.
  2. ACTIVE SET A = the top-K shedding (load,period) pairs (the near-max loads);
     held fixed for the step (the argmax is the min-max analog of Palma's sort).
  3. K reverse adjoints: gradient ROW ∂pshed_a/∂w for each a ∈ A (seed e_a).
  4. EPIGRAPH LP over a trust region → step.
  5. Backtracking line search on the TRUE global max-shed (monotone descent),
     best-iterate tracking, re-identify A next iteration.

Reverse adjoints per iter = K (vs the MILP path's T·N forward Jacobian columns).
Correctness of the fixed active set is enforced by the line search: an accepted
step must reduce the TRUE max-shed (computed from a real resolve), so an
incomplete active set only costs step quality, never validity.

INCLUDE ORDER: reuses `build_fw_implicit_model`, `fw_solve_primal!`, `fw_vjp!`
from `frank_wolfe_palma.jl` — include that first.

Author: Claude (with guidance from Amanda); 2026-05-31.
=#

using JuMP
import MathOptInterface as MOI
using Printf

"""
    minmax_active_set(pshed, K; tol=1e-9) -> Vector{Int}

Flattened indices of the K largest SHEDDING entries of `pshed` (the near-max
active set). Loads with shed ≤ tol are excluded (a non-shedding load can't be the
max and its gradient row is ~0, so it wastes an adjoint). Returns ≤ K indices.
"""
function minmax_active_set(pshed::Vector{Float64}, K::Int; tol::Float64 = 1e-9)
    shedding = findall(>(tol), pshed)
    isempty(shedding) && return Int[argmax(pshed)]          # degenerate: nothing shed
    k = min(K, length(shedding))
    order = sortperm(pshed[shedding]; rev = true)
    return shedding[order[1:k]]
end

"""
    minmax_reverse_grads(model, pshed_vars, weight_params, active_idx)
        -> Vector{Vector{Float64}}

One reverse adjoint per active load: `g[a] = ∂pshed_{active_idx[a]}/∂w` (length m),
i.e. the active ROW of the lower-level Jacobian (seed = e_{active_idx[a]}, no sign
flip — we want pshed itself, not served). K solves total. `model` must be at the
primal solution for the current weights.
"""
function minmax_reverse_grads(model::JuMP.Model, pshed_vars::Vector{JuMP.VariableRef},
                              weight_params::Vector{JuMP.VariableRef}, active_idx::Vector{Int})
    grads = Vector{Vector{Float64}}(undef, length(active_idx))
    for (a, idx) in enumerate(active_idx)
        v = zeros(length(pshed_vars))
        v[idx] = 1.0
        grads[a] = fw_vjp!(model, pshed_vars, weight_params, v)   # = ∂pshed_idx/∂w
    end
    return grads
end

"""
    minmax_lp_step(w_k, pshed, active_idx, grads; w_lo, w_hi, Δ, lp_optimizer) -> w_new

Epigraph LP for one SLP iteration (no CC — weights enter unrescaled):

    min  z
    s.t. z ≥ pshed[a] + grads[a]·Δw     for a in active_idx   (epigraph)
         -Δ ≤ Δw ≤ Δ                                          (trust region)
         w_lo - w_k ≤ Δw ≤ w_hi - w_k                         (weight box)

Returns w_k + Δw clamped to the box. On non-optimal LP status returns w_k.
"""
function minmax_lp_step(w_k::Vector{Float64}, pshed::Vector{Float64},
                        active_idx::Vector{Int}, grads::Vector{Vector{Float64}};
                        w_lo::Vector{Float64}, w_hi::Vector{Float64},
                        Δ::Float64, lp_optimizer)
    m = length(w_k)
    model = JuMP.Model(lp_optimizer)
    set_silent(model)
    @variable(model, z)
    @variable(model, dw[1:m])
    for (a, idx) in enumerate(active_idx)
        @constraint(model, z >= pshed[idx] + sum(grads[a][j] * dw[j] for j in 1:m))
    end
    for j in 1:m
        @constraint(model, dw[j] <=  Δ)
        @constraint(model, dw[j] >= -Δ)
        @constraint(model, dw[j] <= w_hi[j] - w_k[j])
        @constraint(model, dw[j] >= w_lo[j] - w_k[j])
    end
    @objective(model, Min, z)
    optimize!(model)
    st = termination_status(model)
    if st ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
        @warn "[SLP-minmax] LP status $st — taking zero step this iter"
        return copy(w_k)
    end
    w_new = w_k .+ JuMP.value.(dw)
    @inbounds for j in 1:m
        w_new[j] = clamp(w_new[j], w_lo[j], w_hi[j])
    end
    return w_new
end

"""
    slp_minmax(mn_data; lp_optimizer, critical_ids, w_bounds, critical_cap,
               trust_radius, active_set_size, max_iters, tol, min_step, verbose)
        -> NamedTuple

Sequential epigraph-LP min-max bilevel upper level. Minimizes the single global
max shed over all (load,period). Per iteration: 1 primal + K adjoints + 1 LP +
backtracking. Returns `(; weights, pshed, max_shed, total_shed, converged,
slp_iters, n_primal, n_adjoint, history, weight_ids, pshed_nw_ids)`.

`active_set_size` (K) trades reverse-adjoint count against the min-max model's
fidelity at the flat top; larger K models a wider plateau but costs K adjoints/iter.
"""
function slp_minmax(mn_data::Dict{String,Any};
                    lp_optimizer,
                    critical_ids::Vector{Int} = Int[],
                    w_bounds::Tuple{Float64,Float64} = (1.0, 10.0),
                    critical_cap::Float64 = 100.0,
                    trust_radius::Float64 = 0.5,
                    active_set_size::Int = 12,
                    max_iters::Int = 30,
                    tol::Real = 1e-4,
                    min_step::Float64 = 1e-3,
                    verbose::Bool = true)
    w_min, w_max = w_bounds

    _, wparams0, _, weight_ids, pshed_nw_ids, _ = build_fw_implicit_model(mn_data)
    m = length(wparams0)
    n = length(weight_ids)
    @assert m % n == 0
    T = m ÷ n

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

    function solve_at(wv::Vector{Float64})
        mdl, wp, pv, _, _, _ = build_fw_implicit_model(mn_data)
        ps, status = fw_solve_primal!(mdl, wp, pv, wv)
        n_primal[] += 1
        return mdl, wp, pv, ps, status
    end
    merit(ps) = maximum(ps)                         # global max shed (lower = fairer)

    cur_model, cur_wp, cur_pv, pshed, st0 = solve_at(w)
    if st0 ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
        @warn "[SLP-minmax] initial primal status $st0"
    end
    obj = merit(pshed)

    best_w     = copy(w)
    best_obj   = obj
    best_pshed = copy(pshed)
    converged  = false
    slp_iter   = 0
    history = NamedTuple{(:iter, :obj, :step, :total_shed),
                         Tuple{Int,Float64,Float64,Float64}}[]

    for it in 0:(max_iters - 1)
        active = minmax_active_set(pshed, active_set_size)
        grads  = minmax_reverse_grads(cur_model, cur_pv, cur_wp, active)
        n_adjoint[] += length(active)
        w_lp = minmax_lp_step(w, pshed, active, grads;
                              w_lo = w_lo, w_hi = w_hi, Δ = trust_radius,
                              lp_optimizer = lp_optimizer)
        d = w_lp .- w
        γ = 1.0
        accepted = false
        obj_prev = obj
        while γ ≥ min_step
            w_try = w .+ γ .* d
            tm, twp, tpv, tps, tst = solve_at(w_try)
            # Reject non-converged trials (see slp_cc_palma.jl): an infeasible/
            # iteration-limited Ipopt solve returns finite pshed but is unsafe to
            # adjoint-differentiate next iteration. Only converged steps accepted.
            feas = tst in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED)
            to = feas ? merit(tps) : Inf
            if feas && isfinite(to) && to < obj - 1e-10
                w = w_try; cur_model = tm; cur_wp = twp; cur_pv = tpv
                pshed = tps; obj = to; accepted = true
                break
            end
            γ *= 0.5
        end
        tot = sum(pshed)
        push!(history, (iter = it, obj = obj, step = accepted ? γ : 0.0, total_shed = tot))
        if obj < best_obj
            best_obj = obj; best_w .= w; best_pshed .= pshed
        end
        verbose && (it % 5 == 0 || it == max_iters - 1 || !accepted) &&
            @info @sprintf("[SLP-minmax] iter %3d  max_shed=%.6e  step=%.3e  Σshed=%.3e  |A|=%d",
                           it, obj, accepted ? γ : 0.0, tot, length(active))
        slp_iter = it + 1
        if !accepted
            verbose && @info @sprintf("[SLP-minmax] iter %3d  no descent (γ<%.1e) — stationary", it, min_step)
            converged = true
            break
        end
        if abs(obj_prev - obj) ≤ tol * (1 + abs(obj))
            converged = true
            break
        end
    end

    verbose && @info @sprintf("[SLP-minmax] done: %d iters, %s, best max_shed=%.6e, primal=%d adjoint=%d",
                              slp_iter, converged ? "converged" : "max_iters",
                              best_obj, n_primal[], n_adjoint[])
    return (; weights = copy(best_w),
              pshed = copy(best_pshed),
              max_shed = best_obj,
              total_shed = sum(best_pshed),
              converged = converged,
              slp_iters = slp_iter,
              n_primal = n_primal[],
              n_adjoint = n_adjoint[],
              history = history,
              weight_ids = weight_ids,
              pshed_nw_ids = pshed_nw_ids)
end
