# Create the other fairness functions

# Function to compute Jain's Fairness Index
# With peak_time_costs (peak charges): max Σ_t λ[t] * Jain_t
# With reg>0 and pd supplied: subtract λ[t]·reg·Σ_i pshed_new_t[i] / Σ_i pd_t[i]
# With alpha ∈ [0,1] and pd supplied: convex combination of efficiency and Jain.
#   - alpha=0: pure efficiency (min weighted shed)
#   - alpha=1: pure Jain fairness
function jains_fairness_index(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}; critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[], peak_time_costs::Vector{Float64}=Float64[], n_loads::Int=0, reg::Float64=1e-4, pd::Vector{Float64}=Float64[], alpha::Float64=1.0, weight_budget::Float64=Inf)
    model = JuMP.Model(Ipopt.Optimizer)
    m = length(pshed_prev)
    n_per_period = n_loads > 0 ? n_loads : m
    @variable(model, weights_new[1:m] .>= 1.0)
    for id in 1:m
        lid_idx = ((id - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        if load_id in critical_ids
            @constraint(model, weights_new[id] <= 100.0)
        else
            @constraint(model, weights_new[id] <= 10.0)
        end
    end
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] <= TRUST_RADIUS)
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] >= -TRUST_RADIUS)
    @expression(model, pshed_new[i = 1:m],
       pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:m)
    )

    # Guard: if all pshed values are zero, no inequality to reduce — return unchanged
    if all(pshed_prev .== 0.0)
        @warn "[jains_fairness_index] All pshed_prev values are zero — returning unchanged weights"
        return pshed_prev, weights_prev, MOI.OPTIMAL
    end

    # Per-period decomposition with peak charge weighting
    @assert m % n_per_period == 0 "m=$m must be divisible by n_per_period=$n_per_period"
    n_periods = m ÷ n_per_period
    n = n_per_period
    λ = isempty(peak_time_costs) ? ones(n_periods) : peak_time_costs
    @assert length(λ) == n_periods "peak_time_costs must have length $n_periods, got $(length(λ))"

    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1], got $alpha"
    have_pd = !isempty(pd)
    have_pd && @assert length(pd) == m "pd must have length $m when supplied"

    # Per-period weight budget (upper bound only): Σ_i weights_new_{t,i} ≤ weight_budget
    if isfinite(weight_budget)
        for t in 1:n_periods
            offset = (t - 1) * n
            @constraint(model, sum(weights_new[offset + i] for i in 1:n) <= weight_budget)
        end
    end

    # Weighted sum of per-period Jain indices + convex combination with efficiency
    period_jain_terms = []
    eff_terms = []       # per-period shed/total_demand (for α convex combination)
    reg_terms = []       # orthogonal small efficiency regularizer
    use_reg = reg > 0 && have_pd
    use_alpha = alpha < 1.0 && have_pd
    for t in 1:n_periods
        offset = (t - 1) * n
        sum_pshed_t = sum(pshed_new[offset + i] for i in 1:n)
        sum_pshed_sq_t = sum(pshed_new[offset + i]^2 for i in 1:n)
        push!(period_jain_terms, λ[t] * (sum_pshed_t^2) / (n * sum_pshed_sq_t))
        if have_pd
            total_demand_t = sum(pd[offset + i] for i in 1:n)
            if total_demand_t > 0
                eff_t = λ[t] * sum_pshed_t / total_demand_t
                use_alpha && push!(eff_terms, eff_t)
                use_reg   && push!(reg_terms, reg * eff_t)
            end
        end
    end
    # Max α·Jain − (1−α)·shed/td − reg·shed/td
    fairness_part = alpha * sum(period_jain_terms)
    eff_part = (isempty(eff_terms) ? 0.0 : (1.0 - alpha) * sum(eff_terms)) +
               (isempty(reg_terms) ? 0.0 : sum(reg_terms))
    @objective(model, Max, fairness_part - eff_part)
    JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[jains_fairness_index] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Function to compute the min max of load shed
# With peak_time_costs (peak charges): min Σ_t λ[t] * max_i(pshed_t[i])
# With reg>0 and pd supplied: add λ[t]·reg·Σ_i pshed_new_t[i] / Σ_i pd_t[i]
# With alpha ∈ [0,1] and pd supplied: convex combination of efficiency and min-max.
#   - alpha=0: pure efficiency
#   - alpha=1: pure min-max fairness
function min_max_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}; critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[], peak_time_costs::Vector{Float64}=Float64[], n_loads::Int=0, pd::Vector{Float64}=Float64[],reg::Float64=1e-4, alpha::Float64=1.0, weight_budget::Float64=Inf)
    model = JuMP.Model(Ipopt.Optimizer)
    m = length(pshed_prev)
    n_per_period = n_loads > 0 ? n_loads : m
    @variable(model, weights_new[1:m])
    for id in 1:m
        lid_idx = ((id - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        if load_id in critical_ids
            @constraint(model, weights_new[id] >= 50.0)
            @constraint(model, weights_new[id] <= 100.0)
        else
            @constraint(model, weights_new[id] >= 1.0)
            @constraint(model, weights_new[id] <= 10.0)
        end
    end
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] <= TRUST_RADIUS)
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] >= -TRUST_RADIUS)

    @expression(model, pshed_new[i = 1:m],
         pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:m)
    )

    # Per-period decomposition with peak charge weighting
    @assert m % n_per_period == 0 "m=$m must be divisible by n_per_period=$n_per_period"
    n_periods = m ÷ n_per_period
    n = n_per_period
    λ = isempty(peak_time_costs) ? ones(n_periods) : peak_time_costs
    @assert length(λ) == n_periods "peak_time_costs must have length $n_periods, got $(length(λ))"

    # Per-period max of pshed (unweighted). The upper level's objective is a
    # function of pshed_new; weights enter only as decision variables that shape
    # pshed_new through the Jacobian (dpshed/dw). Keeping this unweighted keeps
    # the formulation linear and matches the role weights play in the bilevel:
    # handles for influencing the lower level, not multipliers in the objective.
    @variable(model, t_period[1:n_periods] >= 0)
    for t in 1:n_periods
        offset = (t - 1) * n
        @constraint(model, [i=1:n], t_period[t] >= pshed_new[offset + i])
    end

    # Per-period weight budget (upper bound only)
    # if isfinite(weight_budget)
    #     for t in 1:n_periods
    #         offset = (t - 1) * n
    #         @constraint(model, sum(weights_new[offset + i] for i in 1:n) <= weight_budget)
    #     end
    # end

    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1], got $alpha"
    reg_terms = []
    eff_terms = []  # per-period shed/total_demand for α convex combination
    have_pd = !isempty(pd)
    have_pd && @assert length(pd) == m "pd must have length $m when supplied"
    use_reg = reg > 0 && have_pd
    use_alpha = alpha < 1.0 && have_pd
    if have_pd
        for t in 1:n_periods
            offset = (t - 1) * n
            total_demand_t = sum(pd[offset + i] for i in 1:n)
            if total_demand_t > 0
                eff_t = λ[t] * sum(pshed_new[offset + i] for i in 1:n) / total_demand_t
                use_alpha && push!(eff_terms, eff_t)
                use_reg   && push!(reg_terms, reg * eff_t)
            end
        end
    end
    fairness_term = sum(λ[t] * t_period[t] for t in 1:n_periods)
    #eff_part = (isempty(eff_terms) ? 0.0 : (1.0 - alpha) * sum(eff_terms)) +
               #(isempty(reg_terms) ? 0.0 : sum(reg_terms))
    @objective(model, Min, fairness_term)
    JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[min_max_load_shed] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Function to compute the proportional fairness of load served
# With peak_time_costs (peak charges): max Σ_t λ[t] * Σ_i log(pref_t[i] - pshed_t[i])
# With reg>0: subtract λ[t]·reg·Σ_i pshed_new_t[i] / Σ_i pd_t[i] (efficiency-aligned regularizer)
# With alpha ∈ [0,1]: convex combination of efficiency and proportional fairness.
#   - alpha=0: pure efficiency (Max -shed)
#   - alpha=1: pure proportional fairness (Max Σ log served)
function proportional_fairness_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}, pd::Vector{Float64}; critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[], peak_time_costs::Vector{Float64}=Float64[], n_loads::Int=0, reg::Float64=1e-4, alpha::Float64=1.0, weight_budget::Float64=Inf)
    model = JuMP.Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "warm_start_init_point", "yes")

    m = length(pshed_prev)
    n_per_period = n_loads > 0 ? n_loads : m
    pref = pd  # reference demand aligned with pshed indexing (built by caller from pshed_ids)
    # Start weights at weights_prev so Ipopt evaluates log at a valid initial point
    @variable(model, weights_new[i=1:m] >= 1.0, start = weights_prev[i])
    for id in 1:m
        lid_idx = ((id - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        if load_id in critical_ids
            @constraint(model, weights_new[id] <= 100.0)
        else
            @constraint(model, weights_new[id] <= 10.0)
        end
    end
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] <= TRUST_RADIUS)
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] >= -TRUST_RADIUS)
    # Use variables (not expressions) for pshed_new so log doesn't nest expressions.
    # Bound pshed_new ∈ [0, pref] so the log argument (pref - pshed_new + c) stays
    # strictly positive — without this, the linearization can predict pshed > pref
    # and Ipopt evaluates log(negative) → INVALID_NUMBER_DETECTED, which kills the
    # bilevel loop after iter 1 once any load is fully shed at the warm-start point.
    @variable(model, 0.0 <= pshed_new[i=1:m] <= pref[i], start = clamp(pshed_prev[i], 0.0, pref[i]))
    @constraint(model, [i=1:m],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:m)
    )

    # Per-period decomposition with peak charge weighting
    @assert m % n_per_period == 0 "m=$m must be divisible by n_per_period=$n_per_period"
    n_periods = m ÷ n_per_period
    n = n_per_period
    λ = isempty(peak_time_costs) ? ones(n_periods) : peak_time_costs
    @assert length(λ) == n_periods "peak_time_costs must have length $n_periods, got $(length(λ))"

    # Per-period weight budget (upper bound only)
    if isfinite(weight_budget)
        for t in 1:n_periods
            offset = (t - 1) * n
            @constraint(model, sum(weights_new[offset + i] for i in 1:n) <= weight_budget)
        end
    end

    @assert 0.0 <= alpha <= 1.0 "alpha must be in [0, 1], got $alpha"
    # Shifted log: uniform constant c added to all loads' served values
    c = 1e-6
    # Precompute per-period total demand for α convex combination and reg
    total_demand_t = [sum(pref[(t-1)*n + i] for i in 1:n) for t in 1:n_periods]
    # (1-α) + reg is the coefficient on the efficiency-aligned term (shed/td, subtracted)
    eff_coef = (1.0 - alpha) + reg
    all_td_pos = all(total_demand_t .> 0)
    if eff_coef > 0 && all_td_pos
        @NLobjective(model, Min,
            -alpha * sum(λ[t] * sum(log(pref[(t-1)*n + i] - pshed_new[(t-1)*n + i] + c) for i in 1:n) for t in 1:n_periods)
            + eff_coef * sum(λ[t] * sum(pshed_new[(t-1)*n + i] for i in 1:n) / total_demand_t[t] for t in 1:n_periods))
    else
        @NLobjective(model, Max, alpha * sum(λ[t] * sum(log(pref[(t-1)*n + i] - pshed_new[(t-1)*n + i] + c) for i in 1:n) for t in 1:n_periods))
    end
    JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[proportional_fairness_load_shed] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Function to compute complete efficiency (alpha fairness) of load shed
# With peak_time_costs (peak charges): min Σ_t λ[t] * Σ_i pshed_t[i]
function efficient_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64};critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[], peak_time_costs::Vector{Float64}=Float64[], n_loads::Int=0)
    model = JuMP.Model(Ipopt.Optimizer)
    m = length(pshed_prev)
    n_per_period = n_loads > 0 ? n_loads : m
    @variable(model, weights_new[1:m] .>= 1.0)
    for id in 1:m
        lid_idx = ((id - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        if load_id in critical_ids
            @constraint(model, weights_new[id] <= 100.0)
        else
            @constraint(model, weights_new[id] <= 10.0)
        end
    end
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] <= TRUST_RADIUS)
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] >= -TRUST_RADIUS)
    @expression(model, pshed_new[i = 1:m],
        pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:m)
    )

    # Per-period decomposition with peak charge weighting
    @assert m % n_per_period == 0 "m=$m must be divisible by n_per_period=$n_per_period"
    n_periods = m ÷ n_per_period
    n = n_per_period
    λ = isempty(peak_time_costs) ? ones(n_periods) : peak_time_costs
    @assert length(λ) == n_periods "peak_time_costs must have length $n_periods, got $(length(λ))"

    @objective(model, Min, sum(λ[t] * sum(pshed_new[(t-1)*n + i] for i in 1:n) for t in 1:n_periods))
    JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[efficient_load_shed] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Function to compute the infinity norm fairness of load shed
function infinity_norm_fairness_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}; critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[], n_loads::Int=0)
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(weights_prev)
    m = length(pshed_prev)
    n_per_period =  m
  
    @variable(model, weights_new[1:n] .>= 1.0)
    for id in 1:n
        load_id = isempty(weight_ids) ? id : weight_ids[id]
        if load_id in critical_ids
            @constraint(model, weights_new[id] <= 100.0)
        else
            @constraint(model, weights_new[id] <= 10.0)
        end
    end
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i] <= TRUST_RADIUS)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i] >= -TRUST_RADIUS)
    @expression(model, pshed_new[i = 1:m],
        pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:m)
    )
    # Epigraph reformulation of min ||pshed_new||_∞: t ≥ |pshed_new[i]| for all i
    @variable(model, t >= 0)
    @constraint(model, [i=1:m], pshed_new[i] <= t)
    @constraint(model, [i=1:m], -pshed_new[i] <= t)
    @objective(model, Min, t)
    #  JuMP._CONSTRAINT_LIMIT_FOR_PRINTING[] = 1E9
    # JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[infinity_norm_fairness_load_shed] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Equality min fairness function
# With peak_time_costs (peak charges): min Σ_t λ[t] * (t_period[t] + Σ_i (pshed_t[i] - t_period[t])²)
function equality_min(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}; critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[], peak_time_costs::Vector{Float64}=Float64[], n_loads::Int=0)
    model = JuMP.Model(Ipopt.Optimizer)
    m = length(pshed_prev)
    n_per_period = n_loads > 0 ? n_loads : m
    @variable(model, weights_new[1:m] .>= 1.0)
    for id in 1:m
        lid_idx = ((id - 1) % n_per_period) + 1
        load_id = isempty(weight_ids) ? lid_idx : weight_ids[lid_idx]
        if load_id in critical_ids
            @constraint(model, weights_new[id] <= 100.0)
        else
            @constraint(model, weights_new[id] <= 10.0)
        end
    end
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] <= TRUST_RADIUS)
    @constraint(model, [i=1:m], weights_new[i]-weights_prev[i] >= -TRUST_RADIUS)
    @expression(model, pshed_new[i = 1:m],
        pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:m)
    )

    # Per-period decomposition with peak charge weighting
    @assert m % n_per_period == 0 "m=$m must be divisible by n_per_period=$n_per_period"
    n_periods = m ÷ n_per_period
    n = n_per_period
    λ = isempty(peak_time_costs) ? ones(n_periods) : peak_time_costs
    @assert length(λ) == n_periods "peak_time_costs must have length $n_periods, got $(length(λ))"

    # Hard equality: every load's shed in period t equals the per-period level t_period[t].
    # Minimize the cost-weighted sum of those levels — efficient + perfectly equal across loads.
    @variable(model, t_period[1:n_periods] >= 0)
    for t in 1:n_periods
        offset = (t - 1) * n
        @constraint(model, [i=1:n], pshed_new[offset + i] == t_period[t])
    end
    @objective(model, Min, sum(λ[t] * t_period[t] for t in 1:n_periods))
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[equality_min] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Define the fairness functions for post processing
#Gini index
function gini_index(x)
    x = sort(x)
    n = length(x)
    return (2 * sum(i * x[i] for i in 1:n) / (n * sum(x))) - (n + 1) / n
end

#Jain's index
function jains_index(x)
    n = length(x)
    sum_x = sum(x)
    sum_x2 = sum(xi^2 for xi in x)
    return (sum_x^2) / (n * sum_x2)
end

#Palma Ratio
function palma_ratio(x)
    sorted_x = sort(x)
    n = length(x)
    top_10_percent = sum(sorted_x[ceil(Int, 0.9n):end])
    bottom_40_percent = sum(sorted_x[1:floor(Int, 0.4n)])
    return top_10_percent / bottom_40_percent
end
#Alpha fairness for alpha=1
function alpha_fairness(x, alpha=1)
    if alpha == 1
        return sum(log(xi) for xi in x)
    else
        return sum((xi^(1 - alpha)) / (1 - alpha) for xi in x)
    end
end

#=============================================================================
 Post-hoc multiperiod fairness metric evaluators

 Numerical (non-JuMP) counterparts to the `objective_mn_*` builders in
 core/objective.jl. Each takes per-period per-load vectors
 (Vector{<:AbstractVector{<:Real}}) from a solved problem and returns the
 peak-cost-weighted metric value, so single-level and bilevel solutions can be
 scored on the same yardstick.
=============================================================================#

"""
Internal: per-period element-wise product `w[t][i] * x[t][i]`, used by the
weighted metric evaluators below. Accepts `weights=nothing` (or empty) to
fall back to uniform weights (equivalent to raw `x`).
"""
function _weight_values(x::Vector{<:AbstractVector{<:Real}},
                        weights::Union{Nothing,Vector{<:AbstractVector{<:Real}}}=nothing)
    (weights === nothing || isempty(weights)) && return x
    @assert length(weights) == length(x) "weights length mismatch with per-period data"
    out = Vector{Vector{Float64}}(undef, length(x))
    for t in 1:length(x)
        @assert length(weights[t]) == length(x[t]) "weights[$t] length mismatch"
        out[t] = Float64[weights[t][i] * x[t][i] for i in eachindex(x[t])]
    end
    return out
end

"""
    weighted_shed_mn(shed; peak_time_costs=Float64[], weights=nothing)

Σ_t λ_t · Σ_i (w_{t,i} · pshed_{t,i}). If `weights` is `nothing`, falls back
to raw shed (uniform weights = 1).
"""
function weighted_shed_mn(shed::Vector{<:AbstractVector{<:Real}};
                          peak_time_costs::Vector{<:Real}=Float64[],
                          weights::Union{Nothing,Vector{<:AbstractVector{<:Real}}}=nothing)
    x = _weight_values(shed, weights)
    T = length(x)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    return sum(λ[t] * sum(x[t]) for t in 1:T)
end

"""
    jain_mn(served; peak_time_costs=Float64[], weights=nothing)

Peak-cost-weighted AVERAGE of per-period Jain's index:
`(Σ_t λ_t · Jain_t) / (Σ_t λ_t)`, where `Jain_t = (Σ_i x_{t,i})² / (n_t · Σ_i x_{t,i}²)`
and `x_{t,i} = w_{t,i} · served_{t,i}` when `weights` is supplied.

Lives in `[1/n, 1]` — 1 = perfectly fair, 1/n = one agent gets everything.

(The JuMP objective uses the unnormalized weighted sum; this post-hoc metric
normalizes so the value is directly interpretable as a fairness level.)
"""
function jain_mn(served::Vector{<:AbstractVector{<:Real}};
                 peak_time_costs::Vector{<:Real}=Float64[],
                 weights::Union{Nothing,Vector{<:AbstractVector{<:Real}}}=nothing)
    x = _weight_values(served, weights)
    T = length(x)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    num = Float64[]
    den = Float64[]
    for t in 1:T
        xt = λ[t] * x[t]
        sx = sum(xt)
        sx2 = sum(xi^2 for xi in xt)
        if sx2 > 0
            push!(num, (sx)^2)
            push!(den, length(xt) * sx2)
        end
    end
    return !isempty(den) ? sum(num ./ den) : 0.0
end

"""
    min_max_mn(shed; peak_time_costs=Float64[], weights=nothing)

Σ_t λ_t · max_i (w_{t,i} · pshed_{t,i}). Matches the shape of
`objective_mn_min_max` (which maxes on `w · pshed` inside the MILP).
"""
function min_max_mn(shed::Vector{<:AbstractVector{<:Real}};
                    peak_time_costs::Vector{<:Real}=Float64[],
                    weights::Union{Nothing,Vector{<:AbstractVector{<:Real}}}=nothing)
    x = _weight_values(shed, weights)
    T = length(x)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    return sum(maximum(λ[t] * x[t]) for t in 1:T)
end

"""
    proportional_mn(served; peak_time_costs=Float64[], epsilon=1e-6, weights=nothing)

Σ_t λ_t · Σ_i log(w_{t,i} · served_{t,i} + ε). Matches the shape of
`objective_mn_proportional_fairness_mld`.
"""
function proportional_mn(served::Vector{<:AbstractVector{<:Real}};
                         peak_time_costs::Vector{<:Real}=Float64[],
                         epsilon::Float64=1e-6,
                         weights::Union{Nothing,Vector{<:AbstractVector{<:Real}}}=nothing)
    x = _weight_values(served, weights)
    T = length(x)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    return sum(λ[t] * sum(log(max(s, 0.0) + epsilon) for s in x[t]) for t in 1:T)
end

"""
    palma_mn(shed; peak_time_costs=Float64[], weights=nothing)

Σ_t λ_t · Palma_t on `w_{t,i} · pshed_{t,i}` per period.
"""
function palma_mn(shed::Vector{<:AbstractVector{<:Real}};
                  peak_time_costs::Vector{<:Real}=Float64[],
                  weights::Union{Nothing,Vector{<:AbstractVector{<:Real}}}=nothing)
    x = _weight_values(shed, weights)
    T = length(x)
    λ = isempty(peak_time_costs) ? ones(T) : peak_time_costs
    @assert length(λ) == T "peak_time_costs must have length $T, got $(length(λ))"
    return sum(palma_ratio(λ[t] * x[t]) for t in 1:T)
end



# Plot the fairness indices for the final load shed values
using Plots
function plot_fairness_indices(res, pshed::Vector{Float64}, weight_ids::Vector{Int}, iteration::Int, exp_folder::String, test_name::String)
    n = length(pshed)
    sum_pshed = sum(pshed)
    load_ref = []
    for (i, load) in sort(ref[:load])
        cons = load["connections"]
        for idx in 1:length(cons)
            push!(load_ref, load["pd"][idx])
        end
    end
    load_ref_sum = sum(load_ref)
    println("Total load in reference: $load_ref_sum")

    gen_ref = []# sum(gen["pg"] for (i,gen) in ref[:gen])
    for (i, gen) in ref[:gen]
        cons = gen["connections"]
        for idx in 1:length(cons)
            push!(gen_ref, gen["pg"][idx])
        end
    end
    gen_ref_sum = sum(gen_ref)
    println("Total generation in reference: $gen_ref_sum")

    gen_soln = []# sum(gen["pg"] for (i,gen) in ref[:gen])
    for (i, gen) in res["gen"]
        for idx in 1:length(gen["pg"])
            push!(gen_soln, gen["pg"][idx])
        end
    end
    gen_soln_sum = sum(gen_soln)
    println("Total generation in solution: $gen_soln_sum")

    #load_served = sum((load["pd"]) for (i,load) in res["load"])
    load_served = []
    load_shed = []
    idxs = sort(parse.(Int,collect(keys(res["load"]))))
    for i in 1:length(idxs)
        load = res["load"][string(i)]
        for idx in 1:length(load["pd"])
            push!(load_served, load["pd"][idx])
        end
        # push!(load_shed, load["pshed"])
    end
    load_served_sum = sum(load_served)

    switch_statuses = Dict{String, Any}()
    for (id, switch) in res["switch"]
        switch_statuses[id] = switch["state"]
    end

    switch_names = Dict{String, Any}()
    for (id, switch) in ref[:switch]
        @info id
        @info switch
        switch_names[string(id)] = switch["name"]
    end
    println("Load served percentage: $(load_served_sum/load_ref_sum*100) %")
    push!(served, (load_served_sum/load_ref_sum)*100)

    # Calculate and print fairness indices
    served_array = collect(load_served./load_ref)
    df_fairness_results = DataFrame(TestName=String[], GiniIndex=Float64[], JainsIndex=Float64[], PalmaRatio=Float64[], AlphaFairness1=Float64[])

    push!(df_fairness_results.TestName, "$test_name")
    push!(df_fairness_results.GiniIndex, gini_index(served_array))
    push!(df_fairness_results.JainsIndex, jains_index(served_array))
    push!(df_fairness_results.PalmaRatio, palma_ratio(served_array))
    push!(df_fairness_results.AlphaFairness1, alpha_fairness(served_array, 1))
    println(df_fairness_results)
    CSV.write("$exp_folder/fairness_indices_iteration_$iteration.csv", df_fairness_results)
    # Plot the fairness indices
    indices = ["Gini Index", "Jain's Index", "Palma Ratio",
                "Alpha Fairness (α=1)"]
    values = [df_fairness_results.GiniIndex[1], df_fairness_results.JainsIndex[1],
              df_fairness_results.PalmaRatio[1], df_fairness_results.AlphaFairness1[1]]
    bar_plot = bar(indices, values, title = "Fairness Indices at Iteration $iteration",
                   ylabel = "Value", legend = false, ylim = (0, maximum(values)*1.2))
    savefig(bar_plot, "$exp_folder/fairness_indices_iteration_$iteration.svg")
    display(bar_plot)
end