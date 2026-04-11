# Create the other fairness functions

# Function to compute Jain's Fairness Index
# With period_weights (peak charges): max Σ_t λ[t] * Jain_t
function jains_fairness_index(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}, critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[]; period_weights::Vector{Float64}=Float64[], n_loads::Int=0)
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
    λ = isempty(period_weights) ? ones(n_periods) : period_weights
    @assert length(λ) == n_periods "period_weights must have length $n_periods, got $(length(λ))"

    # Weighted sum of per-period Jain indices
    period_jain_terms = []
    for t in 1:n_periods
        offset = (t - 1) * n
        sum_pshed_t = sum(pshed_new[offset + i] for i in 1:n)
        sum_pshed_sq_t = sum(pshed_new[offset + i]^2 for i in 1:n)
        push!(period_jain_terms, λ[t] * (sum_pshed_t^2) / (n * sum_pshed_sq_t))
    end
    @objective(model, Max, sum(period_jain_terms))
    JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[jains_fairness_index] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Function to compute the min max of load shed
# With period_weights (peak charges): min Σ_t λ[t] * max_i(pshed_t[i])
function min_max_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}, critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[]; period_weights::Vector{Float64}=Float64[], n_loads::Int=0)
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
    λ = isempty(period_weights) ? ones(n_periods) : period_weights
    @assert length(λ) == n_periods "period_weights must have length $n_periods, got $(length(λ))"

    # Per-period max variables
    @variable(model, t_period[1:n_periods] >= 0)
    for t in 1:n_periods
        offset = (t - 1) * n
        @constraint(model, [i=1:n], t_period[t] >= pshed_new[offset + i])
    end
    @objective(model, Min, sum(λ[t] * t_period[t] for t in 1:n_periods))
    JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[min_max_load_shed] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Function to compute the proportional fairness of load served
# With period_weights (peak charges): max Σ_t λ[t] * Σ_i log(pref_t[i] - pshed_t[i])
function proportional_fairness_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}, pd::Vector{Float64}, critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[]; period_weights::Vector{Float64}=Float64[], n_loads::Int=0)
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
    # Use variables (not expressions) for pshed_new so log doesn't nest expressions
    @variable(model, pshed_new[i=1:m], start = pshed_prev[i])
    @constraint(model, [i=1:m],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:m)
    )

    # Per-period decomposition with peak charge weighting
    @assert m % n_per_period == 0 "m=$m must be divisible by n_per_period=$n_per_period"
    n_periods = m ÷ n_per_period
    n = n_per_period
    λ = isempty(period_weights) ? ones(n_periods) : period_weights
    @assert length(λ) == n_periods "period_weights must have length $n_periods, got $(length(λ))"

    # Shifted log: uniform constant c added to all loads' served values
    c = 1e-6
    @NLobjective(model, Max, sum(λ[t] * sum(log(pref[(t-1)*n + i] - pshed_new[(t-1)*n + i] + c) for i in 1:n) for t in 1:n_periods))
    JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[proportional_fairness_load_shed] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Function to compute complete efficiency (alpha fairness) of load shed
# With period_weights (peak charges): min Σ_t λ[t] * Σ_i pshed_t[i]
function complete_efficiency_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64},math::Dict{String,Any},critical_ids::Vector{Int}, weight_ids::Vector{Int}=Int[]; period_weights::Vector{Float64}=Float64[], n_loads::Int=0)
    model = JuMP.Model(Ipopt.Optimizer)
    m = length(pshed_prev)
    n_per_period = n_loads > 0 ? n_loads : m
    # Determine the total load in the reference case
    total_load_ref = 0.0
    for (i, load) in math["load"]
        cons = load["connections"]
        for idx in 1:length(cons)
            total_load_ref += load["pd"][idx]
        end
    end
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
    λ = isempty(period_weights) ? ones(n_periods) : period_weights
    @assert length(λ) == n_periods "period_weights must have length $n_periods, got $(length(λ))"

    @objective(model, Min, sum(λ[t] * sum(pshed_new[(t-1)*n + i] for i in 1:n) for t in 1:n_periods))
    JuMP.set_silent(model)
    optimize!(model)
    status = termination_status(model)
    if status ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]
        @warn "[complete_efficiency_load_shed] Solver did not converge: $status"
    end
    return value.(pshed_new), value.(weights_new), status
end

# Function to compute the infinity norm fairness of load shed
function infinity_norm_fairness_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}, critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[])
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(weights_prev)
    pshed_new = JuMP.@variable(
    model,pshed_new[j in keys(pshed_val)] in JuMP.Parameter(pshed_val[j]),
    base_name = "pshed_new"
        )
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
    #@variable(model, t >= 0)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    # @constraint(model, [i in 1:length(pshed_new)],
    #     pshed_new[i] <= t
    # )
    @objective(model, Min, norm(pshed_new, Inf))
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
# With period_weights (peak charges): min Σ_t λ[t] * (t_period[t] + Σ_i (pshed_t[i] - t_period[t])²)
function equality_min(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}, critical_ids::Vector{Int}=Int[], weight_ids::Vector{Int}=Int[]; period_weights::Vector{Float64}=Float64[], n_loads::Int=0)
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
    λ = isempty(period_weights) ? ones(n_periods) : period_weights
    @assert length(λ) == n_periods "period_weights must have length $n_periods, got $(length(λ))"

    # Per-period max variables with quadratic penalty encouraging equal shedding
    @variable(model, t_period[1:n_periods] >= 0)
    for t in 1:n_periods
        offset = (t - 1) * n
        @constraint(model, [i=1:n], t_period[t] >= pshed_new[offset + i])
    end
    @objective(model, Min, sum(λ[t] * (t_period[t] + sum((pshed_new[(t-1)*n + i] - t_period[t])^2 for i in 1:n)) for t in 1:n_periods))
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