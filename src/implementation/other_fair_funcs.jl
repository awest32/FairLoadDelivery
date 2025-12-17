# Create the other fairness functions 
using FairLoadDelivery
using JuMP, Ipopt, Gurobi

# Function to compute Jain's Fairness Index
function jains_fairness_index(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    n = length(pshed_prev)
    @variable(model, pshed_new[1:n] >= 0)
    @variable(model, weights_new[1:n] >= 0)
    @constraint(model, [i=1:n], weights_new[i] <= 10)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i]<= 0.1)
    @constraint(model, [i in 1:n],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:n)
    )
    sum_pshed = sum(pshed_new)
    sum_pshed_squared = sum(pshed_new[i]^2 for i in 1:n)
    if sum_pshed == 0.0
        return 0.0
    end
    
    fairness_index = (sum_pshed^2) / (n * sum_pshed_squared)
    @objective(model, Max, fairness_index)
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Function to compute the min max of load shed 
# pshed is a vector of load shed values
# updating pshed with the gradietn dpshed_dw
# with respect the the change in weights w, w_prev
# optimizing pshed_new and weights_new
function min_max_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, pshed_new[1:length(pshed_prev)] >= 0)
    @variable(model, weights_new[1:length(weights_prev)] >= 0)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i] <= 10)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i]<= 0.1)

    #@variable(model, t >= 0)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    #@constraint(model, t >= maximum(pshed_new))
    @objective(model, Min, maximum(pshed_new))
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Function to compute the proportional fairness load shed
function proportional_fairness_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, pshed_new[1:length(pshed_prev)] >= 0)
    @variable(model, weights_new[1:length(weights_prev)] >= 0)
    @constraint(model, weights_new[1:length(weights_prev)] .<= 10)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i]<= 0.1)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    @objective(model, Max, sum(log(pshed_new[i] + 1e-6) for i in 1:length(pshed_new))) # Adding a small constant to avoid log(0)
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Function to compute complete efficiency (alpha fairness) of load shed
function complete_efficiency_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, pshed_new[1:length(pshed_prev)] >= 0)
    @variable(model, weights_new[1:length(weights_prev)] >= 0)
    @constraint(model, weights_new[1:length(weights_prev)] .<= 10)
    @constraint(model, [i=1:length(weights_prev)], weights_new[i]-weights_prev[i]<= 0.1)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    @objective(model, Min, sum(pshed_new[i] for i in 1:length(pshed_new)))
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end

# Function to compute the infinity norm fairness of load shed
function infinity_norm_fairness_load_shed(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64})
    model = JuMP.Model(Ipopt.Optimizer)
    @variable(model, pshed_new[1:length(pshed_prev)] >= 0)
    @variable(model, weights_new[1:length(weights_prev)] >= 0)
    @variable(model, t >= 0)
    @constraint(model, [i in 1:length(pshed_prev)],
        pshed_new[i] == pshed_prev[i] + sum(dpshed_dw[i,j] * (weights_new[j] - weights_prev[j]) for j in 1:length(weights_prev))
    )
    @constraint(model, [i in 1:length(pshed_new)],
        pshed_new[i] <= t
    )
    @objective(model, Min, t)
    JuMP.set_silent(model)
    optimize!(model)
    return value.(pshed_new), value.(weights_new)
end