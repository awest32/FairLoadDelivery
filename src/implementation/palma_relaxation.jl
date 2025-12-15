# This script is used to test the constraint matrix formulation for the Palma ratio linearization using the Charnes-Cooper transformation
# I will be using a JuMP model to test the symbolic calculation of the matrix
using FairLoadDelivery
using PowerModelsDistribution
using JuMP
using LinearAlgebra
using Ipopt, Gurobi

model = JuMP.Model()

# Define a three bus system
# n = 13
# P = rand(n)
function lin_palma(n::Int, P::Vector{Float64})
    # Create the JuMP model
    model = JuMP.Model()
    # Define a dummy vector for x
    @variable(model, x_hat[1:n^2] >= 0)
    #@variable(model, x_tilde[1:n^2] >= 0)
    @variable(model, a[1:n^2], Bin)
    @variable(model, x_raw[1:n] >= 0)
    @variable(model, σ >= 1e-6)

    # First permutation matrix constraint to sort the rows Ax == 1
    A = zeros(n,n^2)
    for i in 1:n
        if i == 1
            A[i, i:n] .= 1.0
        else
            A[i, (i-1)*n+1:i*n] .= 1.0
        end
    end
    #@constraint(model, A*a.<= 1.0)
    #@constraint(model, -A*a .<= -1.0)

    # Second permutation matrix constraint to sort the columns A^T * x == 1
    AT = zeros(n,n^2)
    #γ = 1
    for i in 1:n
        for j in 1:n
            row = zeros(n)
            row[i] = 1.0
            if j == 1
                AT[i,j:n] = row
            else
                AT[i,(j-1)*n+1:j*n] = row
            end
        end
        #γ += 1
    end
    #@constraint(model, AT*a .<= 1.0)
    #@constraint(model, -AT*a .<= -1.0)

    # Create the ascending sorting constraint matrix -T ⋅ \tilde{x} ≤ 0
    # This is the difference matrix
    T = zeros(n,n^2)
    for i in 1:n
        if i == 1
            T[i, i:n] .= 1.0
            T[i, n+1:(i+1)*n] .-= 1.0
        else
            T[i, (i-1)*n+1:i*n] .= 1.0
            if i == n
                T[i, (i-1)*n+1:i*n] .= 1.0
            else
                T[i, i*n+1:(i+1)*n] .-= 1.0
            end
        end
    end
    #@constraint(model, -T*x_hat.<= 0.0)

    #############################################################
    # Implement th McCormick envelopes for the bilinear terms
    ##############################################################
    # Start with \hat{x}_{ij} ≤ 0
    # Stagger In matrices
    # lower_bound_x = zeros(n^2,n^2)
    # for i in 1:n
    #     if i == 1
    #         lower_bound_x[i:n, i:n] .= Matrix(I, n, n)
    #     else
    #         lower_bound_x[(i-1)*n+1:i*n, (i-1)*n+1:i*n] .=  Matrix(I, n, n)
    #     end
    # end
    n_squared_identity = Diagonal(ones(n^2))
    #@constraint(model, -n_squared_identity * x_hat .<= 0)

    # second McCormick envelope -\hat{x}_{ij} +x_j +a_ijP_j ≤ P_j
    n_identity = Diagonal(ones(n))
    # Construct the a_ijP_j matrix
    A_ij_P_j = zeros(n^2,n^2)

    for i in 1:n
    A_ij_P_j[(i-1)*n+1:i*n, (i-1)*n+1:i*n] .= Diagonal(P)
    end
    A_ij_P_j
    # Construct the x_j matrix so that each block corresponds with the appropriate x_ij
    x_j = zeros(n^2,n)
    P_out = zeros(n^2)
    for j in 1:n
        if j == 1
            x_j[j:n,j:n] = Matrix(I, n, n)
            P_out[j:n] .= P
        else
            x_j[(j-1)*n+1:j*n, 1:n] = Matrix(I, n, n)
            P_out[(j-1)*n+1:j*n] .= P
        end
    end
    x_j
    #@constraint(model, -n_squared_identity * x_hat .+ x_j .+ A_ij_P_j * a .<= P_out)

    # third McCormick envelope -\hat{x}_{ij} - a_ijP_j ≤ 0
    #@constraint(model, n_squared_identity * x_hat .- A_ij_P_j * a .<= 0)

    # fourth McCormick envelope \hat{x}_{ij} - x_j ≤ 0
    #@constraint(model, n_squared_identity * x_hat .- x_j .<= 0)

    # Create the big F matrix for the Palma ratio constraints
    F = [zeros(n,n^2) A zeros(n,n) 
        zeros(n,n^2) -A zeros(n,n)
        zeros(n,n^2) AT zeros(n,n) 
        zeros(n,n^2) -AT zeros(n,n) 
        -T zeros(n,n^2) zeros(n,n) 
        # -n_squared_identity zeros(n^2,n^2)  zeros(n^2,n) 
        n_squared_identity  -A_ij_P_j  zeros(n^2,n)
        -n_squared_identity A_ij_P_j x_j
        n_squared_identity zeros(n^2,n^2)  -x_j
    ]

    x_tilde = vcat(x_hat, a, x_raw)
    y = x_tilde
    # Store the right hand side of the constraints into a vector named g
    g = [ones(n)
        -ones(n)
        ones(n)
        -ones(n)
        zeros(n)
        # zeros(n^2)
        P_out
        zeros(n^2)
        zeros(n^2)
    ]

    @constraint(model, F * y .<= g*σ)
    A_long = [A zeros(n,n^2) zeros(n,n)]
    # Create the vectors to extract the top 10% of load shed
    top_10_percent_indices = zeros(n)
    top_10_percent_indices[ceil(Int, 0.9*n):n] .= 1.0
    bottom_40_percent_indices = zeros(n)
    bottom_40_percent_indices[1:floor(Int, 0.4*n)] .= 1.0
    obj = transpose(top_10_percent_indices)*A_long*y
    denominator_constraint = transpose(bottom_40_percent_indices)*A_long*y
    @constraint(model, denominator_constraint .== σ)
    @objective(model, Max, obj)

    set_optimizer(model, Gurobi.Optimizer)
    optimize!(model)
    return value.(x_tilde), value(σ)
end

#x, aux = lin_palma(n, P)
# eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.5, ["675a"])

# dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math, ipopt)
# pd = Float64[]
# for i in pshed_ids
#     push!(pd, sum(math["load"][string(i)]["pd"]))
# end
# pshed_prev = pshed_val
# weights_prev = weight_vals
# dpshed_dw = dpshed
#critical_id = [4]
function lin_palma_w_grad_input(dpshed_dw::Matrix{Float64}, pshed_prev::Vector{Float64}, weights_prev::Vector{Float64}, pd::Vector{Float64})
    # Notation explantaion:
    # dpshed_dw: matrix of partial derivatives of pshed with respect to weights (loads)
    # pshed_prev: vector of previous load shed values   
    # w_prev: vector of previous weights
    # x_hat: A*pshed_new = A*x to match the notation for the Charnes-Cooper transformation; uses McCormick envelopes to linearize this term
    # A: permutation matrix to sort the loads in ascending order, 'a' represents each entry
    # x_raw: load shed values (x) we will be optimizing over
    # σ: auxiliary variable from the Charnes-Cooper transformation
    # Implementing the Palma ratio linearization with gradient input
    #  \min_{\boldsymbol{w\in[\underline{w},\overline{w}]}} \frac{ \sum_{d \in \text{Top10\%}}P^{shed}_{d}(w)}{ \sum_{d \in \text{Bottom40\%}} P^{shed}_{d}(w)}\\
    #  \textsf{s.t.} \quad P^{shed}_i = P^{shed}_{i-1} + \nabla_wP^{shed}_{i-1}^*\Delta w\\
    #   w_c = 10\overline{w} \quad \forall c\in\mathcal{C}

    n = length(pshed_prev)
    # Create the JuMP model
    model = JuMP.Model()
    set_optimizer_attribute(model, "MIPGap", 1e-6)
    set_optimizer_attribute(model, "TimeLimit", 60)

    @variable(model, x_hat[1:n^2] >= 0)
    @variable(model, a[1:n^2], Bin)
    # Create new load shed variables 
    #@variable(model, x_raw[1:n] >= 0)
    @variable(model, pshed_new[i=1:n] >=0*pshed_prev[i])# in Parameter(pshed_prev[i]))

    @variable(model, σ >= 1e-6, start = 1.0)

    # Create new weight variables
    @variable(model, weights_new[1:n] >= 0)
    # for i in 1:n
    #     if i in critical_id
    #         @constraint(model, weights_new[i] == 1000)
    #     else
    #         @constraint(model, weights_new[i] <= 10)
    #     end
    # end
    # # Create the load shed update constraints
    # for i in 1:n
    #     @constraint(model, pshed_new[i] <= pshed_prev[i] + sum(dpshed_dw[i,j]*(weights_new[j]-weights_prev[j]) for j in 1:n))
    #     #@constraint(model, pshed_new[i] >= pshed_prev[i] + sum(dpshed_dw[i,j]*(weights_new[j]-weights_prev[j]) for j in 1:n))
    # end

    # First permutation matrix constraint to sort the rows Ax == 1
    A = zeros(n,n^2)
    for i in 1:n
        if i == 1
            A[i, i:n] .= 1.0
        else
            A[i, (i-1)*n+1:i*n] .= 1.0
        end
    end
    #@constraint(model, A*a.<= 1.0)
    #@constraint(model, -A*a .<= -1.0)

    # Second permutation matrix constraint to sort the columns A^T * x == 1
    AT = zeros(n,n^2)
    #γ = 1
    for i in 1:n
        for j in 1:n
            row = zeros(n)
            row[i] = 1.0
            if j == 1
                AT[i,j:n] = row
            else
                AT[i,(j-1)*n+1:j*n] = row
            end
        end
        #γ += 1
    end
    #@constraint(model, AT*a .<= 1.0)
    #@constraint(model, -AT*a .<= -1.0)

    # Create the ascending sorting constraint matrix -T ⋅ \tilde{x} ≤ 0
    # This is the difference matrix
    T = zeros(n,n^2)
    for i in 1:n
        if i == 1
            T[i, i:n] .= 1.0
            T[i, n+1:(i+1)*n] .-= 1.0
        else
            T[i, (i-1)*n+1:i*n] .= 1.0
            if i == n
                T[i, (i-1)*n+1:i*n] .= 1.0
            else
                T[i, i*n+1:(i+1)*n] .-= 1.0
            end
        end
    end
    #@constraint(model, -T*x_hat.<= 0.0)

    #############################################################
    # Implement th McCormick envelopes for the bilinear terms
    ##############################################################
    # Start with \hat{x}_{ij} ≤ 0
    # Stagger In matrices
    # lower_bound_x = zeros(n^2,n^2)
    # for i in 1:n
    #     if i == 1
    #         lower_bound_x[i:n, i:n] .= Matrix(I, n, n)
    #     else
    #         lower_bound_x[(i-1)*n+1:i*n, (i-1)*n+1:i*n] .=  Matrix(I, n, n)
    #     end
    # end
    n_squared_identity = Diagonal(ones(n^2))
    #@constraint(model, -n_squared_identity * x_hat .<= 0)

    # second McCormick envelope -\hat{x}_{ij} +x_j +a_ijP_j ≤ P_j
    n_identity = Diagonal(ones(n))
    # Construct the a_ijP_j matrix
    A_ij_P_j = zeros(n^2,n^2)

    # Set the upper limit for the sorted load shed for each node (x_raw) to be the demand (pd) at that node
    P = pd
    for i in 1:n
    A_ij_P_j[(i-1)*n+1:i*n, (i-1)*n+1:i*n] .= Diagonal(P)
    end
    #A_ij_P_j
    # Construct the x_j matrix so that each block corresponds with the appropriate x_ij
    x_j = zeros(n^2,n)
    P_out = zeros(n^2)
    for j in 1:n
        if j == 1
            x_j[j:n,j:n] = Matrix(I, n, n)
            P_out[j:n] .= P
        else
            x_j[(j-1)*n+1:j*n, 1:n] = Matrix(I, n, n)
            P_out[(j-1)*n+1:j*n] .= P
        end
    end
    #x_j
    #@constraint(model, -n_squared_identity * x_hat .+ x_j .+ A_ij_P_j * a .<= P_out)

    # third McCormick envelope -\hat{x}_{ij} - a_ijP_j ≤ 0
    #@constraint(model, n_squared_identity * x_hat .- A_ij_P_j * a .<= 0)

    # fourth McCormick envelope \hat{x}_{ij} - x_j ≤ 0
    #@constraint(model, n_squared_identity * x_hat .- x_j .<= 0)

    # Create the big F matrix for the Palma ratio constraints
    F = [zeros(n,n^2) A zeros(n,n) zeros(n,n)
        zeros(n,n^2) -A zeros(n,n) zeros(n,n)
        zeros(n,n^2) AT zeros(n,n) zeros(n,n)
        zeros(n,n^2) -AT zeros(n,n) zeros(n,n)
        -T zeros(n,n^2) zeros(n,n) zeros(n,n)
        # -n_squared_identity zeros(n^2,n^2)  zeros(n^2,n) 
        n_squared_identity  -A_ij_P_j  zeros(n^2,n) zeros(n^2,n)
        -n_squared_identity A_ij_P_j x_j zeros(n^2,n)
        n_squared_identity zeros(n^2,n^2)  -x_j zeros(n^2,n)
        zeros(n,2*n^2) n_identity -dpshed_dw
        zeros(n,2*n^2) -n_identity dpshed_dw
    ]

    x_tilde = vcat(x_hat, a, pshed_new, (weights_new.-weights_prev))
    y = x_tilde
    # Store the right hand side of the constraints into a vector named g
    g = [ones(n)
        -ones(n)
        ones(n)
        -ones(n)
        zeros(n)
        # zeros(n^2)
        P_out
        zeros(n^2)
        zeros(n^2)
        pshed_prev
        -pshed_prev
    ]

    @constraint(model, F * y .<= g*σ)
    A_long = [A zeros(n,n^2) zeros(n,2*n)]
    # Create the vectors to extract the top 10% of load shed
    top_10_percent_indices = zeros(n)
    top_10_percent_indices[ceil(Int, 0.9*n):n] .= 1.0
    bottom_40_percent_indices = zeros(n)
    bottom_40_percent_indices[1:floor(Int, 0.4*n)] .= 1.0
    obj = transpose(top_10_percent_indices)*A_long*y
    denominator_constraint = transpose(bottom_40_percent_indices)*A_long*y
    @constraint(model, denominator_constraint .== σ)
    @objective(model, Max, obj)

    set_optimizer(model, Gurobi.Optimizer)
    optimize!(model)
    #  value.(pshed_new)
    #    value(σ)
    #    value.(weights_new)
    return value.(pshed_new), value.(weights_new),value(σ)
end


 #pshed_new, weights_new, sigma = lin_palma_w_grad_input(dpshed, pshed_val, weight_vals, pd)