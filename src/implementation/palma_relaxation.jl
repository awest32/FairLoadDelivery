# This script is used to test the constraint matrix formulation for the Palma ratio linearization using the Charnes-Cooper transformation
# I will be using a JuMP model to test the symbolic calculation of the matrix
using FairLoadDelivery
using PowerModelsDistribution
using JuMP
using LinearAlgebra
using Ipopt, Gurobi

#model = JuMP.Model()

# Define a three bus system
#  n = 2
#  P = [1.0, 3.0]
function lin_palma(n::Int, P::Vector{Float64})
    # Create the JuMP model
    model = JuMP.Model()
    # Define a dummy vector for x
    @variable(model, x_hat[1:n^2] >= 0)
    #@variable(model, x_tilde[1:n^2] >= 0)
    @variable(model, a[1:n^2], Bin)
    @variable(model, x_raw[1:n] >= 0)
    @constraint(model, x_raw .== P)
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
    n_squared_identity = LinearAlgebra.Diagonal(ones(n^2))
    #@constraint(model, -n_squared_identity * x_hat .<= 0)

    # second McCormick envelope -\hat{x}_{ij} +x_j +a_ijP_j ≤ P_j
    n_identity = LinearAlgebra.Diagonal(ones(n))
    # Construct the a_ijP_j matrix
    A_ij_P_j = zeros(n^2,n^2)

    for i in 1:n
    A_ij_P_j[(i-1)*n+1:i*n, (i-1)*n+1:i*n] .= LinearAlgebra.Diagonal(P)
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
        T zeros(n,n^2) zeros(n,n)
        -n_squared_identity zeros(n^2,n^2)  zeros(n^2,n) 
        n_squared_identity  -A_ij_P_j  zeros(n^2,n)
        -n_squared_identity A_ij_P_j x_j
        n_squared_identity zeros(n^2,n^2)  -x_j
    ]

    x_tilde = vcat(x_hat, a, x_raw)
    y = x_tilde
    #value.(y)
    # Store the right hand side of the constraints into a vector named g
     g = [ones(n)
        -ones(n)
        ones(n)
        -ones(n)
        zeros(n)
        zeros(n^2)
        zeros(n^2)
        P_out
        zeros(n^2)
    ]

    @constraint(model, F * y .<= g)
    A_long = [A zeros(n,n^2) zeros(n,n)]
    # Create the vectors to extract the top 10% of load shed
    top_10_percent_indices = zeros(n)
    top_10_percent_indices[ceil(Int, 0.9*n):n] .= 1.0
    bottom_40_percent_indices = zeros(n)
    bottom_40_percent_indices[1:ceil(Int, 0.4*n)] .= 1.0
    obj = transpose(top_10_percent_indices)*A_long*y
    denominator_constraint = transpose(bottom_40_percent_indices)*A_long*y
    @constraint(model, denominator_constraint .== 1)
    @objective(model, Max, obj)

    set_optimizer(model, Gurobi.Optimizer)
    optimize!(model)
    return value.(x_hat), value.(a), value.(x_raw)#, value(σ)
end
#y = lin_palma(n, P)

#x, aux = lin_palma(n, P)
# eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.5, ["675a"])

# dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math, ipopt,1)
# pd = Float64[]
# for i in pshed_ids
#     push!(pd, sum(math["load"][string(i)]["pd"]))
# end
# # pshed_prev = pshed_val
# # weights_prev = weight_vals
# # dpshed_dw = dpshed
# #critical_id = [4]
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
    # pshed_prev = pshed_val
    # weights_prev = weight_vals
    # dpshed_dw = dpshed
    n = length(pshed_prev)
    # Create the JuMP model
    model = JuMP.Model()
    set_optimizer_attribute(model, "MIPGap", 1e-6)
    set_optimizer_attribute(model, "TimeLimit", 60)

    #@variable(model, x_hat[1:n^2] >= 0)
    #@variable(model, a[1:n^2], Bin)
    # Create new load shed variables 
    #@variable(model, pshed_new[i=1:n] >=0*pshed_prev[i])

    # Charnes-Cooper auxiliary variables
    @variable(model, σ >= 1e-6)
    @variable(model, y_xhat[1:n^2] >= 0)
    @variable(model, y_a[1:n^2], Bin)      # or Bin if enforced
    @variable(model, y_pshed[1:n] >= 0) # pshed_new
    @variable(model, y_w[1:n] >=0)            # FREE (signed)

    y = vcat(y_xhat, y_a, y_pshed, (y_w-weights_prev))

    # # Create new weight variables
    #@variable(model, weights_new[1:n] >= 0)

    @constraint(model, y_w .<= 10)
    # Create a infinity norm trust region on the weights
    # ϵ = 0.1

    # @constraint(model, [i in eachindex(y_w)],
    #     y_w[i] - weights_prev[i] <= ϵ
    # )

    # @constraint(model, [i in eachindex(y_w)],
    #     weights_prev[i] - y_w[i] <= ϵ
    # )
    # for i in 1:n
    #     if i in critical_id
    #         @constraint(model, weights_new[i] == 1000)
    #     else
    #         @constraint(model, weights_new[i] <= 10)
    #     end
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

    A_row = zeros(n, n^2)
    for i in 1:n
        for j in 1:n
            idx = i + (j-1)*n  # column-major index of a[i,j] in vec(a)
            A_row[i, idx] = 1.0
        end
    end
    A_col = zeros(n, n^2)
    for j in 1:n
        for i in 1:n
            idx = i + (j-1)*n  # same index in vec(a)
            A_col[j, idx] = 1.0
        end
    end


    # Create the big F matrix for the Palma ratio constraints
    F = [zeros(n,n^2) A_row zeros(n,n) zeros(n,n)
        zeros(n,n^2) -A_row zeros(n,n) zeros(n,n)
        zeros(n,n^2) A_col zeros(n,n) zeros(n,n)
        zeros(n,n^2) -A_col zeros(n,n) zeros(n,n)
        -T zeros(n,n^2) zeros(n,n) zeros(n,n)
        #############
        # McCormick envelopes
        ############
        -n_squared_identity zeros(n^2,n^2) zeros(n^2,n) zeros(n^2,n) 
        n_squared_identity  -A_ij_P_j  zeros(n^2,n) zeros(n^2,n)
        -n_squared_identity A_ij_P_j x_j zeros(n^2,n)
        n_squared_identity zeros(n^2,n^2) -x_j zeros(n^2,n)
        #############
        # BIG M
        ##############
        #-n_squared_identity zeros(n^2,n^2) zeros(n^2,n) zeros(n^2,n) 
        #n_squared_identity  -A_ij_P_j -x_j zeros(n^2,n)
        #-n_squared_identity -A_ij_P_j x_j zeros(n^2,n)
        #n_squared_identity -A_ij_P_j zeros(n^2,n) zeros(n^2,n)
        #zeros(n,2*n^2) n_identity -dpshed_dw
        #zeros(n,2*n^2) -n_identity dpshed_dw
    ]

   # x_tilde = vcat(x_hat, a, pshed_new, (weights_new.-weights_prev))
    # Store the right hand side of the constraints into a vector named g
    g = [ones(n)
        -ones(n)
        ones(n)
        -ones(n)
        zeros(n)
        zeros(n^2)
        -P_out
        zeros(n^2)
        zeros(n^2)
        #zeros(n^2)
        #-P_out
        #-P_out
        #zeros(n^2)
        #pshed_prev
        #-pshed_prev
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
   # @constraint(model, denominator_constraint .== σ)
    @objective(model, Max, 0)

    set_optimizer(model, Gurobi.Optimizer)
    # Disable dual reductions
    set_attribute(model, "DualReductions", 0)

    optimize!(model)
    #  value.(pshed_new)
    #    value(σ)
    #    value.(weights_new)
    @info termination_status(model)
    println("Termination: ", termination_status(model))
    println("Primal status: ", primal_status(model))
    println("Dual status: ", dual_status(model))

   return Array(value.(y_pshed)), Array(value.(weights_prev .+ y_w)), value(σ)
end


# y = lin_palma_w_grad_input(dpshed, pshed_val, weight_vals, pd)

# Plot the weights per load 
function plot_weights_per_load(weights_new, weight_ids, k, save_path)
    weights_plot = bar(weight_ids, weights_new, title = "Fair Load Weights per Load - Iteration $k", xlabel = "Load ID", ylabel = "Fair Load Weight", legend = false)
    savefig(weights_plot, "$save_path/fair_load_weights_per_load_k$(k).svg")
    println("Weights plot saved as $save_path/fair_load_weights_per_load_k$(k).svg")
end

# # eng, math, lbs, critical_id = setup_network( "ieee_13_aw_edit/motivation_b.dss", 0.5, ["675a"])
# # # Initial fair load weights
# # fair_weights = Float64[]
# # for (load_id, load) in (math["load"])
# #     push!(fair_weights, load["weight"])
# # end
# # # Order the load using the indices from the pshed_ids
# # pd = Float64[]
# # for i in pshed_ids
# #     push!(pd, sum(math["load"][string(i)]["pd"]))
# # end
# # dpshed, pshed_val, pshed_ids, weight_vals, weight_ids = lower_level_soln(math, fair_weights, 1)
