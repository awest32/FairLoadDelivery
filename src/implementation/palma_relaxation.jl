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

    # Permutation matrix row sum constraint: Σ_j a_{ij} = 1 for all i
    # Using column-major indexing: a[k] = a_{i,j} where k = i + (j-1)*n
    A_row = zeros(n, n^2)
    for i in 1:n
        for j in 1:n
            idx = i + (j-1)*n  # column-major index
            A_row[i, idx] = 1.0
        end
    end

    # Permutation matrix column sum constraint: Σ_i a_{ij} = 1 for all j
    A_col = zeros(n, n^2)
    for j in 1:n
        for i in 1:n
            idx = i + (j-1)*n  # column-major index
            A_col[j, idx] = 1.0
        end
    end

    # Sorting constraint matrix: enforces ascending order S_k <= S_{k+1}
    # T * x_hat <= 0 means sorted[k] - sorted[k+1] <= 0
    # Using column-major indexing for consistency
    T = zeros(n-1, n^2)
    for k in 1:n-1
        for j in 1:n
            idx_k = k + (j-1)*n
            idx_k1 = (k+1) + (j-1)*n
            T[k, idx_k] = 1.0
            T[k, idx_k1] = -1.0
        end
    end

    #############################################################
    # Implement the McCormick envelopes for the bilinear terms
    # u_ij = a_ij * x_j where a_ij ∈ {0,1} and x_j ∈ [0, P_j]
    # Envelope 1: u >= 0 (eq. 19)
    # Envelope 2: u >= x̄·a + x - x̄ (eq. 20)
    # Envelope 3: u <= x̄·a (eq. 21)
    # Envelope 4: u <= x (eq. 22)
    ##############################################################
    n_squared_identity = LinearAlgebra.Diagonal(ones(n^2))
    #@constraint(model, -n_squared_identity * x_hat .<= 0)

    # Construct A_ij_P_j matrix for McCormick: represents a_ij * P_j term
    # Using column-major indexing: entry (i,j) maps to index i + (j-1)*n
    A_ij_P_j = zeros(n^2, n^2)
    for j in 1:n
        for i in 1:n
            idx = i + (j-1)*n  # column-major index for (i,j)
            A_ij_P_j[idx, idx] = P[j]  # P_j is the upper bound for x_j
        end
    end

    # Construct x_j matrix: maps x[j] to all entries (i,j) in the bilinear term
    # x_j[idx, j] = 1 where idx = i + (j-1)*n for all i
    x_j = zeros(n^2, n)
    P_out = zeros(n^2)
    for j in 1:n
        for i in 1:n
            idx = i + (j-1)*n
            x_j[idx, j] = 1.0
            P_out[idx] = P[j]
        end
    end

    # Create the big F matrix for the Palma ratio constraints
    # Variable vector: y = [x_hat, a, x_raw]
    # Constraints:
    #   Row 1-2: Row sum of permutation matrix = 1 (A_row * a = 1)
    #   Row 3-4: Column sum of permutation matrix = 1 (A_col * a = 1)
    #   Row 5: Sorting constraint (T * x_hat <= 0)
    #   Row 6-9: McCormick envelopes
    F = [zeros(n,n^2) A_row zeros(n,n)           # A_row * a <= 1
        zeros(n,n^2) -A_row zeros(n,n)           # -A_row * a <= -1 (i.e., A_row * a >= 1)
        zeros(n,n^2) A_col zeros(n,n)            # A_col * a <= 1
        zeros(n,n^2) -A_col zeros(n,n)           # -A_col * a <= -1 (i.e., A_col * a >= 1)
        T zeros(n-1,n^2) zeros(n-1,n)            # T * x_hat <= 0 (ascending order)
        -n_squared_identity zeros(n^2,n^2) zeros(n^2,n)   # -u <= 0 (envelope 1: u >= 0)
        n_squared_identity -A_ij_P_j zeros(n^2,n)         # u - a*P <= 0 (envelope 3: u <= a*P)
        -n_squared_identity A_ij_P_j x_j                  # -u + a*P + x <= P (envelope 2: u >= a*P + x - P)
        n_squared_identity zeros(n^2,n^2) -x_j            # u - x <= 0 (envelope 4: u <= x)
    ]

    x_tilde = vcat(x_hat, a, x_raw)
    y = x_tilde

    # Right-hand side vector g
    # Matches the constraint structure of F
    g = [ones(n)           # A_row * a <= 1
        -ones(n)           # -A_row * a <= -1
        ones(n)            # A_col * a <= 1
        -ones(n)           # -A_col * a <= -1
        zeros(n-1)         # T * x_hat <= 0 (n-1 rows now)
        zeros(n^2)         # -u <= 0 (envelope 1)
        zeros(n^2)         # u - a*P <= 0 (envelope 3)
        P_out              # -u + a*P + x <= P (envelope 2)
        zeros(n^2)         # u - x <= 0 (envelope 4)
    ]

    # Verify dimensions before adding constraint
    @assert size(F, 2) == length(y) "F columns ($(size(F, 2))) must match y length ($(length(y)))"
    @assert size(F, 1) == length(g) "F rows ($(size(F, 1))) must match g length ($(length(g)))"

    @constraint(model, F * y .<= g)
    A_long = [A_row zeros(n,n^2) zeros(n,n)]
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

    # Sorting constraint matrix: enforces ascending order S_k <= S_{k+1}
    # T * x_hat <= 0 means sorted[k] - sorted[k+1] <= 0
    # Using column-major indexing for consistency with A_row, A_col
    T = zeros(n-1, n^2)
    for k in 1:n-1
        for j in 1:n
            idx_k = k + (j-1)*n
            idx_k1 = (k+1) + (j-1)*n
            T[k, idx_k] = 1.0
            T[k, idx_k1] = -1.0
        end
    end

    #############################################################
    # Implement the McCormick envelopes for the bilinear terms
    # u_ij = a_ij * x_j where a_ij ∈ {0,1} and x_j ∈ [0, P_j]
    # Envelope 1: u >= 0 (eq. 19)
    # Envelope 2: u >= x̄·a + x - x̄ (eq. 20)
    # Envelope 3: u <= x̄·a (eq. 21)
    # Envelope 4: u <= x (eq. 22)
    ##############################################################
    n_squared_identity = Diagonal(ones(n^2))

    # Upper bound for load shed is the demand (pd) at each node
    P = pd

    # Construct A_ij_P_j matrix for McCormick: represents a_ij * P_j term
    # Using column-major indexing: entry (i,j) maps to index i + (j-1)*n
    A_ij_P_j = zeros(n^2, n^2)
    for j in 1:n
        for i in 1:n
            idx = i + (j-1)*n  # column-major index for (i,j)
            A_ij_P_j[idx, idx] = P[j]  # P_j is the upper bound for x_j
        end
    end

    # Construct x_j matrix: maps x[j] to all entries (i,j) in the bilinear term
    # x_j[idx, j] = 1 where idx = i + (j-1)*n for all i
    x_j = zeros(n^2, n)
    P_out = zeros(n^2)
    for j in 1:n
        for i in 1:n
            idx = i + (j-1)*n
            x_j[idx, j] = 1.0
            P_out[idx] = P[j]
        end
    end

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
    # Variable vector: y = [y_xhat, y_a, y_pshed, (y_w - weights_prev)]
    # Constraints:
    #   Row 1-2: Row sum of permutation matrix = 1 (A_row * a = 1)
    #   Row 3-4: Column sum of permutation matrix = 1 (A_col * a = 1)
    #   Row 5: Sorting constraint (T * x_hat <= 0)
    #   Row 6-9: McCormick envelopes
    F = [zeros(n,n^2) A_row zeros(n,n) zeros(n,n)           # A_row * a <= 1
        zeros(n,n^2) -A_row zeros(n,n) zeros(n,n)           # -A_row * a <= -1
        zeros(n,n^2) A_col zeros(n,n) zeros(n,n)            # A_col * a <= 1
        zeros(n,n^2) -A_col zeros(n,n) zeros(n,n)           # -A_col * a <= -1
        -T zeros(n-1,n^2) zeros(n-1,n) zeros(n-1,n)         # -T * x_hat <= 0 (note: T is n-1 rows now)
        # McCormick envelopes
        -n_squared_identity zeros(n^2,n^2) zeros(n^2,n) zeros(n^2,n)   # -u <= 0 (envelope 1: u >= 0)
        n_squared_identity -A_ij_P_j zeros(n^2,n) zeros(n^2,n)         # u - a*P <= 0 (envelope 3: u <= a*P)
        -n_squared_identity A_ij_P_j x_j zeros(n^2,n)                  # -u + a*P + x <= P (envelope 2)
        n_squared_identity zeros(n^2,n^2) -x_j zeros(n^2,n)            # u - x <= 0 (envelope 4: u <= x)
    ]

    # Right-hand side vector g (FIXED: McCormick envelope RHS values)
    # Matches the constraint structure of F
    g = [ones(n)           # A_row * a <= 1
        -ones(n)           # -A_row * a <= -1
        ones(n)            # A_col * a <= 1
        -ones(n)           # -A_col * a <= -1
        zeros(n-1)         # -T * x_hat <= 0 (n-1 rows now)
        zeros(n^2)         # -u <= 0 (envelope 1)
        zeros(n^2)         # u - a*P <= 0 (envelope 3) - FIXED: was -P_out
        P_out              # -u + a*P + x <= P (envelope 2) - FIXED: was zeros
        zeros(n^2)         # u - x <= 0 (envelope 4)
    ]

    # Verify dimensions before adding constraint
    @assert size(F, 2) == length(y) "F columns ($(size(F, 2))) must match y length ($(length(y)))"
    @assert size(F, 1) == length(g) "F rows ($(size(F, 1))) must match g length ($(length(g)))"

    @constraint(model, F * y .<= g*σ)
    A_long = [A_row zeros(n,n^2) zeros(n,2*n)]  # FIXED: use A_row instead of undefined A
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
