# using JuMP
# using LinearAlgebra
# using Gurobi  # or any MILP solver

# # ----------------------------------------------------------
# # Palma Ratio Calculation
# # ----------------------------------------------------------
# # function palma(pshed::Dict, dpshed::Matrix{Float64}, w::Vector{Float64}, critical_indices::Vector{Any}, solver=Gurobi.Optimizer)
    
# #     # Preserve original order of keys
# #     keys_orig = collect(keys(pshed))
# #     S_prev = [pshed[k] for k in keys_orig]   # previous load shed vector
# #     dSdw = dpshed                             # ∂S/∂w, assume already in correct order
    
# #     n = length(S_prev)
# #     m = length(S_prev)
    
# #     non_critical_indices = setdiff(1:n, critical_indices)

# #     # Upper/lower bounds for weights
# #     w_lower = 10*ones(n)
# #     w_upper = 100*ones(n)
# #     w_upper[critical_indices] .= 10000   # special upper bound for critical loads

# #     epsilon = 0.0001
# #     pmax = 1400

# #     # Top 10% for objective
# #     k_top = max(1, round(Int, 0.10 * m))
# #     # Bottom 40% for Palma normalization
# #     k_bottom = max(1, round(Int, 0.40 * m))

# #     # ----------------------------------------------------------
# #     # BUILD MILP MODEL
# #     # ----------------------------------------------------------
# #     model = JuMP.Model(solver)

# #     # ------------------ VARIABLES -----------------------------
# #     @variable(model, w_lower[i] <= w_var[i=1:n] <= w_upper[i])
# #     @variable(model, zeta[1:m] >= 0)
# #     @variable(model, sigma[1:m] >= 0)
# #     @variable(model, S_k[1:m])

# #     # Permutation matrix: binary variables
# #     @variable(model, A[1:m,1:m], Bin)#upper_bound=1, lower_bound=0)

# #     # ------------------ PERMUTATION CONSTRAINTS ----------------
# #     one_vec = ones(m)
# #     @constraint(model, A*one_vec .== one_vec)
# #     @constraint(model, transpose(A)*one_vec .== one_vec)

# #     # ------------------ AFFINE UPDATE OF S_k ------------------
# #     for i in 1:m
# #         @constraint(model, S_k[i] == S_prev[i] + sum(dSdw[i,j]*(w_var[j]-weights_prev[j]) for j in 1:n))
# #     end

# #     # ------------------ PALMA CONSTRAINTS ---------------------
# #     c = zeros(m)
# #     c[1:k_bottom] .= 1.0

# #     @constraint(model, transpose(c)*(S_k.*zeta) == 1)
# #     #@constraint(model, [i=1:m], sigma[i] == sum(A[j,i]*sum(S_k[j,k]*zeta[k] for k=1:m) for j=1:m))
# #     @constraint(model, transpose(A)*(S_k*transpose(zeta)) .== sigma)
# #     @constraint(model, sigma .>= A*epsilon)
# #     @constraint(model, sigma .>= A*pmax)
# #     @constraint(model, sigma .<= pmax .- A*pmax .+ S_k*transpose(zeta))
# #     @constraint(model, sigma .<= S_k*transpose(zeta) .- epsilon .+ A*epsilon)

# #     # ------------------ Critical Weights ----------------------------
# #     for idx in critical_indices
# #         @constraint(model, w_var[idx] == w_upper[idx])
# #     end
# #     # ------------------ OBJECTIVE ----------------------------
# #     a = zeros(m)
# #     a[(m - k_top + 1):m] .= 1.0
# #     @objective(model, Max, transpose(a)*(S_k.*zeta))

# #     # ------------------ SOLVE -------------------------------
# #     optimize!(model)

# #     println("Status: ", termination_status(model))
# #     palma_ratio = objective_value(model)
# #     return palma_ratio
# # end

# using JuMP, Gurobi

# function palma_cc(pshed::Dict,
#                      dpshed::Matrix{Float64},
#                      w_prev::Vector{Float64},
#                      critical_indices::Vector{Int},
#                      solver = Gurobi.Optimizer)

#     # ---------------------------
#     # Problem dimensions
#     # ---------------------------
#     keys_orig = collect(keys(pshed))
#     S_prev = [pshed[k] for k in keys_orig]
#     dSdw = dpshed
#     n = length(S_prev)   # number of loads
#     m = n                # number of sorted positions

#     # Weight bounds
#     w_lower = 10 .* ones(n)
#     w_upper = 100 .* ones(n)
#     w_upper[critical_indices] .= 10000

#     # Palma segments
#     k_top = max(1, round(Int, 0.10*m))
#     k_bottom = max(1, round(Int, 0.40*m))

#     # Bounds needed for McCormick envelopes
#     # You may tighten these if you know better bounds.
#     S_min = minimum(S_prev) .- 2000.0
#     S_max = maximum(S_prev) .+ 2000.0

#     t_min = 0.0
#     t_max = 1.0          # You may tighten, but 1.0 is safe.

#     # ---------------------------
#     # Build model
#     # ---------------------------
#     model = Model(solver)

#     # Variables
#     @variable(model, w_lower[i] <= w[i=1:n] <= w_upper[i])
#     @variable(model, S_k[i=1:m], lower_bound=S_min, upper_bound=S_max)
#     @variable(model, t_min <= t <= t_max)
#     @variable(model, y[i=1:m] >= 0)   # scaled S_k, but NOT equal to t*S_k due to envelopes
#     @variable(model, A[1:m,1:m], Bin) # permutation matrix

#     # ---------------------------
#     # Permutation constraints
#     # ---------------------------
#     ones_vec = ones(m)
#     @constraint(model, A * ones_vec .== ones_vec)
#     @constraint(model, A' * ones_vec .== ones_vec)

#     # ---------------------------
#     # Affine update S_k = S_prev + dSdw * (w - w_prev)
#     # ---------------------------
#     for i in 1:m
#         @constraint(model,
#             S_k[i] == S_prev[i] + sum(dSdw[i,j] * (w[j] - w_prev[j]) for j in 1:n)
#         )
#     end

#     # ---------------------------
#     # Charnes–Cooper denominator normalization:
#     # sum(bottom 40%) y = 1
#     # ---------------------------
#     @constraint(model, sum(y[1:k_bottom]) == 1)

#     # ---------------------------
#     # McCormick envelopes for y[i] = t * S_k[i]
#     # ---------------------------
#     for i in 1:m
#         # Variable ranges
#         S_L = S_min
#         S_U = S_max

#         # McCormick 4 inequalities:
#         @constraint(model, y[i] >= S_L*t + t_min*S_k[i] - S_L*t_min)
#         @constraint(model, y[i] >= S_U*t + t_max*S_k[i] - S_U*t_max)
#         @constraint(model, y[i] <= S_U*t + t_min*S_k[i] - S_U*t_min)
#         @constraint(model, y[i] <= S_L*t + t_max*S_k[i] - S_L*t_max)
#     end

#     # ---------------------------
#     # Critical weight fixing
#     # ---------------------------
#     for idx in critical_indices
#         @constraint(model, w[idx] == w_upper[idx])
#     end

#     # ---------------------------
#     # Objective: maximize top 10% of y
#     # ---------------------------
#     @objective(model, Max, sum(y[(m-k_top+1):m]))

#     # ---------------------------
#     # Solve
#     # ---------------------------
#     optimize!(model)

#     println("\nSolver Status: ", termination_status(model))

#     # Recover original Palma ratio from S_k
#     S_k_val = value.(S_k)
#     palma_ratio = sum(S_k_val[(m-k_top+1):m]) / sum(S_k_val[1:k_bottom])

#     return palma_ratio, S_k_val, value.(w)
# end


# # ------------------ EXAMPLE USAGE --------------------------
# critical_indices = critical_id  # from earlier code
# solver = ipopt 
# palma_ratio = palma_cc(pshed, dpshed_mat, weights_prev, critical_id, Gurobi.Optimizer)
# println("Palma ratio: ", palma_ratio)

# # function palma_diagnostic(pshed::Dict, dpshed::Matrix{Float64}, w::Vector{Float64},
# #                           critical_indices::Vector{Any}, solver = Gurobi.Optimizer)

# #     keys_orig = collect(keys(pshed))
# #     S_prev = [pshed[k] for k in keys_orig]
# #     dSdw = dpshed

# #     n = length(S_prev)
# #     m = length(S_prev)
# #     non_critical_indices = setdiff(1:n, critical_indices)

# #     w_lower = 10*ones(n)
# #     w_upper = 100*ones(n)
# #     w_upper[critical_indices] .= 10000

# #     epsilon = 0.0001
# #     pmax = 1400

# #     k_top = max(1, round(Int, 0.10*m))
# #     k_bottom = max(1, round(Int, 0.40*m))

# #     model = JuMP.Model(solver)

# #     @variable(model, w_lower[i] <= w_var[i=1:n] <= w_upper[i])
# #     @variable(model, S_k[1:m])

# #     # Slacks dictionary (name → vector of vars)
# #     slack = Dict{String, Vector{JuMP.VariableRef}}()

# #     # --- PERMUTATION MATRIX ---
# #     @variable(model, A[1:m,1:m], Bin)

# #     # Slacks for A*1 == 1
# #     @variable(model, sA1_pos[1:m] >= 0)
# #     @variable(model, sA1_neg[1:m] >= 0)
# #     slack["A_row"] = [sA1_pos; sA1_neg]

# #     @constraint(model, A*ones(m) .== 1 .+ sA1_pos - sA1_neg)

# #     # Slacks for Aᵀ*1 == 1
# #     @variable(model, sA2_pos[1:m] >= 0)
# #     @variable(model, sA2_neg[1:m] >= 0)
# #     slack["A_col"] = [sA2_pos; sA2_neg]

# #     @constraint(model, transpose(A)*ones(m) .== 1 .+ sA2_pos - sA2_neg)

# #     # --- AFFINE UPDATE S_k ---
# #     @variable(model, sSk_pos[1:m] >= 0)
# #     @variable(model, sSk_neg[1:m] >= 0)
# #     slack["S_k"] = [sSk_pos; sSk_neg]

# #     for i in 1:m
# #         expr = S_prev[i] + sum(dSdw[i,j]*(w_var[j]-w[j]) for j in 1:n)
# #         @constraint(model, S_k[i] == expr + sSk_pos[i] - sSk_neg[i])
# #     end

# #     # --- PALMA RELATED CONSTRAINTS ---
# #     @variable(model, zeta[1:m] >= 0)
# #     @variable(model, sigma[1:m] >= 0)

# #     # 1) cᵀ(S_k .* zeta) == 1
# #     @variable(model, sP1_pos >= 0)
# #     @variable(model, sP1_neg >= 0)
# #     slack["palma_c"] = [sP1_pos; sP1_neg]

# #     c = zeros(m); c[1:k_bottom] .= 1
# #     @constraint(model, dot(c, S_k .* zeta) == 1 + sP1_pos - sP1_neg)

# #     # 2) sigma == A*(S_k*zetaᵀ)
# #     @variable(model, sP2_pos[1:m] >= 0)
# #     @variable(model, sP2_neg[1:m] >= 0)
# #     slack["palma_sigma_match"] = [sP2_pos; sP2_neg]

# #     @constraint(model, transpose(A)*(S_k*transpose(zeta)) .== sigma .+ sP2_pos - sP2_neg)

# #     # --- CRITICAL WEIGHTS ---
# #     @variable(model, sCrit_pos[i=1:length(critical_indices)] >= 0)
# #     @variable(model, sCrit_neg[i=1:length(critical_indices)] >= 0)
# #     slack["crit"] = [sCrit_pos; sCrit_neg]

# #     for (idx, crit_idx) in enumerate(critical_indices)
# #         @constraint(model, w_var[crit_idx] == w_upper[crit_idx] + sCrit_pos[idx] - sCrit_neg[idx])
# #     end

# #     # --- OBJECTIVE: MINIMIZE ALL SLACKS ---
# #     all_slacks = vcat(values(slack)...)
# #      a = zeros(m)
# #     a[(m - k_top + 1):m] .= 1.0
# #     @objective(model, Min, sum(all_slacks) + transpose(a)*(S_k.*zeta))

# #     optimize!(model)

# #     println("Diagnostic feasibility result:")
# #     println("Status: ", termination_status(model))
# #     println("Min slack sum = ", objective_value(model))

# #     # Print only nonzero slacks
# #     println("\nNon-zero slacks:")
# #     for (name, vec) in slack
# #         for (i, v) in enumerate(vec)
# #             if abs(value(v)) > 1e-6
# #                 println("$name[$i] = $(value(v))")
# #             end
# #         end
# #     end

# #     return model
# # end
# # # ------------------ DIAGNOSTIC USAGE --------------------------
# #     diag_model = palma_diagnostic(pshed, dpshed_mat, weights_prev, critical_id, Gurobi.Optimizer)
# This script is used to test the constraint matrix formulation for the Palma ratio linearization using the Charnes-Cooper transformation
# I will be using a JuMP model to test the symbolic calculation of the matrix

using JuMP
using LinearAlgebra
using Ipopt, Gurobi

model = JuMP.Model()

# Define a three bus system
n = length(pshed)
# Extract the load from the reference data
pd = []
for k in 1:n
    push!(pd, sum(ref[:load][k]["pd"]))
end
P = pd
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
   A_ij_P_j[(i-1)*n+1:i*n, (i-1)*n+1:i*n] .= Matrix(I,n,n).*(P)
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