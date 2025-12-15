# This script is used to test the constraint matrix formulation for the Palma ratio linearization using the Charnes-Cooper transformation
# I will be using a JuMP model to test the symbolic calculation of the matrix

using JuMP
using LinearAlgebra
using Ipopt, Gurobi

model = JuMP.Model()

# Define a three bus system
n = 13
P = rand(n)
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