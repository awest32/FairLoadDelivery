# Test script to verify the correctness of Matrix F construction
# for the Palma ratio linearization with McCormick envelopes
#
# This script tests:
# 1. Matrix dimension consistency
# 2. Permutation matrix constraints (doubly stochastic)
# 3. Sorting constraint produces ascending order
# 4. McCormick envelope bounds are correct

using LinearAlgebra
using Test

println("="^60)
println("Testing Matrix F Construction for Palma Ratio Linearization")
println("="^60)

# Test with small n for easy verification
n = 3
P = [1.0, 2.0, 3.0]  # Upper bounds for x_j

println("\nTest parameters: n = $n, P = $P")

# ============================================================
# Build matrices using the FIXED column-major indexing
# ============================================================

println("\n--- Building matrices with column-major indexing ---")

# Permutation matrix row sum constraint
A_row = zeros(n, n^2)
for i in 1:n
    for j in 1:n
        idx = i + (j-1)*n  # column-major
        A_row[i, idx] = 1.0
    end
end

# Permutation matrix column sum constraint
A_col = zeros(n, n^2)
for j in 1:n
    for i in 1:n
        idx = i + (j-1)*n  # column-major
        A_col[j, idx] = 1.0
    end
end

# Sorting constraint (n-1 rows)
T = zeros(n-1, n^2)
for k in 1:n-1
    for j in 1:n
        idx_k = k + (j-1)*n
        idx_k1 = (k+1) + (j-1)*n
        T[k, idx_k] = 1.0
        T[k, idx_k1] = -1.0
    end
end

# McCormick matrices
n_squared_identity = Diagonal(ones(n^2))

A_ij_P_j = zeros(n^2, n^2)
for j in 1:n
    for i in 1:n
        idx = i + (j-1)*n
        A_ij_P_j[idx, idx] = P[j]
    end
end

x_j = zeros(n^2, n)
P_out = zeros(n^2)
for j in 1:n
    for i in 1:n
        idx = i + (j-1)*n
        x_j[idx, j] = 1.0
        P_out[idx] = P[j]
    end
end

# Build F matrix (for lin_palma with 3-column variable vector)
F = [zeros(n,n^2) A_row zeros(n,n)
    zeros(n,n^2) -A_row zeros(n,n)
    zeros(n,n^2) A_col zeros(n,n)
    zeros(n,n^2) -A_col zeros(n,n)
    T zeros(n-1,n^2) zeros(n-1,n)
    -n_squared_identity zeros(n^2,n^2) zeros(n^2,n)
    n_squared_identity -A_ij_P_j zeros(n^2,n)
    -n_squared_identity A_ij_P_j x_j
    n_squared_identity zeros(n^2,n^2) -x_j
]

# Build g vector (FIXED)
g = [ones(n)
    -ones(n)
    ones(n)
    -ones(n)
    zeros(n-1)
    zeros(n^2)
    zeros(n^2)    # envelope 3: u - a*P <= 0
    P_out         # envelope 2: -u + a*P + x <= P
    zeros(n^2)
]

# Variable vector dimensions
y_dim = 2*n^2 + n  # x_hat (n²) + a (n²) + x_raw (n)

println("A_row size: $(size(A_row))")
println("A_col size: $(size(A_col))")
println("T size: $(size(T))")
println("F size: $(size(F))")
println("g size: $(size(g))")
println("y dimension: $y_dim")

# ============================================================
# Test 1: Dimension consistency
# ============================================================
println("\n--- Test 1: Dimension Consistency ---")

@testset "Dimension Consistency" begin
    @test size(F, 2) == y_dim
    @test size(F, 1) == length(g)
    @test size(A_row) == (n, n^2)
    @test size(A_col) == (n, n^2)
    @test size(T) == (n-1, n^2)
    @test size(A_ij_P_j) == (n^2, n^2)
    @test size(x_j) == (n^2, n)
    @test length(P_out) == n^2
end

# ============================================================
# Test 2: Permutation constraints (doubly stochastic)
# ============================================================
println("\n--- Test 2: Permutation Constraints ---")

# Create a valid permutation matrix (identity)
# In column-major vectorization: a[k] = 1 if k = i + (i-1)*n (diagonal)
a_identity = zeros(n^2)
for i in 1:n
    idx = i + (i-1)*n  # diagonal in column-major
    a_identity[idx] = 1.0
end

println("Identity permutation vector (column-major):")
println("  a = $a_identity")

@testset "Permutation Constraints" begin
    # Row sums should equal 1
    row_sums = A_row * a_identity
    @test all(isapprox.(row_sums, 1.0, atol=1e-10))
    println("  Row sums: $row_sums ✓")

    # Column sums should equal 1
    col_sums = A_col * a_identity
    @test all(isapprox.(col_sums, 1.0, atol=1e-10))
    println("  Column sums: $col_sums ✓")
end

# Test with a different permutation: swap elements 1 and 2
# Permutation: [2, 1, 3] means position 1 gets element 2, position 2 gets element 1
# In matrix form: a[1,2]=1, a[2,1]=1, a[3,3]=1
a_swap = zeros(n^2)
a_swap[1 + (2-1)*n] = 1.0  # a[1,2] = 1 (row 1, col 2)
a_swap[2 + (1-1)*n] = 1.0  # a[2,1] = 1 (row 2, col 1)
a_swap[3 + (3-1)*n] = 1.0  # a[3,3] = 1 (row 3, col 3)

println("\nSwap permutation (swap pos 1 and 2):")
println("  a = $a_swap")

@testset "Swap Permutation" begin
    row_sums = A_row * a_swap
    @test all(isapprox.(row_sums, 1.0, atol=1e-10))
    println("  Row sums: $row_sums ✓")

    col_sums = A_col * a_swap
    @test all(isapprox.(col_sums, 1.0, atol=1e-10))
    println("  Column sums: $col_sums ✓")
end

# ============================================================
# Test 3: Sorting constraint
# ============================================================
println("\n--- Test 3: Sorting Constraint ---")

# For x = [3.0, 1.0, 2.0], the sorted version is [1.0, 2.0, 3.0]
# This requires permutation a where:
#   sorted[1] = x[2] → a[1,2] = 1
#   sorted[2] = x[3] → a[2,3] = 1
#   sorted[3] = x[1] → a[3,1] = 1

x_unsorted = [3.0, 1.0, 2.0]
a_sort = zeros(n^2)
a_sort[1 + (2-1)*n] = 1.0  # a[1,2] = 1 → sorted[1] gets x[2] = 1.0
a_sort[2 + (3-1)*n] = 1.0  # a[2,3] = 1 → sorted[2] gets x[3] = 2.0
a_sort[3 + (1-1)*n] = 1.0  # a[3,1] = 1 → sorted[3] gets x[1] = 3.0

# Compute x_hat = a * x (bilinear product, represented as n² vector)
# x_hat[idx] = a[i,j] * x[j] where idx = i + (j-1)*n
x_hat = zeros(n^2)
for j in 1:n
    for i in 1:n
        idx = i + (j-1)*n
        x_hat[idx] = a_sort[idx] * x_unsorted[j]
    end
end

# The sorted values are obtained by summing x_hat over each row i
sorted_values = zeros(n)
for i in 1:n
    for j in 1:n
        idx = i + (j-1)*n
        sorted_values[i] += x_hat[idx]
    end
end

println("Unsorted x: $x_unsorted")
println("Sorted values: $sorted_values")

@testset "Sorting Constraint" begin
    # Verify sorted values are [1.0, 2.0, 3.0]
    @test sorted_values ≈ [1.0, 2.0, 3.0]

    # Verify T * x_hat <= 0 (ascending order)
    sort_constraint = T * x_hat
    println("T * x_hat = $sort_constraint (should be <= 0)")
    @test all(sort_constraint .<= 1e-10)
    println("  Sorting constraint satisfied ✓")
end

# ============================================================
# Test 4: McCormick Envelope Bounds
# ============================================================
println("\n--- Test 4: McCormick Envelope Bounds ---")

# Test McCormick envelopes for a specific (i,j) pair
# For a[i,j] ∈ {0,1} and x[j] ∈ [0, P[j]], u[i,j] = a[i,j] * x[j]
# Envelopes:
#   u >= 0
#   u >= P[j]*a + x - P[j]
#   u <= P[j]*a
#   u <= x

@testset "McCormick Envelopes" begin
    for (a_val, x_val, j) in [(0.0, 0.5, 1), (1.0, 0.5, 1), (0.0, 1.5, 2), (1.0, 1.5, 2)]
        u_true = a_val * x_val
        P_j = P[j]

        # Envelope bounds
        lower1 = 0.0
        lower2 = P_j * a_val + x_val - P_j
        upper1 = P_j * a_val
        upper2 = x_val

        lower_bound = max(lower1, lower2)
        upper_bound = min(upper1, upper2)

        println("\na=$a_val, x=$x_val, j=$j, P_j=$P_j")
        println("  True u = a*x = $u_true")
        println("  Envelope 1 (u >= 0): $lower1")
        println("  Envelope 2 (u >= P*a + x - P): $lower2")
        println("  Envelope 3 (u <= P*a): $upper1")
        println("  Envelope 4 (u <= x): $upper2")
        println("  Bounds: [$lower_bound, $upper_bound]")

        @test lower_bound <= u_true + 1e-10
        @test u_true <= upper_bound + 1e-10

        # For binary a, the bounds should be tight
        if a_val == 0.0 || a_val == 1.0
            @test isapprox(lower_bound, u_true, atol=1e-10) || isapprox(upper_bound, u_true, atol=1e-10)
        end
    end
end

# ============================================================
# Test 5: Full constraint system F * y <= g
# ============================================================
println("\n--- Test 5: Full Constraint System ---")

# Build a feasible y vector
# y = [x_hat, a, x_raw] where x_hat = bilinear terms, a = permutation, x_raw = loads

# Use x_raw values within bounds [0, P[j]] for each j
# P = [1.0, 2.0, 3.0], so use x_raw = [0.5, 1.5, 2.5]
x_raw_bounded = [0.5, 1.5, 2.5]

# Use identity permutation (simpler for testing)
a = a_identity

# Compute x_hat as the McCormick relaxation (for binary a, x_hat = a * x)
x_hat_full = zeros(n^2)
for j in 1:n
    for i in 1:n
        idx = i + (j-1)*n
        x_hat_full[idx] = a[idx] * x_raw_bounded[j]
    end
end

y = vcat(x_hat_full, a, x_raw_bounded)

println("y vector:")
println("  x_hat = $x_hat_full")
println("  a = $a")
println("  x_raw = $x_raw_bounded")

# Evaluate F * y
Fy = F * y

@testset "Full Constraint System" begin
    violations = Fy .- g
    max_violation = maximum(violations)

    println("\nConstraint violations (F*y - g):")
    println("  Max violation: $max_violation")

    # All constraints should be satisfied (F*y <= g)
    @test max_violation <= 1e-10
    println("  All constraints satisfied ✓")

    # Check specific constraint blocks
    offset = 0

    # Row sum constraints: A_row * a = 1
    row_block = Fy[offset+1:offset+n]
    @test all(row_block .<= 1.0 + 1e-10)
    offset += n

    # -A_row * a <= -1 (i.e., A_row * a >= 1)
    negrow_block = Fy[offset+1:offset+n]
    @test all(negrow_block .<= -1.0 + 1e-10)
    offset += n

    # Column sum constraints
    col_block = Fy[offset+1:offset+n]
    @test all(col_block .<= 1.0 + 1e-10)
    offset += n

    negcol_block = Fy[offset+1:offset+n]
    @test all(negcol_block .<= -1.0 + 1e-10)
    offset += n

    # Sorting constraint: T * x_hat <= 0
    sort_block = Fy[offset+1:offset+n-1]
    @test all(sort_block .<= 1e-10)
    println("  Sorting constraints: max = $(maximum(sort_block)) ✓")
    offset += n-1

    # McCormick envelope 1: -u <= 0 (i.e., u >= 0)
    env1_block = Fy[offset+1:offset+n^2]
    @test all(env1_block .<= 1e-10)
    println("  McCormick envelope 1 (u >= 0): max = $(maximum(env1_block)) ✓")
    offset += n^2

    # McCormick envelope 3: u - a*P <= 0
    env3_block = Fy[offset+1:offset+n^2]
    @test all(env3_block .<= 1e-10)
    println("  McCormick envelope 3 (u <= a*P): max = $(maximum(env3_block)) ✓")
    offset += n^2

    # McCormick envelope 2: -u + a*P + x <= P
    env2_block = Fy[offset+1:offset+n^2]
    @test all(env2_block .<= P_out .+ 1e-10)
    println("  McCormick envelope 2 (u >= a*P + x - P): max violation = $(maximum(env2_block .- P_out)) ✓")
    offset += n^2

    # McCormick envelope 4: u - x <= 0
    env4_block = Fy[offset+1:offset+n^2]
    @test all(env4_block .<= 1e-10)
    println("  McCormick envelope 4 (u <= x): max = $(maximum(env4_block)) ✓")
end

println("\n" * "="^60)
println("All tests passed!")
println("="^60)
