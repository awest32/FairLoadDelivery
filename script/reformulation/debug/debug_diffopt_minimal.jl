#=
Minimal DiffOpt debugging script
Tests forward differentiation on increasingly complex problems:
1. Simple LP with parameter in objective
2. Simple QP with parameter in objective
3. Simple NLP mimicking MLD structure

Run with: julia --project=. script/reformulation/debug_diffopt_minimal.jl
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))  # debug → reformulation → script → root

using JuMP
using DiffOpt
using Ipopt
using LinearAlgebra
using Printf

println("="^70)
println("DIFFOPT MINIMAL DEBUGGING")
println("="^70)
println()

#======================================================================
TEST 1: Simple LP - min w*x subject to x >= 1
Expected: x* = 1, dx/dw = 0 (constraint binding, not objective)
======================================================================#
println("TEST 1: Simple LP with parameter in objective")
println("-"^50)

model1 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model1)

@variable(model1, w in Parameter(2.0))  # weight parameter
@variable(model1, x >= 1.0)             # decision variable
@objective(model1, Min, w * x)

optimize!(model1)
x_opt = value(x)
println("  Optimal x = $x_opt (expected: 1.0)")

# Forward differentiation: dw = 1
DiffOpt.set_forward_parameter(model1, w, 1.0)
DiffOpt.forward_differentiate!(model1)
dx_dw = DiffOpt.get_forward_variable(model1, x)
println("  dx/dw = $dx_dw (expected: 0, since constraint x>=1 is binding)")
println()

#======================================================================
TEST 2: LP with slack - max w*x subject to x <= 10
Expected: x* = 10, dx/dw = 0 (upper bound binding)
======================================================================#
println("TEST 2: LP with upper bound binding")
println("-"^50)

model2 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model2)

@variable(model2, w2 in Parameter(2.0))
@variable(model2, 0 <= x2 <= 10)
@objective(model2, Max, w2 * x2)

optimize!(model2)
x2_opt = value(x2)
println("  Optimal x = $x2_opt (expected: 10.0)")

DiffOpt.set_forward_parameter(model2, w2, 1.0)
DiffOpt.forward_differentiate!(model2)
dx2_dw = DiffOpt.get_forward_variable(model2, x2)
println("  dx/dw = $dx2_dw (expected: 0, since x=10 is binding)")
println()

#======================================================================
TEST 3: LP with trade-off - max w1*x1 + w2*x2 subject to x1 + x2 <= 10
Expected: dx1/dw1 > 0 (increasing w1 shifts allocation to x1)
======================================================================#
println("TEST 3: LP with trade-off (resource allocation)")
println("-"^50)

model3 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model3)

@variable(model3, w1 in Parameter(1.0))
@variable(model3, w2 in Parameter(2.0))
@variable(model3, x1 >= 0)
@variable(model3, x2 >= 0)
@constraint(model3, x1 + x2 <= 10)
@objective(model3, Max, w1 * x1 + w2 * x2)

optimize!(model3)
println("  Optimal: x1 = $(round(value(x1), digits=4)), x2 = $(round(value(x2), digits=4))")
println("  (Expected: x1=0, x2=10 since w2 > w1)")

# Perturb w1 by 1.0
DiffOpt.set_forward_parameter(model3, w1, 1.0)
DiffOpt.set_forward_parameter(model3, w2, 0.0)
DiffOpt.forward_differentiate!(model3)

dx1_dw1 = DiffOpt.get_forward_variable(model3, x1)
dx2_dw1 = DiffOpt.get_forward_variable(model3, x2)
println("  dx1/dw1 = $(round(dx1_dw1, digits=4)) (expected: >= 0)")
println("  dx2/dw1 = $(round(dx2_dw1, digits=4)) (expected: <= 0)")
println()

#======================================================================
TEST 4: Mimic MLD structure - max w*pd subject to pshed = demand - pd
Here pd is "power delivered", pshed is "power shed"
Expected: d(pshed)/dw < 0 (increasing weight → more delivery → less shed)
======================================================================#
println("TEST 4: MLD-like structure (max weighted delivery)")
println("-"^50)

model4 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model4)

demand = 100.0  # constant
@variable(model4, w4 in Parameter(1.0))  # weight on this load
@variable(model4, 0 <= pd <= demand)      # power delivered
@variable(model4, pshed >= 0)             # power shed
@constraint(model4, pshed_def, pshed == demand - pd)
@objective(model4, Max, w4 * pd)

optimize!(model4)
println("  Optimal: pd = $(round(value(pd), digits=4)), pshed = $(round(value(pshed), digits=4))")

DiffOpt.set_forward_parameter(model4, w4, 1.0)
DiffOpt.forward_differentiate!(model4)

dpd_dw = DiffOpt.get_forward_variable(model4, pd)
dpshed_dw = DiffOpt.get_forward_variable(model4, pshed)
println("  dpd/dw = $(round(dpd_dw, digits=4)) (expected: 0, pd at upper bound)")
println("  dpshed/dw = $(round(dpshed_dw, digits=4)) (expected: 0, since pd at upper bound)")
println()

#======================================================================
TEST 5: Two competing loads - max w1*pd1 + w2*pd2 subject to capacity
This is the simplest case that captures network resource competition
======================================================================#
println("TEST 5: Two competing loads with capacity constraint")
println("-"^50)

model5 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model5)

demand1, demand2 = 100.0, 100.0
capacity = 120.0  # Can't serve both fully

@variable(model5, w1_5 in Parameter(1.0))
@variable(model5, w2_5 in Parameter(1.0))
@variable(model5, 0 <= pd1 <= demand1)
@variable(model5, 0 <= pd2 <= demand2)
@variable(model5, pshed1 >= 0)
@variable(model5, pshed2 >= 0)
@constraint(model5, pshed1 == demand1 - pd1)
@constraint(model5, pshed2 == demand2 - pd2)
@constraint(model5, capacity_con, pd1 + pd2 <= capacity)
@objective(model5, Max, w1_5 * pd1 + w2_5 * pd2)

optimize!(model5)
println("  Optimal: pd1=$(round(value(pd1),digits=2)), pd2=$(round(value(pd2),digits=2))")
println("  Optimal: pshed1=$(round(value(pshed1),digits=2)), pshed2=$(round(value(pshed2),digits=2))")
println("  (Expected: pd1=pd2=60 due to equal weights)")

# Increase w1: should shift capacity to pd1, decreasing pshed1, increasing pshed2
DiffOpt.set_forward_parameter(model5, w1_5, 1.0)
DiffOpt.set_forward_parameter(model5, w2_5, 0.0)
DiffOpt.forward_differentiate!(model5)

dpshed1_dw1 = DiffOpt.get_forward_variable(model5, pshed1)
dpshed2_dw1 = DiffOpt.get_forward_variable(model5, pshed2)
println("  dpshed1/dw1 = $(round(dpshed1_dw1, digits=4)) (expected: < 0)")
println("  dpshed2/dw1 = $(round(dpshed2_dw1, digits=4)) (expected: > 0)")
println()

#======================================================================
TEST 6: Same as Test 5 but with quadratic regularization
This tests NLP capability
======================================================================#
println("TEST 6: Two loads with quadratic regularization (NLP)")
println("-"^50)

model6 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model6)

@variable(model6, w1_6 in Parameter(1.0))
@variable(model6, w2_6 in Parameter(1.0))
@variable(model6, 0 <= pd1_6 <= demand1)
@variable(model6, 0 <= pd2_6 <= demand2)
@variable(model6, pshed1_6 >= 0)
@variable(model6, pshed2_6 >= 0)
@constraint(model6, pshed1_6 == demand1 - pd1_6)
@constraint(model6, pshed2_6 == demand2 - pd2_6)
@constraint(model6, pd1_6 + pd2_6 <= capacity)
# Quadratic regularization to make it smooth
@objective(model6, Max, w1_6 * pd1_6 + w2_6 * pd2_6 - 0.001*(pd1_6^2 + pd2_6^2))

optimize!(model6)
println("  Optimal: pd1=$(round(value(pd1_6),digits=2)), pd2=$(round(value(pd2_6),digits=2))")
println("  Optimal: pshed1=$(round(value(pshed1_6),digits=2)), pshed2=$(round(value(pshed2_6),digits=2))")

DiffOpt.set_forward_parameter(model6, w1_6, 1.0)
DiffOpt.set_forward_parameter(model6, w2_6, 0.0)
DiffOpt.forward_differentiate!(model6)

dpshed1_6_dw1 = DiffOpt.get_forward_variable(model6, pshed1_6)
dpshed2_6_dw1 = DiffOpt.get_forward_variable(model6, pshed2_6)
println("  dpshed1/dw1 = $(round(dpshed1_6_dw1, digits=4)) (expected: < 0)")
println("  dpshed2/dw1 = $(round(dpshed2_6_dw1, digits=4)) (expected: > 0)")
println()

#======================================================================
TEST 7: Check if re-optimization changes sensitivities
======================================================================#
println("TEST 7: Re-optimization effect on sensitivities")
println("-"^50)

model7 = Model(() -> DiffOpt.diff_optimizer(Ipopt.Optimizer))
set_silent(model7)

@variable(model7, w7 in Parameter(1.0))
@variable(model7, 0 <= x7 <= 10)
@objective(model7, Max, w7 * x7 - 0.1*x7^2)

optimize!(model7)
println("  After first optimize: x = $(round(value(x7), digits=4))")

DiffOpt.set_forward_parameter(model7, w7, 1.0)
DiffOpt.forward_differentiate!(model7)
dx_dw_before = DiffOpt.get_forward_variable(model7, x7)
println("  dx/dw (before re-optimize) = $(round(dx_dw_before, digits=4))")

# Re-optimize (should give same solution)
optimize!(model7)
DiffOpt.forward_differentiate!(model7)
dx_dw_after = DiffOpt.get_forward_variable(model7, x7)
println("  dx/dw (after re-optimize) = $(round(dx_dw_after, digits=4))")
println("  Difference: $(abs(dx_dw_before - dx_dw_after))")
println()

#======================================================================
SUMMARY
======================================================================#
println("="^70)
println("SUMMARY")
println("="^70)
println()
println("If Tests 5-6 show NEGATIVE dpshed1/dw1, DiffOpt is working correctly.")
println("If they show POSITIVE values, there's a fundamental issue with the API usage.")
println()
println("Compare these results to the full MLD model to identify where the issue arises.")
